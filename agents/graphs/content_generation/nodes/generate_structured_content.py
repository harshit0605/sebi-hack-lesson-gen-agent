from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    ContentBlockModel,
    LessonModel,
    ContentAnalysisResult,
    AnchorModel,
    VoiceScriptModel,
    BlockType,
    SourceType,
)
from langchain_core.output_parsers import PydanticOutputParser
from agents.graphs.content_generation.llm import content_generator_llm
from agents.graphs.content_generation.prompts.generate_structured_content import (
    GENERATE_STRUCTURED_CONTENT_PROMPTS,
)
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime


async def generate_structured_content(
    state: LessonCreationState,
) -> LessonCreationState:
    """Generate all content blocks, anchors, and voice scripts using Pydantic models"""

    lessons = state.get("new_lessons", []) + state.get("updated_lessons", [])
    all_blocks = []
    all_anchors = []
    all_voice_scripts = []

    try:
        for lesson in lessons:
            # Generate content blocks for each lesson
            blocks = await generate_blocks_for_lesson_structured(
                lesson, state["content_analysis"], state
            )
            all_blocks.extend(blocks)

            # Generate anchors for the lesson
            anchors = await generate_anchors_for_lesson(
                lesson, state["pdf_content"], state["chunk_id"], state["page_numbers"]
            )
            all_anchors.extend(anchors)

            # Link blocks to anchors
            linked_blocks = await link_blocks_to_anchors(blocks, anchors)

            # Update lesson with actual content references
            lesson.blocks = [
                f"block_{block.type.value}_{i}" for i, block in enumerate(linked_blocks)
            ]
            lesson.anchors = [
                f"anchor_{anchor.source_type.value}_{anchor.short_label}"
                for anchor in anchors
            ]

            # Generate voice script and update lesson
            voice_script = await generate_voice_script_for_lesson(
                lesson, linked_blocks, anchors
            )
            if voice_script:
                lesson.voice_script_id = f"voice_{lesson.slug}_v1"
                all_voice_scripts.append(voice_script)
                lesson.voice_ready = True

            # Update all_blocks with the linked blocks for this lesson
            all_blocks = [
                block for block in all_blocks if block.lesson_id != lesson.slug
            ] + linked_blocks

        # Validate generated content
        validation_results = await validate_generated_content(
            all_blocks, all_anchors, all_voice_scripts
        )

        if validation_results["valid"]:
            state["content_blocks"] = all_blocks
            state["anchors"] = all_anchors
            state["voice_scripts"] = all_voice_scripts
            state["current_step"] = "content_generated"

            logging.info(
                f"Generated content: {len(all_blocks)} blocks, {len(all_anchors)} anchors, {len(all_voice_scripts)} voice scripts"
            )
        else:
            state["validation_errors"].extend(validation_results["errors"])
            logging.error(
                f"Content generation validation failed: {validation_results['errors']}"
            )

    except Exception as e:
        error_msg = f"Failed to generate structured content: {str(e)}"
        state["validation_errors"].append(error_msg)
        logging.error(error_msg)

    return state


async def generate_blocks_for_lesson_structured(
    lesson: LessonModel,
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate content blocks with enhanced context and structured output parsing"""

    blocks = []

    # Get enhanced context from state
    integration_plan = state.get("integration_plan")
    journey_plan = state.get("journey_creation_plan")

    # Build journey context
    journey_context = ""
    if journey_plan:
        journey_context = f"""
        Journey Context:
        - Journey id: {journey_plan.slug}
        - Journey Title: {journey_plan.title}
        - Level: {journey_plan.level}
        - Target Audience: {journey_plan.target_audience}
        - Key Topics: {", ".join(journey_plan.key_topics)}
        """

    # Generate concept blocks using enhanced prompts
    concept_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS["concept_blocks"]

    messages = concept_prompt.format_messages(
        lesson_title=lesson.title,
        objectives=lesson.learning_objectives,
        key_concepts=content_analysis.key_concepts,
        sebi_themes=content_analysis.sebi_themes,
        learning_opportunities=content_analysis.learning_opportunities,
        journey_context=journey_context,
        integration_rationale=integration_plan.rationale if integration_plan else "",
        page_numbers=state.get("page_numbers", []),
    )

    concept_blocks = await content_generator_llm.ainvoke_with_structured_output(
        messages, List[ContentBlockModel]
    )

    blocks.extend(concept_blocks)

    # Generate example blocks
    example_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS["example_blocks"]

    example_messages = example_prompt.format_messages(
        lesson_title=lesson.title,
        objectives=lesson.learning_objectives,
        key_concepts=content_analysis.key_concepts,
        sebi_themes=content_analysis.sebi_themes,
        learning_opportunities=content_analysis.learning_opportunities,
        journey_context=journey_context,
        integration_rationale=integration_plan.rationale if integration_plan else "",
        page_numbers=state.get("page_numbers", []),
    )

    example_blocks = await content_generator_llm.ainvoke_with_structured_output(
        example_messages, List[ContentBlockModel]
    )

    blocks.extend(example_blocks)

    # Generate quiz blocks
    quiz_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS["quiz_blocks"]

    quiz_messages = quiz_prompt.format_messages(
        lesson_title=lesson.title,
        objectives=lesson.learning_objectives,
        key_concepts=content_analysis.key_concepts,
        sebi_themes=content_analysis.sebi_themes,
        learning_opportunities=content_analysis.learning_opportunities,
        journey_context=journey_context,
        integration_rationale=integration_plan.rationale if integration_plan else "",
        page_numbers=state.get("page_numbers", []),
    )

    quiz_blocks = await content_generator_llm.ainvoke_with_structured_output(
        quiz_messages, List[ContentBlockModel]
    )

    blocks.extend(quiz_blocks)

    # Generate reflection blocks
    reflection_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS["reflection_blocks"]

    reflection_messages = reflection_prompt.format_messages(
        lesson_title=lesson.title,
        objectives=lesson.learning_objectives,
        key_concepts=content_analysis.key_concepts,
        sebi_themes=content_analysis.sebi_themes,
        learning_opportunities=content_analysis.learning_opportunities,
        journey_context=journey_context,
        integration_rationale=integration_plan.rationale if integration_plan else "",
        page_numbers=state.get("page_numbers", []),
    )

    reflection_blocks = await content_generator_llm.ainvoke_with_structured_output(
        reflection_messages, List[ContentBlockModel]
    )

    blocks.extend(reflection_blocks)

    # Generate interactive blocks if needed
    if len(blocks) < 3:  # Ensure minimum content variety
        interactive_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS["interactive_blocks"]

        interactive_messages = interactive_prompt.format_messages(
            lesson_title=lesson.title,
            objectives=lesson.learning_objectives,
            key_concepts=content_analysis.key_concepts,
            sebi_themes=content_analysis.sebi_themes,
            learning_opportunities=content_analysis.learning_opportunities,
            journey_context=journey_context,
            integration_rationale=integration_plan.rationale
            if integration_plan
            else "",
            page_numbers=state.get("page_numbers", []),
        )

        interactive_blocks = await content_generator_llm.ainvoke_with_structured_output(
            interactive_messages, List[ContentBlockModel]
        )

        blocks.extend(interactive_blocks)

    logging.info(f"Generated {len(blocks)} content blocks for lesson: {lesson.title}")
    return blocks


async def generate_anchors_for_lesson(
    lesson: LessonModel, pdf_content: str, chunk_id: str, page_numbers: List[int]
) -> List[AnchorModel]:
    """Generate SEBI anchors for a lesson based on PDF content and lesson context"""

    try:
        # Extract relevant content sections for anchor creation
        content_sections = extract_content_sections(pdf_content, lesson)

        # Generate different types of anchors
        anchors = []

        # 1. Primary concept anchors
        concept_anchors = await generate_concept_anchors(
            lesson, content_sections, chunk_id, page_numbers
        )
        anchors.extend(concept_anchors)

        # 2. SEBI guideline anchors
        guideline_anchors = await generate_guideline_anchors(
            lesson, content_sections, chunk_id, page_numbers
        )
        anchors.extend(guideline_anchors)

        # 3. Regulatory framework anchors
        framework_anchors = await generate_framework_anchors(
            lesson, content_sections, chunk_id, page_numbers
        )
        anchors.extend(framework_anchors)

        # 4. Practical example anchors
        example_anchors = await generate_example_anchors(
            lesson, content_sections, chunk_id, page_numbers
        )
        anchors.extend(example_anchors)

        # Validate and deduplicate anchors
        anchors = validate_and_deduplicate_anchors(anchors)

        logging.info(f"Generated {len(anchors)} anchors for lesson: {lesson.title}")
        return anchors

    except Exception as e:
        logging.error(f"Failed to generate anchors for lesson {lesson.title}: {str(e)}")
        return []


def extract_content_sections(pdf_content: str, lesson: LessonModel) -> Dict[str, str]:
    """Extract relevant sections from PDF content based on lesson objectives"""

    sections = {}

    # Split content into paragraphs for analysis
    paragraphs = pdf_content.split("\n\n")

    # Create keyword sets for different types of content
    concept_keywords = set()
    for objective in lesson.learning_objectives:
        concept_keywords.update(objective.lower().split())

    guideline_keywords = {
        "sebi",
        "regulation",
        "guideline",
        "circular",
        "notification",
        "compliance",
        "disclosure",
        "requirement",
        "framework",
    }

    example_keywords = {
        "example",
        "illustration",
        "case",
        "scenario",
        "instance",
        "demonstration",
        "application",
        "practical",
    }

    risk_keywords = {
        "risk",
        "riskometer",
        "volatility",
        "diversification",
        "assessment",
        "management",
        "mitigation",
    }

    # Categorize paragraphs based on content
    for para in paragraphs:
        para_lower = para.lower()
        para_words = set(para_lower.split())

        # Check for concept-related content
        if concept_keywords.intersection(para_words):
            if "concepts" not in sections:
                sections["concepts"] = []
            sections["concepts"].append(para)

        # Check for guideline content
        if guideline_keywords.intersection(para_words):
            if "guidelines" not in sections:
                sections["guidelines"] = []
            sections["guidelines"].append(para)

        # Check for examples
        if example_keywords.intersection(para_words):
            if "examples" not in sections:
                sections["examples"] = []
            sections["examples"].append(para)

        # Check for risk-related content
        if risk_keywords.intersection(para_words):
            if "risk" not in sections:
                sections["risk"] = []
            sections["risk"].append(para)

    # Convert lists to strings
    for key, value in sections.items():
        if isinstance(value, list):
            sections[key] = "\n\n".join(value)

    return sections


async def generate_concept_anchors(
    lesson: LessonModel,
    content_sections: Dict[str, str],
    chunk_id: str,
    page_numbers: List[int],
) -> List[AnchorModel]:
    """Generate anchors for key concepts in the lesson"""

    if "concepts" not in content_sections:
        return []

    parser = PydanticOutputParser(pydantic_object=List[AnchorModel])

    concept_anchor_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS[
        "concept_anchors"
    ].partial(format_instructions=parser.get_format_instructions())

    try:
        messages = concept_anchor_prompt.format_messages(
            lesson_title=lesson.title,
            learning_objectives=lesson.learning_objectives,
            content_sections=content_sections.get("concepts", "")[
                :2000
            ],  # Limit length
        )

        response = await content_generator_llm.ainvoke(messages)
        anchors = parser.parse(response.content)

        # Set metadata for all anchors
        for anchor in anchors:
            anchor.created_from_chunk = chunk_id
            anchor.page_numbers = page_numbers
            anchor.last_verified_at = datetime.now()
            anchor.source_type = SourceType.SEBI_PDF
            anchor.relevance_tags.extend(["concept", "definition"])

            if not anchor.confidence_score:
                anchor.confidence_score = 0.8

        return anchors

    except Exception as e:
        logging.error(f"Failed to generate concept anchors: {str(e)}")
        return []


async def generate_guideline_anchors(
    lesson: LessonModel,
    content_sections: Dict[str, str],
    chunk_id: str,
    page_numbers: List[int],
) -> List[AnchorModel]:
    """Generate anchors for SEBI guidelines and regulations"""

    if "guidelines" not in content_sections:
        return []

    parser = PydanticOutputParser(pydantic_object=List[AnchorModel])

    guideline_anchor_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS[
        "guideline_anchors"
    ].partial(format_instructions=parser.get_format_instructions())

    try:
        # Extract focus areas from lesson tags and objectives
        focus_areas = lesson.tags + [
            obj.split()[0] for obj in lesson.learning_objectives
        ]

        messages = guideline_anchor_prompt.format_messages(
            lesson_title=lesson.title,
            focus_areas=focus_areas,
            guideline_content=content_sections.get("guidelines", "")[:2000],
        )

        response = await content_generator_llm.ainvoke(messages)
        anchors = parser.parse(response.content)

        # Set metadata for guideline anchors
        for anchor in anchors:
            anchor.created_from_chunk = chunk_id
            anchor.page_numbers = page_numbers
            anchor.last_verified_at = datetime.now()
            anchor.source_type = determine_sebi_source_type(anchor.excerpt)
            anchor.relevance_tags.extend(["guideline", "regulation", "compliance"])
            anchor.confidence_score = 0.9  # High confidence for regulatory content

        return anchors

    except Exception as e:
        logging.error(f"Failed to generate guideline anchors: {str(e)}")
        return []


async def generate_framework_anchors(
    lesson: LessonModel,
    content_sections: Dict[str, str],
    chunk_id: str,
    page_numbers: List[int],
) -> List[AnchorModel]:
    """Generate anchors for regulatory frameworks and structures"""

    risk_content = content_sections.get("risk", "")
    general_content = content_sections.get("concepts", "")

    if not risk_content and not general_content:
        return []

    framework_content = risk_content + "\n\n" + general_content

    parser = PydanticOutputParser(pydantic_object=List[AnchorModel])

    framework_anchor_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS[
        "framework_anchors"
    ].partial(format_instructions=parser.get_format_instructions())

    try:
        messages = framework_anchor_prompt.format_messages(
            lesson_title=lesson.title,
            learning_objectives=lesson.learning_objectives[
                :3
            ],  # Limit to prevent overflow
            framework_content=framework_content[:2000],
        )

        response = await content_generator_llm.ainvoke(messages)
        anchors = parser.parse(response.content)

        # Set metadata for framework anchors
        for anchor in anchors:
            anchor.created_from_chunk = chunk_id
            anchor.page_numbers = page_numbers
            anchor.last_verified_at = datetime.now()
            anchor.source_type = SourceType.SEBI_PDF
            anchor.relevance_tags.extend(["framework", "structure", "methodology"])
            anchor.confidence_score = 0.85

        return anchors

    except Exception as e:
        logging.error(f"Failed to generate framework anchors: {str(e)}")
        return []


async def generate_example_anchors(
    lesson: LessonModel,
    content_sections: Dict[str, str],
    chunk_id: str,
    page_numbers: List[int],
) -> List[AnchorModel]:
    """Generate anchors for practical examples and illustrations"""

    if "examples" not in content_sections:
        return []

    parser = PydanticOutputParser(pydantic_object=List[AnchorModel])

    example_anchor_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS[
        "example_anchors"
    ].partial(format_instructions=parser.get_format_instructions())

    try:
        messages = example_anchor_prompt.format_messages(
            lesson_title=lesson.title,
            learning_objectives=lesson.learning_objectives,
            example_content=content_sections.get("examples", "")[:2000],
        )

        response = await content_generator_llm.ainvoke(messages)
        anchors = parser.parse(response.content)

        # Set metadata for example anchors
        for anchor in anchors:
            anchor.created_from_chunk = chunk_id
            anchor.page_numbers = page_numbers
            anchor.last_verified_at = datetime.now()
            anchor.source_type = SourceType.SEBI_PDF
            anchor.relevance_tags.extend(["example", "practical", "application"])
            anchor.confidence_score = 0.75

        return anchors

    except Exception as e:
        logging.error(f"Failed to generate example anchors: {str(e)}")
        return []


def determine_sebi_source_type(excerpt: str) -> SourceType:
    """Determine the most appropriate SEBI source type based on content"""

    excerpt_lower = excerpt.lower()

    if "circular" in excerpt_lower:
        return SourceType.SEBI_CIRCULAR
    elif "investor" in excerpt_lower and (
        "portal" in excerpt_lower or "education" in excerpt_lower
    ):
        return SourceType.SEBI_INVESTOR_PORTAL
    elif "nism" in excerpt_lower or "certification" in excerpt_lower:
        return SourceType.NISM_MODULE
    else:
        return SourceType.SEBI_PDF


def validate_and_deduplicate_anchors(anchors: List[AnchorModel]) -> List[AnchorModel]:
    """Validate anchors and remove duplicates based on content similarity"""

    if not anchors:
        return []

    # Remove exact duplicates
    seen_excerpts = set()
    unique_anchors = []

    for anchor in anchors:
        excerpt_key = anchor.excerpt.strip().lower()[:100]  # First 100 chars as key
        if excerpt_key not in seen_excerpts:
            seen_excerpts.add(excerpt_key)
            unique_anchors.append(anchor)

    # Validate anchor content
    validated_anchors = []
    for anchor in unique_anchors:
        if validate_anchor_content(anchor):
            validated_anchors.append(anchor)

    return validated_anchors


def validate_anchor_content(anchor: AnchorModel) -> bool:
    """Validate that an anchor has sufficient and appropriate content"""

    # Check minimum content requirements
    if len(anchor.excerpt.strip()) < 20:
        return False

    if not anchor.title.strip():
        return False

    if not anchor.short_label.strip():
        return False

    # Check for meaningful content (not just whitespace or fragments)
    words = anchor.excerpt.split()
    if len(words) < 5:
        return False

    # Ensure relevance tags are present
    if not anchor.relevance_tags:
        return False

    return True


async def link_blocks_to_anchors(
    blocks: List[ContentBlockModel], anchors: List[AnchorModel]
) -> List[ContentBlockModel]:
    """Link content blocks to relevant anchors based on content similarity"""

    for block in blocks:
        relevant_anchors = find_relevant_anchors_for_block(block, anchors)
        block.anchor_ids = [
            f"{anchor.source_type.value}_{anchor.short_label}"
            for anchor in relevant_anchors
        ]

    return blocks


def find_relevant_anchors_for_block(
    block: ContentBlockModel, anchors: List[AnchorModel]
) -> List[AnchorModel]:
    """Find anchors that are relevant to a specific content block"""

    relevant_anchors = []

    # Extract keywords from block content based on type
    block_keywords = extract_block_keywords(block)

    # Score anchors based on relevance
    anchor_scores = []
    for anchor in anchors:
        score = calculate_anchor_relevance_score(block, anchor, block_keywords)
        if score > 0.3:  # Minimum relevance threshold
            anchor_scores.append((anchor, score))

    # Sort by relevance score and return top matches
    anchor_scores.sort(key=lambda x: x[1], reverse=True)
    relevant_anchors = [
        anchor for anchor, score in anchor_scores[:3]
    ]  # Max 3 anchors per block

    return relevant_anchors


def extract_block_keywords(block: ContentBlockModel) -> set:
    """Extract keywords from a content block based on its type and payload"""

    keywords = set()

    if block.type == BlockType.CONCEPT:
        payload = block.payload
        if hasattr(payload, "heading"):
            keywords.update(payload.heading.lower().split())
        if hasattr(payload, "key_terms"):
            keywords.update([term.lower() for term in payload.key_terms])

    elif block.type == BlockType.EXAMPLE:
        payload = block.payload
        if hasattr(payload, "scenario_title"):
            keywords.update(payload.scenario_title.lower().split())

    elif block.type == BlockType.QUIZ:
        payload = block.payload
        if hasattr(payload, "items"):
            for item in payload.items:
                keywords.update(
                    item.stem.lower().split()[:5]
                )  # First 5 words of question

    # Remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
    }
    keywords = keywords - stop_words

    return keywords


def calculate_anchor_relevance_score(
    block: ContentBlockModel, anchor: AnchorModel, block_keywords: set
) -> float:
    """Calculate relevance score between a block and an anchor"""

    score = 0.0

    # Keyword overlap score
    anchor_words = set(anchor.excerpt.lower().split())
    keyword_overlap = len(block_keywords.intersection(anchor_words))
    if block_keywords:
        score += (keyword_overlap / len(block_keywords)) * 0.4

    # Block type and anchor tag alignment
    type_tag_mapping = {
        BlockType.CONCEPT: ["concept", "definition", "framework"],
        BlockType.EXAMPLE: ["example", "practical", "application"],
        BlockType.QUIZ: ["concept", "definition"],
        BlockType.REFLECTION: ["practical", "application"],
        BlockType.CALLOUT: ["guideline", "regulation", "compliance"],
    }

    if block.type in type_tag_mapping:
        relevant_tags = type_tag_mapping[block.type]
        tag_overlap = len(set(relevant_tags).intersection(set(anchor.relevance_tags)))
        score += (tag_overlap / len(relevant_tags)) * 0.3

    # Anchor confidence boost
    score += anchor.confidence_score * 0.2

    # Lesson-specific boost (if anchor was created for same lesson)
    if anchor.created_from_chunk and block.lesson_id in anchor.created_from_chunk:
        score += 0.1

    return min(score, 1.0)  # Cap at 1.0


async def generate_voice_script_for_lesson(
    lesson: LessonModel, blocks: List[ContentBlockModel], anchors: List[AnchorModel]
) -> Optional[VoiceScriptModel]:
    """Generate voice script for Socratic learning"""

    parser = PydanticOutputParser(pydantic_object=VoiceScriptModel)

    voice_script_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS["voice_script"].partial(
        format_instructions=parser.get_format_instructions()
    )

    try:
        # Prepare summaries
        blocks_summary = []
        for block in blocks[:5]:  # Limit to prevent overflow
            if block.type == BlockType.CONCEPT:
                blocks_summary.append(
                    f"Concept: {getattr(block.payload, 'heading', 'Unknown')}"
                )
            elif block.type == BlockType.EXAMPLE:
                blocks_summary.append(
                    f"Example: {getattr(block.payload, 'scenario_title', 'Unknown')}"
                )

        anchors_summary = [
            f"{anchor.short_label}: {anchor.excerpt[:50]}..." for anchor in anchors[:5]
        ]

        messages = voice_script_prompt.format_messages(
            lesson_title=lesson.title,
            learning_objectives=lesson.learning_objectives,
            duration=lesson.estimated_minutes,
            blocks_summary="; ".join(blocks_summary),
            anchors_summary="; ".join(anchors_summary),
        )

        response = await content_generator_llm.ainvoke(messages)
        voice_script = parser.parse(response.content)

        # Set lesson reference
        voice_script.lesson_id = lesson.slug

        return voice_script

    except Exception as e:
        logging.error(
            f"Failed to generate voice script for lesson {lesson.title}: {str(e)}"
        )
        return None


async def validate_generated_content(
    blocks: List[ContentBlockModel],
    anchors: List[AnchorModel],
    voice_scripts: List[VoiceScriptModel],
) -> Dict[str, Any]:
    """Validate all generated content for quality and consistency"""

    validation_results = {"valid": True, "errors": [], "warnings": []}

    # Validate blocks
    for block in blocks:
        block_validation = validate_content_block(block)
        if not block_validation["valid"]:
            validation_results["errors"].extend(block_validation["errors"])
            validation_results["valid"] = False

    # Validate anchors
    for anchor in anchors:
        if not validate_anchor_content(anchor):
            validation_results["errors"].append(f"Invalid anchor: {anchor.short_label}")
            validation_results["valid"] = False

    # Validate voice scripts
    for script in voice_scripts:
        script_validation = validate_voice_script(script)
        if not script_validation["valid"]:
            validation_results["errors"].extend(script_validation["errors"])
            validation_results["valid"] = False

    # Cross-validation: ensure blocks have anchors
    orphaned_blocks = [block for block in blocks if not block.anchor_ids]
    if orphaned_blocks:
        validation_results["warnings"].append(
            f"{len(orphaned_blocks)} blocks have no anchors"
        )

    return validation_results


def validate_content_block(block: ContentBlockModel) -> Dict[str, Any]:
    """Validate a single content block"""

    result = {"valid": True, "errors": []}

    try:
        # Validate payload based on block type
        if block.type == BlockType.CONCEPT:
            if (
                not hasattr(block.payload, "heading")
                or not block.payload.heading.strip()
            ):
                result["errors"].append("Concept block missing heading")
                result["valid"] = False

        elif block.type == BlockType.EXAMPLE:
            if (
                not hasattr(block.payload, "scenario_md")
                or not block.payload.scenario_md.strip()
            ):
                result["errors"].append("Example block missing scenario")
                result["valid"] = False

        elif block.type == BlockType.QUIZ:
            if not hasattr(block.payload, "items") or not block.payload.items:
                result["errors"].append("Quiz block has no questions")
                result["valid"] = False

        # Validate metadata
        if not block.metadata.source_text:
            result["errors"].append("Block missing source text metadata")
            result["valid"] = False

    except Exception as e:
        result["errors"].append(f"Block validation error: {str(e)}")
        result["valid"] = False

    return result


def validate_voice_script(script: VoiceScriptModel) -> Dict[str, Any]:
    """Validate a voice script"""

    result = {"valid": True, "errors": []}

    try:
        if not script.steps:
            result["errors"].append("Voice script has no steps")
            result["valid"] = False

        for i, step in enumerate(script.steps):
            if not step.prompt.strip():
                result["errors"].append(f"Voice script step {i + 1} has empty prompt")
                result["valid"] = False

            if not step.expected_points:
                result["errors"].append(
                    f"Voice script step {i + 1} has no expected points"
                )
                result["valid"] = False

    except Exception as e:
        result["errors"].append(f"Voice script validation error: {str(e)}")
        result["valid"] = False

    return result
