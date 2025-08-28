"""
Reusable content generation utilities for lesson creation.

This module provides functions to generate content blocks, anchors, and voice scripts
that can be used across different lesson creation strategies.
"""

from typing import List, Dict, Any
import logging
# datetime import removed - not needed for simplified anchor model

from agents.graphs.content_generation.models import (
    LessonModel,
    ContentBlockModel,
    AnchorModel,
    VoiceScriptModel,
    ContentAnalysisResult,
    BlockType,
    SourceType,
    LessonContentBlocks,
)
from typing import Optional
from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.llm import generate_lessons_llm
from agents.graphs.content_generation.prompts.generate_structured_content import (
    GENERATE_STRUCTURED_CONTENT_PROMPTS,
)


async def generate_complete_lesson_content(
    lesson: LessonModel,
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> Dict[str, Any]:
    """
    Generate all content for a lesson including blocks, anchors, and voice scripts.

    Returns:
        Dict containing 'blocks', 'anchors', 'voice_scripts' lists
    """

    # Generate content blocks and anchors together in one LLM call
    blocks, anchors = await generate_blocks_for_lesson(lesson, content_analysis, state)

    # Ensure each block is linked to at least one anchor at block level
    blocks = await link_blocks_to_anchors(blocks, anchors)

    # Generate voice script
    voice_scripts = []
    voice_script = await generate_voice_script_for_lesson(lesson, blocks, anchors)
    if voice_script:
        voice_scripts.append(voice_script)

    return {"blocks": blocks, "anchors": anchors, "voice_scripts": voice_scripts}


async def generate_blocks_for_lesson(
    lesson: LessonModel,
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> tuple[List[LessonContentBlocks], List[AnchorModel]]:
    """Generate all content blocks for a lesson in a single LLM call"""

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

    # Use unified prompt to generate all blocks and anchors in one call
    unified_prompt = GENERATE_STRUCTURED_CONTENT_PROMPTS["unified_lesson_content"]

    messages = unified_prompt.format_messages(
        lesson_title=lesson.title,
        lesson_slug=lesson.slug,
        objectives=lesson.learning_objectives,
        key_concepts=content_analysis.key_concepts,
        sebi_themes=content_analysis.sebi_themes,
        learning_opportunities=content_analysis.learning_opportunities,
        journey_context=journey_context,
        integration_rationale=integration_plan.rationale if integration_plan else "",
        page_numbers=state.get("page_numbers", []),
        pdf_content=state.get("pdf_content", ""),
        chunk_id=state.get("chunk_id", ""),
    )

    # Single LLM call to generate all content blocks
    lesson_content = await generate_lessons_llm.ainvoke_with_structured_output(
        messages, LessonContentBlocks
    )

    # Combine all blocks into a single list
    all_blocks = []
    all_blocks.extend(lesson_content.concept_blocks)
    all_blocks.extend(lesson_content.example_blocks)
    all_blocks.extend(lesson_content.quiz_blocks)
    all_blocks.extend(lesson_content.reflection_blocks)
    all_blocks.extend(lesson_content.interactive_blocks)

    logging.info(
        f"Generated {len(all_blocks)} content blocks and {len(lesson_content.anchors)} anchors for lesson: {lesson.title}"
    )
    return all_blocks, lesson_content.anchors


# Individual block generation functions removed - now using unified generation


async def generate_anchors_for_lesson(
    lesson: LessonModel, pdf_content: str, chunk_id: str, page_numbers: List[int]
) -> List[AnchorModel]:
    """Generate SEBI anchors for a lesson based on PDF content"""

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

        logging.info(f"Generated {len(anchors)} anchors for lesson: {lesson.title}")
        return anchors

    except Exception as e:
        logging.error(f"Failed to generate anchors for lesson {lesson.title}: {str(e)}")
        return []


def extract_content_sections(pdf_content: str, lesson: LessonModel) -> Dict[str, str]:
    """Extract relevant content sections from PDF based on lesson objectives"""

    # Simple implementation - can be enhanced with more sophisticated extraction
    sections = {
        "full_content": pdf_content,
        "lesson_relevant": pdf_content[:2000],  # First 2000 chars as relevant section
    }

    return sections


async def generate_concept_anchors(
    lesson: LessonModel,
    content_sections: Dict[str, str],
    chunk_id: str,
    page_numbers: List[int],
) -> List[AnchorModel]:
    """Generate concept-based anchors"""

    anchors = []

    for i, objective in enumerate(lesson.learning_objectives[:3]):  # Limit to 3 anchors
        anchor = AnchorModel(
            short_label=f"concept_{i + 1}",
            excerpt=content_sections.get("lesson_relevant", "")[:500],
            source_type=SourceType.SEBI_PDF,
            created_from_chunk=chunk_id,
        )
        anchors.append(anchor)

    return anchors


async def generate_guideline_anchors(
    lesson: LessonModel,
    content_sections: Dict[str, str],
    chunk_id: str,
    page_numbers: List[int],
) -> List[AnchorModel]:
    """Generate SEBI guideline-based anchors"""

    anchors = []

    # Create guideline anchor if content seems regulatory
    if any(
        keyword in content_sections.get("full_content", "").lower()
        for keyword in ["sebi", "regulation", "guideline", "compliance"]
    ):
        anchor = AnchorModel(
            short_label="sebi_guideline",
            excerpt=content_sections.get("lesson_relevant", "")[:500],
            source_type=SourceType.SEBI_PDF,
            created_from_chunk=chunk_id,
        )
        anchors.append(anchor)

    return anchors


async def generate_framework_anchors(
    lesson: LessonModel,
    content_sections: Dict[str, str],
    chunk_id: str,
    page_numbers: List[int],
) -> List[AnchorModel]:
    """Generate regulatory framework anchors"""

    anchors = []

    # Create framework anchor for structural content
    if any(
        keyword in content_sections.get("full_content", "").lower()
        for keyword in ["framework", "structure", "process", "procedure"]
    ):
        anchor = AnchorModel(
            short_label="regulatory_framework",
            excerpt=content_sections.get("lesson_relevant", "")[:500],
            source_type=SourceType.SEBI_PDF,
            created_from_chunk=chunk_id,
        )
        anchors.append(anchor)

    return anchors


async def link_blocks_to_anchors(
    blocks: List[ContentBlockModel], anchors: List[AnchorModel]
) -> List[ContentBlockModel]:
    """Link content blocks to relevant anchors"""

    # Simple linking strategy with quiz-item aggregation
    available_labels = [a.short_label for a in anchors]
    for i, block in enumerate(blocks):
        # Aggregate quiz item anchors if present
        try:
            if block.type == BlockType.QUIZ and hasattr(block, "payload") and hasattr(block.payload, "items"):
                aggregated: List[str] = []
                for item in block.payload.items:
                    for aid in getattr(item, "anchor_ids", []) or []:
                        if aid in available_labels and aid not in aggregated:
                            aggregated.append(aid)
                if aggregated:
                    block.anchor_ids = aggregated
                    continue  # already satisfied for this block
        except Exception:
            # Fall through to default linking if any issues
            pass

        # Default positional linking to ensure at least one anchor
        if not getattr(block, "anchor_ids", []):
            if anchors:
                linked = anchors[i % len(anchors)].short_label
                block.anchor_ids = [linked]

    return blocks


async def generate_voice_script_for_lesson(
    lesson: LessonModel,
    blocks: List[ContentBlockModel],
    anchors: List[AnchorModel],
) -> Optional[VoiceScriptModel]:
    """Generate voice script for lesson based on content blocks"""

    try:
        from agents.graphs.content_generation.models import VoiceStep, VoiceCheckpoint

        # Create voice steps based on lesson objectives
        steps = []
        for i, objective in enumerate(
            lesson.learning_objectives[:4]
        ):  # Limit to 4 steps
            step = VoiceStep(
                step_number=i + 1,
                prompt=f"Let's explore: {objective}",
                expected_points=[objective],
                hints=["Think about how this applies to your financial decisions"],
                checkpoint=VoiceCheckpoint(
                    question=f"Can you explain {objective.split()[0]} in your own words?",
                    pass_criteria=[
                        "Shows understanding of the concept",
                        "Can relate to personal experience",
                    ],
                    feedback_positive="Excellent! You've grasped this concept well.",
                    feedback_negative="Let's review this concept with some examples.",
                ),
                anchor_ids=[anchor.short_label for anchor in anchors[:2]]
                if anchors
                else [],
                estimated_duration_seconds=300,
            )
            steps.append(step)

        # Ensure we have at least one step
        if not steps:
            steps = [
                VoiceStep(
                    step_number=1,
                    prompt=f"Welcome to {lesson.title}. Let's begin learning together.",
                    expected_points=["Basic understanding"],
                    hints=["Take your time to think"],
                    checkpoint=VoiceCheckpoint(
                        question="Are you ready to start learning?",
                        pass_criteria=["Shows engagement"],
                        feedback_positive="Great! Let's continue.",
                        feedback_negative="No worries, we'll go step by step.",
                    ),
                    anchor_ids=[],
                    estimated_duration_seconds=180,
                )
            ]

        voice_script = VoiceScriptModel(
            lesson_id=lesson.slug,
            steps=steps,
            total_estimated_minutes=lesson.estimated_minutes,
            difficulty_adjustments=[],
        )

        return voice_script

    except Exception as e:
        logging.error(
            f"Failed to generate voice script for lesson {lesson.title}: {str(e)}"
        )
        return None


def populate_lesson_with_content(
    lesson: LessonModel, content_data: Dict[str, Any]
) -> LessonModel:
    """Populate lesson model with generated content references with consistent IDs"""

    blocks = content_data.get("blocks", [])
    anchors = content_data.get("anchors", [])
    voice_scripts = content_data.get("voice_scripts", [])

    # Generate consistent block IDs based on lesson slug and block type
    lesson.blocks = [
        f"{lesson.slug}_block_{block.type.value}_{i}" for i, block in enumerate(blocks)
    ]

    # Generate consistent anchor IDs based on lesson slug and anchor type
    lesson.anchors = [anchor.short_label for anchor in anchors]

    # Set voice script with consistent ID
    if voice_scripts:
        lesson.voice_script_id = f"{lesson.slug}_voice_script"
        lesson.voice_ready = True

    # Generate consistent quiz and interactive IDs
    quiz_counter = 0
    interactive_counter = 0

    lesson.quiz_ids = []
    lesson.interactive_ids = []

    for i, block in enumerate(blocks):
        if block.type == BlockType.QUIZ:
            lesson.quiz_ids.append(f"{lesson.slug}_quiz_{quiz_counter}")
            quiz_counter += 1
        elif block.type == BlockType.INTERACTIVE:
            lesson.interactive_ids.append(
                f"{lesson.slug}_interactive_{interactive_counter}"
            )
            interactive_counter += 1

    return lesson
