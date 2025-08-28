"""
Concept block generators for different integration scenarios.
"""

from typing import List, Optional
import logging
from langchain_core.output_parsers import PydanticOutputParser

from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.llm import (
    content_generator_llm,
    get_system_message,
    TaskType,
)
from agents.graphs.content_generation.prompts import CONTENT_GENERATION_PROMPTS
from agents.graphs.content_generation.models import (
    LessonModel,
    ContentAnalysisResult,
    ContentBlockModel,
    ContentBlockMetadata,
    BlockType,
    LessonContentDistribution,
    JourneyCreationPlan,
)


async def generate_concept_blocks_for_extension(
    extended_lesson: LessonModel,
    new_concepts: List[str],
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate concept blocks for new concepts being added to extend an existing lesson"""

    concept_blocks = []

    parser = PydanticOutputParser(pydantic_object=List[ContentBlockModel])

    # Use centralized content generation prompts
    extension_concept_prompt = CONTENT_GENERATION_PROMPTS["concept_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    try:
        prompt = extension_concept_prompt.format(
            lesson_title=extended_lesson.title,
            learning_objectives=extended_lesson.learning_objectives,
            new_concepts=new_concepts,
            sebi_themes=content_analysis.sebi_themes,
            complexity_level=content_analysis.complexity_level,
            content_type=content_analysis.content_type,
            content_sample=state["pdf_content"][:1000],
        )

        response = await content_generator_llm.ainvoke(prompt)
        raw_blocks = parser.parse(response.content)

        # Process and enhance each block
        for i, block in enumerate(raw_blocks):
            if block.type == BlockType.CONCEPT:
                # Ensure proper lesson linking and metadata
                block.lesson_id = extended_lesson.slug
                block.order = (
                    len(extended_lesson.blocks) + i + 1
                )  # Continue from existing blocks

                # Enhance payload with extension context
                if hasattr(block.payload, "rich_text_md"):
                    from ..utils.content_enhancers import enhance_extension_content

                    block.payload.rich_text_md = enhance_extension_content(
                        block.payload.rich_text_md,
                        extended_lesson.title,
                        new_concepts[min(i, len(new_concepts) - 1)],
                    )

                # Set metadata
                block.metadata = ContentBlockMetadata(
                    source_text=f"Extension concept: {new_concepts[min(i, len(new_concepts) - 1)]}",
                    generation_confidence=0.85,
                    manual_review_needed=False,
                    integration_notes=f"Added to extend lesson with new concept: {new_concepts[min(i, len(new_concepts) - 1)]}",
                )

                concept_blocks.append(block)

        logging.info(
            f"Generated {len(concept_blocks)} concept blocks for lesson extension"
        )
        return concept_blocks

    except Exception as e:
        logging.error(f"Failed to generate concept blocks for extension: {str(e)}")
        return []


async def generate_enhanced_concept_block(
    lesson: LessonModel,
    overlap_concept: str,
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> Optional[ContentBlockModel]:
    """Generate enhanced concept block that merges overlapping concepts"""

    parser = PydanticOutputParser(pydantic_object=ContentBlockModel)

    # Use centralized content generation prompts
    enhanced_concept_prompt = CONTENT_GENERATION_PROMPTS["concept_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    try:
        prompt = enhanced_concept_prompt.format(
            lesson_title=lesson.title,
            overlap_concept=overlap_concept,
            key_concepts=content_analysis.key_concepts,
            sebi_themes=content_analysis.sebi_themes,
            learning_opportunities=content_analysis.learning_opportunities,
            content_sample=state["pdf_content"][:1000],
        )

        response = await content_generator_llm.ainvoke(prompt)
        enhanced_block = parser.parse(response.content)

        # Configure enhanced concept block
        enhanced_block.lesson_id = lesson.slug
        enhanced_block.type = BlockType.CONCEPT
        enhanced_block.order = 10  # Place early in lesson

        # Add enhancement context to content
        if hasattr(enhanced_block.payload, "rich_text_md"):
            from ..utils.content_enhancers import add_enhancement_context

            enhanced_block.payload.rich_text_md = add_enhancement_context(
                enhanced_block.payload.rich_text_md, overlap_concept
            )

        # Set metadata
        enhanced_block.metadata = ContentBlockMetadata(
            source_text=f"Enhanced overlapping concept: {overlap_concept}",
            generation_confidence=0.9,
            integration_notes=f"Enhanced understanding of: {overlap_concept}",
        )

        return enhanced_block

    except Exception as e:
        logging.error(f"Failed to generate enhanced concept block: {str(e)}")
        return None


async def generate_concept_blocks_from_distribution(
    lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate concept blocks using structured lesson distribution data"""

    concept_blocks = []

    parser = PydanticOutputParser(pydantic_object=List[ContentBlockModel])

    # Use centralized content generation prompts with structured data
    distribution_concept_prompt = CONTENT_GENERATION_PROMPTS["concept_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    # Build journey context for better alignment
    journey_context = ""
    if journey_plan:
        journey_context = f"""
Journey Context:
- Journey id: {journey_plan.slug}
- Journey title: {journey_plan.title}
- Level: {journey_plan.level}
- Target Audience: {journey_plan.target_audience}
- Key Topics: {", ".join(journey_plan.key_topics)}
- Prerequisites: {", ".join(journey_plan.prerequisites)}
"""

    try:
        prompt = distribution_concept_prompt.format(
            lesson_title=lesson_dist.lesson_title,
            concepts_to_cover=lesson_dist.concepts_to_cover,
            learning_objectives=lesson_dist.learning_objectives,
            integration_type=lesson_dist.integration_type,
            estimated_duration=lesson_dist.estimated_duration_minutes,
            prerequisite_concepts=lesson_dist.prerequisite_concepts,
            journey_context=journey_context,
            content_sample=state["pdf_content"][:1000],
        )

        response = await content_generator_llm.ainvoke(prompt)
        raw_blocks = parser.parse(response.content)

        # Process and enhance each block with distribution context
        for i, block in enumerate(raw_blocks):
            if block.type == BlockType.CONCEPT:
                # Ensure proper lesson linking and metadata
                block.lesson_id = lesson.slug
                block.order = (i + 1) * 10  # Space blocks for insertion

                # Enhance payload with distribution context
                if hasattr(block.payload, "rich_text_md"):
                    from ..utils.content_enhancers import enhance_distribution_content

                    block.payload.rich_text_md = enhance_distribution_content(
                        block.payload.rich_text_md,
                        lesson_dist.concepts_to_cover[
                            min(i, len(lesson_dist.concepts_to_cover) - 1)
                        ],
                        lesson_dist.learning_objectives,
                        journey_context,
                    )

                # Set metadata with distribution information
                block.metadata = ContentBlockMetadata(
                    source_text=f"Distribution concept: {lesson_dist.concepts_to_cover[min(i, len(lesson_dist.concepts_to_cover) - 1)]}",
                    generation_confidence=0.9,
                    manual_review_needed=False,
                    integration_notes=f"Generated from structured distribution for {lesson_dist.integration_type}",
                )

                concept_blocks.append(block)

        logging.info(
            f"Generated {len(concept_blocks)} concept blocks from distribution for {lesson_dist.lesson_title}"
        )
        return concept_blocks

    except Exception as e:
        logging.error(f"Failed to generate concept blocks from distribution: {str(e)}")
        return []


async def generate_targeted_concept_blocks(
    lesson: LessonModel,
    target_concepts: List[str],
    learning_objectives: List[str],
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate concept blocks for specific target concepts with learning objectives"""

    concept_blocks = []

    parser = PydanticOutputParser(pydantic_object=List[ContentBlockModel])

    # Use centralized content generation prompts
    targeted_concept_prompt = CONTENT_GENERATION_PROMPTS["concept_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    # Build journey context
    journey_context = ""
    if journey_plan:
        journey_context = f"""
Journey Context:

- Journey id: {journey_plan.slug}
- Journey Title: {journey_plan.title}
- Level: {journey_plan.level}
- Target Audience: {journey_plan.target_audience}
- Total Duration: {journey_plan.total_estimated_hours} hours
"""

    try:
        prompt = targeted_concept_prompt.format(
            lesson_title=lesson.title,
            target_concepts=target_concepts,
            learning_objectives=learning_objectives,
            journey_context=journey_context,
            content_sample=state["pdf_content"][:1000],
        )

        response = await content_generator_llm.ainvoke(prompt)
        raw_blocks = parser.parse(response.content)

        # Process each targeted concept block
        for i, block in enumerate(raw_blocks):
            if block.type == BlockType.CONCEPT:
                block.lesson_id = lesson.slug
                block.order = (i + 1) * 15  # Space for targeted insertion

                # Enhance with targeted context
                if hasattr(block.payload, "rich_text_md"):
                    from ..utils.content_enhancers import enhance_targeted_concept

                    block.payload.rich_text_md = enhance_targeted_concept(
                        block.payload.rich_text_md,
                        target_concepts[min(i, len(target_concepts) - 1)],
                        learning_objectives,
                    )

                # Set metadata for targeted generation
                block.metadata = ContentBlockMetadata(
                    source_text=f"Targeted concept: {target_concepts[min(i, len(target_concepts) - 1)]}",
                    generation_confidence=0.85,
                    integration_notes="Targeted for specific learning objective alignment",
                )

                concept_blocks.append(block)

        logging.info(f"Generated {len(concept_blocks)} targeted concept blocks")
        return concept_blocks

    except Exception as e:
        logging.error(f"Failed to generate targeted concept blocks: {str(e)}")
        return []


async def generate_concept_block_for_merge(
    merged_lesson: LessonModel,
    concept: str,
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> Optional[ContentBlockModel]:
    """Generate a concept block for merged content that doesn't overlap with existing concepts"""

    parser = PydanticOutputParser(pydantic_object=ContentBlockModel)

    # Use centralized content generation prompts
    merge_concept_prompt = CONTENT_GENERATION_PROMPTS["concept_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    try:
        prompt = merge_concept_prompt.format(
            lesson_title=merged_lesson.title,
            concept=concept,
            key_concepts=content_analysis.key_concepts,
            sebi_themes=content_analysis.sebi_themes,
            learning_opportunities=content_analysis.learning_opportunities,
            content_sample=state["pdf_content"][:1000],
        )

        response = await content_generator_llm.ainvoke(prompt)
        concept_block = parser.parse(response.content)

        # Configure block for merged lesson
        concept_block.lesson_id = merged_lesson.slug
        concept_block.type = BlockType.CONCEPT
        concept_block.order = 50  # Place in middle section

        # Enhance concept content for merge context
        if hasattr(concept_block.payload, "rich_text_md"):
            from ..utils.content_enhancers import enhance_merged_concept_content

            concept_block.payload.rich_text_md = enhance_merged_concept_content(
                concept_block.payload.rich_text_md, concept, merged_lesson.title
            )

        # Set metadata
        concept_block.metadata = ContentBlockMetadata(
            source_text=f"New concept in merged lesson: {concept}",
            generation_confidence=0.8,
            integration_notes=f"Non-overlapping concept added during merge: {concept}",
        )

        logging.info(f"Generated concept block for merge: {concept}")
        return concept_block

    except Exception as e:
        logging.error(f"Failed to generate concept block for merge: {str(e)}")
        return None
