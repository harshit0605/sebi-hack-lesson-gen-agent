"""
Example block generators for different integration scenarios.
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
    ContentBlockModel,
    ContentBlockMetadata,
    BlockType,
    ExampleBlockPayload,
    ExistingContentMapping,
    ContentAnalysisResult,
    LessonContentDistribution,
    JourneyCreationPlan,
)


async def generate_example_blocks_for_extension(
    extended_lesson: LessonModel, sebi_themes: List[str], state: LessonCreationState
) -> List[ContentBlockModel]:
    """Generate example blocks for new SEBI themes being added to extend a lesson"""

    example_blocks = []

    parser = PydanticOutputParser(pydantic_object=List[ContentBlockModel])

    # Use centralized content generation prompts
    extension_example_prompt = CONTENT_GENERATION_PROMPTS["example_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    try:
        prompt = extension_example_prompt.format(
            lesson_title=extended_lesson.title,
            learning_objectives=extended_lesson.learning_objectives,
            sebi_themes=sebi_themes,
            content_sample=state["pdf_content"][:1200],
            page_numbers=state["page_numbers"],
        )

        response = await content_generator_llm.ainvoke(prompt)
        raw_blocks = parser.parse(response.content)

        # Process each example block
        for i, block in enumerate(raw_blocks):
            if block.type == BlockType.EXAMPLE:
                # Set proper lesson linking
                block.lesson_id = extended_lesson.slug
                block.order = (
                    len(extended_lesson.blocks) + 100 + i
                )  # Place after concepts

                # Enhance example payload
                if hasattr(block.payload, "scenario_md"):
                    from ..utils.content_enhancers import enhance_example_scenario

                    block.payload.scenario_md = enhance_example_scenario(
                        block.payload.scenario_md,
                        sebi_themes[min(i, len(sebi_themes) - 1)],
                    )

                # Ensure Indian context flag
                block.payload.indian_context = True

                # Set metadata
                block.metadata = ContentBlockMetadata(
                    source_text=f"Extension example for SEBI theme: {sebi_themes[min(i, len(sebi_themes) - 1)]}",
                    generation_confidence=0.8,
                    integration_notes=f"Example added to illustrate: {sebi_themes[min(i, len(sebi_themes) - 1)]}",
                )

                example_blocks.append(block)

        logging.info(
            f"Generated {len(example_blocks)} example blocks for lesson extension"
        )
        return example_blocks

    except Exception as e:
        logging.error(f"Failed to generate example blocks for extension: {str(e)}")
        return []


async def generate_comparison_example_block(
    lesson: LessonModel,
    mapping: ExistingContentMapping,
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> Optional[ContentBlockModel]:
    """Generate example block comparing old and new understanding"""

    try:
        block = ContentBlockModel(
            lesson_id=lesson.slug,
            type=BlockType.EXAMPLE,
            order=2,
            payload=ExampleBlockPayload(
                scenario_title="Evolution of Understanding",
                scenario_md=f"Compare traditional understanding with enhanced SEBI-aligned perspectives on {', '.join(mapping.overlap_concepts[:2])}.",
                qa_pairs=[
                    {
                        "question": "What's the traditional view?",
                        "answer": "Traditional perspective based on general market knowledge.",
                    },
                    {
                        "question": "How does SEBI guidance enhance this?",
                        "answer": "SEBI guidelines provide additional regulatory context and investor protection measures.",
                    },
                ],
                indian_context=True,
            ),
            anchor_ids=[],
            metadata=ContentBlockMetadata(
                source_text=f"Comparison example for overlapping concepts: {', '.join(mapping.overlap_concepts)}",
                generation_confidence=0.7,
                integration_notes="Highlights evolution from existing to merged understanding",
            ),
        )
        return block
    except Exception as e:
        logging.error(f"Failed to generate comparison example block: {str(e)}")
        return None


async def generate_example_blocks_from_distribution(
    lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate example blocks using structured lesson distribution data"""

    example_blocks = []

    parser = PydanticOutputParser(pydantic_object=List[ContentBlockModel])

    # Use centralized content generation prompts with structured data
    distribution_example_prompt = CONTENT_GENERATION_PROMPTS["example_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    # Build journey context for better alignment
    journey_context = ""
    if journey_plan:
        journey_context = f"""
Journey Context:
- Journey id: {journey_plan.slug}
- Journey Title: {journey_plan.title}
- Level: {journey_plan.level}
- Target Audience: {journey_plan.target_audience}
- Key Topics: {", ".join(journey_plan.key_topics)}
- Prerequisites: {", ".join(journey_plan.prerequisites)}
"""

    try:
        prompt = distribution_example_prompt.format(
            lesson_title=lesson_dist.lesson_title,
            concepts_to_cover=lesson_dist.concepts_to_cover,
            learning_objectives=lesson_dist.learning_objectives,
            integration_type=lesson_dist.integration_type,
            estimated_duration=lesson_dist.estimated_duration_minutes,
            prerequisite_concepts=lesson_dist.prerequisite_concepts,
            journey_context=journey_context,
            content_sample=state["pdf_content"][:1200],
            page_numbers=state["page_numbers"],
        )

        response = await content_generator_llm.ainvoke(prompt)
        raw_blocks = parser.parse(response.content)

        # Process and enhance each block with distribution context
        for i, block in enumerate(raw_blocks):
            if block.type == BlockType.EXAMPLE:
                # Ensure proper lesson linking and metadata
                block.lesson_id = lesson.slug
                block.order = 100 + (i * 10)  # Place after concepts

                # Enhance example payload with distribution context
                if hasattr(block.payload, "scenario_md"):
                    from ..utils.content_enhancers import enhance_distribution_example

                    block.payload.scenario_md = enhance_distribution_example(
                        block.payload.scenario_md,
                        lesson_dist.concepts_to_cover,
                        lesson_dist.learning_objectives,
                        journey_context,
                    )

                # Ensure Indian context flag
                block.payload.indian_context = True

                # Set metadata with distribution information
                block.metadata = ContentBlockMetadata(
                    source_text=f"Distribution example for {lesson_dist.integration_type}",
                    generation_confidence=0.9,
                    integration_notes=f"Generated from structured distribution with {len(lesson_dist.concepts_to_cover)} concepts",
                )

                example_blocks.append(block)

        logging.info(
            f"Generated {len(example_blocks)} example blocks from distribution for {lesson_dist.lesson_title}"
        )
        return example_blocks

    except Exception as e:
        logging.error(f"Failed to generate example blocks from distribution: {str(e)}")
        return []


async def generate_objective_aligned_examples(
    lesson: LessonModel,
    learning_objectives: List[str],
    target_concepts: List[str],
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate example blocks specifically aligned with learning objectives"""

    example_blocks = []

    parser = PydanticOutputParser(pydantic_object=List[ContentBlockModel])

    # Use centralized content generation prompts
    objective_example_prompt = CONTENT_GENERATION_PROMPTS["example_block"].partial(
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
        prompt = objective_example_prompt.format(
            lesson_title=lesson.title,
            learning_objectives=learning_objectives,
            target_concepts=target_concepts,
            journey_context=journey_context,
            content_sample=state["pdf_content"][:1200],
            page_numbers=state["page_numbers"],
        )

        response = await content_generator_llm.ainvoke(prompt)
        raw_blocks = parser.parse(response.content)

        # Process each objective-aligned example block
        for i, block in enumerate(raw_blocks):
            if block.type == BlockType.EXAMPLE:
                block.lesson_id = lesson.slug
                block.order = 150 + (i * 10)  # Place after distribution examples

                # Enhance with objective alignment
                if hasattr(block.payload, "scenario_md"):
                    from ..utils.content_enhancers import enhance_objective_example

                    block.payload.scenario_md = enhance_objective_example(
                        block.payload.scenario_md,
                        learning_objectives[min(i, len(learning_objectives) - 1)],
                        target_concepts,
                    )

                # Ensure Indian context
                block.payload.indian_context = True

                # Set metadata for objective alignment
                block.metadata = ContentBlockMetadata(
                    source_text=f"Objective-aligned example: {learning_objectives[min(i, len(learning_objectives) - 1)]}",
                    generation_confidence=0.85,
                    integration_notes="Generated to specifically address learning objectives",
                )

                example_blocks.append(block)

        logging.info(
            f"Generated {len(example_blocks)} objective-aligned example blocks"
        )
        return example_blocks

    except Exception as e:
        logging.error(f"Failed to generate objective-aligned example blocks: {str(e)}")
        return []
