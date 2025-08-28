"""
Strategy for extending existing lessons with new content.
"""

from typing import List, Optional
import logging

from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    LessonModel,
    LessonCreationModel,
    ContentAnalysisResult,
    ContentIntegrationPlan,
    ContentIntegrationAction,
    ContentBlockModel,
    LessonContentDistribution,
    JourneyCreationPlan,
)
from agents.graphs.content_generation.llm import lesson_creator_llm
from agents.graphs.content_generation.prompts import LESSON_CREATION_PROMPTS

# Import block generators - will be created in blocks/ directory
from agents.graphs.content_generation.nodes.blocks.concept_blocks import (
    generate_concept_blocks_for_extension,
)
from agents.graphs.content_generation.nodes.blocks.example_blocks import (
    generate_example_blocks_for_extension,
)
from agents.graphs.content_generation.nodes.blocks.reflection_blocks import (
    generate_integration_reflection_block,
)


async def extend_existing_lessons(state: LessonCreationState) -> LessonCreationState:
    """Extend existing lessons using structured content distribution"""

    integration_plan = state["integration_plan"]
    existing_lessons = state["existing_lessons"]
    journey_plan = state.get("journey_creation_plan")
    existing_journeys = state.get("existing_journeys", [])

    # Get lesson distributions for extensions
    extension_distributions = [
        dist
        for dist in integration_plan.content_distribution
        if dist.integration_type == "extend_existing"
    ]

    updated_lessons = []

    for lesson_dist in extension_distributions:
        # Find the existing lesson by ID
        existing_lesson = next(
            (
                lesson
                for lesson in existing_lessons
                if str(lesson.slug) == lesson_dist.lesson_id
                or str(lesson.title) == lesson_dist.lesson_id
            ),
            None,
        )

        if not existing_lesson:
            state["validation_errors"].append(
                f"Target lesson not found: {lesson_dist.lesson_id}"
            )
            continue

        try:
            # Create extension using structured distribution
            extended_lesson = await create_lesson_extension_from_distribution(
                existing_lesson, lesson_dist, integration_plan, journey_plan, state
            )

            if extended_lesson:
                updated_lessons.append(extended_lesson)
                logging.info(
                    f"Extended lesson: {existing_lesson.title} with structured content"
                )

        except Exception as e:
            state["validation_errors"].append(
                f"Failed to extend lesson {existing_lesson.title}: {str(e)}"
            )

    state["updated_lessons"] = updated_lessons
    return state


async def create_lesson_extension_from_distribution(
    existing_lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    integration_plan: ContentIntegrationPlan,
    journey_plan: JourneyCreationPlan,
    state: LessonCreationState,
) -> Optional[LessonModel]:
    """Create an extended version using structured content distribution"""

    prompt_template = LESSON_CREATION_PROMPTS["extend_lesson"]

    # Use structured data from distribution
    journey_context = ""
    if journey_plan:
        journey_context = f"""
        Journey Context:
        - Journey id: {journey_plan.slug}
        - Journey Title: {journey_plan.title}
        - Level: {journey_plan.level}
        - Target Audience: {journey_plan.target_audience}
        """

    # Get content analysis for raw content context
    content_analysis = state.get("content_analysis")

    prompt = prompt_template.format_messages(
        lesson_title=existing_lesson.title,
        current_objectives=existing_lesson.learning_objectives,
        current_duration=existing_lesson.estimated_minutes,
        concepts_to_cover=lesson_dist.concepts_to_cover,
        learning_objectives=lesson_dist.learning_objectives,
        estimated_duration=lesson_dist.estimated_duration_minutes,
        prerequisite_concepts=lesson_dist.prerequisite_concepts,
        key_concepts=content_analysis.key_concepts if content_analysis else [],
        sebi_themes=content_analysis.sebi_themes if content_analysis else [],
        learning_opportunities=content_analysis.learning_opportunities
        if content_analysis
        else [],
        journey_context=journey_context,
        integration_rationale=integration_plan.rationale,
        page_numbers=state["page_numbers"],
    )

    try:
        lesson_creation = await lesson_creator_llm.ainvoke_with_structured_output(
            prompt, LessonCreationModel
        )

        # Convert to full LessonModel with empty content fields (will be populated later)
        extended_lesson = LessonModel(
            **lesson_creation.model_dump(),
            blocks=[],  # Will be populated by generate_complete_lessons node
            anchors=[],  # Will be populated by generate_complete_lessons node
            voice_ready=False,
            voice_script_id=None,  # Will be set by generate_complete_lessons node
            quiz_ids=[],  # Will be populated by generate_complete_lessons node
            interactive_ids=[],  # Will be populated by generate_complete_lessons node
        )

        # Preserve original IDs and update metadata
        extended_lesson.journey_id = existing_lesson.journey_id
        extended_lesson.version = existing_lesson.version + 1
        extended_lesson.estimated_minutes = (
            existing_lesson.estimated_minutes + lesson_dist.estimated_duration_minutes
        )

        # Merge learning objectives
        extended_lesson.learning_objectives = list(
            set(existing_lesson.learning_objectives + lesson_dist.learning_objectives)
        )

        # Update prerequisites
        extended_lesson.prerequisites = list(
            set(existing_lesson.prerequisites + lesson_dist.prerequisite_concepts)
        )

        extended_lesson.metadata.integration_action = (
            ContentIntegrationAction.EXTEND_EXISTING_LESSON
        )
        extended_lesson.metadata.related_existing_lessons = [existing_lesson.slug]
        extended_lesson.metadata.source_pages.extend(state["page_numbers"])
        extended_lesson.metadata.chunk_id = state["chunk_id"]

        # Generate extension blocks using structured data
        extension_blocks = await generate_extension_blocks_from_distribution(
            extended_lesson, existing_lesson, lesson_dist, state
        )
        if extension_blocks:
            extended_lesson.blocks.extend(
                [block._id for block in extension_blocks if hasattr(block, "_id")]
            )

            if "content_blocks" not in state:
                state["content_blocks"] = []
            state["content_blocks"].extend(extension_blocks)

        return extended_lesson

    except Exception as e:
        logging.error(f"Failed to create lesson extension: {str(e)}")
        return None


async def generate_extension_blocks_from_distribution(
    extended_lesson: LessonModel,
    original_lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate extension blocks using structured lesson distribution"""

    extension_blocks = []

    # Generate concept blocks for specific concepts from distribution
    if lesson_dist.concepts_to_cover:
        concept_blocks = await generate_concept_blocks_for_extension(
            extended_lesson, lesson_dist.concepts_to_cover, lesson_dist, state
        )
        extension_blocks.extend(concept_blocks)

    # Generate example blocks based on learning objectives
    if lesson_dist.learning_objectives:
        example_blocks = await generate_example_blocks_for_extension(
            extended_lesson, lesson_dist.learning_objectives, state
        )
        extension_blocks.extend(example_blocks)

    # Generate integration reflection block
    reflection_block = await generate_integration_reflection_block(
        extended_lesson, original_lesson, state
    )
    if reflection_block:
        extension_blocks.append(reflection_block)

    return extension_blocks
