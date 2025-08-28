"""
Strategy for merging new content with existing lessons when there's significant overlap.
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
    ExistingContentMapping,
)
from agents.graphs.content_generation.llm import lesson_creator_llm
from agents.graphs.content_generation.prompts import LESSON_CREATION_PROMPTS

# Import block generators
from agents.graphs.content_generation.nodes.blocks.concept_blocks import (
    generate_enhanced_concept_block,
)
from agents.graphs.content_generation.nodes.blocks.example_blocks import (
    generate_comparison_example_block,
)
from agents.graphs.content_generation.nodes.blocks.quiz_blocks import (
    generate_merged_quiz_block,
)


async def merge_with_existing_lessons(
    state: LessonCreationState,
) -> LessonCreationState:
    """Merge content using structured content distribution"""

    integration_plan = state["integration_plan"]
    existing_lessons = state["existing_lessons"]
    journey_plan = state.get("journey_creation_plan")
    existing_journeys = state.get("existing_journeys", [])

    # Get lesson distributions for merging
    merge_distributions = [
        dist
        for dist in integration_plan.content_distribution
        if dist.integration_type == "merge_content"
    ]

    updated_lessons = []

    for lesson_dist in merge_distributions:
        existing_lesson = next(
            (
                lesson
                for lesson in existing_lessons
                if lesson.slug == lesson_dist.lesson_id
            ),
            None,
        )

        if not existing_lesson:
            state["validation_errors"].append(
                f"Target lesson not found: {lesson_dist.lesson_id}"
            )
            continue

        try:
            merged_lesson = await create_merged_lesson_from_distribution(
                existing_lesson, lesson_dist, integration_plan, journey_plan, state
            )

            if merged_lesson:
                updated_lessons.append(merged_lesson)
                logging.info(
                    f"Merged structured content into lesson: {existing_lesson.title}"
                )

        except Exception as e:
            state["validation_errors"].append(
                f"Failed to merge with lesson {existing_lesson.title}: {str(e)}"
            )

    state["updated_lessons"] = updated_lessons
    return state


async def create_merged_lesson_from_distribution(
    existing_lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    integration_plan: ContentIntegrationPlan,
    journey_plan: JourneyCreationPlan,
    state: LessonCreationState,
) -> Optional[LessonModel]:
    """Create a merged lesson using structured content distribution"""

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
        merged_lesson = LessonModel(
            **lesson_creation.model_dump(),
            blocks=[],  # Will be populated by generate_complete_lessons node
            anchors=[],  # Will be populated by generate_complete_lessons node
            voice_ready=False,
            voice_script_id=None,  # Will be set by generate_complete_lessons node
            quiz_ids=[],  # Will be populated by generate_complete_lessons node
            interactive_ids=[],  # Will be populated by generate_complete_lessons node
        )

        # Update lesson with merged content
        merged_lesson.journey_id = existing_lesson.journey_id
        merged_lesson.version = existing_lesson.version + 1
        merged_lesson.estimated_minutes = (
            existing_lesson.estimated_minutes + lesson_dist.estimated_duration_minutes
        )

        # Intelligently merge learning objectives
        merged_lesson.learning_objectives = list(
            set(existing_lesson.learning_objectives + lesson_dist.learning_objectives)
        )

        # Update prerequisites
        merged_lesson.prerequisites = list(
            set(existing_lesson.prerequisites + lesson_dist.prerequisite_concepts)
        )

        merged_lesson.metadata.integration_action = (
            ContentIntegrationAction.MERGE_WITH_EXISTING
        )
        merged_lesson.metadata.related_existing_lessons = [existing_lesson.slug]
        merged_lesson.metadata.source_pages.extend(state["page_numbers"])
        merged_lesson.metadata.chunk_id = state["chunk_id"]

        # Generate merged content blocks using structured data
        merged_blocks = await generate_merged_content_blocks_from_distribution(
            merged_lesson, existing_lesson, lesson_dist, state
        )

        if merged_blocks:
            merged_lesson.blocks = [
                block.lesson_id + "_" + str(block.order) for block in merged_blocks
            ]

            if "content_blocks" not in state:
                state["content_blocks"] = []
            state["content_blocks"].extend(merged_blocks)

        return merged_lesson

    except Exception as e:
        logging.error(f"Failed to create merged lesson: {str(e)}")
        return None


async def generate_merged_content_blocks_from_distribution(
    merged_lesson: LessonModel,
    original_lesson: LessonModel,
    lesson_dist,
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate merged content blocks using structured lesson distribution"""

    merged_blocks = []

    # Generate enhanced concept blocks for specific concepts from distribution
    for concept in lesson_dist.concepts_to_cover:
        enhanced_block = await generate_enhanced_concept_block(
            merged_lesson, concept, lesson_dist, state
        )
        if enhanced_block:
            merged_blocks.append(enhanced_block)

    # Generate example blocks based on learning objectives
    if lesson_dist.learning_objectives:
        comparison_block = await generate_comparison_example_block(
            merged_lesson, lesson_dist, original_lesson, state
        )
        if comparison_block:
            merged_blocks.append(comparison_block)

    # Generate updated quiz reflecting merged content
    merged_quiz = await generate_merged_quiz_block(
        merged_lesson, original_lesson, lesson_dist, state
    )
    if merged_quiz:
        merged_blocks.append(merged_quiz)

    return merged_blocks


# Keep original function for backward compatibility
async def create_merged_lesson(
    existing_lesson: LessonModel,
    mapping: ExistingContentMapping,
    content_analysis: ContentAnalysisResult,
    integration_plan: ContentIntegrationPlan,
    state: LessonCreationState,
) -> Optional[LessonModel]:
    """Legacy function - kept for backward compatibility"""
    logging.warning(
        "Using legacy create_merged_lesson - should use distribution-based approach"
    )
    return None
