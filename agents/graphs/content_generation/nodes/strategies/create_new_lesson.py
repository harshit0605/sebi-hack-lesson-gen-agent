"""
Strategy for creating new lessons from content analysis.
"""

import logging

from typing import Optional
from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    LessonModel,
    LessonContentDistribution,
    ContentIntegrationPlan,
    JourneyCreationPlan,
    LessonCreationModel,
)
from agents.graphs.content_generation.llm import (
    lesson_creator_llm,
)
from agents.graphs.content_generation.prompts import LESSON_CREATION_PROMPTS


async def create_new_lessons(state: LessonCreationState):
    """Create new lessons from structured content distribution"""

    integration_plan = state["integration_plan"]
    journey_plan = state.get("journey_creation_plan")
    existing_journeys = state.get("existing_journeys", [])

    # Get lesson distributions for new lessons
    new_lesson_distributions = [
        dist
        for dist in integration_plan.content_distribution
        if dist.integration_type == "new_lesson"
    ]

    if not new_lesson_distributions:
        logging.warning("No new lesson distributions found in integration plan")
        return state

    new_lessons = []

    for lesson_dist in new_lesson_distributions:
        try:
            lesson = await create_lesson_from_distribution(
                lesson_dist, integration_plan, journey_plan, state, existing_journeys
            )
            if lesson:
                new_lessons.append(lesson)
        except Exception as e:
            state["validation_errors"].append(
                f"Failed to create lesson '{lesson_dist.lesson_title}': {str(e)}"
            )

    state["new_lessons"] = new_lessons
    logging.info(f"Created {len(new_lessons)} new lessons from distributions")

    return state


async def create_lesson_from_distribution(
    lesson_dist: LessonContentDistribution,
    integration_plan: ContentIntegrationPlan,
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
    existing_journeys: list,
) -> LessonModel:
    """Create a lesson from a structured content distribution"""

    prompt_template = LESSON_CREATION_PROMPTS["create_new_lesson"]

    # Use structured data from distribution
    journey_context = ""
    if journey_plan:
        journey_context = f"""
        Journey Context:
        - Journey id: {journey_plan.slug}
        - Journey title: {journey_plan.title}
        - Level: {journey_plan.level}
        - Target Audience: {journey_plan.target_audience}
        - Key Topics: {", ".join(journey_plan.key_topics)}
        """

    # Get content analysis for raw content context
    content_analysis = state.get("content_analysis")

    prompt = prompt_template.format_messages(
        lesson_title=lesson_dist.lesson_title,
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

    lesson_creation = await lesson_creator_llm.ainvoke_with_structured_output(
        prompt, LessonCreationModel
    )

    # Convert to full LessonModel with empty content fields (will be populated later)
    lesson = LessonModel(
        **lesson_creation.model_dump(),
        blocks=[],  # Will be populated by generate_complete_lessons node
        anchors=[],  # Will be populated by generate_complete_lessons node
        voice_ready=False,
        voice_script_id=None,  # Will be set by generate_complete_lessons node
        quiz_ids=[],  # Will be populated by generate_complete_lessons node
        interactive_ids=[],  # Will be populated by generate_complete_lessons node
    )

    # Set metadata from distribution
    lesson.estimated_minutes = lesson_dist.estimated_duration_minutes
    lesson.learning_objectives = lesson_dist.learning_objectives
    lesson.prerequisites = lesson_dist.prerequisite_concepts

    # Set journey context - handle both new and existing journey scenarios
    if journey_plan:
        # New journey scenario
        lesson.journey_id = journey_plan.slug
    else:
        # Existing journey scenario - find target journey from integration plan
        target_journey_id = _get_target_journey_id(
            lesson_dist, existing_journeys, integration_plan
        )
        if target_journey_id:
            lesson.journey_id = target_journey_id
        else:
            # Fallback to first existing journey if available
            if existing_journeys:
                lesson.journey_id = existing_journeys[0].slug
            else:
                raise ValueError(
                    f"No journey context available for lesson: {lesson_dist.lesson_title}"
                )

    return lesson


def _get_target_journey_id(lesson_dist, existing_journeys, integration_plan):
    """
    Determine the target journey ID for a lesson based on integration plan context.

    Args:
        lesson_dist: LessonContentDistribution object
        existing_journeys: List of existing JourneyModel objects
        integration_plan: ContentIntegrationPlan object

    Returns:
        str: Journey slug/ID or None if not found
    """
    # If integration plan specifies a target journey, use that
    if (
        hasattr(integration_plan, "target_journey_id")
        and integration_plan.target_journey_id
    ):
        return integration_plan.target_journey_id

    # Look for journey context in the lesson distribution
    if hasattr(lesson_dist, "journey_id") and lesson_dist.journey_id:
        return lesson_dist.journey_id

    # Try to match based on lesson content and existing journey topics
    if existing_journeys:
        # Simple heuristic: match based on lesson topics with journey topics
        lesson_topics = set(lesson_dist.concepts_to_cover)

        best_match = None
        best_score = 0

        for journey in existing_journeys:
            # Calculate topic overlap
            journey_topics = (
                set(journey.tags)
                if hasattr(journey, "tags") and journey.tags
                else set()
            )
            if hasattr(journey, "sebi_topics") and journey.sebi_topics:
                journey_topics.update(journey.sebi_topics)

            overlap = len(lesson_topics.intersection(journey_topics))
            if overlap > best_score:
                best_score = overlap
                best_match = journey.slug

        return best_match

    return None
