from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    JourneyModel,
)
import re
from typing import List
import logging


async def create_new_journey_if_needed(
    state: LessonCreationState,
) -> LessonCreationState:
    """Create a new learning journey if the content doesn't fit existing ones"""

    journey_plan = state.get("journey_creation_plan")
    if not journey_plan:
        state["current_step"] = "no_journey_creation_needed"
        return state

    try:
        # Generate journey slug from title
        # slug = generate_slug(journey_plan.title)
        slug = journey_plan.slug

        # Ensure slug uniqueness
        # existing_slugs = [journey.slug for journey in state["existing_journeys"]]
        # slug = ensure_unique_slug(slug, existing_slugs)

        # Use outcomes directly from journey plan
        outcomes = journey_plan.outcomes

        # Create the new journey model
        new_journey = JourneyModel(
            slug=slug,
            title=journey_plan.title,
            description=journey_plan.description,
            level=journey_plan.level,
            outcomes=outcomes,
            prerequisites=journey_plan.prerequisites,
            estimated_hours=journey_plan.total_estimated_hours,
            tags=journey_plan.key_topics,
            order=len(state["existing_journeys_list"]) + 1,  # Place at end
            sebi_topics=journey_plan.key_topics,
            status="draft",
        )

        # Validate the journey model
        new_journey.model_validate(new_journey.model_dump())

        # Add to state
        if "new_journeys" not in state:
            state["new_journeys"] = []
        state["new_journeys"].append(new_journey)

        # Update integration plan to use the new journey
        if state["integration_plan"]:
            state["integration_plan"].new_journey_needed = True
            state[
                "integration_plan"
            ].rationale += f" Created new journey: {new_journey.title}"

        state["current_step"] = "new_journey_created"

        logging.info(f"New journey created: {new_journey.title} ({new_journey.slug})")

    except Exception as e:
        error_msg = f"Failed to create new journey: {str(e)}"
        state["validation_errors"].append(error_msg)
        logging.error(error_msg)

    return state


# Removed generate_learning_outcomes_from_structure function
# Outcomes are now generated directly in evaluate_journey_fit node


def generate_slug(title: str) -> str:
    """Generate URL-friendly slug from title"""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug[:50]  # Limit length


def ensure_unique_slug(slug: str, existing_slugs: List[str]) -> str:
    """Ensure slug is unique by appending number if needed"""
    original_slug = slug
    counter = 1

    while slug in existing_slugs:
        slug = f"{original_slug}-{counter}"
        counter += 1

    return slug
