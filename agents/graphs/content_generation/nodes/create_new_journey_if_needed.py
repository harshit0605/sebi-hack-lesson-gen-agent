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
        return {"current_step": "no_journey_creation_needed"}

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

        # Prepare partial updates
        updates: LessonCreationState = {
            "new_journeys": state.get("new_journeys", []) + [new_journey],
            "current_step": "new_journey_created",
        }

        # Update integration plan to use the new journey (copy to avoid in-place mutation)
        if state.get("integration_plan"):
            try:
                plan_copy = state["integration_plan"].model_copy(deep=True)
            except Exception:
                # Fallback to using the original if deep copy unsupported
                plan_copy = state["integration_plan"]
            plan_copy.new_journey_needed = True
            plan_copy.rationale = (plan_copy.rationale or "") + (
                f" Created new journey: {new_journey.title}"
            )
            updates["integration_plan"] = plan_copy

        logging.info(f"New journey created: {new_journey.title} ({new_journey.slug})")
        return updates

    except Exception as e:
        error_msg = f"Failed to create new journey: {str(e)}"
        logging.error(error_msg)
        return {
            "validation_errors": state.get("validation_errors", []) + [error_msg],
            "current_step": "journey_creation_failed",
        }


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
