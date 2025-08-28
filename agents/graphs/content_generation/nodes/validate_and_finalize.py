from agents.graphs.content_generation.state import LessonCreationState
from typing import Literal


async def validate_and_finalize(state: LessonCreationState) -> LessonCreationState:
    """Validate all generated content using Pydantic models and business rules"""

    validation_errors = []

    # Validate Pydantic model constraints
    try:
        # Validate journeys
        for journey in state.get("new_journeys", []):
            journey.model_validate(journey.model_dump())

        # Validate lessons
        for lesson in state.get("new_lessons", []) + state.get("updated_lessons", []):
            lesson.model_validate(lesson.model_dump())

            # Business rule validation
            if lesson.estimated_minutes < 10 or lesson.estimated_minutes > 45:
                validation_errors.append(
                    f"Lesson '{lesson.title}' has invalid duration: {lesson.estimated_minutes} minutes"
                )

        # Validate content blocks
        for block in state.get("content_blocks", []):
            block.model_validate(block.model_dump())

            # Ensure each block has at least one anchor
            if not block.anchor_ids:
                validation_errors.append(
                    f"Block '{block.type}' in lesson '{block.lesson_id}' has no SEBI anchors"
                )

        # Validate anchors
        for anchor in state.get("anchors", []):
            anchor.model_validate(anchor.model_dump())

    except Exception as e:
        validation_errors.append(f"Pydantic validation error: {str(e)}")

    return {"validation_errors": validation_errors, "current_step": "validated"}


def validation_gate(
    state: LessonCreationState,
) -> Literal["valid", "needs_review", "has_errors"]:
    """Determine next step based on validation results"""

    errors = state.get("validation_errors", [])
    # quality_metrics = state.get("quality_metrics", {})

    if errors:
        retry_count = state.get("retry_count", 0)
        if retry_count < 2:
            # Do not mutate state here; increment handled in error node
            return "has_errors"
        else:
            # Do not mutate state here; request handled in human review node
            return "needs_review"

    return "valid"

    # Check quality thresholds
    # avg_quality = (
    #     sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0.5
    # )

    # if avg_quality >= 0.8:
    #     return "valid"
    # else:
    #     return "needs_review"
