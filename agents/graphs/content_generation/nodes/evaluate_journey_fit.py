from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    JourneyCreationPlan,
)
from agents.graphs.content_generation.llm import lesson_creator_llm
from agents.graphs.content_generation.prompts.evaluate_journey_fit import (
    EVALUATE_JOURNEY_FIT_PROMPTS,
)
from typing import Literal


async def evaluate_journey_fit(state: LessonCreationState) -> LessonCreationState:
    """Evaluate if new content fits existing journeys or needs a new one.

    Returns only: `journey_creation_plan` (if created) and `current_step`.
    """

    integration_plan = state["integration_plan"]
    existing_journeys = state["existing_journeys"]

    updates: LessonCreationState = {}

    if integration_plan.new_journey_needed:
        journey_prompt = EVALUATE_JOURNEY_FIT_PROMPTS["journey_creation_planning"]

        messages = journey_prompt.format_messages(
            content_distribution=[
                d.model_dump_json() for d in integration_plan.content_distribution
            ],
            existing_journeys=existing_journeys,
            integration_rationale=integration_plan.rationale,
        )

        journey_plan = await lesson_creator_llm.ainvoke_with_structured_output(
            messages, JourneyCreationPlan
        )
        updates["journey_creation_plan"] = journey_plan

    updates["current_step"] = "journey_evaluated"
    return updates


def journey_decision_gate(
    state: LessonCreationState,
) -> Literal["fits_existing", "needs_new_journey", "unclear"]:
    """Decide whether to create new journey, use existing, or request review"""

    integration_plan = state["integration_plan"]

    if integration_plan.new_journey_needed:
        if (
            state.get("journey_creation_plan")
            and state["journey_creation_plan"].justification
        ):
            return "needs_new_journey"
        else:
            return "unclear"
    else:
        return "fits_existing"
