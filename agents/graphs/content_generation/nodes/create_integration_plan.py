from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    ContentIntegrationPlan,
)
from agents.graphs.content_generation.llm import integration_planner_llm
from agents.graphs.content_generation.prompts.create_integration_plan import (
    CREATE_INTEGRATION_PLAN_PROMPTS,
)


async def create_integration_plan(state: LessonCreationState) -> LessonCreationState:
    """Create a detailed plan for integrating new content with existing content.

    Returns only: `integration_plan`, `current_step`.
    """

    content_analysis = state.get("content_analysis", {})
    mappings = state.get("existing_content_mappings", [])
    existing_journeys = state.get("existing_journeys", [])

    integration_prompt = CREATE_INTEGRATION_PLAN_PROMPTS["integration_planning"]

    # Handle empty mappings case explicitly
    if not mappings or len(mappings) == 0:
        mappings_text = (
            "EMPTY - No existing lessons found with relevant content overlap"
        )
    else:
        mappings_text = "\n".join([m.model_dump_json() for m in mappings])

    messages = integration_prompt.format_messages(
        content_analysis=content_analysis.model_dump_json(),
        mappings=mappings_text,
        existing_journeys=existing_journeys,
    )

    # print(messages)

    integration_plan = await integration_planner_llm.ainvoke_with_structured_output(
        messages, ContentIntegrationPlan
    )

    return {"integration_plan": integration_plan, "current_step": "integration_planned"}
