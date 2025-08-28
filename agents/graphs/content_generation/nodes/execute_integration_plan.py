"""Main orchestrator for executing integration plans.

This module acts as a lightweight orchestrator that delegates to specific strategy implementations.
"""

import logging
from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import ContentIntegrationAction

# Import strategy implementations
from agents.graphs.content_generation.nodes.strategies import (
    create_new_lessons,
    extend_existing_lessons,
    merge_with_existing_lessons,
    split_into_multiple_lessons,
)


async def execute_integration_plan(state: LessonCreationState) -> LessonCreationState:
    """Execute the integration plan and update/create content accordingly"""

    integration_plan = state["integration_plan"]

    try:
        if integration_plan.action == ContentIntegrationAction.CREATE_NEW_LESSON:
            await create_new_lessons(state)
        elif integration_plan.action == ContentIntegrationAction.EXTEND_EXISTING_LESSON:
            await extend_existing_lessons(state)
        elif integration_plan.action == ContentIntegrationAction.MERGE_WITH_EXISTING:
            await merge_with_existing_lessons(state)
        elif integration_plan.action == ContentIntegrationAction.SPLIT_INTO_MULTIPLE:
            await split_into_multiple_lessons(state)
        else:
            state["validation_errors"].append(
                f"Unknown integration action: {integration_plan.action}"
            )

        state["current_step"] = "integration_executed"
        logging.info(f"Integration plan executed: {integration_plan.action.value}")

    except Exception as e:
        error_msg = f"Failed to execute integration plan: {str(e)}"
        state["validation_errors"].append(error_msg)
        logging.error(error_msg)

    return state
