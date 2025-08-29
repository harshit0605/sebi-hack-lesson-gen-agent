"""
Node for persisting generated content to MongoDB database
"""

import logging
from typing import Dict
from agents.graphs.content_generation.state import LessonCreationState

from agents.graphs.content_generation.db.persist_content import (
    persist_generated_content,
)


# async def persist_to_database_node(state: LessonCreationState) -> Command[str]:
#     """
#     Persist all generated content (journeys, lessons, content blocks, anchors) to MongoDB.

#     This node handles:
#     1. Persisting anchors (no dependencies)
#     2. Persisting journeys (references anchors)
#     3. Persisting lessons (references journeys and anchors)
#     4. Persisting content blocks (references lessons and anchors)
#     5. Updating lesson-block relationships
#     6. Updating processing state

#     Args:
#         state: LessonCreationState containing generated content

#     Returns:
#         Command routing to next step or completion
#     """

#     try:
#         logging.info(f"Starting database persistence for session {state['session_id']}")

#         # Validate required data exists
#         if not state["content_blocks"]:
#             error_msg = "No content blocks to persist"
#             logging.error(error_msg)
#             return Command(
#                 goto="error",
#                 update={
#                     "validation_errors": state["validation_errors"] + [error_msg],
#                     "current_step": "persist_failed",
#                 },
#             )

#         if not state["lessons"]:
#             error_msg = "No lessons to persist"
#             logging.error(error_msg)
#             return Command(
#                 goto="error",
#                 update={
#                     "validation_errors": state["validation_errors"] + [error_msg],
#                     "current_step": "persist_failed",
#                 },
#             )

#         # Convert Pydantic models to dicts for MongoDB
#         new_journeys_dict = []
#         if state["new_journeys"]:
#             new_journeys_dict = [
#                 journey.model_dump() for journey in state["new_journeys"]
#             ]

#         lessons_dict = [lesson.model_dump() for lesson in state["lessons"]]
#         content_blocks_dict = [block.model_dump() for block in state["content_blocks"]]

#         anchors_dict = []
#         if state["anchors"]:
#             anchors_dict = [anchor.model_dump() for anchor in state["anchors"]]

#         integration_plan_dict = {}
#         if state["integration_plan"]:
#             integration_plan_dict = state["integration_plan"].model_dump()

#         # Persist to database
#         persistence_result = await persist_generated_content(
#             session_id=state["session_id"],
#             new_journeys=new_journeys_dict,
#             lessons=lessons_dict,
#             content_blocks=content_blocks_dict,
#             anchors=anchors_dict,
#             integration_plan=integration_plan_dict,
#         )

#         if not persistence_result["success"]:
#             error_msg = f"Database persistence failed: {persistence_result.get('error', 'Unknown error')}"
#             logging.error(error_msg)

#             return Command(
#                 goto="error",
#                 update={
#                     "validation_errors": state["validation_errors"]
#                     + [error_msg]
#                     + persistence_result.get("errors", []),
#                     "current_step": "persist_failed",
#                     "processing_history": {
#                         **state["processing_history"],
#                         "persistence_attempt": {
#                             "success": False,
#                             "errors": persistence_result.get("errors", []),
#                             "partial_results": persistence_result.get(
#                                 "created_ids", {}
#                             ),
#                         },
#                     },
#                 },
#             )

#         # Success - update state with persistence results
#         logging.info("Successfully persisted content to database")
#         logging.info(f"Created IDs: {persistence_result['created_ids']}")

#         return Command(
#             goto="complete",
#             update={
#                 "current_step": "content_persisted",
#                 "processing_history": {
#                     **state["processing_history"],
#                     "persistence_result": {
#                         "success": True,
#                         "created_ids": persistence_result["created_ids"],
#                         "summary": {
#                             "journeys_created": len(
#                                 persistence_result["created_ids"]["journeys"]
#                             ),
#                             "lessons_created": len(
#                                 persistence_result["created_ids"]["lessons"]
#                             ),
#                             "content_blocks_created": len(
#                                 persistence_result["created_ids"]["content_blocks"]
#                             ),
#                             "anchors_created": len(
#                                 persistence_result["created_ids"]["anchors"]
#                             ),
#                         },
#                     },
#                 },
#             },
#         )

#     except Exception as e:
#         error_msg = f"Critical error in persist_to_database_node: {str(e)}"
#         logging.error(error_msg, exc_info=True)

#         return Command(
#             goto="error",
#             update={
#                 "validation_errors": state["validation_errors"] + [error_msg],
#                 "current_step": "persist_failed",
#                 "processing_history": {
#                     **state["processing_history"],
#                     "persistence_error": {
#                         "error": error_msg,
#                         "step": "persist_to_database_node",
#                     },
#                 },
#             },
#         )


async def persist_to_database_node(state: LessonCreationState) -> LessonCreationState:
    """
    Persist all generated content (journeys, lessons, content blocks, anchors) to MongoDB.

    This node handles:
    1. Persisting anchors (no dependencies)
    2. Persisting journeys (references anchors)
    3. Persisting lessons (references journeys and anchors)
    4. Persisting content blocks (references lessons and anchors)
    5. Updating lesson-block relationships
    6. Updating processing state

    Args:
        state: LessonCreationState containing generated content

    Returns:
        Partial state update with keys it modifies only
    """

    try:
        logging.info(f"Starting database persistence for session {state['session_id']}")

        # Validate required data exists
        if not state["content_blocks"]:
            error_msg = "No content blocks to persist"
            logging.error(error_msg)
            return {
                "validation_errors": state.get("validation_errors", []) + [error_msg],
                "current_step": "persist_failed",
            }

        if not state["lessons"]:
            error_msg = "No lessons to persist"
            logging.error(error_msg)
            return {
                "validation_errors": state.get("validation_errors", []) + [error_msg],
                "current_step": "persist_failed",
            }

        # Convert Pydantic models to dicts for MongoDB
        new_journeys_dict = []
        if state.get("new_journeys"):
            new_journeys_dict = [
                journey.model_dump() for journey in state["new_journeys"]
            ]

        lessons_dict = [lesson.model_dump() for lesson in state["lessons"]]
        content_blocks_dict = [block.model_dump() for block in state["content_blocks"]]

        anchors_dict = []
        if state["anchors"]:
            anchors_dict = [anchor.model_dump() for anchor in state["anchors"]]

        # Enrich anchors with state-derived source metadata to avoid relying on LLM output
        if anchors_dict:
            source_url = state.get("source_url", "")
            # Fallback to original pdf_source if source_url not explicitly set
            if not source_url:
                source_url = state.get("pdf_source", "")
            source_type = state.get("source_type", "SEBI_PDF")

            for a in anchors_dict:
                a.setdefault("source_url", source_url)
                a.setdefault("source_type", source_type)
                # Do not persist page_numbers or created_from_chunk on anchors (per design)

        integration_plan_dict = {}
        if state["integration_plan"]:
            integration_plan_dict = state["integration_plan"].model_dump()

        # Persist to database
        persistence_result = await persist_generated_content(
            session_id=state["session_id"],
            new_journeys=new_journeys_dict,
            lessons=lessons_dict,
            content_blocks=content_blocks_dict,
            anchors=anchors_dict,
            integration_plan=integration_plan_dict,
        )

        if not persistence_result["success"]:
            error_msg = f"Database persistence failed: {persistence_result.get('error', 'Unknown error')}"
            logging.error(error_msg)
            return {
                "validation_errors": state.get("validation_errors", []) + [error_msg],
                "current_step": "persist_failed",
            }

        # Success - update state with persistence results
        logging.info("Successfully persisted content to database")
        logging.info(f"Created IDs: {persistence_result['created_ids']}")
        return {"current_step": "content_persisted"}

    except Exception as e:
        error_msg = f"Critical error in persist_to_database_node: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return {
            "validation_errors": state.get("validation_errors", []) + [error_msg],
            "current_step": "persist_failed",
        }


def create_persistence_summary(created_ids: Dict[str, list]) -> str:
    """Create a human-readable summary of what was persisted"""

    summary_parts = []

    if created_ids.get("journeys"):
        summary_parts.append(f"{len(created_ids['journeys'])} learning journey(s)")

    if created_ids.get("lessons"):
        summary_parts.append(f"{len(created_ids['lessons'])} lesson(s)")

    if created_ids.get("content_blocks"):
        summary_parts.append(f"{len(created_ids['content_blocks'])} content block(s)")

    if created_ids.get("anchors"):
        summary_parts.append(f"{len(created_ids['anchors'])} anchor(s)")

    if summary_parts:
        return f"Successfully persisted: {', '.join(summary_parts)}"
    else:
        return "No content was persisted"
