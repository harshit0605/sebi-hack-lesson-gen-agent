from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.db import (
    load_journeys_from_db,
    load_lessons_from_db,
    load_anchors_from_db,
    load_processing_history,
)

import logging


async def load_existing_content(state: LessonCreationState) -> LessonCreationState:
    """Load existing journeys, lessons, and anchors for context.

    Returns only updated keys: `existing_journeys`, `existing_journeys_list`,
    `processing_history`, `current_step` and appends to `validation_errors` on failure.
    """

    try:
        # Load from MongoDB with error handling
        existing_journeys_data = load_journeys_from_db()
        # existing_lessons_data = load_lessons_from_db()
        # existing_anchors_data = load_anchors_from_db()

        # Load processing history for session context
        # processing_history = load_processing_history(state.get("session_id"))

        # Convert ObjectIds to strings before creating Pydantic models
        def convert_objectids_to_strings(data):
            """Recursively convert ObjectId objects to strings"""
            if isinstance(data, dict):
                return {
                    key: convert_objectids_to_strings(value)
                    for key, value in data.items()
                }
            elif isinstance(data, list):
                return [convert_objectids_to_strings(item) for item in data]
            elif hasattr(data, "__class__") and "ObjectId" in str(type(data)):
                return str(data)
            else:
                return data

        # Convert ObjectIds in all data
        existing_journeys_data = convert_objectids_to_strings(existing_journeys_data)
        # existing_lessons_data = convert_objectids_to_strings(existing_lessons_data)
        # existing_anchors_data = convert_objectids_to_strings(existing_anchors_data)

        # Convert to Pydantic models for validation and type safety
        journeys_summary = "\n".join(
            [
                f"- **Journey Id: {journey['slug']}, Journey Title: {journey['title']}, Journey Level: {journey['level']}, Journey Description: {journey['description']} **"
                for journey in existing_journeys_data
            ]
        )
        # Return partial updates
        return {
            "existing_journeys": journeys_summary,
            "existing_journeys_list": existing_journeys_data,
            "processing_history": {},
            "current_step": "content_loaded",
        }
        # state["existing_lessons"] = [
        #     LessonModel(**lesson) for lesson in existing_lessons_data
        # ]
        # state["existing_anchors"] = [
        #     AnchorModel(**anchor) for anchor in existing_anchors_data
        # ]
        # state["processing_history"] = processing_history
    except Exception as e:
        error_msg = f"Failed to load existing content: {str(e)}"
        logging.error(error_msg)

        # Return partial failure state
        return {
            "validation_errors": state.get("validation_errors", []) + [error_msg],
            "existing_journeys": [],
            "existing_lessons": [],
            "existing_anchors": [],
            "processing_history": {},
            "current_step": "content_load_failed",
        }
