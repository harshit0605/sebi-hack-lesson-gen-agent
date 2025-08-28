from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.db import (
    load_journeys_from_db,
    load_lessons_from_db,
    load_anchors_from_db,
    load_processing_history,
)

import logging


async def load_existing_content(state: LessonCreationState) -> LessonCreationState:
    """Load existing journeys, lessons, and anchors for context"""

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
        state["existing_journeys"] = journeys_summary
        state["existing_journeys_list"] = existing_journeys_data
        # state["existing_lessons"] = [
        #     LessonModel(**lesson) for lesson in existing_lessons_data
        # ]
        # state["existing_anchors"] = [
        #     AnchorModel(**anchor) for anchor in existing_anchors_data
        # ]
        # state["processing_history"] = processing_history
        state["processing_history"] = {}

        # Log loading statistics
        # logging.info(f"""
        # Existing content loaded successfully:
        # - Journeys: {len(state["existing_journeys"])}
        # - Lessons: {len(state["existing_lessons"])}
        # - Anchors: {len(state["existing_anchors"])}
        # # - Processing history: {len(processing_history.get("chunks_processed", []))} chunks
        # """)

        state["current_step"] = "content_loaded"

    except Exception as e:
        # raise
        error_msg = f"Failed to load existing content: {str(e)}"
        logging.error(error_msg)
        state["validation_errors"].append(error_msg)

        # Initialize with empty collections on failure
        state["existing_journeys"] = []
        state["existing_lessons"] = []
        state["existing_anchors"] = []
        state["processing_history"] = {}
        state["current_step"] = "content_load_failed"

    return state
