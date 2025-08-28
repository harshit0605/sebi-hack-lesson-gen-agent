from typing import Dict, Any
from .db import get_database
import logging


def load_processing_history(session_id: str) -> Dict[str, Any]:
    """Load processing history for a session to maintain context across chunks"""

    try:
        db = get_database()

        # Find the processing state for this session
        processing_state = db.processing_states.find_one({"session_id": session_id})

        if processing_state:
            # Convert ObjectId to string
            processing_state["_id"] = str(processing_state["_id"])

            # Convert ObjectIds in nested arrays
            for chunk in processing_state.get("chunks_processed", []):
                if "lessons_created" in chunk:
                    chunk["lessons_created"] = [
                        str(oid) for oid in chunk["lessons_created"]
                    ]
                if "anchors_created" in chunk:
                    chunk["anchors_created"] = [
                        str(oid) for oid in chunk["anchors_created"]
                    ]

            logging.info(f"Loaded processing history for session {session_id}")
            return processing_state
        else:
            logging.info(f"No processing history found for session {session_id}")
            return {
                "session_id": session_id,
                "chunks_processed": [],
                "journey_assignments": [],
                "global_context": {
                    "key_themes": [],
                    "recurring_concepts": [],
                    "sebi_guidelines_referenced": [],
                },
                "status": "new",
            }

    except Exception as e:
        logging.error(
            f"Error loading processing history for session {session_id}: {str(e)}"
        )
        return {
            "session_id": session_id,
            "chunks_processed": [],
            "journey_assignments": [],
            "global_context": {},
            "status": "error",
        }
