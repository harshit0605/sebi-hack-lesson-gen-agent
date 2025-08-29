from typing import Dict, Any, List
from .db import get_database
import logging
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from datetime import datetime


async def persist_generated_content(
    session_id: str,
    new_journeys: List[Dict[str, Any]],
    lessons: List[Dict[str, Any]],
    content_blocks: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
    integration_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Persist all generated content to MongoDB with proper foreign key relationships.

    Args:
        session_id: Processing session identifier
        new_journeys: List of journey objects to create
        lessons: List of lesson objects to create
        content_blocks: List of content block objects to create
        anchors: List of anchor objects to create
        integration_plan: Integration plan metadata

    Returns:
        Dict with persistence results and created object IDs
    """

    try:
        db = get_database()

        # Track results
        results = {
            "success": True,
            "created_ids": {
                "journeys": [],
                "lessons": [],
                "content_blocks": [],
                "anchors": [],
            },
            "errors": [],
            "session_id": session_id,
        }

        # 1. First persist anchors (no dependencies)
        anchor_id_mapping = {}
        if anchors:
            logging.info(f"Persisting {len(anchors)} anchors...")
            for anchor in anchors:
                try:
                    # Prepare anchor document
                    anchor_doc = prepare_anchor_document(anchor, session_id)

                    # Insert anchor
                    result = db.anchors.insert_one(anchor_doc)
                    anchor_id = result.inserted_id

                    # Map old ID to new ObjectId for reference updates
                    if "id" in anchor:
                        anchor_id_mapping[anchor["id"]] = anchor_id

                    # Also map by short_label which is commonly used in content blocks
                    if "short_label" in anchor:
                        anchor_id_mapping[anchor["short_label"]] = anchor_id

                    results["created_ids"]["anchors"].append(str(anchor_id))
                    logging.debug(f"Created anchor: {anchor_id}")

                except DuplicateKeyError:
                    error_msg = (
                        f"Duplicate anchor: {anchor.get('short_label', 'unknown')}"
                    )
                    results["errors"].append(error_msg)
                    logging.warning(error_msg)
                except Exception as e:
                    error_msg = f"Failed to create anchor {anchor.get('short_label', 'unknown')}: {str(e)}"
                    results["errors"].append(error_msg)
                    logging.error(error_msg)

        # 2. Persist journeys and build journey mapping (includes existing journeys)
        journey_id_mapping = {}
        
        # First, load existing journeys to build mapping for lesson references
        existing_journeys = db.learning_journeys.find({})
        existing_journey_count = 0
        for journey in existing_journeys:
            journey_slug = journey["slug"]
            journey_id_mapping[journey_slug] = journey["_id"]
            # Also map common variations
            journey_slug_underscore = journey_slug.replace("-", "_")
            journey_id_mapping[journey_slug_underscore] = journey["_id"]
            existing_journey_count += 1
        
        logging.info(f"Loaded {existing_journey_count} existing journeys for lesson mapping")
        
        # Then persist new journeys if any
        if new_journeys:
            logging.info(f"Persisting {len(new_journeys)} journeys...")
            for journey in new_journeys:
                try:
                    # Prepare journey document
                    journey_doc = prepare_journey_document(journey, anchor_id_mapping)

                    # Insert journey
                    result = db.learning_journeys.insert_one(journey_doc)
                    journey_id = result.inserted_id

                    # Map slug to ObjectId for lesson references
                    journey_slug = journey["slug"]
                    journey_id_mapping[journey_slug] = journey_id

                    # Also map common variations for lesson references
                    journey_slug_underscore = journey_slug.replace("-", "_")
                    journey_id_mapping[journey_slug_underscore] = journey_id

                    logging.debug(
                        f"Created journey: {journey_id} with slug mappings: {journey_slug}, {journey_slug_underscore}"
                    )

                    results["created_ids"]["journeys"].append(str(journey_id))

                except DuplicateKeyError:
                    error_msg = (
                        f"Duplicate journey slug: {journey.get('slug', 'unknown')}"
                    )
                    results["errors"].append(error_msg)
                    logging.warning(error_msg)
                except Exception as e:
                    error_msg = f"Failed to create journey {journey.get('slug', 'unknown')}: {str(e)}"
                    results["errors"].append(error_msg)
                    logging.error(error_msg)

        # 3. Persist lessons (depends on journeys)
        lesson_id_mapping = {}
        if lessons:
            logging.info(f"Persisting {len(lessons)} lessons...")
            for lesson in lessons:
                try:
                    # Prepare lesson document
                    lesson_doc = prepare_lesson_document(
                        lesson, journey_id_mapping, anchor_id_mapping
                    )

                    # Insert lesson
                    result = db.lessons.insert_one(lesson_doc)
                    lesson_id = result.inserted_id

                    # Map both slug and potential variations for content block references
                    lesson_slug = lesson["slug"]
                    lesson_id_mapping[lesson_slug] = lesson_id

                    # Also map common variations (hyphens to underscores, etc.)
                    lesson_slug_underscore = lesson_slug.replace("-", "_")
                    lesson_id_mapping[lesson_slug_underscore] = lesson_id

                    results["created_ids"]["lessons"].append(str(lesson_id))
                    logging.debug(
                        f"Created lesson: {lesson_id} with slug mappings: {lesson_slug}, {lesson_slug_underscore}"
                    )

                except DuplicateKeyError:
                    error_msg = (
                        f"Duplicate lesson slug: {lesson.get('slug', 'unknown')}"
                    )
                    results["errors"].append(error_msg)
                    logging.warning(error_msg)
                except Exception as e:
                    error_msg = f"Failed to create lesson {lesson.get('slug', 'unknown')}: {str(e)}"
                    results["errors"].append(error_msg)
                    logging.error(error_msg)

        # 4. Persist content blocks (depends on lessons and anchors)
        if content_blocks:
            logging.info(f"Persisting {len(content_blocks)} content blocks...")
            for block in content_blocks:
                try:
                    # Prepare content block document
                    block_doc = prepare_content_block_document(
                        block, lesson_id_mapping, anchor_id_mapping
                    )

                    # Insert content block
                    result = db.content_blocks.insert_one(block_doc)
                    block_id = result.inserted_id

                    results["created_ids"]["content_blocks"].append(str(block_id))
                    logging.debug(f"Created content block: {block_id}")

                except Exception as e:
                    error_msg = f"Failed to create content block for lesson {block.get('lesson_id', 'unknown')}: {str(e)}"
                    results["errors"].append(error_msg)
                    logging.error(error_msg)

        # 5. Update lessons with content block references
        update_lesson_block_references(
            db, lesson_id_mapping, results["created_ids"]["content_blocks"]
        )

        # Update processing state
        update_processing_state(
            db, session_id, results["created_ids"], integration_plan
        )

        logging.info(f"Successfully persisted content for session {session_id}")
        logging.info(
            f"Created: {len(results['created_ids']['journeys'])} journeys, "
            f"{len(results['created_ids']['lessons'])} lessons, "
            f"{len(results['created_ids']['content_blocks'])} blocks, "
            f"{len(results['created_ids']['anchors'])} anchors"
        )

        return results

    except Exception as e:
        error_msg = (
            f"Critical error persisting content for session {session_id}: {str(e)}"
        )
        logging.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "session_id": session_id,
            "created_ids": {
                "journeys": [],
                "lessons": [],
                "content_blocks": [],
                "anchors": [],
            },
            "errors": [error_msg],
        }


def prepare_anchor_document(anchor: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Prepare anchor document for MongoDB insertion"""

    return {
        "source_type": anchor.get("source_type", "SEBI_PDF"),
        "title": anchor.get("title", ""),
        "short_label": anchor.get("short_label", ""),
        "excerpt": anchor.get("excerpt", ""),
        "source_url": anchor.get("source_url", ""),
        "document_title": anchor.get("document_title", ""),
        "section": anchor.get("section", ""),
        "last_verified_at": datetime.utcnow(),
        "relevance_tags": anchor.get("relevance_tags", []),
        "confidence_score": anchor.get("confidence_score", 0.8),
        "created_at": datetime.utcnow(),
    }


def prepare_journey_document(
    journey: Dict[str, Any], anchor_id_mapping: Dict[str, ObjectId]
) -> Dict[str, Any]:
    """Prepare journey document for MongoDB insertion"""

    # Convert anchor references
    outcomes = []
    for outcome in journey.get("outcomes", []):
        outcome_doc = {
            "outcome": outcome.get("outcome", ""),
            "assessment_criteria": outcome.get("assessment_criteria", []),
        }

        # Convert anchor ID references
        # Ensure anchor_ids list is initialized before appending
        outcome_doc["anchor_ids"] = []
        for anchor_ref in outcome.get("anchor_ids", []):
            if anchor_ref in anchor_id_mapping:
                outcome_doc["anchor_ids"].append(anchor_id_mapping[anchor_ref])

        outcomes.append(outcome_doc)

    return {
        "slug": journey["slug"],
        "title": journey["title"],
        "description": journey["description"],
        "level": journey["level"],
        "outcomes": outcomes,
        "prerequisites": journey.get("prerequisites", []),
        "estimated_hours": journey.get("estimated_hours", 0),
        "tags": journey.get("tags", []),
        "order": journey.get("order", 0),
        "sebi_topics": journey.get("sebi_topics", []),
        "status": journey.get("status", "draft"),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


def prepare_lesson_document(
    lesson: Dict[str, Any],
    journey_id_mapping: Dict[str, ObjectId],
    anchor_id_mapping: Dict[str, ObjectId],
) -> Dict[str, Any]:
    """Prepare lesson document for MongoDB insertion"""

    # Get journey ObjectId
    lesson_journey_id = lesson["journey_id"]
    journey_id = journey_id_mapping.get(lesson_journey_id)
    if not journey_id:
        # Try alternative mappings
        journey_id = journey_id_mapping.get(lesson_journey_id.replace("-", "_"))
        if not journey_id:
            available_journeys = list(journey_id_mapping.keys())
            raise ValueError(
                f"Journey not found for lesson: {lesson.get('slug', 'unknown')}. "
                f"Looking for journey_id: '{lesson_journey_id}'. "
                f"Available journeys: {available_journeys}"
            )

    # Convert anchor references
    anchor_refs = []
    for anchor_ref in lesson.get("anchors", []):
        if anchor_ref in anchor_id_mapping:
            anchor_refs.append(anchor_id_mapping[anchor_ref])

    return {
        "journey_id": journey_id,
        "slug": lesson["slug"],
        "title": lesson["title"],
        "subtitle": lesson.get("subtitle", ""),
        "unit": lesson.get("unit", ""),
        "estimated_minutes": lesson.get("estimated_minutes", 30),
        "difficulty": lesson.get("difficulty", "easy"),
        "order": lesson.get("order", 1),
        "learning_objectives": lesson.get("learning_objectives", []),
        "blocks": [],  # Will be updated after content blocks are created
        "anchors": anchor_refs,
        "voice_ready": lesson.get("voice_ready", False),
        "voice_script_id": None,
        "quiz_ids": [],  # Will be populated from content blocks
        "interactive_ids": [],  # Will be populated from content blocks
        "prerequisites": lesson.get("prerequisites", []),
        "tags": lesson.get("tags", []),
        "metadata": {
            "source_pages": lesson.get("metadata", {}).get("source_pages", []),
            "chunk_id": lesson.get("metadata", {}).get("chunk_id", ""),
            "overlap_handled": lesson.get("metadata", {}).get("overlap_handled", False),
            "quality_score": lesson.get("metadata", {}).get("quality_score", 0.8),
            "review_status": lesson.get("metadata", {}).get(
                "review_status", "generated"
            ),
            "sebi_compliance_checked": lesson.get("metadata", {}).get(
                "sebi_compliance_checked", False
            ),
            "integration_action": lesson.get("metadata", {}).get(
                "integration_action", "create_new_lesson"
            ),
            "related_existing_lessons": lesson.get("metadata", {}).get(
                "related_existing_lessons", []
            ),
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "version": lesson.get("version", 1),
    }


def prepare_content_block_document(
    block: Dict[str, Any],
    lesson_id_mapping: Dict[str, ObjectId],
    anchor_id_mapping: Dict[str, ObjectId],
) -> Dict[str, Any]:
    """Prepare content block document for MongoDB insertion"""

    # Get lesson ObjectId
    block_lesson_id = block["lesson_id"]
    lesson_id = lesson_id_mapping.get(block_lesson_id)
    if not lesson_id:
        # Try alternative mappings
        lesson_id = lesson_id_mapping.get(block_lesson_id.replace("-", "_"))
        if not lesson_id:
            lesson_id = lesson_id_mapping.get(block_lesson_id.replace("_", "-"))
            if not lesson_id:
                available_lessons = list(lesson_id_mapping.keys())
                raise ValueError(
                    f"Lesson not found for content block: {block_lesson_id}. "
                    f"Available lessons: {available_lessons}"
                )

    # Convert anchor references
    anchor_refs = []
    for anchor_ref in block.get("anchor_ids", []):
        if anchor_ref in anchor_id_mapping:
            anchor_refs.append(anchor_id_mapping[anchor_ref])

    return {
        "lesson_id": lesson_id,
        "type": block["type"],
        "order": block.get("order", 1),
        "payload": block.get("payload", {}),
        "anchor_ids": anchor_refs,
        "metadata": {
            "source_text": block.get("metadata", {}).get("source_text", ""),
            "generation_confidence": block.get("metadata", {}).get(
                "generation_confidence", 0.8
            ),
            "manual_review_needed": block.get("metadata", {}).get(
                "manual_review_needed", False
            ),
            "integration_notes": block.get("metadata", {}).get("integration_notes", ""),
        },
        "created_at": datetime.utcnow(),
    }


def update_lesson_block_references(
    db, lesson_id_mapping: Dict[str, ObjectId], content_block_ids: List[str]
) -> None:
    """Update lessons with references to their content blocks"""

    try:
        # Get all content blocks to build lesson -> blocks mapping
        block_cursor = db.content_blocks.find(
            {"_id": {"$in": [ObjectId(bid) for bid in content_block_ids]}},
            {"lesson_id": 1, "type": 1, "_id": 1},
        )

        lesson_blocks = {}
        quiz_ids = {}
        interactive_ids = {}

        for block in block_cursor:
            lesson_id = block["lesson_id"]
            block_id = block["_id"]
            block_type = block["type"]

            if lesson_id not in lesson_blocks:
                lesson_blocks[lesson_id] = []
                quiz_ids[lesson_id] = []
                interactive_ids[lesson_id] = []

            lesson_blocks[lesson_id].append(block_id)

            if block_type == "quiz":
                quiz_ids[lesson_id].append(block_id)
            elif block_type == "interactive":
                interactive_ids[lesson_id].append(block_id)

        # Update each lesson with its block references
        for lesson_id, block_list in lesson_blocks.items():
            db.lessons.update_one(
                {"_id": lesson_id},
                {
                    "$set": {
                        "blocks": block_list,
                        "quiz_ids": quiz_ids.get(lesson_id, []),
                        "interactive_ids": interactive_ids.get(lesson_id, []),
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

        logging.info(f"Updated {len(lesson_blocks)} lessons with block references")

    except Exception as e:
        logging.error(f"Failed to update lesson block references: {str(e)}")
        raise


def update_processing_state(
    db,
    session_id: str,
    created_ids: Dict[str, List[str]],
    integration_plan: Dict[str, Any],
) -> None:
    """Update processing state with created content IDs"""

    try:
        # Convert string IDs back to ObjectIds for storage
        lessons_created = [ObjectId(lid) for lid in created_ids["lessons"]]
        anchors_created = [ObjectId(aid) for aid in created_ids["anchors"]]

        # Update processing state
        db.processing_states.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "chunks_processed": {
                        "chunk_id": f"complete_lessons_{datetime.utcnow().isoformat()}",
                        "pages": integration_plan.get("source_pages", []),
                        "processed_at": datetime.utcnow(),
                        "lessons_created": lessons_created,
                        "anchors_created": anchors_created,
                        "overlap_with_previous": False,
                    }
                },
                "$set": {"status": "completed", "updated_at": datetime.utcnow()},
            },
            upsert=True,
        )

        logging.info(f"Updated processing state for session {session_id}")

    except Exception as e:
        logging.error(f"Failed to update processing state: {str(e)}")
        raise
