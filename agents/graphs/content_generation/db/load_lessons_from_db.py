from typing import Optional, List, Dict, Any
from .db import get_database
import logging
from pymongo.errors import PyMongoError
from pymongo import ASCENDING


def load_lessons_from_db(
    journey_id: Optional[str] = None,
    limit: Optional[int] = None,
    include_blocks: bool = False,
    review_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load lessons from MongoDB with optional filtering and related data"""

    try:
        db = get_database()
        collection = db.lessons

        # Build query filter
        query = {}
        if journey_id:
            query["journey_id"] = journey_id
        if review_status:
            query["metadata.review_status"] = review_status

        # Build aggregation pipeline
        pipeline = [
            {"$match": query},
            {
                "$lookup": {
                    "from": "learning_journeys",
                    "localField": "journey_id",
                    "foreignField": "slug",
                    "as": "journey_info",
                }
            },
            {
                "$addFields": {
                    "journey_title": {"$arrayElemAt": ["$journey_info.title", 0]},
                    "journey_level": {"$arrayElemAt": ["$journey_info.level", 0]},
                }
            },
        ]

        # Optionally include content blocks
        if include_blocks:
            pipeline.extend(
                [
                    {
                        "$lookup": {
                            "from": "content_blocks",
                            "localField": "slug",
                            "foreignField": "lesson_id",
                            "as": "content_blocks",
                        }
                    },
                    {
                        "$addFields": {
                            "content_blocks": {
                                "$sortArray": {
                                    "input": "$content_blocks",
                                    "sortBy": {"order": 1},
                                }
                            },
                            "block_count": {"$size": "$content_blocks"},
                            "block_types": {"$setUnion": ["$content_blocks.type", []]},
                        }
                    },
                ]
            )

        # Add final projections and sorting
        pipeline.extend(
            [
                {
                    "$project": {
                        "journey_info": 0  # Remove lookup array
                    }
                },
                {"$sort": {"journey_id": ASCENDING, "order": ASCENDING}},
            ]
        )

        if limit:
            pipeline.append({"$limit": limit})

        # Execute aggregation
        cursor = collection.aggregate(pipeline)
        lessons = cursor.to_list(length=None)

        # Convert ObjectIds to strings
        for lesson in lessons:
            if "_id" in lesson:
                lesson["_id"] = str(lesson["_id"])

            # Convert content block ObjectIds if included
            if include_blocks:
                for block in lesson.get("content_blocks", []):
                    if "_id" in block:
                        block["_id"] = str(block["_id"])

        logging.info(f"Loaded {len(lessons)} lessons from database")
        return lessons

    except PyMongoError as e:
        logging.error(f"MongoDB error loading lessons: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading lessons: {str(e)}")
        return []


def load_lesson_by_slug(
    slug: str, include_full_content: bool = True
) -> Optional[Dict[str, Any]]:
    """Load a specific lesson with full content details"""

    try:
        db = get_database()

        pipeline = [
            {"$match": {"slug": slug}},
            {
                "$lookup": {
                    "from": "content_blocks",
                    "localField": "slug",
                    "foreignField": "lesson_id",
                    "as": "content_blocks",
                }
            },
            {
                "$lookup": {
                    "from": "anchors",
                    "localField": "anchors",
                    "foreignField": "_id",
                    "as": "anchor_details",
                }
            },
        ]

        if include_full_content:
            pipeline.extend(
                [
                    {
                        "$lookup": {
                            "from": "voice_scripts",
                            "localField": "voice_script_id",
                            "foreignField": "lesson_id",
                            "as": "voice_script",
                        }
                    },
                    {
                        "$addFields": {
                            "voice_script": {"$arrayElemAt": ["$voice_script", 0]}
                        }
                    },
                ]
            )

        # Sort content blocks
        pipeline.append(
            {
                "$addFields": {
                    "content_blocks": {
                        "$sortArray": {
                            "input": "$content_blocks",
                            "sortBy": {"order": 1},
                        }
                    }
                }
            }
        )

        cursor = db.lessons.aggregate(pipeline)
        result = cursor.to_list(length=1)

        if result:
            lesson = result[0]
            lesson["_id"] = str(lesson["_id"])

            # Convert nested ObjectIds
            for block in lesson.get("content_blocks", []):
                block["_id"] = str(block["_id"])

            for anchor in lesson.get("anchor_details", []):
                anchor["_id"] = str(anchor["_id"])

            if lesson.get("voice_script") and "_id" in lesson["voice_script"]:
                lesson["voice_script"]["_id"] = str(lesson["voice_script"]["_id"])

            return lesson

        return None

    except Exception as e:
        logging.error(f"Error loading lesson by slug {slug}: {str(e)}")
        return None
