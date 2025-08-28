from typing import Optional, List, Dict, Any
from .db import get_database
from datetime import datetime, timedelta
from pymongo.errors import PyMongoError
from pymongo import DESCENDING
import logging


def load_anchors_from_db(
    source_type: Optional[str] = None,
    relevance_tags: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = None,
    created_from_chunk: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load anchors from MongoDB with comprehensive filtering options"""

    try:
        db = get_database()
        collection = db.anchors

        # Build query filter
        query = {}

        if source_type:
            query["source_type"] = source_type

        if relevance_tags:
            query["relevance_tags"] = {"$in": relevance_tags}

        if min_confidence is not None:
            query["confidence_score"] = {"$gte": min_confidence}

        if created_from_chunk:
            query["created_from_chunk"] = created_from_chunk

        # Only include recently verified anchors (within last 6 months)
        six_months_ago = datetime.now() - timedelta(days=180)
        query["last_verified_at"] = {"$gte": six_months_ago}

        # Build aggregation pipeline with usage statistics
        pipeline = [
            {"$match": query},
            {
                "$lookup": {
                    "from": "content_blocks",
                    "localField": "_id",
                    "foreignField": "anchor_ids",
                    "as": "usage_blocks",
                }
            },
            {
                "$addFields": {
                    "usage_count": {"$size": "$usage_blocks"},
                    "used_in_lessons": {"$setUnion": ["$usage_blocks.lesson_id", []]},
                }
            },
            {
                "$project": {
                    "usage_blocks": 0  # Remove the lookup array
                }
            },
            {
                "$sort": {
                    "confidence_score": DESCENDING,
                    "usage_count": DESCENDING,
                    "last_verified_at": DESCENDING,
                }
            },
        ]

        if limit:
            pipeline.append({"$limit": limit})

        # Execute aggregation
        cursor = collection.aggregate(pipeline)
        anchors = cursor.to_list(length=None)

        # Convert ObjectIds to strings
        for anchor in anchors:
            if "_id" in anchor:
                anchor["_id"] = str(anchor["_id"])

        logging.info(f"Loaded {len(anchors)} anchors from database")
        return anchors

    except PyMongoError as e:
        logging.error(f"MongoDB error loading anchors: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading anchors: {str(e)}")
        return []


def search_anchors_by_content(
    search_text: str, source_types: Optional[List[str]] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    """Search anchors by content using text search"""

    try:
        db = get_database()

        # Build text search query
        query = {"$text": {"$search": search_text}}

        if source_types:
            query["source_type"] = {"$in": source_types}

        # Find anchors with text search scoring
        cursor = (
            db.anchors.find(query, {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"}), ("confidence_score", DESCENDING)])
            .limit(limit)
        )

        anchors = cursor.to_list(length=None)

        # Convert ObjectIds
        for anchor in anchors:
            anchor["_id"] = str(anchor["_id"])

        return anchors

    except Exception as e:
        logging.error(f"Error searching anchors: {str(e)}")
        return []


def get_anchor_usage_statistics() -> Dict[str, Any]:
    """Get statistics about anchor usage across lessons"""

    try:
        db = get_database()

        pipeline = [
            {
                "$group": {
                    "_id": "$source_type",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"},
                    "latest_update": {"$max": "$last_verified_at"},
                }
            },
            {"$sort": {"count": DESCENDING}},
        ]

        cursor = db.anchors.aggregate(pipeline)
        stats = cursor.to_list(length=None)

        # Get total anchor count
        total_anchors = db.anchors.count_documents({})

        # Get usage statistics
        usage_pipeline = [
            {
                "$lookup": {
                    "from": "content_blocks",
                    "localField": "_id",
                    "foreignField": "anchor_ids",
                    "as": "blocks",
                }
            },
            {"$match": {"blocks": {"$ne": []}}},
            {"$count": "used_anchors"},
        ]

        usage_cursor = db.anchors.aggregate(usage_pipeline)
        usage_result = usage_cursor.to_list(length=1)
        used_anchors = usage_result[0]["used_anchors"] if usage_result else 0

        return {
            "total_anchors": total_anchors,
            "used_anchors": used_anchors,
            "unused_anchors": total_anchors - used_anchors,
            "usage_rate": (used_anchors / total_anchors * 100)
            if total_anchors > 0
            else 0,
            "by_source_type": stats,
        }

    except Exception as e:
        logging.error(f"Error getting anchor statistics: {str(e)}")
        return {}
