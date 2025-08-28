from typing import Optional, List, Dict, Any
from .db import get_database
import logging
from pymongo.errors import PyMongoError
from pymongo import ASCENDING


def load_journeys_from_db(
    limit: Optional[int] = None,
    status_filter: Optional[str] = None,
    level_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load learning journeys from MongoDB with optional filtering"""

    try:
        db = get_database()
        collection = db.learning_journeys

        # Build query filter
        query = {}
        if status_filter:
            query["status"] = status_filter
        if level_filter:
            query["level"] = level_filter

        # Build aggregation pipeline for enriched data
        pipeline = [
            {"$match": query},
            {
                "$lookup": {
                    "from": "lessons",
                    "localField": "slug",
                    "foreignField": "journey_id",
                    "as": "lessons",
                }
            },
            {
                "$addFields": {
                    "lesson_count": {"$size": "$lessons"},
                    "avg_lesson_duration": {"$avg": "$lessons.estimated_minutes"},
                }
            },
            {
                "$project": {
                    "lessons": 0  # Remove lessons array to keep response size manageable
                }
            },
            {"$sort": {"order": ASCENDING}},
        ]

        if limit:
            pipeline.append({"$limit": limit})

        # Execute aggregation
        cursor = collection.aggregate(pipeline)
        journeys = cursor.to_list(length=None)

        # Convert ObjectId to string for JSON serialization
        for journey in journeys:
            if "_id" in journey:
                journey["_id"] = str(journey["_id"])

        logging.info(f"Loaded {len(journeys)} journeys from database")
        return journeys

    except PyMongoError as e:
        logging.error(f"MongoDB error loading journeys: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading journeys: {str(e)}")
        return []


def load_journey_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """Load a specific journey by slug with full lesson details"""

    try:
        db = get_database()

        pipeline = [
            {"$match": {"slug": slug}},
            {
                "$lookup": {
                    "from": "lessons",
                    "localField": "slug",
                    "foreignField": "journey_id",
                    "as": "lessons",
                }
            },
            {
                "$addFields": {
                    "lessons": {
                        "$sortArray": {"input": "$lessons", "sortBy": {"order": 1}}
                    }
                }
            },
        ]

        cursor = db.learning_journeys.aggregate(pipeline)
        result = cursor.to_list(length=1)

        if result:
            journey = result[0]
            journey["_id"] = str(journey["_id"])

            # Convert lesson ObjectIds
            for lesson in journey.get("lessons", []):
                lesson["_id"] = str(lesson["_id"])

            return journey

        return None

    except Exception as e:
        logging.error(f"Error loading journey by slug {slug}: {str(e)}")
        return None


# # db.py - Update to use Motor
# import motor.motor_asyncio
# from typing import Optional, List, Dict, Any
# import logging
# from pymongo import ASCENDING
# import os


# # Replace your get_database function
# def get_async_database():
#     """Get async MongoDB database connection"""
#     MONGODB_URL = os.getenv(
#         "MONGODB_URL",
#         "mongodb+srv://karnatak95:E1gwr8l8n9FKlZwF@cluster0.pmvh3ud.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
#     )
#     client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
#     return client.your_database_name


# async def load_journeys_from_db_async(
#     limit: Optional[int] = None,
#     status_filter: Optional[str] = None,
#     level_filter: Optional[str] = None,
# ) -> List[Dict[str, Any]]:
#     """Load learning journeys from MongoDB with optional filtering - ASYNC VERSION"""

#     try:
#         db = get_async_database()
#         collection = db.learning_journeys

#         # Build query filter
#         query = {}
#         if status_filter:
#             query["status"] = status_filter
#         if level_filter:
#             query["level"] = level_filter

#         # Build aggregation pipeline for enriched data
#         pipeline = [
#             {"$match": query},
#             {
#                 "$lookup": {
#                     "from": "lessons",
#                     "localField": "slug",
#                     "foreignField": "journey_id",
#                     "as": "lessons",
#                 }
#             },
#             {
#                 "$addFields": {
#                     "lesson_count": {"$size": "$lessons"},
#                     "avg_lesson_duration": {"$avg": "$lessons.estimated_minutes"},
#                 }
#             },
#             {
#                 "$project": {
#                     "lessons": 0  # Remove lessons array to keep response size manageable
#                 }
#             },
#             {"$sort": {"order": ASCENDING}},
#         ]

#         if limit:
#             pipeline.append({"$limit": limit})

#         # Execute aggregation - NOW ASYNC
#         cursor = collection.aggregate(pipeline)
#         journeys = await cursor.to_list(length=None)  # This is now async

#         # Convert ObjectId to string for JSON serialization
#         for journey in journeys:
#             if "_id" in journey:
#                 journey["_id"] = str(journey["_id"])

#         logging.info(f"Loaded {len(journeys)} journeys from database")
#         return journeys

#     except Exception as e:
#         logging.error(f"Error loading journeys: {str(e)}")
#         return []
