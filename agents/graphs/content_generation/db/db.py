import os
import logging
from typing import Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError
from pymongo.server_api import ServerApi


# MongoDB configuration
MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb+srv://karnatak95:E1gwr8l8n9FKlZwF@cluster0.pmvh3ud.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "sebi_lesson_creator")
print(MONGODB_URL)
print(DATABASE_NAME)
# Global MongoDB client (initialized once)
_mongodb_client: Optional[MongoClient] = None
_database = None


def get_database():
    """Get MongoDB database connection with connection pooling (sync)"""
    global _mongodb_client, _database

    if _mongodb_client is None:
        _mongodb_client = MongoClient(
            MONGODB_URL,
            server_api=ServerApi("1"),
            # maxPoolSize=50,
            # minPoolSize=10,
            # maxIdleTimeMS=45000,
            # waitQueueTimeoutMS=5000,
            # serverSelectionTimeoutMS=5000,
        )
        _database = _mongodb_client[DATABASE_NAME]

        # Optional: force an initial server selection to fail fast if unreachable
        # Remove if not desired.
        _mongodb_client.admin.command("ping")

        # Create indexes for optimal performance
        create_database_indexes(_database)

    return _database


def create_database_indexes(db):
    """Create indexes for optimal query performance (sync)"""
    try:
        # Journey indexes
        db.learning_journeys.create_index([("slug", ASCENDING)], unique=True)
        db.learning_journeys.create_index([("level", ASCENDING), ("order", ASCENDING)])
        db.learning_journeys.create_index([("status", ASCENDING)])

        # Lesson indexes
        db.lessons.create_index([("slug", ASCENDING)], unique=True)
        db.lessons.create_index([("journey_id", ASCENDING), ("order", ASCENDING)])
        db.lessons.create_index([("metadata.review_status", ASCENDING)])
        db.lessons.create_index([("tags", ASCENDING)])

        # Anchor indexes
        db.anchors.create_index(
            [("source_type", ASCENDING), ("relevance_tags", ASCENDING)]
        )
        db.anchors.create_index([("created_from_chunk", ASCENDING)])
        db.anchors.create_index([("confidence_score", DESCENDING)])
        db.anchors.create_index([("last_verified_at", DESCENDING)])

        # Processing state indexes
        db.processing_states.create_index([("session_id", ASCENDING)], unique=True)
        db.processing_states.create_index(
            [("status", ASCENDING), ("updated_at", DESCENDING)]
        )

        logging.info("Database indexes created successfully")
    except PyMongoError as e:
        logging.warning(f"Index creation warning: {str(e)}")
