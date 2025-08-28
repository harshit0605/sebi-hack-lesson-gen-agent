"""
Content generation nodes for lesson creation workflow.
"""

from .load_existing_content import load_existing_content
from .analyze_new_content import analyze_new_content
from .map_to_existing_content import map_to_existing_content
from .create_integration_plan import create_integration_plan
from .evaluate_journey_fit import evaluate_journey_fit
from .create_new_journey_if_needed import create_new_journey_if_needed
from .generate_complete_lessons import generate_complete_lessons
from .persist_to_database import persist_to_database_node as persist_to_database
from .validate_and_finalize import validate_and_finalize
from .handle_validation_errors import handle_validation_errors
from .request_human_review import request_human_review
from .generate_lesson_content import generate_lesson_content_node

# Legacy imports (deprecated)
from .execute_integration_plan import execute_integration_plan
from .generate_structured_content import generate_structured_content

__all__ = [
    "load_existing_content",
    "analyze_new_content",
    "map_to_existing_content",
    "create_integration_plan",
    "evaluate_journey_fit",
    "create_new_journey_if_needed",
    "generate_complete_lessons",
    "persist_to_database",
    "validate_and_finalize",
    "handle_validation_errors",
    "request_human_review",
    "generate_lesson_content_node",
    # Legacy (deprecated)
    "execute_integration_plan",
    "generate_structured_content",
]
