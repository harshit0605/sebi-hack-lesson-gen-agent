"""
Integration strategies for content generation.
Provides different approaches for integrating new content with existing lessons.
"""

from .create_new_lesson import create_new_lessons
from .extend_existing_lesson import extend_existing_lessons
from .merge_existing_lessons import merge_with_existing_lessons
from .split_into_lessons import split_into_multiple_lessons

__all__ = [
    "create_new_lessons",
    "extend_existing_lessons", 
    "merge_with_existing_lessons",
    "split_into_multiple_lessons"
]
