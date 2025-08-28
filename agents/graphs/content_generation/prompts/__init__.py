"""
Prompt templates for SEBI lesson creation system.
Organized by task type for better maintainability.
"""

from .content_analysis import CONTENT_ANALYSIS_PROMPTS
from .lesson_creation import LESSON_CREATION_PROMPTS  
from .content_generation import CONTENT_GENERATION_PROMPTS
from .integration_planning import INTEGRATION_PLANNING_PROMPTS
from .quiz_generation import QUIZ_GENERATION_PROMPTS

__all__ = [
    "CONTENT_ANALYSIS_PROMPTS",
    "LESSON_CREATION_PROMPTS", 
    "CONTENT_GENERATION_PROMPTS",
    "INTEGRATION_PLANNING_PROMPTS",
    "QUIZ_GENERATION_PROMPTS"
]
