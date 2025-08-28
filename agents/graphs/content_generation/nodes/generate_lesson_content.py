"""Generate lesson content including blocks, anchors, and voice scripts."""

import logging

from agents.graphs.content_generation.utils.content_generators import (
    generate_complete_lesson_content,
    populate_lesson_with_content,
)
from agents.graphs.content_generation.state import LessonCreationState


async def generate_lesson_content_node(
    state: LessonCreationState,
) -> LessonCreationState:
    """
    LangGraph node to generate content for a single lesson in parallel execution.

    This node expects:
    - lesson_metadata: LessonModel in state (passed via Send API)
    - content_analysis: ContentAnalysisResult in state (shared across all parallel executions)

    Returns updated state with generated content for this lesson.
    """

    try:
        # Extract lesson metadata passed via Send API
        lesson_metadata = state.get("lesson_metadata")
        if not lesson_metadata:
            raise ValueError(
                "No lesson_metadata found in state for parallel content generation"
            )

        # Get shared content analysis from state
        content_analysis = state["content_analysis"]

        logging.info(f"Generating content for lesson: {lesson_metadata.title}")

        # Generate all content for this lesson
        content_data = await generate_complete_lesson_content(
            lesson_metadata, content_analysis, state
        )

        # Populate lesson with content references
        complete_lesson = populate_lesson_with_content(lesson_metadata, content_data)

        # Return state with generated content for this specific lesson
        # This will be accumulated in lesson_content_results by the reducer
        return {
            "lesson_content_results": [{
                "lesson": complete_lesson,
                "blocks": content_data["blocks"],
                "anchors": content_data["anchors"],
                "voice_scripts": content_data["voice_scripts"],
            }]
        }

    except Exception as e:
        error_msg = f"Failed to generate content for lesson {lesson_metadata.title if lesson_metadata else 'unknown'}: {str(e)}"
        logging.error(error_msg)

        return {
            "lesson_content_results": [{
                "error": error_msg,
                "lesson": lesson_metadata,
                "blocks": [],
                "anchors": [],
                "voice_scripts": [],
            }]
        }
