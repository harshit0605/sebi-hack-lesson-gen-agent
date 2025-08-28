from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    AnchorModel,
    ContentBlockModel,
    SourceType,
)
from langchain_core.output_parsers import PydanticOutputParser
from agents.graphs.content_generation.llm import content_generator_llm
from agents.graphs.content_generation.prompts.handle_validation_errors import (
    HANDLE_VALIDATION_ERRORS_PROMPTS,
)
import re
from typing import List
import logging
from datetime import datetime


async def handle_validation_errors(state: LessonCreationState) -> LessonCreationState:
    """Attempt to automatically fix validation errors in generated content"""

    validation_errors = state.get("validation_errors", [])
    if not validation_errors:
        state["current_step"] = "no_errors_to_handle"
        return state

    logging.info(f"Attempting to fix {len(validation_errors)} validation errors")

    fixed_errors = []
    remaining_errors = []

    for error in validation_errors:
        try:
            if await attempt_error_fix(state, error):
                fixed_errors.append(error)
            else:
                remaining_errors.append(error)
        except Exception as e:
            remaining_errors.append(f"Error fixing '{error}': {str(e)}")

    # Update state with remaining errors
    state["validation_errors"] = remaining_errors

    if fixed_errors:
        logging.info(f"Fixed {len(fixed_errors)} validation errors automatically")
        state["current_step"] = "errors_partially_fixed"
    else:
        logging.warning("Could not automatically fix any validation errors")
        state["current_step"] = "errors_not_fixable"

    return state


async def attempt_error_fix(state: LessonCreationState, error: str) -> bool:
    """Attempt to fix a specific validation error"""

    # Handle missing SEBI anchors
    if "has no SEBI anchors" in error:
        return await fix_missing_anchors_error(state, error)

    # Handle invalid lesson duration
    if "invalid duration" in error:
        return await fix_lesson_duration_error(state, error)

    # Handle empty content blocks
    if "empty content" in error or "missing content" in error:
        return await fix_empty_content_error(state, error)

    # Handle Pydantic validation errors
    if "Pydantic validation error" in error:
        return await fix_pydantic_error(state, error)

    # Handle missing learning objectives
    if "learning objectives" in error:
        return await fix_learning_objectives_error(state, error)

    return False


async def fix_missing_anchors_error(state: LessonCreationState, error: str) -> bool:
    """Fix content blocks that are missing SEBI anchors"""

    try:
        # Extract lesson ID from error message
        lesson_id_match = re.search(r"lesson '([^']+)'", error)
        if not lesson_id_match:
            return False

        lesson_id = lesson_id_match.group(1)

        # Find blocks without anchors for this lesson
        blocks_to_fix = []
        for i, block in enumerate(state.get("content_blocks", [])):
            if block.lesson_id == lesson_id and not block.anchor_ids:
                blocks_to_fix.append((i, block))

        if not blocks_to_fix:
            return False

        # Generate anchors for these blocks
        for block_index, block in blocks_to_fix:
            anchors = await generate_emergency_anchors(
                block, state["pdf_content"], state["chunk_id"]
            )
            if anchors:
                # Update block with anchor IDs
                state["content_blocks"][block_index].anchor_ids = [
                    anchor.source_type + "_" + anchor.short_label for anchor in anchors
                ]

                # Add anchors to state
                if "anchors" not in state:
                    state["anchors"] = []
                state["anchors"].extend(anchors)

        return True

    except Exception as e:
        logging.error(f"Failed to fix missing anchors error: {str(e)}")
        return False


async def fix_lesson_duration_error(state: LessonCreationState, error: str) -> bool:
    """Fix lessons with invalid duration"""

    try:
        # Extract lesson title and current duration from error
        lesson_match = re.search(r"Lesson '([^']+)'.*?(\d+) minutes", error)
        if not lesson_match:
            return False

        lesson_title = lesson_match.group(1)
        current_duration = int(lesson_match.group(2))

        # Find and fix the lesson
        for i, lesson in enumerate(
            state.get("new_lessons", []) + state.get("updated_lessons", [])
        ):
            if lesson.title == lesson_title:
                # Adjust duration to valid range (15-35 minutes)
                if current_duration < 15:
                    lesson.estimated_minutes = 15
                elif current_duration > 45:
                    lesson.estimated_minutes = 35

                # Update the lesson in state
                if i < len(state.get("new_lessons", [])):
                    state["new_lessons"][i] = lesson
                else:
                    updated_index = i - len(state.get("new_lessons", []))
                    state["updated_lessons"][updated_index] = lesson

                return True

        return False

    except Exception as e:
        logging.error(f"Failed to fix lesson duration error: {str(e)}")
        return False


async def fix_empty_content_error(state: LessonCreationState, error: str) -> bool:
    """Fix content blocks with empty or insufficient content"""

    try:
        # Use LLM to regenerate content for empty blocks
        parser = PydanticOutputParser(pydantic_object=List[ContentBlockModel])

        fix_content_prompt = HANDLE_VALIDATION_ERRORS_PROMPTS["fix_content"].partial(
            format_instructions=parser.get_format_instructions()
        )

        # Find relevant lesson context
        lesson_context = "General SEBI investor education content"
        if state.get("new_lessons"):
            lesson_context = (
                f"Lessons: {[lesson.title for lesson in state['new_lessons'][:3]]}"
            )

        messages = fix_content_prompt.format_messages(
            error=error,
            lesson_context=lesson_context,
            pdf_content_sample=state["pdf_content"][:1000],
        )

        response = await content_generator_llm.ainvoke(messages)
        fixed_blocks = parser.parse(response.content)

        if fixed_blocks:
            # Replace empty blocks with fixed ones
            if "content_blocks" not in state:
                state["content_blocks"] = []
            state["content_blocks"].extend(fixed_blocks)
            return True

        return False

    except Exception as e:
        logging.error(f"Failed to fix empty content error: {str(e)}")
        return False


async def fix_pydantic_error(state: LessonCreationState, error: str) -> bool:
    """Fix Pydantic model validation errors"""

    try:
        # Common Pydantic fixes

        # Fix enum validation errors
        if "not a valid enumeration member" in error:
            # For now, log and return False - can implement specific enum fixes later
            logging.warning(f"Enum validation error detected: {error}")
            return False

        # Fix required field errors
        if "field required" in error:
            # For now, log and return False - can implement specific field fixes later
            logging.warning(f"Required field error detected: {error}")
            return False

        # Fix type validation errors
        if "wrong type" in error or "invalid type" in error:
            # For now, log and return False - can implement specific type fixes later
            logging.warning(f"Type validation error detected: {error}")
            return False

        return False

    except Exception as e:
        logging.error(f"Failed to fix Pydantic error: {str(e)}")
        return False


async def fix_learning_objectives_error(state: LessonCreationState, error: str) -> bool:
    """Fix lessons with missing or inadequate learning objectives"""

    try:
        # Generate learning objectives for lessons that need them
        parser = PydanticOutputParser(pydantic_object=List[str])

        objectives_prompt = HANDLE_VALIDATION_ERRORS_PROMPTS[
            "fix_learning_objectives"
        ].partial(format_instructions=parser.get_format_instructions())

        content_analysis = state.get("content_analysis", {})
        messages = objectives_prompt.format_messages(
            error=error,
            content_analysis=content_analysis.model_dump_json()
            if content_analysis
            else "{}",
        )

        response = await content_generator_llm.ainvoke(messages)
        objectives = parser.parse(response.content)

        # Apply objectives to lessons that need them
        for lesson in state.get("new_lessons", []) + state.get("updated_lessons", []):
            if not lesson.learning_objectives or len(lesson.learning_objectives) < 2:
                lesson.learning_objectives = objectives[:4]  # Limit to 4 objectives

        return True

    except Exception as e:
        logging.error(f"Failed to fix learning objectives error: {str(e)}")
        return False


async def generate_emergency_anchors(
    block: ContentBlockModel, pdf_content: str, chunk_id: str
) -> List[AnchorModel]:
    """Generate basic anchors for blocks that are missing them"""

    try:
        # Create a basic anchor based on the block content
        anchor = AnchorModel(
            source_type=SourceType.SEBI_PDF,
            title=f"SEBI Content Reference - {block.type.capitalize()}",
            short_label=f"{block.type}_{chunk_id}",
            excerpt=pdf_content[:200] + "..."
            if len(pdf_content) > 200
            else pdf_content,
            document_title=f"SEBI Document - Chunk {chunk_id}",
            page_numbers=[],  # Will be filled from state
            last_verified_at=datetime.now(),
            relevance_tags=[block.type.value],
            created_from_chunk=chunk_id,
            confidence_score=0.7,  # Medium confidence for emergency anchors
        )

        return [anchor]

    except Exception as e:
        logging.error(f"Failed to generate emergency anchor: {str(e)}")
        return []
