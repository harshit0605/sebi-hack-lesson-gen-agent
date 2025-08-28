"""
Unified lesson generation node that creates complete lessons with all content.

This node replaces the two-step process of execute_integration_plan + generate_structured_content
with a single, efficient lesson generation process.
"""

import logging
from typing import List, Dict, Any
from langgraph.types import Send

from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    ContentIntegrationAction,
    LessonModel,
    ContentBlockModel,
    AnchorModel,
    VoiceScriptModel,
)


# Import strategy implementations
from agents.graphs.content_generation.nodes.strategies import (
    create_new_lessons,
    extend_existing_lessons,
    merge_with_existing_lessons,
    split_into_multiple_lessons,
)


async def generate_complete_lessons(state: LessonCreationState) -> LessonCreationState:
    """
    Generate lesson metadata and prepare for parallel content generation.

    This function:
    1. Executes the appropriate integration strategy
    2. Stores lesson metadata in state for conditional edge processing
    3. Returns updated state
    """

    try:
        # Step 1: Execute integration strategy to create lesson metadata
        lessons_metadata = await execute_integration_strategy(state)

        if not lessons_metadata:
            state["validation_errors"].append(
                "No lessons generated from integration strategy"
            )
            state["lessons_for_content_generation"] = []
        else:
            # Store lessons metadata for conditional edge processing
            state["lessons_for_content_generation"] = lessons_metadata
            logging.info(
                f"Prepared {len(lessons_metadata)} lessons for parallel content generation"
            )
        print(
            "lessons_for_content_generation length",
            len(state.get("lessons_for_content_generation", [])),
        )
        return state

    except Exception as e:
        error_msg = f"Failed to prepare lesson generation: {str(e)}"
        state["validation_errors"].append(error_msg)
        logging.error(error_msg)
        state["lessons_for_content_generation"] = []
        return state


async def execute_integration_strategy(state: LessonCreationState) -> List[LessonModel]:
    """
    Execute the appropriate integration strategy and return lesson metadata.

    This function delegates to the existing strategy implementations but expects
    them to return complete LessonModel objects (not just metadata).
    """

    integration_plan = state["integration_plan"]

    try:
        if integration_plan.action == ContentIntegrationAction.CREATE_NEW_LESSON:
            await create_new_lessons(state)
        elif integration_plan.action == ContentIntegrationAction.EXTEND_EXISTING_LESSON:
            await extend_existing_lessons(state)
        elif integration_plan.action == ContentIntegrationAction.MERGE_WITH_EXISTING:
            await merge_with_existing_lessons(state)
        elif integration_plan.action == ContentIntegrationAction.SPLIT_INTO_MULTIPLE:
            await split_into_multiple_lessons(state)
        else:
            raise ValueError(f"Unknown integration action: {integration_plan.action}")

        # Extract lessons from state (strategies update state with lessons)
        lessons = state.get("new_lessons", []) + state.get("updated_lessons", [])

        logging.info(f"Integration strategy executed: {integration_plan.action.value}")
        state.pop("new_lessons", None)
        state.pop("updated_lessons", None)

        return lessons

    except Exception as e:
        error_msg = f"Failed to execute integration strategy: {str(e)}"
        logging.error(error_msg)
        raise


async def validate_complete_lessons(
    lessons: List[LessonModel],
    blocks: List[ContentBlockModel],
    anchors: List[AnchorModel],
    voice_scripts: List[VoiceScriptModel],
) -> Dict[str, Any]:
    """
    Validate that all generated lessons and content are complete and consistent.

    Returns:
        Dict with 'valid' boolean and 'errors' list
    """

    errors = []

    # Validate lessons
    for lesson in lessons:
        if not lesson.title or not lesson.slug:
            errors.append(f"Lesson missing title or slug: {lesson}")

        if not lesson.learning_objectives:
            errors.append(f"Lesson missing learning objectives: {lesson.title}")

        if not lesson.blocks:
            errors.append(f"Lesson missing content blocks: {lesson.title}")

    # Validate blocks
    lesson_ids = {lesson.slug for lesson in lessons}
    anchor_labels = {anchor.short_label for anchor in anchors}
    for block in blocks:
        if block.lesson_id not in lesson_ids:
            errors.append(f"Block references non-existent lesson: {block.lesson_id}")
        # Validate each referenced anchor exists by short_label
        for aid in getattr(block, "anchor_ids", []) or []:
            if aid not in anchor_labels:
                errors.append(
                    f"Block references unknown anchor '{aid}' in lesson {block.lesson_id}"
                )

    # Validate voice scripts
    for script in voice_scripts:
        if script.lesson_id not in lesson_ids:
            errors.append(
                f"Voice script references non-existent lesson: {script.lesson_id}"
            )

    return {"valid": len(errors) == 0, "errors": errors}


def continue_to_lesson_content_generation(state: LessonCreationState):
    """
    Conditional edge function that creates Send commands for parallel lesson content generation.

    This function is called by LangGraph's conditional edge to determine parallel execution.
    """
    lessons_metadata = state.get("lessons_for_content_generation", [])
    print(
        "Node: continue_to_lesson_content_generation -> lessons_metadata length",
        len(lessons_metadata),
    )
    if not lessons_metadata:
        # No lessons to process, skip to collection
        return "collect_lesson_results"

    # Create Send commands for parallel execution
    send_commands = []
    for lesson_meta in lessons_metadata:
        logging.info(f"Creating Send command for lesson: {lesson_meta.title}")

        # Create state for this specific lesson
        lesson_state = {
            **state,  # Copy shared state
            "lesson_metadata": lesson_meta,  # Add specific lesson metadata
        }

        # Create Send command for parallel execution
        send_commands.append(Send("generate_lesson_content_node", lesson_state))

    logging.info(f"Created {len(send_commands)} parallel content generation tasks")
    print(
        "Node: continue_to_lesson_content_generation -> send_commands length",
        len(send_commands),
    )
    return send_commands


async def collect_lesson_results(state: LessonCreationState) -> LessonCreationState:
    """
    Collect and aggregate results from parallel lesson content generation.

    This is the reduce step in the map-reduce pattern.
    """

    all_lessons = []
    all_blocks = []
    all_anchors = []
    all_voice_scripts = []

    try:
        # Get results from parallel execution
        lesson_results = state.get("lesson_content_results", [])

        if not lesson_results:
            state["validation_errors"].append(
                "No lesson content results found from parallel execution"
            )
            return state

        # Process each lesson result
        for result in lesson_results:
            if "error" in result:
                state["validation_errors"].append(result["error"])
                continue

            # Extract content from successful results
            lesson = result.get("lesson")
            blocks = result.get("blocks", [])
            anchors = result.get("anchors", [])
            voice_scripts = result.get("voice_scripts", [])

            if lesson:
                all_lessons.append(lesson)
                all_blocks.extend(blocks)
                all_anchors.extend(anchors)
                all_voice_scripts.extend(voice_scripts)

                logging.info(f"Collected content for lesson: {lesson.title}")

        # Step 3: Validate and store aggregated results
        if all_lessons:
            # Store complete lessons and content
            state["lessons"] = all_lessons
            state["content_blocks"] = all_blocks
            state["anchors"] = all_anchors
            state["voice_scripts"] = all_voice_scripts

            # Update state tracking
            state["current_step"] = "lessons_generated"

            # Clear old partial lesson fields
            state.pop("new_lessons", None)
            state.pop("updated_lessons", None)
            state.pop("lesson_content_results", None)

            # Validate complete lessons
            validation_result = await validate_complete_lessons(
                all_lessons, all_blocks, all_anchors, all_voice_scripts
            )

            if not validation_result["valid"]:
                state["validation_errors"].extend(validation_result["errors"])

            # Generate statistics
            stats = get_lesson_generation_stats(
                all_lessons, all_blocks, all_anchors, all_voice_scripts
            )

            logging.info(
                f"Successfully collected {len(all_lessons)} complete lessons with "
                f"{len(all_blocks)} blocks, {len(all_anchors)} anchors, "
                f"{len(all_voice_scripts)} voice scripts. Stats: {stats}"
            )
        else:
            state["validation_errors"].append(
                "No complete lessons were collected from parallel execution"
            )

    except Exception as e:
        error_msg = f"Failed to collect lesson results: {str(e)}"
        state["validation_errors"].append(error_msg)
        logging.error(error_msg)

    return state


def get_lesson_generation_stats(
    lessons: List[LessonModel],
    blocks: List[ContentBlockModel],
    anchors: List[AnchorModel],
    voice_scripts: List[VoiceScriptModel],
) -> Dict[str, Any]:
    """Get statistics about generated lesson content"""

    return {
        "total_lessons": len(lessons),
        "total_blocks": len(blocks),
        "total_anchors": len(anchors),
        "total_voice_scripts": len(voice_scripts),
        "avg_blocks_per_lesson": len(blocks) / len(lessons) if lessons else 0,
        "avg_anchors_per_lesson": len(anchors) / len(lessons) if lessons else 0,
        "lessons_with_voice": sum(1 for lesson in lessons if lesson.voice_ready),
        "block_types": {
            block_type: sum(1 for block in blocks if block.type.value == block_type)
            for block_type in set(block.type.value for block in blocks)
        }
        if blocks
        else {},
    }
