"""
Unified lesson generation node that creates complete lessons with all content.

This node replaces the two-step process of execute_integration_plan + generate_structured_content
with a single, efficient lesson generation process.
"""

import logging
from typing import List, Dict, Any, Tuple
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
    extend_existing_lessons,
    merge_with_existing_lessons,
    split_into_multiple_lessons,
)
from agents.graphs.content_generation.nodes.strategies.create_new_lesson import (
    create_lesson_from_distribution,
)


async def generate_complete_lessons(state: LessonCreationState) -> LessonCreationState:
    """
    Generate lesson metadata and prepare for parallel content generation.

    This function:
    1. Executes the appropriate integration strategy
    2. For CREATE_NEW, prepares distributions for metadata fan-out
    3. Stores lesson metadata in state for conditional edge processing (non CREATE_NEW)
    4. Returns updated state
    """

    try:
        # Step 1: Execute integration strategy or prepare distributions
        lessons_metadata, strat_updates = await execute_integration_strategy(state)

        updates: Dict[str, Any] = {}

        # If we are in CREATE_NEW flow, we only prepared distributions; metadata will be created via fan-out
        dists = strat_updates.get("lesson_distributions_for_creation") or state.get("lesson_distributions_for_creation")
        if dists:
            logging.info(
                f"Prepared {len(dists)} lesson distributions for metadata fan-out"
            )
            # Include distributions to persist this update in merged state
            updates["lesson_distributions_for_creation"] = dists
        else:
            # Non CREATE_NEW flows should have lessons collected already
            if not lessons_metadata:
                logging.warning("No lessons generated from integration strategy")
                updates["validation_errors"] = state.get("validation_errors", []) + [
                    "No lessons generated from integration strategy"
                ]
                updates["lessons_for_content_generation"] = []
            else:
                updates["lessons_for_content_generation"] = lessons_metadata
                logging.info(
                    f"Prepared {len(lessons_metadata)} lessons for parallel content generation"
                )

        print(
            "lessons_for_content_generation length",
            len(updates.get("lessons_for_content_generation", state.get("lessons_for_content_generation", []))),
        )
        # Merge any additional strategy-produced partial updates (non-destructive)
        if strat_updates:
            updates.update(strat_updates)

        return updates

    except Exception as e:
        error_msg = f"Failed to prepare lesson generation: {str(e)}"
        logging.error(error_msg)
        return {
            "validation_errors": state.get("validation_errors", []) + [error_msg],
            "lessons_for_content_generation": [],
        }


async def execute_integration_strategy(state: LessonCreationState) -> Tuple[List[LessonModel], Dict[str, Any]]:
    """
    Execute the appropriate integration strategy and return lesson metadata.

    This function delegates to the existing strategy implementations but expects
    them to return complete LessonModel objects (not just metadata).
    """

    integration_plan = state["integration_plan"]
    partial_updates: Dict[str, Any] = {}

    try:
        if integration_plan.action == ContentIntegrationAction.CREATE_NEW_LESSON:
            # For create-new, do not create metadata serially; prepare distributions for fan-out
            new_lesson_distributions = [
                dist
                for dist in integration_plan.content_distribution
                if dist.integration_type == "new_lesson"
            ]
            # Return distributions via partial updates instead of mutating state
            partial_updates["lesson_distributions_for_creation"] = new_lesson_distributions
            logging.info(
                f"Prepared {len(new_lesson_distributions)} distributions for create-new metadata generation"
            )
            return [], partial_updates
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
        # Do not mutate state here; return lessons only
        return lessons, partial_updates

    except Exception as e:
        error_msg = f"Failed to execute integration strategy: {str(e)}"
        logging.error(error_msg)
        raise


def continue_to_lesson_metadata_generation(state: LessonCreationState):
    """
    Conditional edge that fans out per-distribution lesson metadata creation for CREATE_NEW flow.
    If no distributions exist, skip directly to metadata collection.
    """
    dists = state.get("lesson_distributions_for_creation", [])
    if not dists:
        return "collect_lesson_metadata"

    send_cmds = []
    for dist in dists:
        logging.info(f"Creating Send for lesson metadata: {getattr(dist, 'lesson_title', 'untitled')}")
        send_cmds.append(
            Send(
                "create_lesson_metadata_node",
                {**state, "lesson_distribution": dist},
            )
        )
    return send_cmds


async def create_lesson_metadata_node(state: LessonCreationState) -> LessonCreationState:
    """
    Create a single LessonModel (metadata only) from a LessonContentDistribution.
    Returns reducer-friendly payload in lesson_metadata_results.
    """
    try:
        dist = state.get("lesson_distribution")
        if not dist:
            raise ValueError("lesson_distribution missing in state")

        integration_plan = state["integration_plan"]
        journey_plan = state.get("journey_creation_plan")
        existing_journeys = state.get("existing_journeys_list", [])

        lesson = await create_lesson_from_distribution(
            dist, integration_plan, journey_plan, state, existing_journeys
        )

        return {"lesson_metadata_results": [lesson]}
    except Exception as e:
        logging.error(f"Failed to create lesson metadata: {str(e)}")
        return {"lesson_metadata_results": []}


async def collect_lesson_metadata(state: LessonCreationState) -> LessonCreationState:
    """
    Collect metadata results and set lessons_for_content_generation.
    If results are empty and lessons_for_content_generation already exists (non CREATE_NEW flows), no-op.
    """
    results = state.get("lesson_metadata_results", [])

    if results:
        logging.info(
            f"Collected {len(results)} lesson metadata items for content generation"
        )
        return {"lessons_for_content_generation": results}

    # Non CREATE_NEW flows might have populated lessons_for_content_generation already
    if not state.get("lessons_for_content_generation"):
        return {"lessons_for_content_generation": []}

    # No changes
    return {}

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

    all_lessons: List[LessonModel] = []
    all_blocks: List[ContentBlockModel] = []
    all_anchors: List[AnchorModel] = []
    all_voice_scripts: List[VoiceScriptModel] = []

    try:
        # Get results from parallel execution
        lesson_results = state.get("lesson_content_results", [])

        if not lesson_results:
            return {
                "validation_errors": state.get("validation_errors", [])
                + ["No lesson content results found from parallel execution"],
            }

        # Process each lesson result
        aggregated_errors = list(state.get("validation_errors", []))
        for result in lesson_results:
            if "error" in result:
                # accumulate error, will return merged errors later
                aggregated_errors.append(result["error"])
                # continue collecting other results; errors will be returned below
                # Note: we don't update updates dict per iteration to avoid duplicates
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
            # Validate complete lessons
            validation_result = await validate_complete_lessons(
                all_lessons, all_blocks, all_anchors, all_voice_scripts
            )

            updates: Dict[str, Any] = {
                "lessons": all_lessons,
                "content_blocks": all_blocks,
                "anchors": all_anchors,
                "voice_scripts": all_voice_scripts,
                "current_step": "lessons_generated",
            }

            if not validation_result["valid"]:
                updates["validation_errors"] = aggregated_errors + validation_result["errors"]
            elif aggregated_errors:
                updates["validation_errors"] = aggregated_errors

            # Generate statistics
            stats = get_lesson_generation_stats(
                all_lessons, all_blocks, all_anchors, all_voice_scripts
            )

            logging.info(
                f"Successfully collected {len(all_lessons)} complete lessons with "
                f"{len(all_blocks)} blocks, {len(all_anchors)} anchors, "
                f"{len(all_voice_scripts)} voice scripts. Stats: {stats}"
            )
            return updates
        else:
            return {
                "validation_errors": aggregated_errors
                + ["No complete lessons were collected from parallel execution"],
            }

    except Exception as e:
        error_msg = f"Failed to collect lesson results: {str(e)}"
        logging.error(error_msg)
        return {"validation_errors": state.get("validation_errors", []) + [error_msg]}


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
