import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from langgraph.graph import StateGraph, END
from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.nodes import (
    load_existing_content,
    analyze_new_content,
    map_to_existing_content,
    create_integration_plan,
    evaluate_journey_fit,
    create_new_journey_if_needed,
    persist_to_database,
    validate_and_finalize,
    handle_validation_errors,
    request_human_review,
)

# Import corrected parallel lesson generation nodes
from agents.graphs.content_generation.nodes.generate_complete_lessons import (
    generate_complete_lessons,
    continue_to_lesson_metadata_generation,
    create_lesson_metadata_node,
    collect_lesson_metadata,
    continue_to_lesson_content_generation,
    collect_lesson_results,
)
from agents.graphs.content_generation.nodes.generate_lesson_content import (
    generate_lesson_content_node,
)

from agents.graphs.content_generation.nodes.evaluate_journey_fit import (
    journey_decision_gate,
)
from agents.graphs.content_generation.nodes.validate_and_finalize import validation_gate

from dotenv import load_dotenv

load_dotenv()


def create_enhanced_lesson_creation_graph():
    workflow = StateGraph(LessonCreationState)

    # Enhanced nodes with unified lesson generation
    workflow.add_node("load_existing_content", load_existing_content)
    workflow.add_node("analyze_new_content", analyze_new_content)
    workflow.add_node("map_to_existing_content", map_to_existing_content)
    workflow.add_node("create_integration_plan", create_integration_plan)
    workflow.add_node("evaluate_journey_fit", evaluate_journey_fit)
    workflow.add_node("create_new_journey", create_new_journey_if_needed)

    # Parallel lesson content generation nodes
    workflow.add_node("generate_complete_lessons", generate_complete_lessons)
    workflow.add_node("create_lesson_metadata_node", create_lesson_metadata_node)
    workflow.add_node("collect_lesson_metadata", collect_lesson_metadata)
    workflow.add_node("generate_lesson_content_node", generate_lesson_content_node)
    workflow.add_node("collect_lesson_results", collect_lesson_results)

    workflow.add_node("persist_to_database", persist_to_database)
    workflow.add_node("validate_and_finalize", validate_and_finalize)

    # Error handling
    workflow.add_node("handle_validation_errors", handle_validation_errors)
    workflow.add_node("request_human_review", request_human_review)

    # Flow definition
    workflow.set_entry_point("load_existing_content")

    workflow.add_edge("load_existing_content", "analyze_new_content")
    workflow.add_edge("analyze_new_content", "create_integration_plan")
    # workflow.add_edge("map_to_existing_content", "create_integration_plan")
    workflow.add_edge("create_integration_plan", "evaluate_journey_fit")

    workflow.add_conditional_edges(
        "evaluate_journey_fit",
        journey_decision_gate,
        {
            "fits_existing": "generate_complete_lessons",
            "needs_new_journey": "create_new_journey",
            "unclear": "request_human_review",
        },
    )

    workflow.add_edge("create_new_journey", "generate_complete_lessons")

    # First: fan-out metadata generation (for CREATE_NEW) or skip to collector
    workflow.add_conditional_edges(
        "generate_complete_lessons",
        continue_to_lesson_metadata_generation,
        {
            "create_lesson_metadata_node": "create_lesson_metadata_node",
            "collect_lesson_metadata": "collect_lesson_metadata",
        },
    )

    # Edge from metadata nodes to metadata collector
    workflow.add_edge("create_lesson_metadata_node", "collect_lesson_metadata")

    # Then: fan-out content generation based on collected lesson metadata
    workflow.add_conditional_edges(
        "collect_lesson_metadata",
        continue_to_lesson_content_generation,
        {
            "generate_lesson_content_node": "generate_lesson_content_node",
            "collect_lesson_results": "collect_lesson_results",
        },
    )

    # Edge from parallel content nodes to collection
    workflow.add_edge("generate_lesson_content_node", "collect_lesson_results")
    workflow.add_edge("collect_lesson_results", "persist_to_database")
    workflow.add_edge("persist_to_database", "validate_and_finalize")

    workflow.add_conditional_edges(
        "validate_and_finalize",
        validation_gate,
        {
            "valid": END,
            "needs_review": "request_human_review",
            "has_errors": "handle_validation_errors",
        },
    )

    workflow.add_edge("handle_validation_errors", "generate_complete_lessons")
    workflow.add_edge("request_human_review", END)

    return workflow.compile()


agent = create_enhanced_lesson_creation_graph()
