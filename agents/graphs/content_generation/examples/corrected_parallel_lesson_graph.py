"""
Corrected implementation of parallel lesson content generation using LangGraph Send API.

This example shows the proper way to set up conditional edges with Send commands
based on the LangGraph documentation pattern.
"""

import sys

sys.path.append("../../../../")

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.nodes.generate_complete_lessons import (
    generate_complete_lessons,
    continue_to_lesson_content_generation,
    collect_lesson_results,
)
from agents.graphs.content_generation.nodes.generate_lesson_content import (
    generate_lesson_content_node,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_corrected_parallel_lesson_graph():
    """
    Create a properly structured LangGraph for parallel lesson content generation.

    Graph structure:
    START -> generate_complete_lessons -> [conditional edge] -> parallel lesson nodes -> collect_lesson_results -> END
    """

    # Create the state graph
    builder = StateGraph(LessonCreationState)

    # Add nodes
    builder.add_node("generate_complete_lessons", generate_complete_lessons)
    builder.add_node("generate_lesson_content_node", generate_lesson_content_node)
    builder.add_node("collect_lesson_results", collect_lesson_results)

    # Add edges
    builder.add_edge(START, "generate_complete_lessons")

    # CRITICAL: Use conditional edge for Send commands
    # This is the correct way to handle Send API in LangGraph
    builder.add_conditional_edges(
        "generate_complete_lessons",
        continue_to_lesson_content_generation,  # This function returns Send commands or node name
        [
            "generate_lesson_content_node",
            "collect_lesson_results",
        ],  # Possible destinations
    )

    # Edge from parallel nodes to collection
    builder.add_edge("generate_lesson_content_node", "collect_lesson_results")
    builder.add_edge("collect_lesson_results", END)

    # Compile the graph
    graph = builder.compile()
    return graph


def demonstrate_send_api_pattern():
    """
    Demonstrate the corrected Send API pattern.
    """

    logger.info("=== CORRECTED Send API Pattern ===")
    logger.info("1. Node Function (generate_complete_lessons):")
    logger.info("   - Executes integration strategy")
    logger.info("   - Stores lesson metadata in state")
    logger.info("   - Returns updated state (NOT Send commands)")

    logger.info(
        "\n2. Conditional Edge Function (continue_to_lesson_content_generation):"
    )
    logger.info("   - Reads lesson metadata from state")
    logger.info("   - Creates Send commands for parallel execution")
    logger.info("   - Returns list of Send objects OR node name")

    logger.info("\n3. Graph Structure:")
    logger.info("   - Uses add_conditional_edges() for Send commands")
    logger.info("   - Conditional edge function handles Send creation")
    logger.info("   - Parallel nodes automatically route to collection")

    logger.info("\n=== Key Differences from Previous Implementation ===")
    logger.info("❌ WRONG: Node function returns Send commands directly")
    logger.info("✅ CORRECT: Conditional edge function returns Send commands")
    logger.info("❌ WRONG: Node function tries to control routing")
    logger.info("✅ CORRECT: LangGraph handles routing via conditional edges")


if __name__ == "__main__":
    # Demonstrate the corrected pattern
    demonstrate_send_api_pattern()

    # Create the corrected graph
    graph = create_corrected_parallel_lesson_graph()

    print("\n" + "=" * 60)
    print("CORRECTED SEND API IMPLEMENTATION")
    print("=" * 60)
    print("\nKey fixes applied:")
    print("1. generate_complete_lessons() now returns state, not Send commands")
    print("2. Added continue_to_lesson_content_generation() conditional edge function")
    print("3. Updated graph to use add_conditional_edges() for Send commands")
    print("4. Added lessons_for_content_generation field to state")
    print("\nThe graph is now properly structured for LangGraph Send API.")
