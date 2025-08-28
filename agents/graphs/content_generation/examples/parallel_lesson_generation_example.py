"""
Example demonstrating parallel lesson content generation using LangGraph Send API.

This example shows how the map-reduce pattern works for generating lesson content
in parallel, with proper state management and result aggregation.
"""

import asyncio
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Mock imports for demonstration (replace with actual imports)
from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.nodes.generate_complete_lessons import (
    generate_complete_lessons,
    collect_lesson_results,
)
from agents.graphs.content_generation.nodes.generate_lesson_content import (
    generate_lesson_content_node,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_parallel_lesson_generation_graph():
    """
    Create a LangGraph that demonstrates parallel lesson content generation.
    
    Graph structure:
    START -> generate_complete_lessons -> [parallel lesson content nodes] -> collect_lesson_results -> END
    """
    
    # Create the state graph
    builder = StateGraph(LessonCreationState)
    
    # Add nodes
    builder.add_node("generate_complete_lessons", generate_complete_lessons)
    builder.add_node("generate_lesson_content_node", generate_lesson_content_node)
    builder.add_node("collect_lesson_results", collect_lesson_results)
    
    # Add edges
    builder.add_edge(START, "generate_complete_lessons")
    # generate_complete_lessons returns Send commands for parallel execution
    # LangGraph automatically handles the parallel execution and routing
    builder.add_edge("generate_lesson_content_node", "collect_lesson_results")
    builder.add_edge("collect_lesson_results", END)
    
    # Compile the graph
    graph = builder.compile()
    return graph


async def run_parallel_lesson_generation_example():
    """
    Run an example of parallel lesson content generation.
    """
    
    logger.info("Starting parallel lesson generation example...")
    
    # Create the graph
    graph = create_parallel_lesson_generation_graph()
    
    # Mock initial state (replace with actual data)
    initial_state = {
        "pdf_content": "Sample SEBI regulatory content about mutual funds...",
        "page_numbers": [1, 2, 3],
        "session_id": "example_session",
        "chunk_id": "example_chunk",
        "existing_journeys": [],
        "existing_lessons": [],
        "existing_anchors": [],
        "processing_history": {},
        "content_analysis": {
            "key_concepts": ["Mutual Funds", "Risk Assessment", "Regulatory Compliance"],
            "sebi_themes": ["Investor Protection", "Market Regulation"],
            "learning_opportunities": ["Understanding fund types", "Risk evaluation"]
        },
        "existing_content_mappings": [],
        "integration_plan": {
            "action": "CREATE_NEW_LESSON",
            "rationale": "New content requires separate lessons"
        },
        "journey_creation_plan": None,
        "new_journeys": [],
        "lessons": [],
        "content_blocks": [],
        "anchors": [],
        "voice_scripts": [],
        "lesson_content_results": [],  # This will accumulate parallel results
        "new_lessons": [],
        "updated_lessons": [],
        "quality_metrics": {},
        "processing_flags": [],
        "validation_errors": [],
        "current_step": "initial",
        "requires_human_review": False,
        "retry_count": 0,
    }
    
    try:
        # Execute the graph
        logger.info("Executing parallel lesson generation graph...")
        result = await graph.ainvoke(initial_state)
        
        # Display results
        logger.info("Parallel lesson generation completed!")
        logger.info(f"Generated {len(result.get('lessons', []))} lessons")
        logger.info(f"Generated {len(result.get('content_blocks', []))} content blocks")
        logger.info(f"Generated {len(result.get('anchors', []))} anchors")
        logger.info(f"Generated {len(result.get('voice_scripts', []))} voice scripts")
        
        if result.get('validation_errors'):
            logger.warning(f"Validation errors: {result['validation_errors']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in parallel lesson generation: {str(e)}")
        raise


def demonstrate_send_api_pattern():
    """
    Demonstrate the Send API pattern used in the implementation.
    """
    
    logger.info("\n=== Send API Pattern Demonstration ===")
    logger.info("1. generate_complete_lessons() creates Send commands:")
    logger.info("   - Each lesson gets its own Send(node_name, lesson_state)")
    logger.info("   - LangGraph executes all Send commands in parallel")
    logger.info("   - Results accumulate in lesson_content_results via reducer")
    
    logger.info("\n2. generate_lesson_content_node() processes individual lessons:")
    logger.info("   - Receives lesson_metadata via Send API")
    logger.info("   - Generates content for that specific lesson")
    logger.info("   - Returns results to be accumulated by reducer")
    
    logger.info("\n3. collect_lesson_results() aggregates all results:")
    logger.info("   - Collects results from lesson_content_results")
    logger.info("   - Validates and combines all generated content")
    logger.info("   - Updates final state with complete lessons")
    
    logger.info("\n=== Key Benefits ===")
    logger.info("✓ Parallel execution for faster processing")
    logger.info("✓ Automatic result aggregation via reducers")
    logger.info("✓ Error handling for individual lessons")
    logger.info("✓ Scalable to any number of lessons")


if __name__ == "__main__":
    # Demonstrate the Send API pattern
    demonstrate_send_api_pattern()
    
    # Run the example (commented out since it requires actual implementations)
    # asyncio.run(run_parallel_lesson_generation_example())
    
    print("\n" + "="*60)
    print("PARALLEL LESSON GENERATION IMPLEMENTATION COMPLETE")
    print("="*60)
    print("\nKey files updated:")
    print("1. generate_complete_lessons.py - Map phase with Send API")
    print("2. generate_lesson_content.py - Worker node for parallel execution")
    print("3. state.py - Added reducer for parallel result accumulation")
    print("4. This example file - Demonstrates usage pattern")
    print("\nThe implementation follows LangGraph best practices:")
    print("- Uses Send API for dynamic parallel task creation")
    print("- Implements proper state reducers for concurrent updates")
    print("- Handles errors gracefully in parallel execution")
    print("- Provides clear separation between map and reduce phases")
