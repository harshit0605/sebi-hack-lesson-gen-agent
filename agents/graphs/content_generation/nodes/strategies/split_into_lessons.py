"""
Strategy for splitting dense content into multiple focused lessons.
"""

from typing import List, Dict, Any, Optional
import logging

from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    LessonModel,
    LessonCreationModel,
    ContentAnalysisResult,
    ContentIntegrationPlan,
    ContentIntegrationAction,
    LessonMetadata,
    LessonContentDistribution,
    JourneyCreationPlan,
    ContentBlockModel,
)
from agents.graphs.content_generation.llm import (
    lesson_creator_llm,
    get_system_message,
    TaskType,
)
from agents.graphs.content_generation.prompts import LESSON_CREATION_PROMPTS

# Import the block generator that will be created
from agents.graphs.content_generation.nodes.blocks.split_lesson_blocks import (
    generate_blocks_for_split_lesson as _generate_blocks,
)


async def split_into_multiple_lessons(
    state: LessonCreationState,
) -> LessonCreationState:
    """Split dense content into multiple focused lessons using structured content distribution"""

    integration_plan = state["integration_plan"]
    journey_plan = state.get("journey_creation_plan")
    existing_journeys = state.get("existing_journeys", [])

    try:
        # Use structured content distribution for lesson creation (eliminates redundant LLM calls)
        if integration_plan.content_distribution:
            # Filter for split-type distributions only
            split_distributions = [
                dist
                for dist in integration_plan.content_distribution
                if dist.integration_type == "split_lesson"
            ]

            if split_distributions:
                split_lessons = await create_lessons_from_distribution(
                    split_distributions,
                    integration_plan,
                    journey_plan,
                    state,
                )
            else:
                logging.warning("No split lesson distributions found")
                split_lessons = []
        else:
            # Fallback only if absolutely no content distribution exists
            logging.warning("No content distribution available - using legacy approach")
            content_analysis = state["content_analysis"]
            split_strategy = await determine_split_strategy(
                content_analysis, integration_plan
            )
            split_lessons = await create_split_lessons(
                split_strategy, content_analysis, integration_plan, state
            )

        if split_lessons:
            state["new_lessons"] = split_lessons
            logging.info(f"Split content into {len(split_lessons)} lessons")
        else:
            state["validation_errors"].append(
                "Failed to split content into multiple lessons"
            )

    except Exception as e:
        state["validation_errors"].append(f"Failed to split content: {str(e)}")

    return state


async def create_lessons_from_distribution(
    content_distribution: List[LessonContentDistribution],
    integration_plan: ContentIntegrationPlan,
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
) -> List[LessonModel]:
    """Create multiple lessons from structured content distribution"""

    lessons = []

    for i, lesson_dist in enumerate(content_distribution):
        try:
            lesson = await create_lesson_from_distribution(
                lesson_dist, integration_plan, journey_plan, state, i
            )

            if lesson:
                lessons.append(lesson)

        except Exception as e:
            logging.error(f"Failed to create lesson from distribution {i}: {str(e)}")
            continue

    # Set up lesson dependencies based on distribution prerequisites
    lessons = setup_distribution_dependencies(lessons, content_distribution)

    return lessons


async def create_lesson_from_distribution(
    lesson_dist: LessonContentDistribution,
    integration_plan: ContentIntegrationPlan,
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
    lesson_index: int,
) -> Optional[LessonModel]:
    """Create a single lesson from content distribution"""

    # Use centralized lesson creation prompts with journey context
    lesson_prompt = LESSON_CREATION_PROMPTS["create_lesson"].partial(
        system_message=get_system_message(TaskType.LESSON_CREATION),
    )

    # Build journey context for better alignment
    journey_context = ""
    if journey_plan:
        journey_context = f"""
Journey Context:
- Journey id: {journey_plan.slug}
- Journey Title: {journey_plan.title}
- Target Audience: {journey_plan.target_audience}
- Total Duration: {journey_plan.total_estimated_hours} hours
- Key Topics: {", ".join(journey_plan.key_topics)}
- Prerequisites: {", ".join(journey_plan.prerequisites)}
"""

    # Get content analysis for raw content context
    content_analysis = state.get("content_analysis")

    prompt = lesson_prompt.format_messages(
        title=lesson_dist.lesson_title,
        concepts=lesson_dist.concepts_to_cover,
        learning_objectives=lesson_dist.learning_objectives,
        integration_type=lesson_dist.integration_type,
        estimated_duration=lesson_dist.estimated_duration_minutes,
        prerequisite_concepts=lesson_dist.prerequisite_concepts,
        key_concepts=content_analysis.key_concepts if content_analysis else [],
        sebi_themes=content_analysis.sebi_themes if content_analysis else [],
        learning_opportunities=content_analysis.learning_opportunities
        if content_analysis
        else [],
        journey_context=journey_context,
        integration_rationale=integration_plan.rationale,
    )

    try:
        lesson_creation = await lesson_creator_llm.ainvoke_with_structured_output(
            prompt, LessonCreationModel
        )

        # Convert to full LessonModel with empty content fields (will be populated later)
        lesson = LessonModel(
            **lesson_creation.model_dump(),
            blocks=[],  # Will be populated by generate_complete_lessons node
            anchors=[],  # Will be populated by generate_complete_lessons node
            voice_ready=False,
            voice_script_id=None,  # Will be set by generate_complete_lessons node
            quiz_ids=[],  # Will be populated by generate_complete_lessons node
            interactive_ids=[],  # Will be populated by generate_complete_lessons node
        )

        # Set metadata for split lesson from distribution
        lesson.metadata = LessonMetadata(
            source_pages=state["page_numbers"],
            chunk_id=state["chunk_id"],
            integration_action=ContentIntegrationAction.SPLIT_INTO_MULTIPLE,
            overlap_handled=True,
            review_status="generated",
        )

        # Generate content blocks directly from distribution data (no additional LLM calls)
        lesson_blocks = generate_blocks_from_distribution_data(
            lesson, lesson_dist, state
        )

        if lesson_blocks:
            lesson.blocks = [f"{lesson.slug}_{block.order}" for block in lesson_blocks]

            # Add blocks to state
            if "content_blocks" not in state:
                state["content_blocks"] = []
            state["content_blocks"].extend(lesson_blocks)

        return lesson

    except Exception as e:
        logging.error(f"Failed to create lesson from distribution: {str(e)}")
        return None


def setup_distribution_dependencies(
    lessons: List[LessonModel], content_distribution: List[LessonContentDistribution]
) -> List[LessonModel]:
    """Set up prerequisite relationships between lessons based on distribution data"""

    dist_lookup = {dist.lesson_title: dist for dist in content_distribution}

    for lesson in lessons:
        if lesson.title in dist_lookup:
            dist = dist_lookup[lesson.title]

            # Set up dependencies based on prerequisite concepts
            if dist.prerequisite_concepts:
                # Find lessons that cover prerequisite concepts
                prereq_lessons = []
                for other_lesson in lessons:
                    if other_lesson.title != lesson.title:
                        other_dist = dist_lookup.get(other_lesson.title)
                        if other_dist:
                            # Check if other lesson covers any prerequisite concepts
                            concept_overlap = set(dist.prerequisite_concepts) & set(
                                other_dist.concepts_to_cover
                            )
                            if concept_overlap:
                                prereq_lessons.append(other_lesson.slug)

                if prereq_lessons:
                    lesson.prerequisites = prereq_lessons

    return lessons


def generate_blocks_from_distribution_data(
    lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate content blocks directly from distribution data without LLM calls"""

    from agents.graphs.content_generation.models import (
        ContentBlockModel,
        ContentBlockMetadata,
        BlockType,
        ConceptBlockPayload,
        ExampleBlockPayload,
        ReflectionBlockPayload,
    )

    blocks = []

    # Generate concept blocks for each concept in distribution
    for i, concept in enumerate(lesson_dist.concepts_to_cover):
        concept_block = ContentBlockModel(
            lesson_id=lesson.slug,
            type=BlockType.CONCEPT,
            order=(i + 1) * 10,
            payload=ConceptBlockPayload(
                heading=concept.title()
                if hasattr(concept, "title")
                else str(concept).title(),
                rich_text_md=f"## {str(concept).title()}\n\nThis concept is essential for understanding {lesson_dist.integration_type} in the context of SEBI guidelines and investor education.",
                sebi_context=f"This concept relates to SEBI's investor education initiatives and supports the learning objective: {lesson_dist.learning_objectives[0] if lesson_dist.learning_objectives else 'financial literacy'}.",
            ),
            anchor_ids=[],
            metadata=ContentBlockMetadata(
                source_text=f"Generated from distribution concept: {concept}",
                generation_confidence=0.85,
                integration_notes=f"Part of {lesson_dist.integration_type} strategy",
            ),
        )
        blocks.append(concept_block)

    # Generate example block if we have learning objectives
    if lesson_dist.learning_objectives:
        example_block = ContentBlockModel(
            lesson_id=lesson.slug,
            type=BlockType.EXAMPLE,
            order=100,
            payload=ExampleBlockPayload(
                scenario_title=f"Practical Application: {lesson_dist.lesson_title}",
                scenario_md=f"Real-world scenario demonstrating {lesson_dist.learning_objectives[0]} in Indian financial markets.",
                qa_pairs=[
                    {
                        "question": f"How does this apply to {lesson_dist.concepts_to_cover[0] if lesson_dist.concepts_to_cover else 'investing'}?",
                        "answer": "This concept helps investors make informed decisions by providing clear guidelines and practical knowledge for Indian market conditions.",
                    }
                ],
                indian_context=True,
            ),
            anchor_ids=[],
            metadata=ContentBlockMetadata(
                source_text=f"Generated example for objectives: {', '.join(lesson_dist.learning_objectives[:2])}",
                generation_confidence=0.8,
                integration_notes=f"Supports {lesson_dist.integration_type} learning objectives",
            ),
        )
        blocks.append(example_block)

    # Generate reflection block for lesson synthesis
    reflection_block = ContentBlockModel(
        lesson_id=lesson.slug,
        type=BlockType.REFLECTION,
        order=200,
        payload=ReflectionBlockPayload(
            prompt_md=f"Reflect on how the concepts in this lesson ({lesson_dist.lesson_title}) connect to your understanding of {lesson_dist.integration_type}.",
            guidance_md="Consider practical applications and how these concepts help in making informed investment decisions according to SEBI guidelines.",
            min_chars=150,
            reflection_type="application",
            sample_responses=[
                f"The concepts help me understand {lesson_dist.integration_type} better by...",
                "I can apply this knowledge when making decisions about...",
            ],
        ),
        anchor_ids=[],
        metadata=ContentBlockMetadata(
            source_text=f"Generated reflection for lesson: {lesson_dist.lesson_title}",
            generation_confidence=0.8,
            integration_notes=f"Synthesis activity for {lesson_dist.integration_type}",
        ),
    )
    blocks.append(reflection_block)

    return blocks


async def generate_blocks_for_distribution_lesson(
    lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    integration_plan: ContentIntegrationPlan,
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
):
    """Generate content blocks for a lesson created from distribution"""

    # Convert distribution data to group format for compatibility with existing block generator
    group = {
        "title": lesson_dist.lesson_title,
        "concepts": lesson_dist.concepts_to_cover,
        "learning_objectives": lesson_dist.learning_objectives,
        "focus_area": lesson_dist.integration_type,
        "estimated_duration": lesson_dist.estimated_duration_minutes,
        "prerequisite_concepts": lesson_dist.prerequisite_concepts,
    }

    return await _generate_blocks(lesson, group, state.get("content_analysis"), state)


async def determine_split_strategy(
    content_analysis: ContentAnalysisResult, integration_plan: ContentIntegrationPlan
) -> Dict[str, Any]:
    """Determine how to optimally split the content into multiple lessons"""

    # Create a simple strategy prompt for splitting content
    strategy_prompt = f"""
    Based on the content analysis, determine how to split this content into multiple focused lessons.
    
    Content Analysis:
    - Key Concepts: {content_analysis.key_concepts}
    - SEBI Themes: {content_analysis.sebi_themes}
    - Learning Opportunities: {content_analysis.learning_opportunities}
    - Complexity Level: {content_analysis.complexity_level}
    - Estimated Lesson Count: {content_analysis.estimated_lesson_count}
    
    Integration Rationale: {integration_plan.rationale}
    
    Provide a JSON response with the following structure:
    {{
        "lesson_groups": [
            {{
                "title": "Lesson title",
                "concepts": ["concept1", "concept2"],
                "themes": ["theme1", "theme2"],
                "focus_area": "Main focus area",
                "estimated_duration": 25
            }}
        ],
        "dependencies": {{"lesson_title": ["prerequisite_lesson"]}},
        "progression_order": ["lesson1", "lesson2", "lesson3"]
    }}
    """

    response = await lesson_creator_llm.ainvoke(strategy_prompt)

    try:
        import json

        strategy = json.loads(response.content)
        return strategy
    except json.JSONDecodeError:
        logging.error("Failed to parse strategy JSON response")
        return {"lesson_groups": [], "dependencies": {}, "progression_order": []}


async def create_split_lessons(
    split_strategy: Dict[str, Any],
    content_analysis: ContentAnalysisResult,
    integration_plan: ContentIntegrationPlan,
    state: LessonCreationState,
) -> List[LessonModel]:
    """Create multiple lessons based on the split strategy"""

    lessons = []
    lesson_groups = split_strategy.get("lesson_groups", [])

    for i, group in enumerate(lesson_groups):
        try:
            lesson = await create_lesson_from_group(
                group, i, split_strategy, content_analysis, integration_plan, state
            )

            if lesson:
                lessons.append(lesson)

        except Exception as e:
            logging.error(f"Failed to create lesson from group {i}: {str(e)}")
            continue

    # Set up lesson dependencies based on strategy
    lessons = setup_lesson_dependencies(lessons, split_strategy)

    return lessons


async def create_lesson_from_group(
    group: Dict[str, Any],
    group_index: int,
    split_strategy: Dict[str, Any],
    content_analysis: ContentAnalysisResult,
    integration_plan: ContentIntegrationPlan,
    state: LessonCreationState,
) -> Optional[LessonModel]:
    """Create a single lesson from a content group"""

    # Use centralized lesson creation prompts
    lesson_prompt = LESSON_CREATION_PROMPTS["create_lesson"].partial(
        system_message=get_system_message(TaskType.LESSON_CREATION),
    )

    # Get content analysis for raw content context
    content_analysis = state.get("content_analysis")

    prompt = lesson_prompt.format_messages(
        title=group.get("title", f"Lesson {group_index + 1}"),
        concepts=group.get("concepts", []),
        learning_objectives=group.get("learning_objectives", []),
        integration_type="split_lesson",
        estimated_duration=group.get("estimated_duration", 20),
        prerequisite_concepts=group.get("prerequisite_concepts", []),
        key_concepts=content_analysis.key_concepts if content_analysis else [],
        sebi_themes=content_analysis.sebi_themes if content_analysis else [],
        learning_opportunities=content_analysis.learning_opportunities
        if content_analysis
        else [],
        journey_context="",
        integration_rationale=integration_plan.rationale,
    )

    try:
        lesson_creation = await lesson_creator_llm.ainvoke_with_structured_output(
            prompt, LessonCreationModel
        )

        # Convert to full LessonModel with empty content fields (will be populated later)
        lesson = LessonModel(
            **lesson_creation.model_dump(),
            blocks=[],  # Will be populated by generate_complete_lessons node
            anchors=[],  # Will be populated by generate_complete_lessons node
            voice_ready=False,
            voice_script_id=None,  # Will be set by generate_complete_lessons node
            quiz_ids=[],  # Will be populated by generate_complete_lessons node
            interactive_ids=[],  # Will be populated by generate_complete_lessons node
        )

        # Set metadata for split lesson
        lesson.metadata = LessonMetadata(
            source_pages=state["page_numbers"],
            chunk_id=state["chunk_id"],
            integration_action=ContentIntegrationAction.SPLIT_INTO_MULTIPLE,
            overlap_handled=True,
            review_status="generated",
        )

        return lesson

    except Exception as e:
        logging.error(f"Failed to create lesson from group: {str(e)}")
        return None


def setup_lesson_dependencies(
    lessons: List[LessonModel], split_strategy: Dict[str, Any]
) -> List[LessonModel]:
    """Set up prerequisite relationships between split lessons"""

    dependencies = split_strategy.get("dependencies", {})
    progression_order = split_strategy.get("progression_order", [])

    # Create lesson lookup by title
    lesson_lookup = {lesson.title: lesson for lesson in lessons}

    for lesson in lessons:
        # Set up dependencies based on strategy
        if lesson.title in dependencies:
            prereq_titles = dependencies[lesson.title]
            lesson.prerequisites = [
                lesson_lookup[title].slug
                for title in prereq_titles
                if title in lesson_lookup
            ]

        # Set up sequential dependencies based on progression order
        if lesson.title in progression_order:
            lesson_index = progression_order.index(lesson.title)
            if lesson_index > 0:
                previous_lesson_title = progression_order[lesson_index - 1]
                if previous_lesson_title in lesson_lookup:
                    if not lesson.prerequisites:
                        lesson.prerequisites = []
                    lesson.prerequisites.append(
                        lesson_lookup[previous_lesson_title].slug
                    )

        # Set lesson order based on progression
        if lesson.title in progression_order:
            lesson.order = progression_order.index(lesson.title) + 1

    return lessons


async def generate_blocks_for_split_lesson(
    lesson: LessonModel,
    group: Dict[str, Any],
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
):
    """Generate content blocks specifically for a split lesson"""

    return await _generate_blocks(lesson, group, content_analysis, state)
