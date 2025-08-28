"""
Quiz block generators for merged lessons.
"""

from typing import Optional, List
import logging
from langchain_core.output_parsers import PydanticOutputParser

from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.llm import (
    content_generator_llm,
    get_system_message,
    TaskType,
)
from agents.graphs.content_generation.prompts import QUIZ_GENERATION_PROMPTS
from agents.graphs.content_generation.models import (
    LessonModel,
    ContentBlockModel,
    ContentBlockMetadata,
    BlockType,
    ContentAnalysisResult,
    LessonContentDistribution,
    JourneyCreationPlan,
)


async def generate_merged_quiz_block(
    merged_lesson: LessonModel,
    original_lesson: LessonModel,
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> Optional[ContentBlockModel]:
    """Generate a quiz block that tests understanding of merged content"""

    parser = PydanticOutputParser(pydantic_object=ContentBlockModel)

    # Use centralized quiz generation prompts
    merged_quiz_prompt = QUIZ_GENERATION_PROMPTS["merged_lesson_quiz"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    try:
        prompt = merged_quiz_prompt.format(
            original_title=original_lesson.title,
            original_objectives=original_lesson.learning_objectives[:3],
            merged_title=merged_lesson.title,
            merged_objectives=merged_lesson.learning_objectives,
            key_concepts=content_analysis.key_concepts,
            sebi_themes=content_analysis.sebi_themes,
        )

        response = await content_generator_llm.ainvoke(prompt)
        quiz_block = parser.parse(response.content)

        # Configure quiz block for merged lesson
        quiz_block.lesson_id = merged_lesson.slug
        quiz_block.type = BlockType.QUIZ
        quiz_block.order = 900  # Place near end of lesson

        # Enhance quiz items with merge-specific rationales
        if hasattr(quiz_block.payload, "items"):
            for item in quiz_block.payload.items:
                from ..utils.content_enhancers import enhance_quiz_rationale_for_merge

                item.rationale = enhance_quiz_rationale_for_merge(
                    item.rationale, original_lesson.title, merged_lesson.title
                )

        # Set metadata
        quiz_block.metadata = ContentBlockMetadata(
            source_text=f"Merged quiz covering: {original_lesson.title} + new content",
            generation_confidence=0.85,
            integration_notes="Quiz tests integrated understanding of merged content",
        )

        logging.info(f"Generated merged quiz block for lesson: {merged_lesson.title}")
        return quiz_block

    except Exception as e:
        logging.error(f"Failed to generate merged quiz block: {str(e)}")
        return None


async def generate_quiz_block_from_distribution(
    lesson: LessonModel,
    lesson_dist: LessonContentDistribution,
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
) -> Optional[ContentBlockModel]:
    """Generate quiz block using structured lesson distribution data"""

    parser = PydanticOutputParser(pydantic_object=ContentBlockModel)

    # Use centralized quiz generation prompts with structured data
    distribution_quiz_prompt = QUIZ_GENERATION_PROMPTS["merged_lesson_quiz"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    # Build journey context for better alignment
    journey_context = ""
    if journey_plan:
        journey_context = f"""
Journey Context:
- Journey id: {journey_plan.slug}
- Journey Title: {journey_plan.title}
- Level: {journey_plan.level}
- Target Audience: {journey_plan.target_audience}
- Key Topics: {", ".join(journey_plan.key_topics)}
- Prerequisites: {", ".join(journey_plan.prerequisites)}
"""

    try:
        prompt = distribution_quiz_prompt.format(
            lesson_title=lesson_dist.lesson_title,
            concepts_to_cover=lesson_dist.concepts_to_cover,
            learning_objectives=lesson_dist.learning_objectives,
            integration_type=lesson_dist.integration_type,
            estimated_duration=lesson_dist.estimated_duration_minutes,
            prerequisite_concepts=lesson_dist.prerequisite_concepts,
            journey_context=journey_context,
        )

        response = await content_generator_llm.ainvoke(prompt)
        quiz_block = parser.parse(response.content)

        # Configure quiz block for distribution-based lesson
        quiz_block.lesson_id = lesson.slug
        quiz_block.type = BlockType.QUIZ
        quiz_block.order = 900  # Place near end of lesson

        # Enhance quiz items with distribution-specific context
        if hasattr(quiz_block.payload, "items"):
            for i, item in enumerate(quiz_block.payload.items):
                from ..utils.content_enhancers import enhance_quiz_with_objectives

                target_objective = lesson_dist.learning_objectives[
                    min(i, len(lesson_dist.learning_objectives) - 1)
                ]
                item.rationale = enhance_quiz_with_objectives(
                    item.rationale, target_objective, lesson_dist.integration_type
                )

        # Set metadata with distribution information
        quiz_block.metadata = ContentBlockMetadata(
            source_text=f"Distribution quiz for {lesson_dist.integration_type}",
            generation_confidence=0.9,
            integration_notes=f"Quiz aligned with {len(lesson_dist.learning_objectives)} learning objectives",
        )

        logging.info(
            f"Generated distribution quiz block for lesson: {lesson_dist.lesson_title}"
        )
        return quiz_block

    except Exception as e:
        logging.error(f"Failed to generate distribution quiz block: {str(e)}")
        return None


async def generate_objective_assessment_quiz(
    lesson: LessonModel,
    learning_objectives: List[str],
    target_concepts: List[str],
    journey_plan: Optional[JourneyCreationPlan],
    state: LessonCreationState,
) -> Optional[ContentBlockModel]:
    """Generate quiz block specifically designed to assess learning objectives"""

    parser = PydanticOutputParser(pydantic_object=ContentBlockModel)

    # Use centralized quiz generation prompts
    objective_quiz_prompt = QUIZ_GENERATION_PROMPTS["merged_lesson_quiz"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    # Build journey context
    journey_context = ""
    if journey_plan:
        journey_context = f"""
Journey Context:
- Journey id: {journey_plan.slug}
- Journey Title: {journey_plan.title}
- Level: {journey_plan.level}
- Target Audience: {journey_plan.target_audience}
"""

    try:
        prompt = objective_quiz_prompt.format(
            lesson_title=lesson.title,
            learning_objectives=learning_objectives,
            target_concepts=target_concepts,
            journey_context=journey_context,
        )

        response = await content_generator_llm.ainvoke(prompt)
        quiz_block = parser.parse(response.content)

        # Configure quiz block for objective assessment
        quiz_block.lesson_id = lesson.slug
        quiz_block.type = BlockType.QUIZ
        quiz_block.order = 950  # Place at end for assessment

        # Enhance quiz items with objective-specific rationales
        if hasattr(quiz_block.payload, "items"):
            for i, item in enumerate(quiz_block.payload.items):
                from ..utils.content_enhancers import enhance_assessment_rationale

                target_objective = learning_objectives[
                    min(i, len(learning_objectives) - 1)
                ]
                item.rationale = enhance_assessment_rationale(
                    item.rationale, target_objective
                )

        # Set metadata for objective assessment
        quiz_block.metadata = ContentBlockMetadata(
            source_text=f"Objective assessment quiz for {len(learning_objectives)} objectives",
            generation_confidence=0.85,
            integration_notes="Quiz designed to assess specific learning objective achievement",
        )

        logging.info(f"Generated objective assessment quiz for lesson: {lesson.title}")
        return quiz_block

    except Exception as e:
        logging.error(f"Failed to generate objective assessment quiz: {str(e)}")
        return None
