"""
Reflection block generators for integration scenarios.
"""

from typing import Optional
import logging
from langchain_core.output_parsers import PydanticOutputParser

from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.llm import (
    content_generator_llm,
    get_system_message,
    TaskType,
)
from agents.graphs.content_generation.prompts import CONTENT_GENERATION_PROMPTS
from agents.graphs.content_generation.models import (
    LessonModel,
    ContentBlockModel,
    ContentBlockMetadata,
    BlockType,
)


async def generate_integration_reflection_block(
    extended_lesson: LessonModel,
    original_lesson: LessonModel,
    state: LessonCreationState,
) -> Optional[ContentBlockModel]:
    """Generate a reflection block that helps students integrate new and existing content"""

    parser = PydanticOutputParser(pydantic_object=ContentBlockModel)

    # Use centralized content generation prompts - reflection type
    integration_reflection_prompt = CONTENT_GENERATION_PROMPTS["concept_block"].partial(
        format_instructions=parser.get_format_instructions(),
        system_message=get_system_message(TaskType.CONTENT_GENERATION),
    )

    try:
        # Extract integration context
        integration_notes = (
            extended_lesson.metadata.integration_notes or "General content extension"
        )

        prompt = integration_reflection_prompt.format(
            original_title=original_lesson.title,
            original_objectives=original_lesson.learning_objectives[:3],
            extended_title=extended_lesson.title,
            extended_objectives=extended_lesson.learning_objectives[:5],
            page_numbers=state["page_numbers"],
            integration_notes=integration_notes,
        )

        response = await content_generator_llm.ainvoke(prompt)
        reflection_block = parser.parse(response.content)

        # Ensure proper block configuration
        reflection_block.lesson_id = extended_lesson.slug
        reflection_block.type = BlockType.REFLECTION
        reflection_block.order = 999  # Place at end of lesson

        # Enhance reflection prompt if needed
        if hasattr(reflection_block.payload, "prompt_md"):
            from ..utils.content_enhancers import enhance_integration_prompt

            reflection_block.payload.prompt_md = enhance_integration_prompt(
                reflection_block.payload.prompt_md,
                original_lesson.title,
                extended_lesson.title,
            )

        # Set metadata
        reflection_block.metadata = ContentBlockMetadata(
            source_text=f"Integration reflection for extended lesson: {extended_lesson.title}",
            generation_confidence=0.9,
            integration_notes="Synthesizes original and extended content",
        )

        logging.info(
            f"Generated integration reflection block for lesson: {extended_lesson.title}"
        )
        return reflection_block

    except Exception as e:
        logging.error(f"Failed to generate integration reflection block: {str(e)}")
        return None
