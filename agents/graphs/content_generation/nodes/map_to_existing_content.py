from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import ExistingContentMappings
from agents.graphs.content_generation.llm import llm
from agents.graphs.content_generation.prompts.map_to_existing_content import (
    MAP_TO_EXISTING_CONTENT_PROMPTS,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


async def map_to_existing_content(state: LessonCreationState) -> LessonCreationState:
    """Map new content to existing lessons and identify integration opportunities"""

    content_analysis = state["content_analysis"]
    # existing_lessons = state["existing_lessons"]
    journeys_summary = state["existing_journeys"]

    # if len(existing_lessons) == 0:
    #     state["existing_content_mappings"] = []
    #     state["current_step"] = "content_mapped"
    #     return state
    if journeys_summary == "":
        state["existing_content_mappings"] = []
        state["current_step"] = "content_mapped"
        return state

    # mapping_prompt = MAP_TO_EXISTING_CONTENT_PROMPTS["content_mapping"]

    # Prepare existing lessons summary with journey context
    # lessons_summary = "\n".join(
    #     [
    #         f"- **Lesson slug: {lesson.slug}** **Lesson Title: {lesson.title}** (Journey id: {lesson.journey_id}, Topics: {', '.join(lesson.learning_objectives)})"
    #         for lesson in existing_lessons
    #     ]
    # )

    # messages = mapping_prompt.format_messages(
    #     key_concepts="\n".join(content_analysis.key_concepts),
    #     sebi_themes="\n".join(content_analysis.sebi_themes),
    #     learning_opportunities="\n".join(content_analysis.learning_opportunities),
    #     existing_journeys_summary=journeys_summary,
    #     # existing_lessons_summary=lessons_summary,
    # )

    # # Fix: Use the model class directly, not List[Model]
    # mappings = await llm.ainvoke_with_structured_output(
    #     messages, ExistingContentMappings
    # )

    # # Ensure mappings is always a list
    # if not isinstance(mappings, list):
    #     mappings = [mappings] if mappings else []

    # state["existing_content_mappings"] = mappings
    state["existing_content_mappings"] = []
    state["current_step"] = "content_mapped"
    return state
