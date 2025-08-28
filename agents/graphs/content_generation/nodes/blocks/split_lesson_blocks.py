"""
Block generators specifically for split lessons.
"""

from typing import List, Dict, Any
from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import (
    LessonModel,
    ContentAnalysisResult,
    ContentBlockModel,
    ContentBlockMetadata,
    BlockType,
    ConceptBlockPayload,
    ExampleBlockPayload,
    ReflectionBlockPayload,
)


async def generate_blocks_for_split_lesson(
    lesson: LessonModel,
    group: Dict[str, Any],
    content_analysis: ContentAnalysisResult,
    state: LessonCreationState,
) -> List[ContentBlockModel]:
    """Generate content blocks specifically for a split lesson"""

    blocks = []

    # Generate concept blocks for group concepts
    group_concepts = group.get("concepts", [])
    for i, concept in enumerate(group_concepts):
        concept_block = ContentBlockModel(
            lesson_id=lesson.slug,
            type=BlockType.CONCEPT,
            order=i + 1,
            payload=ConceptBlockPayload(
                heading=concept.title(),
                rich_text_md=f"## {concept.title()}\n\nDetailed explanation of {concept} in the context of SEBI guidelines and investor education.",
                sebi_context=f"This concept relates to SEBI's investor education initiatives and {group.get('focus_area', 'financial literacy')}.",
            ),
            anchor_ids=[],
            metadata=ContentBlockMetadata(
                source_text=f"Generated from split content group focusing on {concept}",
                generation_confidence=0.8,
            ),
        )
        blocks.append(concept_block)

    # Generate example block for group themes
    group_themes = group.get("themes", [])
    if group_themes:
        example_block = ContentBlockModel(
            lesson_id=lesson.slug,
            type=BlockType.EXAMPLE,
            order=len(blocks) + 1,
            payload=ExampleBlockPayload(
                scenario_title=f"Practical Application: {group.get('focus_area', 'Concept Application')}",
                scenario_md=f"Real-world scenario demonstrating {', '.join(group_themes[:2])} in Indian financial markets.",
                qa_pairs=[
                    {
                        "question": f"How does {group_themes[0]} apply in practice?",
                        "answer": "Detailed practical explanation with Indian market context.",
                    }
                ],
                indian_context=True,
            ),
            anchor_ids=[],
            metadata=ContentBlockMetadata(
                source_text=f"Generated example for themes: {', '.join(group_themes)}",
                generation_confidence=0.7,
            ),
        )
        blocks.append(example_block)

    # Generate reflection block for lesson synthesis
    reflection_block = ContentBlockModel(
        lesson_id=lesson.slug,
        type=BlockType.REFLECTION,
        order=len(blocks) + 1,
        payload=ReflectionBlockPayload(
            prompt_md=f"Reflect on how the concepts in this lesson ({lesson.title}) connect to your understanding of {group.get('focus_area', 'financial markets')}.",
            guidance_md="Consider practical applications and how these concepts help in making informed investment decisions.",
            min_chars=150,
            reflection_type="application",
            sample_responses=[
                f"The concepts help me understand {group.get('focus_area', 'the topic')} better by...",
                f"I can apply this knowledge when making decisions about...",
            ],
        ),
        anchor_ids=[],
        metadata=ContentBlockMetadata(
            source_text=f"Generated reflection for lesson focus: {group.get('focus_area')}",
            generation_confidence=0.8,
        ),
    )
    blocks.append(reflection_block)

    return blocks
