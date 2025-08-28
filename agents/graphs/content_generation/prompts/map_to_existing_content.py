"""
Prompts for content mapping node.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

MAP_TO_EXISTING_CONTENT_SYSTEM_MESSAGE = """You are an expert educational content analyst specializing in content relationship analysis and integration planning. Your role is to identify how new educational content relates to existing lessons and determine optimal integration opportunities.

Focus on semantic similarities, conceptual overlaps, and pedagogical coherence when mapping content relationships."""

MAP_TO_EXISTING_CONTENT_PROMPTS = {
    "content_mapping": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                MAP_TO_EXISTING_CONTENT_SYSTEM_MESSAGE
            ),
            HumanMessagePromptTemplate.from_template("""
        Analyze how this new content relates to existing lessons and identify integration opportunities.
        
        New Content Analysis:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        Existing Learning Journeys:
        {existing_journeys_summary}
        
        For each existing lesson that could be related, provide:
        1. Similarity score (0.0-1.0)
        2. Overlapping concepts
        3. Integration potential (high/medium/low)
        4. Consider the journey context and level when determining integration potential
        
        """),
        ]
    )
}
