"""
Prompts for journey fit evaluation node.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

EVALUATE_JOURNEY_FIT_SYSTEM_MESSAGE = """You are an expert curriculum designer specializing in learning pathway optimization and educational content organization. Your role is to create new learning journeys based on detailed lesson structure analysis.

You will receive a detailed content distribution that specifies exactly what lessons should be created, their content, duration, and learning objectives. Use this information to design a coherent learning journey that organizes these lessons in optimal pedagogical sequence.

Consider learning objectives, content coherence, complexity progression, prerequisite relationships, and total learning time when designing the journey structure."""

EVALUATE_JOURNEY_FIT_PROMPTS = {
    "journey_creation_planning": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                EVALUATE_JOURNEY_FIT_SYSTEM_MESSAGE
            ),
            HumanMessagePromptTemplate.from_template("""
        Create a plan for a new learning journey based on the detailed lesson structure from content integration analysis.
        
        Content Distribution (Lesson Structure): {content_distribution}
        
        Existing Journeys: {existing_journeys}
        
        Integration Rationale: {integration_rationale}
        
        Design a new journey that:
        1. Has a unique slug different from existing journeys
        2. Leverages the detailed lesson structure provided in content_distribution
        3. Organizes lessons in logical learning progression
        4. Doesn't overlap significantly with existing journeys
        5. Follows SEBI educational guidelines
        6. Has appropriate scope and complexity based on lesson estimates
        7. Calculates total duration from individual lesson estimates
        8. Identifies prerequisites from lesson dependency analysis
        
        Use the lesson structure to inform:
        - Journey title and description
        - Total estimated hours (sum of lesson durations)
        - Learning progression and dependencies
        - Target audience based on lesson complexity
        - Key topics from lesson concepts
        - Learning outcomes (synthesize from lesson objectives into 4-6 high-level outcomes)
        
        """),
        ]
    )
}
