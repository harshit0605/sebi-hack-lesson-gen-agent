"""Lesson creation prompt templates"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# System message for lesson creation
LESSON_CREATION_SYSTEM_MESSAGE = """You are an expert instructional designer specialized in creating SEBI-compliant financial education content. Your role is to design engaging, pedagogically sound lessons that help Indian investors make informed decisions.

Your lessons should:
- Follow adult learning principles
- Include practical Indian market examples
- Maintain SEBI regulatory compliance
- Use clear, accessible language
- Progress logically from basic to advanced concepts
- Include interactive elements and assessments
- Reference appropriate SEBI sources and guidelines"""

LESSON_CREATION_PROMPTS = {
    "create_new_lesson": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(LESSON_CREATION_SYSTEM_MESSAGE),
            HumanMessagePromptTemplate.from_template("""
        Create a comprehensive educational lesson from the structured content distribution for SEBI investor education.
        
        Lesson Details:
        Title: {lesson_title}
        Concepts to Cover: {concepts_to_cover}
        Learning Objectives: {learning_objectives}
        Estimated Duration: {estimated_duration} minutes
        Prerequisite Concepts: {prerequisite_concepts}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Pages: {page_numbers}
        Integration Rationale: {integration_rationale}
        
        Create a lesson that:
        1. Uses the provided learning objectives as-is
        2. Covers all specified concepts comprehensively
        3. Leverages key concepts and SEBI themes from the raw content
        4. Includes practical Indian market examples
        5. Maintains SEBI regulatory compliance
        6. Progresses logically from basic to advanced concepts
        7. Respects the estimated duration
        8. Builds on prerequisite concepts appropriately
        
        """),
        ]
    ),
    "extend_lesson": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(LESSON_CREATION_SYSTEM_MESSAGE),
            HumanMessagePromptTemplate.from_template("""
        Extend an existing lesson with new content while maintaining educational coherence.
        
        Existing Lesson: {lesson_title}
        Current Objectives: {current_objectives}
        Current Duration: {current_duration} minutes
        
        New Content to Integrate:
        Concepts to Cover: {concepts_to_cover}
        Learning Objectives: {learning_objectives}
        Estimated Duration: {estimated_duration} minutes
        Prerequisite Concepts: {prerequisite_concepts}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Pages: {page_numbers}
        Integration Rationale: {integration_rationale}
        
        Extension Strategy:
        1. Build upon existing foundation
        2. Add complementary concepts seamlessly  
        3. Leverage key concepts and SEBI themes from raw content
        4. Maintain learning progression
        5. Merge objectives intelligently
        6. Keep total duration reasonable
        
        """),
        ]
    ),
    "create_lesson": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(LESSON_CREATION_SYSTEM_MESSAGE),
            HumanMessagePromptTemplate.from_template("""
        Create a focused lesson from the provided structured data.
        
        Lesson Details:
        Title: {title}
        Concepts: {concepts}
        Learning Objectives: {learning_objectives}
        Integration Type: {integration_type}
        Estimated Duration: {estimated_duration} minutes
        Prerequisite Concepts: {prerequisite_concepts}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        
        Create a well-structured lesson that leverages the raw content analysis context while following the structured lesson details.
        
        """),
        ]
    ),
}
