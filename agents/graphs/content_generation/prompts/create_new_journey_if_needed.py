"""
Prompts for creating new learning journeys.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

CREATE_NEW_JOURNEY_SYSTEM_MESSAGE = """You are an expert instructional designer specialized in creating SEBI-compliant financial education content. Generate specific, measurable learning outcomes that align with SEBI investor education goals.

Focus on creating learning outcomes that are:
- Specific and measurable
- Aligned with SEBI regulatory requirements
- Appropriate for the target audience
- Practical and actionable for investors"""

CREATE_NEW_JOURNEY_PROMPTS = {
    "learning_outcomes": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(CREATE_NEW_JOURNEY_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate specific, measurable learning outcomes for this new learning journey based on the detailed lesson structure.
        
        Journey Plan:
        Title: {title}
        Description: {description}
        Key Topics: {key_topics}
        Target Audience: {target_audience}
        Total Lessons: {total_lessons}
        
        Lesson Objectives (from lesson structure):
        {lesson_objectives}
        
        Create 4-6 high-level learning outcomes that:
        1. Synthesize the detailed lesson objectives into broader journey outcomes
        2. Are specific and measurable
        3. Aligned with SEBI investor education goals
        4. Appropriate for the target audience
        5. Cover the key topics comprehensively
        6. Focus on practical application and understanding
        
        Each outcome should start with action verbs like "Understand", "Explain", "Apply", "Analyze", "Evaluate"
        
        """)
    ])
}
