"""Quiz generation prompt templates"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System message for quiz generation
QUIZ_GENERATION_SYSTEM_MESSAGE = """You are an expert assessment designer for financial education. Create fair, comprehensive, and educationally valuable questions that test understanding of SEBI-related concepts and investor education topics.

Your assessments should:
- Test application, not just memorization
- Include scenario-based questions
- Provide clear, educational rationales
- Use appropriate difficulty levels
- Reference SEBI guidelines in explanations
- Help reinforce key learning objectives"""

QUIZ_GENERATION_PROMPTS = {
    "generate_quiz": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(QUIZ_GENERATION_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create a comprehensive quiz for the given lesson content.
        
        Lesson: {lesson_title}
        Learning Objectives: {learning_objectives}
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        
        Create a quiz that:
        1. Tests understanding of key concepts (6-8 questions)
        2. Includes scenario-based questions
        3. Uses varied question types (MCQ, true/false)
        4. Tests practical application, not memorization
        5. Provides clear educational rationales
        6. References SEBI guidelines in explanations
        7. Uses appropriate difficulty levels
        
        Focus on helping investors make informed decisions.
        
        {format_instructions}
        """)
    ]),
    
    "merged_lesson_quiz": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(QUIZ_GENERATION_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create a comprehensive quiz block for a merged lesson that tests both original and new content.
        
        Original Lesson: {original_title}
        Original Objectives: {original_objectives}
        
        Merged Lesson: {merged_title}
        Merged Objectives: {merged_objectives}
        
        New Content Analysis:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        
        Create a quiz that:
        1. Tests understanding of both original and new concepts
        2. Includes 6-8 questions covering merged content comprehensively
        3. Has questions that require integration of old and new knowledge
        4. Includes SEBI-specific scenarios and applications
        5. Uses varied question types (MCQ, true/false, scenario-based)
        6. Provides clear rationales linking back to SEBI guidelines
        7. Tests practical application, not just memorization
        
        Ensure the quiz feels cohesive and tests the unified lesson content.
        
        {format_instructions}
        """)
    ])
}
