"""Content generation prompt templates"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System message for content generation
CONTENT_GENERATION_SYSTEM_MESSAGE = """You are an expert educational content creator specializing in SEBI investor education materials. Generate clear, engaging, and pedagogically effective content that helps Indian investors understand complex financial concepts.

Ensure all content:
- Uses plain language accessible to retail investors
- Includes relevant Indian market examples
- Maintains accuracy with SEBI guidelines
- Provides practical, actionable insights
- Uses appropriate disclaimers and risk warnings
- Supports learning objectives effectively"""

CONTENT_GENERATION_PROMPTS = {
    "concept_block": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(CONTENT_GENERATION_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create a concept block for the given topic in SEBI investor education context.
        
        Lesson: {lesson_title}
        Concept: {concept}
        Learning Context: {learning_objectives}
        
        Content Analysis:
        Key Themes: {sebi_themes}
        Complexity Level: {complexity_level}
        
        Generate a concept block that:
        1. Explains the concept clearly and concisely
        2. Includes SEBI regulatory context
        3. Provides practical Indian market examples
        4. Uses accessible language for retail investors
        5. Connects to broader lesson themes
        6. Includes appropriate risk warnings
        
        {format_instructions}
        """)
    ]),
    
    "example_block": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(CONTENT_GENERATION_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create a practical example block for SEBI investor education.
        
        Lesson: {lesson_title}
        Themes to Illustrate: {sebi_themes}
        Context: {content_sample}
        
        Create an example that:
        1. Uses real Indian market scenarios
        2. Shows practical application of concepts
        3. References SEBI guidelines appropriately
        4. Includes both correct and incorrect approaches
        5. Helps investors make better decisions
        6. Uses step-by-step illustrations
        
        {format_instructions}
        """)
    ])
}
