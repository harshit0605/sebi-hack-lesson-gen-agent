"""
Prompts for validation error handling node.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

HANDLE_VALIDATION_ERRORS_SYSTEM_MESSAGE = """You are an expert educational content validator and error correction specialist for SEBI investor education materials. Your role is to identify and fix validation errors in generated content to ensure quality, compliance, and educational effectiveness.

Focus on maintaining content integrity while correcting structural, formatting, and validation issues."""

HANDLE_VALIDATION_ERRORS_PROMPTS = {
    "fix_content": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(HANDLE_VALIDATION_ERRORS_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Fix empty or insufficient content blocks based on the lesson context and PDF content.
        
        Error: {error}
        
        Lesson Context: {lesson_context}
        
        PDF Content Sample: {pdf_content_sample}
        
        Generate proper content blocks with:
        1. Meaningful content that addresses the learning objectives
        2. Appropriate SEBI context and examples
        3. Clear, educational explanations
        4. Proper payload structure for each block type
        
        {format_instructions}
        """)
    ]),
    
    "fix_learning_objectives": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(HANDLE_VALIDATION_ERRORS_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate 3-5 specific learning objectives for lessons based on the content analysis.
        
        Error: {error}
        Content Analysis: {content_analysis}
        
        Create objectives that:
        1. Start with action verbs (Understand, Explain, Apply, Analyze)
        2. Are specific and measurable
        3. Align with SEBI investor education goals
        4. Are appropriate for the lesson content
        
        {format_instructions}
        """)
    ])
}
