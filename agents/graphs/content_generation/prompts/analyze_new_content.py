"""
Prompts for content analysis node.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

ANALYZE_CONTENT_SYSTEM_MESSAGE = """You are an expert educational content analyst specializing in SEBI (Securities and Exchange Board of India) investor education materials. Your role is to analyze PDF content and extract key concepts, learning opportunities, and educational themes that align with SEBI's investor protection and education mandates.

Focus on identifying:
- Key financial concepts and regulatory information
- Learning opportunities for Indian investors
- Risk management and compliance themes
- Market structure and investment principles
- Practical applications and real-world scenarios"""

ANALYZE_CONTENT_PROMPTS = {
    "content_analysis": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(ANALYZE_CONTENT_SYSTEM_MESSAGE),
            HumanMessagePromptTemplate.from_template("""
        Analyze this SEBI document content chunk and provide a comprehensive analysis for educational lesson creation.
        
        PDF Content (Pages {page_numbers}):
        {pdf_content}
        
        Chunk ID: {chunk_id}
        
        Analyze and extract:
        1. Key financial concepts and definitions (be specific and comprehensive)
        2. SEBI-specific themes, regulations, and guidelines mentioned
        3. Learning opportunities and learning moments
        4. Complexity level of the content (beginner/intermediate/advanced)
        5. Estimate how many focused lessons this content could create
        6. Content type classification
        
        Focus on:
        - Investor education themes
        - Risk management concepts  
        - Market structure information
        - Regulatory compliance aspects
        - Learning opportunities
        - Practical examples and scenarios
        - Technical definitions that need explanation
        
        Be thorough but concise. Each concept should be actionable for lesson creation.
        
        """),
        ]
    )
}
