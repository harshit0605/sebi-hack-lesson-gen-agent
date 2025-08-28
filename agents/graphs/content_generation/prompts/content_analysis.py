"""Content analysis prompt templates"""

from langchain_core.prompts import PromptTemplate

CONTENT_ANALYSIS_PROMPTS = {
    "analyze_content": PromptTemplate(
        template="""
        Analyze the following educational content and extract key information for SEBI investor education.
        
        Content: {content}
        Content Type: {content_type}
        Pages: {page_numbers}
        
        Extract:
        1. Key financial concepts and definitions
        2. SEBI-related themes and regulations
        3. Learning opportunities for investors
        4. Complexity level assessment
        5. Potential lesson topics
        6. Risk-related content
        7. Practical applications for Indian markets
        
        Focus on content that helps retail investors make informed decisions.
        
        """,
        input_variables=["content", "content_type", "page_numbers"],
    )
}
