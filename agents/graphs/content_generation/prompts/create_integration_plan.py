"""
Prompts for integration planning node.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

CREATE_INTEGRATION_PLAN_SYSTEM_MESSAGE = """You are an expert curriculum designer specializing in educational content integration and learning pathway optimization. Your role is to determine how new educational content should be integrated with existing SEBI investor education materials.

Consider pedagogical principles, content coherence, and optimal learning progression when making integration decisions."""

CREATE_INTEGRATION_PLAN_PROMPTS = {
    "integration_planning": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                CREATE_INTEGRATION_PLAN_SYSTEM_MESSAGE
            ),
            HumanMessagePromptTemplate.from_template("""
        Create a detailed integration plan for new content based on the analysis and existing lesson mappings.
        
        Content Analysis: {content_analysis}
        
        Existing Lesson Mappings: {mappings}
        
        Existing Journeys: {existing_journeys}
        
        CRITICAL DECISION RULES:
        
        1. **If mappings are EMPTY or contain NO relevant existing lessons:**
           - ALWAYS set new_journey_needed = true
           - Use action = "create_new_lesson"
           - This indicates first-time content or completely new topic areas
        
        2. **If mappings show WEAK or NO overlap with existing lessons:**
           - Set new_journey_needed = true
           - Use action = "create_new_lesson"
           - Create new learning pathway for distinct content
        
        3. **If mappings show STRONG overlap (>70 percent content similarity):**
           - Set new_journey_needed = false
           - Choose appropriate action:
             - "extend_existing_lesson" - Add to existing lesson
             - "merge_with_existing" - Combine with existing content
             - "split_into_multiple" - Distribute across multiple lessons
        
        Integration Strategy Options:
        - CREATE_NEW_LESSON: Content is distinct and warrants new lessons
        - EXTEND_EXISTING_LESSON: Content can enhance existing lessons  
        - MERGE_WITH_EXISTING: Content overlaps significantly and should be merged
        - SPLIT_INTO_MULTIPLE: Content should be split across multiple lessons
        
        IMPORTANT: Empty or minimal mappings indicate NEW content that requires NEW journeys.
        Do not default to extending existing content when no clear relationships exist.
        
        Provide detailed rationale explaining your decision based on the mapping analysis.
        
        """),
        ]
    )
}
