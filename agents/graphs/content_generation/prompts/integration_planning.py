"""Integration planning prompt templates"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System message for integration planning
INTEGRATION_PLANNING_SYSTEM_MESSAGE = """You are an expert curriculum designer specializing in educational content integration and learning pathway optimization. Your role is to determine how new educational content should be integrated with existing SEBI investor education materials.

Consider:
- Learning progression and prerequisites
- Content overlap and redundancy
- Optimal lesson sequencing
- Learner cognitive load
- Educational coherence and flow
- Assessment and reinforcement opportunities"""

INTEGRATION_PLANNING_PROMPTS = {
    "create_integration_plan": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTEGRATION_PLANNING_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Analyze content and existing lessons to determine the optimal integration strategy.
        
        Content Analysis:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        Complexity Level: {complexity_level}
        
        Existing Content Mapping:
        Existing Lessons: {existing_lessons}
        Overlap Analysis: {overlap_analysis}
        
        Determine the best integration approach:
        1. CREATE_NEW_LESSON - If content is sufficiently unique
        2. EXTEND_EXISTING_LESSON - If content complements existing lesson
        3. MERGE_WITH_EXISTING - If content overlaps significantly
        4. SPLIT_INTO_MULTIPLE - If content is too dense for single lesson
        
        Consider:
        - Learning progression and prerequisites
        - Content overlap and redundancy
        - Optimal lesson sequencing
        - Learner cognitive load
        - Educational coherence
        
        {format_instructions}
        """)
    ])
}
