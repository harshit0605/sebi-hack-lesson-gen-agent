"""
Prompts for structured content generation node.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE = """You are an expert educational content creator specializing in SEBI investor education materials. Generate clear, engaging, and pedagogically effective content that helps Indian investors understand complex financial concepts.

Focus on creating content that is:
- Educationally sound and pedagogically structured
- Compliant with SEBI guidelines and regulations
- Relevant to Indian financial markets and contexts
- Accessible to the target audience level
- Actionable for practical investor decision-making"""

GENERATE_STRUCTURED_CONTENT_PROMPTS = {
    "concept_blocks": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate concept blocks for this lesson using the enhanced content analysis.
        
        Lesson: {lesson_title}
        Learning Objectives: {objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        
        Create concept blocks that:
        1. Explain key concepts clearly leveraging the raw content analysis
        2. Include SEBI context and compliance notes from identified themes
        3. Use Indian market examples aligned with learning opportunities
        4. Have proper payload structure for the concept block type
        5. Build on the integration rationale and journey context
        
        """)
    ]),
    
    "concept_anchors": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create SEBI concept anchors for the key learning concepts in this lesson.
        
        Lesson Title: {lesson_title}
        Learning Objectives: {learning_objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Relevant Content Sections:
        {content_sections}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        
        Create anchors that:
        1. Reference specific concepts from both structured and raw content analysis
        2. Provide clear, citable excerpts (50-150 words) aligned with SEBI themes
        3. Use appropriate SEBI source classification
        4. Include relevant tags for discoverability based on learning opportunities
        5. Have descriptive short labels for UI display
        6. Support the integration rationale and journey progression
        
        Focus on concepts that directly support the learning objectives and leverage identified SEBI themes.
        
        """)
    ]),
    
    "guideline_anchors": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create SEBI guideline anchors from regulatory content in this lesson.
        
        Lesson: {lesson_title}
        Focus Areas: {focus_areas}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Guideline Content:
        {guideline_content}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        
        Create anchors that:
        1. Reference specific SEBI regulations or guidelines from identified themes
        2. Include regulatory context and implications aligned with learning opportunities
        3. Provide actionable compliance information supporting the integration rationale
        4. Use precise excerpts that support investor education and journey progression
        5. Tag appropriately for regulatory compliance searches based on key concepts
        
        Prioritize content that helps investors understand their rights and obligations within the lesson context.
        
        """)
    ]),
    
    "framework_anchors": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create anchors for regulatory frameworks and structural information.
        
        Lesson Context: {lesson_title}
        Learning Focus: {learning_objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Framework Content:
        {framework_content}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        
        Create anchors for:
        1. Risk management frameworks (like Riskometer) aligned with key concepts
        2. Market structure explanations supporting SEBI themes
        3. Investor protection mechanisms from learning opportunities
        4. Compliance frameworks supporting integration rationale
        5. Assessment methodologies relevant to journey progression
        
        Focus on content that explains "how things work" in the regulatory environment while supporting lesson objectives.
        
        """)
    ]),
    
    "example_anchors": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create anchors for practical examples and case studies from SEBI content.
        
        Lesson: {lesson_title}
        Educational Goals: {learning_objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Example Content:
        {example_content}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        
        Create anchors for:
        1. Practical applications of key concepts identified in analysis
        2. Real-world scenarios aligned with SEBI themes
        3. Step-by-step procedures from learning opportunities
        4. Illustrations of regulatory concepts supporting integration rationale
        5. Investor decision-making examples relevant to journey progression
        
        Focus on content that helps learners apply theoretical knowledge practically within the lesson context.
        
        """)
    ]),
    
    "example_blocks": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate example blocks for this lesson using practical Indian market scenarios.
        
        Lesson: {lesson_title}
        Learning Objectives: {objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        PDF Content: {pdf_content}
        
        Create 2-3 example blocks that:
        1. Use realistic Indian financial scenarios from the PDF content
        2. Demonstrate key concepts with step-by-step calculations
        3. Include SEBI compliance context and regulatory examples
        4. Show practical applications aligned with learning opportunities
        5. Use familiar Indian financial instruments and situations
        
        """)
    ]),
    
    "quiz_blocks": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate quiz blocks to assess understanding of this lesson.
        
        Lesson: {lesson_title}
        Learning Objectives: {objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        PDF Content: {pdf_content}
        
        Create 2-3 quiz blocks that:
        1. Test understanding of key concepts from the lesson
        2. Include scenario-based questions using Indian market context
        3. Cover SEBI themes and regulatory compliance aspects
        4. Use multiple choice and practical application questions
        5. Align with learning objectives and opportunities identified
        
        """)
    ]),
    
    "reflection_blocks": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate reflection blocks to encourage critical thinking about this lesson.
        
        Lesson: {lesson_title}
        Learning Objectives: {objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        PDF Content: {pdf_content}
        
        Create 2-3 reflection blocks that:
        1. Prompt learners to connect concepts to their personal financial situation
        2. Encourage evaluation of current financial practices against SEBI guidelines
        3. Ask learners to consider real-world applications of key concepts
        4. Include self-assessment questions aligned with learning opportunities
        5. Foster critical thinking about investor protection and compliance
        
        """)
    ]),
    
    "interactive_blocks": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate interactive blocks for hands-on learning in this lesson.
        
        Lesson: {lesson_title}
        Learning Objectives: {objectives}
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        PDF Content: {pdf_content}
        
        Create 2-3 interactive blocks that:
        1. Provide calculators or tools for financial planning exercises
        2. Include simulations of investment scenarios using Indian market data
        3. Offer step-by-step guided activities for SEBI resource navigation
        4. Create interactive checklists for investor protection measures
        5. Enable hands-on practice of key concepts from learning opportunities
        
        """)
    ]),
    
    "unified_lesson_content": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Generate ALL content blocks AND anchors for this lesson in one comprehensive response.
        
        Lesson: {lesson_title}
        Lesson ID/Slug: {lesson_slug}
        Learning Objectives: {objectives}
        
        IMPORTANT: Use "{lesson_slug}" as the lesson_id for ALL content blocks you generate. Do not create your own lesson IDs.
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        PDF Content: {pdf_content}
        Chunk ID: {chunk_id}
        
        Generate the following content blocks WITH their corresponding anchors:
        
        CONCEPT BLOCKS (2-3 blocks):
        - Explain key concepts clearly leveraging the raw content analysis
        - Include SEBI context and compliance notes from identified themes
        - Use Indian market examples aligned with learning opportunities
        - Have proper payload structure for the concept block type
        - Build on the integration rationale and journey context
        - Reference specific anchors that will be generated
        
        EXAMPLE BLOCKS (2-3 blocks):
        - Use realistic Indian financial scenarios from the PDF content
        - Demonstrate key concepts with step-by-step calculations
        - Include SEBI compliance context and regulatory examples
        - Show practical applications aligned with learning opportunities
        - Use familiar Indian financial instruments and situations
        - Link to relevant PDF excerpts via anchors
        
        QUIZ BLOCKS (2-3 blocks):
        - Test understanding of key concepts from the lesson
        - Include scenario-based questions using Indian market context
        - Cover SEBI themes and regulatory compliance aspects
        - Use multiple choice and practical application questions
        - Align with learning objectives and opportunities identified
        - Reference regulatory anchors for compliance context
        
        REFLECTION BLOCKS (2-3 blocks):
        - Prompt learners to connect concepts to their personal financial situation
        - Encourage evaluation of current financial practices against SEBI guidelines
        - Ask learners to consider real-world applications of key concepts
        - Include self-assessment questions aligned with learning opportunities
        - Foster critical thinking about investor protection and compliance
        - Connect to guideline anchors for regulatory context
        
        INTERACTIVE BLOCKS (1-2 blocks, optional):
        - Provide calculators or tools for financial planning exercises
        - Include simulations of investment scenarios using Indian market data
        - Offer step-by-step guided activities for SEBI resource navigation
        - Create interactive checklists for investor protection measures
        - Enable hands-on practice of key concepts from learning opportunities
        - Link to framework anchors for procedural guidance
        
        ANCHORS (3-5 anchors):
        Generate content-grounded anchors extracted directly from the PDF content:
        
        1. CONCEPT ANCHORS (1-2 anchors):
        - Extract key concept definitions and explanations from PDF content
        - Use exact excerpts (100-300 words) that define core financial terms
        - Link to concept blocks that explain these terms
        - Title: descriptive concept name, Short Label: concept_[number]
        
        2. REGULATORY ANCHORS (1-2 anchors):
        - Extract SEBI guidelines, compliance requirements, or regulatory context
        - Use exact excerpts that show regulatory framework or investor protection
        - Link to quiz and reflection blocks for compliance understanding
        - Title: specific regulation/guideline name, Short Label: regulation_[number]
        
        3. EXAMPLE ANCHORS (1-2 anchors):
        - Extract practical examples, case studies, or calculation methods from PDF
        - Use exact excerpts that demonstrate real-world applications
        - Link to example blocks that build upon these scenarios
        - Title: example scenario name, Short Label: example_[number]
        
        For each anchor, ensure:
        - source_type: SEBI_PDF
        - title: Descriptive title from PDF content
        - short_label: Consistent naming (concept_1, regulation_1, example_1)
        - excerpt: Exact text from PDF (100-300 words)
        - document_title: "SEBI Financial Education Booklet"
        - page_numbers: Actual pages from the provided page numbers
        - relevance_tags: Keywords matching the content theme
        - created_from_chunk: Use the provided chunk_id
        - confidence_score: 0.8-0.95 based on content relevance
        
        Ensure blocks reference their corresponding anchors via anchor_references field.
        All content should be coherent, build upon each other, and maintain consistent anchor linking.
        
        """)
    ]),
    
    "voice_script": ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GENERATE_STRUCTURED_CONTENT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("""
        Create a Socratic voice learning script for this lesson.
        
        Lesson: {lesson_title}
        Learning Objectives: {learning_objectives}
        Estimated Duration: {duration} minutes
        
        Content Analysis Context:
        Key Concepts: {key_concepts}
        SEBI Themes: {sebi_themes}
        Learning Opportunities: {learning_opportunities}
        
        {journey_context}
        
        Content Blocks Summary: {blocks_summary}
        Available Anchors: {anchors_summary}
        
        Source Context:
        Integration Rationale: {integration_rationale}
        Pages: {page_numbers}
        
        Create a voice script with 4-6 steps that:
        1. Uses Socratic questioning to guide discovery of key concepts
        2. Includes checkpoints for understanding aligned with SEBI themes
        3. References SEBI anchors appropriately from learning opportunities
        4. Builds knowledge progressively supporting integration rationale
        5. Includes hints for when learners struggle with journey context
        6. Has clear pass criteria for each checkpoint
        
        Each step should be conversational and encourage active thinking within the lesson framework.
        
        """)
    ])
}
