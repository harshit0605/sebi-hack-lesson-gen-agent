from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal

# datetime import removed - not needed for simplified anchor model
from enum import Enum


class JourneyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class BlockType(str, Enum):
    CONCEPT = "concept"
    EXAMPLE = "example"
    INTERACTIVE = "interactive"
    QUIZ = "quiz"
    REFLECTION = "reflection"
    CALLOUT = "callout"


class SourceType(str, Enum):
    SEBI_PDF = "SEBI_PDF"
    SEBI_CIRCULAR = "SEBI_CIRCULAR"
    SEBI_INVESTOR_PORTAL = "SEBI_INVESTOR_PORTAL"
    NISM_MODULE = "NISM_MODULE"


class ContentIntegrationAction(str, Enum):
    CREATE_NEW_LESSON = "create_new_lesson"
    EXTEND_EXISTING_LESSON = "extend_existing_lesson"
    MERGE_WITH_EXISTING = "merge_with_existing"
    SPLIT_INTO_MULTIPLE = "split_into_multiple"


# Core Pydantic models


class LearningOutcome(BaseModel):
    outcome: str = Field(
        description="A specific, measurable learning outcome starting with action verbs like 'Understand', 'Explain', 'Apply', 'Analyze', 'Evaluate'. Should be aligned with SEBI investor education goals and appropriate for the target audience."
    )
    assessment_criteria: List[str] = Field(
        default_factory=list,
        description="Specific, measurable criteria to assess if the learning outcome has been achieved. Each criterion should be concrete and observable, focusing on practical application and understanding.",
    )


class JourneyModel(BaseModel):
    slug: str = Field(
        description="URL-friendly identifier for the journey (e.g., 'basic-investing', 'mutual-funds-101'). Should be lowercase with hyphens."
    )
    title: str = Field(
        description="Compelling, clear title for the learning journey that immediately conveys the value to Indian investors (e.g., 'Master the Basics of Smart Investing')."
    )
    description: str = Field(
        description="Engaging 2-3 sentence description explaining what learners will gain from this journey. Should highlight practical benefits and relevance to Indian financial markets."
    )
    level: JourneyLevel = Field(
        description="Appropriate difficulty level based on prerequisite knowledge and complexity of concepts covered."
    )
    outcomes: List[LearningOutcome] = Field(
        description="4-6 comprehensive learning outcomes that cover the key topics and skills learners will master. Should be specific, measurable, and aligned with SEBI investor education goals."
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="List of specific knowledge or skills learners should have before starting this journey. Be specific about financial concepts or prior learning journeys required.",
    )
    estimated_hours: float = Field(
        description="Realistic estimate of total time needed to complete the journey, including lessons, activities, and assessments. Consider Indian learner context and pacing."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Relevant keywords for discoverability (e.g., 'investing', 'mutual-funds', 'risk-management', 'sebi-guidelines'). Focus on terms Indian investors would search for.",
    )
    order: int = Field(
        description="Sequential order in the overall curriculum. Lower numbers indicate foundational content that should be learned first."
    )
    sebi_topics: List[str] = Field(
        default_factory=list,
        description="Specific SEBI regulatory topics, guidelines, or investor protection themes covered in this journey. Reference official SEBI terminology.",
    )
    status: str = Field(
        default="draft",
        description="Current status of the journey: 'draft', 'review', 'published', 'archived'.",
    )


class LessonMetadata(BaseModel):
    source_pages: List[int] = Field(
        description="List of page numbers from the source PDF where this lesson content was extracted from."
    )
    chunk_id: str = Field(
        description="Unique identifier for the content chunk that generated this lesson."
    )
    overlap_handled: bool = Field(
        default=False,
        description="Whether content overlap with existing lessons has been identified and handled.",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality assessment score for the lesson content (0.0 to 1.0).",
    )
    review_status: str = Field(
        default="generated",
        description="Current review status: 'generated', 'reviewed', 'approved', 'rejected'.",
    )
    sebi_compliance_checked: bool = Field(
        default=False,
        description="Whether the lesson has been checked for SEBI regulatory compliance.",
    )
    integration_action: Optional[ContentIntegrationAction] = Field(
        default=None,
        description="The integration action that was applied to create or modify this lesson. Value can be among create_new_lesson, extend_existing_lesson,merge_with_existing and split_into_multiple",
    )
    related_existing_lessons: List[str] = Field(
        default=[],
        description="List of existing lesson IDs that are related to or referenced by this lesson.",
    )


class LessonCreationModel(BaseModel):
    """Model for lesson creation that excludes content fields to avoid token waste"""

    journey_id: str = Field(
        description="ID of the parent learning journey this lesson belongs to."
    )
    slug: str = Field(
        description="URL-friendly identifier for the lesson (e.g., 'understanding-mutual-fund-nav'). Should be descriptive and SEO-friendly."
    )
    title: str = Field(
        description="Clear, engaging lesson title that immediately conveys the learning value. Should be specific and actionable (e.g., 'How to Read and Interpret Mutual Fund NAV')."
    )
    subtitle: Optional[str] = Field(
        default=None,
        description="Optional subtitle providing additional context or learning focus. Use when the main title needs clarification.",
    )
    unit: str = Field(
        description="Thematic unit or module this lesson belongs to within the journey (e.g., 'Mutual Fund Basics', 'Risk Assessment')."
    )
    estimated_minutes: int = Field(
        description="Realistic time estimate for an average Indian learner to complete this lesson, including reading, activities, and reflection. Consider mobile learning patterns."
    )
    difficulty: str = Field(
        description="Difficulty level: 'easy', 'medium', 'hard'. Should align with the complexity of financial concepts and required prior knowledge."
    )
    order: int = Field(
        description="Sequential order within the journey. Ensure logical progression from basic to advanced concepts."
    )
    learning_objectives: List[str] = Field(
        description="3-5 specific, measurable objectives learners will achieve. Start with action verbs and focus on practical skills applicable to Indian financial markets."
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Specific lessons or concepts learners should complete before this lesson. Be explicit about required knowledge.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Searchable keywords relevant to Indian investors (e.g., 'sip', 'tax-saving', 'elss', 'nfo'). Include both English and commonly used Hindi financial terms.",
    )
    metadata: LessonMetadata = Field(
        description="Technical metadata about lesson creation, quality, and integration status."
    )
    version: int = Field(
        default=1, description="Version number for content tracking and updates."
    )


class LessonModel(BaseModel):
    journey_id: str = Field(
        description="ID of the parent learning journey this lesson belongs to."
    )
    slug: str = Field(
        description="URL-friendly identifier for the lesson (e.g., 'understanding-mutual-fund-nav'). Should be descriptive and SEO-friendly."
    )
    title: str = Field(
        description="Clear, engaging lesson title that immediately conveys the learning value. Should be specific and actionable (e.g., 'How to Read and Interpret Mutual Fund NAV')."
    )
    subtitle: Optional[str] = Field(
        default=None,
        description="Optional subtitle providing additional context or learning focus. Use when the main title needs clarification.",
    )
    unit: str = Field(
        description="Thematic unit or module this lesson belongs to within the journey (e.g., 'Mutual Fund Basics', 'Risk Assessment')."
    )
    estimated_minutes: int = Field(
        description="Realistic time estimate for an average Indian learner to complete this lesson, including reading, activities, and reflection. Consider mobile learning patterns."
    )
    difficulty: str = Field(
        description="Difficulty level: 'easy', 'medium', 'hard'. Should align with the complexity of financial concepts and required prior knowledge."
    )
    order: int = Field(
        description="Sequential order within the journey. Ensure logical progression from basic to advanced concepts."
    )
    learning_objectives: List[str] = Field(
        description="3-5 specific, measurable objectives learners will achieve. Start with action verbs and focus on practical skills applicable to Indian financial markets."
    )
    blocks: List[str] = Field(
        default_factory=list,
        description="Ordered list of content block IDs that make up this lesson. Blocks should flow logically from concept introduction to practical application.",
    )
    anchors: List[str] = Field(
        default_factory=list,
        description="SEBI regulatory source references that provide authoritative backing for the lesson content. Essential for compliance and credibility.",
    )
    voice_ready: bool = Field(
        default=False,
        description="Whether this lesson has been optimized for voice-based learning delivery.",
    )
    voice_script_id: Optional[str] = Field(
        default=None,
        description="ID of the associated voice script if voice_ready is True.",
    )
    quiz_ids: List[str] = Field(
        default_factory=list,
        description="IDs of assessment quizzes associated with this lesson for knowledge validation.",
    )
    interactive_ids: List[str] = Field(
        default_factory=list,
        description="IDs of interactive elements like calculators, simulations, or decision trees that enhance learning.",
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Specific lessons or concepts learners should complete before this lesson. Be explicit about required knowledge.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Searchable keywords relevant to Indian investors (e.g., 'sip', 'tax-saving', 'elss', 'nfo'). Include both English and commonly used Hindi financial terms.",
    )
    metadata: LessonMetadata = Field(
        description="Technical metadata about lesson creation, quality, and integration status."
    )
    version: int = Field(
        default=1, description="Version number for content tracking and updates."
    )


# Content block payload models
class ConceptBlockPayload(BaseModel):
    heading: str = Field(
        description="Clear, descriptive heading that captures the core concept being taught. Should be specific and immediately understandable to Indian investors."
    )
    rich_text_md: str = Field(
        description="Comprehensive explanation in Markdown format. Include definitions, examples relevant to Indian markets, and practical applications. Use clear language appropriate for the target audience."
    )
    image_key: Optional[str] = Field(
        default=None,
        description="Optional reference to a relevant image, chart, or diagram that enhances understanding of the concept.",
    )
    key_terms: List[str] = Field(
        default_factory=list,
        description="Important financial terms introduced in this concept block. Include both English and commonly used Hindi terms where applicable.",
    )
    sebi_context: str = Field(
        description="Specific connection to SEBI guidelines, regulations, or investor protection measures. Explain how this concept relates to regulatory compliance and investor safety."
    )


class QAPair(BaseModel):
    question: str = Field(
        description="Question that guides critical thinking about the scenario"
    )
    answer: str = Field(
        description="Clear, educational answer that explains the concept"
    )


class ExampleBlockPayload(BaseModel):
    scenario_title: str = Field(
        description="Engaging title for the example scenario that clearly indicates the learning context (e.g., 'Choosing Between ELSS and PPF for Tax Saving')."
    )
    scenario_md: str = Field(
        description="Detailed scenario in Markdown format using realistic Indian financial situations. Include specific numbers, timeframes, and outcomes that learners can relate to."
    )
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="Question-answer pairs that help learners think through the scenario. Questions should guide critical thinking and practical application.",
    )
    indian_context: bool = Field(
        default=True,
        description="Whether the example uses Indian financial products, regulations, and market conditions. Should almost always be True for SEBI education.",
    )


class WidgetConfig(BaseModel):
    title: str = Field(description="Title of the interactive widget")
    description: str = Field(description="Brief description of what the widget does")
    parameters: List[str] = Field(
        description="List of input parameters or fields required"
    )
    default_values: List[str] = Field(
        default_factory=list, description="Default values for parameters"
    )


class ScoringRubric(BaseModel):
    criteria: List[str] = Field(description="List of evaluation criteria")
    scoring_method: str = Field(
        description="How the scoring is calculated (e.g., 'percentage', 'points', 'qualitative')"
    )
    passing_threshold: str = Field(
        description="Minimum requirement to pass (e.g., '70%', '3 out of 5')"
    )


class InteractiveBlockPayload(BaseModel):
    widget_kind: str = Field(
        description="Type of interactive element: 'calculator', 'simulation', 'decision_tree', 'drag_drop', 'scenario_builder'. Choose based on learning objectives."
    )
    widget_config: WidgetConfig = Field(
        description="Configuration parameters specific to the widget type. Include all necessary settings for the interactive element to function properly."
    )
    instructions_md: str = Field(
        description="Clear, step-by-step instructions in Markdown format explaining how to use the interactive element and what learners should focus on."
    )
    expected_outcomes: List[str] = Field(
        description="Specific learning outcomes or insights learners should gain from completing this interactive element."
    )
    scoring_rubric: ScoringRubric = Field(
        description="Criteria for evaluating learner performance or understanding, if applicable. Include both quantitative and qualitative measures."
    )
    fallback_content: str = Field(
        description="Static content to display if the interactive element cannot load. Should still provide the core learning value."
    )


class QuizChoice(BaseModel):
    text: str = Field(
        description="Clear, concise answer choice that is realistic and plausible. Avoid obviously wrong answers that don't test understanding."
    )
    correct: bool = Field(
        description="Whether this choice is correct. Ensure only one choice per question is marked as correct."
    )
    explanation: str = Field(
        description="Detailed explanation of why this choice is correct or incorrect. Include educational value even for wrong answers to reinforce learning."
    )


class QuizItem(BaseModel):
    stem: str = Field(
        description="Clear, unambiguous question that tests specific knowledge or application. Use realistic scenarios relevant to Indian investors."
    )
    choices: List[QuizChoice] = Field(
        description="3-4 plausible answer choices. Include common misconceptions as incorrect options to address typical learner confusion."
    )
    rationale: str = Field(
        description="Comprehensive explanation of the correct answer and why other options are incorrect. Connect to broader learning objectives."
    )
    anchor_ids: List[str] = Field(
        description="SEBI source references that support the correct answer. Essential for regulatory accuracy and credibility."
    )
    difficulty: str = Field(
        description="Question difficulty: 'easy', 'medium', 'hard'. Should align with lesson complexity and learning objectives."
    )


class QuizBlockPayload(BaseModel):
    quiz_type: str = Field(
        description="Type of quiz: 'knowledge_check', 'application', 'scenario_based', 'regulatory_compliance'. Choose based on learning objectives."
    )
    items: List[QuizItem] = Field(
        description="Collection of quiz questions that comprehensively assess the lesson's learning objectives. Ensure progressive difficulty."
    )
    pass_threshold: float = Field(
        default=70.0,
        description="Minimum percentage score required to pass the quiz. Adjust based on content criticality and learner level.",
    )


class ReflectionBlockPayload(BaseModel):
    prompt_md: str = Field(
        description="Thought-provoking reflection prompt in Markdown format that encourages learners to connect concepts to their personal financial situation."
    )
    guidance_md: str = Field(
        description="Helpful guidance in Markdown format with tips for meaningful reflection and examples of what good responses might include."
    )
    min_chars: int = Field(
        default=100,
        description="Minimum character count for reflection responses to ensure thoughtful engagement.",
    )
    reflection_type: str = Field(
        description="Type of reflection: 'personal_application', 'case_analysis', 'goal_setting', 'risk_assessment'. Choose based on learning objectives."
    )
    sample_responses: List[str] = Field(
        default_factory=list,
        description="Example responses that demonstrate the depth and type of thinking expected. Help learners understand the reflection quality expected.",
    )


class CalloutBlockPayload(BaseModel):
    style: str = Field(
        description="Visual style and urgency level: 'warning' (high risk/caution), 'info' (helpful information), 'compliance' (regulatory requirement), 'tip' (best practice), 'sebi_guideline' (official guidance)."
    )
    title: str = Field(
        description="Attention-grabbing title that immediately conveys the importance and relevance of the callout message."
    )
    text_md: str = Field(
        description="Concise, impactful message in Markdown format. Focus on actionable information or critical warnings that protect investors."
    )
    icon: str = Field(
        description="Appropriate icon identifier that reinforces the callout style and helps with visual recognition."
    )
    dismissible: bool = Field(
        default=True,
        description="Whether learners can dismiss this callout. Set to False for critical regulatory warnings or compliance information.",
    )


class ContentBlockMetadata(BaseModel):
    source_text: str = Field(
        description="Original text from SEBI document that this content block was generated from. Used for traceability and quality assurance."
    )
    generation_confidence: float = Field(
        description="Confidence score (0.0-1.0) indicating the quality and accuracy of the generated content. Higher scores indicate better alignment with source material."
    )
    manual_review_needed: bool = Field(
        default=False,
        description="Whether this content block requires human review before publication. Set to True for complex regulatory topics or low confidence scores.",
    )
    integration_notes: Optional[str] = Field(
        default=None,
        description="Notes about how this block integrates with existing content or special considerations for placement within the lesson.",
    )


class ContentBlockModel(BaseModel):
    lesson_id: str = Field(
        description="ID of the lesson this content block belongs to."
    )
    type: BlockType = Field(
        description="Type of content block that determines the learning approach and user interaction pattern."
    )
    order: int = Field(
        description="Sequential order within the lesson. Ensure logical flow from concept introduction through examples to application and assessment."
    )
    payload: Union[
        ConceptBlockPayload,
        ExampleBlockPayload,
        InteractiveBlockPayload,
        QuizBlockPayload,
        ReflectionBlockPayload,
        CalloutBlockPayload,
    ] = Field(
        description="Content-specific data structure that varies based on the block type. Contains all the educational content and configuration for this block."
    )
    anchor_ids: List[str] = Field(
        default_factory=list,
        description="SEBI regulatory source references that provide authoritative backing for this content block. Critical for compliance and credibility. Ids should match the short_label of the anchors generated.",
    )
    metadata: ContentBlockMetadata = Field(
        description="Technical metadata about content generation, quality assessment, and integration requirements."
    )


class AnchorModel(BaseModel):
    title: str = Field(
        description="Brief title of the anchor which can be used for display in UI"
    )
    short_label: str = Field(
        description="Brief label for quick identification (e.g., 'concept_1', 'regulation_1')"
    )
    excerpt: str = Field(
        description="Relevant text excerpt from the source content. Keep it concise."
    )
    # source_type: SourceType = Field(
    #     description="Type of source document (PDF, circular, etc.)"
    # )

    # created_from_chunk: str = Field(
    #     description="ID of the document chunk this anchor was extracted from"
    # )


class ConceptBlock(ContentBlockModel):
    """Concept block for explaining key learning concepts"""

    type: Literal[BlockType.CONCEPT] = BlockType.CONCEPT
    payload: ConceptBlockPayload = Field(
        description="Concept block payload containing the concept explanation, examples, and regulatory context."
    )


class ExampleBlock(ContentBlockModel):
    """Example block with practical scenarios and calculations"""

    type: Literal[BlockType.EXAMPLE] = BlockType.EXAMPLE
    payload: ExampleBlockPayload = Field(
        description="Example block payload containing the example scenario, calculations, and regulatory context."
    )


class QuizBlock(ContentBlockModel):
    """Quiz block for assessment with questions and answers"""

    type: Literal[BlockType.QUIZ] = BlockType.QUIZ
    payload: QuizBlockPayload = Field(
        description="Quiz block payload containing the quiz questions, answers, and regulatory context."
    )


class ReflectionBlock(ContentBlockModel):
    """Reflection block for critical thinking and self-assessment"""

    type: Literal[BlockType.REFLECTION] = BlockType.REFLECTION
    payload: ReflectionBlockPayload = Field(
        description="Reflection block payload containing the reflection prompt, guidance, and regulatory context."
    )


class InteractiveBlock(ContentBlockModel):
    """Interactive block for hands-on learning activities"""

    type: Literal[BlockType.INTERACTIVE] = BlockType.INTERACTIVE
    payload: InteractiveBlockPayload = Field(
        description="Interactive block payload containing the interactive activity, instructions, and regulatory context."
    )


class LessonContentBlocks(BaseModel):
    """Unified model for generating all content blocks and anchors for a lesson in one LLM call"""

    concept_blocks: List[ConceptBlock] = Field(
        description="2-3 concept blocks explaining key learning concepts with SEBI context and regulatory themes"
    )
    example_blocks: List[ExampleBlock] = Field(
        description="2-3 example blocks with practical Indian market scenarios, step-by-step calculations, and real-world applications"
    )
    quiz_blocks: List[QuizBlock] = Field(
        description="2-3 quiz blocks for assessment with scenario-based questions, multiple choice, and practical applications"
    )
    reflection_blocks: List[ReflectionBlock] = Field(
        description="2-3 reflection blocks for critical thinking, personal financial situation connection, and self-assessment"
    )
    interactive_blocks: List[InteractiveBlock] = Field(
        default_factory=list,
        description="Optional 1-2 interactive blocks for hands-on learning (calculators, simulations, guided activities)",
    )
    anchors: List[AnchorModel] = Field(
        description="3-5 content-grounded anchors extracted from PDF content, each linked to specific blocks and providing regulatory context"
    )


class VoiceCheckpoint(BaseModel):
    question: str
    pass_criteria: List[str]
    feedback_positive: str
    feedback_negative: str


class VoiceStep(BaseModel):
    step_number: int
    prompt: str
    expected_points: List[str]
    hints: List[str]
    checkpoint: VoiceCheckpoint
    anchor_ids: List[str]
    estimated_duration_seconds: int


class DifficultyAdjustment(BaseModel):
    level: str = Field(
        description="Difficulty level (e.g., 'beginner', 'intermediate', 'advanced')"
    )
    adjustments: List[str] = Field(
        description="List of specific adjustments for this difficulty level"
    )
    time_multiplier: float = Field(
        description="Time adjustment factor (e.g., 1.5 for slower pace)"
    )


class VoiceScriptModel(BaseModel):
    lesson_id: str
    steps: List[VoiceStep]
    total_estimated_minutes: int
    difficulty_adjustments: List[DifficultyAdjustment] = Field(default_factory=list)


# Content integration models
class ContentAnalysisResult(BaseModel):
    key_concepts: List[str] = Field(
        description="Comprehensive list of atmost 30 key financial concepts extracted from the content. Each concept should be specific, clearly defined, and relevant to Indian investors. Include both basic and advanced concepts with clear explanations."
    )
    sebi_themes: List[str] = Field(
        description="Specific SEBI regulatory themes, guidelines, and investor protection measures identified in the content. Reference official SEBI terminology and current regulatory priorities."
    )
    learning_opportunities: List[str] = Field(
        description="Detailed learning opportunities and lesson ideas derived from the content. Each opportunity should specify the learning objective, target audience, and practical application for Indian investors."
    )
    complexity_level: str = Field(
        description="Overall complexity assessment: 'Beginner' (basic financial concepts), 'Intermediate' (requires some financial knowledge), 'Advanced' (complex financial instruments and strategies)."
    )
    estimated_lesson_count: int = Field(
        description="Realistic estimate of how many focused lessons this content could generate. Consider content depth, concept complexity, and optimal lesson length for effective learning."
    )
    content_type: Literal["regulatory", "educational", "practical"] = Field(
        description="Primary content classification: 'regulatory' (rules and compliance), 'educational' (concepts and knowledge), 'practical' (application and tools)."
    )


class ExistingContentMapping(BaseModel):
    lesson_id: str = Field(
        description="Unique identifier of the existing lesson that has potential overlap or integration opportunity."
    )
    lesson_title: str = Field(
        description="Title of the existing lesson for easy identification and context."
    )
    journey: str = Field(
        description="Learning journey that the existing lesson belongs to."
    )
    similarity_score: float = Field(
        description="Quantitative similarity score (0.0-1.0) indicating how closely the new content aligns with the existing lesson. Higher scores suggest stronger integration potential."
    )
    overlap_concepts: List[str] = Field(
        description="Specific concepts, topics, or learning objectives that overlap between the new content and existing lesson. Be precise about the nature of the overlap."
    )
    integration_potential: str = Field(
        description="Assessment of integration feasibility: 'high' (strong overlap, easy integration), 'medium' (some overlap, moderate effort), 'low' (minimal overlap, significant effort required)."
    )


class ExistingContentMappings(BaseModel):
    mappings: List[ExistingContentMapping] = Field(
        description="List of existing content mappings for the lesson."
    )


class LessonContentDistribution(BaseModel):
    lesson_id: str = Field(
        description="Unique identifier of the lesson (existing or new) that will contain this content."
    )
    lesson_title: str = Field(
        description="Clear, descriptive title for the lesson. For new lessons, create an engaging title. For existing lessons, use the current title."
    )
    concepts_to_cover: List[str] = Field(
        description="Specific concepts from the content analysis that this lesson will cover. Be precise about which concepts belong in this lesson."
    )
    integration_type: Literal["new_lesson", "extend_existing", "merge_content"] = Field(
        description="How this content will be integrated: 'new_lesson' (completely new), 'extend_existing' (add to existing), 'merge_content' (combine with existing)."
    )
    estimated_duration_minutes: int = Field(
        description="Realistic estimate of additional time this content will add to the lesson, considering the concepts' complexity and depth."
    )
    prerequisite_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts that learners must understand before tackling this lesson's content. Reference specific concepts from other lessons.",
    )
    learning_objectives: List[str] = Field(
        description="3-5 specific learning objectives that learners will achieve from this lesson's content. Start with action verbs and focus on practical application."
    )


class ContentIntegrationPlan(BaseModel):
    action: ContentIntegrationAction = Field(
        description="Primary integration strategy based on content analysis and existing content mappings. Choose the approach that best serves learner progression and content coherence."
    )
    target_lessons: List[str] = Field(
        default_factory=list,
        description="Specific existing lesson IDs that will be modified as part of this integration plan. Include only lessons that will be directly affected.",
    )
    new_lessons_needed: int = Field(
        default=0,
        description="Number of completely new lessons required to properly cover the new content. Consider optimal lesson length and learning objectives.",
    )
    new_journey_needed: bool = Field(
        default=False,
        description="Whether the content requires creating an entirely new learning journey because it doesn't fit well within existing journey structures.",
    )
    rationale: str = Field(
        description="Comprehensive explanation of why this integration approach was chosen. Include pedagogical reasoning, content coherence considerations, and learner experience impact."
    )
    content_distribution: List[LessonContentDistribution] = Field(
        description="Detailed breakdown of how content will be distributed across lessons. Each item specifies exactly what concepts go into which lesson and how they integrate."
    )


class JourneyCreationPlan(BaseModel):
    title: str = Field(
        description="Compelling title for the new learning journey that clearly communicates value to Indian investors. Should be specific and outcome-focused."
    )
    slug: str = Field(
        description="URL-friendly identifier for the journey. Should be descriptive and follow naming conventions (lowercase with hyphens)."
    )
    description: str = Field(
        description="Engaging description that explains what learners will achieve and why this journey is valuable. Highlight practical benefits and relevance to Indian financial markets."
    )
    level: JourneyLevel = Field(
        description="Appropriate difficulty level based on the complexity of concepts and required prerequisite knowledge."
    )
    order: int = Field(description="Order of the journey in the curriculum.")
    justification: str = Field(
        description="Detailed explanation of why a new journey is necessary rather than integrating with existing journeys. Include pedagogical and structural reasoning."
    )
    outcomes: List[LearningOutcome] = Field(
        description="4-6 high-level learning outcomes synthesized from the lesson structure. Should be specific, measurable, and aligned with SEBI investor education goals."
    )
    total_estimated_hours: float = Field(
        description="Total journey duration calculated from individual lesson estimates in the lesson structure. Should reflect realistic completion time for target audience."
    )
    key_topics: List[str] = Field(
        description="Core topics and concepts this journey will cover. Should be comprehensive yet focused, with clear relevance to SEBI investor education goals. Limit the list to 5-7 topics."
    )
    target_audience: str = Field(
        description="Specific description of the intended learners, including their background, knowledge level, and learning goals. Be specific about Indian investor context."
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="If this journey requires any prerequisites from existing journeys, list them here. Otherwise, leave this empty.",
    )
