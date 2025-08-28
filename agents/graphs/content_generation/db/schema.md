// Core collections with relationships
// 1. Journeys collection
{
    _id: ObjectId,
    slug: String, // "market-basics", "risk-diversification"
    title: String,
    description: String,
    level: String, // "beginner", "intermediate", "advanced"
    outcomes: [
        {
      id: String,
      text: String,
      anchor_ids: [ObjectId
            ]
        }
    ],
    prerequisites: [ObjectId
    ], // journey dependencies
    estimated_hours: Number,
    tags: [String
    ],
    order: Number,
    created_at: Date,
    updated_at: Date,
    sebi_topics: [String
    ], // mapped SEBI topic areas
    status: String // "draft", "published", "archived"
}
// 2. Lessons collection
{
    _id: ObjectId,
    journey_id: ObjectId,
    slug: String,
    title: String,
    subtitle: String,
    unit: String, // "Unit 1: Market Structure"
    estimated_minutes: Number,
    difficulty: String, // "easy", "medium", "hard"
    order: Number,
    learning_objectives: [String
    ],
    blocks: [ObjectId
    ], // references to content_blocks
    anchors: [ObjectId
    ], // SEBI sources used
    voice_ready: Boolean,
    voice_script_id: ObjectId,
    quiz_ids: [ObjectId
    ],
    interactive_ids: [ObjectId
    ],
    prerequisites: [ObjectId
    ], // lesson dependencies
    tags: [String
    ],
    metadata: {
      source_pages: [Number
        ], // PDF pages this lesson was created from
      chunk_id: String, // tracking which PDF chunk
      overlap_handled: Boolean,
      quality_score: Number, // 0-100
      review_status: String, // "generated", "reviewed", "approved"
      sebi_compliance_checked: Boolean
    },
    created_at: Date,
    updated_at: Date,
    version: Number
}
// 3. Content blocks collection
{
    _id: ObjectId,
    lesson_id: ObjectId,
    type: String, // "concept", "example", "interactive", "quiz", "reflection", "callout"
    order: Number,
    payload: {}, // type-specific structured data
    anchor_ids: [ObjectId
    ],
    metadata: {
      source_text: String, // original PDF text this was derived from
      generation_confidence: Number,
      manual_review_needed: Boolean
    },
    created_at: Date
}
// 4. Block payload schemas by type
// Concept block payload:
{
    heading: String,
    rich_text_md: String,
    image_key: String, // optional
    key_terms: [String
    ],
    sebi_context: String // how this relates to SEBI guidelines
}
// Example block payload:
{
    scenario_title: String,
    scenario_md: String,
    qa_pairs: [
        {
      question: String,
      answer: String
        }
    ],
    indian_context: Boolean // uses Indian market examples
}
// Interactive block payload:
{
    widget_kind: String, // "risk_slider", "riskometer_interpret", "order_ticket"
    widget_config: {},
    instructions_md: String,
    expected_outcomes: [String
    ],
    scoring_rubric: {},
    fallback_content: String // if interactive fails
}
// Quiz block payload:
{
    quiz_type: String, // "mcq", "multi_select", "true_false", "ordering"
    items: [
        {
      stem: String,
      choices: [
                {
        text: String,
        correct: Boolean,
        explanation: String
                }
            ],
      rationale: String,
      anchor_ids: [ObjectId
            ],
      difficulty: String
        }
    ],
    pass_threshold: Number // percentage to pass
}
// Reflection block payload:
{
    prompt_md: String,
    guidance_md: String,
    min_chars: Number,
    reflection_type: String, // "journal", "teach_back", "application"
    sample_responses: [String
    ] // good example responses
}
// Callout block payload:
{
    style: String, // "warning", "info", "compliance", "tip", "sebi_guideline"
    title: String,
    text_md: String,
    icon: String,
    dismissible: Boolean
}
// 5. Anchors collection (SEBI sources)
{
    _id: ObjectId,
    source_type: String, // "SEBI_PDF", "SEBI_CIRCULAR", "SEBI_INVESTOR_PORTAL", "NISM_MODULE"
    title: String,
    short_label: String, // for UI chips
    excerpt: String, // key excerpt for context
    source_url: String,
    document_title: String,
    page_numbers: [Number
    ], // for PDF sources
    section: String, // chapter/section reference
    last_verified_at: Date,
    relevance_tags: [String
    ], // "risk", "mutual_funds", "trading"
    created_from_chunk: String, // which processing chunk created this
    confidence_score: Number // how confident we are in this anchor
}
// 6. Voice scripts collection
{
    _id: ObjectId,
    lesson_id: ObjectId,
    steps: [
        {
      step_number: Number,
      prompt: String,
      expected_points: [String
            ],
      hints: [String
            ],
      checkpoint: {
        type: String, // "mcq", "short_answer", "comprehension"
        data: {}, // checkpoint-specific data
        pass_criteria: String
            },
      anchor_ids: [ObjectId
            ],
      estimated_duration_seconds: Number
        }
    ],
    total_estimated_minutes: Number,
    difficulty_adjustments: {
      beginner: {}, // script modifications for beginners
      advanced: {} // script modifications for advanced learners
    },
    created_at: Date
}
// 7. Processing state collection (for agent context)
{
    _id: ObjectId,
    session_id: String, // unique ID for this PDF processing session
    pdf_metadata: {
      title: String,
      total_pages: Number,
      sebi_document_type: String
    },
    chunks_processed: [
        {
      chunk_id: String,
      pages: [Number
            ],
      processed_at: Date,
      lessons_created: [ObjectId
            ],
      anchors_created: [ObjectId
            ],
      overlap_with_previous: Boolean
        }
    ],
    journey_assignments: [
        {
      journey_slug: String,
      confidence: Number,
      content_coverage: [String
            ] // topics covered
        }
    ],
    global_context: {
      key_themes: [String
        ],
      recurring_concepts: [String
        ],
      sebi_guidelines_referenced: [String
        ]
    },
    next_expected_pages: [Number
    ],
    status: String, // "in_progress", "completed", "error"
    created_at: Date,
    updated_at: Date
}
// 8. Quality metrics collection
{
    _id: ObjectId,
    lesson_id: ObjectId,
    metrics: {
      content_completeness: Number, // 0-100
      sebi_alignment: Number, // 0-100
      educational_value: Number, // 0-100
      block_diversity: Number, // variety of block types
      anchor_quality: Number, // quality of SEBI references
      quiz_difficulty_balance: Number,
      estimated_engagement: Number
    },
    flags: [
        {
      type: String, // "missing_anchor", "too_complex", "insufficient_examples"
      severity: String, // "low", "medium", "high"
      description: String,
      auto_fixable: Boolean
        }
    ],
    review_notes: String,
    approved_by: String,
    approved_at: Date,
    created_at: Date
}