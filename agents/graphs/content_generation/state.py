from typing import TypedDict, List, Optional, Dict, Any, Annotated
import operator
from agents.graphs.content_generation.models import (
    JourneyModel,
    LessonModel,
    AnchorModel,
    ContentAnalysisResult,
    ExistingContentMapping,
    ContentIntegrationPlan,
    JourneyCreationPlan,
    LessonContentBlocks,
    VoiceScriptModel,
    LessonContentDistribution,
)

import uuid


class LessonCreationState(TypedDict, total=False):
    # Input
    pdf_content: str
    page_numbers: List[int]
    # original source context propagated from parent graph
    source_url: str
    source_type: str
    session_id: str = f"session_{uuid.uuid4()}"
    chunk_id: str = f"chunk_{uuid.uuid4()}"

    # Context from previous processing
    existing_journeys: str
    existing_journeys_list: List[Dict[str, Any]]
    # existing_lessons: List[LessonModel]
    # existing_anchors: List[AnchorModel]
    processing_history: Dict[str, Any]

    # Analysis results
    content_analysis: ContentAnalysisResult
    existing_content_mappings: List[ExistingContentMapping] = []
    integration_plan: ContentIntegrationPlan
    journey_creation_plan: Optional[JourneyCreationPlan]

    # Generated content (Pydantic models)
    new_journeys: List[JourneyModel]
    lessons: List[LessonModel]  # Unified field for all generated lessons
    content_blocks: List[LessonContentBlocks]
    anchors: List[AnchorModel]
    voice_scripts: List[VoiceScriptModel]

    # Lesson metadata for parallel processing
    lessons_for_content_generation: List[LessonModel]
    lesson_distributions_for_creation: List[LessonContentDistribution]

    # Parallel execution results (with reducer for concurrent updates)
    lesson_content_results: Annotated[List[Dict[str, Any]], operator.add]
    lesson_metadata_results: Annotated[List[LessonModel], operator.add]

    # Legacy fields (deprecated - use 'lessons' instead)
    new_lessons: Optional[List[LessonModel]]  # For backward compatibility
    updated_lessons: Optional[List[LessonModel]]  # For backward compatibility

    # Quality and control
    quality_metrics: Dict[str, float]
    processing_flags: List[Dict[str, Any]]
    validation_errors: List[str]

    # Flow control
    current_step: str
    requires_human_review: bool = False
    retry_count: int = 0
