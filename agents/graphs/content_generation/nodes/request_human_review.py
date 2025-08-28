from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.models import ContentIntegrationAction
from typing import Dict, Any
import logging
from datetime import datetime


async def request_human_review(state: LessonCreationState) -> LessonCreationState:
    """Prepare content for human review with comprehensive context and recommendations"""

    # Compile review context
    review_context = await compile_review_context(state)

    # Generate review recommendations
    recommendations = await generate_review_recommendations(state)

    # Create review record
    review_record = {
        "session_id": state["session_id"],
        "chunk_id": state["chunk_id"],
        "timestamp": datetime.now().isoformat(),
        "review_context": review_context,
        "recommendations": recommendations,
        "priority": determine_review_priority(state),
        "estimated_review_time": estimate_review_time(state),
        "auto_fix_attempted": state.get("retry_count", 0) > 0,
        "validation_errors": state.get("validation_errors", []),
        "quality_metrics": state.get("quality_metrics", {}),
        "content_summary": generate_content_summary(state),
    }

    # Store review record for human reviewer
    await store_review_record(review_record)

    logging.info(
        f"Content queued for human review - Priority: {review_record['priority']}, Session: {state['session_id']}"
    )

    # Return partial state update only
    return {
        "requires_human_review": True,
        "review_record": review_record,
        "current_step": "awaiting_human_review",
    }


async def compile_review_context(state: LessonCreationState) -> Dict[str, Any]:
    """Compile comprehensive context for human reviewer"""

    context = {
        "content_analysis": state.get("content_analysis"),
        "integration_plan": state.get("integration_plan"),
        "journey_creation_plan": state.get("journey_creation_plan"),
        "processing_history": {
            "chunks_processed": len(
                state.get("processing_history", {}).get("chunks_processed", [])
            ),
            "total_lessons_created": len(state.get("new_lessons", [])),
            "total_journeys_created": len(state.get("new_journeys", [])),
            "existing_content_mappings": len(
                state.get("existing_content_mappings", [])
            ),
        },
        "content_statistics": {
            "pdf_pages": len(state.get("page_numbers", [])),
            "estimated_lesson_count": state.get("content_analysis", {}).get(
                "estimated_lesson_count", 0
            )
            if state.get("content_analysis")
            else 0,
            "concepts_identified": len(
                state.get("content_analysis", {}).get("key_concepts", [])
            )
            if state.get("content_analysis")
            else 0,
            "sebi_themes": len(state.get("content_analysis", {}).get("sebi_themes", []))
            if state.get("content_analysis")
            else 0,
        },
        "generated_content": {
            "new_lessons": len(state.get("new_lessons", [])),
            "updated_lessons": len(state.get("updated_lessons", [])),
            "content_blocks": len(state.get("content_blocks", [])),
            "anchors": len(state.get("anchors", [])),
            "voice_scripts": len(state.get("voice_scripts", [])),
        },
    }

    return context


async def generate_review_recommendations(state: LessonCreationState) -> Dict[str, Any]:
    """Generate specific recommendations for human reviewer"""

    recommendations = {
        "critical_issues": [],
        "suggested_fixes": [],
        "content_improvements": [],
        "compliance_checks": [],
        "quality_enhancements": [],
    }

    # Analyze validation errors for critical issues
    validation_errors = state.get("validation_errors", [])
    for error in validation_errors:
        if "missing SEBI" in error or "compliance" in error:
            recommendations["critical_issues"].append(
                {
                    "issue": error,
                    "severity": "high",
                    "action": "Manual review of SEBI compliance required",
                }
            )
        elif "duration" in error or "content" in error:
            recommendations["suggested_fixes"].append(
                {
                    "issue": error,
                    "severity": "medium",
                    "action": "Adjust content structure or lesson sizing",
                }
            )

    # Quality-based recommendations
    quality_metrics = state.get("quality_metrics", {})
    if quality_metrics.get("sebi_alignment", 0) < 0.8:
        recommendations["compliance_checks"].append(
            {
                "metric": "SEBI Alignment",
                "score": quality_metrics.get("sebi_alignment", 0),
                "recommendation": "Review content for SEBI guideline alignment and add proper anchors",
            }
        )

    if quality_metrics.get("educational_value", 0) < 0.7:
        recommendations["content_improvements"].append(
            {
                "metric": "Educational Value",
                "score": quality_metrics.get("educational_value", 0),
                "recommendation": "Enhance examples, add more interactive elements, improve learning objectives",
            }
        )

    # Integration-specific recommendations
    integration_plan = state.get("integration_plan")
    if (
        integration_plan
        and integration_plan.action == ContentIntegrationAction.CREATE_NEW_JOURNEY
    ):
        recommendations["quality_enhancements"].append(
            {
                "area": "New Journey Creation",
                "recommendation": f"Review new journey '{state.get('journey_creation_plan', {}).get('title', 'Unknown')}' for uniqueness and educational value",
            }
        )

    return recommendations


def determine_review_priority(state: LessonCreationState) -> str:
    """Determine priority level for human review"""

    validation_errors = state.get("validation_errors", [])
    quality_metrics = state.get("quality_metrics", {})

    # High priority conditions
    if any(
        "compliance" in error.lower() or "sebi" in error.lower()
        for error in validation_errors
    ):
        return "high"

    if quality_metrics.get("sebi_alignment", 1.0) < 0.6:
        return "high"

    if state.get("journey_creation_plan") is not None:
        return "high"

    # Medium priority conditions
    if len(validation_errors) > 3:
        return "medium"

    avg_quality = (
        sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0.5
    )
    if avg_quality < 0.7:
        return "medium"

    # Low priority (default)
    return "low"


def estimate_review_time(state: LessonCreationState) -> int:
    """Estimate review time in minutes"""

    base_time = 10  # Base review time

    # Add time based on content volume
    lessons_count = len(state.get("new_lessons", []))
    blocks_count = len(state.get("content_blocks", []))

    content_time = (lessons_count * 5) + (blocks_count * 2)

    # Add time for errors
    error_time = len(state.get("validation_errors", [])) * 3

    # Add time for new journey creation
    journey_time = 15 if state.get("journey_creation_plan") else 0

    total_time = base_time + content_time + error_time + journey_time

    return min(total_time, 120)  # Cap at 2 hours


def generate_content_summary(state: LessonCreationState) -> Dict[str, Any]:
    """Generate a summary of the content for quick review"""

    summary = {}

    # Content analysis summary
    if state.get("content_analysis"):
        analysis = state["content_analysis"]
        summary["content_focus"] = {
            "key_concepts": analysis.key_concepts[:5],  # Top 5 concepts
            "primary_themes": analysis.sebi_themes[:3],  # Top 3 themes
            "complexity": analysis.complexity_level,
            "content_type": analysis.content_type,
        }

    # Generated lessons summary
    if state.get("new_lessons"):
        lessons = state["new_lessons"]
        summary["lessons_created"] = [
            {
                "title": lesson.title,
                "duration": lesson.estimated_minutes,
                "objectives_count": len(lesson.learning_objectives),
                "blocks_count": len(lesson.blocks),
            }
            for lesson in lessons[:3]  # First 3 lessons
        ]

    # Integration summary
    if state.get("integration_plan"):
        plan = state["integration_plan"]
        summary["integration"] = {
            "action": plan.action.value,
            "rationale": plan.rationale[:100] + "..."
            if len(plan.rationale) > 100
            else plan.rationale,
            "affects_existing": len(plan.target_lessons) > 0,
        }

    return summary


async def store_review_record(review_record: Dict[str, Any]) -> None:
    """Store review record in database for human reviewer"""

    try:
        # In a real implementation, this would store to MongoDB
        # For now, we'll log the review record
        logging.info(f"Review record created: {review_record['session_id']}")

        # You would implement database storage here:
        # await db.review_queue.insert_one(review_record)

    except Exception as e:
        logging.error(f"Failed to store review record: {str(e)}")
