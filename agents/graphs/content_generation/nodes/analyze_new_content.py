from agents.graphs.content_generation.models import ContentAnalysisResult
from agents.graphs.content_generation.state import LessonCreationState
from agents.graphs.content_generation.llm import content_analyzer_llm
from agents.graphs.content_generation.prompts.analyze_new_content import (
    ANALYZE_CONTENT_PROMPTS,
)
import logging


async def analyze_new_content(state: LessonCreationState) -> LessonCreationState:
    """Analyze new PDF content chunk to extract key concepts and learning opportunities.

    Returns only the keys it updates: `content_analysis`, `current_step`, and
    appends to `validation_errors` on failure.
    """

    pdf_content = state.get("pdf_content", "")
    page_numbers = state.get("page_numbers", [])
    chunk_id = state.get("chunk_id", "")

    analysis_prompt = ANALYZE_CONTENT_PROMPTS["content_analysis"]

    try:
        messages = analysis_prompt.format_messages(
            pdf_content=pdf_content,  # Limit content length for LLM
            page_numbers=", ".join(map(str, page_numbers)),
            chunk_id=chunk_id,
        )

        content_analysis = await content_analyzer_llm.ainvoke_with_structured_output(
            messages, ContentAnalysisResult
        )

        # Enhance analysis with additional processing
        content_analysis = await enhance_content_analysis(content_analysis, pdf_content)

        logging.info(
            f"Content analysis completed for chunk {chunk_id}: {len(content_analysis.key_concepts)} concepts identified"
        )

        # Return partial state with only modified keys
        return {
            "content_analysis": content_analysis,
            "current_step": "content_analyzed",
        }

    except Exception as e:
        error_msg = f"Content analysis failed: {str(e)}"
        logging.error(f"Content analysis error for chunk {chunk_id}: {str(e)}")
        return {
            "validation_errors": state.get("validation_errors", []) + [error_msg]
        }


async def enhance_content_analysis(
    analysis: ContentAnalysisResult, pdf_content: str
) -> ContentAnalysisResult:
    """Enhance content analysis with additional processing and validation"""

    # Extract additional SEBI-specific terms and concepts
    sebi_keywords = [
        "riskometer",
        "mutual fund",
        "systematic risk",
        "unsystematic risk",
        "diversification",
        "asset allocation",
        "KIM",
        "SID",
        "NAV",
        "expense ratio",
        "exit load",
        "scheme",
        "AMC",
        "trustee",
        "custodian",
        "registrar",
        "algorithmic trading",
        "high frequency trading",
        "market making",
        "arbitrage",
        "derivatives",
        "futures",
        "options",
        "equity",
        "debt",
    ]

    found_keywords = []
    content_lower = pdf_content.lower()

    for keyword in sebi_keywords:
        if keyword.lower() in content_lower:
            found_keywords.append(keyword)

    # Add found keywords to key concepts if not already present
    existing_concepts_lower = [concept.lower() for concept in analysis.key_concepts]
    for keyword in found_keywords:
        if keyword.lower() not in existing_concepts_lower:
            analysis.key_concepts.append(keyword.title())

    # Validate and adjust lesson count estimate
    content_length = len(pdf_content.split())
    if content_length < 500:
        analysis.estimated_lesson_count = max(1, analysis.estimated_lesson_count)
    elif content_length > 3000:
        analysis.estimated_lesson_count = min(5, analysis.estimated_lesson_count)

    # Determine content type if not already set appropriately
    if (
        "regulation" in content_lower
        or "guideline" in content_lower
        or "circular" in content_lower
    ):
        analysis.content_type = "regulatory"
    elif (
        "example" in content_lower
        or "case study" in content_lower
        or "scenario" in content_lower
    ):
        analysis.content_type = "practical"
    elif "investor" in content_lower and "education" in content_lower:
        analysis.content_type = "educational"

    return analysis
