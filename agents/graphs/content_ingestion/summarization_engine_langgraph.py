"""
LangGraph-based SummarizationEngine Agent using StateGraph patterns.
Provides LLM-based summarization with multiple output formats and quality scoring.
"""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncio
import logging
import re

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from base_agent import (
    BaseLangGraphAgent,
    SummarizationState,
    LangGraphConfig,
    AgentStatus,
)
from llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


class SummarizationEngineAgent(BaseLangGraphAgent):
    """LangGraph-based SummarizationEngine with multiple output formats."""

    def __init__(
        self, llm_adapter: LLMAdapter, config: Optional[LangGraphConfig] = None
    ):
        super().__init__("SummarizationEngine", config)
        self.llm_adapter = llm_adapter
        self.semaphore = asyncio.Semaphore(3)  # Concurrent processing limit

        # Summary format configurations
        self.format_configs = {
            "tweet": {
                "name": "Tweet Format",
                "max_length": 280,
                "style": "concise, engaging, hashtag-friendly",
                "target_audience": "social media users",
            },
            "paragraph": {
                "name": "Paragraph Summary",
                "max_length": 500,
                "style": "comprehensive yet concise",
                "target_audience": "general readers",
            },
            "detailed": {
                "name": "Detailed Summary",
                "max_length": 1500,
                "style": "thorough analysis with key insights",
                "target_audience": "investors and analysts",
            },
            "bullet_points": {
                "name": "Bullet Points",
                "max_length": 800,
                "style": "structured list of key points",
                "target_audience": "quick reference",
            },
            "infographic": {
                "name": "Infographic Description",
                "max_length": 1000,
                "style": "visual elements and data points",
                "target_audience": "visual content creators",
            },
        }

    def build_graph(self) -> StateGraph:
        """Build the SummarizationEngine LangGraph workflow."""

        def prepare_summarization(state: SummarizationState) -> SummarizationState:
            """Prepare summarization request and validate inputs."""
            content = state.get("content", "")
            summary_formats = state.get("summary_formats", [])

            if not content:
                state["messages"].append(
                    AIMessage(content="No content provided for summarization")
                )
                state["status"] = AgentStatus.FAILED
                state["last_error"] = "No content provided"
                return state

            if not summary_formats:
                # Default to paragraph format if none specified
                summary_formats = ["paragraph"]
                state["summary_formats"] = summary_formats

            # Validate format availability
            invalid_formats = [
                fmt for fmt in summary_formats if fmt not in self.format_configs
            ]
            if invalid_formats:
                state["messages"].append(
                    AIMessage(content=f"Invalid summary formats: {invalid_formats}")
                )
                state["status"] = AgentStatus.FAILED
                state["last_error"] = f"Invalid formats: {invalid_formats}"
                return state

            # Initialize tracking dictionaries
            state["summaries"] = {}
            state["key_concepts"] = []
            state["quality_scores"] = {}

            state["messages"].append(
                AIMessage(
                    content=f"Prepared summarization for {len(summary_formats)} formats"
                )
            )

            return state

        async def extract_key_concepts(state: SummarizationState) -> SummarizationState:
            """Extract key concepts and themes from the content."""
            content = state.get("content", "")

            try:
                concepts_prompt = f"""
Analyze the following financial education content and extract key concepts:

{content[:2000]}

Extract:
1. Main financial concepts (max 5)
2. Key investment principles mentioned
3. Important terminology used
4. Risk factors discussed
5. Regulatory aspects (SEBI guidelines, etc.)

Format as a simple list of key concepts, one per line.
"""

                response = await self.llm_adapter.generate(
                    prompt=concepts_prompt, max_tokens=300, temperature=0.3
                )

                # Parse key concepts from response
                concepts_text = response.text.strip()
                key_concepts = [
                    line.strip().lstrip("•-123456789. ")
                    for line in concepts_text.split("\n")
                    if line.strip()
                    and not line.startswith(("Extract:", "Format:", "Analyze:"))
                ]

                state["key_concepts"] = key_concepts[:10]  # Limit to top 10

                state["messages"].append(
                    AIMessage(
                        content=f"Extracted {len(state['key_concepts'])} key concepts"
                    )
                )

            except Exception as e:
                self.logger.warning(f"Failed to extract key concepts: {e}")
                state["key_concepts"] = []

            return state

        async def generate_summaries(state: SummarizationState) -> SummarizationState:
            """Generate summaries in multiple formats concurrently."""
            content = state.get("content", "")
            summary_formats = state.get("summary_formats", [])
            key_concepts = state.get("key_concepts", [])

            try:
                # Generate summaries concurrently
                summary_tasks = [
                    self._generate_format_summary(content, fmt, key_concepts)
                    for fmt in summary_formats
                ]

                summary_results = await asyncio.gather(
                    *summary_tasks, return_exceptions=True
                )

                # Process results
                summaries = {}
                quality_scores = {}

                for i, result in enumerate(summary_results):
                    fmt = summary_formats[i]

                    if isinstance(result, Exception):
                        self.logger.error(f"Summarization failed for {fmt}: {result}")
                        summaries[fmt] = f"Summarization failed: {str(result)}"
                        quality_scores[fmt] = 0.0
                    else:
                        summary_text, quality_score = result
                        summaries[fmt] = summary_text
                        quality_scores[fmt] = quality_score

                state["summaries"] = summaries
                state["quality_scores"] = quality_scores

                successful_summaries = sum(
                    1 for score in quality_scores.values() if score > 0.5
                )

                state["messages"].append(
                    AIMessage(
                        content=f"Generated {successful_summaries}/{len(summary_formats)} high-quality summaries"
                    )
                )

            except Exception as e:
                state["last_error"] = f"Summary generation error: {str(e)}"
                state["status"] = AgentStatus.FAILED
                self.logger.error(f"Summary generation failed: {e}")

            return state

        async def enhance_summaries(state: SummarizationState) -> SummarizationState:
            """Enhance summaries with additional context and formatting."""
            summaries = state.get("summaries", {})
            key_concepts = state.get("key_concepts", [])

            try:
                enhanced_summaries = {}

                for fmt, summary in summaries.items():
                    if state["quality_scores"].get(fmt, 0) < 0.5:
                        continue  # Skip low-quality summaries

                    format_config = self.format_configs[fmt]

                    # Add format-specific enhancements
                    if fmt == "tweet":
                        enhanced = self._enhance_tweet_format(summary, key_concepts)
                    elif fmt == "infographic":
                        enhanced = self._enhance_infographic_format(
                            summary, key_concepts
                        )
                    elif fmt == "bullet_points":
                        enhanced = self._enhance_bullet_format(summary, key_concepts)
                    else:
                        enhanced = summary

                    enhanced_summaries[fmt] = enhanced

                # Update state with enhanced summaries
                state["summaries"].update(enhanced_summaries)

                state["messages"].append(
                    AIMessage(
                        content=f"Enhanced {len(enhanced_summaries)} summary formats"
                    )
                )

            except Exception as e:
                self.logger.warning(f"Summary enhancement failed: {e}")
                # Continue with original summaries

            return state

        def check_completion(
            state: SummarizationState,
        ) -> Literal["completed", "retry", "failed"]:
            """Check if summarization process is complete."""
            summaries = state.get("summaries", {})
            summary_formats = state.get("summary_formats", [])
            quality_scores = state.get("quality_scores", {})
            retry_count = state.get("retry_count", 0)

            if not summaries or not summary_formats:
                return "failed"

            # Check if we have summaries for all formats
            missing_formats = [fmt for fmt in summary_formats if fmt not in summaries]

            if missing_formats:
                if retry_count < 2:
                    return "retry"
                else:
                    return "failed"

            # Check quality threshold
            high_quality_count = sum(
                1 for score in quality_scores.values() if score >= 0.6
            )

            if high_quality_count < len(summary_formats) * 0.5 and retry_count < 2:
                return "retry"

            return "completed"

        def finalize_success(state: SummarizationState) -> SummarizationState:
            """Finalize successful summarization."""
            summaries = state.get("summaries", {})
            quality_scores = state.get("quality_scores", {})

            avg_quality = sum(quality_scores.values()) / max(len(quality_scores), 1)

            # Create metadata
            state["metadata"] = {
                "summary_count": len(summaries),
                "average_quality": avg_quality,
                "key_concepts_count": len(state.get("key_concepts", [])),
                "processing_timestamp": datetime.now().isoformat(),
            }

            state["messages"].append(
                AIMessage(
                    content=f"SummarizationEngine completed successfully. "
                    f"Generated {len(summaries)} summaries with avg quality {avg_quality:.2f}"
                )
            )
            state["status"] = AgentStatus.COMPLETED
            return state

        def handle_retry(state: SummarizationState) -> SummarizationState:
            """Handle retry for failed summarizations."""
            retry_count = state.get("retry_count", 0) + 1
            state["retry_count"] = retry_count

            # Reset low-quality summaries for retry
            quality_scores = state.get("quality_scores", {})
            summaries = state.get("summaries", {})

            retry_formats = []
            for fmt, score in quality_scores.items():
                if score < 0.5:
                    retry_formats.append(fmt)
                    if fmt in summaries:
                        del summaries[fmt]
                    if fmt in quality_scores:
                        del quality_scores[fmt]

            state["messages"].append(
                AIMessage(
                    content=f"Retrying summarization for {len(retry_formats)} formats (attempt {retry_count})"
                )
            )
            state["status"] = AgentStatus.RETRYING
            return state

        def handle_failure(state: SummarizationState) -> SummarizationState:
            """Handle final failure state."""
            error_msg = state.get("last_error", "Summary quality below threshold")

            state["messages"].append(
                AIMessage(content=f"SummarizationEngine failed: {error_msg}")
            )
            state["status"] = AgentStatus.FAILED
            return state

        # Build the StateGraph
        workflow = StateGraph(SummarizationState)

        # Add nodes
        workflow.add_node("prepare_summarization", prepare_summarization)
        workflow.add_node("extract_key_concepts", extract_key_concepts)
        workflow.add_node("generate_summaries", generate_summaries)
        workflow.add_node("enhance_summaries", enhance_summaries)
        workflow.add_node("finalize_success", finalize_success)
        workflow.add_node("handle_retry", handle_retry)
        workflow.add_node("handle_failure", handle_failure)

        # Add edges
        workflow.add_edge(START, "prepare_summarization")
        workflow.add_edge("prepare_summarization", "extract_key_concepts")
        workflow.add_edge("extract_key_concepts", "generate_summaries")
        workflow.add_edge("generate_summaries", "enhance_summaries")

        # Conditional edges based on completion check
        workflow.add_conditional_edges(
            "enhance_summaries",
            check_completion,
            {
                "completed": "finalize_success",
                "retry": "handle_retry",
                "failed": "handle_failure",
            },
        )

        workflow.add_edge("handle_retry", "generate_summaries")
        workflow.add_edge("finalize_success", END)
        workflow.add_edge("handle_failure", END)

        return workflow

    async def _generate_format_summary(
        self, content: str, format_type: str, key_concepts: List[str]
    ) -> tuple[str, float]:
        """Generate summary for a specific format."""
        async with self.semaphore:
            try:
                format_config = self.format_configs[format_type]
                concepts_context = ", ".join(key_concepts[:5]) if key_concepts else ""

                prompt = f"""
Create a {format_config["name"]} of the following financial education content.

Content: {content[:2000]}

Requirements:
- Style: {format_config["style"]}
- Target audience: {format_config["target_audience"]}
- Maximum length: {format_config["max_length"]} characters
- Focus on key concepts: {concepts_context}
- Maintain financial accuracy and compliance
- Use appropriate tone for investor education

{self._get_format_specific_instructions(format_type)}

Provide only the summary without explanations.
"""

                response = await self.llm_adapter.generate(
                    prompt=prompt,
                    max_tokens=min(format_config["max_length"] // 2, 1000),
                    temperature=0.4,
                )

                summary_text = response.text.strip()

                # Calculate quality score
                quality_score = self._calculate_summary_quality(
                    content, summary_text, format_type, format_config
                )

                return summary_text, quality_score

            except Exception as e:
                self.logger.error(f"Summary generation failed for {format_type}: {e}")
                raise e

    def _get_format_specific_instructions(self, format_type: str) -> str:
        """Get format-specific instructions."""
        instructions = {
            "tweet": """
- Include relevant hashtags (#investing #SEBI #financialeducation)
- Use engaging language that encourages learning
- Mention key takeaway or call-to-action
""",
            "paragraph": """
- Start with main topic/concept
- Include 2-3 supporting points
- End with practical application or next steps
""",
            "detailed": """
- Provide comprehensive overview
- Include context and background
- Add regulatory considerations if relevant
- Mention risk factors and considerations
""",
            "bullet_points": """
- Use clear, parallel structure
- Start each point with action verbs
- Include quantitative data where available
- Maximum 7 bullet points
""",
            "infographic": """
- Describe visual elements (charts, icons, data)
- Suggest color coding for different concepts
- Include numerical data and percentages
- Specify layout suggestions (header, sections, footer)
""",
        }
        return instructions.get(format_type, "")

    def _calculate_summary_quality(
        self, original: str, summary: str, format_type: str, config: Dict[str, Any]
    ) -> float:
        """Calculate quality score for summary."""
        if not summary or summary == original:
            return 0.0

        quality_score = 0.5  # Base score

        # Length appropriateness
        max_length = config["max_length"]
        if len(summary) <= max_length:
            quality_score += 0.2
        elif len(summary) <= max_length * 1.1:  # 10% tolerance
            quality_score += 0.1

        # Content preservation (key terms)
        financial_terms = [
            "investment",
            "portfolio",
            "risk",
            "return",
            "market",
            "fund",
            "stock",
            "bond",
        ]
        original_terms = [
            term for term in financial_terms if term.lower() in original.lower()
        ]
        preserved_terms = [
            term for term in original_terms if term.lower() in summary.lower()
        ]

        if original_terms:
            preservation_ratio = len(preserved_terms) / len(original_terms)
            quality_score += preservation_ratio * 0.2

        # Format-specific quality checks
        if format_type == "tweet" and "#" in summary:
            quality_score += 0.1
        elif format_type == "bullet_points" and (
            "•" in summary or summary.count("\n") >= 2
        ):
            quality_score += 0.1
        elif format_type == "detailed" and len(summary.split()) >= 100:
            quality_score += 0.1

        return min(quality_score, 1.0)

    def _enhance_tweet_format(self, summary: str, key_concepts: List[str]) -> str:
        """Enhance tweet format with hashtags and engagement."""
        if "#" not in summary:
            hashtags = ["#InvestorEducation", "#SEBI", "#FinancialLiteracy"]
            if key_concepts:
                concept_tags = [
                    f"#{concept.replace(' ', '').title()}"
                    for concept in key_concepts[:2]
                ]
                hashtags.extend(concept_tags)

            # Add hashtags while respecting character limit
            available_space = 280 - len(summary) - 1  # -1 for space
            hashtag_text = " " + " ".join(hashtags)

            if len(hashtag_text) <= available_space:
                summary += hashtag_text
            else:
                summary += " #InvestorEducation"

        return summary

    def _enhance_infographic_format(self, summary: str, key_concepts: List[str]) -> str:
        """Enhance infographic format with visual descriptions."""
        if "visual" not in summary.lower() and "chart" not in summary.lower():
            visual_suggestions = "\n\nVisual Elements:\n"
            visual_suggestions += "- Header: Main concept with SEBI logo\n"
            visual_suggestions += "- Central graphic: Flow chart or process diagram\n"
            visual_suggestions += (
                "- Data visualization: Key statistics in charts/graphs\n"
            )
            visual_suggestions += "- Footer: Disclaimer and educational resources\n"

            if len(summary) + len(visual_suggestions) <= 1000:
                summary += visual_suggestions

        return summary

    def _enhance_bullet_format(self, summary: str, key_concepts: List[str]) -> str:
        """Enhance bullet point format structure."""
        lines = summary.split("\n")
        enhanced_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith(("•", "-", "*")):
                # Add bullet point if missing
                if not re.match(r"^\d+\.", line):  # Don't modify numbered lists
                    line = f"• {line}"
            enhanced_lines.append(line)

        return "\n".join(enhanced_lines)


# Example usage and testing
async def test_summarization_engine():
    """Test the LangGraph-based SummarizationEngine agent."""
    from llm_adapter import LLMAdapter, LLMConfig, LLMProvider

    # Setup LLM adapter
    config = LLMConfig(
        provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo", api_key="your-api-key"
    )

    llm_adapter = LLMAdapter(config)
    summarizer = SummarizationEngineAgent(llm_adapter=llm_adapter)

    # Compile the graph
    summarizer.compile()

    # Create initial state
    initial_state = SummarizationState(
        messages=[HumanMessage(content="Generate summaries in multiple formats")],
        status=AgentStatus.READY,
        agent_name="SummarizationEngine",
        content="""
Portfolio diversification is one of the most fundamental principles of sound investment management. 
SEBI guidelines emphasize the importance of spreading investments across different asset classes, 
sectors, and geographical regions to minimize risk while optimizing returns.

Key diversification strategies include:

1. Asset Class Diversification: Allocating investments across equities, debt securities, commodities, 
and real estate to reduce correlation risk.

2. Sector Diversification: Within equity investments, spreading holdings across technology, healthcare, 
finance, manufacturing, and other sectors to avoid concentration risk.

3. Geographical Diversification: Including domestic and international investments to benefit from 
different economic cycles and currency movements.

4. Market Cap Diversification: Combining large-cap, mid-cap, and small-cap stocks to balance 
growth potential with stability.

Risk Management Benefits:
- Reduces portfolio volatility
- Protects against sector-specific downturns  
- Provides more consistent long-term returns
- Helps maintain liquidity across different market conditions

Implementation Considerations:
- Regular portfolio rebalancing (quarterly or semi-annually)
- Understanding correlation between different investments
- Monitoring expense ratios and transaction costs
- Aligning diversification with investment goals and risk tolerance

SEBI regulations require mutual funds to maintain minimum diversification standards, 
with limits on single stock exposure and sector concentration to protect retail investors.
""",
        summary_formats=["tweet", "paragraph", "bullet_points", "detailed"],
    )

    # Execute the workflow
    final_state = await summarizer.execute(initial_state)

    # Print results
    print(f"Final Status: {final_state['status']}")
    print(f"Generated Summaries: {list(final_state.get('summaries', {}).keys())}")
    print(f"Quality Scores: {final_state.get('quality_scores', {})}")
    print(f"Key Concepts: {final_state.get('key_concepts', [])}")
    print(f"Messages: {[msg.content for msg in final_state['messages']]}")

    return final_state


if __name__ == "__main__":
    # asyncio.run(test_summarization_engine())
    pass
