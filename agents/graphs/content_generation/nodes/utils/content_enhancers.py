"""
Content enhancement utilities for different integration scenarios.
"""


def add_enhancement_context(content_md: str, concept: str) -> str:
    """Add context showing how the concept understanding is being enhanced"""

    enhancement_intro = f"## Enhanced Understanding: {concept.title()}\n\n"
    enhancement_intro += "*Building on foundational knowledge, we now explore deeper dimensions of this concept.*\n\n"

    return enhancement_intro + content_md


def enhance_extension_content(content_md: str, lesson_title: str, concept: str) -> str:
    """Enhance content markdown to better integrate with existing lesson"""

    # Add connecting language if not already present
    connecting_phrases = [
        "Building on our earlier discussion",
        "Extending our understanding",
        "To further our exploration of",
        "Complementing what we've learned",
    ]

    if not any(phrase.lower() in content_md.lower() for phrase in connecting_phrases):
        content_md = f"Building on our earlier discussion in this lesson, {content_md}"

    # Add concept-specific enhancements
    if "riskometer" in concept.lower():
        content_md += "\n\n*This extends our understanding of risk assessment in the context of SEBI's investor protection framework.*"
    elif "diversification" in concept.lower():
        content_md += "\n\n*This concept builds upon the risk management principles we've established.*"

    return content_md


def enhance_example_scenario(scenario_md: str, sebi_theme: str) -> str:
    """Enhance example scenario with theme-specific SEBI context"""

    # Add SEBI-specific enhancements based on theme
    if "mutual fund" in sebi_theme.lower():
        scenario_md += "\n\n**SEBI Protection**: This scenario demonstrates how SEBI's mutual fund regulations protect investor interests through mandatory disclosures and risk labeling."

    elif "risk" in sebi_theme.lower():
        scenario_md += "\n\n**SEBI Framework**: This example shows how SEBI's risk disclosure requirements help investors make informed decisions."

    elif "diversification" in sebi_theme.lower():
        scenario_md += "\n\n**Regulatory Context**: SEBI encourages diversification as a key principle of prudent investing, as reflected in mutual fund portfolio construction guidelines."

    return scenario_md


def enhance_integration_prompt(
    prompt_md: str, original_title: str, extended_title: str
) -> str:
    """Enhance reflection prompt to better facilitate integration"""

    # Add specific integration guidance
    integration_guidance = f"""

**Integration Questions to Consider:**
- How do the new concepts enhance your understanding from the original lesson on {original_title}?
- What connections can you draw between the foundational concepts and the extended material?
- How would you explain this integrated knowledge to a fellow investor?
- What practical steps would you take differently now with this enhanced understanding?

"""

    return prompt_md + integration_guidance


def enhance_merged_concept_content(
    content_md: str, concept: str, lesson_title: str
) -> str:
    """Enhance concept content to fit better in merged lesson context"""

    # Add connecting language for merge context
    if not any(
        phrase in content_md.lower()
        for phrase in ["in addition", "complementing", "building upon"]
    ):
        content_md = (
            f"In addition to our core discussion in {lesson_title}, {content_md}"
        )

    # Add concept-specific contextual notes
    if "compliance" in concept.lower():
        content_md += "\n\n*This compliance aspect reinforces the regulatory framework we've been exploring.*"
    elif "disclosure" in concept.lower():
        content_md += "\n\n*These disclosure requirements work in conjunction with other investor protection measures.*"

    return content_md


def enhance_quiz_rationale_for_merge(
    rationale: str, original_title: str, merged_title: str
) -> str:
    """Enhance quiz rationale to emphasize the integrated nature of merged content"""

    # Add context about how this question tests integrated understanding
    if "sebi" in rationale.lower() or "regulation" in rationale.lower():
        rationale += " This question integrates regulatory concepts from our foundational discussion with enhanced SEBI guidelines."
    elif "risk" in rationale.lower():
        rationale += f" This demonstrates how risk concepts from {original_title} are enhanced by additional SEBI perspectives."
    else:
        rationale += " This question tests your integrated understanding of concepts spanning the enhanced lesson content."

    return rationale
