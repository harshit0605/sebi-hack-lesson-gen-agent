import asyncio
import os
import sys
from types import SimpleNamespace
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from prototype.agents.graphs.content_harvester_langgraph import ContentHarvesterAgent
from prototype.agents.graphs.base_agent import AgentStatus, ContentHarvesterState

# Ensure project root is on sys.path for absolute imports
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class StubLLMAdapter:
    """Minimal stub for LLMAdapter with the same generate(prompt=...) API.
    Returns an object with a .text attribute.
    """

    async def generate(self, prompt: str, **kwargs):
        return SimpleNamespace(
            text="Stub summary: extracted content was processed successfully."
        )


async def main() -> Dict[str, Any]:
    # Instantiate agent with stub LLM
    llm_adapter = StubLLMAdapter()
    harvester = ContentHarvesterAgent(llm_adapter=llm_adapter)
    harvester.compile()

    # Define sample sources (1 PDF, 1 Web, and 1 Unsupported to exercise validation)
    sources: List[Dict[str, Any]] = [
        {
            "content_id": "pdf_001",
            "source": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "content_type": "pdf",
        },
        {
            "content_id": "web_001",
            "source": "https://example.com/",
            "content_type": "web",
        },
        {
            "content_id": "bad_001",
            "source": "https://example.com/not-used",
            "content_type": "txt",
        },
    ]

    initial_state = ContentHarvesterState(
        messages=[HumanMessage(content="Harvest the provided sources")],
        status=AgentStatus.READY,
        agent_name="ContentHarvester",
        content_sources=sources,
        error_count=0,
        retry_count=0,
    )

    final_state = await harvester.execute(initial_state)

    # Basic reporting
    status = final_state.get("status")
    print(f"Final Status: {status}")
    print(f"Error Count: {final_state.get('error_count')}")
    extracted = final_state.get("extracted_content", []) or []
    print(f"Extracted Content Count: {len(extracted)}")
    parsing_status = final_state.get("parsing_status", {}) or {}
    print(f"Parsing Status: {parsing_status}")
    msgs = final_state.get("messages", []) or []
    print("Messages:")
    for m in msgs:
        # m is a LangChain message object; access .content safely
        content = getattr(m, "content", str(m))
        print(f"- {content}")

    return final_state


if __name__ == "__main__":
    asyncio.run(main())
