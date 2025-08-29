import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


@pytest.mark.asyncio
async def test_orchestrator_smoke_runs_three_chunks(monkeypatch):
    # Import the module under test
    import agents.graphs.chunking_orchestrator.graph as orch_graph

    # Track calls to the content generation stub
    calls: List[Dict[str, Any]] = []

    async def stub_content_generation(state: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate processing of the chunk without DB/LLM
        calls.append({
            "chunk_id": state.get("chunk_id"),
            "pages": state.get("page_numbers"),
        })
        return {"last_run_status": f"subgraph_ok:{state.get('chunk_id')}"}

    class Doc:
        def __init__(self, content: str, metadata: Dict[str, Any]):
            self.page_content = content
            self.metadata = metadata

    def fake_chunker(pdf_path: str, *, mode: str, pages_per_chunk: int, manual_ranges=None):
        # Always return 3 deterministic chunks
        return [
            Doc("Chunk A", {"pages": [1, 2], "chunk_id": "c1", "label": "L1"}),
            Doc("Chunk B", {"pages": [3], "chunk_id": "c2", "label": "L2"}),
            Doc("Chunk C", {"pages": [4, 5], "chunk_id": "c3", "label": "L3"}),
        ]

    # Patch the subgraph node and PDF chunker used by the orchestrator
    monkeypatch.setattr(orch_graph, "content_generation_graph", stub_content_generation)
    monkeypatch.setattr(orch_graph, "create_test_chunks_from_pdf", fake_chunker)

    # Build graph AFTER patching so the compiled graph uses our stubs
    compiled = orch_graph.create_chunking_orchestrator_graph()

    # Create a temporary file to satisfy resolve_pdf_source local path check
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        temp_pdf = tmp.name
    try:
        initial_state: Dict[str, Any] = {
            "pdf_source": temp_pdf,
            "chunk_mode": "auto",
            "pages_per_chunk": 10,
            "session_id": "test-session-123",
        }

        final_state = await compiled.ainvoke(initial_state)  # type: ignore[attr-defined]

        # Validate orchestrator iterated through all 3 chunks
        assert final_state.get("total_chunks") == 3
        assert final_state.get("current_index") == 3
        assert final_state.get("done") is True
        # Ensure our stub was called exactly once per chunk in order
        assert [c["chunk_id"] for c in calls] == ["c1", "c2", "c3"]
    finally:
        try:
            Path(temp_pdf).unlink(missing_ok=True)
        except Exception:
            pass
