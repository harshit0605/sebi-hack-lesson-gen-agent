import pytest

from agents.graphs.chunking_orchestrator.graph import (
    prepare_next_chunk,
    loop_gate,
)


@pytest.mark.asyncio
async def test_prepare_next_chunk_sets_shared_keys():
    state = {
        "chunks": [
            {
                "content": "Hello World",
                "pages": [1, 2],
                "chunk_id": "chunk_1",
            }
        ],
        "current_index": 0,
    }

    result = await prepare_next_chunk(state)  # type: ignore[arg-type]

    assert result["pdf_content"] == "Hello World"
    assert result["page_numbers"] == [1, 2]
    assert result["chunk_id"] == "chunk_1"
    assert result["last_run_status"].startswith("prepared:")


@pytest.mark.asyncio
async def test_prepare_next_chunk_index_out_of_range():
    state = {
        "chunks": [
            {
                "content": "A",
                "pages": [1],
                "chunk_id": "chunk_1",
            }
        ],
        "current_index": 5,
    }

    result = await prepare_next_chunk(state)  # type: ignore[arg-type]

    assert result["last_run_status"] == "skip_no_chunk"
    assert "pdf_content" not in result
    assert "page_numbers" not in result
    assert "chunk_id" not in result


def test_loop_gate_continue_and_end():
    assert loop_gate({"done": False}) == "continue"  # type: ignore[arg-type]
    assert loop_gate({"done": True}) == "end"  # type: ignore[arg-type]
