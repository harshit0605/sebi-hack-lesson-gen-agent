import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from langgraph.graph import StateGraph, START, END

from agents.graphs.chunking_orchestrator.state import ChunkOrchestrationState
from agents.graphs.content_generation.graph import agent as content_generation_graph
from agents.graphs.content_generation.utils.pdf_chunker import (
    create_test_chunks_from_pdf,
)

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

logger = logging.getLogger(__name__)


async def resolve_pdf_source(state: ChunkOrchestrationState) -> ChunkOrchestrationState:
    """Resolve pdf_source (URL or local path) to a local file path.
    Returns: {"pdf_local_path", "last_run_status"}
    """
    src = state.get("pdf_source", "")
    if not src:
        return {
            "errors": state.get("errors", []) + ["pdf_source missing"],
            "last_run_status": "error",
        }

    # If it's a URL, download to temp file
    if src.lower().startswith("http://") or src.lower().startswith("https://"):
        if aiohttp is None:
            return {
                "errors": state.get("errors", [])
                + ["aiohttp not available to download URL"],
                "last_run_status": "error",
            }
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(src) as resp:
                    if resp.status != 200:
                        return {
                            "errors": state.get("errors", [])
                            + [f"Failed to download PDF: HTTP {resp.status}"],
                            "last_run_status": "error",
                        }
                    with open(tmp_path, "wb") as f:
                        while True:
                            chunk = await resp.content.read(1 << 14)
                            if not chunk:
                                break
                            f.write(chunk)
            logger.info(f"Downloaded PDF to {tmp_path}")
            return {"pdf_local_path": tmp_path, "last_run_status": "downloaded"}
        except Exception as e:  # pragma: no cover
            return {
                "errors": state.get("errors", []) + [f"Download error: {e}"],
                "last_run_status": "error",
            }
    else:
        # Assume local path
        p = Path(src)
        if not p.exists():
            return {
                "errors": state.get("errors", [])
                + [f"Local PDF not found: {p.as_posix()}"],
                "last_run_status": "error",
            }
        return {"pdf_local_path": p.as_posix(), "last_run_status": "resolved"}


async def chunk_pdf(state: ChunkOrchestrationState) -> ChunkOrchestrationState:
    """Chunk the PDF into logical sections using create_test_chunks_from_pdf.
    Returns: {"chunks", "total_chunks", "last_run_status"}
    """
    pdf_path = state.get("pdf_local_path")
    if not pdf_path:
        return {
            "errors": state.get("errors", []) + ["pdf_local_path missing"],
            "last_run_status": "error",
        }

    mode = state.get("chunk_mode", "auto")
    pages_per_chunk = int(state.get("pages_per_chunk", 10) or 10)
    manual_ranges = state.get("manual_ranges", None)

    try:
        documents = create_test_chunks_from_pdf(
            pdf_path, mode=mode, pages_per_chunk=pages_per_chunk, manual_ranges=manual_ranges
        )
        # Convert to simple dicts stored in parent state only
        chunks: List[Dict[str, Any]] = []
        for i, d in enumerate(documents):
            meta = d.metadata or {}
            chunks.append(
                {
                    "content": d.page_content or "",
                    "pages": meta.get("pages", []),
                    "chunk_id": meta.get("chunk_id") or f"chunk_{i+1}",
                    "label": meta.get("label") or meta.get("chapter_info"),
                }
            )
        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "last_run_status": f"chunked:{len(chunks)}",
        }
    except Exception as e:
        return {
            "errors": state.get("errors", []) + [f"Chunking failed: {e}"],
            "last_run_status": "error",
        }


async def init_cursor(state: ChunkOrchestrationState) -> ChunkOrchestrationState:
    total = int(state.get("total_chunks", 0) or 0)
    if total == 0:
        return {
            "errors": state.get("errors", []) + ["No chunks to process"],
            "done": True,
            "last_run_status": "no_chunks",
        }
    return {"current_index": 0, "done": False, "last_run_status": "cursor_init"}


async def prepare_next_chunk(state: ChunkOrchestrationState) -> ChunkOrchestrationState:
    """Set shared keys for the subgraph based on the current chunk.
    This enables adding the subgraph as a node with shared state.
    Returns: {pdf_content, page_numbers, chunk_id, source_url, source_type, last_run_status}
    """
    idx = int(state.get("current_index", 0) or 0)
    chunks = state.get("chunks", []) or []
    if idx >= len(chunks):
        return {"last_run_status": "skip_no_chunk"}

    chunk = chunks[idx]
    return {
        "pdf_content": chunk.get("content", ""),
        "page_numbers": chunk.get("pages", []),
        "chunk_id": chunk.get("chunk_id", f"chunk_{idx+1}"),
        # Propagate original source URL/path and type so subgraph can persist anchors correctly
        "source_url": state.get("pdf_source", ""),
        "source_type": state.get("source_type", "SEBI_PDF"),
        "last_run_status": f"prepared:{chunk.get('chunk_id', f'chunk_{idx+1}')}",
    }


async def advance_cursor(state: ChunkOrchestrationState) -> ChunkOrchestrationState:
    idx = int(state.get("current_index", 0) or 0) + 1
    total = int(state.get("total_chunks", 0) or 0)
    done = idx >= total
    return {"current_index": idx, "done": done, "last_run_status": "advanced"}


def loop_gate(state: ChunkOrchestrationState) -> str:
    return "continue" if not state.get("done", False) else "end"


async def cleanup_after_chunk(state: ChunkOrchestrationState) -> ChunkOrchestrationState:
    """Cleanup large transient keys after subgraph processing.
    Returns: {pdf_content, page_numbers, last_run_status}
    """
    return {"pdf_content": "", "page_numbers": [], "last_run_status": "cleaned"}


def create_chunking_orchestrator_graph():
    builder = StateGraph(ChunkOrchestrationState)

    builder.add_node("resolve_pdf_source", resolve_pdf_source)
    builder.add_node("chunk_pdf", chunk_pdf)
    builder.add_node("init_cursor", init_cursor)
    builder.add_node("prepare_next_chunk", prepare_next_chunk)
    # Insert compiled subgraph as a node with shared keys
    builder.add_node("content_generation", content_generation_graph)
    builder.add_node("cleanup_after_chunk", cleanup_after_chunk)
    builder.add_node("advance_cursor", advance_cursor)

    builder.add_edge(START, "resolve_pdf_source")
    builder.add_edge("resolve_pdf_source", "chunk_pdf")
    builder.add_edge("chunk_pdf", "init_cursor")
    builder.add_edge("init_cursor", "prepare_next_chunk")
    builder.add_edge("prepare_next_chunk", "content_generation")
    builder.add_edge("content_generation", "cleanup_after_chunk")
    builder.add_edge("cleanup_after_chunk", "advance_cursor")

    builder.add_conditional_edges(
        "advance_cursor",
        loop_gate,
        {
            "continue": "prepare_next_chunk",
            "end": END,
        },
    )

    return builder.compile()


orchestrator = create_chunking_orchestrator_graph()
