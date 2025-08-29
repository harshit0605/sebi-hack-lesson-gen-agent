import asyncio
import argparse
import logging
from typing import Any, Dict

from agents.graphs.chunking_orchestrator.graph import orchestrator

logging.basicConfig(level=logging.INFO)


async def main(pdf_source: str, mode: str = "auto", pages_per_chunk: int = 10, session_id: str | None = None):
    state: Dict[str, Any] = {
        "pdf_source": pdf_source,
        "chunk_mode": mode,
        "pages_per_chunk": pages_per_chunk,
    }
    if session_id:
        state["session_id"] = session_id

    async for update in orchestrator.astream(
        state,
        stream_mode="updates",
        subgraphs=True,
    ):
        print(update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_source", help="Local path or URL to PDF")
    parser.add_argument("--mode", default="auto", choices=["auto", "manual", "smart"], help="Chunking mode")
    parser.add_argument("--pages-per-chunk", type=int, default=10)
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()
    asyncio.run(main(args.pdf_source, args.mode, args.pages_per_chunk, args.session_id))
