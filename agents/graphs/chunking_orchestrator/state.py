from typing import TypedDict, List, Dict, Any


class ChunkOrchestrationState(TypedDict, total=False):
    # Input
    pdf_source: str  # local path or http(s) URL
    source_type: str  # e.g., 'SEBI_PDF'
    chunk_mode: str  # 'auto' | 'manual' | 'smart'
    pages_per_chunk: int
    manual_ranges: List[Dict[str, Any]]
    session_id: str

    # Derived chunking data (persisted only in parent graph)
    pdf_local_path: str
    chunks: List[Dict[str, Any]]  # each: {content:str, pages:List[int], chunk_id:str, label?:str}
    current_index: int
    total_chunks: int

    # Shared keys with subgraph (so subgraph can be added as a node)
    pdf_content: str
    page_numbers: List[int]
    chunk_id: str
    source_url: str
    source_type: str

    # Execution tracking
    last_run_status: str
    errors: List[str]
    done: bool
