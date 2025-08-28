"""PDF chunking utilities for SEBI document processing.
Supports both automatic equal spacing and manual logical section chunking using LangChain Documents.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import re

from langchain.schema import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Collapse multiple newlines into a single newline
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def extract_pages_pymupdf(pdf_path: Union[str, Path]) -> List[Document]:
    """
    Extract text per page using PyMuPDF and return one LangChain Document per page.
    Page numbers are 1-based in metadata for user-friendly handling.
    """
    docs: List[Document] = []
    pdf_path = str(pdf_path)
    
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            # 1-based page index for UX
            page_num = i + 1
            docs.append(
                Document(
                    page_content=text or "",
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "total_pages": len(doc),
                    },
                )
            )
    return docs


def make_equal_ranges(n_pages: int, span: int) -> List[Tuple[int, int]]:
    """
    Create equal-sized page ranges like [(1,10), (11,20), ...].
    """
    ranges: List[Tuple[int, int]] = []
    start = 1
    while start <= n_pages:
        end = min(start + span - 1, n_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges


def make_documents_for_ranges(page_docs: List[Document], ranges: List[Tuple[int, int]]) -> List[Document]:
    """
    Merge per-page Documents into one Document per range.
    Preserves provenance: start_page, end_page, list of pages, and source.
    """
    # Map page number -> Document for quick lookup
    by_page: Dict[int, Document] = {d.metadata["page"]: d for d in page_docs}
    out: List[Document] = []
    
    for (start, end) in ranges:
        selected = [by_page[p] for p in range(start, end + 1) if p in by_page]
        text = "\n".join(d.page_content for d in selected)
        if not selected:
            continue
            
        meta: Dict[str, Any] = {
            "source": selected[0].metadata.get("source"),
            "start_page": start,
            "end_page": end,
            "pages": [d.metadata["page"] for d in selected],
            "total_pages": selected[0].metadata.get("total_pages"),
            "chunk_id": f"chunk_{start}_{end}",
            "word_count": len(text.split()),
        }
        out.append(Document(page_content=text, metadata=meta))
    
    return out

class SEBIPDFChunker:
    """
    PDF chunker specifically designed for SEBI documents using LangChain Documents.
    Supports automatic equal spacing and manual logical section chunking.
    """
    
    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        self.page_docs: List[Document] = []
        self._load_document()
    
    def _load_document(self):
        """Load the PDF document and extract pages as LangChain Documents."""
        try:
            self.page_docs = extract_pages_pymupdf(self.pdf_path)
            logger.info(f"Loaded PDF with {len(self.page_docs)} pages: {self.pdf_path}")
        except Exception as e:
            logger.error(f"Failed to load PDF {self.pdf_path}: {str(e)}")
            raise
    
    def get_total_pages(self) -> int:
        """Get total number of pages in the document."""
        return len(self.page_docs)
    
    def chunk_by_equal_pages(self, pages_per_chunk: int = 10) -> List[Document]:
        """
        Chunk document into equal page ranges using LangChain Documents.
        
        Args:
            pages_per_chunk: Number of pages per chunk
            
        Returns:
            List of LangChain Document objects
        """
        total_pages = self.get_total_pages()
        ranges = make_equal_ranges(total_pages, pages_per_chunk)
        chunks = make_documents_for_ranges(self.page_docs, ranges)
        
        logger.info(f"Created {len(chunks)} equal chunks with {pages_per_chunk} pages each")
        return chunks
    
    def chunk_by_manual_ranges(self, ranges: List[Dict[str, Any]]) -> List[Document]:
        """
        Chunk document by manually specified page ranges using LangChain Documents.
        
        Args:
            ranges: List of dictionaries with 'start', 'end', and optional 'label'
                   Example: [{'start': 1, 'end': 15, 'label': 'Introduction'},
                            {'start': 16, 'end': 30, 'label': 'Risk Management'}]
        
        Returns:
            List of LangChain Document objects
        """
        total_pages = self.get_total_pages()
        
        # Convert dict ranges to tuple ranges and validate
        valid_ranges = []
        for i, range_dict in enumerate(ranges):
            start = range_dict['start']
            end = range_dict['end']
            label = range_dict.get('label', f"Manual-chunk {i + 1}")
            
            # Validate range
            if start < 1 or end > total_pages:
                logger.warning(f"Range {start}-{end} exceeds document bounds (1-{total_pages})")
                continue
                
            valid_ranges.append((start, end))
        
        chunks = make_documents_for_ranges(self.page_docs, valid_ranges)
        
        # Add labels to metadata
        for i, (chunk, range_dict) in enumerate(zip(chunks, ranges)):
            if i < len(ranges):
                label = range_dict.get('label', f"Manual-chunk {i + 1}")
                chunk.metadata['label'] = label
                chunk.metadata['chapter_info'] = label
        
        logger.info(f"Created {len(chunks)} manual chunks")
        return chunks
    
    def analyze_document_structure(self) -> Dict[str, Any]:
        """
        Analyze the document structure to suggest logical chunking points.
        Looks for headings, table of contents, and section breaks.
        """
        structure_info = {
            'total_pages': self.get_total_pages(),
            'suggested_breaks': [],
            'headings': [],
            'toc_pages': [],
            'chapter_markers': []
        }
        
        # Analyze first 10 pages for TOC and structure
        for page_num in range(1, min(11, self.get_total_pages() + 1)):
            # Get page document
            page_doc = next((doc for doc in self.page_docs if doc.metadata['page'] == page_num), None)
            if not page_doc:
                continue
                
            text = page_doc.page_content
            
            # Look for table of contents indicators
            if self._is_toc_page(text):
                structure_info['toc_pages'].append(page_num)
            
            # Extract potential headings
            headings = self._extract_headings(text, page_num)
            structure_info['headings'].extend(headings)
            
            # Look for chapter markers
            chapters = self._extract_chapter_markers(text, page_num)
            structure_info['chapter_markers'].extend(chapters)
        
        # Generate suggested breaks based on structure
        structure_info['suggested_breaks'] = self._suggest_logical_breaks(structure_info)
        
        return structure_info
    
    def _detect_tables_in_pages(self, page_numbers: List[int]) -> bool:
        """Detect if pages contain tables."""
        try:
            with fitz.open(str(self.pdf_path)) as doc:
                for page_num in page_numbers:
                    if 1 <= page_num <= len(doc):
                        page = doc[page_num - 1]
                        tables = page.find_tables()
                        if tables:
                            return True
            return False
        except Exception:
            return False
    
    def _detect_images_in_pages(self, page_numbers: List[int]) -> bool:
        """Detect if pages contain images."""
        try:
            with fitz.open(str(self.pdf_path)) as doc:
                for page_num in page_numbers:
                    if 1 <= page_num <= len(doc):
                        page = doc[page_num - 1]
                        images = page.get_images()
                        if images:
                            return True
            return False
        except Exception:
            return False
    
    def _is_toc_page(self, text: str) -> bool:
        """Check if page appears to be a table of contents."""
        toc_indicators = [
            'table of contents', 'contents', 'index',
            'chapter', 'section', '..............', '........'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in toc_indicators)
    
    def _extract_headings(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract potential headings from page text."""
        headings = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for numbered headings (1., 1.1, etc.)
            if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', line):
                headings.append({
                    'text': line[:100],  # First 100 chars
                    'page': page_num,
                    'type': 'numbered'
                })
            
            # Look for all-caps headings
            elif len(line) > 5 and line.isupper() and len(line) < 100:
                headings.append({
                    'text': line,
                    'page': page_num,
                    'type': 'caps'
                })
        
        return headings
    
    def _extract_chapter_markers(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract chapter markers from text."""
        markers = []
        
        # Look for "Chapter X" patterns
        chapter_pattern = r'(?i)chapter\s+(\d+|[ivx]+)\s*[:\-]?\s*(.{0,100})'
        matches = re.finditer(chapter_pattern, text)
        
        for match in matches:
            markers.append({
                'chapter_num': match.group(1),
                'title': match.group(2).strip(),
                'page': page_num
            })
        
        return markers
    
    def _suggest_logical_breaks(self, structure_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest logical break points for chunking."""
        breaks = []
        total_pages = structure_info['total_pages']
        
        # If we have chapter markers, use those
        if structure_info['chapter_markers']:
            prev_page = 1
            for i, marker in enumerate(structure_info['chapter_markers']):
                if i > 0:  # Skip first chapter for start point
                    breaks.append({
                        'start': prev_page,
                        'end': marker['page'] - 1,
                        'label': f"Chapter {i}",
                        'confidence': 0.9
                    })
                prev_page = marker['page']
            
            # Add final chapter
            breaks.append({
                'start': prev_page,
                'end': total_pages,
                'label': f"Chapter {len(structure_info['chapter_markers'])}",
                'confidence': 0.9
            })
        
        # Fallback to equal spacing if no clear structure
        else:
            chunk_size = max(10, total_pages // 5)  # Aim for ~5 chunks
            for start in range(1, total_pages + 1, chunk_size):
                end = min(start + chunk_size - 1, total_pages)
                breaks.append({
                    'start': start,
                    'end': end,
                    'label': f"Section {len(breaks) + 1}",
                    'confidence': 0.5
                })
        
        return breaks
    
    def get_sample_text(self, max_pages: int = 3) -> str:
        """Get sample text from first few pages for preview."""
        if not self.page_docs:
            return ""
        
        sample_pages = self.page_docs[:max_pages]
        return "\n\n".join(doc.page_content for doc in sample_pages)

def create_test_chunks_from_pdf(
    pdf_path: Union[str, Path],
    mode: str = "auto",
    pages_per_chunk: int = 10,
    manual_ranges: Optional[List[Dict[str, Any]]] = None
) -> List[Document]:
    """
    Convenience function to create test chunks from a SEBI PDF using LangChain Documents.
    
    Args:
        pdf_path: Path to the PDF file
        mode: "auto" for equal spacing, "manual" for custom ranges, "smart" for structure-based
        pages_per_chunk: Pages per chunk for auto mode
        manual_ranges: List of range dicts for manual mode
    
    Returns:
        List of LangChain Document objects ready for testing
    """
    chunker = SEBIPDFChunker(pdf_path)
    
    if mode == "auto":
        return chunker.chunk_by_equal_pages(pages_per_chunk)
    
    elif mode == "manual":
        if not manual_ranges:
            raise ValueError("Manual ranges required for manual mode")
        return chunker.chunk_by_manual_ranges(manual_ranges)
    
    elif mode == "smart":
        # Analyze structure and use suggested breaks
        structure = chunker.analyze_document_structure()
        suggested_ranges = structure['suggested_breaks']
        
        if suggested_ranges:
            return chunker.chunk_by_manual_ranges(suggested_ranges)
        else:
            logger.warning("No clear structure found, falling back to auto chunking")
            return chunker.chunk_by_equal_pages(pages_per_chunk)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'manual', or 'smart'")
