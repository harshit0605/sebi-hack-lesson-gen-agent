"""
LangGraph-based ContentHarvester Agent using StateGraph patterns.
Extracts content from PDFs, web pages, and other sources using PyMuPDF integration.
"""
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncio
import logging
import aiohttp
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage

from .base_agent import (
    BaseLangGraphAgent,
    ContentHarvesterState,
    ContentHarvesterMetadata,
    AgentStatus,
    LangGraphConfig,
    create_error_node,
    create_completion_node,
    should_retry,
    validate_state_with_pydantic,
    create_validated_state_update,
)
from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

class ContentHarvesterAgent(BaseLangGraphAgent):
    """LangGraph-based ContentHarvester with PyMuPDF integration."""
    
    def __init__(self, llm_adapter: LLMAdapter, config: Optional[LangGraphConfig] = None):
        super().__init__("ContentHarvester", config)
        self.llm_adapter = llm_adapter
        self.semaphore = asyncio.Semaphore(5)  # Concurrent processing limit
        
    def build_graph(self) -> StateGraph:
        """Build the ContentHarvester LangGraph workflow."""
        
        def validate_sources(state: ContentHarvesterState) -> Command[Literal["download_content", "handle_failure"]]:
            """Validate content sources and prepare for download."""
            # Validate state using Pydantic model
            validated_state = validate_state_with_pydantic(state, ContentHarvesterMetadata)
            
            content_sources = validated_state.get('content_sources', [])
            
            if not content_sources:
                messages = validated_state.get('messages', [])
                messages.append(AIMessage(content="No content sources provided for harvesting"))
                
                updates = create_validated_state_update({
                    'messages': messages,
                    'status': AgentStatus.FAILED,
                    'last_error': "No content sources provided"
                }, ContentHarvesterMetadata)
                
                return Command(goto="handle_failure", update=updates)
            
            # Initialize tracking dictionaries
            parsing_status: Dict[str, str] = {}
            source_metadata: Dict[str, Any] = {}
            unsupported: List[str] = []
            supported_types = {"pdf", "web", "html"}
            
            for source in content_sources:
                content_id = source.get('content_id', 'unknown')
                ctype = source.get('content_type', 'unknown')
                if ctype not in supported_types:
                    parsing_status[content_id] = 'UNSUPPORTED_TYPE'
                    unsupported.append(content_id)
                else:
                    parsing_status[content_id] = 'PENDING'
                source_metadata[content_id] = {
                    'source': source.get('source', ''),
                    'content_type': source.get('content_type', 'unknown'),
                    'added_at': datetime.now().isoformat()
                }
            
            messages = validated_state.get('messages', [])
            messages.append(AIMessage(content=f"Validated {len(content_sources)} content sources for processing"))
            if unsupported:
                messages.append(AIMessage(content=f"Rejected unsupported content types for IDs: {', '.join(unsupported)}"))
            
            updates = create_validated_state_update({
                'messages': messages,
                'parsing_status': parsing_status,
                'source_metadata': source_metadata,
                'extracted_content': [],
                'error_count': validated_state.get('error_count', 0) + len(unsupported)
            }, ContentHarvesterMetadata)
            
            return Command(goto="download_content", update=updates)
        
        async def download_content(state: ContentHarvesterState) -> Command[Literal["extract_content", "handle_failure"]]:
            """Download content from web sources."""
            # Validate state using Pydantic model
            validated_state = validate_state_with_pydantic(state, ContentHarvesterMetadata)
            content_sources = validated_state.get('content_sources', [])
            
            try:
                download_tasks = []
                actionable_sources: List[Dict[str, Any]] = []
                for source in content_sources:
                    if source.get('content_type') in ['pdf', 'web', 'html']:
                        download_tasks.append(self._download_single_source(source))
                        actionable_sources.append(source)
                
                successful_downloads = []
                parsing_status = validated_state.get('parsing_status', {})
                error_count = validated_state.get('error_count', 0)
                
                if download_tasks:
                    downloaded_content = await asyncio.gather(*download_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(downloaded_content):
                        content_id = actionable_sources[i].get('content_id', f'source_{i}')
                        if isinstance(result, Exception):
                            parsing_status[content_id] = 'DOWNLOAD_FAILED'
                            self.logger.error(f"Download failed for {content_id}: {result}")
                            error_count += 1
                        else:
                            successful_downloads.append(result)
                            # mark as downloaded to aid debugging
                            if parsing_status.get(content_id) == 'PENDING':
                                parsing_status[content_id] = 'DOWNLOADED'
                
                messages = validated_state.get('messages', [])
                messages.append(AIMessage(content=f"Downloaded {len(successful_downloads)} files successfully"))
                
                updates = create_validated_state_update({
                    'messages': messages,
                    'downloaded_files': successful_downloads,
                    'parsing_status': parsing_status,
                    'error_count': error_count
                }, ContentHarvesterMetadata)
                
                return Command(goto="extract_content", update=updates)
                
            except Exception as e:
                self.logger.error(f"Content download failed: {e}")
                messages = validated_state.get('messages', [])
                
                updates = create_validated_state_update({
                    'messages': messages,
                    'last_error': f"Download error: {str(e)}",
                    'status': AgentStatus.FAILED,
                    'error_count': validated_state.get('error_count', 0) + 1
                }, ContentHarvesterMetadata)
                
                return Command(goto="handle_failure", update=updates)
        
        async def extract_content(state: ContentHarvesterState) -> Command[Literal["finalize_success", "handle_retry", "handle_failure"]]:
            """Extract content from downloaded files using PyMuPDF and BeautifulSoup."""
            validated_state = validate_state_with_pydantic(state, ContentHarvesterMetadata)
            downloaded_files = validated_state.get('downloaded_files', [])
            parsing_status: Dict[str, str] = validated_state.get('parsing_status', {})
            error_count = validated_state.get('error_count', 0)
            extracted_content: List[Dict[str, Any]] = []

            try:
                extraction_tasks = []
                actionable_items = []
                for item in downloaded_files:
                    cid = item.get('content_id', 'unknown')
                    # Only extract for those not already marked failed/unsupported
                    if parsing_status.get(cid) in (None, 'PENDING', 'DOWNLOADED', 'RETRYING'):
                        extraction_tasks.append(self._extract_single_content(item))
                        actionable_items.append(item)

                if extraction_tasks:
                    extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                    for i, result in enumerate(extraction_results):
                        content_id = actionable_items[i].get('content_id', f'source_{i}')
                        if isinstance(result, Exception):
                            parsing_status[content_id] = 'EXTRACTION_FAILED'
                            self.logger.error(f"Extraction failed for {content_id}: {result}")
                            error_count += 1
                        else:
                            parsing_status[content_id] = 'COMPLETED'
                            extracted_content.append(result)

                messages = validated_state.get('messages', [])

                # Generate summary using LLM
                if extracted_content:
                    summary = await self._generate_content_summary(extracted_content)
                    messages.append(AIMessage(content=summary))

                # Determine next step based on completion status
                actionable_total = sum(1 for v in parsing_status.values() if v != 'UNSUPPORTED_TYPE')
                completed_count = sum(1 for v in parsing_status.values() if v == 'COMPLETED')
                failed_count = sum(1 for v in parsing_status.values() if 'FAILED' in v)
                retry_count = validated_state.get('retry_count', 0)

                if actionable_total > 0 and completed_count == actionable_total:
                    goto_node = "finalize_success"
                elif failed_count > 0 and retry_count < 3:
                    goto_node = "handle_retry"
                else:
                    goto_node = "handle_failure"

                updates = create_validated_state_update({
                    'messages': messages,
                    'extracted_content': extracted_content + validated_state.get('extracted_content', []),
                    'parsing_status': parsing_status,
                    'error_count': error_count
                }, ContentHarvesterMetadata)

                return Command(goto=goto_node, update=updates)

            except Exception as e:
                self.logger.error(f"Content extraction failed: {e}")
                messages = validated_state.get('messages', [])

                updates = create_validated_state_update({
                    'messages': messages,
                    'last_error': f"Extraction error: {str(e)}",
                    'status': AgentStatus.FAILED,
                    'error_count': error_count + 1
                }, ContentHarvesterMetadata)

                return Command(goto="handle_failure", update=updates)
        
        def check_completion(state: ContentHarvesterState) -> Literal["completed", "failed", "retry"]:
            """Check if content harvesting is complete."""
            parsing_status = state.get('parsing_status', {})
            
            if not parsing_status:
                return "failed"
            
            completed_count = sum(1 for status in parsing_status.values() if status == 'COMPLETED')
            failed_count = sum(1 for status in parsing_status.values() if 'FAILED' in status)
            total_count = len(parsing_status)
            
            if completed_count == total_count:
                return "completed"
            elif failed_count > 0 and state.get('retry_count', 0) < 3:
                return "retry"
            else:
                return "failed"
        
        def finalize_success(state: ContentHarvesterState) -> Command[None]:
            """Finalize successful content harvesting."""
            extracted_content = getattr(state, 'extracted_content', []) if hasattr(state, 'extracted_content') else state.get('extracted_content', [])
            extracted_count = len(extracted_content)
            
            messages = getattr(state, 'messages', []) if hasattr(state, 'messages') else state.get('messages', [])
            messages.append(AIMessage(content=f"ContentHarvester completed successfully. Extracted {extracted_count} content items."))
            
            return Command(
                goto=END,
                update={
                    'messages': messages,
                    'status': AgentStatus.COMPLETED
                }
            )
        
        def handle_retry(state: ContentHarvesterState) -> Command[Literal["download_content"]]:
            """Handle retry logic for failed extractions."""
            retry_count = getattr(state, 'retry_count', 0) if hasattr(state, 'retry_count') else state.get('retry_count', 0)
            retry_count += 1
            
            # Reset failed statuses for retry
            parsing_status = getattr(state, 'parsing_status', {}) if hasattr(state, 'parsing_status') else state.get('parsing_status', {})
            for content_id, status in parsing_status.items():
                if 'FAILED' in status:
                    parsing_status[content_id] = 'PENDING'
            
            messages = getattr(state, 'messages', []) if hasattr(state, 'messages') else state.get('messages', [])
            messages.append(AIMessage(content=f"Retrying content extraction (attempt {retry_count})"))
            
            return Command(
                goto="download_content",
                update={
                    'messages': messages,
                    'retry_count': retry_count,
                    'parsing_status': parsing_status,
                    'status': AgentStatus.RETRYING
                }
            )
        
        def handle_failure(state: ContentHarvesterState) -> Command[None]:
            """Handle final failure state."""
            error_msg = getattr(state, 'last_error', 'Unknown error occurred') if hasattr(state, 'last_error') else state.get('last_error', 'Unknown error occurred')
            
            messages = getattr(state, 'messages', []) if hasattr(state, 'messages') else state.get('messages', [])
            messages.append(AIMessage(content=f"ContentHarvester failed: {error_msg}"))
            
            return Command(
                goto=END,
                update={
                    'messages': messages,
                    'status': AgentStatus.FAILED
                }
            )
        
        # Build the StateGraph
        workflow = StateGraph(ContentHarvesterState)
        
        # Add nodes
        workflow.add_node("validate_sources", validate_sources)
        workflow.add_node("download_content", download_content)
        workflow.add_node("extract_content", extract_content)
        workflow.add_node("finalize_success", finalize_success)
        workflow.add_node("handle_retry", handle_retry)
        workflow.add_node("handle_failure", handle_failure)
        
        # Add edges - simplified since Command pattern handles routing
        workflow.add_edge(START, "validate_sources")
        workflow.add_edge("finalize_success", END)
        workflow.add_edge("handle_failure", END)
        
        return workflow
    
    async def _download_single_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Download content from a single source."""
        async with self.semaphore:
            source_url = source.get('source', '')
            content_type = source.get('content_type', 'unknown')
            
            if content_type in ['web', 'html']:
                return await self._download_web_content(source)
            elif content_type == 'pdf':
                return await self._download_pdf_content(source)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
    
    async def _download_web_content(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Download web content."""
        url = source.get('source', '')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return {
                        'content_id': source.get('content_id'),
                        'source': url,
                        'content_type': 'web',
                        'raw_content': content,
                        'downloaded_at': datetime.now().isoformat()
                    }
                else:
                    raise aiohttp.ClientError(f"HTTP {response.status} for URL: {url}")
    
    async def _download_pdf_content(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Download PDF content."""
        url = source.get('source', '')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    return {
                        'content_id': source.get('content_id'),
                        'source': url,
                        'content_type': 'pdf',
                        'raw_content': content,
                        'downloaded_at': datetime.now().isoformat()
                    }
                else:
                    raise aiohttp.ClientError(f"HTTP {response.status} for PDF: {url}")
    
    async def _extract_single_content(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from a single downloaded source dict (has raw_content)."""
        content_type = source.get('content_type', 'unknown')
        if content_type == 'pdf':
            return await self._extract_pdf_content(source)
        elif content_type in ['web', 'html']:
            return await self._extract_web_content(source)
        else:
            raise ValueError(f"Unsupported content type for extraction: {content_type}")
    
    async def _extract_pdf_content(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from PDF bytes using PyMuPDF."""
        content_id = source.get('content_id')
        source_url = source.get('source', '')
        pdf_bytes: Optional[bytes] = source.get('raw_content')

        if pdf_bytes is None and source_url:
            # Fallback: fetch PDF if bytes not present
            async with aiohttp.ClientSession() as session:
                async with session.get(source_url) as response:
                    response.raise_for_status()
                    pdf_bytes = await response.read()

        if not pdf_bytes:
            raise ValueError(f"No PDF bytes available for content_id={content_id}")

        text_parts: List[str] = []
        pages_count = 0
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages_count = doc.page_count
            for page in doc:
                text_parts.append(page.get_text("text"))
        extracted_text = "\n".join(text_parts).strip()

        return {
            'content_id': content_id,
            'source': source_url,
            'content_type': 'pdf',
            'extracted_text': extracted_text,
            'metadata': {
                'pages_count': pages_count,
                'extraction_method': 'pymupdf',
                'extracted_at': datetime.now().isoformat()
            }
        }
    
    async def _extract_web_content(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Extract readable text from HTML using BeautifulSoup. Falls back to fetching if needed."""
        content_id = source.get('content_id')
        source_url = source.get('source', '')
        html: Optional[str] = source.get('raw_content')

        if html is None and source_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(source_url) as response:
                    response.raise_for_status()
                    html = await response.text()

        if not html:
            raise ValueError(f"No HTML content available for content_id={content_id}")

        soup = BeautifulSoup(html, 'html.parser')
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        extracted_text = soup.get_text(separator='\n')
        # Normalize whitespace
        extracted_text = "\n".join(line.strip() for line in extracted_text.splitlines() if line.strip())

        return {
            'content_id': content_id,
            'source': source_url,
            'content_type': 'web',
            'extracted_text': extracted_text,
            'metadata': {
                'extraction_method': 'beautifulsoup',
                'extracted_at': datetime.now().isoformat()
            }
        }
    
    async def _generate_content_summary(self, extracted_content: List[Dict[str, Any]]) -> str:
        """Generate summary of extracted content using LLM."""
        try:
            content_texts = [item.get('extracted_text', '') for item in extracted_content]
            combined_text = '\n\n'.join(content_texts)[:2000]  # Limit for LLM
            
            prompt = f"""
Analyze the following extracted content and provide a brief summary:

{combined_text}

Provide:
1. Main topics covered
2. Content type and source summary  
3. Key financial concepts mentioned
4. Relevance for investor education

Format as a concise summary in 2-3 sentences.
"""
            
            response = await self.llm_adapter.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            return f"Content Summary: {response.text}"
            
        except Exception as e:
            self.logger.warning(f"Failed to generate content summary: {e}")
            return f"Successfully extracted content from {len(extracted_content)} sources"

# Example usage and testing
async def test_content_harvester():
    """Test the LangGraph-based ContentHarvester agent."""
    from llm_adapter import LLMAdapter, LLMConfig, LLMProvider
    
    # Setup LLM adapter
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    
    llm_adapter = LLMAdapter(config)
    harvester = ContentHarvesterAgent(llm_adapter=llm_adapter)
    
    # Compile the graph
    harvester.compile()
    
    # Create initial state
    initial_state = ContentHarvesterState(
        messages=[HumanMessage(content="Extract content from provided sources")],
        status=AgentStatus.READY,
        agent_name="ContentHarvester",
        content_sources=[
            {
                'content_id': 'test001',
                'source': 'https://example.com/financial-guide.pdf',
                'content_type': 'pdf'
            },
            {
                'content_id': 'test002',
                'source': 'https://example.com/investment-basics.html',
                'content_type': 'web'
            }
        ]
    )
    
    # Execute the workflow
    final_state = await harvester.execute(initial_state)
    
    # Print results
    print(f"Final Status: {final_state['status']}")
    print(f"Extracted Content Count: {len(final_state.get('extracted_content', []))}")
    print(f"Messages: {[msg.content for msg in final_state['messages']]}")
    
    return final_state

if __name__ == "__main__":
    # asyncio.run(test_content_harvester())
    pass
