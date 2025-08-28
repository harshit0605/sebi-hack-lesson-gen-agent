"""
LangGraph-based SearchMaster Agent using StateGraph patterns.
Provides semantic search and intelligent recommendations with user preference learning.
"""
from typing import Dict, Any, List, Optional, Literal, Tuple
from datetime import datetime
import asyncio
import logging
import hashlib
import numpy as np

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from base_agent import (
    BaseLangGraphAgent,
    SearchState,
    LangGraphConfig,
    AgentStatus
)
from llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

class SearchMasterAgent(BaseLangGraphAgent):
    """LangGraph-based SearchMaster with semantic search and recommendations."""
    
    def __init__(self, llm_adapter: LLMAdapter, config: Optional[LangGraphConfig] = None):
        super().__init__("SearchMaster", config)
        self.llm_adapter = llm_adapter
        self.semaphore = asyncio.Semaphore(3)  # Concurrent processing limit
        
        # Content index for search (in production, use vector database)
        self.content_index = {}
        self.embedding_cache = {}
        
        # User preference tracking
        self.user_preferences = {}
        
    def build_graph(self) -> StateGraph:
        """Build the SearchMaster LangGraph workflow."""
        
        def prepare_search(state: SearchState) -> SearchState:
            """Prepare search request and validate inputs."""
            query = state.get('query', '')
            search_filters = state.get('search_filters', {})
            
            if not query:
                state['messages'].append(
                    AIMessage(content="No search query provided")
                )
                state['status'] = AgentStatus.FAILED
                state['last_error'] = "No search query provided"
                return state
            
            # Initialize search tracking
            state['search_results'] = []
            state['recommendations'] = []
            state['user_preferences'] = {}
            
            # Extract user ID for personalization
            user_id = search_filters.get('user_id', 'anonymous')
            state['user_id'] = user_id
            
            state['messages'].append(
                AIMessage(content=f"Prepared search for query: '{query[:50]}...'")
            )
            
            return state
        
        async def generate_search_embeddings(state: SearchState) -> SearchState:
            """Generate embeddings for the search query."""
            query = state.get('query', '')
            
            try:
                # Generate embedding for search query
                query_embedding = await self._generate_embedding(query)
                state['query_embedding'] = query_embedding
                
                state['messages'].append(
                    AIMessage(content="Generated search embeddings")
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to generate embeddings: {e}")
                # Continue with text-based search
                state['query_embedding'] = None
            
            return state
        
        async def perform_semantic_search(state: SearchState) -> SearchState:
            """Perform semantic search using embeddings and filters."""
            query = state.get('query', '')
            query_embedding = state.get('query_embedding')
            search_filters = state.get('search_filters', {})
            
            try:
                if query_embedding and self.content_index:
                    # Semantic search with embeddings
                    search_results = await self._semantic_search_with_embeddings(
                        query_embedding, search_filters
                    )
                else:
                    # Fallback to text-based search
                    search_results = await self._text_based_search(query, search_filters)
                
                # Rank and filter results
                ranked_results = self._rank_search_results(search_results, query, search_filters)
                
                state['search_results'] = ranked_results
                
                state['messages'].append(
                    AIMessage(content=f"Found {len(ranked_results)} search results")
                )
                
            except Exception as e:
                state['last_error'] = f"Search error: {str(e)}"
                state['status'] = AgentStatus.FAILED
                self.logger.error(f"Search failed: {e}")
            
            return state
        
        async def generate_recommendations(state: SearchState) -> SearchState:
            """Generate personalized recommendations based on search context."""
            search_results = state.get('search_results', [])
            user_id = state.get('user_id', 'anonymous')
            query = state.get('query', '')
            
            try:
                if search_results:
                    # Generate content-based recommendations
                    content_recommendations = await self._generate_content_recommendations(
                        search_results, query
                    )
                    
                    # Get user-based recommendations if available
                    user_recommendations = self._get_user_based_recommendations(user_id)
                    
                    # Combine and rank recommendations
                    combined_recommendations = self._combine_recommendations(
                        content_recommendations, user_recommendations
                    )
                    
                    state['recommendations'] = combined_recommendations
                    
                    state['messages'].append(
                        AIMessage(content=f"Generated {len(combined_recommendations)} recommendations")
                    )
                else:
                    state['recommendations'] = []
                
            except Exception as e:
                self.logger.warning(f"Recommendation generation failed: {e}")
                state['recommendations'] = []
            
            return state
        
        async def enhance_with_query_suggestions(state: SearchState) -> SearchState:
            """Generate related query suggestions using LLM."""
            query = state.get('query', '')
            search_results = state.get('search_results', [])
            
            try:
                suggestions = await self._generate_query_suggestions(query, search_results)
                state['query_suggestions'] = suggestions
                
                state['messages'].append(
                    AIMessage(content=f"Generated {len(suggestions)} query suggestions")
                )
                
            except Exception as e:
                self.logger.warning(f"Query suggestion generation failed: {e}")
                state['query_suggestions'] = []
            
            return state
        
        def update_user_preferences(state: SearchState) -> SearchState:
            """Update user preferences based on search behavior."""
            user_id = state.get('user_id', 'anonymous')
            search_results = state.get('search_results', [])
            query = state.get('query', '')
            
            if user_id != 'anonymous' and search_results:
                # Extract preferences from search behavior
                categories = []
                content_types = []
                
                for result in search_results[:5]:  # Top 5 results
                    result_categories = result.get('categories', [])
                    categories.extend(result_categories)
                    
                    content_type = result.get('content_type', 'unknown')
                    content_types.append(content_type)
                
                # Update user preference store
                if user_id not in self.user_preferences:
                    self.user_preferences[user_id] = {
                        'categories': [],
                        'content_types': [],
                        'search_history': [],
                        'last_updated': datetime.now().isoformat()
                    }
                
                user_prefs = self.user_preferences[user_id]
                
                # Update categories (keep most recent 20)
                user_prefs['categories'].extend(categories)
                user_prefs['categories'] = list(set(user_prefs['categories']))[-20:]
                
                # Update content types (keep most recent 10)
                user_prefs['content_types'].extend(content_types)
                user_prefs['content_types'] = list(set(user_prefs['content_types']))[-10:]
                
                # Update search history (keep most recent 50)
                user_prefs['search_history'].append({
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'results_count': len(search_results)
                })
                user_prefs['search_history'] = user_prefs['search_history'][-50:]
                
                user_prefs['last_updated'] = datetime.now().isoformat()
                
                state['user_preferences'] = user_prefs
                
            return state
        
        def check_completion(state: SearchState) -> Literal["completed", "retry", "failed"]:
            """Check if search process is complete."""
            search_results = state.get('search_results', [])
            query = state.get('query', '')
            retry_count = state.get('retry_count', 0)
            
            if not query:
                return "failed"
            
            # Check if we have reasonable results
            if len(search_results) > 0:
                return "completed"
            elif retry_count < 2:
                return "retry"
            else:
                return "failed"
        
        def finalize_success(state: SearchState) -> SearchState:
            """Finalize successful search."""
            search_results = state.get('search_results', [])
            recommendations = state.get('recommendations', [])
            query_suggestions = state.get('query_suggestions', [])
            
            # Create search metadata
            state['metadata'] = {
                'search_results_count': len(search_results),
                'recommendations_count': len(recommendations),
                'query_suggestions_count': len(query_suggestions),
                'search_timestamp': datetime.now().isoformat(),
                'user_id': state.get('user_id', 'anonymous')
            }
            
            state['messages'].append(
                AIMessage(content=f"SearchMaster completed successfully. "
                                f"Found {len(search_results)} results, {len(recommendations)} recommendations, "
                                f"and {len(query_suggestions)} query suggestions.")
            )
            state['status'] = AgentStatus.COMPLETED
            return state
        
        def handle_retry(state: SearchState) -> SearchState:
            """Handle retry with expanded search parameters."""
            retry_count = state.get('retry_count', 0) + 1
            state['retry_count'] = retry_count
            
            # Expand search filters for retry
            search_filters = state.get('search_filters', {})
            search_filters['expand_search'] = True
            state['search_filters'] = search_filters
            
            state['messages'].append(
                AIMessage(content=f"Retrying search with expanded parameters (attempt {retry_count})")
            )
            state['status'] = AgentStatus.RETRYING
            return state
        
        def handle_failure(state: SearchState) -> SearchState:
            """Handle search failure."""
            error_msg = state.get('last_error', 'No search results found')
            
            state['messages'].append(
                AIMessage(content=f"SearchMaster failed: {error_msg}")
            )
            state['status'] = AgentStatus.FAILED
            return state
        
        # Build the StateGraph
        workflow = StateGraph(SearchState)
        
        # Add nodes
        workflow.add_node("prepare_search", prepare_search)
        workflow.add_node("generate_search_embeddings", generate_search_embeddings)
        workflow.add_node("perform_semantic_search", perform_semantic_search)
        workflow.add_node("generate_recommendations", generate_recommendations)
        workflow.add_node("enhance_with_query_suggestions", enhance_with_query_suggestions)
        workflow.add_node("update_user_preferences", update_user_preferences)
        workflow.add_node("finalize_success", finalize_success)
        workflow.add_node("handle_retry", handle_retry)
        workflow.add_node("handle_failure", handle_failure)
        
        # Add edges
        workflow.add_edge(START, "prepare_search")
        workflow.add_edge("prepare_search", "generate_search_embeddings")
        workflow.add_edge("generate_search_embeddings", "perform_semantic_search")
        workflow.add_edge("perform_semantic_search", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "enhance_with_query_suggestions")
        workflow.add_edge("enhance_with_query_suggestions", "update_user_preferences")
        
        # Conditional edges based on completion check
        workflow.add_conditional_edges(
            "update_user_preferences",
            check_completion,
            {
                "completed": "finalize_success",
                "retry": "handle_retry",
                "failed": "handle_failure"
            }
        )
        
        workflow.add_edge("handle_retry", "perform_semantic_search")
        workflow.add_edge("finalize_success", END)
        workflow.add_edge("handle_failure", END)
        
        return workflow
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (mock implementation)."""
        # In production, use proper embedding models
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Mock embedding based on text characteristics
        words = text.lower().split()
        embedding = [0.0] * 384  # Standard embedding size
        
        # Simple hash-based pseudo-embedding
        for i, word in enumerate(words[:20]):
            word_hash = hash(word) % 384
            embedding[word_hash] = (embedding[word_hash] + 1.0) / len(words)
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    async def _semantic_search_with_embeddings(self, query_embedding: List[float], 
                                             search_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        results = []
        
        # Mock search results based on content index
        for content_id, content_item in self.content_index.items():
            similarity = self._calculate_cosine_similarity(
                query_embedding, content_item.get('embedding', [])
            )
            
            if similarity > 0.3:  # Minimum similarity threshold
                result = {
                    'content_id': content_id,
                    'title': content_item.get('title', ''),
                    'content_snippet': content_item.get('content', '')[:200],
                    'content_type': content_item.get('content_type', 'unknown'),
                    'categories': content_item.get('categories', []),
                    'similarity_score': similarity,
                    'metadata': content_item.get('metadata', {})
                }
                results.append(result)
        
        return results
    
    async def _text_based_search(self, query: str, search_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback text-based search."""
        # Mock text-based search results
        query_words = set(query.lower().split())
        results = []
        
        # Create mock results based on query
        mock_results = [
            {
                'content_id': 'search_001',
                'title': f'Investment Guide Related to {query}',
                'content_snippet': f'This content discusses {query} and related financial concepts...',
                'content_type': 'guide',
                'categories': ['investment', 'education'],
                'similarity_score': 0.8,
                'metadata': {'source': 'financial_education_db'}
            },
            {
                'content_id': 'search_002', 
                'title': f'SEBI Guidelines on {query}',
                'content_snippet': f'Regulatory aspects of {query} according to SEBI guidelines...',
                'content_type': 'regulatory',
                'categories': ['regulation', 'compliance'],
                'similarity_score': 0.7,
                'metadata': {'source': 'regulatory_db'}
            }
        ]
        
        return mock_results
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _rank_search_results(self, results: List[Dict[str, Any]], query: str, 
                           filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank search results by relevance and filters."""
        # Apply filters
        filtered_results = []
        for result in results:
            if self._passes_filters(result, filters):
                filtered_results.append(result)
        
        # Sort by similarity score
        filtered_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Limit results
        max_results = filters.get('max_results', 10)
        return filtered_results[:max_results]
    
    def _passes_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if result passes search filters."""
        # Language filter
        if 'language' in filters and result.get('language') != filters['language']:
            return False
        
        # Content type filter
        if 'content_types' in filters:
            if result.get('content_type') not in filters['content_types']:
                return False
        
        # Category filter
        if 'categories' in filters:
            result_categories = set(result.get('categories', []))
            filter_categories = set(filters['categories'])
            if not result_categories.intersection(filter_categories):
                return False
        
        return True
    
    async def _generate_content_recommendations(self, search_results: List[Dict[str, Any]], 
                                              query: str) -> List[Dict[str, Any]]:
        """Generate content-based recommendations."""
        recommendations = []
        
        if not search_results:
            return recommendations
        
        # Extract categories from top search results
        categories = set()
        for result in search_results[:3]:
            categories.update(result.get('categories', []))
        
        # Generate recommendations based on categories
        for content_id, content_item in self.content_index.items():
            content_categories = set(content_item.get('categories', []))
            
            # Skip items already in search results
            if content_id in [r.get('content_id') for r in search_results]:
                continue
            
            # Check category overlap
            category_overlap = len(categories.intersection(content_categories))
            if category_overlap > 0:
                recommendation = {
                    'content_id': content_id,
                    'title': content_item.get('title', ''),
                    'content_snippet': content_item.get('content', '')[:150],
                    'content_type': content_item.get('content_type', 'unknown'),
                    'categories': list(content_categories),
                    'recommendation_score': category_overlap / len(categories),
                    'recommendation_reason': f'Related to {query} via shared categories'
                }
                recommendations.append(recommendation)
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        return recommendations[:5]  # Top 5 recommendations
    
    def _get_user_based_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user-based recommendations from preference history."""
        if user_id == 'anonymous' or user_id not in self.user_preferences:
            return []
        
        user_prefs = self.user_preferences[user_id]
        preferred_categories = user_prefs.get('categories', [])
        preferred_types = user_prefs.get('content_types', [])
        
        recommendations = []
        
        # Find content matching user preferences
        for content_id, content_item in self.content_index.items():
            content_categories = set(content_item.get('categories', []))
            preferred_cats = set(preferred_categories)
            
            category_match = len(content_categories.intersection(preferred_cats))
            type_match = 1 if content_item.get('content_type') in preferred_types else 0
            
            if category_match > 0 or type_match > 0:
                recommendation = {
                    'content_id': content_id,
                    'title': content_item.get('title', ''),
                    'content_snippet': content_item.get('content', '')[:150],
                    'recommendation_score': category_match * 0.7 + type_match * 0.3,
                    'recommendation_reason': 'Based on your reading history'
                }
                recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        return recommendations[:3]  # Top 3 user-based recommendations
    
    def _combine_recommendations(self, content_recs: List[Dict[str, Any]], 
                               user_recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine content-based and user-based recommendations."""
        combined = []
        seen_ids = set()
        
        # Add user-based recommendations first (higher priority)
        for rec in user_recs:
            if rec['content_id'] not in seen_ids:
                combined.append(rec)
                seen_ids.add(rec['content_id'])
        
        # Add content-based recommendations
        for rec in content_recs:
            if rec['content_id'] not in seen_ids and len(combined) < 8:
                combined.append(rec)
                seen_ids.add(rec['content_id'])
        
        return combined
    
    async def _generate_query_suggestions(self, query: str, 
                                        search_results: List[Dict[str, Any]]) -> List[str]:
        """Generate related query suggestions using LLM."""
        try:
            # Extract context from search results
            context = ""
            if search_results:
                categories = set()
                for result in search_results[:3]:
                    categories.update(result.get('categories', []))
                context = f"Related topics found: {', '.join(list(categories)[:5])}"
            
            prompt = f"""
Based on the financial education search query: "{query}"
{context}

Generate 3 related search suggestions that would help users find more relevant investor education content.

Focus on:
- SEBI regulations and guidelines
- Investment concepts and strategies
- Risk management and portfolio diversification
- Mutual funds and market instruments
- Financial planning and wealth creation

Format as a simple list:
1. [suggestion 1]
2. [suggestion 2]
3. [suggestion 3]
"""
            
            response = await self.llm_adapter.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            )
            
            # Parse suggestions from response
            suggestions = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '-', '*'))):
                    suggestion = line.split('.', 1)[-1].strip()
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions[:3]
            
        except Exception as e:
            self.logger.error(f"Failed to generate query suggestions: {e}")
            return []

# Example usage and testing
async def test_search_master():
    """Test the LangGraph-based SearchMaster agent."""
    from llm_adapter import LLMAdapter, LLMConfig, LLMProvider
    
    # Setup LLM adapter
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    
    llm_adapter = LLMAdapter(config)
    search_agent = SearchMasterAgent(llm_adapter=llm_adapter)
    
    # Add mock content to index
    search_agent.content_index = {
        'content_001': {
            'title': 'Portfolio Diversification Guide',
            'content': 'Learn about portfolio diversification strategies...',
            'content_type': 'guide',
            'categories': ['investment', 'portfolio', 'risk_management'],
            'embedding': await search_agent._generate_embedding('portfolio diversification risk management'),
            'metadata': {'author': 'SEBI Education', 'date': '2024-01-01'}
        },
        'content_002': {
            'title': 'SIP Investment Basics',
            'content': 'Understanding systematic investment plans...',
            'content_type': 'tutorial',
            'categories': ['sip', 'investment', 'mutual_funds'],
            'embedding': await search_agent._generate_embedding('SIP systematic investment plan mutual funds'),
            'metadata': {'author': 'AMC Guide', 'date': '2024-01-15'}
        }
    }
    
    # Compile the graph
    search_agent.compile()
    
    # Create initial state
    initial_state = SearchState(
        messages=[HumanMessage(content="Search for investment guidance")],
        status=AgentStatus.READY,
        agent_name="SearchMaster",
        query="portfolio diversification risk management",
        search_filters={
            'user_id': 'test_user_123',
            'categories': ['investment', 'portfolio'],
            'max_results': 10
        }
    )
    
    # Execute the workflow
    final_state = await search_agent.execute(initial_state)
    
    # Print results
    print(f"Final Status: {final_state['status']}")
    print(f"Search Results: {len(final_state.get('search_results', []))}")
    print(f"Recommendations: {len(final_state.get('recommendations', []))}")
    print(f"Query Suggestions: {final_state.get('query_suggestions', [])}")
    print(f"Messages: {[msg.content for msg in final_state['messages']]}")
    
    return final_state

if __name__ == "__main__":
    # asyncio.run(test_search_master())
    pass
