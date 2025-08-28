"""
LangGraph Multi-Agent Orchestrator using Supervisor Pattern.
Coordinates between ContentHarvester, TranslationOrchestrator, SummarizationEngine, 
QuizCraftsman, and SearchMaster agents using proper LangGraph patterns.
"""
from typing import Dict, Any, List, Optional, Literal, Annotated
from datetime import datetime
import asyncio
import logging

from pydantic import Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .base_agent import (
    BaseLangGraphAgent,
    BaseAgentState,
    AgentMetadata,
    AgentStatus,
    LangGraphConfig,
    validate_state_with_pydantic,
    create_validated_state_update,
)
from llm_adapter import LLMAdapter
from content_harvester_langgraph import ContentHarvesterAgent
from translation_orchestrator_langgraph import TranslationOrchestratorAgent
from summarization_engine_langgraph import SummarizationEngineAgent
from quiz_craftsman_langgraph import QuizCraftsmanAgent
from search_master_langgraph import SearchMasterAgent

logger = logging.getLogger(__name__)

# Multi-agent orchestrator state using Pydantic
class OrchestrationState(BaseAgentState):
    """State for the multi-agent orchestrator using Pydantic."""
    workflow_type: str
    workflow_status: AgentStatus = Field(default=AgentStatus.READY)
    
    # Input data
    content_sources: List[Dict[str, Any]] = Field(default_factory=list)
    content_text: str = ""
    target_languages: List[str] = Field(default_factory=list)
    summary_formats: List[str] = Field(default_factory=list)
    quiz_type: str = "multiple_choice"
    quiz_difficulty: str = "medium"
    search_query: str = ""
    search_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent results
    harvested_content: List[Dict[str, Any]] = Field(default_factory=list)
    translations: Dict[str, str] = Field(default_factory=dict)
    summaries: Dict[str, str] = Field(default_factory=dict)
    quiz_questions: List[Dict[str, Any]] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Orchestration metadata
    next_agent: str = ""
    completed_agents: List[str] = Field(default_factory=list)
    agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    workflow_metadata: Dict[str, Any] = Field(default_factory=dict)

class LangGraphOrchestrator(BaseLangGraphAgent):
    """LangGraph-based multi-agent orchestrator using supervisor pattern."""
    
    def __init__(self, llm_adapter: LLMAdapter, config: Optional[LangGraphConfig] = None):
        super().__init__("MultiAgentOrchestrator", config)
        self.llm_adapter = llm_adapter
        
        # Initialize individual agents
        self.content_harvester = ContentHarvesterAgent(llm_adapter, config)
        self.translation_orchestrator = TranslationOrchestratorAgent(llm_adapter, config)
        self.summarization_engine = SummarizationEngineAgent(llm_adapter, config)
        self.quiz_craftsman = QuizCraftsmanAgent(llm_adapter, config)
        self.search_master = SearchMasterAgent(llm_adapter, config)
        
        # Compile all agents
        self._compile_agents()
        
        # Available workflow types
        self.workflow_types = {
            'content_processing': 'Extract and process content from sources',
            'multilingual_content': 'Create multilingual educational content',
            'quiz_generation': 'Generate quizzes from content',
            'search_and_recommend': 'Search content and provide recommendations',
            'full_pipeline': 'Complete content pipeline with all agents'
        }
    
    def _compile_agents(self):
        """Compile all individual agents."""
        try:
            self.content_harvester.compile()
            self.translation_orchestrator.compile()
            self.summarization_engine.compile()
            self.quiz_craftsman.compile()
            self.search_master.compile()
            self.logger.info("All individual agents compiled successfully")
        except Exception as e:
            self.logger.error(f"Failed to compile agents: {e}")
            raise
    
    def build_graph(self) -> StateGraph:
        """Build the multi-agent orchestrator graph using supervisor pattern."""
        
        def supervisor_node(state: OrchestrationState) -> Command[Literal["content_harvester", "translation", "summarization", "quiz_generation", "search", "END"]]:
            """Supervisor node that routes to next agent based on workflow type."""
            try:
                # Validate state using Pydantic model
                validated_state = validate_state_with_pydantic(state, AgentMetadata)
                
                workflow_type = validated_state.get('workflow_type', 'unknown')
                completed_agents = validated_state.get('completed_agents', [])
                messages = validated_state.get('messages', [])
                
                # Determine next agent based on workflow and completion status
                next_agent = self._determine_next_agent_sync(workflow_type, completed_agents)
                
                if next_agent == "END":
                    updates = create_validated_state_update({
                        'workflow_status': AgentStatus.COMPLETED,
                        'messages': messages + [
                            AIMessage(content="Multi-agent workflow completed successfully")
                        ]
                    }, AgentMetadata)
                    
                    return Command(goto="END", update=updates)
                else:
                    agent_mapping = {
                        'content_harvester': 'content_harvester',
                        'translation_orchestrator': 'translation',
                        'summarization_engine': 'summarization', 
                        'quiz_craftsman': 'quiz_generation',
                        'search_master': 'search'
                    }
                    
                    goto_agent = agent_mapping.get(next_agent, "END")
                    
                    updates = create_validated_state_update({
                        'next_agent': next_agent,
                        'messages': messages + [
                            AIMessage(content=f"Supervisor routing to: {next_agent}")
                        ]
                    }, AgentMetadata)
                    
                    return Command(goto=goto_agent, update=updates)
                
            except Exception as e:
                self.logger.error(f"Supervisor routing failed: {e}")
                messages = state.get('messages', [])
                
                updates = create_validated_state_update({
                    'workflow_status': AgentStatus.FAILED,
                    'messages': messages + [
                        AIMessage(content=f"Supervisor error: {str(e)}")
                    ]
                }, AgentMetadata)
                
                return Command(goto="END", update=updates)
        
        async def content_harvester_node(state: OrchestrationState) -> Command[Literal["supervisor"]]:
            """Execute ContentHarvester agent."""
            try:
                content_sources = state.get('content_sources', [])
                
                # Create ContentHarvester state
                harvester_state = {
                    'messages': [HumanMessage(content="Extract content from sources")],
                    'status': AgentStatus.READY,
                    'agent_name': 'ContentHarvester',
                    'content_sources': content_sources
                }
                
                # Execute ContentHarvester
                result_state = await self.content_harvester.execute(harvester_state)
                
                if result_state['status'] == AgentStatus.COMPLETED:
                    extracted_content = result_state.get('extracted_content', [])
                    
                    # Store combined content text for next agents
                    combined_text = '\n\n'.join([
                        item.get('extracted_text', '') for item in extracted_content
                    ])
                    
                    # Update completed agents
                    completed = state.get('completed_agents', [])
                    completed.append('content_harvester')
                    
                    # Store agent result
                    agent_results = state.get('agent_results', {})
                    agent_results['content_harvester'] = {
                        'status': 'completed',
                        'extracted_content_count': len(extracted_content),
                        'combined_text_length': len(combined_text)
                    }
                    
                    return Command(
                        goto="supervisor",
                        update={
                            'harvested_content': extracted_content,
                            'content_text': combined_text,
                            'completed_agents': completed,
                            'agent_results': agent_results,
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"ContentHarvester completed: extracted {len(extracted_content)} items")
                            ]
                        }
                    )
                else:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"ContentHarvester failed: {result_state.get('last_error', 'Unknown error')}")
                            ]
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"ContentHarvester execution failed: {e}")
                return Command(
                    goto="supervisor",
                    update={
                        'messages': state.get('messages', []) + [
                            AIMessage(content=f"ContentHarvester error: {str(e)}")
                        ]
                    }
                )
        
        async def translation_node(state: OrchestrationState) -> Command[Literal["supervisor"]]:
            """Execute TranslationOrchestrator agent."""
            try:
                content_text = state.get('content_text', '')
                target_languages = state.get('target_languages', ['hi', 'ta'])
                
                if not content_text:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content="No content available for translation")
                            ]
                        }
                    )
                
                # Create TranslationOrchestrator state
                translation_state = {
                    'messages': [HumanMessage(content="Translate content to target languages")],
                    'status': AgentStatus.READY,
                    'agent_name': 'TranslationOrchestrator',
                    'source_text': content_text,
                    'source_language': 'en',
                    'target_languages': target_languages
                }
                
                # Execute TranslationOrchestrator
                result_state = await self.translation_orchestrator.execute(translation_state)
                
                if result_state['status'] == AgentStatus.COMPLETED:
                    translations = result_state.get('translations', {})
                    
                    # Update completed agents
                    completed = state.get('completed_agents', [])
                    completed.append('translation_orchestrator')
                    
                    # Store agent result
                    agent_results = state.get('agent_results', {})
                    agent_results['translation_orchestrator'] = {
                        'status': 'completed',
                        'languages_translated': list(translations.keys()),
                        'quality_scores': result_state.get('quality_scores', {})
                    }
                    
                    return Command(
                        goto="supervisor",
                        update={
                            'translations': translations,
                            'completed_agents': completed,
                            'agent_results': agent_results,
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"TranslationOrchestrator completed: translated to {len(translations)} languages")
                            ]
                        }
                    )
                else:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"TranslationOrchestrator failed: {result_state.get('last_error', 'Unknown error')}")
                            ]
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"TranslationOrchestrator execution failed: {e}")
                return Command(
                    goto="supervisor",
                    update={
                        'messages': state.get('messages', []) + [
                            AIMessage(content=f"TranslationOrchestrator error: {str(e)}")
                        ]
                    }
                )
        
        async def summarization_node(state: OrchestrationState) -> Command[Literal["supervisor"]]:
            """Execute SummarizationEngine agent."""
            try:
                content_text = state.get('content_text', '')
                summary_formats = state.get('summary_formats', ['paragraph', 'bullet_points'])
                
                if not content_text:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content="No content available for summarization")
                            ]
                        }
                    )
                
                # Create SummarizationEngine state
                summarization_state = {
                    'messages': [HumanMessage(content="Generate summaries in multiple formats")],
                    'status': AgentStatus.READY,
                    'agent_name': 'SummarizationEngine',
                    'content': content_text,
                    'summary_formats': summary_formats
                }
                
                # Execute SummarizationEngine
                result_state = await self.summarization_engine.execute(summarization_state)
                
                if result_state['status'] == AgentStatus.COMPLETED:
                    summaries = result_state.get('summaries', {})
                    
                    # Update completed agents
                    completed = state.get('completed_agents', [])
                    completed.append('summarization_engine')
                    
                    # Store agent result
                    agent_results = state.get('agent_results', {})
                    agent_results['summarization_engine'] = {
                        'status': 'completed',
                        'summary_formats': list(summaries.keys()),
                        'quality_scores': result_state.get('quality_scores', {}),
                        'key_concepts': result_state.get('key_concepts', [])
                    }
                    
                    return Command(
                        goto="supervisor",
                        update={
                            'summaries': summaries,
                            'completed_agents': completed,
                            'agent_results': agent_results,
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"SummarizationEngine completed: generated {len(summaries)} summary formats")
                            ]
                        }
                    )
                else:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"SummarizationEngine failed: {result_state.get('last_error', 'Unknown error')}")
                            ]
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"SummarizationEngine execution failed: {e}")
                return Command(
                    goto="supervisor",
                    update={
                        'messages': state.get('messages', []) + [
                            AIMessage(content=f"SummarizationEngine error: {str(e)}")
                        ]
                    }
                )
        
        async def quiz_generation_node(state: OrchestrationState) -> Command[Literal["supervisor"]]:
            """Execute QuizCraftsman agent."""
            try:
                content_text = state.get('content_text', '')
                quiz_type = state.get('quiz_type', 'multiple_choice')
                quiz_difficulty = state.get('quiz_difficulty', 'medium')
                
                if not content_text:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content="No content available for quiz generation")
                            ]
                        }
                    )
                
                # Create QuizCraftsman state
                quiz_state = {
                    'messages': [HumanMessage(content="Generate quiz questions from content")],
                    'status': AgentStatus.READY,
                    'agent_name': 'QuizCraftsman',
                    'content': content_text,
                    'quiz_type': quiz_type,
                    'difficulty_level': quiz_difficulty
                }
                
                # Execute QuizCraftsman
                result_state = await self.quiz_craftsman.execute(quiz_state)
                
                if result_state['status'] == AgentStatus.COMPLETED:
                    quiz_questions = result_state.get('questions', [])
                    
                    # Update completed agents
                    completed = state.get('completed_agents', [])
                    completed.append('quiz_craftsman')
                    
                    # Store agent result
                    agent_results = state.get('agent_results', {})
                    agent_results['quiz_craftsman'] = {
                        'status': 'completed',
                        'questions_generated': len(quiz_questions),
                        'quiz_type': quiz_type,
                        'difficulty_level': quiz_difficulty,
                        'quiz_metadata': result_state.get('quiz_metadata', {})
                    }
                    
                    return Command(
                        goto="supervisor",
                        update={
                            'quiz_questions': quiz_questions,
                            'completed_agents': completed,
                            'agent_results': agent_results,
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"QuizCraftsman completed: generated {len(quiz_questions)} questions")
                            ]
                        }
                    )
                else:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"QuizCraftsman failed: {result_state.get('last_error', 'Unknown error')}")
                            ]
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"QuizCraftsman execution failed: {e}")
                return Command(
                    goto="supervisor",
                    update={
                        'messages': state.get('messages', []) + [
                            AIMessage(content=f"QuizCraftsman error: {str(e)}")
                        ]
                    }
                )
        
        async def search_node(state: OrchestrationState) -> Command[Literal["supervisor"]]:
            """Execute SearchMaster agent."""
            try:
                search_query = state.get('search_query', '')
                search_filters = state.get('search_filters', {})
                
                if not search_query:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content="No search query provided")
                            ]
                        }
                    )
                
                # Create SearchMaster state
                search_state = {
                    'messages': [HumanMessage(content="Search for relevant content")],
                    'status': AgentStatus.READY,
                    'agent_name': 'SearchMaster',
                    'query': search_query,
                    'search_filters': search_filters
                }
                
                # Execute SearchMaster
                result_state = await self.search_master.execute(search_state)
                
                if result_state['status'] == AgentStatus.COMPLETED:
                    search_results = result_state.get('search_results', [])
                    recommendations = result_state.get('recommendations', [])
                    
                    # Update completed agents
                    completed = state.get('completed_agents', [])
                    completed.append('search_master')
                    
                    # Store agent result
                    agent_results = state.get('agent_results', {})
                    agent_results['search_master'] = {
                        'status': 'completed',
                        'search_results_count': len(search_results),
                        'recommendations_count': len(recommendations),
                        'query_suggestions': result_state.get('query_suggestions', [])
                    }
                    
                    return Command(
                        goto="supervisor",
                        update={
                            'search_results': search_results,
                            'recommendations': recommendations,
                            'completed_agents': completed,
                            'agent_results': agent_results,
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"SearchMaster completed: found {len(search_results)} results, {len(recommendations)} recommendations")
                            ]
                        }
                    )
                else:
                    return Command(
                        goto="supervisor",
                        update={
                            'messages': state.get('messages', []) + [
                                AIMessage(content=f"SearchMaster failed: {result_state.get('last_error', 'Unknown error')}")
                            ]
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"SearchMaster execution failed: {e}")
                return Command(
                    goto="supervisor",
                    update={
                        'messages': state.get('messages', []) + [
                            AIMessage(content=f"SearchMaster error: {str(e)}")
                        ]
                    }
                )
        
        
        # Build the StateGraph
        workflow = StateGraph(OrchestrationState)
        
        # Add nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("content_harvester", content_harvester_node)
        workflow.add_node("translation", translation_node)
        workflow.add_node("summarization", summarization_node)
        workflow.add_node("quiz_generation", quiz_generation_node)
        workflow.add_node("search", search_node)
        
        # Add edges - supervisor node handles routing via Command pattern
        workflow.add_edge(START, "supervisor")
        
        return workflow
    
    def _determine_next_agent_sync(self, workflow_type: str, completed_agents: List[str], 
                                   state: OrchestrationState) -> str:
        """Determine the next agent to execute based on workflow type and progress."""
        
        workflow_sequences = {
            'content_processing': ['content_harvester'],
            'multilingual_content': ['content_harvester', 'translation_orchestrator', 'summarization_engine'],
            'quiz_generation': ['content_harvester', 'summarization_engine', 'quiz_craftsman'],
            'search_and_recommend': ['search_master'],
            'full_pipeline': ['content_harvester', 'translation_orchestrator', 'summarization_engine', 'quiz_craftsman']
        }
        
        sequence = workflow_sequences.get(workflow_type, [])
        
        if not sequence:
            return "END"
        
        # Find next agent in sequence that hasn't been completed
        for agent in sequence:
            if agent not in completed_agents:
                return agent
        
        return "END"
    
    async def execute_workflow(self, workflow_type: str, **kwargs) -> OrchestrationState:
        """Execute a specific workflow type with given parameters."""
        if not self.compiled_graph:
            raise RuntimeError("Orchestrator graph not compiled")
        
        # Create initial state
        initial_state = OrchestrationState(
            messages=[HumanMessage(content=f"Starting {workflow_type} workflow")],
            workflow_type=workflow_type,
            workflow_status=AgentStatus.READY,
            completed_agents=[],
            agent_results={},
            workflow_metadata={
                'started_at': datetime.now().isoformat(),
                'workflow_type': workflow_type
            },
            **kwargs
        )
        
        # Execute the graph
        final_state = await self.execute(initial_state)
        
        # Update final metadata
        final_state['workflow_metadata']['completed_at'] = datetime.now().isoformat()
        final_state['workflow_metadata']['final_status'] = final_state['workflow_status'].value
        
        return final_state

# Example usage and testing
async def test_langgraph_orchestrator():
    """Test the LangGraph multi-agent orchestrator."""
    from llm_adapter import LLMAdapter, LLMConfig, LLMProvider
    
    # Setup LLM adapter
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    
    llm_adapter = LLMAdapter(config)
    orchestrator = LangGraphOrchestrator(llm_adapter=llm_adapter)
    
    # Compile the orchestrator
    orchestrator.compile()
    
    # Test multilingual content workflow
    final_state = await orchestrator.execute_workflow(
        workflow_type='multilingual_content',
        content_sources=[
            {
                'content_id': 'test001',
                'source': 'https://example.com/investment-guide.pdf',
                'content_type': 'pdf'
            }
        ],
        target_languages=['hi', 'ta'],
        summary_formats=['paragraph', 'bullet_points']
    )
    
    # Print results
    print(f"Workflow Status: {final_state['workflow_status']}")
    print(f"Completed Agents: {final_state.get('completed_agents', [])}")
    print(f"Agent Results: {final_state.get('agent_results', {})}")
    print(f"Messages: {[msg.content for msg in final_state['messages']]}")
    
    return final_state

if __name__ == "__main__":
    # asyncio.run(test_langgraph_orchestrator())
    pass
