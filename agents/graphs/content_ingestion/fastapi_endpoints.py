"""
FastAPI endpoints for LangGraph-based agents and workflows.
Integrates with the existing SEBI Education Backend infrastructure.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import asyncio
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from langgraph_orchestrator import LangGraphOrchestrator
from llm_adapter import LLMAdapter, LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class WorkflowType(str, Enum):
    """Available workflow types."""
    CONTENT_PROCESSING = "content_processing"
    MULTILINGUAL_CONTENT = "multilingual_content"
    QUIZ_GENERATION = "quiz_generation"
    SEARCH_AND_RECOMMEND = "search_and_recommend"
    FULL_PIPELINE = "full_pipeline"

class ContentSource(BaseModel):
    """Content source specification."""
    content_id: str = Field(..., description="Unique identifier for the content")
    source: str = Field(..., description="URL or path to the content source")
    content_type: str = Field(..., description="Type of content: pdf, web, html")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class WorkflowRequest(BaseModel):
    """Base workflow request."""
    workflow_type: WorkflowType
    user_id: Optional[str] = Field(default="anonymous", description="User identifier for personalization")
    priority: Optional[str] = Field(default="medium", description="Processing priority")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ContentProcessingRequest(WorkflowRequest):
    """Content processing workflow request."""
    workflow_type: WorkflowType = WorkflowType.CONTENT_PROCESSING
    content_sources: List[ContentSource] = Field(..., description="List of content sources to process")

class MultilingualContentRequest(WorkflowRequest):
    """Multilingual content creation workflow request."""
    workflow_type: WorkflowType = WorkflowType.MULTILINGUAL_CONTENT
    content_sources: List[ContentSource] = Field(..., description="List of content sources")
    target_languages: List[str] = Field(default=["hi", "ta"], description="Target languages for translation")
    summary_formats: List[str] = Field(default=["paragraph", "bullet_points"], description="Summary formats to generate")

class QuizGenerationRequest(WorkflowRequest):
    """Quiz generation workflow request."""
    workflow_type: WorkflowType = WorkflowType.QUIZ_GENERATION
    content_sources: List[ContentSource] = Field(..., description="Content sources for quiz generation")
    quiz_type: str = Field(default="multiple_choice", description="Type of quiz questions")
    difficulty_level: str = Field(default="medium", description="Difficulty level: easy, medium, hard")

class SearchRequest(WorkflowRequest):
    """Search and recommendation workflow request."""
    workflow_type: WorkflowType = WorkflowType.SEARCH_AND_RECOMMEND
    search_query: str = Field(..., description="Search query")
    search_filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    max_results: int = Field(default=10, description="Maximum number of results")

class FullPipelineRequest(WorkflowRequest):
    """Full pipeline workflow request."""
    workflow_type: WorkflowType = WorkflowType.FULL_PIPELINE
    content_sources: List[ContentSource] = Field(..., description="Content sources")
    target_languages: List[str] = Field(default=["hi", "ta"], description="Target languages")
    summary_formats: List[str] = Field(default=["paragraph", "bullet_points"], description="Summary formats")
    quiz_type: str = Field(default="multiple_choice", description="Quiz type")
    difficulty_level: str = Field(default="medium", description="Quiz difficulty")

class WorkflowResponse(BaseModel):
    """Workflow execution response."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    status: str = Field(..., description="Workflow status")
    workflow_type: str = Field(..., description="Type of workflow executed")
    completed_agents: List[str] = Field(default_factory=list, description="List of completed agents")
    agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Results from each agent")
    execution_time: Optional[float] = Field(None, description="Total execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")

class AgentStatus(BaseModel):
    """Individual agent status."""
    agent_name: str
    status: str
    processing_time: Optional[float] = None
    result_summary: Optional[str] = None
    error_message: Optional[str] = None

class SystemHealthResponse(BaseModel):
    """System health check response."""
    status: str
    agents_health: Dict[str, bool]
    llm_adapter_status: str
    uptime_seconds: float
    total_workflows_executed: int

# Global orchestrator instance
orchestrator: Optional[LangGraphOrchestrator] = None
workflow_executions: Dict[str, Dict[str, Any]] = {}

def get_orchestrator() -> LangGraphOrchestrator:
    """Dependency to get the orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="LangGraph orchestrator not initialized")
    return orchestrator

def initialize_orchestrator():
    """Initialize the LangGraph orchestrator."""
    global orchestrator
    
    try:
        # Configure LLM adapter (in production, use environment variables)
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="your-api-key"  # Replace with actual API key
        )
        
        llm_adapter = LLMAdapter(llm_config)
        orchestrator = LangGraphOrchestrator(llm_adapter=llm_adapter)
        
        logger.info("LangGraph orchestrator initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise

# FastAPI app instance
app = FastAPI(
    title="SEBI Education LangGraph Agents API",
    description="Multi-agent system for financial education content processing using LangGraph",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "workflows",
            "description": "Multi-agent workflow operations"
        },
        {
            "name": "health",
            "description": "System health and monitoring"
        }
    ]
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    initialize_orchestrator()

@app.get("/health", response_model=SystemHealthResponse, tags=["health"])
async def health_check(orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)):
    """Check system health and agent status."""
    try:
        # Check individual agent health
        agents_health = {
            "content_harvester": await orchestrator.content_harvester.health_check(),
            "translation_orchestrator": await orchestrator.translation_orchestrator.health_check(), 
            "summarization_engine": await orchestrator.summarization_engine.health_check(),
            "quiz_craftsman": await orchestrator.quiz_craftsman.health_check(),
            "search_master": await orchestrator.search_master.health_check()
        }
        
        # Check LLM adapter health
        llm_health = await orchestrator.llm_adapter.health_check_all()
        llm_status = "healthy" if any(llm_health.values()) else "unhealthy"
        
        # Get performance stats
        stats = orchestrator.get_performance_stats()
        
        return SystemHealthResponse(
            status="healthy" if all(agents_health.values()) else "degraded",
            agents_health=agents_health,
            llm_adapter_status=llm_status,
            uptime_seconds=stats.get("uptime_seconds", 0),
            total_workflows_executed=stats.get("total_executions", 0)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/workflows/content-processing", response_model=WorkflowResponse, tags=["workflows"])
async def execute_content_processing(
    request: ContentProcessingRequest,
    background_tasks: BackgroundTasks,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
):
    """Execute content processing workflow."""
    try:
        workflow_id = f"content_proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id}"
        
        # Convert content sources to dict format
        content_sources = [source.dict() for source in request.content_sources]
        
        # Execute workflow
        start_time = asyncio.get_event_loop().time()
        final_state = await orchestrator.execute_workflow(
            workflow_type="content_processing",
            content_sources=content_sources
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Store execution record
        workflow_executions[workflow_id] = {
            "request": request.dict(),
            "final_state": final_state,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=final_state.get("workflow_status", "unknown").value,
            workflow_type=request.workflow_type.value,
            completed_agents=final_state.get("completed_agents", []),
            agent_results=final_state.get("agent_results", {}),
            execution_time=execution_time,
            metadata=final_state.get("workflow_metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Content processing workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.post("/workflows/multilingual-content", response_model=WorkflowResponse, tags=["workflows"])
async def execute_multilingual_content(
    request: MultilingualContentRequest,
    background_tasks: BackgroundTasks,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
):
    """Execute multilingual content creation workflow."""
    try:
        workflow_id = f"multilingual_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id}"
        
        # Convert request to workflow parameters
        content_sources = [source.dict() for source in request.content_sources]
        
        start_time = asyncio.get_event_loop().time()
        final_state = await orchestrator.execute_workflow(
            workflow_type="multilingual_content",
            content_sources=content_sources,
            target_languages=request.target_languages,
            summary_formats=request.summary_formats
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        workflow_executions[workflow_id] = {
            "request": request.dict(),
            "final_state": final_state,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=final_state.get("workflow_status", "unknown").value,
            workflow_type=request.workflow_type.value,
            completed_agents=final_state.get("completed_agents", []),
            agent_results=final_state.get("agent_results", {}),
            execution_time=execution_time,
            metadata=final_state.get("workflow_metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Multilingual content workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.post("/workflows/quiz-generation", response_model=WorkflowResponse, tags=["workflows"])
async def execute_quiz_generation(
    request: QuizGenerationRequest,
    background_tasks: BackgroundTasks,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
):
    """Execute quiz generation workflow."""
    try:
        workflow_id = f"quiz_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id}"
        
        content_sources = [source.dict() for source in request.content_sources]
        
        start_time = asyncio.get_event_loop().time()
        final_state = await orchestrator.execute_workflow(
            workflow_type="quiz_generation",
            content_sources=content_sources,
            quiz_type=request.quiz_type,
            quiz_difficulty=request.difficulty_level
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        workflow_executions[workflow_id] = {
            "request": request.dict(),
            "final_state": final_state,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=final_state.get("workflow_status", "unknown").value,
            workflow_type=request.workflow_type.value,
            completed_agents=final_state.get("completed_agents", []),
            agent_results=final_state.get("agent_results", {}),
            execution_time=execution_time,
            metadata=final_state.get("workflow_metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Quiz generation workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.post("/workflows/search", response_model=WorkflowResponse, tags=["workflows"])
async def execute_search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
):
    """Execute search and recommendation workflow."""
    try:
        workflow_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id}"
        
        # Add user_id to search filters
        search_filters = request.search_filters.copy()
        search_filters["user_id"] = request.user_id
        search_filters["max_results"] = request.max_results
        
        start_time = asyncio.get_event_loop().time()
        final_state = await orchestrator.execute_workflow(
            workflow_type="search_and_recommend",
            search_query=request.search_query,
            search_filters=search_filters
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        workflow_executions[workflow_id] = {
            "request": request.dict(),
            "final_state": final_state,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=final_state.get("workflow_status", "unknown").value,
            workflow_type=request.workflow_type.value,
            completed_agents=final_state.get("completed_agents", []),
            agent_results=final_state.get("agent_results", {}),
            execution_time=execution_time,
            metadata=final_state.get("workflow_metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Search workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.post("/workflows/full-pipeline", response_model=WorkflowResponse, tags=["workflows"])
async def execute_full_pipeline(
    request: FullPipelineRequest,
    background_tasks: BackgroundTasks,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
):
    """Execute full content processing pipeline."""
    try:
        workflow_id = f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id}"
        
        content_sources = [source.dict() for source in request.content_sources]
        
        start_time = asyncio.get_event_loop().time()
        final_state = await orchestrator.execute_workflow(
            workflow_type="full_pipeline",
            content_sources=content_sources,
            target_languages=request.target_languages,
            summary_formats=request.summary_formats,
            quiz_type=request.quiz_type,
            quiz_difficulty=request.difficulty_level
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        workflow_executions[workflow_id] = {
            "request": request.dict(),
            "final_state": final_state,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=final_state.get("workflow_status", "unknown").value,
            workflow_type=request.workflow_type.value,
            completed_agents=final_state.get("completed_agents", []),
            agent_results=final_state.get("agent_results", {}),
            execution_time=execution_time,
            metadata=final_state.get("workflow_metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Full pipeline workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.get("/workflows/{workflow_id}", response_model=Dict[str, Any], tags=["workflows"])
async def get_workflow_status(workflow_id: str):
    """Get status and results of a specific workflow execution."""
    if workflow_id not in workflow_executions:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_record = workflow_executions[workflow_id]
    
    return {
        "workflow_id": workflow_id,
        "status": execution_record["final_state"].get("workflow_status", "unknown"),
        "execution_time": execution_record["execution_time"],
        "timestamp": execution_record["timestamp"],
        "request": execution_record["request"],
        "results": {
            "completed_agents": execution_record["final_state"].get("completed_agents", []),
            "agent_results": execution_record["final_state"].get("agent_results", {}),
            "harvested_content": execution_record["final_state"].get("harvested_content", []),
            "translations": execution_record["final_state"].get("translations", {}),
            "summaries": execution_record["final_state"].get("summaries", {}),
            "quiz_questions": execution_record["final_state"].get("quiz_questions", []),
            "search_results": execution_record["final_state"].get("search_results", []),
            "recommendations": execution_record["final_state"].get("recommendations", [])
        },
        "metadata": execution_record["final_state"].get("workflow_metadata", {})
    }

@app.get("/workflows", response_model=List[Dict[str, Any]], tags=["workflows"])
async def list_workflows(limit: int = 50, user_id: Optional[str] = None):
    """List recent workflow executions."""
    executions = []
    
    for workflow_id, record in list(workflow_executions.items())[-limit:]:
        # Filter by user_id if provided
        if user_id and record["request"].get("user_id") != user_id:
            continue
            
        executions.append({
            "workflow_id": workflow_id,
            "workflow_type": record["request"]["workflow_type"],
            "status": record["final_state"].get("workflow_status", "unknown"),
            "execution_time": record["execution_time"],
            "timestamp": record["timestamp"],
            "user_id": record["request"].get("user_id", "anonymous")
        })
    
    return executions

@app.get("/agents/performance", response_model=Dict[str, Any], tags=["health"])
async def get_agent_performance(orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)):
    """Get performance statistics for all agents."""
    try:
        return {
            "orchestrator": orchestrator.get_performance_stats(),
            "content_harvester": orchestrator.content_harvester.get_performance_stats(),
            "translation_orchestrator": orchestrator.translation_orchestrator.get_performance_stats(),
            "summarization_engine": orchestrator.summarization_engine.get_performance_stats(),
            "quiz_craftsman": orchestrator.quiz_craftsman.get_performance_stats(),
            "search_master": orchestrator.search_master.get_performance_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get agent performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance stats")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "fastapi_endpoints:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
