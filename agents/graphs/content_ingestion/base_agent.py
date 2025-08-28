"""
LangGraph-based agent implementations using StateGraph patterns.
Provides proper LangGraph integration with state management and workflow orchestration.
"""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
import logging
import time

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState, END
 
from langgraph.types import Command
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for processing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(Enum):
    """Status of agent execution."""

    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


# Pydantic models for validation (standalone, not for hybrid use)
class AgentMetadata(BaseModel):
    """Pydantic model for agent metadata validation."""

    status: AgentStatus = Field(default=AgentStatus.READY)
    agent_name: str
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    error_count: int = Field(default=0, ge=0)
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = Field(default=0, ge=0)

    class Config:
        arbitrary_types_allowed = True


class ContentHarvesterMetadata(AgentMetadata):
    """Pydantic model for content harvester specific fields."""

    content_sources: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_content: List[Dict[str, Any]] = Field(default_factory=list)
    parsing_status: Dict[str, str] = Field(default_factory=dict)
    source_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    downloaded_files: List[Dict[str, Any]] = Field(default_factory=list)


class TranslationMetadata(AgentMetadata):
    """Pydantic model for translation specific fields."""

    source_text: str = ""
    source_language: str = "en"
    target_languages: List[str] = Field(default_factory=list)
    translations: Dict[str, str] = Field(default_factory=dict)
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    cached_translations: Dict[str, str] = Field(default_factory=dict)


class SummarizationMetadata(AgentMetadata):
    """Pydantic model for summarization specific fields."""

    content: str = ""
    summary_formats: List[str] = Field(default_factory=list)
    summaries: Dict[str, str] = Field(default_factory=dict)
    key_concepts: List[str] = Field(default_factory=list)
    quality_scores: Dict[str, float] = Field(default_factory=dict)


class QuizMetadata(AgentMetadata):
    """Pydantic model for quiz generation specific fields."""

    content: str = ""
    difficulty_level: str = "medium"
    quiz_type: str = "multiple_choice"
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    quiz_metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchMetadata(AgentMetadata):
    """Pydantic model for search specific fields."""

    query: str = ""
    search_filters: Dict[str, Any] = Field(default_factory=dict)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


# MessagesState-based schemas (pure LangGraph approach)
class BaseAgentState(MessagesState):
    """Base state structure extending MessagesState for LangGraph agents."""

    status: Optional[AgentStatus] = None
    agent_name: Optional[str] = None
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    error_count: Optional[int] = None
    last_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    retry_count: Optional[int] = None


class ContentHarvesterState(MessagesState):
    """State for content harvesting workflows using MessagesState."""

    status: Optional[AgentStatus] = None
    agent_name: Optional[str] = None
    content_sources: Optional[List[Dict[str, Any]]] = None
    extracted_content: Optional[List[Dict[str, Any]]] = None
    parsing_status: Optional[Dict[str, str]] = None
    source_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    downloaded_files: Optional[List[Dict[str, Any]]] = None
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    error_count: Optional[int] = None
    last_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    retry_count: Optional[int] = None


class TranslationState(MessagesState):
    """State for translation workflows using MessagesState."""

    status: Optional[AgentStatus] = None
    agent_name: Optional[str] = None
    source_text: Optional[str] = None
    source_language: Optional[str] = None
    target_languages: Optional[List[str]] = None
    translations: Optional[Dict[str, str]] = None
    quality_scores: Optional[Dict[str, float]] = None
    cached_translations: Optional[Dict[str, str]] = None
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    error_count: Optional[int] = None
    last_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SummarizationState(MessagesState):
    """State for summarization workflows using MessagesState."""

    status: Optional[AgentStatus] = None
    agent_name: Optional[str] = None
    content: Optional[str] = None
    summary_formats: Optional[List[str]] = None
    summaries: Optional[Dict[str, str]] = None
    key_concepts: Optional[List[str]] = None
    quality_scores: Optional[Dict[str, float]] = None
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    error_count: Optional[int] = None
    last_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QuizState(MessagesState):
    """State for quiz generation workflows using MessagesState."""

    status: Optional[AgentStatus] = None
    agent_name: Optional[str] = None
    content: Optional[str] = None
    difficulty_level: Optional[str] = None
    quiz_type: Optional[str] = None
    questions: Optional[List[Dict[str, Any]]] = None
    quiz_metadata: Optional[Dict[str, Any]] = None
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    error_count: Optional[int] = None
    last_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchState(MessagesState):
    """State for search and recommendation workflows using MessagesState."""

    status: Optional[AgentStatus] = None
    agent_name: Optional[str] = None
    query: Optional[str] = None
    search_filters: Optional[Dict[str, Any]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    error_count: Optional[int] = None
    last_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LangGraphConfig:
    """Configuration for LangGraph agents."""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: int = 300
    enable_streaming: bool = False
    debug_mode: bool = False


class BaseLangGraphAgent:
    """Base class for LangGraph-based agents."""

    def __init__(self, agent_name: str, config: Optional[LangGraphConfig] = None):
        self.agent_name = agent_name
        self.config = config or LangGraphConfig()
        self.logger = logging.getLogger(f"langgraph.{agent_name.lower()}")
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None

        # Performance tracking
        self.start_time = time.time()
        self.total_executions = 0
        self.successful_executions = 0

    def create_initial_state(self, **kwargs) -> BaseAgentState:
        """Create initial state for the agent."""
        return BaseAgentState(
            messages=[],
            status=AgentStatus.READY,
            agent_name=self.agent_name,
            error_count=0,
            retry_count=0,
            metadata=kwargs.get("metadata", {}),
        )

    def build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_graph method")

    def compile(self) -> None:
        """Compile the graph for execution."""
        if not self.graph:
            self.graph = self.build_graph()
        self.compiled_graph = self.graph.compile()
        self.logger.info(f"Compiled LangGraph for agent: {self.agent_name}")

    async def execute(self, initial_state: BaseAgentState) -> BaseAgentState:
        """Execute the compiled graph with given initial state."""
        if not self.compiled_graph:
            raise RuntimeError(f"Graph not compiled for agent: {self.agent_name}")

        self.total_executions += 1
        start_time = time.time()

        try:
            # Update state with processing start
            initial_state["processing_start"] = datetime.now(timezone.utc)
            initial_state["status"] = AgentStatus.PROCESSING

            # Execute the graph
            final_state = await self.compiled_graph.ainvoke(initial_state)

            # Update final state
            final_state["processing_end"] = datetime.now(timezone.utc)
            current_status = final_state.get("status") if hasattr(final_state, "get") else getattr(final_state, "status", None)
            if current_status is None:
                final_state["status"] = AgentStatus.COMPLETED

            if (final_state.get("status") if hasattr(final_state, "get") else getattr(final_state, "status", None)) == AgentStatus.COMPLETED:
                self.successful_executions += 1
            processing_time = time.time() - start_time

            self.logger.info(
                f"Agent {self.agent_name} completed successfully in {processing_time:.2f}s"
            )

            return final_state

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            self.logger.error(
                f"Agent {self.agent_name} failed after {processing_time:.2f}s: {error_msg}"
            )

            # Update state with error info
            initial_state["status"] = AgentStatus.FAILED
            initial_state["processing_end"] = datetime.now(timezone.utc)
            initial_state["last_error"] = error_msg
            initial_state["error_count"] = initial_state.get("error_count", 0) + 1

            return initial_state

    def get_graph_visualization(self) -> bytes:
        """Get graph visualization as PNG bytes."""
        if not self.compiled_graph:
            raise RuntimeError("Graph not compiled")

        try:
            return self.compiled_graph.get_graph().draw_mermaid_png()
        except Exception as e:
            self.logger.warning(f"Could not generate graph visualization: {e}")
            return b""

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        uptime = time.time() - self.start_time
        success_rate = (
            self.successful_executions / max(self.total_executions, 1)
        ) * 100

        return {
            "agent_name": self.agent_name,
            "uptime_seconds": uptime,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate_percent": success_rate,
            "graph_compiled": self.compiled_graph is not None,
        }


# Utility functions for common LangGraph patterns (simplified, no hybrid validation)
def create_error_node(agent_name: str):
    """Create a standard error handling node using Command pattern."""

    def handle_error(state: BaseAgentState) -> Command[None]:
        error_msg = f"Error in {agent_name}: {getattr(state, 'last_error', 'Unknown error') if hasattr(state, 'last_error') else state.get('last_error', 'Unknown error')}"

        messages = (
            getattr(state, "messages", [])
            if hasattr(state, "messages")
            else state.get("messages", [])
        )
        messages.append(AIMessage(content=error_msg))

        updates = {
            "messages": messages,
            "status": AgentStatus.FAILED,
            "processing_end": datetime.now(timezone.utc),
        }

        return Command(goto=END, update=updates)

    return handle_error


def create_completion_node(agent_name: str):
    """Create a standard completion node using Command pattern."""

    def handle_completion(state: BaseAgentState) -> Command[None]:
        completion_msg = f"{agent_name} completed successfully"

        messages = (
            getattr(state, "messages", [])
            if hasattr(state, "messages")
            else state.get("messages", [])
        )
        messages.append(AIMessage(content=completion_msg))

        updates = {
            "messages": messages,
            "status": AgentStatus.COMPLETED,
            "processing_end": datetime.now(timezone.utc),
        }

        return Command(goto=END, update=updates)

    return handle_completion


def should_retry(state: BaseAgentState) -> Literal["retry", "error"]:
    """Standard retry logic for failed operations."""
    retry_count = (
        getattr(state, "retry_count", 0)
        if hasattr(state, "retry_count")
        else state.get("retry_count", 0)
    )
    error_count = (
        getattr(state, "error_count", 0)
        if hasattr(state, "error_count")
        else state.get("error_count", 0)
    )

    if error_count > 0 and retry_count < 3:
        return "retry"
    return "error"


# --- Soft validation helpers for Pydantic models used by agents ---
def _state_to_dict(state: BaseAgentState) -> Dict[str, Any]:
    """Best-effort conversion of a MessagesState-like object to a plain dict.

    Supports both attribute and mapping styles used by LangGraph MessagesState.
    """
    # MessagesState is dict-like; ensure we don't accidentally drop keys
    try:
        # If it's already a mapping
        return dict(state)  # type: ignore[arg-type]
    except Exception:
        # Fallback to attribute access for common fields
        data: Dict[str, Any] = {}
        for key in [
            "messages",
            "status",
            "agent_name",
            "processing_start",
            "processing_end",
            "error_count",
            "last_error",
            "metadata",
            "retry_count",
            # content-harvester specific
            "content_sources",
            "extracted_content",
            "parsing_status",
            "source_metadata",
            "downloaded_files",
        ]:
            if hasattr(state, key):
                data[key] = getattr(state, key)
        return data


def validate_state_with_pydantic(state: BaseAgentState, model_cls: type[BaseModel]) -> Dict[str, Any]:
    """Soft-validate state against a Pydantic model and return a dict view.

    We do not raise on validation errors; instead we log and return the original state
    as a dict to keep the workflow resilient even when optional fields are missing.
    """
    data = _state_to_dict(state)
    try:
        # Attempt to validate; if required fields are missing, this may raise.
        if hasattr(model_cls, "model_validate"):
            model_cls.model_validate(data)  # type: ignore[attr-defined]
        else:
            model_cls.parse_obj(data)  # type: ignore[attr-defined]
        return data
    except Exception as e:
        logger.debug(f"Soft validation warning for {model_cls.__name__}: {e}")
        return data


def create_validated_state_update(updates: Dict[str, Any], model_cls: type[BaseModel]) -> Dict[str, Any]:
    """Soft-validate a prospective state update and return it unchanged on failure.

    This function is intentionally permissive: it returns `updates` even if full
    validation would fail due to missing optional fields in the broader state.
    """
    try:
        if hasattr(model_cls, "model_validate"):
            model_cls.model_validate(updates)  # type: ignore[attr-defined]
        else:
            model_cls.parse_obj(updates)  # type: ignore[attr-defined]
        return updates
    except Exception as e:
        logger.debug(f"State update soft validation warning for {model_cls.__name__}: {e}")
        return updates
