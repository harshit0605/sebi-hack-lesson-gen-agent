"""
LLM Client Configuration for SEBI Lesson Creation System
Supports multiple LLM providers with fallback options and specialized configurations
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import backoff
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
import tiktoken


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class TaskType(str, Enum):
    CONTENT_ANALYSIS = "content_analysis"
    LESSON_CREATION = "lesson_creation"
    CONTENT_GENERATION = "content_generation"
    QUIZ_GENERATION = "quiz_generation"
    VOICE_SCRIPT = "voice_script"
    VALIDATION = "validation"
    INTEGRATION_PLANNING = "integration_planning"


@dataclass
class LLMConfig:
    """Configuration for LLM models"""

    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    max_retries: int
    timeout: int
    rate_limit_requests_per_minute: int
    context_window: int


# Model configurations optimized for different tasks
MODEL_CONFIGS = {
    TaskType.CONTENT_ANALYSIS: {
        LLMProvider.OPENAI: LLMConfig(
            model_name="gpt-4o",
            max_tokens=4000,
            temperature=0.3,
            top_p=0.9,
            max_retries=3,
            timeout=60,
            rate_limit_requests_per_minute=50,
            context_window=128000,
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.3,
            top_p=0.9,
            max_retries=3,
            timeout=60,
            rate_limit_requests_per_minute=50,
            context_window=200000,
        ),
    },
    TaskType.LESSON_CREATION: {
        LLMProvider.OPENAI: LLMConfig(
            model_name="gpt-4o",
            max_tokens=6000,
            temperature=0.4,
            top_p=0.9,
            max_retries=3,
            timeout=90,
            rate_limit_requests_per_minute=40,
            context_window=128000,
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=6000,
            temperature=0.4,
            top_p=0.9,
            max_retries=3,
            timeout=90,
            rate_limit_requests_per_minute=40,
            context_window=200000,
        ),
    },
    TaskType.CONTENT_GENERATION: {
        LLMProvider.OPENAI: LLMConfig(
            model_name="gpt-4o",
            max_tokens=3000,
            temperature=0.5,
            top_p=0.9,
            max_retries=3,
            timeout=75,
            rate_limit_requests_per_minute=45,
            context_window=128000,
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.5,
            top_p=0.9,
            max_retries=3,
            timeout=75,
            rate_limit_requests_per_minute=45,
            context_window=200000,
        ),
    },
    TaskType.QUIZ_GENERATION: {
        LLMProvider.OPENAI: LLMConfig(
            model_name="gpt-4o",
            max_tokens=2000,
            temperature=0.6,
            top_p=0.9,
            max_retries=3,
            timeout=45,
            rate_limit_requests_per_minute=60,
            context_window=128000,
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            model_name="claude-3-5-haiku-20241022",
            max_tokens=2000,
            temperature=0.6,
            top_p=0.9,
            max_retries=3,
            timeout=45,
            rate_limit_requests_per_minute=60,
            context_window=200000,
        ),
    },
    TaskType.VOICE_SCRIPT: {
        LLMProvider.OPENAI: LLMConfig(
            model_name="gpt-4o",
            max_tokens=3000,
            temperature=0.7,
            top_p=0.9,
            max_retries=3,
            timeout=60,
            rate_limit_requests_per_minute=50,
            context_window=128000,
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.7,
            top_p=0.9,
            max_retries=3,
            timeout=60,
            rate_limit_requests_per_minute=50,
            context_window=200000,
        ),
    },
    TaskType.VALIDATION: {
        LLMProvider.OPENAI: LLMConfig(
            model_name="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.2,
            top_p=0.8,
            max_retries=2,
            timeout=30,
            rate_limit_requests_per_minute=80,
            context_window=128000,
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            model_name="claude-3-5-haiku-20241022",
            max_tokens=1000,
            temperature=0.2,
            top_p=0.8,
            max_retries=2,
            timeout=30,
            rate_limit_requests_per_minute=80,
            context_window=200000,
        ),
    },
    TaskType.INTEGRATION_PLANNING: {
        LLMProvider.OPENAI: LLMConfig(
            model_name="gpt-4o",
            max_tokens=2000,
            temperature=0.3,
            top_p=0.9,
            max_retries=3,
            timeout=60,
            rate_limit_requests_per_minute=50,
            context_window=128000,
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.3,
            top_p=0.9,
            max_retries=3,
            timeout=60,
            rate_limit_requests_per_minute=50,
            context_window=200000,
        ),
    },
}


class LLMClientManager:
    """Manages LLM clients with fallback support and rate limiting"""

    def __init__(self):
        self.primary_provider = LLMProvider(os.getenv("PRIMARY_LLM_PROVIDER", "openai"))
        self.fallback_provider = LLMProvider(
            os.getenv("FALLBACK_LLM_PROVIDER", "anthropic")
        )
        self.clients = {}
        self.rate_limiters = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM clients for different providers"""

        # OpenAI Client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.clients[LLMProvider.OPENAI] = ChatOpenAI(
                api_key=openai_api_key,
            )
            logging.info("OpenAI client initialized")

        # Anthropic Client
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.clients[LLMProvider.ANTHROPIC] = ChatAnthropic(
                api_key=anthropic_api_key
            )
            logging.info("Anthropic client initialized")

        # Azure OpenAI Client (if configured)
        # azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        # if azure_endpoint and azure_api_key:
        #     from langchain_openai import AzureChatOpenAI

        #     self.clients[LLMProvider.AZURE_OPENAI] = AzureChatOpenAI(
        #         api_key=azure_api_key,
        #         azure_endpoint=azure_endpoint,
        #         api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        #         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
        #     )
        #     logging.info("Azure OpenAI client initialized")

        if not self.clients:
            raise ValueError("No LLM clients could be initialized. Check API keys.")

    def get_client(
        self, task_type: TaskType, provider: Optional[LLMProvider] = None
    ) -> BaseChatModel:
        """Get appropriate LLM client for a specific task"""

        target_provider = provider or self.primary_provider

        # Check if requested provider is available
        if target_provider not in self.clients:
            logging.warning(f"Provider {target_provider} not available, using fallback")
            target_provider = self.fallback_provider

        if target_provider not in self.clients:
            raise ValueError(f"No available LLM provider for task {task_type}")

        # Get client and configure for task
        client = self.clients[target_provider]
        config = MODEL_CONFIGS[task_type][target_provider]

        # Update client configuration
        client.model = config.model_name
        client.max_tokens = config.max_tokens
        client.temperature = config.temperature
        client.model_kwargs = {
            "top_p": config.top_p,
        }
        client.max_retries = config.max_retries
        client.request_timeout = config.timeout

        return client

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        giveup=lambda e: "rate_limit" not in str(e).lower(),
    )
    async def ainvoke_with_fallback(
        self,
        messages: Union[str, List[Dict[str, str]]],
        task_type: TaskType,
        provider: Optional[LLMProvider] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """Invoke LLM with automatic fallback on failure"""

        # Prepare messages
        if isinstance(messages, str):
            formatted_messages = []
            if system_message:
                formatted_messages.append(SystemMessage(content=system_message))
            formatted_messages.append(HumanMessage(content=messages))
        else:
            formatted_messages = [
                SystemMessage(content=msg["content"])
                if msg["role"] == "system"
                else HumanMessage(content=msg["content"])
                for msg in messages
            ]

        # Try primary provider first
        try:
            client = self.get_client(task_type, provider)
            response = await client.ainvoke(formatted_messages)
            return response.content

        except Exception as e:
            logging.error(f"Primary LLM call failed: {str(e)}")

            # Try fallback provider if different from primary
            if (provider or self.primary_provider) != self.fallback_provider:
                try:
                    logging.info("Attempting fallback provider")
                    fallback_client = self.get_client(task_type, self.fallback_provider)
                    response = await fallback_client.ainvoke(formatted_messages)
                    return response.content

                except Exception as fallback_error:
                    logging.error(f"Fallback LLM call failed: {str(fallback_error)}")

            # Re-raise the original exception if all attempts fail
            raise e

    def estimate_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Estimate token count for text"""
        try:
            if "gpt" in model.lower():
                encoding = tiktoken.encoding_for_model(
                    model if model in tiktoken.list_encoding_names() else "gpt-4"
                )
                return len(encoding.encode(text))
            elif "claude" in model.lower():
                # Rough approximation for Claude (4 characters per token)
                return len(text) // 4
            else:
                # Fallback estimation
                return len(text.split())
        except Exception:
            # Fallback to word count approximation
            return len(text.split())

    def validate_context_length(
        self, text: str, task_type: TaskType, provider: Optional[LLMProvider] = None
    ) -> bool:
        """Validate if text fits within context window"""
        target_provider = provider or self.primary_provider
        config = MODEL_CONFIGS[task_type][target_provider]

        estimated_tokens = self.estimate_tokens(text, config.model_name)

        # Reserve tokens for response and system message
        reserved_tokens = config.max_tokens + 500
        available_tokens = config.context_window - reserved_tokens

        return estimated_tokens <= available_tokens


# Global LLM manager instance
llm_manager = LLMClientManager()


# Convenience functions for backward compatibility
async def get_llm_for_task(
    task_type: TaskType, provider: Optional[LLMProvider] = None
) -> BaseChatModel:
    """Get LLM client for specific task"""
    return llm_manager.get_client(task_type, provider)


class LLMClient:
    """Unified LLM client interface for the application"""

    def __init__(self, task_type: TaskType = TaskType.CONTENT_ANALYSIS):
        self.task_type = task_type
        self.manager = llm_manager

    async def ainvoke(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Invoke LLM with automatic provider management"""
        return await self.manager.ainvoke_with_fallback(
            prompt, self.task_type, system_message=system_message
        )

    def with_task_type(self, task_type: TaskType) -> "LLMClient":
        """Create a new client instance with different task type"""
        return LLMClient(task_type)


# Default LLM instance for general use
llm = LLMClient(TaskType.CONTENT_ANALYSIS)

# Task-specific LLM instances
content_analyzer_llm = LLMClient(TaskType.CONTENT_ANALYSIS)
lesson_creator_llm = LLMClient(TaskType.LESSON_CREATION)
content_generator_llm = LLMClient(TaskType.CONTENT_GENERATION)
quiz_generator_llm = LLMClient(TaskType.QUIZ_GENERATION)
voice_script_llm = LLMClient(TaskType.VOICE_SCRIPT)
validator_llm = LLMClient(TaskType.VALIDATION)
integration_planner_llm = LLMClient(TaskType.INTEGRATION_PLANNING)

# System messages for different contexts
SYSTEM_MESSAGES = {
    TaskType.CONTENT_ANALYSIS: """You are an expert educational content analyst specializing in SEBI (Securities and Exchange Board of India) investor education materials. Your role is to analyze PDF content and extract key concepts, learning opportunities, and educational themes that align with SEBI's investor protection and education mandates.

Focus on:
- Financial literacy concepts
- Risk assessment and management
- Regulatory compliance themes
- Investor rights and protections
- Market structure and operations
- Practical applications for Indian investors

Always ground your analysis in SEBI's regulatory framework and investor education objectives.""",
    TaskType.LESSON_CREATION: """You are an expert instructional designer specialized in creating SEBI-compliant financial education content. Your role is to design engaging, pedagogically sound lessons that help Indian investors make informed decisions.

Your lessons should:
- Follow adult learning principles
- Include practical Indian market examples
- Maintain SEBI regulatory compliance
- Use clear, accessible language
- Progress logically from basic to advanced concepts
- Include interactive elements and assessments
- Reference appropriate SEBI sources and guidelines""",
    TaskType.CONTENT_GENERATION: """You are an expert educational content creator specializing in SEBI investor education materials. Generate clear, engaging, and pedagogically effective content that helps Indian investors understand complex financial concepts.

Ensure all content:
- Uses plain language accessible to retail investors
- Includes relevant Indian market examples
- Maintains accuracy with SEBI guidelines
- Provides practical, actionable insights
- Uses appropriate disclaimers and risk warnings
- Supports learning objectives effectively""",
    TaskType.QUIZ_GENERATION: """You are an expert assessment designer for financial education. Create fair, comprehensive, and educationally valuable questions that test understanding of SEBI-related concepts and investor education topics.

Your assessments should:
- Test application, not just memorization
- Include scenario-based questions
- Provide clear, educational rationales
- Use appropriate difficulty levels
- Reference SEBI guidelines in explanations
- Help reinforce key learning objectives""",
    TaskType.VOICE_SCRIPT: """You are an expert educational dialogue designer specializing in Socratic learning methods for financial education. Create engaging voice-based learning experiences that guide students through discovery-based learning about SEBI concepts and investor education.

Your voice scripts should:
- Use conversational, encouraging tone
- Ask probing questions that lead to insights
- Provide appropriate hints and guidance
- Include checkpoints for understanding
- Maintain engagement throughout
- Connect concepts to practical applications""",
    TaskType.VALIDATION: """You are a quality assurance specialist for SEBI investor education content. Your role is to validate content accuracy, compliance, and educational effectiveness.

Validate:
- SEBI guideline alignment and accuracy
- Educational content quality and clarity
- Proper risk disclosures and disclaimers
- Appropriate examples and scenarios
- Learning objective achievement
- Content completeness and coherence""",
    TaskType.INTEGRATION_PLANNING: """You are an expert curriculum designer specializing in educational content integration and learning pathway optimization. Your role is to determine how new educational content should be integrated with existing SEBI investor education materials.

Consider:
- Learning progression and prerequisites
- Content overlap and redundancy
- Optimal lesson sequencing
- Learner cognitive load
- Educational coherence and flow
- Assessment and reinforcement opportunities""",
}


# Utility functions
def get_system_message(task_type: TaskType) -> str:
    """Get appropriate system message for task type"""
    return SYSTEM_MESSAGES.get(task_type, SYSTEM_MESSAGES[TaskType.CONTENT_ANALYSIS])


async def test_llm_connectivity():
    """Test LLM connectivity and functionality"""
    try:
        test_client = LLMClient(TaskType.VALIDATION)
        response = await test_client.ainvoke(
            "Respond with 'LLM connection successful' to confirm connectivity.",
            system_message="You are a system test assistant.",
        )
        logging.info(f"LLM connectivity test: {response}")
        return True
    except Exception as e:
        logging.error(f"LLM connectivity test failed: {str(e)}")
        return False


# Configuration validation
def validate_llm_configuration():
    """Validate LLM configuration and API keys"""
    errors = []
    warnings = []

    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        errors.append("No API keys found for OpenAI or Anthropic")

    # Check model availability
    for task_type in TaskType:
        primary_available = llm_manager.primary_provider in MODEL_CONFIGS[task_type]
        fallback_available = llm_manager.fallback_provider in MODEL_CONFIGS[task_type]

        if not primary_available and not fallback_available:
            errors.append(f"No model configuration for task {task_type}")
        elif not primary_available:
            warnings.append(f"Primary provider not configured for task {task_type}")

    if errors:
        raise ValueError(f"LLM configuration errors: {errors}")

    if warnings:
        logging.warning(f"LLM configuration warnings: {warnings}")

    logging.info("LLM configuration validation passed")


# Initialize and validate on import
try:
    validate_llm_configuration()
    logging.info("LLM client initialized successfully")
except Exception as e:
    logging.error(f"LLM client initialization failed: {str(e)}")
    raise
