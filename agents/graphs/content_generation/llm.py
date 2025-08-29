"""
Simplified LLM Client for SEBI Lesson Creation System - MVP Version
"""

import os
import logging
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage
from typing import List, TypeVar, Type, Union
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel | List[BaseModel])


class ModelProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class TaskType(str, Enum):
    CONTENT_ANALYSIS = "content_analysis"
    LESSON_CREATION = "lesson_creation"
    CONTENT_GENERATION = "content_generation"
    QUIZ_GENERATION = "quiz_generation"
    VOICE_SCRIPT = "voice_script"
    VALIDATION = "validation"
    INTEGRATION_PLANNING = "integration_planning"
    GENERATE_LESSONS = "generate_lessons"


class LLMClient:
    """Multi-provider LLM client supporting OpenAI and Gemini models"""

    def __init__(
        self,
        task_type: TaskType = TaskType.CONTENT_ANALYSIS,
        provider: ModelProvider = None,
    ):
        self.task_type = task_type
        self.provider = provider or self._get_default_provider()
        self._client = None

    def _get_default_provider(self) -> ModelProvider:
        """Get default provider based on environment variables"""
        return ModelProvider.GEMINI
        # if os.getenv("OPENAI_API_KEY"):
        #     return ModelProvider.OPENAI
        # elif os.getenv("GOOGLE_API_KEY"):
        #     return ModelProvider.GEMINI
        # else:
        #     # Default to OpenAI if no keys are set
        #     return ModelProvider.OPENAI

    @property
    def client(self) -> Union[ChatOpenAI, ChatGoogleGenerativeAI]:
        """Lazy initialization of LLM client based on provider"""
        if self._client is None:
            config = self._get_config()

            if self.provider == ModelProvider.OPENAI:
                self._client = self._create_openai_client(config)
            elif self.provider == ModelProvider.GEMINI:
                self._client = self._create_gemini_client(config)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        return self._client

    def _create_openai_client(self, config: dict) -> ChatOpenAI:
        """Create OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI provider"
            )

        return ChatOpenAI(
            model=config["openai_model"],
            api_key=api_key,
            # reasoning_effort="high",
        )

    def _create_gemini_client(self, config: dict) -> ChatGoogleGenerativeAI:
        """Create Gemini client"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini provider"
            )

        return ChatGoogleGenerativeAI(
            model=config["gemini_model"],
            google_api_key=api_key,
            disable_streaming=True,
            tags=["nostream"],
        )

    def _get_config(self) -> dict:
        """Get configuration based on task type with both OpenAI and Gemini models"""
        configs = {
            TaskType.CONTENT_ANALYSIS: {
                "openai_model": "gpt-4.1",
                "gemini_model": "gemini-2.5-flash",
            },
            TaskType.LESSON_CREATION: {
                "openai_model": "o4-mini",
                "gemini_model": "gemini-2.5-flash",
            },
            TaskType.CONTENT_GENERATION: {
                "openai_model": "o4-mini",
                "gemini_model": "gemini-2.5-flash",
            },
            TaskType.QUIZ_GENERATION: {
                "openai_model": "o4-mini",
                "gemini_model": "gemini-2.5-flash",
            },
            TaskType.INTEGRATION_PLANNING: {
                "openai_model": "o4-mini",
                "gemini_model": "gemini-2.5-flash",
            },
            TaskType.VALIDATION: {
                "openai_model": "o4-mini",
                "gemini_model": "gemini-2.5-flash",
            },
            TaskType.VOICE_SCRIPT: {
                "openai_model": "o4-mini",
                "gemini_model": "gemini-2.5-flash",
            },
            TaskType.GENERATE_LESSONS: {
                "openai_model": "o4-mini",
                "gemini_model": "gemini-2.5-pro",
            },
        }
        return configs.get(self.task_type, configs[TaskType.CONTENT_ANALYSIS])

    async def ainvoke_with_structured_output(
        self, messages: List[BaseMessage], schema: Type[T]
    ) -> T:
        """Simple LLM invocation with structured output"""
        try:
            response = await self.client.with_structured_output(
                schema=schema,
                method="function_calling",
            ).ainvoke(messages)
            return response
        except Exception as e:
            logging.error(f"LLM call failed: {str(e)}")
            raise

    async def ainvoke(self, messages: List[BaseMessage]) -> str:
        """Simple LLM invocation"""
        try:
            response = await self.client.ainvoke(messages)
            return response
        except Exception as e:
            logging.error(f"LLM call failed: {str(e)}")
            raise


# Task-specific LLM instances (auto-detect provider based on available API keys)
content_analyzer_llm = LLMClient(TaskType.CONTENT_ANALYSIS)
lesson_creator_llm = LLMClient(TaskType.LESSON_CREATION)
content_generator_llm = LLMClient(TaskType.CONTENT_GENERATION)
quiz_generator_llm = LLMClient(TaskType.QUIZ_GENERATION)
voice_script_llm = LLMClient(TaskType.VOICE_SCRIPT)
validator_llm = LLMClient(TaskType.VALIDATION)
integration_planner_llm = LLMClient(TaskType.INTEGRATION_PLANNING)
generate_lessons_llm = LLMClient(TaskType.GENERATE_LESSONS)

# Provider-specific instances for explicit usage
openai_content_generator = LLMClient(TaskType.CONTENT_GENERATION, ModelProvider.OPENAI)
gemini_content_generator = LLMClient(TaskType.CONTENT_GENERATION, ModelProvider.GEMINI)
openai_lesson_creator = LLMClient(TaskType.LESSON_CREATION, ModelProvider.OPENAI)
gemini_lesson_creator = LLMClient(TaskType.LESSON_CREATION, ModelProvider.GEMINI)

# Default LLM instance for general use
llm = content_generator_llm

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


def create_llm_client(task_type: TaskType, provider: ModelProvider = None) -> LLMClient:
    """Factory function to create LLM client with specific provider"""
    return LLMClient(task_type, provider)


def get_available_providers() -> List[ModelProvider]:
    """Get list of available providers based on environment variables"""
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append(ModelProvider.OPENAI)
    if os.getenv("GOOGLE_API_KEY"):
        providers.append(ModelProvider.GEMINI)
    return providers


def get_model_info(provider: ModelProvider, task_type: TaskType) -> dict:
    """Get model information for a specific provider and task type"""
    client = LLMClient(task_type)
    config = client._get_config()

    if provider == ModelProvider.OPENAI:
        return {
            "provider": "OpenAI",
            "model": config["openai_model"],
        }
    elif provider == ModelProvider.GEMINI:
        return {
            "provider": "Google Gemini",
            "model": config["gemini_model"],
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")
