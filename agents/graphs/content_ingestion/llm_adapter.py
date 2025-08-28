"""
LLM Adapter for pluggable model support (OpenAI, Claude, HuggingFace).
Provides a unified interface for different LLM providers.
"""
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: float = 30.0
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}

@dataclass
class LLMResponse:
    """Unified response from LLM providers."""
    text: str
    finish_reason: str
    usage: Dict[str, int]
    provider: LLMProvider
    model: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"llm.{config.provider.value}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text from prompt with streaming."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider")
        return self._client
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                **{k: v for k, v in self.config.custom_params.items() if k not in ['max_tokens', 'temperature']}
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                provider=LLMProvider.OPENAI,
                model=self.config.model_name
            )
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                stream=True,
                **{k: v for k, v in self.config.custom_params.items() if k not in ['max_tokens', 'temperature']}
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
    
    @property 
    def client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("anthropic package is required for Claude provider")
        return self._client
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Claude API."""
        try:
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in self.config.custom_params.items() if k not in ['max_tokens', 'temperature']}
            )
            
            return LLMResponse(
                text=response.content[0].text,
                finish_reason=response.stop_reason,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                provider=LLMProvider.CLAUDE,
                model=self.config.model_name
            )
        except Exception as e:
            self.logger.error(f"Claude API error: {str(e)}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        try:
            stream = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **{k: v for k, v in self.config.custom_params.items() if k not in ['max_tokens', 'temperature']}
            )
            
            async for chunk in stream:
                if chunk.type == 'content_block_delta':
                    yield chunk.delta.text
        except Exception as e:
            self.logger.error(f"Claude streaming error: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check Claude API health."""
        try:
            # Simple test message
            await self.generate("Hi", max_tokens=1)
            return True
        except Exception:
            return False

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Transformers provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._pipeline = None
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.config.model_name,
                    device_map="auto" if self.config.custom_params.get("use_gpu", False) else "cpu",
                    **{k: v for k, v in self.config.custom_params.items() if k != "use_gpu"}
                )
            except ImportError:
                raise ImportError("transformers package is required for HuggingFace provider")
        return self._pipeline
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using HuggingFace model."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.pipeline(
                    prompt,
                    max_length=kwargs.get('max_tokens', self.config.max_tokens) + len(prompt.split()),
                    temperature=kwargs.get('temperature', self.config.temperature),
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
            )
            
            generated_text = response[0]['generated_text']
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return LLMResponse(
                text=generated_text,
                finish_reason="stop",
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split())
                },
                provider=LLMProvider.HUGGINGFACE,
                model=self.config.model_name
            )
        except Exception as e:
            self.logger.error(f"HuggingFace error: {str(e)}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming (simplified for HF)."""
        response = await self.generate(prompt, **kwargs)
        # Simple word-by-word streaming simulation
        words = response.text.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay
    
    async def health_check(self) -> bool:
        """Check HuggingFace model health."""
        try:
            await self.generate("test", max_tokens=1)
            return True
        except Exception:
            return False

class LLMAdapter:
    """Main adapter class for managing different LLM providers."""
    
    def __init__(self, default_config: Optional[LLMConfig] = None):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider_name: Optional[str] = None
        
        if default_config:
            self.add_provider("default", default_config)
            self.default_provider_name = "default"
    
    def add_provider(self, name: str, config: LLMConfig):
        """Add a new LLM provider."""
        if config.provider == LLMProvider.OPENAI:
            provider = OpenAIProvider(config)
        elif config.provider == LLMProvider.CLAUDE:
            provider = ClaudeProvider(config)
        elif config.provider == LLMProvider.HUGGINGFACE:
            provider = HuggingFaceProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        self.providers[name] = provider
        
        if self.default_provider_name is None:
            self.default_provider_name = name
    
    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        """Get provider by name or default."""
        provider_name = name or self.default_provider_name
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")
        return self.providers[provider_name]
    
    async def generate(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate text using specified or default provider."""
        provider = self.get_provider(provider_name)
        return await provider.generate(prompt, **kwargs)
    
    async def generate_stream(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming using specified or default provider."""
        provider = self.get_provider(provider_name)
        async for chunk in provider.generate_stream(prompt, **kwargs):
            yield chunk
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all providers."""
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {str(e)}")
                results[name] = False
        return results

# Predefined configurations for common models
COMMON_CONFIGS = {
    "gpt-4": LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        max_tokens=2000,
        temperature=0.7
    ),
    "gpt-3.5-turbo": LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        max_tokens=1500,
        temperature=0.7
    ),
    "claude-3-sonnet": LLMConfig(
        provider=LLMProvider.CLAUDE,
        model_name="claude-3-sonnet-20240229",
        max_tokens=2000,
        temperature=0.7
    ),
    "claude-3-haiku": LLMConfig(
        provider=LLMProvider.CLAUDE,
        model_name="claude-3-haiku-20240307",
        max_tokens=1500,
        temperature=0.7
    ),
    "mistral-7b": LLMConfig(
        provider=LLMProvider.HUGGINGFACE,
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        max_tokens=1000,
        temperature=0.7,
        custom_params={"use_gpu": True}
    )
}

def create_adapter_from_config(config_name: str, api_key: str) -> LLMAdapter:
    """Create LLM adapter from predefined configuration."""
    if config_name not in COMMON_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    config = COMMON_CONFIGS[config_name]
    config.api_key = api_key
    
    return LLMAdapter(config)
