"""
LLM Integration for Master Orchestrator
Provides unified interface for multiple LLM providers with optimization
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import time
import json

import structlog
import openai
import anthropic
import google.generativeai as genai
from pydantic import BaseModel, Field

from .config import LLMProviderConfig

logger = structlog.get_logger()


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    VLLM = "vllm"
    LOCAL = "local"


class LLMRequest(BaseModel):
    """LLM request model."""
    
    messages: List[Dict[str, str]] = Field(description="Chat messages")
    model: Optional[str] = Field(default=None, description="Specific model to use")
    max_tokens: int = Field(default=4096, description="Maximum tokens")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    stream: bool = Field(default=False, description="Enable streaming")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available tools")


class LLMResponse(BaseModel):
    """LLM response model."""
    
    content: str = Field(description="Generated content")
    provider: str = Field(description="Provider used")
    model: str = Field(description="Model used")
    tokens_used: int = Field(default=0, description="Tokens consumed")
    latency_ms: int = Field(description="Response latency in milliseconds")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMMetrics(BaseModel):
    """LLM usage metrics."""
    
    total_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    average_latency_ms: float = Field(default=0.0)
    success_rate: float = Field(default=0.0)
    provider_usage: Dict[str, int] = Field(default_factory=dict)
    error_count: int = Field(default=0)


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.logger = structlog.get_logger(f"llm.{config.name}")
        self.metrics = LLMMetrics()
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM."""
        raise NotImplementedError
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM."""
        raise NotImplementedError
    
    def get_cost_estimate(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage."""
        # Default cost estimation - override in specific providers
        return tokens * 0.00001  # $0.01 per 1K tokens


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # OpenAI pricing (as of 2024)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI."""
        start_time = time.time()
        
        try:
            messages = request.messages.copy()
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            model = request.model or self.config.model
            
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
            
            if request.tools:
                kwargs["tools"] = request.tools
                kwargs["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**kwargs)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = self.get_cost_estimate(tokens_used, model)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_tokens += tokens_used
            self.metrics.total_cost_usd += cost
            self.metrics.provider_usage[self.config.name] = self.metrics.provider_usage.get(self.config.name, 0) + 1
            
            return LLMResponse(
                content=content,
                provider=self.config.name,
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_usd=cost,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump() if response.usage else {}
                }
            )
            
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI."""
        try:
            messages = request.messages.copy()
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            model = request.model or self.config.model
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    def get_cost_estimate(self, tokens: int, model: str) -> float:
        """Estimate OpenAI cost."""
        if model in self.pricing:
            # Simplified cost calculation (assuming 50/50 input/output split)
            input_cost = (tokens * 0.5) * (self.pricing[model]["input"] / 1000)
            output_cost = (tokens * 0.5) * (self.pricing[model]["output"] / 1000)
            return input_cost + output_cost
        return super().get_cost_estimate(tokens, model)


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        
        # Anthropic pricing
        self.pricing = {
            "claude-3-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic."""
        start_time = time.time()
        
        try:
            model = request.model or self.config.model
            
            # Convert messages format
            messages = []
            system_prompt = request.system_prompt or ""
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    messages.append(msg)
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                messages=messages
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            content = ""
            if response.content:
                content = "".join([block.text for block in response.content if hasattr(block, 'text')])
            
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = self.get_cost_estimate(tokens_used, model)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_tokens += tokens_used
            self.metrics.total_cost_usd += cost
            self.metrics.provider_usage[self.config.name] = self.metrics.provider_usage.get(self.config.name, 0) + 1
            
            return LLMResponse(
                content=content,
                provider=self.config.name,
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_usd=cost,
                metadata={
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
            )
            
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def get_cost_estimate(self, tokens: int, model: str) -> float:
        """Estimate Anthropic cost."""
        if model in self.pricing:
            # Simplified cost calculation
            input_cost = (tokens * 0.5) * (self.pricing[model]["input"] / 1000)
            output_cost = (tokens * 0.5) * (self.pricing[model]["output"] / 1000)
            return input_cost + output_cost
        return super().get_cost_estimate(tokens, model)


class GoogleProvider(LLMProvider):
    """Google provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Google."""
        start_time = time.time()
        
        try:
            # Convert messages to Google format
            prompt_parts = []
            if request.system_prompt:
                prompt_parts.append(f"System: {request.system_prompt}")
            
            for msg in request.messages:
                role = "Human" if msg["role"] == "user" else "Assistant"
                prompt_parts.append(f"{role}: {msg['content']}")
            
            prompt = "\n\n".join(prompt_parts)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            content = response.text if response.text else ""
            tokens_used = self.model.count_tokens(prompt).total_tokens
            cost = self.get_cost_estimate(tokens_used, self.config.model)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_tokens += tokens_used
            self.metrics.total_cost_usd += cost
            self.metrics.provider_usage[self.config.name] = self.metrics.provider_usage.get(self.config.name, 0) + 1
            
            return LLMResponse(
                content=content,
                provider=self.config.name,
                model=self.config.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_usd=cost,
                metadata={
                    "safety_ratings": [rating.to_dict() for rating in response.candidates[0].safety_ratings] if response.candidates else []
                }
            )
            
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.error(f"Google generation failed: {e}")
            raise


class vLLMProvider(LLMProvider):
    """vLLM local provider implementation."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
        # Initialize vLLM client similar to OpenAI
        self.client = openai.AsyncOpenAI(
            api_key="dummy",  # vLLM doesn't require real API key
            base_url=f"{self.base_url}/v1"
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using vLLM."""
        start_time = time.time()
        
        try:
            messages = request.messages.copy()
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            model = request.model or self.config.model
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Local models have no cost
            cost = 0.0
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_tokens += tokens_used
            self.metrics.provider_usage[self.config.name] = self.metrics.provider_usage.get(self.config.name, 0) + 1
            
            return LLMResponse(
                content=content,
                provider=self.config.name,
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_usd=cost,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "local": True
                }
            )
            
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.error(f"vLLM generation failed: {e}")
            raise


class LLMManager:
    """
    LLM Manager for Master Orchestrator.
    
    Provides unified interface for multiple LLM providers with
    automatic optimization, load balancing, and cost management.
    """
    
    def __init__(self, provider_configs: List[LLMProviderConfig]):
        self.logger = structlog.get_logger("llm_manager")
        self.providers: Dict[str, LLMProvider] = {}
        self.load_balancer = LLMLoadBalancer()
        self.cost_optimizer = LLMCostOptimizer()
        
        # Initialize providers
        for config in provider_configs:
            if config.enabled:
                self._create_provider(config)
    
    def _create_provider(self, config: LLMProviderConfig) -> None:
        """Create provider instance based on configuration."""
        try:
            if config.name == "openai":
                provider = OpenAIProvider(config)
            elif config.name == "anthropic":
                provider = AnthropicProvider(config)
            elif config.name == "google":
                provider = GoogleProvider(config)
            elif config.name == "vllm":
                provider = vLLMProvider(config)
            else:
                self.logger.warning(f"Unknown provider type: {config.name}")
                return
            
            self.providers[config.name] = provider
            self.logger.info(f"Initialized LLM provider: {config.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize provider {config.name}: {e}")
    
    async def generate(
        self,
        request: LLMRequest,
        preferred_provider: Optional[str] = None
    ) -> LLMResponse:
        """Generate response with automatic provider selection."""
        
        # Select optimal provider
        provider_name = preferred_provider or self.select_optimal_provider(request)
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider not available: {provider_name}")
        
        provider = self.providers[provider_name]
        
        try:
            response = await provider.generate(request)
            
            # Update load balancer metrics
            self.load_balancer.record_request(provider_name, response.latency_ms, True)
            
            self.logger.debug(
                f"Generated response via {provider_name}",
                tokens=response.tokens_used,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd
            )
            
            return response
            
        except Exception as e:
            self.load_balancer.record_request(provider_name, 0, False)
            self.logger.error(f"Generation failed with {provider_name}: {e}")
            
            # Try fallback provider
            fallback_provider = self.get_fallback_provider(provider_name)
            if fallback_provider and fallback_provider != provider_name:
                self.logger.info(f"Trying fallback provider: {fallback_provider}")
                return await self.generate(request, fallback_provider)
            
            raise
    
    def select_optimal_provider(self, request: LLMRequest) -> str:
        """Select optimal provider based on request and current metrics."""
        
        # Cost-based selection for batch processing
        if not request.stream and request.max_tokens > 1000:
            return self.cost_optimizer.get_cheapest_provider(self.providers, request)
        
        # Performance-based selection for interactive use
        return self.load_balancer.get_best_provider(self.providers)
    
    def get_fallback_provider(self, failed_provider: str) -> Optional[str]:
        """Get fallback provider for failed requests."""
        available_providers = [name for name in self.providers.keys() if name != failed_provider]
        
        if not available_providers:
            return None
        
        # Return provider with best reliability
        return self.load_balancer.get_most_reliable_provider(available_providers)
    
    async def stream_generate(
        self,
        request: LLMRequest,
        preferred_provider: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        
        provider_name = preferred_provider or self.select_optimal_provider(request)
        provider = self.providers[provider_name]
        
        async for chunk in provider.stream_generate(request):
            yield chunk
    
    def get_metrics(self) -> Dict[str, LLMMetrics]:
        """Get metrics for all providers."""
        return {name: provider.metrics for name, provider in self.providers.items()}
    
    def get_total_cost(self) -> float:
        """Get total cost across all providers."""
        return sum(provider.metrics.total_cost_usd for provider in self.providers.values())


class LLMLoadBalancer:
    """Load balancer for LLM providers."""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    def record_request(self, provider: str, latency_ms: int, success: bool) -> None:
        """Record request metrics."""
        if provider not in self.metrics:
            self.metrics[provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_latency_ms": 0,
                "failures": 0
            }
        
        self.metrics[provider]["total_requests"] += 1
        if success:
            self.metrics[provider]["successful_requests"] += 1
            self.metrics[provider]["total_latency_ms"] += latency_ms
        else:
            self.metrics[provider]["failures"] += 1
    
    def get_best_provider(self, providers: Dict[str, LLMProvider]) -> str:
        """Get provider with best performance."""
        best_provider = None
        best_score = -1
        
        for provider_name in providers.keys():
            score = self._calculate_provider_score(provider_name)
            if score > best_score:
                best_score = score
                best_provider = provider_name
        
        return best_provider or list(providers.keys())[0]
    
    def get_most_reliable_provider(self, provider_names: List[str]) -> str:
        """Get most reliable provider."""
        best_provider = provider_names[0]
        best_reliability = 0
        
        for provider_name in provider_names:
            reliability = self._calculate_reliability(provider_name)
            if reliability > best_reliability:
                best_reliability = reliability
                best_provider = provider_name
        
        return best_provider
    
    def _calculate_provider_score(self, provider: str) -> float:
        """Calculate provider performance score."""
        if provider not in self.metrics:
            return 0.5  # Neutral score for new providers
        
        metrics = self.metrics[provider]
        total_requests = metrics["total_requests"]
        
        if total_requests == 0:
            return 0.5
        
        success_rate = metrics["successful_requests"] / total_requests
        avg_latency = metrics["total_latency_ms"] / max(metrics["successful_requests"], 1)
        
        # Score based on success rate and inverse latency
        latency_score = 1.0 / (1.0 + avg_latency / 1000.0)  # Normalize latency
        
        return (success_rate * 0.7) + (latency_score * 0.3)
    
    def _calculate_reliability(self, provider: str) -> float:
        """Calculate provider reliability."""
        if provider not in self.metrics:
            return 0.5
        
        metrics = self.metrics[provider]
        total_requests = metrics["total_requests"]
        
        if total_requests == 0:
            return 0.5
        
        return metrics["successful_requests"] / total_requests


class LLMCostOptimizer:
    """Cost optimizer for LLM usage."""
    
    def get_cheapest_provider(
        self,
        providers: Dict[str, LLMProvider],
        request: LLMRequest
    ) -> str:
        """Get cheapest provider for given request."""
        
        estimated_tokens = min(request.max_tokens, 2000)  # Estimate based on max_tokens
        
        cheapest_provider = None
        lowest_cost = float('inf')
        
        for name, provider in providers.items():
            # Estimate cost
            cost = provider.get_cost_estimate(estimated_tokens, provider.config.model)
            
            if cost < lowest_cost:
                lowest_cost = cost
                cheapest_provider = name
        
        return cheapest_provider or list(providers.keys())[0]