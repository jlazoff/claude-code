#!/usr/bin/env python3

"""
LiteLLM Integration Manager
Unified LLM interface across all providers with configuration management,
load balancing, fallback strategies, and comprehensive monitoring
"""

import os
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import random

import litellm
from litellm import completion, acompletion, embedding, cost_per_token
import structlog

from unified_config import get_config_manager, APIKeys, EnvironmentConfig

logger = structlog.get_logger()

@dataclass
class LLMModelConfig:
    """Configuration for a specific LLM model."""
    
    model_name: str
    provider: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    max_retries: int = 3
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    context_window: int = 4096
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_vision: bool = False
    priority: int = 1  # 1 = highest priority
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class LLMUsageStats:
    """Usage statistics for LLM models."""
    
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    average_latency: float = 0.0
    last_used: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class LLMResponse:
    """Standardized LLM response with metadata."""
    
    content: str
    model: str
    provider: str
    usage: Dict[str, Any]
    latency: float
    cost: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Round-robin and weighted load balancer for LLM models."""
    
    def __init__(self):
        self.model_weights: Dict[str, float] = {}
        self.model_usage_count: Dict[str, int] = {}
        self.model_last_used: Dict[str, datetime] = {}
        self.round_robin_index = 0
        self._lock = threading.Lock()
    
    def add_model(self, model_name: str, weight: float = 1.0):
        """Add a model to the load balancer."""
        with self._lock:
            self.model_weights[model_name] = weight
            self.model_usage_count[model_name] = 0
            self.model_last_used[model_name] = datetime.utcnow()
    
    def get_next_model(self, available_models: List[str], strategy: str = "round_robin") -> str:
        """Get the next model based on load balancing strategy."""
        if not available_models:
            raise ValueError("No available models for load balancing")
        
        with self._lock:
            if strategy == "round_robin":
                model = available_models[self.round_robin_index % len(available_models)]
                self.round_robin_index = (self.round_robin_index + 1) % len(available_models)
                return model
            
            elif strategy == "weighted":
                # Calculate weights for available models
                weights = []
                for model in available_models:
                    weight = self.model_weights.get(model, 1.0)
                    # Reduce weight based on recent usage
                    recent_usage = self.model_usage_count.get(model, 0)
                    adjusted_weight = weight / (1 + recent_usage * 0.1)
                    weights.append(adjusted_weight)
                
                # Weighted random selection
                total_weight = sum(weights)
                if total_weight > 0:
                    rand_val = random.uniform(0, total_weight)
                    cumulative_weight = 0
                    for i, weight in enumerate(weights):
                        cumulative_weight += weight
                        if rand_val <= cumulative_weight:
                            return available_models[i]
                
                return available_models[0]
            
            elif strategy == "least_used":
                # Select model with least recent usage
                least_used_model = min(
                    available_models,
                    key=lambda m: self.model_usage_count.get(m, 0)
                )
                return least_used_model
            
            else:
                return available_models[0]
    
    def record_usage(self, model_name: str):
        """Record usage of a model."""
        with self._lock:
            self.model_usage_count[model_name] = self.model_usage_count.get(model_name, 0) + 1
            self.model_last_used[model_name] = datetime.utcnow()


class LiteLLMManager:
    """Comprehensive LiteLLM management with configuration, monitoring, and optimization."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.models: Dict[str, LLMModelConfig] = {}
        self.usage_stats: Dict[str, LLMUsageStats] = {}
        self.load_balancer = LoadBalancer()
        self.fallback_chain: List[str] = []
        
        # Configuration
        self.default_timeout = 60
        self.default_max_retries = 3
        self.enable_caching = True
        self.enable_load_balancing = True
        self.enable_fallback = True
        
        # Monitoring
        self.request_queue = Queue()
        self.metrics_cache = {}
        self.error_tracking = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize with current configuration
        self._initialize_models()
        self._setup_litellm()
        
        # Start background monitoring
        self._start_monitoring()
        
        logger.info("LiteLLMManager initialized", 
                   models=len(self.models),
                   environment=self.config_manager.current_environment)
    
    def _initialize_models(self):
        """Initialize LLM models from configuration."""
        api_keys = self.config_manager.get_api_keys()
        current_config = self.config_manager.get_current_config()
        
        # OpenAI Models
        if api_keys.openai_api_key:
            self._add_openai_models(api_keys.openai_api_key)
        
        # Anthropic Models
        if api_keys.anthropic_api_key:
            self._add_anthropic_models(api_keys.anthropic_api_key)
        
        # Google Models
        if api_keys.gemini_api_key:
            self._add_google_models(api_keys.gemini_api_key)
        
        # Local Models
        self._add_local_models(api_keys)
        
        # Setup fallback chain
        self._setup_fallback_chain()
        
        logger.info("Models initialized", 
                   total_models=len(self.models),
                   enabled_models=len([m for m in self.models.values() if m.enabled]))
    
    def _add_openai_models(self, api_key: str):
        """Add OpenAI models."""
        openai_models = [
            ("gpt-4o", 4096, 128000, True, True, True),
            ("gpt-4o-mini", 4096, 128000, True, True, True),
            ("gpt-4-turbo", 4096, 128000, True, True, True),
            ("gpt-4", 4096, 8192, True, True, False),
            ("gpt-3.5-turbo", 4096, 16384, True, True, False),
            ("o1-preview", 32768, 128000, False, False, False),
            ("o1-mini", 65536, 128000, False, False, False),
        ]
        
        for model_name, max_tokens, context_window, streaming, functions, vision in openai_models:
            self.models[model_name] = LLMModelConfig(
                model_name=model_name,
                provider="openai",
                api_key=api_key,
                max_tokens=max_tokens,
                context_window=context_window,
                supports_streaming=streaming,
                supports_functions=functions,
                supports_vision=vision,
                priority=1 if "gpt-4o" in model_name else 2,
                tags=["openai", "chat", "general"]
            )
            self.load_balancer.add_model(model_name, weight=2.0 if "gpt-4o" in model_name else 1.0)
    
    def _add_anthropic_models(self, api_key: str):
        """Add Anthropic models."""
        anthropic_models = [
            ("claude-3-5-sonnet-20241022", 8192, 200000, True, True, True),
            ("claude-3-5-haiku-20241022", 8192, 200000, True, True, True),
            ("claude-3-opus-20240229", 4096, 200000, True, True, True),
            ("claude-3-sonnet-20240229", 4096, 200000, True, True, True),
            ("claude-3-haiku-20240307", 4096, 200000, True, True, True),
        ]
        
        for model_name, max_tokens, context_window, streaming, functions, vision in anthropic_models:
            self.models[model_name] = LLMModelConfig(
                model_name=model_name,
                provider="anthropic",
                api_key=api_key,
                max_tokens=max_tokens,
                context_window=context_window,
                supports_streaming=streaming,
                supports_functions=functions,
                supports_vision=vision,
                priority=1,
                tags=["anthropic", "claude", "chat", "general"]
            )
            self.load_balancer.add_model(model_name, weight=2.0)
    
    def _add_google_models(self, api_key: str):
        """Add Google models."""
        google_models = [
            ("gemini-1.5-pro", 8192, 2000000, True, True, True),
            ("gemini-1.5-flash", 8192, 1000000, True, True, True),
            ("gemini-pro", 2048, 32768, True, True, False),
        ]
        
        for model_name, max_tokens, context_window, streaming, functions, vision in google_models:
            self.models[model_name] = LLMModelConfig(
                model_name=model_name,
                provider="google",
                api_key=api_key,
                max_tokens=max_tokens,
                context_window=context_window,
                supports_streaming=streaming,
                supports_functions=functions,
                supports_vision=vision,
                priority=2,
                tags=["google", "gemini", "chat", "general"]
            )
            self.load_balancer.add_model(model_name, weight=1.5)
    
    def _add_local_models(self, api_keys: APIKeys):
        """Add local LLM models."""
        # Ollama models
        if api_keys.ollama_endpoint:
            local_models = [
                ("ollama/llama3.1:8b", "ollama"),
                ("ollama/codellama:13b", "ollama"),
                ("ollama/mistral:7b", "ollama"),
                ("ollama/phi3:mini", "ollama"),
            ]
            
            for model_name, provider in local_models:
                self.models[model_name] = LLMModelConfig(
                    model_name=model_name,
                    provider=provider,
                    api_base=api_keys.ollama_endpoint,
                    max_tokens=4096,
                    context_window=8192,
                    cost_per_input_token=0.0,
                    cost_per_output_token=0.0,
                    priority=3,
                    tags=["local", "ollama", "chat"]
                )
                self.load_balancer.add_model(model_name, weight=0.5)
        
        # vLLM models
        if api_keys.vllm_endpoint:
            self.models["vllm/local-model"] = LLMModelConfig(
                model_name="vllm/local-model",
                provider="vllm",
                api_base=api_keys.vllm_endpoint,
                max_tokens=4096,
                context_window=8192,
                cost_per_input_token=0.0,
                cost_per_output_token=0.0,
                priority=3,
                tags=["local", "vllm", "chat"]
            )
            self.load_balancer.add_model("vllm/local-model", weight=0.5)
    
    def _setup_fallback_chain(self):
        """Setup fallback chain based on priority and availability."""
        # Sort models by priority and enabled status
        available_models = [
            model.model_name for model in self.models.values() 
            if model.enabled
        ]
        
        # Sort by priority (lower number = higher priority)
        available_models.sort(key=lambda m: self.models[m].priority)
        
        self.fallback_chain = available_models
        logger.info("Fallback chain established", chain=self.fallback_chain[:5])  # Show top 5
    
    def _setup_litellm(self):
        """Configure LiteLLM with our settings."""
        # Set API keys in environment
        api_keys = self.config_manager.get_api_keys()
        
        if api_keys.openai_api_key:
            os.environ["OPENAI_API_KEY"] = api_keys.openai_api_key
        if api_keys.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_keys.anthropic_api_key
        if api_keys.gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = api_keys.gemini_api_key
        
        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported parameters
        litellm.set_verbose = False  # Reduce logging noise
        litellm.telemetry = False  # Disable telemetry
        
        logger.info("LiteLLM configured with API keys")
    
    def _start_monitoring(self):
        """Start background monitoring and metrics collection."""
        def monitor_loop():
            while True:
                try:
                    self._collect_metrics()
                    self._cleanup_old_stats()
                    time.sleep(60)  # Collect metrics every minute
                except Exception as e:
                    logger.error("Monitoring error", error=str(e))
        
        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Background monitoring started")
    
    def _collect_metrics(self):
        """Collect and cache performance metrics."""
        current_time = datetime.utcnow()
        
        # Calculate aggregate metrics
        total_requests = sum(stats.total_requests for stats in self.usage_stats.values())
        total_cost = sum(stats.total_cost for stats in self.usage_stats.values())
        success_rate = 0.0
        
        if total_requests > 0:
            successful_requests = sum(stats.successful_requests for stats in self.usage_stats.values())
            success_rate = successful_requests / total_requests
        
        self.metrics_cache = {
            "timestamp": current_time.isoformat(),
            "total_requests": total_requests,
            "total_cost": total_cost,
            "success_rate": success_rate,
            "active_models": len([m for m in self.models.values() if m.enabled]),
            "model_stats": {name: asdict(stats) for name, stats in self.usage_stats.items()}
        }
    
    def _cleanup_old_stats(self):
        """Clean up old statistics to prevent memory bloat."""
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        # Reset stats for models not used in the last week
        for model_name, stats in list(self.usage_stats.items()):
            if stats.last_used:
                last_used = datetime.fromisoformat(stats.last_used)
                if last_used < cutoff_time:
                    # Reset but keep model in stats
                    self.usage_stats[model_name] = LLMUsageStats(model_name=model_name)
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        functions: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete a chat conversation with automatic model selection and fallback."""
        
        start_time = time.time()
        selected_model = None
        response = None
        error_messages = []
        
        # Determine which models to try
        models_to_try = []
        if model and model in self.models:
            models_to_try = [model]
        elif self.enable_load_balancing:
            available_models = [m for m in self.fallback_chain if self.models[m].enabled]
            if available_models:
                primary_model = self.load_balancer.get_next_model(available_models)
                models_to_try = [primary_model]
        
        if self.enable_fallback and not models_to_try:
            models_to_try = [m for m in self.fallback_chain if self.models[m].enabled]
        
        if not models_to_try:
            raise ValueError("No available models configured")
        
        # Try models in order
        for model_name in models_to_try:
            try:
                model_config = self.models[model_name]
                
                # Prepare request parameters
                request_params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or model_config.max_tokens,
                    "timeout": model_config.timeout,
                    **kwargs
                }
                
                # Add functions if supported
                if functions and model_config.supports_functions:
                    request_params["functions"] = functions
                
                # Make the request
                if stream and model_config.supports_streaming:
                    # Handle streaming separately
                    response_obj = await acompletion(**request_params, stream=True)
                    # For streaming, we'll need to collect the full response
                    content = ""
                    async for chunk in response_obj:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                content += delta.content
                    
                    # Create mock usage for streaming
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                else:
                    response_obj = await acompletion(**request_params)
                    content = response_obj.choices[0].message.content
                    usage = response_obj.usage if hasattr(response_obj, 'usage') else {}
                
                # Calculate metrics
                latency = time.time() - start_time
                cost = self._calculate_cost(model_name, usage)
                
                # Create standardized response
                response = LLMResponse(
                    content=content,
                    model=model_name,
                    provider=model_config.provider,
                    usage=usage,
                    latency=latency,
                    cost=cost,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream,
                        "functions_used": bool(functions)
                    }
                )
                
                # Update statistics
                self._update_usage_stats(model_name, usage, cost, latency, success=True)
                self.load_balancer.record_usage(model_name)
                
                selected_model = model_name
                break
                
            except Exception as e:
                error_msg = f"Model {model_name} failed: {str(e)}"
                error_messages.append(error_msg)
                logger.warning("Model request failed", model=model_name, error=str(e))
                
                # Update error statistics
                self._update_usage_stats(model_name, {}, 0, 0, success=False, error=str(e))
                
                # Continue to next model if fallback is enabled
                if not self.enable_fallback:
                    break
        
        if not response:
            error_summary = "; ".join(error_messages)
            raise Exception(f"All models failed: {error_summary}")
        
        logger.info("LLM request completed", 
                   model=selected_model,
                   latency=f"{response.latency:.2f}s",
                   cost=f"${response.cost:.6f}")
        
        return response
    
    def _calculate_cost(self, model_name: str, usage: Dict[str, Any]) -> float:
        """Calculate cost for a request."""
        model_config = self.models.get(model_name)
        if not model_config or not usage:
            return 0.0
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Use configured costs or try to get from LiteLLM
        input_cost = model_config.cost_per_input_token
        output_cost = model_config.cost_per_output_token
        
        if input_cost == 0.0 or output_cost == 0.0:
            try:
                # Try to get cost from LiteLLM
                cost_info = cost_per_token(model_name, prompt_tokens, completion_tokens)
                return cost_info if isinstance(cost_info, (int, float)) else 0.0
            except:
                return 0.0
        
        return (prompt_tokens * input_cost) + (completion_tokens * output_cost)
    
    def _update_usage_stats(self, model_name: str, usage: Dict[str, Any], cost: float, 
                           latency: float, success: bool = True, error: Optional[str] = None):
        """Update usage statistics for a model."""
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = LLMUsageStats(model_name=model_name)
        
        stats = self.usage_stats[model_name]
        stats.total_requests += 1
        stats.last_used = datetime.utcnow().isoformat()
        
        if success:
            stats.successful_requests += 1
            stats.total_input_tokens += usage.get("prompt_tokens", 0)
            stats.total_output_tokens += usage.get("completion_tokens", 0)
            stats.total_cost += cost
            
            # Update rolling average latency
            if stats.average_latency == 0:
                stats.average_latency = latency
            else:
                stats.average_latency = (stats.average_latency * 0.9) + (latency * 0.1)
        else:
            stats.failed_requests += 1
            stats.error_count += 1
            stats.last_error = error
    
    def add_model(self, model_config: LLMModelConfig):
        """Add a new model configuration."""
        self.models[model_config.model_name] = model_config
        self.load_balancer.add_model(model_config.model_name, weight=1.0)
        self._setup_fallback_chain()
        
        logger.info("Model added", model=model_config.model_name, provider=model_config.provider)
    
    def remove_model(self, model_name: str):
        """Remove a model configuration."""
        if model_name in self.models:
            del self.models[model_name]
            self._setup_fallback_chain()
            logger.info("Model removed", model=model_name)
    
    def enable_model(self, model_name: str):
        """Enable a model."""
        if model_name in self.models:
            self.models[model_name].enabled = True
            self._setup_fallback_chain()
            logger.info("Model enabled", model=model_name)
    
    def disable_model(self, model_name: str):
        """Disable a model."""
        if model_name in self.models:
            self.models[model_name].enabled = False
            self._setup_fallback_chain()
            logger.info("Model disabled", model=model_name)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all models."""
        return {
            "total_models": len(self.models),
            "enabled_models": len([m for m in self.models.values() if m.enabled]),
            "providers": list(set(m.provider for m in self.models.values())),
            "fallback_chain": self.fallback_chain,
            "usage_stats": {name: asdict(stats) for name, stats in self.usage_stats.items()},
            "metrics": self.metrics_cache
        }
    
    def get_best_model_for_task(self, task_type: str = "general", 
                               requirements: Dict[str, Any] = None) -> str:
        """Get the best model for a specific task."""
        requirements = requirements or {}
        
        # Filter models based on requirements
        candidates = []
        for model_name, model_config in self.models.items():
            if not model_config.enabled:
                continue
            
            # Check requirements
            if requirements.get("supports_vision") and not model_config.supports_vision:
                continue
            if requirements.get("supports_functions") and not model_config.supports_functions:
                continue
            if requirements.get("max_context") and model_config.context_window < requirements["max_context"]:
                continue
            
            # Check tags
            if task_type != "general" and task_type not in model_config.tags:
                continue
            
            candidates.append((model_name, model_config))
        
        if not candidates:
            return self.fallback_chain[0] if self.fallback_chain else None
        
        # Sort by priority and success rate
        def model_score(item):
            model_name, model_config = item
            stats = self.usage_stats.get(model_name, LLMUsageStats(model_name))
            success_rate = stats.successful_requests / max(stats.total_requests, 1)
            priority_score = 1.0 / model_config.priority  # Lower priority number = higher score
            return priority_score * success_rate
        
        candidates.sort(key=model_score, reverse=True)
        return candidates[0][0]
    
    def save_configuration(self):
        """Save model configurations to the config manager."""
        model_configs = {name: asdict(config) for name, config in self.models.items()}
        self.config_manager.update_system_state(llm_model_configs=model_configs)
        
        usage_stats = {name: asdict(stats) for name, stats in self.usage_stats.items()}
        self.config_manager.update_system_state(llm_usage_stats=usage_stats)
        
        logger.info("LLM configuration saved")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export usage metrics in specified format."""
        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": {name: asdict(config) for name, config in self.models.items()},
            "usage_stats": {name: asdict(stats) for name, stats in self.usage_stats.items()},
            "metrics_cache": self.metrics_cache,
            "fallback_chain": self.fallback_chain
        }
        
        if format.lower() == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            return str(metrics_data)


# Global LiteLLM manager instance
_llm_manager = None

def get_llm_manager() -> LiteLLMManager:
    """Get the global LiteLLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LiteLLMManager()
    return _llm_manager


async def complete_with_best_model(
    messages: List[Dict[str, str]],
    task_type: str = "general",
    **kwargs
) -> LLMResponse:
    """Complete a request using the best model for the task."""
    llm_manager = get_llm_manager()
    best_model = llm_manager.get_best_model_for_task(task_type)
    return await llm_manager.complete(messages, model=best_model, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_litellm_manager():
        print("🤖 Master Orchestrator - LiteLLM Manager")
        print("=" * 50)
        
        # Initialize manager
        llm_manager = get_llm_manager()
        
        # Show status
        status = llm_manager.get_model_status()
        print(f"📊 Total models: {status['total_models']}")
        print(f"✅ Enabled models: {status['enabled_models']}")
        print(f"🏷️ Providers: {', '.join(status['providers'])}")
        print(f"🔄 Fallback chain: {', '.join(status['fallback_chain'][:3])}...")
        
        # Test completion (if we have API keys)
        test_messages = [
            {"role": "user", "content": "Hello! Can you help me with a simple programming question?"}
        ]
        
        try:
            response = await llm_manager.complete(test_messages, temperature=0.7)
            print(f"\n✨ Test completion successful!")
            print(f"🤖 Model: {response.model}")
            print(f"⚡ Latency: {response.latency:.2f}s")
            print(f"💰 Cost: ${response.cost:.6f}")
            print(f"📝 Response: {response.content[:100]}...")
            
        except Exception as e:
            print(f"\n⚠️ Test completion failed: {e}")
            print("💡 This is expected if no API keys are configured")
        
        # Save configuration
        llm_manager.save_configuration()
        print(f"\n💾 Configuration saved")
        
        print(f"\n🎯 Best model for coding: {llm_manager.get_best_model_for_task('coding')}")
        print(f"🎯 Best model for general: {llm_manager.get_best_model_for_task('general')}")
    
    # Run test
    asyncio.run(test_litellm_manager())