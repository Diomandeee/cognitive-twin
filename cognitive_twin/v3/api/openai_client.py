"""
OpenAI API client for CognitiveTwin V3.

Provides GPT 5.2 and Codex integration for data augmentation with
rate limiting, error handling, and cost tracking.
"""

import os
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List
from threading import Lock


@dataclass
class ClientConfig:
    """Configuration for OpenAI client."""
    
    api_key: Optional[str] = None
    organization: Optional[str] = None
    
    # Model settings
    default_model: str = "gpt-5.2"
    codex_model: str = "gpt-5.2-codex"
    
    # Request settings
    default_temperature: float = 0.3
    default_max_tokens: int = 64000
    timeout: int = 60
    
    # Rate limiting
    max_requests_per_minute: int = 500
    max_tokens_per_minute: int = 200000
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Cost tracking
    track_costs: bool = True
    max_cost_per_run: float = 100.0
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.organization is None:
            self.organization = os.environ.get("OPENAI_ORG_ID")


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(
        self,
        max_requests_per_minute: int = 500,
        max_tokens_per_minute: int = 200000,
    ):
        self.max_requests = max_requests_per_minute
        self.max_tokens = max_tokens_per_minute
        
        self.request_tokens = max_requests_per_minute
        self.token_tokens = max_tokens_per_minute
        
        self.last_refill = time.time()
        self.refill_interval = 60.0
        
        self.lock = Lock()
        self._async_lock: Optional[asyncio.Lock] = None
    
    @property
    def async_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock
    
    def _refill(self):
        """Refill token buckets based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed >= self.refill_interval:
            self.request_tokens = self.max_requests
            self.token_tokens = self.max_tokens
            self.last_refill = now
        else:
            fraction = elapsed / self.refill_interval
            self.request_tokens = min(
                self.max_requests,
                self.request_tokens + int(self.max_requests * fraction)
            )
            self.token_tokens = min(
                self.max_tokens,
                self.token_tokens + int(self.max_tokens * fraction)
            )
    
    def wait(self, estimated_tokens: int = 1000):
        """Wait for rate limit, blocking."""
        with self.lock:
            self._refill()
            
            while self.request_tokens <= 0 or self.token_tokens < estimated_tokens:
                time.sleep(0.1)
                self._refill()
            
            self.request_tokens -= 1
            self.token_tokens -= estimated_tokens
    
    async def wait_async(self, estimated_tokens: int = 1000):
        """Wait for rate limit, async."""
        async with self.async_lock:
            self._refill()
            
            while self.request_tokens <= 0 or self.token_tokens < estimated_tokens:
                await asyncio.sleep(0.1)
                self._refill()
            
            self.request_tokens -= 1
            self.token_tokens -= estimated_tokens


@dataclass
class UsageRecord:
    """Record of API usage."""
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


class CostTracker:
    """Track API usage and costs."""
    
    # Pricing per million tokens
    PRICING = {
        "gpt-5.2": {"input": 2.50, "output": 10.00},
        "gpt-5.2-codex": {"input": 5.00, "output": 15.00},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }
    
    def __init__(self):
        self.records: List[UsageRecord] = []
        self.total_cost = 0.0
        self.lock = Lock()
    
    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Record usage."""
        pricing = self.PRICING.get(model, {"input": 10.0, "output": 30.0})
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cost = input_cost + output_cost
        
        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
        
        with self.lock:
            self.records.append(record)
            self.total_cost += cost
    
    def get_summary(self) -> dict:
        """Get usage summary."""
        by_model = {}
        
        for record in self.records:
            if record.model not in by_model:
                by_model[record.model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            
            by_model[record.model]["requests"] += 1
            by_model[record.model]["input_tokens"] += record.input_tokens
            by_model[record.model]["output_tokens"] += record.output_tokens
            by_model[record.model]["cost"] += record.cost
        
        return {
            "total_requests": len(self.records),
            "total_cost": self.total_cost,
            "by_model": by_model,
        }
    
    def check_budget(self, max_cost: float) -> bool:
        """Check if within budget."""
        return self.total_cost <= max_cost


class V3OpenAIClient:
    """OpenAI client for V3 data augmentation."""
    
    def __init__(self, config: Optional[ClientConfig] = None):
        self.config = config or ClientConfig()
        
        # Lazy import openai
        try:
            from openai import OpenAI, AsyncOpenAI
            
            self.client = OpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization,
                timeout=self.config.timeout,
            )
            
            self.async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization,
                timeout=self.config.timeout,
            )
            
            self._openai_available = True
        except ImportError:
            self.client = None
            self.async_client = None
            self._openai_available = False
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=self.config.max_requests_per_minute,
            max_tokens_per_minute=self.config.max_tokens_per_minute,
        )
        
        # Cost tracker
        self.cost_tracker = CostTracker() if self.config.track_costs else None
    
    def verify_connection(self) -> bool:
        """Verify API connection."""
        if not self._openai_available:
            print("OpenAI package not installed")
            return False
        
        try:
            models = self.client.models.list()
            return len(list(models)) > 0
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def chat_complete(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Synchronous chat completion."""
        if not self._openai_available:
            raise RuntimeError("OpenAI package not installed")
        
        temperature = temperature or self.config.default_temperature
        max_tokens = max_tokens or self.config.default_max_tokens
        model = model or self.config.default_model
        
        self.rate_limiter.wait()
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if self.cost_tracker:
            self.cost_tracker.add_usage(
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        return response.choices[0].message.content
    
    async def chat_complete_async(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Asynchronous chat completion."""
        if not self._openai_available:
            raise RuntimeError("OpenAI package not installed")
        
        temperature = temperature or self.config.default_temperature
        max_tokens = max_tokens or self.config.default_max_tokens
        model = model or self.config.default_model
        
        await self.rate_limiter.wait_async()
        
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if self.cost_tracker:
            self.cost_tracker.add_usage(
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        return response.choices[0].message.content
    
    def responses_generate(
        self,
        input_text: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Synchronous Responses API generation (Codex)."""
        if not self._openai_available:
            raise RuntimeError("OpenAI package not installed")
        
        temperature = temperature if temperature is not None else 0.2
        max_tokens = max_tokens or 8192
        
        self.rate_limiter.wait()
        
        # Use responses API if available, fallback to chat
        try:
            response = self.client.responses.create(
                model=self.config.codex_model,
                input=input_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if self.cost_tracker:
                self.cost_tracker.add_usage(
                    model=self.config.codex_model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            
            return response.output
        except AttributeError:
            # Fallback to chat completions
            return self.chat_complete(
                [{"role": "user", "content": input_text}],
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.config.codex_model,
            )
    
    async def responses_generate_async(
        self,
        input_text: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Asynchronous Responses API generation (Codex)."""
        if not self._openai_available:
            raise RuntimeError("OpenAI package not installed")
        
        temperature = temperature if temperature is not None else 0.2
        max_tokens = max_tokens or 8192
        
        await self.rate_limiter.wait_async()
        
        # Use responses API if available, fallback to chat
        try:
            response = await self.async_client.responses.create(
                model=self.config.codex_model,
                input=input_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if self.cost_tracker:
                self.cost_tracker.add_usage(
                    model=self.config.codex_model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            
            return response.output
        except AttributeError:
            # Fallback to chat completions
            return await self.chat_complete_async(
                [{"role": "user", "content": input_text}],
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.config.codex_model,
            )
    
    def get_cost_summary(self) -> dict:
        """Get cost summary."""
        if self.cost_tracker:
            return self.cost_tracker.get_summary()
        return {"total_requests": 0, "total_cost": 0.0, "by_model": {}}
    
    def check_budget(self) -> bool:
        """Check if within budget."""
        if self.cost_tracker:
            return self.cost_tracker.check_budget(self.config.max_cost_per_run)
        return True

