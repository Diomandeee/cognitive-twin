# Phase 6: API Integration

> **Purpose**: Configure and integrate OpenAI GPT 5.2 and GPT 5.2 Codex APIs for V3 data augmentation pipeline, including client setup, rate limiting, error handling, and cost optimization.
>
> **Models**:
> - `gpt-5.2` - General augmentation (conversation rewriting, canonicalization, DPO pairs)
> - `gpt-5.2-codex` - Agentic coding tasks (repo worm, code generation, diff creation)
>
> **Implementation File**: `rag_plusplus/ml/cognitivetwin_v3/api/openai_client.py`

---

## 1. OpenAI API Configuration

### 1.1. API Models

```python
from enum import Enum
from dataclasses import dataclass

class OpenAIModel(str, Enum):
    """Available OpenAI models for V3."""
    
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_CODEX = "gpt-5.2-codex"
    GPT_4_1 = "gpt-4.1"  # Fallback
    GPT_4_TURBO = "gpt-4-turbo"  # Fallback

@dataclass
class ModelConfig:
    """Configuration for a model."""
    
    model_id: str
    context_length: int
    max_output_tokens: int
    supports_responses_api: bool
    supports_chat_api: bool
    price_per_million_input: float
    price_per_million_output: float

MODEL_CONFIGS = {
    OpenAIModel.GPT_5_2: ModelConfig(
        model_id="gpt-5.2",
        context_length=200000,
        max_output_tokens=16384,
        supports_responses_api=True,
        supports_chat_api=True,
        price_per_million_input=2.50,
        price_per_million_output=10.00,
    ),
    OpenAIModel.GPT_5_2_CODEX: ModelConfig(
        model_id="gpt-5.2-codex",
        context_length=200000,
        max_output_tokens=16384,
        supports_responses_api=True,
        supports_chat_api=False,  # Responses API only
        price_per_million_input=5.00,
        price_per_million_output=15.00,
    ),
}
```

### 1.2. Client Configuration

```python
from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class OpenAIClientConfig:
    """Configuration for OpenAI client."""
    
    # Authentication
    api_key: Optional[str] = None
    organization: Optional[str] = None
    
    # Default model settings
    default_model: OpenAIModel = OpenAIModel.GPT_5_2
    codex_model: OpenAIModel = OpenAIModel.GPT_5_2_CODEX
    
    # Request settings
    default_temperature: float = 0.3
    default_max_tokens: int = 4096
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
```

### 1.3. Client Initialization

```python
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import Optional

class V3OpenAIClient:
    """OpenAI client for V3 data augmentation."""
    
    def __init__(self, config: OpenAIClientConfig = None):
        self.config = config or OpenAIClientConfig()
        
        # Initialize sync and async clients
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
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=self.config.max_requests_per_minute,
            max_tokens_per_minute=self.config.max_tokens_per_minute,
        )
        
        # Cost tracker
        self.cost_tracker = CostTracker() if self.config.track_costs else None
    
    def verify_connection(self) -> bool:
        """Verify API connection."""
        try:
            models = self.client.models.list()
            return len(list(models)) > 0
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
```

---

## 2. API Endpoints

### 2.1. Chat Completions API (GPT 5.2)

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatCompletionRequest:
    messages: List[ChatMessage]
    model: str = "gpt-5.2"
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

class ChatCompletionsAPI:
    """Chat Completions API wrapper."""
    
    def __init__(self, client: V3OpenAIClient):
        self.client = client
    
    def complete(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Synchronous chat completion."""
        
        # Rate limit
        self.client.rate_limiter.wait()
        
        response = self.client.client.chat.completions.create(
            model=self.client.config.default_model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Track costs
        if self.client.cost_tracker:
            self.client.cost_tracker.add_usage(
                model=self.client.config.default_model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        return response.choices[0].message.content
    
    async def complete_async(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Asynchronous chat completion."""
        
        await self.client.rate_limiter.wait_async()
        
        response = await self.client.async_client.chat.completions.create(
            model=self.client.config.default_model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if self.client.cost_tracker:
            self.client.cost_tracker.add_usage(
                model=self.client.config.default_model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        return response.choices[0].message.content
    
    async def batch_complete(
        self,
        message_batches: List[List[dict]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        concurrency: int = 5,
    ) -> List[str]:
        """Batch completions with concurrency control."""
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def complete_one(messages):
            async with semaphore:
                return await self.complete_async(messages, temperature, max_tokens)
        
        tasks = [complete_one(msgs) for msgs in message_batches]
        return await asyncio.gather(*tasks)
```

### 2.2. Responses API (GPT 5.2 Codex)

```python
@dataclass
class ResponsesRequest:
    input: str
    model: str = "gpt-5.2-codex"
    temperature: float = 0.2
    max_tokens: int = 8192

class ResponsesAPI:
    """Responses API wrapper for Codex."""
    
    def __init__(self, client: V3OpenAIClient):
        self.client = client
    
    def generate(
        self,
        input_text: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> str:
        """Synchronous code generation."""
        
        self.client.rate_limiter.wait()
        
        response = self.client.client.responses.create(
            model=self.client.config.codex_model.value,
            input=input_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if self.client.cost_tracker:
            self.client.cost_tracker.add_usage(
                model=self.client.config.codex_model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        
        return response.output
    
    async def generate_async(
        self,
        input_text: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> str:
        """Asynchronous code generation."""
        
        await self.client.rate_limiter.wait_async()
        
        response = await self.client.async_client.responses.create(
            model=self.client.config.codex_model.value,
            input=input_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if self.client.cost_tracker:
            self.client.cost_tracker.add_usage(
                model=self.client.config.codex_model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        
        return response.output
```

---

## 3. Rate Limiting

### 3.1. Token Bucket Rate Limiter

```python
import time
import asyncio
from threading import Lock

class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(
        self,
        max_requests_per_minute: int = 500,
        max_tokens_per_minute: int = 200000,
    ):
        self.max_requests = max_requests_per_minute
        self.max_tokens = max_tokens_per_minute
        
        # Token buckets
        self.request_tokens = max_requests_per_minute
        self.token_tokens = max_tokens_per_minute
        
        # Timing
        self.last_refill = time.time()
        self.refill_interval = 60.0  # 1 minute
        
        # Synchronization
        self.lock = Lock()
        self.async_lock = asyncio.Lock()
    
    def _refill(self):
        """Refill token buckets based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed >= self.refill_interval:
            self.request_tokens = self.max_requests
            self.token_tokens = self.max_tokens
            self.last_refill = now
        else:
            # Partial refill
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
                sleep_time = 0.1
                time.sleep(sleep_time)
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
```

### 3.2. Adaptive Rate Limiting

```python
class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts to API responses."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.consecutive_rate_limits = 0
        self.backoff_multiplier = 1.0
    
    def on_success(self):
        """Call on successful request."""
        self.consecutive_rate_limits = 0
        self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.9)
    
    def on_rate_limit(self):
        """Call on rate limit error."""
        self.consecutive_rate_limits += 1
        self.backoff_multiplier = min(10.0, self.backoff_multiplier * 2.0)
    
    def get_wait_time(self) -> float:
        """Get wait time with backoff."""
        base_wait = 60.0 / self.max_requests
        return base_wait * self.backoff_multiplier
```

---

## 4. Error Handling

### 4.1. Error Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class APIErrorType(str, Enum):
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CONTENT_FILTER = "content_filter"
    CONTEXT_LENGTH = "context_length"
    UNKNOWN = "unknown"

@dataclass
class APIError:
    error_type: APIErrorType
    message: str
    status_code: Optional[int] = None
    retry_after: Optional[float] = None
```

### 4.2. Error Handler

```python
from openai import (
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    APIStatusError,
    APITimeoutError,
)

class ErrorHandler:
    """Handle API errors with retry logic."""
    
    def __init__(self, config: OpenAIClientConfig):
        self.config = config
    
    def classify_error(self, error: Exception) -> APIError:
        """Classify an exception into APIError."""
        
        if isinstance(error, RateLimitError):
            retry_after = self._extract_retry_after(error)
            return APIError(
                error_type=APIErrorType.RATE_LIMIT,
                message=str(error),
                status_code=429,
                retry_after=retry_after,
            )
        
        elif isinstance(error, AuthenticationError):
            return APIError(
                error_type=APIErrorType.AUTHENTICATION,
                message=str(error),
                status_code=401,
            )
        
        elif isinstance(error, BadRequestError):
            if "context_length" in str(error).lower():
                return APIError(
                    error_type=APIErrorType.CONTEXT_LENGTH,
                    message=str(error),
                    status_code=400,
                )
            elif "content_filter" in str(error).lower():
                return APIError(
                    error_type=APIErrorType.CONTENT_FILTER,
                    message=str(error),
                    status_code=400,
                )
            return APIError(
                error_type=APIErrorType.INVALID_REQUEST,
                message=str(error),
                status_code=400,
            )
        
        elif isinstance(error, APIStatusError):
            return APIError(
                error_type=APIErrorType.SERVER_ERROR,
                message=str(error),
                status_code=getattr(error, 'status_code', 500),
            )
        
        elif isinstance(error, APITimeoutError):
            return APIError(
                error_type=APIErrorType.TIMEOUT,
                message=str(error),
            )
        
        return APIError(
            error_type=APIErrorType.UNKNOWN,
            message=str(error),
        )
    
    def should_retry(self, error: APIError, attempt: int) -> bool:
        """Determine if error is retryable."""
        
        if attempt >= self.config.max_retries:
            return False
        
        retryable_types = {
            APIErrorType.RATE_LIMIT,
            APIErrorType.SERVER_ERROR,
            APIErrorType.TIMEOUT,
        }
        
        return error.error_type in retryable_types
    
    def get_retry_delay(self, error: APIError, attempt: int) -> float:
        """Calculate retry delay."""
        
        if error.retry_after:
            return error.retry_after
        
        base_delay = self.config.retry_delay
        
        if self.config.exponential_backoff:
            return base_delay * (2 ** attempt)
        
        return base_delay
    
    def _extract_retry_after(self, error: RateLimitError) -> Optional[float]:
        """Extract retry-after from rate limit error."""
        
        import re
        
        message = str(error)
        match = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", message)
        
        if match:
            return float(match.group(1))
        
        return None
```

### 4.3. Retry Decorator

```python
import functools
import asyncio

def with_retry(func):
    """Decorator to add retry logic to API calls."""
    
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        handler = ErrorHandler(self.client.config)
        
        for attempt in range(self.client.config.max_retries + 1):
            try:
                result = await func(self, *args, **kwargs)
                
                if hasattr(self.client, 'rate_limiter'):
                    if hasattr(self.client.rate_limiter, 'on_success'):
                        self.client.rate_limiter.on_success()
                
                return result
            
            except Exception as e:
                error = handler.classify_error(e)
                
                if hasattr(self.client, 'rate_limiter'):
                    if hasattr(self.client.rate_limiter, 'on_rate_limit'):
                        if error.error_type == APIErrorType.RATE_LIMIT:
                            self.client.rate_limiter.on_rate_limit()
                
                if not handler.should_retry(error, attempt):
                    raise
                
                delay = handler.get_retry_delay(error, attempt)
                print(f"Retry {attempt + 1} after {delay:.1f}s: {error.message[:100]}")
                await asyncio.sleep(delay)
        
        raise Exception("Max retries exceeded")
    
    return wrapper
```

---

## 5. Cost Tracking

### 5.1. Cost Tracker

```python
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

@dataclass
class UsageRecord:
    timestamp: datetime
    model: OpenAIModel
    input_tokens: int
    output_tokens: int
    cost: float

class CostTracker:
    """Track API usage and costs."""
    
    def __init__(self):
        self.records: List[UsageRecord] = []
        self.total_cost = 0.0
        self.lock = Lock()
    
    def add_usage(
        self,
        model: OpenAIModel,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record usage."""
        
        config = MODEL_CONFIGS[model]
        
        input_cost = (input_tokens / 1_000_000) * config.price_per_million_input
        output_cost = (output_tokens / 1_000_000) * config.price_per_million_output
        cost = input_cost + output_cost
        
        record = UsageRecord(
            timestamp=datetime.utcnow(),
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
            model_key = record.model.value
            if model_key not in by_model:
                by_model[model_key] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            
            by_model[model_key]["requests"] += 1
            by_model[model_key]["input_tokens"] += record.input_tokens
            by_model[model_key]["output_tokens"] += record.output_tokens
            by_model[model_key]["cost"] += record.cost
        
        return {
            "total_requests": len(self.records),
            "total_cost": self.total_cost,
            "by_model": by_model,
        }
    
    def check_budget(self, max_cost: float) -> bool:
        """Check if within budget."""
        return self.total_cost <= max_cost
```

### 5.2. Cost Guard

```python
class CostGuard:
    """Guard against excessive API costs."""
    
    def __init__(
        self,
        tracker: CostTracker,
        max_cost: float = 100.0,
        warn_at: float = 0.8,
    ):
        self.tracker = tracker
        self.max_cost = max_cost
        self.warn_at = warn_at
        self.warned = False
    
    def check(self) -> bool:
        """Check if within budget. Returns False if over budget."""
        
        current = self.tracker.total_cost
        
        if current >= self.max_cost:
            raise Exception(f"Cost limit exceeded: ${current:.2f} >= ${self.max_cost:.2f}")
        
        if current >= self.max_cost * self.warn_at and not self.warned:
            print(f"WARNING: Cost approaching limit: ${current:.2f} / ${self.max_cost:.2f}")
            self.warned = True
        
        return True
```

---

## 6. Prompt Templates

### 6.1. Corpus Surgery Templates

```python
REWRITER_SYSTEM_PROMPT = """You are a response rewriter for CognitiveTwin V3.

Your task is to rewrite assistant responses that asked for unnecessary permission 
into responses that execute immediately.

RULES:
1. NEVER ask questions when the directive is clear
2. NEVER use phrases like "Would you like me to...", "Should I...", "Can you confirm..."
3. If assumptions are needed, STATE them as declarations, then proceed
4. ALWAYS produce the artifact that was requested
5. Keep the same technical content, just remove permission-seeking

ASSUMPTION PROTOCOL:
- If something is unknown but non-blocking: choose a reasonable default
- State assumptions in ONE line at the start: "Assumptions: ..."
- Then proceed with the implementation/answer
- NO question marks after assumptions

OUTPUT FORMAT:
Return ONLY the rewritten assistant response. No explanations.
"""

CLASSIFIER_PROMPT = """Classify the following assistant response as one of:
- UNJUSTIFIED: Asks permission when not needed (directive was clear)
- JUSTIFIED: Asks for genuinely missing information
- NEUTRAL: Neither permission-seeking nor requiring clarification

User message: {user_message}
Assistant response: {assistant_response}
Directive completeness: {directive_completeness}

Classification:"""
```

### 6.2. Repo Worm Templates

```python
CODEX_SYSTEM_PROMPT = """You are a code generation assistant for CognitiveTwin V3.

RULES:
1. Generate complete, working code that compiles
2. Follow existing patterns and conventions in the codebase
3. Do NOT ask clarifying questions - make reasonable assumptions
4. State assumptions briefly at the start if needed
5. Include proper error handling and edge cases
6. Match the existing code style exactly

OUTPUT FORMAT:
- Provide code in markdown code blocks
- Include brief comments explaining complex logic
- For diffs, use unified diff format

ASSUMPTION PROTOCOL:
If you need to assume something:
- State it in a comment: # Assumption: ...
- Then proceed with implementation
- NO questions
"""

TASK_PROMPT_TEMPLATE = """
REPOSITORY CONTEXT:
{context}

TASK:
{task_description}

CONSTRAINTS:
- Must compile without errors
- Follow existing patterns
- No new dependencies unless necessary
- Do NOT ask questions

Provide the implementation:
"""
```

### 6.3. Conversation Worm Templates

```python
PARAPHRASE_PROMPT = """Generate {count} paraphrases of the user message that:
1. Preserve the exact same intent and meaning
2. Use different wording, sentence structure, or phrasing
3. Maintain the same level of formality
4. Keep any technical terms unchanged

Original message:
{message}

Output format:
PARAPHRASE 1: ...
PARAPHRASE 2: ...
"""

IDEAL_RESPONSE_PROMPT = """Generate the ideal assistant response for CognitiveTwin V3.

The original assistant response asked for permission or clarification when it shouldn't have.
Generate what the assistant SHOULD have said instead.

RULES:
1. Execute immediately - do not ask permission
2. If assumptions are needed, state them briefly then proceed
3. Produce the requested artifact/output
4. Do NOT end with a question
5. Match the technical level and style of the conversation

CONVERSATION CONTEXT:
{context}

USER MESSAGE:
{user_message}

PROBLEMATIC RESPONSE:
{problematic_response}

Generate the ideal response:
"""
```

---

## 7. V3 Augmentation Pipeline

### 7.1. Unified Client

```python
class V3AugmentationClient:
    """Unified client for V3 data augmentation."""
    
    def __init__(self, config: OpenAIClientConfig = None):
        self.client = V3OpenAIClient(config)
        self.chat_api = ChatCompletionsAPI(self.client)
        self.responses_api = ResponsesAPI(self.client)
        self.cost_guard = CostGuard(
            self.client.cost_tracker,
            max_cost=config.max_cost_per_run if config else 100.0,
        )
    
    async def rewrite_assistant_turn(
        self,
        assistant_message: str,
        user_message: str,
        conversation_history: List[dict],
    ) -> str:
        """Rewrite an assistant turn using GPT 5.2."""
        
        self.cost_guard.check()
        
        messages = [
            {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
            {"role": "user", "content": f"""CONVERSATION HISTORY:
{self._format_history(conversation_history)}

USER MESSAGE:
{user_message}

PROBLEMATIC ASSISTANT RESPONSE:
{assistant_message}

Rewrite this response to execute immediately:"""}
        ]
        
        return await self.chat_api.complete_async(
            messages,
            temperature=0.3,
            max_tokens=4096,
        )
    
    async def generate_code(
        self,
        task_description: str,
        context: str,
    ) -> str:
        """Generate code using GPT 5.2 Codex."""
        
        self.cost_guard.check()
        
        prompt = TASK_PROMPT_TEMPLATE.format(
            context=context,
            task_description=task_description,
        )
        
        full_input = f"{CODEX_SYSTEM_PROMPT}\n\n{prompt}"
        
        return await self.responses_api.generate_async(
            full_input,
            temperature=0.2,
            max_tokens=8192,
        )
    
    async def generate_paraphrases(
        self,
        message: str,
        count: int = 2,
    ) -> List[str]:
        """Generate paraphrases of a message."""
        
        self.cost_guard.check()
        
        messages = [
            {"role": "system", "content": "You are a paraphrase generator."},
            {"role": "user", "content": PARAPHRASE_PROMPT.format(
                count=count,
                message=message,
            )}
        ]
        
        response = await self.chat_api.complete_async(
            messages,
            temperature=0.7,
            max_tokens=1024,
        )
        
        # Parse paraphrases
        import re
        paraphrases = []
        for match in re.finditer(r'PARAPHRASE \d+:\s*(.+?)(?=PARAPHRASE \d+:|$)', response, re.DOTALL):
            paraphrases.append(match.group(1).strip())
        
        return paraphrases[:count]
    
    async def generate_ideal_response(
        self,
        user_message: str,
        problematic_response: str,
        context: List[dict],
    ) -> str:
        """Generate ideal response for friction point."""
        
        self.cost_guard.check()
        
        messages = [
            {"role": "system", "content": IDEAL_RESPONSE_PROMPT.format(
                context=self._format_history(context),
                user_message=user_message,
                problematic_response=problematic_response,
            )}
        ]
        
        return await self.chat_api.complete_async(
            messages,
            temperature=0.3,
            max_tokens=4096,
        )
    
    def get_cost_summary(self) -> dict:
        """Get cost summary."""
        return self.client.cost_tracker.get_summary() if self.client.cost_tracker else {}
    
    def _format_history(self, history: List[dict]) -> str:
        """Format conversation history."""
        parts = []
        for msg in history[-6:]:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")[:500]
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)
```

### 7.2. Batch Processing

```python
class BatchProcessor:
    """Process data in batches with progress tracking."""
    
    def __init__(
        self,
        client: V3AugmentationClient,
        batch_size: int = 10,
        concurrency: int = 5,
    ):
        self.client = client
        self.batch_size = batch_size
        self.concurrency = concurrency
    
    async def process_rewrites(
        self,
        items: List[dict],
        progress_callback = None,
    ) -> List[dict]:
        """Process rewrite tasks in batches."""
        
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            tasks = [
                self.client.rewrite_assistant_turn(
                    item["assistant_message"],
                    item["user_message"],
                    item.get("history", []),
                )
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append({**item, "error": str(result)})
                else:
                    results.append({**item, "rewritten": result})
            
            if progress_callback:
                progress_callback(min(i + self.batch_size, len(items)), len(items))
        
        return results
    
    async def process_code_generation(
        self,
        tasks: List[dict],
        progress_callback = None,
    ) -> List[dict]:
        """Process code generation tasks in batches."""
        
        results = []
        
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            
            gen_tasks = [
                self.client.generate_code(
                    task["description"],
                    task["context"],
                )
                for task in batch
            ]
            
            batch_results = await asyncio.gather(*gen_tasks, return_exceptions=True)
            
            for task, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append({**task, "error": str(result)})
                else:
                    results.append({**task, "generated": result})
            
            if progress_callback:
                progress_callback(min(i + self.batch_size, len(tasks)), len(tasks))
        
        return results
```

---

## 8. CLI Interface

```python
import click
import asyncio
from pathlib import Path

@click.group()
def cli():
    """OpenAI API tools for CognitiveTwin V3."""
    pass

@cli.command()
def verify():
    """Verify API connection."""
    
    client = V3OpenAIClient()
    if client.verify_connection():
        click.echo("✓ API connection verified")
    else:
        click.echo("✗ API connection failed")

@cli.command()
@click.option("--input", type=Path, required=True, help="Input JSONL file")
@click.option("--output", type=Path, required=True, help="Output JSONL file")
@click.option("--batch-size", default=10, help="Batch size")
def rewrite(input, output, batch_size):
    """Rewrite assistant turns."""
    
    import json
    
    # Load data
    with open(input) as f:
        items = [json.loads(line) for line in f]
    
    client = V3AugmentationClient()
    processor = BatchProcessor(client, batch_size=batch_size)
    
    def progress(current, total):
        click.echo(f"  Progress: {current}/{total}")
    
    results = asyncio.run(processor.process_rewrites(items, progress))
    
    # Save results
    with open(output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print summary
    summary = client.get_cost_summary()
    click.echo(f"\nComplete!")
    click.echo(f"  Total cost: ${summary.get('total_cost', 0):.2f}")

@cli.command()
def cost_report():
    """Show cost report."""
    
    client = V3AugmentationClient()
    summary = client.get_cost_summary()
    
    click.echo("Cost Summary:")
    click.echo(f"  Total requests: {summary.get('total_requests', 0)}")
    click.echo(f"  Total cost: ${summary.get('total_cost', 0):.2f}")
    
    for model, data in summary.get('by_model', {}).items():
        click.echo(f"\n  {model}:")
        click.echo(f"    Requests: {data['requests']}")
        click.echo(f"    Input tokens: {data['input_tokens']:,}")
        click.echo(f"    Output tokens: {data['output_tokens']:,}")
        click.echo(f"    Cost: ${data['cost']:.2f}")

if __name__ == "__main__":
    cli()
```

---

## 9. Environment Setup

### 9.1. Required Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export OPENAI_ORG_ID="org-..."
export OPENAI_MAX_REQUESTS_PER_MINUTE=500
export OPENAI_MAX_TOKENS_PER_MINUTE=200000
export OPENAI_MAX_COST_PER_RUN=100.0
```

### 9.2. Dependencies

```
# requirements.txt
openai>=1.40.0
asyncio
aiohttp
tenacity
```

### 9.3. Configuration File

```yaml
# config/openai.yaml
api:
  default_model: gpt-5.2
  codex_model: gpt-5.2-codex
  timeout: 60

rate_limits:
  max_requests_per_minute: 500
  max_tokens_per_minute: 200000

retry:
  max_retries: 3
  retry_delay: 1.0
  exponential_backoff: true

costs:
  track_costs: true
  max_cost_per_run: 100.0
  warn_at_percentage: 80
```

