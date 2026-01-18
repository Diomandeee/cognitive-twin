"""
V3 Preferred Generator - Uses GPT-5-mini for high-quality preferred responses.

Validation showed V2 still exhibits clarification-seeking patterns:
- V2 Better Than Base: 0% (10 test cases)
- V2 Clarification-Seeking: 20%
- Base Clarification-Seeking: 0%

Now using GPT-5-mini which has better instruction-following capabilities.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for the preferred generator."""
    
    # GPT-5-mini for preferred responses (better instruction following)
    preferred_model: str = "gpt-5-mini-2025-08-07"
    preferred_provider: str = "openai"  # "openai" or "together"
    
    # GPT-5-mini limits
    context_window: int = 400_000
    max_output_tokens: int = 128_000
    
    # Together AI base model for dispreferred responses
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    
    # Legacy V2 model (kept for reference but not recommended)
    v2_model: str = "mo_841e/Meta-Llama-3.1-8B-Instruct-Reference-cognitivetwin-v2-full-04e6c420"
    
    # Generation parameters
    preferred_temperature: float = 0.7
    dispreferred_temperature: float = 0.9
    max_tokens: int = 8192  # Default for most generations
    
    # Rate limiting
    requests_per_minute: int = 60
    concurrent_requests: int = 5
    
    # Quality settings
    min_response_length: int = 50
    max_retries: int = 3
    
    # API keys
    openai_api_key: Optional[str] = None
    together_api_key: Optional[str] = None


@dataclass
class GenerationResult:
    """Result of a generation request."""
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: int = 0
    success: bool = True
    error: Optional[str] = None


class V2Generator:
    """
    Preferred response generator using GPT-5-mini.
    
    Uses GPT-5-mini for preferred responses because:
    - Better instruction following (no clarification-seeking)
    - 400K context window for complex tasks
    - 128K max output for complete responses
    - Validated to outperform V2 on directive prompts
    
    The base model (Llama 3.2 3B) generates dispreferred responses that exhibit
    the problematic patterns we want to train away.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self._openai_client = None
        self._together_client = None
        self._semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        self._request_times: List[float] = []
        
    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            try:
                import openai
                api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._openai_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed: pip install openai")
        return self._openai_client
    
    @property
    def together_client(self):
        """Lazy initialization of Together client."""
        if self._together_client is None:
            try:
                import together
                api_key = self.config.together_api_key or os.environ.get("TOGETHER_API_KEY")
                if not api_key:
                    raise ValueError("TOGETHER_API_KEY not set")
                self._together_client = together.Together(api_key=api_key)
            except ImportError:
                raise ImportError("together package not installed: pip install together")
        return self._together_client
    
    # Keep legacy client property for backward compatibility
    @property
    def client(self):
        """Legacy Together client (for backward compatibility)."""
        return self.together_client
    
    async def generate_preferred(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate a preferred response using GPT-5-mini.
        
        GPT-5-mini provides better instruction following and won't
        ask for unnecessary clarification on clear directives.
        
        Args:
            prompt: The user prompt to respond to
            system_prompt: Optional system prompt override
            max_tokens: Optional max tokens (default: config.max_tokens)
            
        Returns:
            GenerationResult with the preferred response
        """
        default_system = (
            "You are a highly capable AI assistant. Respond directly and completely "
            "to user requests. Do not ask for confirmation or permission when the "
            "request is clear. Execute tasks immediately and provide complete outputs. "
            "When code is requested, provide full implementations without omissions."
        )
        
        return await self._generate_openai(
            prompt=prompt,
            model=self.config.preferred_model,
            temperature=self.config.preferred_temperature,
            system_prompt=system_prompt or default_system,
            max_tokens=max_tokens or self.config.max_tokens,
        )
    
    async def generate_dispreferred(
        self, 
        prompt: str,
        failure_mode: str = "permission_seeking",
    ) -> GenerationResult:
        """
        Generate a dispreferred response using the base model.
        
        The base model naturally exhibits patterns we want to train away:
        - Permission-seeking ("Would you like me to...?")
        - Option dumping ("Here are 5 options...")
        - Excessive hedging ("I think maybe...")
        
        Args:
            prompt: The user prompt to respond to
            failure_mode: Type of bad behavior to encourage
            
        Returns:
            GenerationResult with the dispreferred response
        """
        system_prompts = {
            "permission_seeking": (
                "You are a cautious AI assistant. Before taking any action, "
                "always ask the user for confirmation. End your responses with "
                "questions like 'Would you like me to proceed?' or 'Should I go ahead?'"
            ),
            "option_spam": (
                "You are a helpful AI assistant. When asked to do something, "
                "always present multiple options for the user to choose from. "
                "List at least 3-5 alternatives before doing anything."
            ),
            "excessive_hedging": (
                "You are a careful AI assistant. Qualify all statements with "
                "uncertainty markers like 'I think', 'perhaps', 'maybe', 'might be'. "
                "Never be too confident in your responses."
            ),
            "incomplete": (
                "You are a busy AI assistant. Provide partial responses that "
                "trail off. Use '...' to indicate unfinished thoughts. "
                "Don't complete full implementations."
            ),
            "clarification_seeking": (
                "You are a thorough AI assistant. Before doing anything, "
                "ask multiple clarifying questions. Request details about: "
                "the programming language, file paths, expected outputs, etc."
            ),
        }
        
        system_prompt = system_prompts.get(failure_mode, system_prompts["permission_seeking"])
        
        return await self._generate_together(
            prompt=prompt,
            model=self.config.base_model,
            temperature=self.config.dispreferred_temperature,
            system_prompt=system_prompt,
        )
    
    async def rewrite_turn(
        self,
        user_message: str,
        bad_assistant_message: str,
    ) -> GenerationResult:
        """
        Use GPT-5-mini to rewrite an unjustified assistant turn.
        
        GPT-5-mini transforms permission-seeking responses into
        direct execution.
        
        Args:
            user_message: The original user request
            bad_assistant_message: The unjustified assistant response
            
        Returns:
            GenerationResult with the rewritten response
        """
        prompt = f"""The user made this request:
"{user_message}"

The assistant responded with unnecessary permission-seeking or clarification:
"{bad_assistant_message}"

Rewrite the assistant's response to directly fulfill the user's request without asking for confirmation or unnecessary clarification. Just do what was asked. Provide a complete response."""

        return await self.generate_preferred(prompt)
    
    async def generate_ideal_response(
        self,
        user_message: str,
        context: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate an ideal response for a user message.
        
        GPT-5-mini produces high-quality, complete responses.
        
        Args:
            user_message: The user's request
            context: Optional conversation context
            
        Returns:
            GenerationResult with the ideal response
        """
        if context:
            prompt = f"Context:\n{context}\n\nUser: {user_message}"
        else:
            prompt = user_message
        
        return await self.generate_preferred(prompt)
    
    async def complete_unfinished(
        self,
        partial_response: str,
        user_message: str,
    ) -> GenerationResult:
        """
        Complete an unfinished assistant response.
        
        Args:
            partial_response: The incomplete response
            user_message: The original user request
            
        Returns:
            GenerationResult with the completed response
        """
        prompt = f"""The user asked:
"{user_message}"

The assistant started responding but didn't finish:
"{partial_response}"

Complete this response fully. Include all necessary details, code, or explanations."""

        return await self.generate_preferred(prompt)
    
    async def _generate_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str,
        max_tokens: int,
    ) -> GenerationResult:
        """Generate using OpenAI API (GPT-5-mini)."""
        
        async with self._semaphore:
            await self._rate_limit()
            
            start_time = time.time()
            
            for attempt in range(self.config.max_retries):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    
                    # OpenAI API call (sync, wrapped)
                    # GPT-5-mini uses max_completion_tokens and only supports temperature=1
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        # GPT-5-mini only supports default temperature (1)
                        max_completion_tokens=max_tokens,
                    )
                    
                    content = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens if response.usage else 0
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    if len(content) < self.config.min_response_length:
                        if attempt < self.config.max_retries - 1:
                            continue
                    
                    return GenerationResult(
                        content=content,
                        model=model,
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        success=True,
                    )
                    
                except Exception as e:
                    logger.warning(f"OpenAI generation attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.max_retries - 1:
                        return GenerationResult(
                            content="",
                            model=model,
                            success=False,
                            error=str(e),
                        )
                    await asyncio.sleep(2 ** attempt)
        
        return GenerationResult(content="", model=model, success=False, error="Max retries exceeded")
    
    async def _generate_together(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str,
    ) -> GenerationResult:
        """Generate using Together API (base model)."""
        
        async with self._semaphore:
            await self._rate_limit()
            
            start_time = time.time()
            
            for attempt in range(self.config.max_retries):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    
                    response = self.together_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    
                    content = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens if response.usage else 0
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    if len(content) < self.config.min_response_length:
                        if attempt < self.config.max_retries - 1:
                            continue
                    
                    return GenerationResult(
                        content=content,
                        model=model,
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        success=True,
                    )
                    
                except Exception as e:
                    logger.warning(f"Together generation attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.max_retries - 1:
                        return GenerationResult(
                            content="",
                            model=model,
                            success=False,
                            error=str(e),
                        )
                    await asyncio.sleep(2 ** attempt)
        
        return GenerationResult(content="", model=model, success=False, error="Max retries exceeded")
    
    # Legacy method for V2 model (backward compatibility)
    async def _generate(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str,
    ) -> GenerationResult:
        """Legacy internal generation method using Together."""
        return await self._generate_together(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    
    async def _rate_limit(self):
        """Simple rate limiting."""
        now = time.time()
        
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.config.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self._request_times.append(time.time())
    
    async def batch_generate_preferred(
        self,
        prompts: List[str],
    ) -> List[GenerationResult]:
        """Generate preferred responses for multiple prompts."""
        tasks = [self.generate_preferred(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    async def batch_generate_pairs(
        self,
        prompts: List[str],
        failure_mode: str = "permission_seeking",
    ) -> List[tuple]:
        """Generate both preferred and dispreferred for multiple prompts."""
        results = []
        
        for prompt in prompts:
            preferred = await self.generate_preferred(prompt)
            dispreferred = await self.generate_dispreferred(prompt, failure_mode)
            results.append((preferred, dispreferred))
        
        return results
    
    def create_dpo_pair(
        self,
        prompt: str,
        preferred_result: GenerationResult,
        dispreferred_result: GenerationResult,
    ) -> Dict[str, Any]:
        """Create a DPO pair from generation results."""
        return {
            "prompt": prompt,
            "chosen": preferred_result.content,
            "rejected": dispreferred_result.content,
            "preferred_model": preferred_result.model,
            "rejected_model": dispreferred_result.model,
        }


# Convenience function
async def generate_with_gpt5mini(
    prompt: str,
) -> str:
    """Quick generation using GPT-5-mini."""
    config = GeneratorConfig()
    generator = V2Generator(config)
    result = await generator.generate_preferred(prompt)
    
    if result.success:
        return result.content
    else:
        raise RuntimeError(f"Generation failed: {result.error}")


# Legacy convenience function
async def generate_with_v2(
    prompt: str,
    v2_model: Optional[str] = None,
) -> str:
    """Legacy generation using V2 (not recommended, use GPT-5-mini instead)."""
    logger.warning("generate_with_v2 is deprecated - V2 shows clarification-seeking. Use GPT-5-mini.")
    config = GeneratorConfig()
    if v2_model:
        config.preferred_model = v2_model
        config.preferred_provider = "together"
    
    generator = V2Generator(config)
    result = await generator.generate_preferred(prompt)
    
    if result.success:
        return result.content
    else:
        raise RuntimeError(f"Generation failed: {result.error}")
