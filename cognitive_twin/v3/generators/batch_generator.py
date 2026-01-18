"""
Batch Generator - Leverages GPT-5-mini's 400K context window for efficient batch generation.

Instead of making individual API calls for each prompt, we batch 50-100 prompts
into a single call with accumulated context, then parse the responses.

This reduces:
- API latency (fewer round trips)
- Cost (shared context tokens)
- Time (parallel batch processing)
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BatchConfig:
    """Configuration for batch generation."""
    
    # Batch sizing
    max_prompts_per_batch: int = 50  # Conservative to fit in 400K
    max_tokens_per_batch: int = 280_000  # Leave room for output
    tokens_per_char: float = 0.25  # Rough estimate
    
    # Output limits
    max_output_tokens: int = 128_000
    target_tokens_per_response: int = 2000  # Average expected response length
    
    # Parallelism
    concurrent_batches: int = 3  # Process 3 batches in parallel
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Quality
    min_response_length: int = 50


# ============================================================================
# Batch Prompt Formatter
# ============================================================================

BATCH_SYSTEM_PROMPT = """You are an AI assistant generating multiple responses in a single batch.

CRITICAL INSTRUCTIONS:
1. You will receive multiple numbered prompts
2. Respond to EACH prompt with a complete, high-quality response
3. Format your output as a JSON array with one object per prompt
4. Each object must have: {{"id": <number>, "response": "<your response>"}}
5. Never ask clarifying questions - execute the task directly
6. Never say "Would you like..." or "Should I..." or "Let me know if..."
7. Provide complete implementations without omissions
8. Use numbered lists for multi-step processes

STYLE GUIDELINES:
{style_context}

OUTPUT FORMAT:
```json
[
  {{"id": 1, "response": "Your complete response to prompt 1..."}},
  {{"id": 2, "response": "Your complete response to prompt 2..."}}
]
```
"""

BATCH_USER_TEMPLATE = """Generate responses for these {count} prompts:

{prompts_section}

Remember: Output a JSON array with one response object per prompt. Execute directly without asking questions."""


@dataclass
class BatchPrompt:
    """A single prompt in a batch."""
    id: int
    prompt: str
    source_id: str
    prompt_type: str = "general"
    context: Optional[str] = None
    

@dataclass
class BatchResult:
    """Result of a batch generation."""
    batch_id: int
    responses: List[Dict[str, Any]]
    tokens_used: int = 0
    latency_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    failed_ids: List[int] = field(default_factory=list)


class BatchPromptFormatter:
    """
    Formats multiple prompts for a single API call.
    
    Packs prompts efficiently while respecting token limits,
    and includes global style context once per batch.
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.config.tokens_per_char)
    
    def create_batches(
        self,
        prompts: List[BatchPrompt],
        style_context: str = "",
    ) -> List[Tuple[List[BatchPrompt], str]]:
        """
        Create batches of prompts with formatted system/user messages.
        
        Args:
            prompts: List of prompts to batch
            style_context: Global style guidelines to include
            
        Returns:
            List of (batch_prompts, formatted_user_message) tuples
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        # Base overhead: system prompt + style context
        system_prompt = BATCH_SYSTEM_PROMPT.format(style_context=style_context or "Be concise and direct.")
        base_overhead = self.estimate_tokens(system_prompt) + 500  # Buffer
        
        for prompt in prompts:
            prompt_tokens = self.estimate_tokens(prompt.prompt) + 100  # ID + formatting
            estimated_response = self.config.target_tokens_per_response
            
            # Check if adding this prompt would exceed limits
            if (len(current_batch) >= self.config.max_prompts_per_batch or
                current_tokens + prompt_tokens + estimated_response > self.config.max_tokens_per_batch - base_overhead):
                # Finalize current batch
                if current_batch:
                    formatted = self._format_batch(current_batch)
                    batches.append((current_batch.copy(), formatted))
                
                # Start new batch
                current_batch = [prompt]
                current_tokens = prompt_tokens + estimated_response
            else:
                current_batch.append(prompt)
                current_tokens += prompt_tokens + estimated_response
        
        # Don't forget the last batch
        if current_batch:
            formatted = self._format_batch(current_batch)
            batches.append((current_batch.copy(), formatted))
        
        logger.info(f"Created {len(batches)} batches from {len(prompts)} prompts")
        return batches
    
    def _format_batch(self, batch: List[BatchPrompt]) -> str:
        """Format a batch of prompts into a user message."""
        prompts_section = ""
        for i, bp in enumerate(batch, 1):
            prompts_section += f"--- PROMPT {bp.id} ---\n"
            if bp.context:
                prompts_section += f"Context: {bp.context[:500]}...\n"
            prompts_section += f"{bp.prompt}\n\n"
        
        return BATCH_USER_TEMPLATE.format(
            count=len(batch),
            prompts_section=prompts_section.strip()
        )
    
    def get_system_prompt(self, style_context: str = "") -> str:
        """Get the system prompt with style context."""
        return BATCH_SYSTEM_PROMPT.format(
            style_context=style_context or "Be concise and direct."
        )


# ============================================================================
# Batch Response Parser
# ============================================================================

class BatchResponseParser:
    """
    Parses batch responses from the model.
    
    Handles various output formats and gracefully recovers from parsing failures.
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
    
    def parse(
        self,
        response_text: str,
        expected_ids: List[int],
    ) -> Tuple[Dict[int, str], List[int]]:
        """
        Parse batch response and extract individual responses.
        
        Args:
            response_text: Raw response from the model
            expected_ids: List of prompt IDs we expect responses for
            
        Returns:
            Tuple of (id_to_response dict, list of failed/missing IDs)
        """
        parsed = {}
        failed = []
        
        # Try JSON parsing first
        json_parsed = self._try_json_parse(response_text)
        if json_parsed:
            for item in json_parsed:
                if isinstance(item, dict) and "id" in item and "response" in item:
                    parsed[item["id"]] = item["response"]
        
        # If JSON failed or incomplete, try regex extraction
        if len(parsed) < len(expected_ids) * 0.5:
            regex_parsed = self._try_regex_parse(response_text)
            for id_, response in regex_parsed.items():
                if id_ not in parsed:
                    parsed[id_] = response
        
        # Identify missing IDs
        for id_ in expected_ids:
            if id_ not in parsed:
                failed.append(id_)
        
        # Validate responses
        for id_, response in list(parsed.items()):
            if len(response) < self.config.min_response_length:
                failed.append(id_)
                del parsed[id_]
        
        return parsed, failed
    
    def _try_json_parse(self, text: str) -> Optional[List[Dict]]:
        """Try to parse response as JSON."""
        # Find JSON array in the response
        matches = re.findall(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue
        
        # Try parsing the whole thing as JSON
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _try_regex_parse(self, text: str) -> Dict[int, str]:
        """Try to extract responses using regex patterns."""
        parsed = {}
        
        # Pattern 1: {"id": N, "response": "..."}
        pattern1 = r'"id"\s*:\s*(\d+)\s*,\s*"response"\s*:\s*"((?:[^"\\]|\\.)*)\"'
        for match in re.finditer(pattern1, text):
            id_ = int(match.group(1))
            response = match.group(2).replace('\\"', '"').replace('\\n', '\n')
            parsed[id_] = response
        
        # Pattern 2: --- RESPONSE N --- followed by content
        pattern2 = r'---\s*RESPONSE\s*(\d+)\s*---\s*(.*?)(?=---\s*RESPONSE|\Z)'
        for match in re.finditer(pattern2, text, re.DOTALL):
            id_ = int(match.group(1))
            response = match.group(2).strip()
            if id_ not in parsed:
                parsed[id_] = response
        
        return parsed


# ============================================================================
# Batch Generator
# ============================================================================

class BatchGenerator:
    """
    Main batch generation orchestrator.
    
    Uses GPT-5-mini's 400K context window to process multiple prompts
    in parallel batches, dramatically reducing API costs and time.
    
    Features:
    - Progress tracking with estimated time remaining
    - Checkpointing for resume on failure
    - Parallel batch processing
    """
    
    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        openai_api_key: Optional[str] = None,
        together_api_key: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.config = config or BatchConfig()
        self.formatter = BatchPromptFormatter(self.config)
        self.parser = BatchResponseParser(self.config)
        self.openai_api_key = openai_api_key
        self.together_api_key = together_api_key
        self.checkpoint_dir = checkpoint_dir
        
        # Stats
        self.total_prompts_processed = 0
        self.total_tokens_used = 0
        self.total_batches_completed = 0
        self.total_failures = 0
        
        # Progress tracking
        self._start_time: Optional[float] = None
        self._total_batches: int = 0
        self._completed_batches: int = 0
        self._checkpoint_data: Dict[str, Any] = {}
    
    async def generate_preferred_batch(
        self,
        prompts: List[BatchPrompt],
        style_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Generate preferred responses for a batch of prompts using GPT-5-mini.
        
        Args:
            prompts: List of prompts to process
            style_context: Global style guidelines
            
        Returns:
            List of response dictionaries with prompt metadata
        """
        if not prompts:
            return []
        
        # Initialize progress tracking
        self._start_time = time.time()
        self._completed_batches = 0
        
        # Load checkpoint if resuming
        start_batch, existing_responses = self._load_checkpoint()
        
        # Create batches
        batches = self.formatter.create_batches(prompts, style_context)
        self._total_batches = len(batches)
        
        logger.info(f"Processing {len(prompts)} prompts in {len(batches)} batches")
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch}")
        
        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_batches)
        
        async def process_with_semaphore(batch_id: int, batch_prompts: List[BatchPrompt], user_msg: str):
            async with semaphore:
                result = await self._process_single_batch(
                    batch_id, batch_prompts, user_msg, style_context, "preferred"
                )
                self._log_progress(batch_id, len(batches), result)
                self._save_checkpoint(batch_id, result.responses)
                return result
        
        # Launch batches (skip already completed ones)
        tasks = []
        for i, (batch_prompts, user_msg) in enumerate(batches):
            if i >= start_batch:
                tasks.append(process_with_semaphore(i, batch_prompts, user_msg))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all responses (include existing from checkpoint)
        all_responses = existing_responses.copy()
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch failed with exception: {result}")
                self.total_failures += 1
            elif isinstance(result, BatchResult):
                all_responses.extend(result.responses)
                self.total_tokens_used += result.tokens_used
                self.total_batches_completed += 1
                self.total_failures += len(result.failed_ids)
        
        self.total_prompts_processed += len(prompts)
        
        # Clear checkpoint on success
        self.clear_checkpoint()
        
        elapsed = time.time() - self._start_time
        logger.info(f"Batch generation complete: {len(all_responses)}/{len(prompts)} successful in {elapsed:.1f}s")
        
        return all_responses
    
    async def generate_dispreferred_batch(
        self,
        prompts: List[BatchPrompt],
        style_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Generate dispreferred responses using base model (Together AI).
        
        Args:
            prompts: List of prompts to process
            style_context: Context to intentionally NOT follow (for contrast)
            
        Returns:
            List of response dictionaries
        """
        if not prompts:
            return []
        
        # For dispreferred, we use smaller batches since Llama has less context
        smaller_config = BatchConfig(
            max_prompts_per_batch=20,
            max_tokens_per_batch=60_000,  # Llama's limit is lower
            concurrent_batches=5,
        )
        formatter = BatchPromptFormatter(smaller_config)
        
        batches = formatter.create_batches(prompts, "")  # No style context for dispreferred
        
        logger.info(f"Processing {len(prompts)} dispreferred prompts in {len(batches)} batches")
        
        semaphore = asyncio.Semaphore(smaller_config.concurrent_batches)
        
        async def process_with_semaphore(batch_id: int, batch_prompts: List[BatchPrompt], user_msg: str):
            async with semaphore:
                return await self._process_single_batch(
                    batch_id, batch_prompts, user_msg, "", "dispreferred"
                )
        
        tasks = []
        for i, (batch_prompts, user_msg) in enumerate(batches):
            tasks.append(process_with_semaphore(i, batch_prompts, user_msg))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_responses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Dispreferred batch failed: {result}")
            elif isinstance(result, BatchResult):
                all_responses.extend(result.responses)
        
        return all_responses
    
    async def generate_dpo_pairs_batch(
        self,
        prompts: List[BatchPrompt],
        style_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Generate DPO pairs (preferred + dispreferred) for a batch of prompts.
        
        Runs both generations in parallel for efficiency.
        
        Returns:
            List of DPO pair dictionaries
        """
        # Run preferred and dispreferred in parallel
        preferred_task = self.generate_preferred_batch(prompts, style_context)
        dispreferred_task = self.generate_dispreferred_batch(prompts, "")
        
        preferred_results, dispreferred_results = await asyncio.gather(
            preferred_task, dispreferred_task
        )
        
        # Match up pairs by prompt ID
        preferred_by_id = {r["prompt_id"]: r for r in preferred_results}
        dispreferred_by_id = {r["prompt_id"]: r for r in dispreferred_results}
        
        dpo_pairs = []
        for prompt in prompts:
            pref = preferred_by_id.get(prompt.id)
            dispref = dispreferred_by_id.get(prompt.id)
            
            if pref and dispref:
                dpo_pairs.append({
                    "prompt_id": prompt.id,
                    "source_id": prompt.source_id,
                    "prompt": prompt.prompt,
                    "chosen": pref["response"],
                    "rejected": dispref["response"],
                    "chosen_model": pref.get("model", "gpt-5-mini"),
                    "rejected_model": dispref.get("model", "llama-3.2-3b"),
                })
        
        logger.info(f"Generated {len(dpo_pairs)} DPO pairs from {len(prompts)} prompts")
        
        return dpo_pairs
    
    async def _process_single_batch(
        self,
        batch_id: int,
        batch_prompts: List[BatchPrompt],
        user_msg: str,
        style_context: str,
        mode: str,  # "preferred" or "dispreferred"
    ) -> BatchResult:
        """Process a single batch through the API."""
        start_time = time.time()
        expected_ids = [bp.id for bp in batch_prompts]
        
        for attempt in range(self.config.max_retries):
            try:
                if mode == "preferred":
                    response_text, tokens = await self._call_openai(user_msg, style_context)
                    model = "gpt-5-mini-2025-08-07"
                else:
                    response_text, tokens = await self._call_together(user_msg)
                    model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
                
                # Parse the response
                parsed, failed = self.parser.parse(response_text, expected_ids)
                
                # Build response list
                responses = []
                for bp in batch_prompts:
                    if bp.id in parsed:
                        responses.append({
                            "prompt_id": bp.id,
                            "source_id": bp.source_id,
                            "prompt": bp.prompt,
                            "response": parsed[bp.id],
                            "model": model,
                        })
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                logger.info(f"Batch {batch_id}: {len(responses)}/{len(batch_prompts)} parsed ({latency_ms}ms)")
                
                return BatchResult(
                    batch_id=batch_id,
                    responses=responses,
                    tokens_used=tokens,
                    latency_ms=latency_ms,
                    success=True,
                    failed_ids=failed,
                )
                
            except Exception as e:
                logger.warning(f"Batch {batch_id} attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return BatchResult(
            batch_id=batch_id,
            responses=[],
            success=False,
            error=f"Failed after {self.config.max_retries} attempts",
            failed_ids=expected_ids,
        )
    
    async def _call_openai(self, user_msg: str, style_context: str) -> Tuple[str, int]:
        """Call OpenAI API for preferred generation."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed: pip install openai")
        
        api_key = self.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        client = openai.OpenAI(api_key=api_key)
        
        system_prompt = self.formatter.get_system_prompt(style_context)
        
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=self.config.max_output_tokens,
            temperature=1.0,  # GPT-5-mini only supports default
        )
        
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        
        return content, tokens
    
    async def _call_together(self, user_msg: str) -> Tuple[str, int]:
        """Call Together AI API for dispreferred generation."""
        try:
            import together
        except ImportError:
            raise ImportError("together package not installed: pip install together")
        
        api_key = self.together_api_key or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY required")
        
        client = together.Together(api_key=api_key)
        
        # Simpler prompt for dispreferred - encourage hedging
        system_prompt = """You are an AI assistant generating multiple responses.
Output format: JSON array with {"id": N, "response": "..."} for each prompt.
Be helpful but thorough - explain options, ask for clarification if needed."""
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=16000,
            temperature=0.9,
        )
        
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        
        return content, tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        return {
            "total_prompts_processed": self.total_prompts_processed,
            "total_tokens_used": self.total_tokens_used,
            "total_batches_completed": self.total_batches_completed,
            "total_failures": self.total_failures,
            "success_rate": (
                (self.total_prompts_processed - self.total_failures) / 
                max(1, self.total_prompts_processed)
            ),
            "elapsed_seconds": elapsed,
            "prompts_per_second": self.total_prompts_processed / max(1, elapsed),
        }
    
    def _log_progress(self, batch_id: int, total_batches: int, batch_result: BatchResult):
        """Log progress with estimated time remaining."""
        self._completed_batches += 1
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        if self._completed_batches > 0 and elapsed > 0:
            avg_time_per_batch = elapsed / self._completed_batches
            remaining_batches = total_batches - self._completed_batches
            eta_seconds = avg_time_per_batch * remaining_batches
            
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds / 3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds / 60:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"
            
            success_count = len(batch_result.responses)
            total_count = success_count + len(batch_result.failed_ids)
            
            logger.info(
                f"    Batch {batch_id + 1}/{total_batches}: "
                f"{success_count}/{total_count} responses, "
                f"{batch_result.tokens_used:,} tokens, "
                f"ETA: {eta_str}"
            )
    
    def _save_checkpoint(self, batch_id: int, responses: List[Dict[str, Any]]):
        """Save checkpoint for resume capability."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"batch_checkpoint.json")
        
        self._checkpoint_data["last_batch_id"] = batch_id
        self._checkpoint_data["responses"] = self._checkpoint_data.get("responses", [])
        self._checkpoint_data["responses"].extend(responses)
        self._checkpoint_data["timestamp"] = time.time()
        
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(self._checkpoint_data, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> Tuple[int, List[Dict[str, Any]]]:
        """Load checkpoint if exists."""
        if not self.checkpoint_dir:
            return 0, []
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"batch_checkpoint.json")
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    last_batch = data.get("last_batch_id", 0)
                    responses = data.get("responses", [])
                    logger.info(f"Resuming from checkpoint: batch {last_batch}, {len(responses)} responses")
                    return last_batch + 1, responses
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        
        return 0, []
    
    def clear_checkpoint(self):
        """Clear checkpoint after successful completion."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"batch_checkpoint.json")
        
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.info("Checkpoint cleared")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")


# Import for os.environ
import os

