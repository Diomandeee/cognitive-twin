"""
Assistant Rewriter for CognitiveTwin V3.

Rewrites unjustified clarification turns into direct execution.
Supports both OpenAI GPT 5.2 and the fine-tuned V2 model (via Together AI).

The V2 model is preferred for rewriting because it has already learned
the user's style and correction patterns.
"""

import re
from typing import List, Optional, Union

from .types import (
    FormatConstraints,
    ValidationResult,
    ClassificationResult,
)
from .constants import (
    STRONG_PERMISSION_PHRASES,
    OPTION_DUMPING_PHRASES,
    PROVIDER_ISMS,
    DIRECTIVE_REWRITE_THRESHOLD,
)

# Import clients - try both
try:
    from ..api.openai_client import V3OpenAIClient
except ImportError:
    V3OpenAIClient = None

try:
    from ..generators.v2_generator import V2Generator, GenerationResult
except ImportError:
    V2Generator = None


# =============================================================================
# REWRITER PROMPTS
# =============================================================================

REWRITER_SYSTEM_PROMPT = """You are a rewriter that transforms assistant responses from permission-seeking to direct execution.

Your task: Take an assistant response that asks for permission or clarification when it's not needed, and rewrite it to execute directly.

Rules:
1. NEVER end with a question
2. NEVER use permission phrases like "Would you like me to...", "Should I...", "Can you confirm..."
3. NEVER dump options when the user didn't ask for them
4. If the user's request has minor ambiguities, make reasonable assumptions and state them
5. Execute the requested task directly
6. Produce the artifact (code, text, analysis) that was requested
7. Match the format constraints specified in the user's message
8. Be concise but complete

Format your response as the assistant's direct answer, ready to replace the original."""

REWRITER_USER_TEMPLATE = """Here is the context:

CONVERSATION HISTORY:
{history}

USER MESSAGE:
{user_message}

ORIGINAL ASSISTANT RESPONSE (unjustified clarification):
{assistant_message}

FORMAT CONSTRAINTS:
{format_constraints}

Rewrite the assistant response to execute directly without asking for permission."""


def format_history(history: List[dict]) -> str:
    """Format conversation history for prompt."""
    if not history:
        return "(No prior messages)"
    
    lines = []
    for turn in history[-6:]:  # Last 6 turns max
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")[:500]
        if len(turn.get("content", "")) > 500:
            content += "..."
        lines.append(f"{role}: {content}")
    
    return "\n\n".join(lines)


def format_constraints(constraints: FormatConstraints) -> str:
    """Format constraints for prompt."""
    parts = []
    
    if constraints.forbid_bullets:
        parts.append("- No bullet points")
    if constraints.require_numbered:
        parts.append("- Use numbered lists")
    if constraints.must_return_code:
        parts.append("- Must include code")
    if constraints.must_return_diff:
        parts.append("- Must include diff")
    if constraints.must_return_json:
        parts.append("- Must return JSON")
    if constraints.must_not_omit:
        parts.append("- Do not omit any content")
    
    if not parts:
        return "(No specific format constraints)"
    
    return "\n".join(parts)


# =============================================================================
# REWRITER FUNCTIONS
# =============================================================================

async def rewrite_assistant_turn(
    assistant_message: str,
    user_message: str,
    conversation_history: List[dict],
    format_constraints: FormatConstraints,
    client: V3OpenAIClient,
    temperature: float = 0.3,
    max_retries: int = 2,
) -> str:
    """
    Rewrite an unjustified clarification into direct execution.
    
    Args:
        assistant_message: Original assistant response
        user_message: Preceding user message
        conversation_history: Prior conversation turns
        format_constraints: Format requirements from user
        client: OpenAI client
        temperature: Generation temperature
        max_retries: Number of retries if validation fails
    
    Returns:
        Rewritten assistant response
    """
    # Build prompt
    user_prompt = REWRITER_USER_TEMPLATE.format(
        history=format_history(conversation_history),
        user_message=user_message,
        assistant_message=assistant_message,
        format_constraints=format_constraints_to_str(format_constraints),
    )
    
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    for attempt in range(max_retries + 1):
        # Generate rewrite
        rewritten = await client.chat_complete_async(
            messages=messages,
            temperature=temperature + (attempt * 0.1),  # Increase temp on retry
        )
        
        # Validate
        validation = validate_rewrite(rewritten, user_message, format_constraints)
        
        if validation.is_valid:
            return rewritten
        
        # Add validation feedback for retry
        if attempt < max_retries:
            messages.append({
                "role": "assistant",
                "content": rewritten,
            })
            messages.append({
                "role": "user",
                "content": f"This rewrite has issues: {', '.join(validation.errors)}. Try again.",
            })
    
    # Return best attempt even if validation failed
    return rewritten


def format_constraints_to_str(constraints: FormatConstraints) -> str:
    """Convert constraints to string for prompt."""
    return format_constraints(constraints)


def rewrite_assistant_turn_sync(
    assistant_message: str,
    user_message: str,
    conversation_history: List[dict],
    format_constraints: FormatConstraints,
    client: V3OpenAIClient,
) -> str:
    """Synchronous wrapper for rewrite_assistant_turn."""
    import asyncio
    
    return asyncio.run(rewrite_assistant_turn(
        assistant_message=assistant_message,
        user_message=user_message,
        conversation_history=conversation_history,
        format_constraints=format_constraints,
        client=client,
    ))


# =============================================================================
# VALIDATION
# =============================================================================

def validate_rewrite(
    rewritten: str,
    user_message: str,
    format_constraints: FormatConstraints,
) -> ValidationResult:
    """
    Validate that a rewritten response meets requirements.
    
    Checks:
    1. Does not end with question mark
    2. Does not contain permission phrases
    3. Contains required artifact if applicable
    4. Follows format constraints
    
    Returns:
        ValidationResult with is_valid and list of errors
    """
    errors = []
    normalized = rewritten.lower()
    
    # Check 1: Does not end with question
    stripped = rewritten.rstrip()
    if stripped.endswith('?'):
        errors.append("ends_with_question")
    
    # Check 2: No permission phrases
    for phrase in STRONG_PERMISSION_PHRASES:
        if phrase in normalized:
            errors.append(f"contains_permission_phrase: {phrase}")
            break  # Only report first
    
    for phrase in OPTION_DUMPING_PHRASES:
        # Only flag option dumping if no options were requested
        if phrase in normalized and "option" not in user_message.lower():
            errors.append(f"contains_option_dump: {phrase}")
            break
    
    # Check 3: Contains required artifact
    if format_constraints.must_return_code:
        if not re.search(r"```[\s\S]*?```", rewritten):
            errors.append("missing_code_block")
    
    if format_constraints.must_return_diff:
        if not re.search(r"^[\+\-]", rewritten, re.MULTILINE):
            errors.append("missing_diff_markers")
    
    if format_constraints.must_return_json:
        if not re.search(r'[{\[].*[}\]]', rewritten, re.DOTALL):
            errors.append("missing_json")
    
    # Check 4: Format constraints
    if format_constraints.forbid_bullets:
        if re.search(r'^\s*[-*•]\s+', rewritten, re.MULTILINE):
            errors.append("contains_bullets")
    
    if format_constraints.require_numbered:
        if not re.search(r'^\s*\d+[.)]\s+', rewritten, re.MULTILINE):
            errors.append("missing_numbered_list")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
    )


def strip_provider_isms(text: str) -> str:
    """Remove provider-specific phrases from text."""
    result = text
    
    for pattern in PROVIDER_ISMS:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up excess whitespace
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'^\s+', '', result)
    
    return result


def should_rewrite(
    classification: ClassificationResult,
) -> bool:
    """
    Determine if a turn should be rewritten.
    
    Conditions:
    - Classification is UNJUSTIFIED
    - Directive completeness >= threshold
    - Exec score == 0 (didn't execute)
    """
    from .types import ClarificationType
    
    if classification.classification != ClarificationType.UNJUSTIFIED:
        return False
    
    if classification.directive_completeness < DIRECTIVE_REWRITE_THRESHOLD:
        return False
    
    if classification.exec_score > 0:
        return False
    
    return True


# =============================================================================
# ASSUMPTION PROTOCOL
# =============================================================================

ASSUMPTION_PATTERNS = {
    "file_type": "Assuming {assumed} format",
    "language": "Assuming {assumed} language",
    "scope": "Assuming {assumed} scope",
    "style": "Assuming {assumed} style",
}


def generate_assumption_prefix(
    user_message: str,
    context: Optional[dict] = None,
) -> str:
    """
    Generate assumption prefix for ambiguous requests.
    
    Instead of asking for clarification, state assumptions explicitly.
    """
    assumptions = []
    user_lower = user_message.lower()
    
    # Detect ambiguities and generate assumptions
    if "code" in user_lower or "function" in user_lower:
        if not any(lang in user_lower for lang in ["python", "javascript", "typescript", "rust"]):
            assumptions.append("Assuming Python (most common in this codebase)")
    
    if "refactor" in user_lower or "rewrite" in user_lower:
        if "```" not in user_message:
            assumptions.append("Assuming the referenced code from context")
    
    if not assumptions:
        return ""
    
    return "**Assumptions**: " + "; ".join(assumptions) + "\n\n"


# =============================================================================
# V2 GENERATOR REWRITING (PREFERRED)
# =============================================================================

async def rewrite_with_v2(
    assistant_message: str,
    user_message: str,
    generator: "V2Generator",
    format_constraints: Optional[FormatConstraints] = None,
    max_retries: int = 2,
) -> str:
    """
    Rewrite an unjustified turn using the fine-tuned V2 model.
    
    The V2 model is preferred for rewriting because it has already learned
    from the user's corrections and naturally produces style-aligned responses.
    
    Args:
        assistant_message: Original assistant response
        user_message: Preceding user message
        generator: V2Generator instance
        format_constraints: Optional format requirements
        max_retries: Number of retries if validation fails
    
    Returns:
        Rewritten assistant response
    """
    if V2Generator is None:
        raise ImportError("V2Generator not available. Install together package.")
    
    # Build format constraints string
    constraint_str = ""
    if format_constraints:
        constraint_str = format_constraints_to_str(format_constraints)
    
    for attempt in range(max_retries + 1):
        # Use V2's rewrite capability
        result = await generator.rewrite_turn(
            user_message=user_message,
            bad_assistant_message=assistant_message,
        )
        
        if not result.success:
            if attempt < max_retries:
                continue
            # Return original on failure
            return assistant_message
        
        rewritten = result.content
        
        # Validate
        if format_constraints:
            validation = validate_rewrite(rewritten, user_message, format_constraints)
            
            if validation.is_valid:
                return rewritten
            
            if attempt < max_retries:
                continue
        else:
            # No format constraints, just basic validation
            stripped = rewritten.rstrip()
            if not stripped.endswith('?'):
                return rewritten
    
    return rewritten


async def rewrite_batch_with_v2(
    turns: List[dict],
    generator: "V2Generator",
    concurrency: int = 5,
) -> List[dict]:
    """
    Rewrite multiple turns using V2 generator.
    
    Args:
        turns: List of dicts with assistant_message and user_message
        generator: V2Generator instance
        concurrency: Max concurrent requests
    
    Returns:
        List of results with original and rewritten content
    """
    import asyncio
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_turn(turn: dict) -> dict:
        async with semaphore:
            constraints = turn.get("format_constraints")
            if constraints is not None and isinstance(constraints, dict):
                constraints = FormatConstraints(**constraints)
            
            try:
                rewritten = await rewrite_with_v2(
                    assistant_message=turn["assistant_message"],
                    user_message=turn["user_message"],
                    generator=generator,
                    format_constraints=constraints,
                )
                
                validation_result = ValidationResult(is_valid=True, errors=[])
                if constraints:
                    validation_result = validate_rewrite(
                        rewritten,
                        turn["user_message"],
                        constraints,
                    )
                
                return {
                    "original": turn["assistant_message"],
                    "rewritten": rewritten,
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "success": True,
                }
            except Exception as e:
                return {
                    "original": turn["assistant_message"],
                    "rewritten": None,
                    "is_valid": False,
                    "errors": [str(e)],
                    "success": False,
                }
    
    results = await asyncio.gather(*[process_turn(t) for t in turns])
    return list(results)


# =============================================================================
# BATCH REWRITING (OpenAI)
# =============================================================================

async def rewrite_batch(
    turns: List[dict],
    client: V3OpenAIClient,
    concurrency: int = 5,
) -> List[dict]:
    """
    Rewrite multiple turns concurrently.
    
    Args:
        turns: List of dicts with keys:
            - assistant_message
            - user_message
            - history (optional)
            - format_constraints (optional)
        client: OpenAI client
        concurrency: Max concurrent requests
    
    Returns:
        List of results with original and rewritten content
    """
    import asyncio
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_turn(turn: dict) -> dict:
        async with semaphore:
            constraints = turn.get("format_constraints")
            if constraints is None:
                from .classifier import extract_format_constraints
                constraints = extract_format_constraints(turn["user_message"])
            elif isinstance(constraints, dict):
                constraints = FormatConstraints(**constraints)
            
            try:
                rewritten = await rewrite_assistant_turn(
                    assistant_message=turn["assistant_message"],
                    user_message=turn["user_message"],
                    conversation_history=turn.get("history", []),
                    format_constraints=constraints,
                    client=client,
                )
                
                validation = validate_rewrite(
                    rewritten,
                    turn["user_message"],
                    constraints,
                )
                
                return {
                    "original": turn["assistant_message"],
                    "rewritten": rewritten,
                    "is_valid": validation.is_valid,
                    "errors": validation.errors,
                    "success": True,
                }
            except Exception as e:
                return {
                    "original": turn["assistant_message"],
                    "rewritten": None,
                    "is_valid": False,
                    "errors": [str(e)],
                    "success": False,
                }
    
    results = await asyncio.gather(*[process_turn(t) for t in turns])
    return list(results)

