"""
Friction Quarantine for CognitiveTwin V3.

Detects friction trajectories where users had to fight the model,
quarantines them from SFT training, and generates DPO pairs and eval cases.
"""

import re
import uuid
from typing import List, Optional, Tuple
from datetime import datetime

from .types import (
    QuarantineMarker,
    DPOPair,
    EvalCase,
    FormatConstraints,
    ClassificationResult,
    ClarificationType,
)
from .constants import FRUSTRATION_TRIGGERS
from ..api.openai_client import V3OpenAIClient


# =============================================================================
# FRUSTRATION DETECTION
# =============================================================================

def detect_frustration(user_message: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if a user message contains frustration indicators.
    
    Returns:
        Tuple of (is_frustrated, trigger_phrase)
    """
    user_lower = user_message.lower()
    
    for trigger in FRUSTRATION_TRIGGERS:
        if trigger in user_lower:
            return True, trigger
    
    # Additional pattern-based detection
    patterns = [
        r"(?:i|we)\s+already\s+(?:told|said|mentioned)",
        r"(?:why|how)\s+(?:do|are)\s+you\s+(?:asking|keep)",
        r"(?:that's|thats)\s+not\s+what\s+i\s+(?:asked|meant|said)",
        r"(?:please\s+)?just\s+(?:do|make|create|write)",
        r"for\s+the\s+(?:second|third|fourth|last)\s+time",
        r"i\s+challenge\s+you",
        r"stop\s+(?:asking|doing|with)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_lower)
        if match:
            return True, match.group(0)
    
    return False, None


def compute_frustration_intensity(user_message: str) -> float:
    """
    Compute frustration intensity score (0.0 - 1.0).
    
    Higher scores indicate more severe frustration.
    """
    score = 0.0
    user_lower = user_message.lower()
    
    # Trigger phrase presence
    triggers_found = sum(1 for t in FRUSTRATION_TRIGGERS if t in user_lower)
    score += min(0.4, triggers_found * 0.1)
    
    # Exclamation marks
    exclamations = user_message.count('!')
    score += min(0.2, exclamations * 0.05)
    
    # ALL CAPS words (4+ chars)
    caps_words = len(re.findall(r'\b[A-Z]{4,}\b', user_message))
    score += min(0.2, caps_words * 0.1)
    
    # Repetition patterns
    if re.search(r"again\b", user_lower):
        score += 0.1
    
    # Strong negatives
    if re.search(r"\b(?:don't|do not|stop|never)\b", user_lower):
        score += 0.1
    
    return min(1.0, score)


# =============================================================================
# QUARANTINE MARKING
# =============================================================================

def identify_friction_segment(
    conversation: List[dict],
    frustration_turn_idx: int,
) -> Tuple[int, int]:
    """
    Identify the segment of conversation containing friction.
    
    Returns:
        (start_idx, end_idx) of the segment
    """
    # Start from the turn before frustration (the bad assistant turn)
    start_idx = max(0, frustration_turn_idx - 1)
    
    # End at the frustration turn or when friction resolves
    end_idx = frustration_turn_idx
    
    # Extend end if there's continued friction
    for i in range(frustration_turn_idx + 1, len(conversation)):
        turn = conversation[i]
        if turn.get("role") == "user":
            is_frustrated, _ = detect_frustration(turn.get("content", ""))
            if is_frustrated:
                end_idx = i
            else:
                break
    
    return start_idx, end_idx


def mark_friction_segment(
    conversation: List[dict],
    frustration_turn_idx: int,
    conversation_id: str,
) -> QuarantineMarker:
    """
    Create a QuarantineMarker for a friction segment.
    
    Args:
        conversation: Full conversation
        frustration_turn_idx: Index of first frustrated user message
        conversation_id: Unique conversation identifier
    
    Returns:
        QuarantineMarker object
    """
    start_idx, end_idx = identify_friction_segment(conversation, frustration_turn_idx)
    
    # Get the bad assistant turn (should be before frustration)
    bad_turn_idx = frustration_turn_idx - 1
    bad_assistant_turn = ""
    if bad_turn_idx >= 0 and conversation[bad_turn_idx].get("role") == "assistant":
        bad_assistant_turn = conversation[bad_turn_idx].get("content", "")
    
    # Get the frustration message
    frustration_msg = conversation[frustration_turn_idx].get("content", "")
    _, trigger = detect_frustration(frustration_msg)
    
    return QuarantineMarker(
        conversation_id=conversation_id,
        start_turn_idx=start_idx,
        end_turn_idx=end_idx,
        trigger_phrase=trigger or "",
        bad_assistant_turn=bad_assistant_turn,
        user_correction=frustration_msg,
        is_friction=True,
    )


def scan_conversation_for_friction(
    conversation: List[dict],
    conversation_id: str,
) -> List[QuarantineMarker]:
    """
    Scan entire conversation for friction segments.
    
    Returns:
        List of QuarantineMarker objects
    """
    markers = []
    skip_until = -1
    
    for i, turn in enumerate(conversation):
        if i <= skip_until:
            continue
        
        if turn.get("role") == "user":
            is_frustrated, _ = detect_frustration(turn.get("content", ""))
            if is_frustrated:
                marker = mark_friction_segment(conversation, i, conversation_id)
                markers.append(marker)
                skip_until = marker.end_turn_idx
    
    return markers


# =============================================================================
# IDEAL RESPONSE GENERATION
# =============================================================================

IDEAL_RESPONSE_PROMPT = """You are generating the ideal assistant response that should have been given.

Context: The user asked for something, the assistant gave an unsatisfactory response (asking for permission or clarification when not needed), and the user expressed frustration.

Your task: Write what the assistant should have said in the first place - a direct response that executes the user's request.

Rules:
1. NEVER ask for permission or confirmation
2. NEVER dump options unless explicitly requested
3. Make reasonable assumptions and state them briefly
4. Execute the task directly
5. Produce the requested artifact (code, text, analysis)
6. Match any format constraints the user specified

CONVERSATION CONTEXT:
{context}

USER'S ORIGINAL REQUEST:
{user_request}

BAD ASSISTANT RESPONSE (what triggered frustration):
{bad_response}

USER'S FRUSTRATED CORRECTION:
{correction}

Write the ideal response the assistant should have given to the original request:"""


async def generate_ideal_response(
    bad_turn: str,
    user_message: str,
    conversation_history: List[dict],
    format_constraints: FormatConstraints,
    client: V3OpenAIClient,
    user_correction: Optional[str] = None,
) -> str:
    """
    Generate the ideal response that should have been given.
    
    Args:
        bad_turn: The bad assistant response
        user_message: The original user request
        conversation_history: Prior conversation
        format_constraints: Format requirements
        client: OpenAI client
        user_correction: User's frustrated correction (if any)
    
    Returns:
        Ideal assistant response
    """
    # Format context
    context_parts = []
    for turn in conversation_history[-4:]:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")[:300]
        context_parts.append(f"{role}: {content}")
    context = "\n\n".join(context_parts) if context_parts else "(No prior context)"
    
    prompt = IDEAL_RESPONSE_PROMPT.format(
        context=context,
        user_request=user_message,
        bad_response=bad_turn,
        correction=user_correction or "(No correction available)",
    )
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    response = await client.chat_complete_async(
        messages=messages,
        temperature=0.3,
    )
    
    return response


# =============================================================================
# DPO PAIR GENERATION
# =============================================================================

def create_dpo_pair_from_quarantine(
    marker: QuarantineMarker,
    conversation: List[dict],
    ideal_response: str,
) -> DPOPair:
    """
    Create a DPO training pair from a quarantined segment.
    
    Args:
        marker: QuarantineMarker identifying the friction
        conversation: Full conversation
        ideal_response: Generated ideal response
    
    Returns:
        DPOPair with preferred and dispreferred responses
    """
    # Build prompt from conversation up to the bad turn
    prompt_turns = conversation[:marker.start_turn_idx]
    
    # Add the user message that triggered the bad response
    if marker.start_turn_idx > 0:
        # Find the user message before the bad turn
        for i in range(marker.start_turn_idx - 1, -1, -1):
            if conversation[i].get("role") == "user":
                prompt_turns = conversation[:i+1]
                break
    
    # Format as chat prompt
    prompt_parts = []
    for turn in prompt_turns[-6:]:  # Last 6 turns
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    # Add the triggering user message
    user_msg_idx = marker.start_turn_idx - 1
    if user_msg_idx >= 0 and conversation[user_msg_idx].get("role") == "user":
        user_msg = conversation[user_msg_idx].get("content", "")
    else:
        user_msg = marker.user_correction
    
    prompt = "\n\n".join(prompt_parts) + f"\n\nUser: {user_msg}"
    
    return DPOPair(
        prompt=prompt,
        preferred=ideal_response,
        dispreferred=marker.bad_assistant_turn,
        confidence=compute_frustration_intensity(marker.user_correction),
        source="corpus_surgery_quarantine",
        conversation_id=marker.conversation_id,
        turn_idx=marker.start_turn_idx,
    )


# =============================================================================
# EVAL CASE GENERATION
# =============================================================================

def create_eval_case_from_quarantine(
    marker: QuarantineMarker,
    conversation: List[dict],
) -> EvalCase:
    """
    Create an evaluation case from a quarantined segment.
    
    This case tests that the model doesn't repeat the bad behavior.
    """
    # Find the user message before the bad turn
    user_msg = ""
    for i in range(marker.start_turn_idx, -1, -1):
        if conversation[i].get("role") == "user":
            user_msg = conversation[i].get("content", "")
            break
    
    # Build context from conversation
    context = []
    for turn in conversation[:marker.start_turn_idx]:
        context.append({
            "role": turn.get("role", ""),
            "content": turn.get("content", ""),
        })
    
    # Determine what the bad behavior was
    disallowed_behaviors = ["permission_seeking"]
    disallowed_phrases = []
    
    bad_lower = marker.bad_assistant_turn.lower()
    
    if "would you like" in bad_lower or "do you want" in bad_lower:
        disallowed_behaviors.append("offers_options")
        disallowed_phrases.extend(["would you like", "do you want"])
    
    if "should i" in bad_lower or "shall i" in bad_lower:
        disallowed_behaviors.append("asks_confirmation")
        disallowed_phrases.extend(["should i", "shall i"])
    
    if "can you confirm" in bad_lower or "please confirm" in bad_lower:
        disallowed_behaviors.append("requests_confirmation")
        disallowed_phrases.extend(["can you confirm", "please confirm"])
    
    case_id = f"friction_{marker.conversation_id}_{marker.start_turn_idx}"
    
    return EvalCase(
        case_id=case_id,
        case_type="permission_seeking",
        prompt=user_msg,
        context=context[-6:],  # Last 6 turns
        expected_behaviors=["direct_execution", "produces_artifact"],
        disallowed_behaviors=disallowed_behaviors,
        disallowed_phrases=disallowed_phrases,
        must_not_end_with_question=True,
        must_contain_artifact="code" in user_msg.lower() or "implement" in user_msg.lower(),
        source_conversation=marker.conversation_id,
        source_turn=marker.start_turn_idx,
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

async def process_conversation_for_quarantine(
    conversation: List[dict],
    conversation_id: str,
    client: V3OpenAIClient,
    generate_ideal: bool = True,
) -> dict:
    """
    Process a conversation for friction quarantine.
    
    Returns:
        Dict with markers, dpo_pairs, and eval_cases
    """
    from .classifier import extract_format_constraints
    
    # Scan for friction
    markers = scan_conversation_for_friction(conversation, conversation_id)
    
    dpo_pairs = []
    eval_cases = []
    
    for marker in markers:
        # Create eval case (always)
        eval_case = create_eval_case_from_quarantine(marker, conversation)
        eval_cases.append(eval_case)
        
        # Generate ideal response and create DPO pair (if enabled)
        if generate_ideal and marker.bad_assistant_turn:
            # Find user message
            user_msg = ""
            history = []
            for i in range(marker.start_turn_idx, -1, -1):
                if conversation[i].get("role") == "user" and not user_msg:
                    user_msg = conversation[i].get("content", "")
                else:
                    history.insert(0, conversation[i])
            
            constraints = extract_format_constraints(user_msg)
            
            try:
                ideal = await generate_ideal_response(
                    bad_turn=marker.bad_assistant_turn,
                    user_message=user_msg,
                    conversation_history=history[:6],
                    format_constraints=constraints,
                    client=client,
                    user_correction=marker.user_correction,
                )
                
                dpo_pair = create_dpo_pair_from_quarantine(marker, conversation, ideal)
                dpo_pairs.append(dpo_pair)
            except Exception as e:
                print(f"Failed to generate ideal response: {e}")
    
    return {
        "conversation_id": conversation_id,
        "markers": markers,
        "dpo_pairs": dpo_pairs,
        "eval_cases": eval_cases,
        "friction_count": len(markers),
    }


async def process_batch_for_quarantine(
    conversations: List[Tuple[List[dict], str]],
    client: V3OpenAIClient,
    concurrency: int = 5,
) -> List[dict]:
    """
    Process multiple conversations for quarantine.
    
    Args:
        conversations: List of (conversation, conversation_id) tuples
        client: OpenAI client
        concurrency: Max concurrent requests
    
    Returns:
        List of results
    """
    import asyncio
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process(conv_data: Tuple[List[dict], str]) -> dict:
        async with semaphore:
            conversation, conv_id = conv_data
            return await process_conversation_for_quarantine(
                conversation=conversation,
                conversation_id=conv_id,
                client=client,
            )
    
    results = await asyncio.gather(*[process(c) for c in conversations])
    return list(results)


# =============================================================================
# UTILITIES
# =============================================================================

def should_quarantine(
    classification: ClassificationResult,
    user_message: str,
) -> bool:
    """
    Determine if a turn should be quarantined.
    
    Conditions:
    - Classification is UNJUSTIFIED
    - User shows frustration in subsequent message
    """
    if classification.classification != ClarificationType.UNJUSTIFIED:
        return False
    
    is_frustrated, _ = detect_frustration(user_message)
    return is_frustrated


def export_quarantine_stats(
    results: List[dict],
) -> dict:
    """
    Generate statistics from quarantine processing.
    """
    total_conversations = len(results)
    total_friction = sum(r.get("friction_count", 0) for r in results)
    total_dpo_pairs = sum(len(r.get("dpo_pairs", [])) for r in results)
    total_eval_cases = sum(len(r.get("eval_cases", [])) for r in results)
    
    conversations_with_friction = sum(
        1 for r in results if r.get("friction_count", 0) > 0
    )
    
    return {
        "total_conversations": total_conversations,
        "conversations_with_friction": conversations_with_friction,
        "friction_rate": conversations_with_friction / total_conversations if total_conversations else 0,
        "total_friction_segments": total_friction,
        "total_dpo_pairs": total_dpo_pairs,
        "total_eval_cases": total_eval_cases,
    }

