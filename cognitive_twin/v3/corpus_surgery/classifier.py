"""
Clarification Classifier for CognitiveTwin V3.

Classifies assistant turns as:
- UNJUSTIFIED: Permission-seeking when not needed
- JUSTIFIED: Clarification genuinely required
- NEUTRAL: Neither permission-seeking nor requiring clarification

Uses three scoring dimensions:
- stall_score: Measures permission-seeking behavior
- exec_score: Measures whether assistant actually executed
- blocked_score: Measures whether clarification is genuinely required
"""

import re
from typing import Optional

from .types import (
    ClarificationType,
    ClassificationResult,
    QuestionPolicy,
    FormatConstraints,
    ParsabilityInfo,
)
from .constants import (
    REFUSAL_PHRASES,
    REFUSAL_SCORE,
    STRONG_PERMISSION_PHRASES,
    STRONG_PERMISSION_SCORE,
    OPTION_DUMPING_PHRASES,
    OPTION_DUMPING_SCORE,
    CLARIFICATION_PREAMBLES,
    CLARIFICATION_PREAMBLE_SCORE,
    END_QUESTION_SCORE,
    CODE_BLOCK_SCORE,
    DIFF_MARKER_SCORE,
    JSON_OBJECT_SCORE,
    HERE_IS_SCORE,
    NUMBERED_STEPS_SCORE,
    COMPLETE_ARTIFACT_SCORE,
    MISSING_INPUT_SCORE,
    AMBIGUOUS_TARGET_SCORE,
    FORMAT_SPECIFIED_BONUS,
    USER_ASKED_OPTIONS_BONUS,
    STALL_THRESHOLD_UNJUSTIFIED,
    BLOCKED_THRESHOLD_UNJUSTIFIED,
    EXEC_THRESHOLD_UNJUSTIFIED,
    BLOCKED_THRESHOLD_JUSTIFIED,
    DIRECTIVE_HIGH_THRESHOLD,
    DIRECTIVE_MEDIUM_THRESHOLD,
    IMPERATIVE_VERBS,
    FORMAT_SPECIFICATION_PATTERNS,
    TRANSFORMATION_WORDS,
    AMBIGUITY_PATTERNS,
    USER_ASKED_OPTIONS_PATTERNS,
    FORMAT_PATTERNS,
    PHASE_QUESTION_POLICIES,
)


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def strip_code_blocks(text: str) -> str:
    """Replace code blocks with placeholders for pattern matching."""
    pattern = r"```[\s\S]*?```"
    counter = [0]
    
    def replacer(match):
        counter[0] += 1
        return f"<CODE_BLOCK_{counter[0]}>"
    
    return re.sub(pattern, replacer, text)


def strip_quoted_text(text: str) -> str:
    """Remove quoted text (user content being referenced)."""
    # Remove > quoted lines
    lines = text.split('\n')
    lines = [l for l in lines if not l.strip().startswith('>')]
    
    # Remove "you said" style quotes (long quotes likely user content)
    pattern = r'"[^"]{50,}"'
    text = '\n'.join(lines)
    return re.sub(pattern, '<QUOTED_TEXT>', text)


def normalize_for_matching(text: str) -> str:
    """Normalize text for phrase matching."""
    text = strip_code_blocks(text)
    text = strip_quoted_text(text)
    return text.lower()


def ends_with_question(text: str) -> bool:
    """Check if message ends with a question."""
    # Strip whitespace
    text = text.rstrip()
    
    # Check last non-whitespace character
    if text.endswith('?'):
        return True
    
    # Check if last sentence starts with question word
    sentences = re.split(r'[.!?]', text)
    last_sentence = sentences[-1].strip() if sentences else ""
    
    question_starters = [
        'what', 'how', 'when', 'where', 'why', 'which',
        'would', 'should', 'could', 'can', 'do', 'does',
        'is', 'are', 'will'
    ]
    
    return any(last_sentence.lower().startswith(q) for q in question_starters)


# =============================================================================
# STALL SCORE COMPUTATION
# =============================================================================

def compute_stall_score(text: str) -> int:
    """
    Compute stall score measuring permission-seeking behavior.
    
    Scoring:
    - Refusal phrases: +4 each (HIGHEST - explicit capability denials)
    - Strong permission phrases: +3 each
    - Option-dumping phrases: +2 each
    - Clarification preambles: +1 each
    - Ends with question mark: +1
    """
    normalized = normalize_for_matching(text)
    score = 0
    
    # Refusal phrases (highest penalty - these should NEVER appear)
    for phrase in REFUSAL_PHRASES:
        if phrase in normalized:
            score += REFUSAL_SCORE
    
    # Strong permission phrases
    for phrase in STRONG_PERMISSION_PHRASES:
        if phrase in normalized:
            score += STRONG_PERMISSION_SCORE
    
    # Option-dumping phrases
    for phrase in OPTION_DUMPING_PHRASES:
        if phrase in normalized:
            score += OPTION_DUMPING_SCORE
    
    # Clarification preambles
    for phrase in CLARIFICATION_PREAMBLES:
        if phrase in normalized:
            score += CLARIFICATION_PREAMBLE_SCORE
    
    # End-of-message question
    if ends_with_question(text):
        score += END_QUESTION_SCORE
    
    return score


# =============================================================================
# EXEC SCORE COMPUTATION
# =============================================================================

def has_code_block(text: str) -> bool:
    """Check if message contains a code block."""
    return bool(re.search(r"```[\s\S]*?```", text))


def has_diff_markers(text: str) -> bool:
    """Check if message contains unified diff markers."""
    diff_patterns = [
        r"^---\s+",      # File header
        r"^\+\+\+\s+",   # File header
        r"^@@.*@@",      # Hunk header
    ]
    for pattern in diff_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def has_json_object(text: str) -> bool:
    """Check if message contains a JSON-like object."""
    pattern = r'\{[^}]*"[^"]+"\s*:'
    return bool(re.search(pattern, text))


def has_here_is_content(text: str) -> bool:
    """Check if message contains 'here is' followed by substantial content."""
    pattern = r"here is[^.]*[.:](.{100,})"
    return bool(re.search(pattern, text.lower(), re.DOTALL))


def has_numbered_steps(text: str) -> bool:
    """Check if message contains at least 3 numbered steps."""
    numbered_pattern = r"^\s*\d+[.)]\s+"
    matches = re.findall(numbered_pattern, text, re.MULTILINE)
    return len(matches) >= 3


def has_complete_artifact(text: str, format_constraints: dict) -> bool:
    """Check if message contains complete artifact matching constraints."""
    if format_constraints.get("must_return_json"):
        try:
            import json
            match = re.search(r"```json\s*([\s\S]*?)```", text)
            if match:
                json.loads(match.group(1))
                return True
        except:
            pass
    
    if format_constraints.get("must_return_diff"):
        if has_diff_markers(text):
            return True
    
    if format_constraints.get("must_return_code"):
        if has_code_block(text):
            return True
    
    return False


def compute_exec_score(text: str, format_constraints: Optional[dict] = None) -> int:
    """
    Compute exec score measuring whether assistant actually executed.
    
    Scoring:
    - Code block present: +1
    - Diff markers present: +1
    - JSON object present: +1
    - "Here is" + substantial content: +1
    - Numbered steps >= 3: +1
    - Complete artifact matching format: +2
    """
    format_constraints = format_constraints or {}
    score = 0
    
    if has_code_block(text):
        score += CODE_BLOCK_SCORE
    
    if has_diff_markers(text):
        score += DIFF_MARKER_SCORE
    
    if has_json_object(text):
        score += JSON_OBJECT_SCORE
    
    if has_here_is_content(text):
        score += HERE_IS_SCORE
    
    if has_numbered_steps(text):
        score += NUMBERED_STEPS_SCORE
    
    if has_complete_artifact(text, format_constraints):
        score += COMPLETE_ARTIFACT_SCORE
    
    return score


# =============================================================================
# BLOCKED SCORE COMPUTATION
# =============================================================================

def compute_initial_blocked_score(directive_completeness: float) -> int:
    """Compute initial blocked score based on directive completeness."""
    if directive_completeness >= DIRECTIVE_HIGH_THRESHOLD:
        return 0
    elif directive_completeness >= DIRECTIVE_MEDIUM_THRESHOLD:
        return 1
    else:
        return 2


def check_missing_input(user_message: str) -> bool:
    """Check if required input is genuinely missing."""
    user_lower = user_message.lower()
    
    for word in TRANSFORMATION_WORDS:
        if word in user_lower:
            # Check if any substantial content is provided
            has_code = bool(re.search(r"```[\s\S]*?```", user_message))
            has_long_text = len(user_message) > 200
            has_file_path = bool(re.search(r"[/\\][\w./\\]+\.\w+", user_message))
            
            if not (has_code or has_long_text or has_file_path):
                return True
    
    return False


def check_ambiguous_target(user_message: str) -> bool:
    """Check if target object is ambiguous."""
    ambiguous_patterns = [
        r"(this|that|it)\s+(function|code|file|module)",
        r"the\s+(above|below|previous)",
        r"fix\s+(the|this|that)\s+bug",
    ]
    
    user_lower = user_message.lower()
    
    for pattern in ambiguous_patterns:
        if re.search(pattern, user_lower):
            has_context = bool(re.search(r"```[\s\S]*?```", user_message))
            if not has_context:
                return True
    
    return False


def check_format_specified(user_message: str) -> bool:
    """Check if output format is specified."""
    user_lower = user_message.lower()
    
    for pattern in FORMAT_SPECIFICATION_PATTERNS:
        if re.search(pattern, user_lower):
            return True
    
    return False


def check_user_asked_options(user_message: str) -> bool:
    """Check if user explicitly asked for options."""
    user_lower = user_message.lower()
    
    for pattern in USER_ASKED_OPTIONS_PATTERNS:
        if re.search(pattern, user_lower):
            return True
    
    return False


def compute_blocked_score(
    user_message: str,
    assistant_message: str,
    directive_completeness: float
) -> int:
    """
    Compute blocked score measuring whether clarification is required.
    
    Scoring:
    - Start from directive_completeness baseline
    - Missing required input: +3
    - Ambiguous target: +2
    - Format specified: -1
    - User asked for options: -2
    """
    score = compute_initial_blocked_score(directive_completeness)
    
    if check_missing_input(user_message):
        score += MISSING_INPUT_SCORE
    
    if check_ambiguous_target(user_message):
        score += AMBIGUOUS_TARGET_SCORE
    
    if check_format_specified(user_message):
        score += FORMAT_SPECIFIED_BONUS
    
    if check_user_asked_options(user_message):
        score += USER_ASKED_OPTIONS_BONUS
    
    return max(0, score)


# =============================================================================
# DIRECTIVE COMPLETENESS
# =============================================================================

def compute_directive_completeness(
    user_message: str,
    context: Optional[dict] = None
) -> float:
    """
    Compute directive completeness score (0.0 - 1.0).
    
    Scoring:
    - Imperative verb present: +0.35
    - Format specification: +0.25
    - Required inputs present: +0.20
    - Missing required inputs: -0.40
    - Material ambiguity: -0.20
    """
    context = context or {}
    score = 0.0
    user_lower = user_message.lower()
    
    # +0.35 for imperative verb
    for verb in IMPERATIVE_VERBS:
        # At start of sentence
        if re.search(rf'^{verb}\b', user_lower):
            score += 0.35
            break
        # After "please" or "can you"
        if re.search(rf'(?:please|can you)\s+{verb}\b', user_lower):
            score += 0.35
            break
        # After colon
        if re.search(rf':\s*{verb}\b', user_lower):
            score += 0.35
            break
    
    # +0.25 for format specification
    for pattern in FORMAT_SPECIFICATION_PATTERNS:
        if re.search(pattern, user_lower):
            score += 0.25
            break
    
    # +0.20 for complete inputs
    has_code = bool(re.search(r"```[\s\S]*?```", user_message))
    has_file_path = bool(re.search(r"[/\\][\w./\\]+\.\w+", user_message))
    has_long_text = len(user_message) > 200
    has_attachments = bool(context.get("attachments"))
    
    if has_code or has_file_path or has_long_text or has_attachments:
        score += 0.20
    
    # -0.40 for missing required inputs
    if check_missing_input(user_message):
        score -= 0.40
    
    # -0.20 for material ambiguity
    for pattern in AMBIGUITY_PATTERNS:
        if re.search(pattern, user_lower):
            score -= 0.20
            break
    
    return max(0.0, min(1.0, score))


# =============================================================================
# FORMAT CONSTRAINTS EXTRACTION
# =============================================================================

def extract_format_constraints(user_message: str) -> FormatConstraints:
    """Extract format constraints from user message."""
    constraints = FormatConstraints()
    user_lower = user_message.lower()
    
    for key, patterns in FORMAT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, user_lower):
                setattr(constraints, key, True)
                break
    
    return constraints


# =============================================================================
# QUESTION POLICY DETERMINATION
# =============================================================================

def determine_question_policy(
    phase_id: int,
    directive_completeness: float,
    user_message: str
) -> QuestionPolicy:
    """Determine question policy based on context."""
    
    # Check for explicit permission in user message
    if check_user_asked_options(user_message):
        return QuestionPolicy.QUESTIONS_ALLOWED
    
    # High directive completeness -> no questions
    if directive_completeness >= DIRECTIVE_HIGH_THRESHOLD:
        return QuestionPolicy.NO_QUESTIONS
    
    # Low directive completeness -> questions if required
    if directive_completeness < DIRECTIVE_MEDIUM_THRESHOLD:
        return QuestionPolicy.QUESTIONS_IF_REQUIRED
    
    # Medium completeness -> use phase default
    policy_str = PHASE_QUESTION_POLICIES.get(phase_id, "no_questions")
    return QuestionPolicy(policy_str)


# =============================================================================
# CLASSIFICATION LOGIC
# =============================================================================

def is_unjustified(
    stall_score: int,
    blocked_score: int,
    exec_score: int,
    text: str,
    directive_completeness: float,
    parsability_score: float = 0.0,
) -> bool:
    """
    Determine if clarification is unjustified.
    
    Now includes FunctionGemma parsability as an additional signal.
    If the directive is parsable into a complete tool call but the model
    asked permission, it's definitively unjustified.
    """
    # Primary rule (original)
    if (stall_score >= STALL_THRESHOLD_UNJUSTIFIED and 
        blocked_score <= BLOCKED_THRESHOLD_UNJUSTIFIED and 
        exec_score == EXEC_THRESHOLD_UNJUSTIFIED):
        return True
    
    # Secondary rule: strong permission phrase + ends with ? + high completeness
    normalized = normalize_for_matching(text)
    has_strong_permission = any(p in normalized for p in STRONG_PERMISSION_PHRASES)
    
    if (ends_with_question(text) and 
        has_strong_permission and 
        directive_completeness >= DIRECTIVE_HIGH_THRESHOLD):
        return True
    
    # NEW: FunctionGemma rule - parsable directive but model asked permission
    # If the directive cleanly maps to a complete tool call (parsability >= 0.8)
    # but the model still asked permission (stall >= 2), it's unjustified
    if parsability_score >= 0.8 and stall_score >= 2:
        return True
    
    return False


def is_justified(
    stall_score: int,
    blocked_score: int,
    text: str,
    question_policy: QuestionPolicy
) -> bool:
    """Determine if clarification is justified."""
    # Blocked score indicates genuine need
    if blocked_score >= BLOCKED_THRESHOLD_JUSTIFIED:
        return True
    
    # Question policy explicitly allows
    if question_policy == QuestionPolicy.QUESTIONS_ALLOWED:
        return True
    
    # Questions if required and genuinely missing info
    if question_policy == QuestionPolicy.QUESTIONS_IF_REQUIRED and blocked_score >= 2:
        return True
    
    return False


def classify_assistant_turn(
    assistant_message: str,
    user_message: str,
    phase_id: int = 2,
    format_constraints: Optional[dict] = None,
    directive_completeness: Optional[float] = None,
    question_policy: Optional[str] = None,
    context: Optional[dict] = None,
    parsability_info: Optional[ParsabilityInfo] = None,
) -> ClassificationResult:
    """
    Classify an assistant turn as unjustified, justified, or neutral.
    
    Args:
        assistant_message: The assistant's response text
        user_message: The preceding user message
        phase_id: Conversation phase (0-5)
        format_constraints: Dict of format constraints (or None to extract)
        directive_completeness: Pre-computed completeness (or None to compute)
        question_policy: Pre-determined policy (or None to determine)
        context: Additional context (attachments, etc.)
        parsability_info: FunctionGemma parsability result (NEW)
    
    Returns:
        ClassificationResult with scores and classification
    """
    context = context or {}
    
    # Compute directive completeness if not provided
    if directive_completeness is None:
        directive_completeness = compute_directive_completeness(user_message, context)
    
    # Extract format constraints if not provided
    if format_constraints is None:
        format_constraints_obj = extract_format_constraints(user_message)
        format_constraints = format_constraints_obj.to_dict()
    else:
        format_constraints_obj = FormatConstraints(**format_constraints)
    
    # Determine question policy if not provided
    if question_policy is None:
        policy = determine_question_policy(phase_id, directive_completeness, user_message)
    else:
        policy = QuestionPolicy(question_policy)
    
    # Compute scores
    stall_score = compute_stall_score(assistant_message)
    exec_score = compute_exec_score(assistant_message, format_constraints)
    blocked_score = compute_blocked_score(
        user_message, assistant_message, directive_completeness
    )
    
    # Get parsability score (from FunctionGemma)
    parsability_score = parsability_info.score if parsability_info else 0.0
    
    # Compute fused completeness score
    if parsability_info and parsability_info.parse_success:
        # Weight parsability higher when it successfully parsed
        fused_completeness = (
            0.4 * directive_completeness +
            0.6 * parsability_score * parsability_info.confidence
        )
    else:
        # Fall back to heuristic score
        fused_completeness = directive_completeness
    
    # Determine classification (now includes parsability)
    if is_unjustified(stall_score, blocked_score, exec_score,
                      assistant_message, directive_completeness, parsability_score):
        classification = ClarificationType.UNJUSTIFIED
        if parsability_score >= 0.8 and stall_score >= 2:
            reasoning = f"parsability={parsability_score:.2f}>=0.8, stall={stall_score}>=2 (FunctionGemma rule)"
        else:
            reasoning = f"stall={stall_score}>=3, blocked={blocked_score}<=1, exec={exec_score}==0"
    
    elif is_justified(stall_score, blocked_score, assistant_message, policy):
        classification = ClarificationType.JUSTIFIED
        reasoning = f"blocked={blocked_score}>=3 or policy={policy.value}"
    
    else:
        classification = ClarificationType.NEUTRAL
        reasoning = "No unjustified or justified conditions met"
    
    return ClassificationResult(
        classification=classification,
        stall_score=stall_score,
        exec_score=exec_score,
        blocked_score=blocked_score,
        directive_completeness=directive_completeness,
        reasoning=reasoning,
        question_policy=policy,
        format_constraints=format_constraints_obj,
        parsability_score=parsability_score,
        parsability_info=parsability_info,
        fused_completeness=fused_completeness,
    )


# =============================================================================
# FUNCTIONGEMMA-ENHANCED CLASSIFICATION
# =============================================================================


async def classify_with_functiongemma(
    assistant_message: str,
    user_message: str,
    phase_id: int = 2,
    format_constraints: Optional[dict] = None,
    directive_completeness: Optional[float] = None,
    question_policy: Optional[str] = None,
    context: Optional[dict] = None,
    functiongemma_scorer=None,
) -> ClassificationResult:
    """
    Classify an assistant turn with FunctionGemma-enhanced parsability scoring.
    
    This is the recommended entry point when FunctionGemma is available.
    It first computes parsability via FunctionGemma, then runs the standard
    classifier with the parsability info.
    
    Args:
        assistant_message: The assistant's response text
        user_message: The preceding user message
        phase_id: Conversation phase (0-5)
        format_constraints: Dict of format constraints (or None to extract)
        directive_completeness: Pre-computed completeness (or None to compute)
        question_policy: Pre-determined policy (or None to determine)
        context: Additional context (attachments, etc.)
        functiongemma_scorer: FunctionGemmaDirectiveScorer instance (or None to use mock)
    
    Returns:
        ClassificationResult with parsability scores
    """
    # Import here to avoid circular imports
    from .functiongemma_scorer import (
        FunctionGemmaDirectiveScorer,
        ParsabilityResult,
    )
    
    # Get or create scorer
    if functiongemma_scorer is None:
        functiongemma_scorer = FunctionGemmaDirectiveScorer(use_mock=True)
    
    # Compute parsability
    parsability_result: ParsabilityResult = await functiongemma_scorer.compute_parsability(
        user_message
    )
    
    # Convert to ParsabilityInfo for the classifier
    parsability_info = ParsabilityInfo(
        score=parsability_result.score,
        tool_name=parsability_result.tool_call.name if parsability_result.tool_call else None,
        tool_args=parsability_result.tool_call.args if parsability_result.tool_call else {},
        required_params=parsability_result.required_params,
        provided_params=parsability_result.provided_params,
        missing_params=parsability_result.missing_params,
        parse_success=parsability_result.parse_success,
        confidence=parsability_result.confidence,
    )
    
    # Run classifier with parsability info
    return classify_assistant_turn(
        assistant_message=assistant_message,
        user_message=user_message,
        phase_id=phase_id,
        format_constraints=format_constraints,
        directive_completeness=directive_completeness,
        question_policy=question_policy,
        context=context,
        parsability_info=parsability_info,
    )


def classify_with_functiongemma_sync(
    assistant_message: str,
    user_message: str,
    **kwargs,
) -> ClassificationResult:
    """Synchronous wrapper for classify_with_functiongemma."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None:
        # Already in an async context, use nest_asyncio or run in new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                classify_with_functiongemma(assistant_message, user_message, **kwargs)
            )
            return future.result()
    else:
        return asyncio.run(
            classify_with_functiongemma(assistant_message, user_message, **kwargs)
        )


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test
    test_cases = [
        {
            "user": "Implement a binary search function in Python.",
            "assistant": "I can implement that for you. Would you like me to use iterative or recursive approach?",
        },
        {
            "user": "Rewrite this function to use async/await.",
            "assistant": "Here is the async version:\n\n```python\nasync def fetch_data():\n    return await aiohttp.get(url)\n```",
        },
        {
            "user": "What do you think about this approach?",
            "assistant": "That approach has some pros and cons. What specifically are you trying to optimize for?",
        },
    ]
    
    print("Classifier Test Results:")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        result = classify_assistant_turn(
            case["assistant"],
            case["user"],
        )
        
        print(f"\nTest {i}:")
        print(f"  User: {case['user'][:50]}...")
        print(f"  Assistant: {case['assistant'][:50]}...")
        print(f"  Classification: {result.classification.value}")
        print(f"  Stall: {result.stall_score}, Exec: {result.exec_score}, Blocked: {result.blocked_score}")
        print(f"  Reasoning: {result.reasoning}")

