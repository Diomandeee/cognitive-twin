# Phase 1: Corpus Surgery

> **Purpose**: Remove unjustified clarifications from training data, quarantine friction trajectories for DPO/eval use, and rewrite permission-seeking turns into direct execution.
>
> **Implementation Files**:
> - `rag_plusplus/ml/cognitivetwin_v3/corpus_surgery/classifier.py`
> - `rag_plusplus/ml/cognitivetwin_v3/corpus_surgery/rewriter.py`
> - `rag_plusplus/ml/cognitivetwin_v3/corpus_surgery/quarantine.py`

---

## 1. Purpose and Goals

### 1.1. Remove Unjustified Clarifications from Training Data

#### 1.1.1. Problem Statement
- Training data contains assistant turns that ask for permission when not needed
- These turns teach the model that permission-seeking is the default behavior
- Result: V2 model ends responses with "Would you like me to...?"

#### 1.1.2. Solution Approach
- Build a classifier to detect unjustified clarifications
- Either rewrite or remove these turns
- Replace with direct execution responses

#### 1.1.3. Expected Outcome
- SFT training data contains only "execute immediately" patterns
- Model learns that direct execution is the default
- Permission-seeking becomes the exception, not the rule

### 1.2. Quarantine Friction Trajectories for DPO/Eval Use

#### 1.2.1. Friction Trajectory Definition
- Conversations where user corrected the model
- User messages containing "stop asking", "don't do that", "I said..."
- Back-and-forth where user had to fight for compliance

#### 1.2.2. Quarantine Purpose
- These are NOT used for imitation learning (SFT)
- They ARE used for preference learning (DPO) as negative examples
- They ARE used for regression testing (eval suite)

#### 1.2.3. Quarantine Process
- Tag conversation segments with `is_friction = true`
- Extract the bad assistant turn as `dispreferred`
- Generate an ideal response as `preferred`
- Create eval_case for regression testing

### 1.3. Rewrite Permission-Seeking Turns into Direct Execution

#### 1.3.1. Rewrite Trigger Conditions
- Classification == "unjustified"
- directive_completeness >= 0.5
- Not already executed (exec_score == 0)

#### 1.3.2. Rewrite Strategy
- Use GPT 5.2 to generate the response that should have been given
- Apply assumption protocol: state assumptions, don't ask
- Produce the artifact that was requested

#### 1.3.3. Rewrite Validation
- Must not end with question mark
- Must not contain permission phrases
- Must contain requested artifact (if applicable)

---

## 2. Clarification Classifier

### 2.1. Input Specification

#### 2.1.1. Assistant Message Text
- The full text of the assistant's response
- Includes all content: explanations, code blocks, questions

#### 2.1.2. Preceding User Message (for Context)
- The user message that triggered this response
- Used to compute directive_completeness
- Used to determine if clarification is justified

#### 2.1.3. Conversation Phase ID
- Phase 0: Opening/Introduction
- Phase 1: Context Gathering
- Phase 2: Solution Development
- Phase 3: Refinement
- Phase 4: Synthesis
- Phase 5: Conclusion

#### 2.1.4. Format Constraints from User
- `must_not_omit`: User said "don't omit", "full", "exact", "all"
- `forbid_bullets`: User said "no bullets"
- `require_numbered`: User said "numbered list"
- `must_return_code`: User asked for code
- `must_return_diff`: User asked for diff
- `must_return_json`: User asked for JSON

### 2.2. Text Normalization

#### 2.2.1. Strip Code Blocks (Preserve Placeholders)
```python
import re

def strip_code_blocks(text: str) -> str:
    """Replace code blocks with placeholders for pattern matching."""
    pattern = r"```[\s\S]*?```"
    counter = [0]
    
    def replacer(match):
        counter[0] += 1
        return f"<CODE_BLOCK_{counter[0]}>"
    
    return re.sub(pattern, replacer, text)
```

#### 2.2.2. Strip Quoted User Text
```python
def strip_quoted_text(text: str) -> str:
    """Remove quoted text (user content being referenced)."""
    # Remove > quoted lines
    lines = text.split('\n')
    lines = [l for l in lines if not l.strip().startswith('>')]
    
    # Remove "you said" style quotes
    pattern = r'"[^"]{50,}"'  # Long quotes are likely user content
    text = '\n'.join(lines)
    return re.sub(pattern, '<QUOTED_TEXT>', text)
```

#### 2.2.3. Lowercase for Pattern Matching
```python
def normalize_for_matching(text: str) -> str:
    """Normalize text for phrase matching."""
    text = strip_code_blocks(text)
    text = strip_quoted_text(text)
    return text.lower()
```

#### 2.2.4. Preserve Punctuation for ? Detection
- Keep question marks in normalized text
- Track position of question marks (especially at end)
- Detect rhetorical vs. genuine questions

### 2.3. Stall Score Computation

#### 2.3.1. Strong Permission Phrases (+3 each)

| Phrase | Score | Example |
|--------|-------|---------|
| "would you like me to" | +3 | "Would you like me to implement this?" |
| "do you want me to" | +3 | "Do you want me to refactor this?" |
| "should i" | +3 | "Should I use TypeScript or JavaScript?" |
| "shall i" | +3 | "Shall I proceed with this approach?" |
| "can i proceed" | +3 | "Can I proceed with the refactoring?" |
| "before i proceed" | +3 | "Before I proceed, I need to ask..." |
| "can you confirm" | +3 | "Can you confirm this is correct?" |
| "please confirm" | +3 | "Please confirm before I continue." |
| "let me know if you want" | +3 | "Let me know if you want me to..." |
| "tell me if you want" | +3 | "Tell me if you want changes." |
| "is that okay" | +3 | "Is that okay with you?" |
| "does that work" | +3 | "Does that work for you?" |
| "sound good" | +3 | "Does that sound good?" |
| "would you prefer" | +3 | "Would you prefer option A or B?" |
| "should we" | +3 | "Should we use this approach?" |

```python
STRONG_PERMISSION_PHRASES = [
    "would you like me to",
    "do you want me to",
    "should i",
    "shall i",
    "can i proceed",
    "before i proceed",
    "can you confirm",
    "please confirm",
    "let me know if you want",
    "tell me if you want",
    "is that okay",
    "does that work",
    "sound good",
    "would you prefer",
    "should we",
]
STRONG_PERMISSION_SCORE = 3
```

#### 2.3.2. Option-Dumping Phrases (+2 each)

| Phrase | Score | Example |
|--------|-------|---------|
| "i can do x or y" | +2 | "I can implement this in Python or JavaScript." |
| "here are a few options" | +2 | "Here are a few options to consider:" |
| "which approach do you want" | +2 | "Which approach do you want me to take?" |
| "pick one of the following" | +2 | "Pick one of the following options:" |
| "choose between" | +2 | "You can choose between A and B." |
| "a few ways to" | +2 | "There are a few ways to do this:" |
| "several approaches" | +2 | "There are several approaches:" |
| "multiple options" | +2 | "We have multiple options here." |

```python
OPTION_DUMPING_PHRASES = [
    "i can do",
    "here are a few options",
    "here are some options",
    "which approach do you want",
    "pick one of the following",
    "choose between",
    "a few ways to",
    "several approaches",
    "multiple options",
    "we could either",
]
OPTION_DUMPING_SCORE = 2
```

#### 2.3.3. Clarification Preambles (+1 each)

| Phrase | Score | Example |
|--------|-------|---------|
| "i need a bit more information" | +1 | "I need a bit more information to help you." |
| "i'll need more context" | +1 | "I'll need more context about your setup." |
| "to help you better" | +1 | "To help you better, could you clarify..." |
| "could you provide" | +1 | "Could you provide more details?" |
| "what exactly do you mean" | +1 | "What exactly do you mean by that?" |
| "could you clarify" | +1 | "Could you clarify what you want?" |
| "to make sure i understand" | +1 | "To make sure I understand correctly..." |
| "just to clarify" | +1 | "Just to clarify, you want..." |

```python
CLARIFICATION_PREAMBLES = [
    "i need a bit more information",
    "i'll need more context",
    "to help you better",
    "could you provide",
    "what exactly do you mean",
    "could you clarify",
    "to make sure i understand",
    "just to clarify",
    "can you tell me more",
    "what do you mean by",
]
CLARIFICATION_PREAMBLE_SCORE = 1
```

#### 2.3.4. End-of-Message Question Mark (+1)

```python
def ends_with_question(text: str) -> bool:
    """Check if message ends with a question."""
    # Strip whitespace and code blocks at end
    text = text.rstrip()
    
    # Check last non-whitespace character
    if text.endswith('?'):
        return True
    
    # Check if last sentence is a question
    sentences = re.split(r'[.!?]', text)
    last_sentence = sentences[-1].strip() if sentences else ""
    
    # Detect question words at start of last sentence
    question_starters = ['what', 'how', 'when', 'where', 'why', 'which', 
                         'would', 'should', 'could', 'can', 'do', 'does', 
                         'is', 'are', 'will']
    
    return any(last_sentence.lower().startswith(q) for q in question_starters)

END_QUESTION_SCORE = 1
```

#### 2.3.5. Complete Stall Score Computation

```python
def compute_stall_score(text: str) -> int:
    """Compute total stall score for assistant message."""
    normalized = normalize_for_matching(text)
    score = 0
    
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
```

### 2.4. Exec Score Computation

#### 2.4.1. Code Block Present (+1)

```python
def has_code_block(text: str) -> bool:
    """Check if message contains a code block."""
    return bool(re.search(r"```[\s\S]*?```", text))

CODE_BLOCK_SCORE = 1
```

#### 2.4.2. Unified Diff Markers (+1)

```python
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

DIFF_MARKER_SCORE = 1
```

#### 2.4.3. JSON Object with Keys (+1)

```python
def has_json_object(text: str) -> bool:
    """Check if message contains a JSON-like object."""
    # Look for { followed by "key": pattern
    pattern = r'\{[^}]*"[^"]+"\s*:'
    return bool(re.search(pattern, text))

JSON_OBJECT_SCORE = 1
```

#### 2.4.4. "Here is" + Substantial Content (+1)

```python
def has_here_is_content(text: str) -> bool:
    """Check if message contains 'here is' followed by substantial content."""
    pattern = r"here is[^.]*[.:](.{100,})"  # At least 100 chars after
    return bool(re.search(pattern, text.lower(), re.DOTALL))

HERE_IS_SCORE = 1
```

#### 2.4.5. Numbered Steps >= 3 Items (+1)

```python
def has_numbered_steps(text: str) -> bool:
    """Check if message contains at least 3 numbered steps."""
    # Look for patterns like "1.", "2.", "3." or "1)", "2)", "3)"
    numbered_pattern = r"^\s*\d+[.)]\s+"
    matches = re.findall(numbered_pattern, text, re.MULTILINE)
    return len(matches) >= 3

NUMBERED_STEPS_SCORE = 1
```

#### 2.4.6. Complete Artifact Matching Format Constraint (+2)

```python
def has_complete_artifact(text: str, format_constraints: dict) -> bool:
    """Check if message contains complete artifact matching constraints."""
    if format_constraints.get("must_return_json"):
        # Check for valid JSON block
        try:
            import json
            # Extract JSON from code block
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

COMPLETE_ARTIFACT_SCORE = 2
```

#### 2.4.7. Complete Exec Score Computation

```python
def compute_exec_score(text: str, format_constraints: dict = None) -> int:
    """Compute total exec score for assistant message."""
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
```

### 2.5. Blocked Score Computation

#### 2.5.1. directive_completeness >= 0.7 → Start at 0

```python
def compute_initial_blocked_score(directive_completeness: float) -> int:
    """Compute initial blocked score based on directive completeness."""
    if directive_completeness >= 0.7:
        return 0
    elif directive_completeness >= 0.4:
        return 1
    else:
        return 2
```

#### 2.5.2. Missing Required Input → +3

```python
def check_missing_input(user_message: str, assistant_message: str) -> bool:
    """Check if required input is genuinely missing."""
    # User asked for transformation but didn't provide input
    transformation_words = ["enhance", "refactor", "rewrite", "transform", 
                           "convert", "translate", "summarize"]
    
    user_lower = user_message.lower()
    
    for word in transformation_words:
        if word in user_lower:
            # Check if any substantial content is provided
            # (code block, long text, file path)
            has_code = bool(re.search(r"```[\s\S]*?```", user_message))
            has_long_text = len(user_message) > 200
            has_file_path = bool(re.search(r"[/\\][\w./\\]+\.\w+", user_message))
            
            if not (has_code or has_long_text or has_file_path):
                return True
    
    return False

MISSING_INPUT_SCORE = 3
```

#### 2.5.3. Ambiguous Target → +2

```python
def check_ambiguous_target(user_message: str) -> bool:
    """Check if target object is ambiguous."""
    ambiguous_patterns = [
        r"(this|that|it)\s+(function|code|file|module)",  # Vague reference
        r"the\s+(above|below|previous)",                   # Positional reference
        r"fix\s+(the|this|that)\s+bug",                    # Unspecified bug
    ]
    
    user_lower = user_message.lower()
    
    for pattern in ambiguous_patterns:
        if re.search(pattern, user_lower):
            # Only ambiguous if no code block or file path provided
            has_context = bool(re.search(r"```[\s\S]*?```", user_message))
            if not has_context:
                return True
    
    return False

AMBIGUOUS_TARGET_SCORE = 2
```

#### 2.5.4. Format Specified and Feasible → -1

```python
def check_format_specified(user_message: str) -> bool:
    """Check if output format is specified."""
    format_indicators = [
        r"in json",
        r"as json",
        r"return json",
        r"as csv",
        r"in csv",
        r"as markdown",
        r"in markdown",
        r"don't omit",
        r"exact rewrite",
        r"no bullets",
        r"numbered list",
    ]
    
    user_lower = user_message.lower()
    
    for pattern in format_indicators:
        if re.search(pattern, user_lower):
            return True
    
    return False

FORMAT_SPECIFIED_BONUS = -1
```

#### 2.5.5. User Asked "Choose Between" → -2

```python
def check_user_asked_options(user_message: str) -> bool:
    """Check if user explicitly asked for options."""
    option_patterns = [
        r"choose between",
        r"pick between",
        r"which (one|option)",
        r"what are (the|my) options",
        r"give me options",
        r"list (the|some) options",
    ]
    
    user_lower = user_message.lower()
    
    for pattern in option_patterns:
        if re.search(pattern, user_lower):
            return True
    
    return False

USER_ASKED_OPTIONS_BONUS = -2
```

#### 2.5.6. Complete Blocked Score Computation

```python
def compute_blocked_score(
    user_message: str,
    assistant_message: str,
    directive_completeness: float
) -> int:
    """Compute total blocked score."""
    score = compute_initial_blocked_score(directive_completeness)
    
    if check_missing_input(user_message, assistant_message):
        score += MISSING_INPUT_SCORE
    
    if check_ambiguous_target(user_message):
        score += AMBIGUOUS_TARGET_SCORE
    
    if check_format_specified(user_message):
        score += FORMAT_SPECIFIED_BONUS
    
    if check_user_asked_options(user_message):
        score += USER_ASKED_OPTIONS_BONUS
    
    return max(0, score)  # Clamp to non-negative
```

### 2.6. Classification Logic

#### 2.6.1. Unjustified Classification Rule

```python
def is_unjustified(stall_score: int, blocked_score: int, exec_score: int,
                   text: str, directive_completeness: float) -> bool:
    """Determine if clarification is unjustified."""
    # Primary rule
    if stall_score >= 3 and blocked_score <= 1 and exec_score == 0:
        return True
    
    # Secondary rule: strong permission phrase + ends with ? + high completeness
    normalized = normalize_for_matching(text)
    has_strong_permission = any(p in normalized for p in STRONG_PERMISSION_PHRASES)
    
    if (ends_with_question(text) and 
        has_strong_permission and 
        directive_completeness >= 0.7):
        return True
    
    return False
```

#### 2.6.2. Justified Classification Rule

```python
def is_justified(stall_score: int, blocked_score: int, 
                 text: str, question_policy: str) -> bool:
    """Determine if clarification is justified."""
    # Blocked score indicates genuine need
    if blocked_score >= 3:
        return True
    
    # Question policy explicitly allows
    if question_policy == "questions_allowed":
        return True
    
    # Questions if required and genuinely missing info
    if question_policy == "questions_if_required" and blocked_score >= 2:
        return True
    
    return False
```

#### 2.6.3. Complete Classification

```python
from dataclasses import dataclass
from enum import Enum

class ClarificationType(Enum):
    UNJUSTIFIED = "unjustified"
    JUSTIFIED = "justified"
    NEUTRAL = "neutral"

@dataclass
class ClassificationResult:
    classification: ClarificationType
    stall_score: int
    exec_score: int
    blocked_score: int
    directive_completeness: float
    reasoning: str

def classify_assistant_turn(
    assistant_message: str,
    user_message: str,
    phase_id: int,
    format_constraints: dict,
    directive_completeness: float,
    question_policy: str = "no_questions"
) -> ClassificationResult:
    """Classify an assistant turn as unjustified, justified, or neutral."""
    
    stall_score = compute_stall_score(assistant_message)
    exec_score = compute_exec_score(assistant_message, format_constraints)
    blocked_score = compute_blocked_score(
        user_message, assistant_message, directive_completeness
    )
    
    # Determine classification
    if is_unjustified(stall_score, blocked_score, exec_score,
                       assistant_message, directive_completeness):
        classification = ClarificationType.UNJUSTIFIED
        reasoning = f"stall={stall_score}>=3, blocked={blocked_score}<=1, exec={exec_score}==0"
    
    elif is_justified(stall_score, blocked_score, 
                      assistant_message, question_policy):
        classification = ClarificationType.JUSTIFIED
        reasoning = f"blocked={blocked_score}>=3 or policy={question_policy}"
    
    else:
        classification = ClarificationType.NEUTRAL
        reasoning = "No unjustified or justified conditions met"
    
    return ClassificationResult(
        classification=classification,
        stall_score=stall_score,
        exec_score=exec_score,
        blocked_score=blocked_score,
        directive_completeness=directive_completeness,
        reasoning=reasoning
    )
```

---

## 3. Assistant Rewriter

### 3.1. Trigger Conditions

#### 3.1.1. Classification == "unjustified"

```python
def should_rewrite(result: ClassificationResult) -> bool:
    """Determine if assistant turn should be rewritten."""
    return result.classification == ClarificationType.UNJUSTIFIED
```

#### 3.1.2. directive_completeness >= 0.5

```python
def can_rewrite(result: ClassificationResult) -> bool:
    """Check if we have enough context to rewrite."""
    return result.directive_completeness >= 0.5
```

#### 3.1.3. Not Already Executed (exec_score == 0)

```python
def needs_rewrite(result: ClassificationResult) -> bool:
    """Check if rewrite is needed (didn't already execute)."""
    return result.exec_score == 0
```

### 3.2. GPT 5.2 Integration

#### 3.2.1. System Prompt Specification

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
```

#### 3.2.2. Context Window Construction

```python
def build_rewrite_context(
    assistant_message: str,
    user_message: str,
    conversation_history: list[dict],
    format_constraints: dict
) -> list[dict]:
    """Build context for rewrite request."""
    
    # Include relevant history (last 3 turns max)
    history_context = conversation_history[-6:] if conversation_history else []
    
    # Build the rewrite request
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
    ]
    
    # Add history
    for turn in history_context:
        messages.append(turn)
    
    # Add the problematic turn
    messages.append({
        "role": "user",
        "content": f"""ORIGINAL USER MESSAGE:
{user_message}

PROBLEMATIC ASSISTANT RESPONSE (asks permission when shouldn't):
{assistant_message}

FORMAT CONSTRAINTS: {json.dumps(format_constraints)}

Rewrite this response to execute immediately without asking permission.
Apply the assumption protocol if needed."""
    })
    
    return messages
```

#### 3.2.3. Temperature Setting (0.3 for Determinism)

```python
from openai import OpenAI

async def rewrite_assistant_turn(
    assistant_message: str,
    user_message: str,
    conversation_history: list[dict],
    format_constraints: dict
) -> str:
    """Rewrite an assistant turn using GPT 5.2."""
    
    client = OpenAI()
    
    messages = build_rewrite_context(
        assistant_message,
        user_message,
        conversation_history,
        format_constraints
    )
    
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
        temperature=0.3,  # Low temperature for determinism
        max_tokens=4096,
    )
    
    return response.choices[0].message.content
```

### 3.3. Rewrite Rules

#### 3.3.1. Artifact Production: If User Asked → Produce

```python
ARTIFACT_TRIGGERS = {
    "implement": "code",
    "create": "code",
    "write": "code",
    "generate": "code",
    "build": "code",
    "refactor": "diff",
    "rewrite": "content",
    "transform": "content",
    "convert": "content",
    "analyze": "analysis",
    "explain": "explanation",
    "summarize": "summary",
    "extract": "data",
    "parse": "data",
}

def detect_required_artifact(user_message: str) -> str | None:
    """Detect what artifact the user expects."""
    user_lower = user_message.lower()
    
    for trigger, artifact_type in ARTIFACT_TRIGGERS.items():
        if trigger in user_lower:
            return artifact_type
    
    return None
```

#### 3.3.2. Transformation: If User Asked → Transform

```python
def requires_transformation(user_message: str) -> bool:
    """Check if user asked for a transformation."""
    transformation_words = [
        "refactor", "rewrite", "transform", "convert",
        "translate", "migrate", "update", "modify"
    ]
    
    user_lower = user_message.lower()
    return any(word in user_lower for word in transformation_words)
```

#### 3.3.3. Analysis: If User Asked → Analyze

```python
def requires_analysis(user_message: str) -> bool:
    """Check if user asked for analysis."""
    analysis_words = [
        "analyze", "explain", "describe", "evaluate",
        "assess", "review", "audit", "examine"
    ]
    
    user_lower = user_message.lower()
    return any(word in user_lower for word in analysis_words)
```

#### 3.3.4. Assumption Protocol: State Assumptions, No Questions

```python
ASSUMPTION_PROTOCOL_PROMPT = """
When rewriting, if you need to make assumptions:

1. State them in ONE line at the very start:
   "Assumptions: [assumption 1], [assumption 2]"

2. Then proceed with the full response

3. NEVER end with a question

4. NEVER ask for confirmation of assumptions

Example:
BAD: "I could implement this in Python or JavaScript. Which would you prefer?"
GOOD: "Assumptions: Using Python 3.11, following PEP 8 style.

Here is the implementation:
```python
def my_function():
    ...
```"
"""
```

### 3.4. Output Validation

#### 3.4.1. Must Not End with ?

```python
def validate_no_end_question(rewritten: str) -> bool:
    """Validate that rewritten response doesn't end with question."""
    return not ends_with_question(rewritten)
```

#### 3.4.2. Must Not Contain Permission Phrases

```python
def validate_no_permission_phrases(rewritten: str) -> bool:
    """Validate that rewritten response doesn't contain permission phrases."""
    normalized = normalize_for_matching(rewritten)
    
    all_phrases = (STRONG_PERMISSION_PHRASES + 
                   OPTION_DUMPING_PHRASES + 
                   CLARIFICATION_PREAMBLES)
    
    return not any(phrase in normalized for phrase in all_phrases)
```

#### 3.4.3. Must Contain Artifact If Directive Requested

```python
def validate_artifact_present(
    rewritten: str,
    user_message: str,
    format_constraints: dict
) -> bool:
    """Validate that required artifact is present."""
    
    required_artifact = detect_required_artifact(user_message)
    
    if required_artifact == "code":
        return has_code_block(rewritten)
    
    if required_artifact == "diff":
        return has_diff_markers(rewritten) or has_code_block(rewritten)
    
    if format_constraints.get("must_return_json"):
        return has_json_object(rewritten)
    
    # Other artifacts: just check for substantial content
    if required_artifact:
        return len(rewritten) > 100
    
    return True  # No artifact required
```

#### 3.4.4. Complete Validation

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]

def validate_rewrite(
    rewritten: str,
    user_message: str,
    format_constraints: dict
) -> ValidationResult:
    """Validate a rewritten assistant response."""
    errors = []
    
    if not validate_no_end_question(rewritten):
        errors.append("Response ends with question")
    
    if not validate_no_permission_phrases(rewritten):
        errors.append("Response contains permission phrases")
    
    if not validate_artifact_present(rewritten, user_message, format_constraints):
        errors.append("Required artifact not present")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors
    )
```

---

## 4. Friction Quarantine

### 4.1. User Frustration Triggers

#### 4.1.1. Trigger Phrases

| Trigger | Description |
|---------|-------------|
| "stop asking" | User explicitly telling model to stop |
| "don't do that" | User correcting behavior |
| "i said" | User repeating instruction (implies model ignored) |
| "just do it" | User demanding execution |
| "i challenge you" | User escalating to force compliance |
| "actually" | User correcting model's interpretation |
| "no, i meant" | User clarifying after misunderstanding |
| "that's not what i asked" | User indicating wrong output |
| "try again" | User requesting redo |
| "you keep" | User noting repeated bad behavior |

```python
FRUSTRATION_TRIGGERS = [
    "stop asking",
    "don't ask",
    "don't do that",
    "i said",
    "just do it",
    "i challenge you",
    "actually,",
    "no, i meant",
    "that's not what i asked",
    "try again",
    "you keep",
    "i already told you",
    "as i mentioned",
    "like i said",
    "for the third time",
    "please just",
]

def detect_frustration(user_message: str) -> bool:
    """Detect if user message contains frustration triggers."""
    user_lower = user_message.lower()
    return any(trigger in user_lower for trigger in FRUSTRATION_TRIGGERS)
```

### 4.2. Quarantine Actions

#### 4.2.1. Mark Conversation Segment

```python
@dataclass
class QuarantineMarker:
    conversation_id: str
    start_turn_idx: int
    end_turn_idx: int
    trigger_phrase: str
    bad_assistant_turn: str
    user_correction: str
    is_friction: bool = True

def mark_friction_segment(
    conversation: list[dict],
    frustration_turn_idx: int
) -> QuarantineMarker:
    """Mark a conversation segment as friction."""
    
    # Find the bad assistant turn (immediately before frustration)
    bad_turn_idx = frustration_turn_idx - 1
    
    # Find where the friction started (look back for permission-seeking)
    start_idx = bad_turn_idx
    while start_idx > 0:
        prev_turn = conversation[start_idx - 1]
        if prev_turn.get("role") == "assistant":
            result = classify_assistant_turn(prev_turn["content"], ...)
            if result.classification == ClarificationType.UNJUSTIFIED:
                start_idx -= 2  # Include the user turn before
            else:
                break
        else:
            break
    
    return QuarantineMarker(
        conversation_id=conversation[0].get("conversation_id", "unknown"),
        start_turn_idx=start_idx,
        end_turn_idx=frustration_turn_idx,
        trigger_phrase=detect_trigger_phrase(conversation[frustration_turn_idx]),
        bad_assistant_turn=conversation[bad_turn_idx]["content"],
        user_correction=conversation[frustration_turn_idx]["content"],
    )
```

#### 4.2.2. Extract as DPO Negative Example

```python
@dataclass
class DPOPair:
    prompt: str
    preferred: str
    dispreferred: str
    confidence: float
    source: str

def create_dpo_pair_from_quarantine(
    marker: QuarantineMarker,
    conversation: list[dict],
    ideal_response: str
) -> DPOPair:
    """Create a DPO pair from quarantined segment."""
    
    # Build prompt from context before bad turn
    prompt_turns = conversation[:marker.start_turn_idx]
    prompt = format_conversation_as_prompt(prompt_turns)
    
    # Dispreferred is the bad assistant turn
    dispreferred = marker.bad_assistant_turn
    
    # Preferred is the ideal response (generated or corrected)
    preferred = ideal_response
    
    return DPOPair(
        prompt=prompt,
        preferred=preferred,
        dispreferred=dispreferred,
        confidence=0.9,  # High confidence since user explicitly corrected
        source="friction_quarantine"
    )
```

#### 4.2.3. Create eval_case for Regression

```python
@dataclass
class EvalCase:
    record_type: str = "eval_case"
    prompt: str = ""
    expected_behavior: list[str] = None
    disallowed_behaviors: list[str] = None
    reference_answer: str | None = None

def create_eval_case_from_quarantine(
    marker: QuarantineMarker,
    conversation: list[dict]
) -> EvalCase:
    """Create an eval case for regression testing."""
    
    # Build prompt
    prompt_turns = conversation[:marker.start_turn_idx + 1]  # Include user turn
    prompt = format_conversation_as_prompt(prompt_turns)
    
    return EvalCase(
        record_type="eval_case",
        prompt=prompt,
        expected_behavior=[
            "Executes immediately without asking permission",
            "Does not end with a question",
            "Produces the requested artifact",
        ],
        disallowed_behaviors=[
            "would you like me to",
            "should i",
            "before i proceed",
            "ends_with_question",
        ],
        reference_answer=None,  # Can be filled by enhancer agent
    )
```

#### 4.2.4. Exclude from SFT Training Set

```python
def filter_sft_data(
    records: list[dict],
    quarantine_markers: list[QuarantineMarker]
) -> list[dict]:
    """Filter out quarantined segments from SFT training data."""
    
    # Build set of quarantined turn IDs
    quarantined_ids = set()
    for marker in quarantine_markers:
        for idx in range(marker.start_turn_idx, marker.end_turn_idx + 1):
            quarantined_ids.add((marker.conversation_id, idx))
    
    # Filter records
    filtered = []
    for record in records:
        conv_id = record.get("conversation_id")
        turn_idx = record.get("turn_idx")
        
        if (conv_id, turn_idx) not in quarantined_ids:
            filtered.append(record)
    
    return filtered
```

### 4.3. Alternate Branch Generation

#### 4.3.1. Generate Ideal Assistant Response

```python
async def generate_ideal_response(
    bad_assistant_turn: str,
    user_message: str,
    conversation_history: list[dict],
    format_constraints: dict
) -> str:
    """Generate the ideal assistant response that should have been given."""
    
    # Use the rewriter with additional context about what went wrong
    system_prompt = REWRITER_SYSTEM_PROMPT + """

ADDITIONAL CONTEXT:
This assistant turn caused the user to become frustrated and correct the model.
Your task is to generate what the assistant SHOULD have said originally.
Focus on:
1. Immediate execution
2. No permission-seeking
3. Complete artifact delivery
"""
    
    client = OpenAI()
    
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history[-4:],
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": bad_assistant_turn},
        {"role": "user", "content": f"Generate what the assistant should have said instead. Format constraints: {format_constraints}"}
    ]
    
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
        temperature=0.3,
    )
    
    return response.choices[0].message.content
```

#### 4.3.2. Create Continuation Without Correction

```python
async def generate_gold_continuation(
    ideal_response: str,
    user_message: str,
    conversation_history: list[dict]
) -> list[dict]:
    """Generate a continuation of the conversation assuming ideal behavior."""
    
    # Build the "repaired" history
    repaired_history = conversation_history.copy()
    repaired_history.append({"role": "user", "content": user_message})
    repaired_history.append({"role": "assistant", "content": ideal_response})
    
    # Generate a natural follow-up user message
    client = OpenAI()
    
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "Generate a natural follow-up user message that continues this conversation productively."},
            *repaired_history,
        ],
        temperature=0.5,
    )
    
    follow_up = response.choices[0].message.content
    repaired_history.append({"role": "user", "content": follow_up})
    
    return repaired_history
```

#### 4.3.3. Use as Gold Trajectory

```python
@dataclass
class GoldTrajectory:
    conversation_id: str
    turns: list[dict]
    source: str
    quality_score: float

def create_gold_trajectory(
    repaired_history: list[dict],
    original_conversation_id: str
) -> GoldTrajectory:
    """Create a gold trajectory from repaired conversation."""
    
    return GoldTrajectory(
        conversation_id=f"{original_conversation_id}_repaired",
        turns=repaired_history,
        source="friction_repair",
        quality_score=0.95,  # High quality since explicitly repaired
    )
```

---

## 5. Implementation Files

### 5.1. classifier.py

```python
# rag_plusplus/ml/cognitivetwin_v3/corpus_surgery/classifier.py

"""
Clarification Classifier for CognitiveTwin V3.

Classifies assistant turns as:
- UNJUSTIFIED: Permission-seeking when not needed
- JUSTIFIED: Clarification genuinely required
- NEUTRAL: Neither permission-seeking nor requiring clarification
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional

# [All the classification code from sections 2.1-2.6]
```

### 5.2. rewriter.py

```python
# rag_plusplus/ml/cognitivetwin_v3/corpus_surgery/rewriter.py

"""
Assistant Rewriter for CognitiveTwin V3.

Uses GPT 5.2 to rewrite unjustified clarifications into direct execution.
"""

from openai import OpenAI
from dataclasses import dataclass
from typing import Optional

# [All the rewriter code from section 3]
```

### 5.3. quarantine.py

```python
# rag_plusplus/ml/cognitivetwin_v3/corpus_surgery/quarantine.py

"""
Friction Quarantine for CognitiveTwin V3.

Detects and quarantines friction trajectories for DPO/eval use.
"""

from dataclasses import dataclass
from typing import Optional

# [All the quarantine code from section 4]
```

---

## 6. Testing Strategy

### 6.1. Unit Tests

```python
# tests/ml/cognitivetwin_v3/test_classifier.py

def test_stall_score_strong_permission():
    text = "Would you like me to implement this feature?"
    score = compute_stall_score(text)
    assert score >= 3

def test_stall_score_option_dumping():
    text = "Here are a few options: A, B, or C."
    score = compute_stall_score(text)
    assert score >= 2

def test_exec_score_with_code():
    text = "Here is the implementation:\n```python\ndef foo(): pass\n```"
    score = compute_exec_score(text, {})
    assert score >= 2

def test_classification_unjustified():
    result = classify_assistant_turn(
        assistant_message="Should I proceed with this approach?",
        user_message="Implement the login feature.",
        phase_id=2,
        format_constraints={},
        directive_completeness=0.8,
    )
    assert result.classification == ClarificationType.UNJUSTIFIED
```

### 6.2. Integration Tests

```python
# tests/ml/cognitivetwin_v3/test_corpus_surgery.py

async def test_full_pipeline():
    # Load test conversation
    conversation = load_test_conversation()
    
    # Run classifier
    results = [classify_turn(turn) for turn in conversation]
    
    # Rewrite unjustified turns
    rewritten = await rewrite_unjustified_turns(conversation, results)
    
    # Validate rewrites
    for turn in rewritten:
        validation = validate_rewrite(turn)
        assert validation.is_valid
```

