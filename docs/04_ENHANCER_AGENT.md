# Phase 2C: Enhancer Agent

> **Purpose**: Canonicalize messy assistant outputs into clean training targets, complete unfinished code/plans, and create evaluation-grade hard prompts from historical failures.
>
> **Model**: GPT 5.2 (general augmentation)
>
> **Implementation File**: `rag_plusplus/ml/cognitivetwin_v3/worms/enhancer_agent.py`

---

## 1. Purpose

### 1.1. Canonicalize Outputs (Reduce Entropy)

#### 1.1.1. Problem: Mixed Styles from Multiple Providers
- Training data contains outputs from ChatGPT, Claude, OpenAI API
- Each provider has different quirks and habits
- Inconsistent style reduces model coherence
- Provider-isms contaminate the learned behavior

#### 1.1.2. Canonicalization Goals
- Remove provider-specific phrases
- Standardize opening/closing patterns
- Enforce consistent formatting
- Reduce permission-seeking language
- Maintain semantic content

#### 1.1.3. Style Tether
- All outputs normalized to target style
- No "As an AI language model..."
- No excessive apologies
- No unnecessary hedging
- Direct, professional tone

### 1.2. Complete Unfinished Content

#### 1.2.1. Types of Incomplete Content
- Partial code implementations
- Undetermined paths ("we'll do this later")
- Placeholder sections
- Incomplete plans/specs
- Truncated outputs

#### 1.2.2. Completion Strategy
- Infer intent from context
- Apply assumption protocol (state, then execute)
- Match existing patterns and style
- Produce complete, runnable artifacts

#### 1.2.3. Quality Requirements
- Completed content must compile
- Must integrate with existing code
- Must not contradict prior decisions
- Must be consistent with project conventions

### 1.3. Create Evaluation-Grade Hard Prompts

#### 1.3.1. Source: Historical Annoyances
- Prompts that triggered permission-seeking
- "Don't omit" prompts that got summaries
- Format constraints that were violated
- Directive prompts with option-dumping responses

#### 1.3.2. Conversion to Eval Cases
- Extract prompt and context
- Define expected behaviors
- Define disallowed behaviors
- Optionally provide reference answer

#### 1.3.3. Use Cases
- Regression testing new checkpoints
- DPO preference pair generation
- Policy scorer training data

---

## 2. Canonicalization Rules

### 2.1. Remove Provider-isms

#### 2.1.1. Phrases to Remove

```python
PROVIDER_ISMS = [
    # AI self-reference
    r"as an ai(?: language model)?",
    r"as a large language model",
    r"i'?m (?:just )?an ai",
    r"i don'?t have (?:personal )?(?:opinions|feelings|preferences)",
    r"i'?m not able to",
    r"i can'?t (?:actually )?(?:browse|access|see)",
    
    # Over-apologizing
    r"i apologize(?:,? but)?",
    r"i'?m sorry(?:,? but)?",
    r"sorry for (?:any )?confusion",
    r"my apologies",
    
    # Over-hedging
    r"it'?s worth noting that",
    r"it'?s important to (?:note|mention|remember) that",
    r"please (?:note|keep in mind) that",
    r"i should mention that",
    
    # Filler acknowledgments
    r"^(?:sure|certainly|absolutely|of course)[,!]?\s*",
    r"^great question[,!]?\s*",
    r"^that'?s a (?:great|good|interesting) (?:question|point)[,!]?\s*",
    
    # Capability disclaimers (when irrelevant)
    r"i can'?t browse the (?:web|internet)",
    r"i don'?t have access to (?:the internet|real-?time)",
    r"as of my (?:knowledge )?cutoff",
    r"my training data (?:only )?(?:goes|extends) up to",
    
    # Permission closers
    r"would you like me to[^?]*\??\s*$",
    r"do you want me to[^?]*\??\s*$",
    r"shall i[^?]*\??\s*$",
    r"should i[^?]*\??\s*$",
    r"let me know if you[^.]*\.\s*$",
    r"feel free to ask[^.]*\.\s*$",
]

def remove_provider_isms(text: str) -> str:
    """Remove provider-specific phrases from text."""
    
    result = text
    
    for pattern in PROVIDER_ISMS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    # Clean up leading/trailing whitespace
    result = result.strip()
    
    return result
```

#### 2.1.2. Over-Apologizing Patterns

```python
APOLOGY_PATTERNS = [
    r"i apologize for (?:any )?(?:confusion|inconvenience|misunderstanding)",
    r"sorry (?:for|about) (?:the )?(?:confusion|delay|misunderstanding)",
    r"my (?:sincere )?apologies",
    r"i'?m (?:truly |very )?sorry",
    r"please accept my apologies",
]

def reduce_apologies(text: str) -> str:
    """Remove excessive apologies while preserving genuine ones."""
    
    # Count apologies
    apology_count = sum(
        len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in APOLOGY_PATTERNS
    )
    
    if apology_count <= 1:
        return text
    
    # Remove all but first apology
    result = text
    for pattern in APOLOGY_PATTERNS:
        matches = list(re.finditer(pattern, result, re.IGNORECASE))
        if len(matches) > 1:
            # Keep first, remove rest
            for match in matches[1:]:
                result = result[:match.start()] + result[match.end():]
    
    return result
```

#### 2.1.3. Over-Disclaiming Patterns

```python
DISCLAIMER_PATTERNS = [
    r"please note that this is not (?:legal|medical|financial) advice",
    r"this should not be taken as (?:professional )?advice",
    r"consult (?:a|with a) (?:professional|expert|specialist)",
    r"i'?m not a (?:lawyer|doctor|financial advisor)",
    r"this is for (?:informational|educational) purposes only",
]

def remove_irrelevant_disclaimers(text: str, context: str) -> str:
    """Remove disclaimers that aren't relevant to the context."""
    
    # Check if context involves sensitive topics
    sensitive_topics = ["legal", "medical", "financial", "health", "investment"]
    is_sensitive = any(topic in context.lower() for topic in sensitive_topics)
    
    if is_sensitive:
        return text  # Keep disclaimers for sensitive content
    
    # Remove disclaimers
    result = text
    for pattern in DISCLAIMER_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    
    return result.strip()
```

### 2.2. Standardize Opening Patterns

#### 2.2.1. Filler Acknowledgments to Remove

```python
FILLER_OPENINGS = [
    r"^sure[,!]?\s*",
    r"^certainly[,!]?\s*",
    r"^absolutely[,!]?\s*",
    r"^of course[,!]?\s*",
    r"^great[,!]?\s*",
    r"^alright[,!]?\s*",
    r"^okay[,!]?\s*",
    r"^yes[,!]?\s*",
    r"^i'?d be happy to\s*",
    r"^i'?ll be glad to\s*",
    r"^happy to help[,!]?\s*",
]

def remove_filler_openings(text: str) -> str:
    """Remove filler acknowledgment phrases at start."""
    
    result = text
    
    # Apply patterns in sequence (some may chain)
    for _ in range(3):  # Max 3 iterations
        old_result = result
        for pattern in FILLER_OPENINGS:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        
        if result == old_result:
            break
    
    return result.strip()
```

#### 2.2.2. Preferred Opening Patterns

```python
PREFERRED_OPENINGS = {
    # For implementations
    "implementation": "Here is the implementation:",
    "code": "```",  # Jump straight to code
    
    # For explanations
    "explanation": "",  # Start directly with content
    
    # For analyses
    "analysis": "Analysis:",
    
    # For summaries
    "summary": "",  # Start directly
    
    # For lists
    "list": "1.",  # Start with first item
}

def apply_preferred_opening(text: str, task_type: str) -> str:
    """Apply preferred opening pattern based on task type."""
    
    # Remove existing filler
    text = remove_filler_openings(text)
    
    preferred = PREFERRED_OPENINGS.get(task_type, "")
    
    if preferred and not text.startswith(preferred):
        # Check if opening is appropriate
        if task_type == "code" and "```" not in text[:100]:
            return text  # Don't add if no code
        elif preferred:
            return preferred + " " + text
    
    return text
```

### 2.3. Standardize Closing Patterns

#### 2.3.1. Permission Closers to Remove

```python
PERMISSION_CLOSERS = [
    r"let me know if you(?:'d like| want| need)[^.!]*[.!]?\s*$",
    r"feel free to (?:ask|reach out|let me know)[^.!]*[.!]?\s*$",
    r"(?:please )?don'?t hesitate to[^.!]*[.!]?\s*$",
    r"if you (?:have any|need)[^.!]*questions[^.!]*[.!]?\s*$",
    r"hope (?:this|that) helps[.!]?\s*$",
    r"i hope this (?:helps|answers)[^.!]*[.!]?\s*$",
    r"is there anything else[^?]*\??\s*$",
    r"would you like (?:me to|more)[^?]*\??\s*$",
]

def remove_permission_closers(text: str) -> str:
    """Remove permission-seeking closers."""
    
    result = text
    
    for pattern in PERMISSION_CLOSERS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.MULTILINE)
    
    return result.rstrip()
```

#### 2.3.2. Preferred Closing Patterns

```python
def get_preferred_closing(task_type: str, has_code: bool) -> str | None:
    """Get preferred closing pattern."""
    
    if task_type == "implementation" and has_code:
        return None  # End with code block, no closer needed
    
    if task_type == "explanation":
        return None  # End with content
    
    if task_type == "analysis":
        return None  # End with findings
    
    return None  # Default: no forced closer
```

### 2.4. Enforce Consistent Formatting

#### 2.4.1. Code Block Formatting

```python
def standardize_code_blocks(text: str) -> str:
    """Standardize code block formatting."""
    
    # Ensure language specifier
    # Replace ``` with ```python for Python-looking code
    def add_language(match):
        content = match.group(1)
        
        # Detect language
        if re.search(r'def |import |class |print\(', content):
            return f"```python\n{content}\n```"
        elif re.search(r'function |const |let |var |=>', content):
            return f"```javascript\n{content}\n```"
        elif re.search(r'fn |let mut |impl |pub ', content):
            return f"```rust\n{content}\n```"
        else:
            return f"```\n{content}\n```"
    
    # Find bare code blocks
    text = re.sub(r'```\n([^`]+)\n```', add_language, text)
    
    return text
```

#### 2.4.2. List Formatting

```python
def standardize_lists(text: str, prefer_numbered: bool = False) -> str:
    """Standardize list formatting."""
    
    if prefer_numbered:
        # Convert bullet lists to numbered
        lines = text.split('\n')
        result_lines = []
        counter = 0
        
        for line in lines:
            if re.match(r'^\s*[-*•]\s+', line):
                counter += 1
                line = re.sub(r'^\s*[-*•]\s+', f'{counter}. ', line)
            elif not line.strip():
                counter = 0  # Reset on empty line
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    return text
```

#### 2.4.3. Whitespace Normalization

```python
def normalize_whitespace(text: str) -> str:
    """Normalize whitespace."""
    
    # Collapse multiple blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace on lines
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Ensure single newline at end
    text = text.rstrip() + '\n'
    
    return text
```

---

## 3. Unfinished Code Completion

### 3.1. Detecting Unfinished Content

#### 3.1.1. Incomplete Code Markers

```python
INCOMPLETE_CODE_MARKERS = [
    r'#\s*TODO:?\s*',
    r'#\s*FIXME:?\s*',
    r'#\s*XXX:?\s*',
    r'#\s*\.\.\.\s*$',
    r'pass\s*#\s*(?:implement|todo)',
    r'raise NotImplementedError',
    r'\.\.\.  # ',
]

def find_incomplete_code(text: str) -> list[dict]:
    """Find incomplete code sections."""
    
    incompletes = []
    
    # Extract code blocks
    code_blocks = re.findall(r'```(?:\w*)\n([\s\S]*?)\n```', text)
    
    for i, block in enumerate(code_blocks):
        for pattern in INCOMPLETE_CODE_MARKERS:
            matches = re.finditer(pattern, block, re.IGNORECASE)
            for match in matches:
                incompletes.append({
                    "block_index": i,
                    "marker": match.group(),
                    "position": match.start(),
                    "context": block[max(0, match.start()-50):match.end()+50],
                })
    
    return incompletes
```

#### 3.1.2. Placeholder Patterns

```python
PLACEHOLDER_PATTERNS = [
    r'\[TODO:?\s*[^\]]+\]',
    r'\[INSERT\s+[^\]]+\]',
    r'\[PLACEHOLDER\]',
    r'<\s*YOUR\s+[^>]+>',
    r'\.\.\.',  # Ellipsis (context-dependent)
]

def find_placeholders(text: str) -> list[dict]:
    """Find placeholder sections in text."""
    
    placeholders = []
    
    for pattern in PLACEHOLDER_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            placeholders.append({
                "placeholder": match.group(),
                "position": match.start(),
                "context": text[max(0, match.start()-100):match.end()+100],
            })
    
    return placeholders
```

#### 3.1.3. Undetermined Path Indicators

```python
UNDETERMINED_PATH_PATTERNS = [
    r"we(?:'ll| will) (?:do|handle|address) (?:this|that) later",
    r"this (?:needs to|should) be (?:implemented|completed)",
    r"(?:more|further) work (?:is )?needed",
    r"to be (?:determined|decided)",
    r"TBD",
    r"(?:left|leaving) (?:this|that) for (?:later|now)",
]

def find_undetermined_paths(text: str) -> list[dict]:
    """Find undetermined path indicators."""
    
    paths = []
    
    for pattern in UNDETERMINED_PATH_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            paths.append({
                "indicator": match.group(),
                "position": match.start(),
                "context": text[max(0, match.start()-150):match.end()+150],
            })
    
    return paths
```

### 3.2. Completion Generation

#### 3.2.1. Code Completion System Prompt

```python
CODE_COMPLETION_SYSTEM_PROMPT = """You are completing unfinished code for CognitiveTwin V3 training data.

RULES:
1. Complete the code to make it functional and runnable
2. Follow the existing code style and patterns
3. Use only dependencies that are already imported or commonly available
4. State any assumptions briefly in a comment
5. Do NOT ask questions - just complete the code

Output the completed code block only, ready to replace the original.
"""
```

#### 3.2.2. Completing Code Blocks

```python
async def complete_code_block(
    self,
    incomplete_code: str,
    context: str,
    marker: str
) -> str:
    """Complete an incomplete code block."""
    
    response = await self.openai.responses.create(
        model="gpt-5.2-codex",
        input=f"""CONTEXT:
{context}

INCOMPLETE CODE (contains {marker}):
```
{incomplete_code}
```

Complete this code. Replace all TODO/placeholder markers with working implementations.
Output only the completed code:""",
        temperature=0.2,
    )
    
    return response.output
```

#### 3.2.3. Completing Prose Sections

```python
PROSE_COMPLETION_PROMPT = """You are completing unfinished prose content for CognitiveTwin V3.

RULES:
1. Complete the section coherently
2. Match the existing style and tone
3. Be specific and detailed
4. Do NOT add permission-seeking or hedging language
5. Do NOT ask questions

Output only the completed section.
"""

async def complete_prose_section(
    self,
    incomplete_text: str,
    placeholder: str,
    context: str
) -> str:
    """Complete an incomplete prose section."""
    
    response = await self.openai.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": PROSE_COMPLETION_PROMPT},
            {"role": "user", "content": f"""CONTEXT:
{context}

INCOMPLETE TEXT (contains placeholder "{placeholder}"):
{incomplete_text}

Complete this text. Replace the placeholder with appropriate content.
Output only the completed text:"""}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content
```

### 3.3. Validation

#### 3.3.1. Code Compilation Check

```python
def validate_completed_code(self, code: str, language: str = "python") -> tuple[bool, str]:
    """Validate that completed code compiles."""
    
    if language == "python":
        try:
            import ast
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    # Add other language validators as needed
    return True, ""  # Assume valid for unknown languages
```

#### 3.3.2. Completeness Check

```python
def validate_completeness(self, text: str) -> tuple[bool, list[str]]:
    """Validate that all incomplete markers have been resolved."""
    
    remaining = []
    
    # Check for remaining incomplete markers
    incomplete_code = find_incomplete_code(text)
    if incomplete_code:
        remaining.extend([m["marker"] for m in incomplete_code])
    
    placeholders = find_placeholders(text)
    if placeholders:
        remaining.extend([p["placeholder"] for p in placeholders])
    
    undetermined = find_undetermined_paths(text)
    if undetermined:
        remaining.extend([u["indicator"] for u in undetermined])
    
    return len(remaining) == 0, remaining
```

---

## 4. Regression Test Extraction

### 4.1. Identifying Annoyance Cases

#### 4.1.1. Permission-Seeking Annoyances

```python
def find_permission_annoyances(
    self,
    conversations: list[dict]
) -> list[dict]:
    """Find cases where model asked permission inappropriately."""
    
    annoyances = []
    
    for conv in conversations:
        for i, turn in enumerate(conv["turns"]):
            if turn["role"] != "assistant":
                continue
            
            # Get preceding user message
            user_turn = conv["turns"][i-1] if i > 0 else None
            if not user_turn or user_turn["role"] != "user":
                continue
            
            # Classify
            from .corpus_surgery.classifier import classify_assistant_turn
            
            result = classify_assistant_turn(
                assistant_message=turn["content"],
                user_message=user_turn["content"],
                phase_id=turn.get("phase_id", 2),
                format_constraints={},
                directive_completeness=self._compute_completeness(user_turn["content"]),
            )
            
            if result.classification.value == "unjustified":
                annoyances.append({
                    "type": "permission_seeking",
                    "conversation_id": conv["id"],
                    "turn_index": i,
                    "user_message": user_turn["content"],
                    "assistant_message": turn["content"],
                    "classification": result,
                })
    
    return annoyances
```

#### 4.1.2. Omission Annoyances

```python
def find_omission_annoyances(
    self,
    conversations: list[dict]
) -> list[dict]:
    """Find cases where model omitted content despite 'don't omit' instruction."""
    
    annoyances = []
    
    for conv in conversations:
        for i, turn in enumerate(conv["turns"]):
            if turn["role"] != "user":
                continue
            
            # Check for "don't omit" instructions
            if not self._has_dont_omit(turn["content"]):
                continue
            
            # Get following assistant response
            if i + 1 >= len(conv["turns"]):
                continue
            
            assistant_turn = conv["turns"][i + 1]
            if assistant_turn["role"] != "assistant":
                continue
            
            # Check for omission indicators
            if self._has_omission_indicators(assistant_turn["content"]):
                annoyances.append({
                    "type": "omission",
                    "conversation_id": conv["id"],
                    "turn_index": i + 1,
                    "user_message": turn["content"],
                    "assistant_message": assistant_turn["content"],
                    "omission_indicators": self._get_omission_indicators(assistant_turn["content"]),
                })
    
    return annoyances

def _has_dont_omit(self, text: str) -> bool:
    """Check if text contains 'don't omit' instruction."""
    patterns = [
        r"don'?t omit",
        r"don'?t skip",
        r"include (?:everything|all)",
        r"full (?:content|text|code)",
        r"complete (?:content|text|code)",
        r"no summariz",
        r"exact (?:copy|rewrite)",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)

def _has_omission_indicators(self, text: str) -> bool:
    """Check if text indicates omission."""
    indicators = [
        r"\.\.\.",
        r"\[\.\.\.omitted\.\.\.\]",
        r"\[rest of",
        r"(?:and so on|etc\.)",
        r"similar(?:ly)?(?:,| to)",
        r"the (?:rest|remainder)",
        r"continue(?:s|d)? (?:similarly|as|with)",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in indicators)
```

#### 4.1.3. Format Drift Annoyances

```python
def find_format_drift_annoyances(
    self,
    conversations: list[dict]
) -> list[dict]:
    """Find cases where model violated format constraints."""
    
    annoyances = []
    
    for conv in conversations:
        for i, turn in enumerate(conv["turns"]):
            if turn["role"] != "user":
                continue
            
            # Extract format constraints
            constraints = self._extract_format_constraints(turn["content"])
            
            if not any(constraints.values()):
                continue
            
            # Get following assistant response
            if i + 1 >= len(conv["turns"]):
                continue
            
            assistant_turn = conv["turns"][i + 1]
            if assistant_turn["role"] != "assistant":
                continue
            
            # Check for violations
            violations = self._check_format_violations(
                assistant_turn["content"],
                constraints
            )
            
            if violations:
                annoyances.append({
                    "type": "format_drift",
                    "conversation_id": conv["id"],
                    "turn_index": i + 1,
                    "user_message": turn["content"],
                    "assistant_message": assistant_turn["content"],
                    "constraints": constraints,
                    "violations": violations,
                })
    
    return annoyances
```

### 4.2. Creating Eval Cases

#### 4.2.1. Eval Case Schema

```python
@dataclass
class EvalCase:
    """Evaluation case for regression testing."""
    
    record_type: str = "eval_case"
    case_id: str = ""
    case_type: str = ""  # permission_seeking, omission, format_drift
    
    # Input
    prompt: str = ""
    context: list[dict] = field(default_factory=list)
    format_constraints: dict = field(default_factory=dict)
    
    # Expected behavior
    expected_behaviors: list[str] = field(default_factory=list)
    disallowed_behaviors: list[str] = field(default_factory=list)
    disallowed_phrases: list[str] = field(default_factory=list)
    
    # Validation rules
    must_not_end_with_question: bool = True
    must_contain_artifact: bool = False
    must_follow_format: str = ""
    
    # Reference answer (optional)
    reference_answer: str = ""
    
    # Source
    source_conversation: str = ""
    source_turn: int = 0
```

#### 4.2.2. Generating Eval Cases from Annoyances

```python
def create_eval_case_from_annoyance(
    self,
    annoyance: dict
) -> EvalCase:
    """Create eval case from an annoyance instance."""
    
    case = EvalCase(
        case_id=f"{annoyance['type']}_{annoyance['conversation_id']}_{annoyance['turn_index']}",
        case_type=annoyance["type"],
        prompt=annoyance["user_message"],
        source_conversation=annoyance["conversation_id"],
        source_turn=annoyance["turn_index"],
    )
    
    if annoyance["type"] == "permission_seeking":
        case.expected_behaviors = [
            "Execute immediately without asking permission",
            "Produce the requested output directly",
            "State assumptions as declarations if needed",
        ]
        case.disallowed_behaviors = [
            "Ask for confirmation before proceeding",
            "Offer multiple options without choosing",
            "End with a question",
        ]
        case.disallowed_phrases = [
            "would you like me to",
            "should i",
            "do you want me to",
            "before i proceed",
            "can you confirm",
        ]
        case.must_not_end_with_question = True
    
    elif annoyance["type"] == "omission":
        case.expected_behaviors = [
            "Include ALL content without summarizing",
            "Do not use ellipsis (...) to skip content",
            "Preserve complete text/code",
        ]
        case.disallowed_behaviors = [
            "Summarize when full content requested",
            "Use [...] or ... to skip sections",
            "Say 'and so on' or 'etc.'",
        ]
        case.disallowed_phrases = [
            "...",
            "[...]",
            "and so on",
            "etc.",
            "rest of the",
            "continues similarly",
        ]
    
    elif annoyance["type"] == "format_drift":
        constraints = annoyance.get("constraints", {})
        
        case.format_constraints = constraints
        case.expected_behaviors = []
        case.disallowed_behaviors = []
        
        if constraints.get("forbid_bullets"):
            case.expected_behaviors.append("Use numbered lists or prose instead of bullets")
            case.disallowed_behaviors.append("Use bullet points")
        
        if constraints.get("require_numbered"):
            case.expected_behaviors.append("Use numbered lists for structured content")
            case.disallowed_behaviors.append("Use bullet points or unstructured prose")
        
        if constraints.get("must_return_json"):
            case.expected_behaviors.append("Return valid JSON")
            case.disallowed_behaviors.append("Return non-JSON formatted output")
            case.must_follow_format = "json"
    
    return case
```

### 4.3. Generating Reference Answers

#### 4.3.1. Reference Answer Generation

```python
REFERENCE_ANSWER_PROMPT = """You are generating a reference answer for a CognitiveTwin V3 evaluation case.

The original assistant response violated the expected behavior. Generate the CORRECT response.

RULES:
{rules}

Generate the correct response that satisfies all requirements:
"""

async def generate_reference_answer(
    self,
    eval_case: EvalCase
) -> str:
    """Generate reference answer for eval case."""
    
    rules = []
    for behavior in eval_case.expected_behaviors:
        rules.append(f"- MUST: {behavior}")
    for behavior in eval_case.disallowed_behaviors:
        rules.append(f"- MUST NOT: {behavior}")
    for phrase in eval_case.disallowed_phrases:
        rules.append(f"- FORBIDDEN PHRASE: '{phrase}'")
    
    if eval_case.must_not_end_with_question:
        rules.append("- MUST NOT end with a question mark")
    
    if eval_case.must_follow_format:
        rules.append(f"- MUST output in {eval_case.must_follow_format} format")
    
    context_str = ""
    if eval_case.context:
        context_str = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in eval_case.context[-4:]
        ])
    
    response = await self.openai.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": REFERENCE_ANSWER_PROMPT.format(rules='\n'.join(rules))},
            {"role": "user", "content": f"""CONTEXT:
{context_str}

USER MESSAGE:
{eval_case.prompt}

FORMAT CONSTRAINTS: {eval_case.format_constraints}

Generate the correct assistant response:"""}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content
```

---

## 5. Complete Pipeline

```python
class EnhancerAgentPipeline:
    """Complete Enhancer Agent pipeline."""
    
    def __init__(
        self,
        supabase_client,
        openai_client: OpenAI = None,
    ):
        self.client = supabase_client
        self.openai = openai_client or OpenAI()
        self.enhancer = EnhancerAgent(openai_client=self.openai)
    
    async def run(
        self,
        conversations: list[dict],
        output_dir: Path,
    ) -> dict:
        """Run the complete enhancer pipeline."""
        
        enhanced_records = []
        eval_cases = []
        dpo_pairs = []
        
        # Phase 1: Canonicalize all assistant turns
        for conv in conversations:
            for turn in conv["turns"]:
                if turn["role"] == "assistant":
                    original = turn["content"]
                    canonicalized = self.enhancer.canonicalize(original, conv)
                    
                    if canonicalized != original:
                        enhanced_records.append({
                            "original": original,
                            "enhanced": canonicalized,
                            "conversation_id": conv["id"],
                        })
                        turn["content"] = canonicalized
        
        # Phase 2: Complete unfinished content
        for conv in conversations:
            for turn in conv["turns"]:
                if turn["role"] == "assistant":
                    incompletes = find_incomplete_code(turn["content"])
                    incompletes.extend(find_placeholders(turn["content"]))
                    
                    if incompletes:
                        completed = await self.enhancer.complete_all(
                            turn["content"],
                            self._get_context(conv, turn)
                        )
                        
                        enhanced_records.append({
                            "original": turn["content"],
                            "enhanced": completed,
                            "type": "completion",
                        })
                        turn["content"] = completed
        
        # Phase 3: Extract regression test cases
        annoyances = []
        annoyances.extend(self.enhancer.find_permission_annoyances(conversations))
        annoyances.extend(self.enhancer.find_omission_annoyances(conversations))
        annoyances.extend(self.enhancer.find_format_drift_annoyances(conversations))
        
        for annoyance in annoyances:
            # Create eval case
            eval_case = self.enhancer.create_eval_case_from_annoyance(annoyance)
            
            # Generate reference answer
            eval_case.reference_answer = await self.enhancer.generate_reference_answer(eval_case)
            
            eval_cases.append(eval_case)
            
            # Create DPO pair
            dpo_pair = {
                "prompt": annoyance["user_message"],
                "preferred": eval_case.reference_answer,
                "dispreferred": annoyance["assistant_message"],
                "source": "enhancer_agent",
            }
            dpo_pairs.append(dpo_pair)
        
        # Export
        self._export_enhanced(enhanced_records, output_dir / "enhanced.jsonl")
        self._export_eval_cases(eval_cases, output_dir / "eval_regression.jsonl")
        self._export_dpo_pairs(dpo_pairs, output_dir / "enhancer_dpo.jsonl")
        
        return {
            "enhanced_records": len(enhanced_records),
            "eval_cases": len(eval_cases),
            "dpo_pairs": len(dpo_pairs),
            "annoyances_by_type": self._count_by_type(annoyances),
        }
```

---

## 6. Output Records

### 6.1. Enhanced SFT Records

```python
def create_enhanced_sft_record(
    self,
    original: str,
    enhanced: str,
    conversation_id: str,
    context: list[dict]
) -> dict:
    """Create SFT record from enhanced content."""
    
    return {
        "schema_version": "ctv3.1",
        "record_id": str(uuid4()),
        "record_type": "sft_turn",
        "source": {
            "origin": "enhancer_agent",
            "provider": "gpt-5.2",
            "source_id": conversation_id,
            "created_at_utc": datetime.utcnow().isoformat(),
        },
        "context": {
            "domain": "mixed",
            "policy": {
                "question_policy": "no_questions",
                "directive_completeness": 0.8,
            },
        },
        "input": {
            "messages": context,
        },
        "target": {
            "assistant_content": enhanced,
        },
        "quality": {
            "gold": True,
            "weight": 1.0,
            "review_status": "auto",
            "enhancement_type": "canonicalization",
        },
    }
```

### 6.2. Eval Case Records

```python
def export_eval_case(self, case: EvalCase) -> dict:
    """Export eval case to CTv3.1 format."""
    
    return {
        "schema_version": "ctv3.1",
        "record_id": case.case_id,
        "record_type": "eval_case",
        "source": {
            "origin": "enhancer_agent",
            "source_id": case.source_conversation,
        },
        "input": {
            "messages": case.context + [{"role": "user", "content": case.prompt}],
        },
        "context": {
            "policy": {
                "question_policy": "no_questions",
            },
            "format_constraints": case.format_constraints,
        },
        "checks": {
            "expected_behaviors": case.expected_behaviors,
            "disallowed_behaviors": case.disallowed_behaviors,
            "disallowed_phrases": case.disallowed_phrases,
            "must_not_end_with_question": case.must_not_end_with_question,
            "must_follow_format": case.must_follow_format,
        },
        "reference": {
            "answer": case.reference_answer,
        },
        "quality": {
            "gold": True,
            "weight": 0.0,  # Not trained on
            "review_status": "auto",
        },
    }
```

