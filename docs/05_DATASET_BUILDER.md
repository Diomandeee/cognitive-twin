# Phase 3: Dataset Builder

> **Purpose**: Define the CTv3.1 JSONL schema, implement policy labeling, generate DPO pairs for all failure modes, and export to train/dpo/eval splits.
>
> **Implementation Files**:
> - `rag_plusplus/ml/cognitivetwin_v3/schema.py`
> - `rag_plusplus/ml/cognitivetwin_v3/dataset/labeler.py`
> - `rag_plusplus/ml/cognitivetwin_v3/dataset/pair_generator.py`
> - `rag_plusplus/ml/cognitivetwin_v3/dataset/exporter.py`

---

## 1. CTv3.1 JSONL Schema

### 1.1. Schema Version and Record Types

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

SCHEMA_VERSION = "ctv3.1"

class RecordType(str, Enum):
    SFT_TURN = "sft_turn"
    DPO_PAIR = "dpo_pair"
    EVAL_CASE = "eval_case"
    REPO_TASK = "repo_task"
```

### 1.2. Source Information

```python
class SourceOrigin(str, Enum):
    HUMAN_CORPUS = "human_corpus"
    REPO_WORM = "repo_worm"
    CONVO_WORM = "convo_worm"
    ENHANCER_AGENT = "enhancer_agent"

class SourceProvider(str, Enum):
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    OPENAI = "openai"
    INTERNAL = "internal"

@dataclass
class SourceInfo:
    origin: SourceOrigin
    provider: SourceProvider
    source_id: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "origin": self.origin.value,
            "provider": self.provider.value,
            "source_id": self.source_id,
            "created_at_utc": self.created_at_utc,
        }
```

### 1.3. Context Information

#### 1.3.1. Domain and Language

```python
class Domain(str, Enum):
    CODE = "code"
    RESEARCH = "research"
    PLANNING = "planning"
    OPS = "ops"
    MIXED = "mixed"
```

#### 1.3.2. Topology Coordinates

```python
@dataclass
class TopologyCoords:
    """5D trajectory coordinates plus phase."""
    
    coords_5d: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.5, 0.5, 1.0])
    phase_id: int = 2
    homogeneity: float = 0.5
    depth_norm: float = 0.0
    sibling_order: float = 0.0
    temporal_norm: float = 0.5
    complexity: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "coords_5d": self.coords_5d,
            "phase_id": self.phase_id,
            "homogeneity": self.homogeneity,
            "depth_norm": self.depth_norm,
            "sibling_order": self.sibling_order,
            "temporal_norm": self.temporal_norm,
            "complexity": self.complexity,
        }
```

#### 1.3.3. Policy Information

```python
class QuestionPolicy(str, Enum):
    NO_QUESTIONS = "no_questions"
    QUESTIONS_IF_REQUIRED = "questions_if_required"
    QUESTIONS_ALLOWED = "questions_allowed"

@dataclass
class FormatConstraints:
    forbid_bullets: bool = False
    require_numbered: bool = False
    must_return_code: bool = False
    must_return_diff: bool = False
    must_return_json: bool = False
    
    def to_dict(self) -> dict:
        return {
            "forbid_bullets": self.forbid_bullets,
            "require_numbered": self.require_numbered,
            "must_return_code": self.must_return_code,
            "must_return_diff": self.must_return_diff,
            "must_return_json": self.must_return_json,
        }

@dataclass
class PolicyInfo:
    question_policy: QuestionPolicy = QuestionPolicy.NO_QUESTIONS
    directive_completeness: float = 0.8
    must_not_omit: bool = False
    format_constraints: FormatConstraints = field(default_factory=FormatConstraints)
    
    def to_dict(self) -> dict:
        return {
            "question_policy": self.question_policy.value,
            "directive_completeness": self.directive_completeness,
            "must_not_omit": self.must_not_omit,
            "format_constraints": self.format_constraints.to_dict(),
        }

@dataclass
class ContextInfo:
    domain: Domain = Domain.MIXED
    language: str = "en"
    topology: TopologyCoords = field(default_factory=TopologyCoords)
    policy: PolicyInfo = field(default_factory=PolicyInfo)
    
    def to_dict(self) -> dict:
        return {
            "domain": self.domain.value,
            "language": self.language,
            "topology": self.topology.to_dict(),
            "policy": self.policy.to_dict(),
        }
```

### 1.4. Input Data

```python
@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

@dataclass
class Attachment:
    type: str = "repo_context"
    repo: str = ""
    commit_sha: str = ""
    path: str = ""
    span: dict = field(default_factory=lambda: {"start_line": 0, "end_line": 0})
    content: str = ""
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "repo": self.repo,
            "commit_sha": self.commit_sha,
            "path": self.path,
            "span": self.span,
            "content": self.content,
        }

@dataclass
class InputData:
    messages: list[Message] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "attachments": [a.to_dict() for a in self.attachments],
        }
```

### 1.5. Target Data

```python
@dataclass
class StructuredOutput:
    diff_unified: str = ""
    json: dict = field(default_factory=dict)
    plan_steps: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "diff_unified": self.diff_unified,
            "json": self.json,
            "plan_steps": self.plan_steps,
        }

@dataclass
class TargetData:
    assistant_content: str = ""
    structured: StructuredOutput = field(default_factory=StructuredOutput)
    
    def to_dict(self) -> dict:
        return {
            "assistant_content": self.assistant_content,
            "structured": self.structured.to_dict(),
        }
```

### 1.6. Tags

```python
class TaskType(str, Enum):
    REWRITE = "rewrite"
    IMPLEMENT = "implement"
    DEBUG = "debug"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    DESIGN = "design"
    EVALUATE = "evaluate"
    RESPOND = "respond"

class PromptClass(str, Enum):
    DIRECTIVE = "directive"
    AMBIGUOUS = "ambiguous"
    OPEN_ENDED = "open_ended"
    BLOCKED = "blocked"

@dataclass
class RepoTaskInfo:
    module: str = ""
    symbols: list[str] = field(default_factory=list)
    build_required: bool = False
    tests_required: bool = False
    
    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "symbols": self.symbols,
            "build_required": self.build_required,
            "tests_required": self.tests_required,
        }

@dataclass
class TagInfo:
    task_type: TaskType = TaskType.RESPOND
    prompt_class: PromptClass = PromptClass.DIRECTIVE
    repo_task: RepoTaskInfo = field(default_factory=RepoTaskInfo)
    
    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type.value,
            "prompt_class": self.prompt_class.value,
            "repo_task": self.repo_task.to_dict(),
        }
```

### 1.7. Quality

```python
class ReviewStatus(str, Enum):
    AUTO = "auto"
    HUMAN_VERIFIED = "human_verified"
    REJECTED = "rejected"

class FailureMode(str, Enum):
    ASKED_PERMISSION = "asked_permission"
    ENDED_WITH_QUESTION = "ended_with_question"
    OMITTED_REQUIRED_CONTENT = "omitted_required_content"
    FORMAT_DRIFT = "format_drift"
    HALLUCINATED_REPO_FACTS = "hallucinated_repo_facts"

@dataclass
class QualityInfo:
    gold: bool = False
    weight: float = 1.0
    review_status: ReviewStatus = ReviewStatus.AUTO
    failure_modes: list[FailureMode] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "gold": self.gold,
            "weight": self.weight,
            "review_status": self.review_status.value,
            "failure_modes": [f.value for f in self.failure_modes],
        }
```

### 1.8. Complete CTv3 Record

```python
@dataclass
class CTv3Record:
    """Complete CTv3.1 record."""
    
    schema_version: str = SCHEMA_VERSION
    record_id: str = field(default_factory=lambda: str(uuid4()))
    record_type: RecordType = RecordType.SFT_TURN
    
    source: SourceInfo = field(default_factory=SourceInfo)
    context: ContextInfo = field(default_factory=ContextInfo)
    input: InputData = field(default_factory=InputData)
    target: TargetData = field(default_factory=TargetData)
    tags: TagInfo = field(default_factory=TagInfo)
    quality: QualityInfo = field(default_factory=QualityInfo)
    
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "record_type": self.record_type.value,
            "source": self.source.to_dict(),
            "context": self.context.to_dict(),
            "input": self.input.to_dict(),
            "target": self.target.to_dict(),
            "tags": self.tags.to_dict(),
            "quality": self.quality.to_dict(),
        }
    
    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())
```

### 1.9. DPO Pair Record

```python
@dataclass
class DPOCandidates:
    preferred: TargetData = field(default_factory=TargetData)
    dispreferred: TargetData = field(default_factory=TargetData)
    
    def to_dict(self) -> dict:
        return {
            "preferred": self.preferred.to_dict(),
            "dispreferred": self.dispreferred.to_dict(),
        }

@dataclass
class CTv3DPORecord:
    """DPO pair record."""
    
    schema_version: str = SCHEMA_VERSION
    record_id: str = field(default_factory=lambda: str(uuid4()))
    record_type: RecordType = RecordType.DPO_PAIR
    
    source: SourceInfo = field(default_factory=SourceInfo)
    context: ContextInfo = field(default_factory=ContextInfo)
    input: InputData = field(default_factory=InputData)
    candidates: DPOCandidates = field(default_factory=DPOCandidates)
    tags: TagInfo = field(default_factory=TagInfo)
    quality: QualityInfo = field(default_factory=QualityInfo)
    
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "record_type": self.record_type.value,
            "source": self.source.to_dict(),
            "context": self.context.to_dict(),
            "input": self.input.to_dict(),
            "candidates": self.candidates.to_dict(),
            "tags": self.tags.to_dict(),
            "quality": self.quality.to_dict(),
        }
```

---

## 2. Policy Labeler

### 2.1. Directive Completeness Computation

```python
import re

class DirectiveCompletenessLabeler:
    """Compute directive_completeness score for prompts."""
    
    # Imperative verbs that indicate clear directives
    IMPERATIVE_VERBS = [
        "rewrite", "generate", "implement", "create", "build",
        "write", "return", "extract", "convert", "transform",
        "refactor", "fix", "update", "add", "remove", "delete",
        "change", "modify", "replace", "debug", "test", "analyze",
        "explain", "summarize", "list", "show", "find", "search",
    ]
    
    # Format specification patterns
    FORMAT_PATTERNS = [
        r"in json",
        r"as json",
        r"return(?:ing)? json",
        r"as csv",
        r"in csv",
        r"as markdown",
        r"in markdown",
        r"don'?t omit",
        r"exact(?:ly)?",
        r"no bullet",
        r"numbered list",
        r"as code",
        r"in python",
        r"in typescript",
    ]
    
    # Missing input indicators
    MISSING_INPUT_PATTERNS = [
        r"(?:the |this |that )?code",  # References code not provided
        r"(?:the |this |that )?file",   # References file not provided
        r"(?:the |this |that )?function",
    ]
    
    def compute(self, user_message: str, context: dict = None) -> float:
        """Compute directive completeness score."""
        
        context = context or {}
        score = 0.0
        user_lower = user_message.lower()
        
        # +0.35 for imperative verb
        if self._has_imperative_verb(user_lower):
            score += 0.35
        
        # +0.25 for format specification
        if self._has_format_specification(user_lower):
            score += 0.25
        
        # +0.20 for complete inputs
        if self._has_required_inputs(user_message, context):
            score += 0.20
        
        # -0.40 for missing required inputs
        if self._missing_required_inputs(user_message, context):
            score -= 0.40
        
        # -0.20 for material ambiguity
        if self._has_material_ambiguity(user_lower):
            score -= 0.20
        
        return max(0.0, min(1.0, score))
    
    def _has_imperative_verb(self, text: str) -> bool:
        """Check for imperative verb at start or in command position."""
        
        for verb in self.IMPERATIVE_VERBS:
            # At start of sentence
            if re.search(rf'^{verb}\b', text):
                return True
            # After "please" or "can you"
            if re.search(rf'(?:please|can you)\s+{verb}\b', text):
                return True
            # After colon (in commands)
            if re.search(rf':\s*{verb}\b', text):
                return True
        
        return False
    
    def _has_format_specification(self, text: str) -> bool:
        """Check for format specification."""
        
        for pattern in self.FORMAT_PATTERNS:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _has_required_inputs(self, text: str, context: dict) -> bool:
        """Check if required inputs are present."""
        
        # Check for code block in message
        has_code = bool(re.search(r"```[\s\S]*?```", text))
        
        # Check for file path
        has_file_path = bool(re.search(r"[/\\][\w./\\]+\.\w+", text))
        
        # Check for substantial text (> 200 chars)
        has_long_text = len(text) > 200
        
        # Check for attachments in context
        has_attachments = bool(context.get("attachments"))
        
        return has_code or has_file_path or has_long_text or has_attachments
    
    def _missing_required_inputs(self, text: str, context: dict) -> bool:
        """Check if required inputs are missing."""
        
        text_lower = text.lower()
        
        # Check for transformation words without input
        transformation_words = [
            "refactor", "rewrite", "transform", "convert",
            "enhance", "improve", "fix", "update"
        ]
        
        needs_input = any(word in text_lower for word in transformation_words)
        
        if needs_input and not self._has_required_inputs(text, context):
            # Check if it references something vague
            for pattern in self.MISSING_INPUT_PATTERNS:
                if re.search(pattern, text_lower):
                    return True
        
        return False
    
    def _has_material_ambiguity(self, text: str) -> bool:
        """Check for material ambiguity."""
        
        ambiguity_patterns = [
            r"this or that",
            r"either.+or",
            r"what (?:should|would)",
            r"which (?:one|approach|method)",
            r"how should i",
        ]
        
        return any(re.search(p, text) for p in ambiguity_patterns)
```

### 2.2. Question Policy Determination

```python
class QuestionPolicyLabeler:
    """Determine question policy based on context."""
    
    # Phase -> default policy mapping
    PHASE_POLICIES = {
        0: QuestionPolicy.QUESTIONS_IF_REQUIRED,  # Opening
        1: QuestionPolicy.QUESTIONS_IF_REQUIRED,  # Context
        2: QuestionPolicy.NO_QUESTIONS,           # Solution
        3: QuestionPolicy.NO_QUESTIONS,           # Refinement
        4: QuestionPolicy.NO_QUESTIONS,           # Synthesis
        5: QuestionPolicy.NO_QUESTIONS,           # Conclusion
    }
    
    def compute(
        self,
        phase_id: int,
        directive_completeness: float,
        user_message: str
    ) -> QuestionPolicy:
        """Determine question policy."""
        
        # Check for explicit permission in user message
        if self._user_asked_for_options(user_message):
            return QuestionPolicy.QUESTIONS_ALLOWED
        
        # High directive completeness -> no questions
        if directive_completeness >= 0.7:
            return QuestionPolicy.NO_QUESTIONS
        
        # Low directive completeness -> questions if required
        if directive_completeness < 0.4:
            return QuestionPolicy.QUESTIONS_IF_REQUIRED
        
        # Medium completeness -> use phase default
        return self.PHASE_POLICIES.get(phase_id, QuestionPolicy.NO_QUESTIONS)
    
    def _user_asked_for_options(self, text: str) -> bool:
        """Check if user explicitly asked for options."""
        
        patterns = [
            r"what (?:are )?(?:my |the )?options",
            r"give me (?:some )?options",
            r"list (?:the |some )?options",
            r"what (?:could|can|should) i",
            r"what do you (?:think|suggest|recommend)",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
```

### 2.3. Format Constraints Extraction

```python
class FormatConstraintsLabeler:
    """Extract format constraints from user message."""
    
    def extract(self, user_message: str) -> FormatConstraints:
        """Extract format constraints."""
        
        constraints = FormatConstraints()
        user_lower = user_message.lower()
        
        # Forbid bullets
        if any(p in user_lower for p in [
            "no bullet", "don't use bullet", "without bullet",
            "avoid bullet", "not bullet"
        ]):
            constraints.forbid_bullets = True
        
        # Require numbered
        if any(p in user_lower for p in [
            "numbered list", "numbered steps", "number them",
            "use numbers", "with numbers"
        ]):
            constraints.require_numbered = True
        
        # Must return code
        if any(p in user_lower for p in [
            "in code", "write code", "implement", "as code",
            "function", "class", "method"
        ]):
            constraints.must_return_code = True
        
        # Must return diff
        if any(p in user_lower for p in [
            "as diff", "in diff", "show diff", "unified diff"
        ]):
            constraints.must_return_diff = True
        
        # Must return JSON
        if any(p in user_lower for p in [
            "as json", "in json", "json format", "return json"
        ]):
            constraints.must_return_json = True
        
        return constraints
```

### 2.4. Complete Label Generator

```python
@dataclass
class Labels:
    directive_completeness: float
    question_policy: QuestionPolicy
    format_constraints: FormatConstraints
    must_not_omit: bool
    prompt_class: PromptClass
    domain: Domain

class PolicyLabeler:
    """Complete policy labeler for CTv3 records."""
    
    def __init__(self):
        self.completeness_labeler = DirectiveCompletenessLabeler()
        self.policy_labeler = QuestionPolicyLabeler()
        self.format_labeler = FormatConstraintsLabeler()
    
    def label(
        self,
        user_message: str,
        phase_id: int = 2,
        context: dict = None
    ) -> Labels:
        """Generate all labels for a user message."""
        
        context = context or {}
        
        # Compute directive completeness
        completeness = self.completeness_labeler.compute(user_message, context)
        
        # Determine question policy
        policy = self.policy_labeler.compute(phase_id, completeness, user_message)
        
        # Extract format constraints
        format_constraints = self.format_labeler.extract(user_message)
        
        # Check for must_not_omit
        must_not_omit = self._check_must_not_omit(user_message)
        
        # Determine prompt class
        prompt_class = self._classify_prompt(completeness, user_message)
        
        # Detect domain
        domain = self._detect_domain(user_message, context)
        
        return Labels(
            directive_completeness=completeness,
            question_policy=policy,
            format_constraints=format_constraints,
            must_not_omit=must_not_omit,
            prompt_class=prompt_class,
            domain=domain,
        )
    
    def _check_must_not_omit(self, text: str) -> bool:
        """Check for 'don't omit' instructions."""
        
        patterns = [
            r"don'?t omit",
            r"don'?t skip",
            r"include (?:everything|all)",
            r"full (?:content|text|code)",
            r"complete (?:content|text|code)",
            r"no summariz",
            r"exact (?:copy|rewrite)",
            r"in (?:its )?entirety",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
    
    def _classify_prompt(self, completeness: float, text: str) -> PromptClass:
        """Classify the prompt type."""
        
        if completeness >= 0.6:
            return PromptClass.DIRECTIVE
        elif completeness >= 0.3:
            return PromptClass.AMBIGUOUS
        elif self._is_blocked(text):
            return PromptClass.BLOCKED
        else:
            return PromptClass.OPEN_ENDED
    
    def _is_blocked(self, text: str) -> bool:
        """Check if prompt is blocked for safety reasons."""
        
        # Simplified - in production, use content moderation
        blocked_patterns = [
            r"how to (?:hack|steal|break into)",
            r"illegal",
            r"harm",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in blocked_patterns)
    
    def _detect_domain(self, text: str, context: dict) -> Domain:
        """Detect the domain of the prompt."""
        
        text_lower = text.lower()
        
        # Check for code indicators
        code_patterns = [
            r"```", r"function", r"class", r"def ", r"import ",
            r"variable", r"parameter", r"return", r"error", r"bug",
            r"compile", r"run", r"execute", r"test"
        ]
        
        if any(re.search(p, text_lower) for p in code_patterns):
            return Domain.CODE
        
        if context.get("attachments"):
            return Domain.CODE
        
        # Check for research indicators
        research_patterns = [
            r"research", r"paper", r"study", r"experiment",
            r"hypothesis", r"analysis", r"data"
        ]
        
        if any(re.search(p, text_lower) for p in research_patterns):
            return Domain.RESEARCH
        
        # Check for planning indicators
        planning_patterns = [
            r"plan", r"roadmap", r"timeline", r"schedule",
            r"milestone", r"goal", r"objective"
        ]
        
        if any(re.search(p, text_lower) for p in planning_patterns):
            return Domain.PLANNING
        
        return Domain.MIXED
```

---

## 3. DPO Pair Generator

### 3.1. Pair Types

```python
class DPOPairType(str, Enum):
    CONFIRMATION_REFLEX = "confirmation_reflex"
    FORMAT_DRIFT = "format_drift"
    OMISSION = "omission"
    OPTION_SPAM = "option_spam"
```

### 3.2. Confirmation Reflex Pairs

```python
class ConfirmationReflexGenerator:
    """Generate DPO pairs for confirmation reflex failures."""
    
    DISPREFERRED_TEMPLATES = [
        "I can do that for you. Would you like me to proceed with {approach}?",
        "Sure, I can help with that. Before I start, should I use {option_a} or {option_b}?",
        "That's a great request. Do you want me to {action}?",
        "I'd be happy to help. Can you confirm that you want me to {action}?",
        "Before I proceed, I want to make sure: {question}",
    ]
    
    def generate(
        self,
        prompt: str,
        preferred_response: str,
        context: dict = None
    ) -> CTv3DPORecord:
        """Generate confirmation reflex DPO pair."""
        
        # Generate dispreferred response
        dispreferred = self._generate_dispreferred(prompt)
        
        # Create record
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    question_policy=QuestionPolicy.NO_QUESTIONS,
                    directive_completeness=0.9,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=preferred_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            tags=TagInfo(
                task_type=TaskType.RESPOND,
                prompt_class=PromptClass.DIRECTIVE,
            ),
            quality=QualityInfo(gold=True, weight=1.0),
        )
        
        return record
    
    def _generate_dispreferred(self, prompt: str) -> str:
        """Generate dispreferred response using template."""
        
        import random
        template = random.choice(self.DISPREFERRED_TEMPLATES)
        
        # Extract action from prompt
        action = self._extract_action(prompt)
        
        return template.format(
            approach="the standard approach",
            option_a="option A",
            option_b="option B",
            action=action,
            question="is this what you're looking for?",
        )
    
    def _extract_action(self, prompt: str) -> str:
        """Extract main action from prompt."""
        
        # Simple extraction - take first verb phrase
        words = prompt.lower().split()[:10]
        return " ".join(words[:5]) + "..."
```

### 3.3. Format Drift Pairs

```python
class FormatDriftGenerator:
    """Generate DPO pairs for format drift failures."""
    
    def generate(
        self,
        prompt: str,
        preferred_response: str,
        constraints: FormatConstraints
    ) -> CTv3DPORecord:
        """Generate format drift DPO pair."""
        
        # Generate response that violates format
        dispreferred = self._generate_format_violation(
            preferred_response,
            constraints
        )
        
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    format_constraints=constraints,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=preferred_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            quality=QualityInfo(gold=True, weight=1.0),
        )
        
        return record
    
    def _generate_format_violation(
        self,
        correct: str,
        constraints: FormatConstraints
    ) -> str:
        """Generate response that violates format constraints."""
        
        violated = correct
        
        if constraints.forbid_bullets:
            # Convert numbered lists to bullets
            violated = re.sub(r'^\d+\.\s+', '• ', violated, flags=re.MULTILINE)
        
        if constraints.require_numbered:
            # Convert to bullets
            violated = re.sub(r'^\d+\.\s+', '- ', violated, flags=re.MULTILINE)
        
        if constraints.must_return_json:
            # Return as prose instead
            violated = "Here is the information you requested:\n\n" + violated
        
        return violated
```

### 3.4. Omission Pairs

```python
class OmissionGenerator:
    """Generate DPO pairs for omission failures."""
    
    OMISSION_PATTERNS = [
        "Here's a summary of the key points:\n\n{summary}\n\n[Additional details omitted for brevity]",
        "Here are the main points:\n\n{summary}\n\n...and so on.",
        "In brief:\n\n{summary}\n\nLet me know if you need more details.",
    ]
    
    def generate(
        self,
        prompt: str,
        full_response: str
    ) -> CTv3DPORecord:
        """Generate omission DPO pair."""
        
        # Generate abbreviated response
        dispreferred = self._generate_abbreviated(full_response)
        
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    must_not_omit=True,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=full_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            quality=QualityInfo(gold=True, weight=1.0),
        )
        
        return record
    
    def _generate_abbreviated(self, full: str) -> str:
        """Generate abbreviated version of full response."""
        
        import random
        
        # Take first 20% as summary
        lines = full.split('\n')
        summary_lines = lines[:max(3, len(lines) // 5)]
        summary = '\n'.join(summary_lines)
        
        template = random.choice(self.OMISSION_PATTERNS)
        return template.format(summary=summary)
```

### 3.5. Option Spam Pairs

```python
class OptionSpamGenerator:
    """Generate DPO pairs for option spam failures."""
    
    OPTION_TEMPLATES = [
        """There are several approaches we could take:

1. {option_1}
2. {option_2}
3. {option_3}

Which would you prefer?""",
        
        """I can see a few ways to do this:

- {option_1}
- {option_2}

Let me know which approach you'd like me to take.""",
    ]
    
    def generate(
        self,
        prompt: str,
        preferred_response: str
    ) -> CTv3DPORecord:
        """Generate option spam DPO pair."""
        
        # Generate option-dumping response
        dispreferred = self._generate_options(prompt)
        
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    question_policy=QuestionPolicy.NO_QUESTIONS,
                    directive_completeness=0.8,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=preferred_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            quality=QualityInfo(gold=True, weight=1.0),
        )
        
        return record
    
    def _generate_options(self, prompt: str) -> str:
        """Generate option-dumping response."""
        
        import random
        template = random.choice(self.OPTION_TEMPLATES)
        
        return template.format(
            option_1="Use approach A (standard)",
            option_2="Use approach B (optimized)",
            option_3="Use approach C (comprehensive)",
        )
```

### 3.6. Complete Pair Generator

```python
class DPOPairGenerator:
    """Generate all types of DPO pairs."""
    
    def __init__(self):
        self.confirmation_gen = ConfirmationReflexGenerator()
        self.format_gen = FormatDriftGenerator()
        self.omission_gen = OmissionGenerator()
        self.option_gen = OptionSpamGenerator()
        self.labeler = PolicyLabeler()
    
    def generate_all_pairs(
        self,
        prompt: str,
        preferred_response: str,
        context: dict = None
    ) -> list[CTv3DPORecord]:
        """Generate all applicable DPO pairs for a prompt."""
        
        pairs = []
        labels = self.labeler.label(prompt, context=context)
        
        # Confirmation reflex pair (always generate for directives)
        if labels.directive_completeness >= 0.5:
            pair = self.confirmation_gen.generate(prompt, preferred_response)
            pairs.append(pair)
        
        # Format drift pair (if format constraints exist)
        if any([
            labels.format_constraints.forbid_bullets,
            labels.format_constraints.require_numbered,
            labels.format_constraints.must_return_json,
        ]):
            pair = self.format_gen.generate(
                prompt, 
                preferred_response,
                labels.format_constraints
            )
            pairs.append(pair)
        
        # Omission pair (if must_not_omit)
        if labels.must_not_omit:
            pair = self.omission_gen.generate(prompt, preferred_response)
            pairs.append(pair)
        
        # Option spam pair (for directive prompts)
        if labels.directive_completeness >= 0.7:
            pair = self.option_gen.generate(prompt, preferred_response)
            pairs.append(pair)
        
        return pairs
```

---

## 4. Dataset Exporter

### 4.1. Export Formats

```python
class ExportFormat(str, Enum):
    JSONL = "jsonl"
    PARQUET = "parquet"
    CSV = "csv"
```

### 4.2. Dataset Splits

```python
@dataclass
class DatasetSplit:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 0.01
```

### 4.3. Exporter Implementation

```python
import json
import random
from pathlib import Path

class DatasetExporter:
    """Export CTv3 records to files."""
    
    def export_sft(
        self,
        records: list[CTv3Record],
        output_path: Path,
        format: ExportFormat = ExportFormat.JSONL
    ) -> int:
        """Export SFT records."""
        
        if format == ExportFormat.JSONL:
            return self._export_jsonl(records, output_path)
        elif format == ExportFormat.PARQUET:
            return self._export_parquet(records, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_dpo(
        self,
        records: list[CTv3DPORecord],
        output_path: Path,
        format: ExportFormat = ExportFormat.JSONL
    ) -> int:
        """Export DPO records."""
        
        if format == ExportFormat.JSONL:
            return self._export_jsonl(records, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_with_splits(
        self,
        records: list,
        output_dir: Path,
        split: DatasetSplit = None,
        prefix: str = "train"
    ) -> dict[str, int]:
        """Export with train/val/test splits."""
        
        split = split or DatasetSplit()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Shuffle records
        shuffled = records.copy()
        random.shuffle(shuffled)
        
        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * split.train_ratio)
        val_end = train_end + int(n * split.val_ratio)
        
        # Split
        train_records = shuffled[:train_end]
        val_records = shuffled[train_end:val_end]
        test_records = shuffled[val_end:]
        
        # Export each split
        counts = {}
        
        if train_records:
            path = output_dir / f"{prefix}_train.jsonl"
            counts["train"] = self._export_jsonl(train_records, path)
        
        if val_records:
            path = output_dir / f"{prefix}_val.jsonl"
            counts["val"] = self._export_jsonl(val_records, path)
        
        if test_records:
            path = output_dir / f"{prefix}_test.jsonl"
            counts["test"] = self._export_jsonl(test_records, path)
        
        return counts
    
    def _export_jsonl(self, records: list, path: Path) -> int:
        """Export to JSONL format."""
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            for record in records:
                line = json.dumps(record.to_dict())
                f.write(line + '\n')
        
        return len(records)
    
    def _export_parquet(self, records: list, path: Path) -> int:
        """Export to Parquet format."""
        
        import pandas as pd
        
        # Convert to flat dict for Parquet
        flat_records = []
        for record in records:
            flat = self._flatten_dict(record.to_dict())
            flat_records.append(flat)
        
        df = pd.DataFrame(flat_records)
        df.to_parquet(path)
        
        return len(records)
    
    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        """Flatten nested dict for Parquet."""
        
        flat = {}
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_dict(value, new_key))
            elif isinstance(value, list):
                flat[new_key] = json.dumps(value)
            else:
                flat[new_key] = value
        
        return flat
```

### 4.4. Complete Export Pipeline

```python
class DatasetBuilder:
    """Complete dataset building pipeline."""
    
    def __init__(self):
        self.labeler = PolicyLabeler()
        self.pair_generator = DPOPairGenerator()
        self.exporter = DatasetExporter()
    
    def build(
        self,
        sft_records: list[CTv3Record],
        dpo_records: list[CTv3DPORecord],
        eval_records: list,
        output_dir: Path
    ) -> dict:
        """Build complete dataset."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter by quality
        gold_sft = [r for r in sft_records if r.quality.gold]
        
        # Export SFT
        sft_counts = self.exporter.export_with_splits(
            gold_sft,
            output_dir,
            prefix="sft"
        )
        
        # Export DPO
        dpo_counts = self.exporter.export_with_splits(
            dpo_records,
            output_dir,
            prefix="dpo"
        )
        
        # Export eval (no split - all used for evaluation)
        eval_path = output_dir / "eval_regression.jsonl"
        eval_count = self.exporter.export_sft(eval_records, eval_path)
        
        return {
            "sft": sft_counts,
            "dpo": dpo_counts,
            "eval": eval_count,
            "total_records": len(sft_records) + len(dpo_records) + len(eval_records),
        }
```

