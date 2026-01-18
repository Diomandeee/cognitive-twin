"""
CTv3.1 Schema for CognitiveTwin V3.

Defines the complete schema for training data:
- Enums for record types, sources, domains, policies
- Dataclasses for structured data
- Complete record types (SFT, DPO, Eval)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


SCHEMA_VERSION = "ctv3.1"


# =============================================================================
# ENUMS
# =============================================================================

class RecordType(str, Enum):
    """Types of training records."""
    SFT_TURN = "sft_turn"
    DPO_PAIR = "dpo_pair"
    EVAL_CASE = "eval_case"
    REPO_TASK = "repo_task"


class SourceOrigin(str, Enum):
    """Origin of training data."""
    HUMAN_CORPUS = "human_corpus"
    REPO_WORM = "repo_worm"
    CONVO_WORM = "convo_worm"
    ENHANCER_AGENT = "enhancer_agent"
    CORPUS_SURGERY = "corpus_surgery"


class SourceProvider(str, Enum):
    """Provider that generated the content."""
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    OPENAI = "openai"
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_CODEX = "gpt-5.2-codex"
    INTERNAL = "internal"


class Domain(str, Enum):
    """Domain of the content."""
    CODE = "code"
    RESEARCH = "research"
    PLANNING = "planning"
    OPS = "ops"
    MIXED = "mixed"


class QuestionPolicy(str, Enum):
    """Policy for questions in responses."""
    NO_QUESTIONS = "no_questions"
    QUESTIONS_IF_REQUIRED = "questions_if_required"
    QUESTIONS_ALLOWED = "questions_allowed"


class TaskType(str, Enum):
    """Type of task."""
    REWRITE = "rewrite"
    IMPLEMENT = "implement"
    DEBUG = "debug"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    DESIGN = "design"
    EVALUATE = "evaluate"
    RESPOND = "respond"
    COMPLETE = "complete"
    TEST = "test"


class PromptClass(str, Enum):
    """Classification of prompts."""
    DIRECTIVE = "directive"
    AMBIGUOUS = "ambiguous"
    OPEN_ENDED = "open_ended"
    BLOCKED = "blocked"


class ReviewStatus(str, Enum):
    """Review status of records."""
    AUTO = "auto"
    HUMAN_VERIFIED = "human_verified"
    REJECTED = "rejected"


class FailureMode(str, Enum):
    """Failure modes for DPO training."""
    ASKED_PERMISSION = "asked_permission"
    ENDED_WITH_QUESTION = "ended_with_question"
    OMITTED_REQUIRED_CONTENT = "omitted_required_content"
    FORMAT_DRIFT = "format_drift"
    HALLUCINATED_REPO_FACTS = "hallucinated_repo_facts"
    OPTION_SPAM = "option_spam"


class DPOPairType(str, Enum):
    """Types of DPO pairs."""
    CONFIRMATION_REFLEX = "confirmation_reflex"
    FORMAT_DRIFT = "format_drift"
    OMISSION = "omission"
    OPTION_SPAM = "option_spam"
    FRICTION_REPAIR = "friction_repair"


# =============================================================================
# SOURCE INFORMATION
# =============================================================================

@dataclass
class SourceInfo:
    """Source information for a record."""
    
    origin: SourceOrigin = SourceOrigin.HUMAN_CORPUS
    provider: SourceProvider = SourceProvider.INTERNAL
    source_id: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "origin": self.origin.value if isinstance(self.origin, Enum) else self.origin,
            "provider": self.provider.value if isinstance(self.provider, Enum) else self.provider,
            "source_id": self.source_id,
            "created_at_utc": self.created_at_utc,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SourceInfo":
        return cls(
            origin=SourceOrigin(data.get("origin", "human_corpus")),
            provider=SourceProvider(data.get("provider", "internal")),
            source_id=data.get("source_id", ""),
            created_at_utc=data.get("created_at_utc", ""),
        )


# =============================================================================
# TOPOLOGY COORDINATES
# =============================================================================

@dataclass
class TopologyCoords:
    """5D trajectory coordinates plus phase."""
    
    coords_5d: list = field(default_factory=lambda: [0.0, 0.0, 0.5, 0.5, 1.0])
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
    
    @classmethod
    def from_dict(cls, data: dict) -> "TopologyCoords":
        return cls(
            coords_5d=data.get("coords_5d", [0.0, 0.0, 0.5, 0.5, 1.0]),
            phase_id=data.get("phase_id", 2),
            homogeneity=data.get("homogeneity", 0.5),
            depth_norm=data.get("depth_norm", 0.0),
            sibling_order=data.get("sibling_order", 0.0),
            temporal_norm=data.get("temporal_norm", 0.5),
            complexity=data.get("complexity", 1.0),
        )


# =============================================================================
# FORMAT CONSTRAINTS
# =============================================================================

@dataclass
class FormatConstraints:
    """Format constraints for responses."""
    
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
    
    @classmethod
    def from_dict(cls, data: dict) -> "FormatConstraints":
        return cls(
            forbid_bullets=data.get("forbid_bullets", False),
            require_numbered=data.get("require_numbered", False),
            must_return_code=data.get("must_return_code", False),
            must_return_diff=data.get("must_return_diff", False),
            must_return_json=data.get("must_return_json", False),
        )
    
    def any_active(self) -> bool:
        """Check if any constraints are active."""
        return any([
            self.forbid_bullets,
            self.require_numbered,
            self.must_return_code,
            self.must_return_diff,
            self.must_return_json,
        ])


# =============================================================================
# POLICY INFORMATION
# =============================================================================

@dataclass
class PolicyInfo:
    """Policy information for a record."""
    
    question_policy: QuestionPolicy = QuestionPolicy.NO_QUESTIONS
    directive_completeness: float = 0.8
    must_not_omit: bool = False
    format_constraints: FormatConstraints = field(default_factory=FormatConstraints)
    
    def to_dict(self) -> dict:
        return {
            "question_policy": self.question_policy.value if isinstance(self.question_policy, Enum) else self.question_policy,
            "directive_completeness": self.directive_completeness,
            "must_not_omit": self.must_not_omit,
            "format_constraints": self.format_constraints.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PolicyInfo":
        return cls(
            question_policy=QuestionPolicy(data.get("question_policy", "no_questions")),
            directive_completeness=data.get("directive_completeness", 0.8),
            must_not_omit=data.get("must_not_omit", False),
            format_constraints=FormatConstraints.from_dict(data.get("format_constraints", {})),
        )


# =============================================================================
# CONTEXT INFORMATION
# =============================================================================

@dataclass
class ContextInfo:
    """Context information for a record."""
    
    domain: Domain = Domain.MIXED
    language: str = "en"
    topology: TopologyCoords = field(default_factory=TopologyCoords)
    policy: PolicyInfo = field(default_factory=PolicyInfo)
    
    def to_dict(self) -> dict:
        return {
            "domain": self.domain.value if isinstance(self.domain, Enum) else self.domain,
            "language": self.language,
            "topology": self.topology.to_dict(),
            "policy": self.policy.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ContextInfo":
        return cls(
            domain=Domain(data.get("domain", "mixed")),
            language=data.get("language", "en"),
            topology=TopologyCoords.from_dict(data.get("topology", {})),
            policy=PolicyInfo.from_dict(data.get("policy", {})),
        )


# =============================================================================
# MESSAGE AND ATTACHMENT
# =============================================================================

@dataclass
class Message:
    """A single message in the conversation."""
    
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
        )


@dataclass
class Attachment:
    """Code or file attachment."""
    
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
    
    @classmethod
    def from_dict(cls, data: dict) -> "Attachment":
        return cls(
            type=data.get("type", "repo_context"),
            repo=data.get("repo", ""),
            commit_sha=data.get("commit_sha", ""),
            path=data.get("path", ""),
            span=data.get("span", {"start_line": 0, "end_line": 0}),
            content=data.get("content", ""),
        )


# =============================================================================
# INPUT DATA
# =============================================================================

@dataclass
class InputData:
    """Input data for a record."""
    
    messages: list = field(default_factory=list)
    attachments: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "messages": [
                m.to_dict() if hasattr(m, 'to_dict') else m
                for m in self.messages
            ],
            "attachments": [
                a.to_dict() if hasattr(a, 'to_dict') else a
                for a in self.attachments
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "InputData":
        return cls(
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            attachments=[Attachment.from_dict(a) for a in data.get("attachments", [])],
        )


# =============================================================================
# TARGET DATA
# =============================================================================

@dataclass
class StructuredOutput:
    """Structured output data."""
    
    diff_unified: str = ""
    json_data: dict = field(default_factory=dict)
    plan_steps: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "diff_unified": self.diff_unified,
            "json": self.json_data,
            "plan_steps": self.plan_steps,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StructuredOutput":
        return cls(
            diff_unified=data.get("diff_unified", ""),
            json_data=data.get("json", {}),
            plan_steps=data.get("plan_steps", []),
        )


@dataclass
class TargetData:
    """Target (expected output) data."""
    
    assistant_content: str = ""
    structured: StructuredOutput = field(default_factory=StructuredOutput)
    
    def to_dict(self) -> dict:
        return {
            "assistant_content": self.assistant_content,
            "structured": self.structured.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TargetData":
        return cls(
            assistant_content=data.get("assistant_content", ""),
            structured=StructuredOutput.from_dict(data.get("structured", {})),
        )


# =============================================================================
# TAG INFORMATION
# =============================================================================

@dataclass
class RepoTaskInfo:
    """Information for repo tasks."""
    
    module: str = ""
    symbols: list = field(default_factory=list)
    build_required: bool = False
    tests_required: bool = False
    
    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "symbols": self.symbols,
            "build_required": self.build_required,
            "tests_required": self.tests_required,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RepoTaskInfo":
        return cls(
            module=data.get("module", ""),
            symbols=data.get("symbols", []),
            build_required=data.get("build_required", False),
            tests_required=data.get("tests_required", False),
        )


@dataclass
class TagInfo:
    """Tag information for a record."""
    
    task_type: TaskType = TaskType.RESPOND
    prompt_class: PromptClass = PromptClass.DIRECTIVE
    repo_task: RepoTaskInfo = field(default_factory=RepoTaskInfo)
    dpo_reason: str = ""
    branch_type: str = ""
    enhancement_type: str = ""
    
    def to_dict(self) -> dict:
        result = {
            "task_type": self.task_type.value if isinstance(self.task_type, Enum) else self.task_type,
            "prompt_class": self.prompt_class.value if isinstance(self.prompt_class, Enum) else self.prompt_class,
            "repo_task": self.repo_task.to_dict(),
        }
        
        if self.dpo_reason:
            result["dpo_reason"] = self.dpo_reason
        if self.branch_type:
            result["branch_type"] = self.branch_type
        if self.enhancement_type:
            result["enhancement_type"] = self.enhancement_type
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "TagInfo":
        return cls(
            task_type=TaskType(data.get("task_type", "respond")),
            prompt_class=PromptClass(data.get("prompt_class", "directive")),
            repo_task=RepoTaskInfo.from_dict(data.get("repo_task", {})),
            dpo_reason=data.get("dpo_reason", ""),
            branch_type=data.get("branch_type", ""),
            enhancement_type=data.get("enhancement_type", ""),
        )


# =============================================================================
# QUALITY INFORMATION
# =============================================================================

@dataclass
class QualityInfo:
    """Quality information for a record."""
    
    gold: bool = False
    weight: float = 1.0
    review_status: ReviewStatus = ReviewStatus.AUTO
    failure_modes: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "gold": self.gold,
            "weight": self.weight,
            "review_status": self.review_status.value if isinstance(self.review_status, Enum) else self.review_status,
            "failure_modes": [
                f.value if isinstance(f, Enum) else f
                for f in self.failure_modes
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QualityInfo":
        return cls(
            gold=data.get("gold", False),
            weight=data.get("weight", 1.0),
            review_status=ReviewStatus(data.get("review_status", "auto")),
            failure_modes=[
                FailureMode(f) if isinstance(f, str) else f
                for f in data.get("failure_modes", [])
            ],
        )


# =============================================================================
# COMPLETE CTv3 RECORD
# =============================================================================

@dataclass
class CTv3Record:
    """Complete CTv3.1 SFT record."""
    
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
            "record_type": self.record_type.value if isinstance(self.record_type, Enum) else self.record_type,
            "source": self.source.to_dict(),
            "context": self.context.to_dict(),
            "input": self.input.to_dict(),
            "target": self.target.to_dict(),
            "tags": self.tags.to_dict(),
            "quality": self.quality.to_dict(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> "CTv3Record":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            record_id=data.get("record_id", str(uuid4())),
            record_type=RecordType(data.get("record_type", "sft_turn")),
            source=SourceInfo.from_dict(data.get("source", {})),
            context=ContextInfo.from_dict(data.get("context", {})),
            input=InputData.from_dict(data.get("input", {})),
            target=TargetData.from_dict(data.get("target", {})),
            tags=TagInfo.from_dict(data.get("tags", {})),
            quality=QualityInfo.from_dict(data.get("quality", {})),
        )


# =============================================================================
# DPO PAIR RECORD
# =============================================================================

@dataclass
class DPOCandidates:
    """Preferred and dispreferred candidates for DPO."""
    
    preferred: TargetData = field(default_factory=TargetData)
    dispreferred: TargetData = field(default_factory=TargetData)
    
    def to_dict(self) -> dict:
        return {
            "preferred": self.preferred.to_dict(),
            "dispreferred": self.dispreferred.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DPOCandidates":
        return cls(
            preferred=TargetData.from_dict(data.get("preferred", {})),
            dispreferred=TargetData.from_dict(data.get("dispreferred", {})),
        )


@dataclass
class CTv3DPORecord:
    """Complete CTv3.1 DPO pair record."""
    
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
            "record_type": self.record_type.value if isinstance(self.record_type, Enum) else self.record_type,
            "source": self.source.to_dict(),
            "context": self.context.to_dict(),
            "input": self.input.to_dict(),
            "candidates": self.candidates.to_dict(),
            "tags": self.tags.to_dict(),
            "quality": self.quality.to_dict(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> "CTv3DPORecord":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            record_id=data.get("record_id", str(uuid4())),
            record_type=RecordType(data.get("record_type", "dpo_pair")),
            source=SourceInfo.from_dict(data.get("source", {})),
            context=ContextInfo.from_dict(data.get("context", {})),
            input=InputData.from_dict(data.get("input", {})),
            candidates=DPOCandidates.from_dict(data.get("candidates", {})),
            tags=TagInfo.from_dict(data.get("tags", {})),
            quality=QualityInfo.from_dict(data.get("quality", {})),
        )


# =============================================================================
# EVAL CASE RECORD
# =============================================================================

@dataclass
class EvalChecks:
    """Checks for evaluation cases."""
    
    expected_behaviors: list = field(default_factory=list)
    disallowed_behaviors: list = field(default_factory=list)
    disallowed_phrases: list = field(default_factory=list)
    must_not_end_with_question: bool = True
    must_follow_format: str = ""
    
    def to_dict(self) -> dict:
        return {
            "expected_behaviors": self.expected_behaviors,
            "disallowed_behaviors": self.disallowed_behaviors,
            "disallowed_phrases": self.disallowed_phrases,
            "must_not_end_with_question": self.must_not_end_with_question,
            "must_follow_format": self.must_follow_format,
        }


@dataclass
class CTv3EvalRecord:
    """Complete CTv3.1 eval case record."""
    
    schema_version: str = SCHEMA_VERSION
    record_id: str = field(default_factory=lambda: str(uuid4()))
    record_type: RecordType = RecordType.EVAL_CASE
    
    source: SourceInfo = field(default_factory=SourceInfo)
    context: ContextInfo = field(default_factory=ContextInfo)
    input: InputData = field(default_factory=InputData)
    checks: EvalChecks = field(default_factory=EvalChecks)
    reference_answer: str = ""
    quality: QualityInfo = field(default_factory=QualityInfo)
    
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "record_type": self.record_type.value if isinstance(self.record_type, Enum) else self.record_type,
            "source": self.source.to_dict(),
            "context": self.context.to_dict(),
            "input": self.input.to_dict(),
            "checks": self.checks.to_dict(),
            "reference": {"answer": self.reference_answer},
            "quality": self.quality.to_dict(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


