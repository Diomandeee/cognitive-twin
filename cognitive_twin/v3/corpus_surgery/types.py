"""
Type definitions for Corpus Surgery module.

Defines dataclasses and enums used across the classifier, rewriter, and quarantine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set
from datetime import datetime


class ClarificationType(str, Enum):
    """Classification of assistant clarification behavior."""
    
    UNJUSTIFIED = "unjustified"  # Permission-seeking when not needed
    JUSTIFIED = "justified"      # Clarification genuinely required
    NEUTRAL = "neutral"          # Neither permission-seeking nor requiring clarification


class QuestionPolicy(str, Enum):
    """Policy governing whether assistant may ask questions."""
    
    NO_QUESTIONS = "no_questions"
    QUESTIONS_IF_REQUIRED = "questions_if_required"
    QUESTIONS_ALLOWED = "questions_allowed"


@dataclass
class FormatConstraints:
    """Format constraints extracted from user message."""
    
    forbid_bullets: bool = False
    require_numbered: bool = False
    must_return_code: bool = False
    must_return_diff: bool = False
    must_return_json: bool = False
    must_not_omit: bool = False
    
    def to_dict(self) -> dict:
        return {
            "forbid_bullets": self.forbid_bullets,
            "require_numbered": self.require_numbered,
            "must_return_code": self.must_return_code,
            "must_return_diff": self.must_return_diff,
            "must_return_json": self.must_return_json,
            "must_not_omit": self.must_not_omit,
        }


@dataclass
class ParsabilityInfo:
    """FunctionGemma parsability information."""
    
    score: float = 0.0
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    required_params: Set[str] = field(default_factory=set)
    provided_params: Set[str] = field(default_factory=set)
    missing_params: Set[str] = field(default_factory=set)
    parse_success: bool = False
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "required_params": list(self.required_params),
            "provided_params": list(self.provided_params),
            "missing_params": list(self.missing_params),
            "parse_success": self.parse_success,
            "confidence": self.confidence,
        }
    
    @property
    def is_complete(self) -> bool:
        """Check if all required parameters were provided."""
        return self.parse_success and self.score >= 1.0


@dataclass
class ClassificationResult:
    """Result of classifying an assistant turn."""
    
    classification: ClarificationType
    stall_score: int
    exec_score: int
    blocked_score: int
    directive_completeness: float
    reasoning: str
    
    # Optional metadata
    question_policy: QuestionPolicy = QuestionPolicy.NO_QUESTIONS
    format_constraints: FormatConstraints = field(default_factory=FormatConstraints)
    
    # FunctionGemma parsability (NEW)
    parsability_score: float = 0.0
    parsability_info: Optional[ParsabilityInfo] = None
    fused_completeness: float = 0.0  # Combined heuristic + parsability
    
    def to_dict(self) -> dict:
        return {
            "classification": self.classification.value,
            "stall_score": self.stall_score,
            "exec_score": self.exec_score,
            "blocked_score": self.blocked_score,
            "directive_completeness": self.directive_completeness,
            "parsability_score": self.parsability_score,
            "fused_completeness": self.fused_completeness,
            "reasoning": self.reasoning,
            "question_policy": self.question_policy.value,
            "format_constraints": self.format_constraints.to_dict(),
            "parsability_info": self.parsability_info.to_dict() if self.parsability_info else None,
        }


@dataclass
class ValidationResult:
    """Result of validating a rewritten response."""
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
        }


@dataclass
class QuarantineMarker:
    """Marks a conversation segment as containing friction."""
    
    conversation_id: str
    start_turn_idx: int
    end_turn_idx: int
    trigger_phrase: str
    bad_assistant_turn: str
    user_correction: str
    is_friction: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "start_turn_idx": self.start_turn_idx,
            "end_turn_idx": self.end_turn_idx,
            "trigger_phrase": self.trigger_phrase,
            "bad_assistant_turn": self.bad_assistant_turn,
            "user_correction": self.user_correction,
            "is_friction": self.is_friction,
            "created_at": self.created_at,
        }


@dataclass
class DPOPair:
    """A preference pair for DPO training."""
    
    prompt: str
    preferred: str
    dispreferred: str
    confidence: float = 0.9
    source: str = "corpus_surgery"
    
    # Optional metadata
    conversation_id: Optional[str] = None
    turn_idx: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.preferred,
            "rejected": self.dispreferred,
            "confidence": self.confidence,
            "source": self.source,
            "conversation_id": self.conversation_id,
            "turn_idx": self.turn_idx,
        }
    
    def to_training_format(self) -> dict:
        """Export in Together AI DPO format."""
        return {
            "prompt": self.prompt,
            "chosen": self.preferred,
            "rejected": self.dispreferred,
        }


@dataclass
class EvalCase:
    """Evaluation case for regression testing."""
    
    case_id: str
    case_type: str  # permission_seeking, omission, format_drift
    prompt: str
    
    # Expected behavior
    expected_behaviors: list[str] = field(default_factory=list)
    disallowed_behaviors: list[str] = field(default_factory=list)
    disallowed_phrases: list[str] = field(default_factory=list)
    
    # Validation rules
    must_not_end_with_question: bool = True
    must_contain_artifact: bool = False
    must_follow_format: str = ""
    
    # Reference answer (optional)
    reference_answer: Optional[str] = None
    
    # Source
    source_conversation: str = ""
    source_turn: int = 0
    
    # Context
    context: list[dict] = field(default_factory=list)
    format_constraints: FormatConstraints = field(default_factory=FormatConstraints)
    
    def to_dict(self) -> dict:
        return {
            "record_type": "eval_case",
            "case_id": self.case_id,
            "case_type": self.case_type,
            "prompt": self.prompt,
            "context": self.context,
            "checks": {
                "expected_behaviors": self.expected_behaviors,
                "disallowed_behaviors": self.disallowed_behaviors,
                "disallowed_phrases": self.disallowed_phrases,
                "must_not_end_with_question": self.must_not_end_with_question,
                "must_contain_artifact": self.must_contain_artifact,
                "must_follow_format": self.must_follow_format,
            },
            "reference": {
                "answer": self.reference_answer,
            },
            "source": {
                "conversation": self.source_conversation,
                "turn": self.source_turn,
            },
            "format_constraints": self.format_constraints.to_dict(),
        }


@dataclass
class ProcessedTurn:
    """Result of processing a single turn."""
    
    original_content: str
    processed_content: str
    classification: ClassificationResult
    action_taken: str  # "kept", "rewritten", "quarantined"
    
    # For rewrites
    rewrite_valid: bool = True
    rewrite_errors: list[str] = field(default_factory=list)
    
    # For quarantine
    quarantine_marker: Optional[QuarantineMarker] = None
    dpo_pair: Optional[DPOPair] = None
    eval_case: Optional[EvalCase] = None


@dataclass
class ProcessedConversation:
    """Result of processing a full conversation."""
    
    conversation_id: str
    original_turns: int
    processed_turns: list[ProcessedTurn]
    
    # Stats
    kept_count: int = 0
    rewritten_count: int = 0
    quarantined_count: int = 0
    
    # Outputs
    sft_turns: list[dict] = field(default_factory=list)
    dpo_pairs: list[DPOPair] = field(default_factory=list)
    eval_cases: list[EvalCase] = field(default_factory=list)

