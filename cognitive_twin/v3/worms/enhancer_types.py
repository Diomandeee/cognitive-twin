"""
Enhancer Agent Types for CognitiveTwin V3.

Defines dataclasses for:
- Configuration
- Evaluation cases
- Annoyance records
- Enhanced records
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnhancerConfig:
    """Configuration for the Enhancer Agent."""
    
    # Feature toggles
    remove_provider_isms: bool = True
    remove_filler_openings: bool = True
    remove_permission_closers: bool = True
    complete_code: bool = True
    complete_prose: bool = True
    extract_eval_cases: bool = True
    
    # Processing limits
    max_concurrent_conversations: int = 5
    max_completion_retries: int = 2
    
    # Apology handling
    max_apologies_to_keep: int = 1
    
    # Disclaimer handling
    keep_sensitive_disclaimers: bool = True
    sensitive_topics: list = field(default_factory=lambda: [
        "legal", "medical", "financial", "health", "investment",
        "tax", "security", "safety"
    ])
    
    # Code completion
    code_completion_model: str = "gpt-5.2"
    code_completion_temperature: float = 0.2
    
    # Prose completion
    prose_completion_model: str = "gpt-5.2"
    prose_completion_temperature: float = 0.3
    
    # Reference answer generation
    reference_answer_model: str = "gpt-5.2"
    reference_answer_temperature: float = 0.3


# =============================================================================
# EVAL CASE
# =============================================================================

@dataclass
class EvalCase:
    """Evaluation case for regression testing."""
    
    record_type: str = "eval_case"
    case_id: str = ""
    case_type: str = ""  # permission_seeking, omission, format_drift
    
    # Input
    prompt: str = ""
    context: list = field(default_factory=list)
    format_constraints: dict = field(default_factory=dict)
    
    # Expected behavior
    expected_behaviors: list = field(default_factory=list)
    disallowed_behaviors: list = field(default_factory=list)
    disallowed_phrases: list = field(default_factory=list)
    
    # Validation rules
    must_not_end_with_question: bool = True
    must_contain_artifact: bool = False
    must_follow_format: str = ""
    
    # Reference answer (optional)
    reference_answer: str = ""
    
    # Source
    source_conversation: str = ""
    source_turn: int = 0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        if not self.case_id:
            self.case_id = f"eval_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> dict:
        return {
            "record_type": self.record_type,
            "case_id": self.case_id,
            "case_type": self.case_type,
            "prompt": self.prompt,
            "context": self.context,
            "format_constraints": self.format_constraints,
            "expected_behaviors": self.expected_behaviors,
            "disallowed_behaviors": self.disallowed_behaviors,
            "disallowed_phrases": self.disallowed_phrases,
            "must_not_end_with_question": self.must_not_end_with_question,
            "must_contain_artifact": self.must_contain_artifact,
            "must_follow_format": self.must_follow_format,
            "reference_answer": self.reference_answer,
            "source_conversation": self.source_conversation,
            "source_turn": self.source_turn,
            "created_at": self.created_at,
        }


# =============================================================================
# ANNOYANCE RECORD
# =============================================================================

@dataclass
class AnnoyanceRecord:
    """Record of an annoyance (policy violation) in conversation."""
    
    type: str  # permission_seeking, omission, format_drift
    conversation_id: str
    turn_index: int
    user_message: str
    assistant_message: str
    
    # Additional context
    classification: Optional[dict] = None
    omission_indicators: list = field(default_factory=list)
    format_constraints: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "classification": self.classification,
            "omission_indicators": self.omission_indicators,
            "format_constraints": self.format_constraints,
            "violations": self.violations,
        }


# =============================================================================
# ENHANCED RECORD
# =============================================================================

@dataclass
class EnhancedRecord:
    """Record of an enhanced (canonicalized/completed) assistant response."""
    
    original: str
    enhanced: str
    enhancement_type: str  # canonicalization, completion, both
    
    # Source
    conversation_id: str = ""
    turn_index: int = 0
    
    # Changes made
    changes_made: list = field(default_factory=list)
    
    # Context (for SFT record creation)
    context: list = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "enhanced": self.enhanced,
            "enhancement_type": self.enhancement_type,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "changes_made": self.changes_made,
            "context": self.context,
            "created_at": self.created_at,
        }


# =============================================================================
# INCOMPLETE CONTENT RECORDS
# =============================================================================

@dataclass
class IncompleteCodeMarker:
    """Record of incomplete code in a response."""
    
    block_index: int
    marker: str
    position: int
    context: str
    language: str = "python"


@dataclass
class PlaceholderMarker:
    """Record of a placeholder in content."""
    
    placeholder: str
    position: int
    context: str


@dataclass
class UndeterminedPath:
    """Record of an undetermined path indicator."""
    
    indicator: str
    position: int
    context: str


# =============================================================================
# STATISTICS
# =============================================================================

@dataclass
class EnhancerStats:
    """Statistics from Enhancer Agent execution."""
    
    conversations_processed: int = 0
    turns_processed: int = 0
    
    # Canonicalization
    provider_isms_removed: int = 0
    filler_openings_removed: int = 0
    permission_closers_removed: int = 0
    apologies_reduced: int = 0
    
    # Completion
    code_blocks_completed: int = 0
    prose_sections_completed: int = 0
    
    # Annoyances found
    permission_annoyances: int = 0
    omission_annoyances: int = 0
    format_drift_annoyances: int = 0
    
    # Output counts
    enhanced_records: int = 0
    eval_cases: int = 0
    dpo_pairs: int = 0
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    
    @property
    def total_annoyances(self) -> int:
        return (
            self.permission_annoyances +
            self.omission_annoyances +
            self.format_drift_annoyances
        )
    
    def to_dict(self) -> dict:
        return {
            "conversations_processed": self.conversations_processed,
            "turns_processed": self.turns_processed,
            "canonicalization": {
                "provider_isms_removed": self.provider_isms_removed,
                "filler_openings_removed": self.filler_openings_removed,
                "permission_closers_removed": self.permission_closers_removed,
                "apologies_reduced": self.apologies_reduced,
            },
            "completion": {
                "code_blocks_completed": self.code_blocks_completed,
                "prose_sections_completed": self.prose_sections_completed,
            },
            "annoyances": {
                "permission_seeking": self.permission_annoyances,
                "omission": self.omission_annoyances,
                "format_drift": self.format_drift_annoyances,
                "total": self.total_annoyances,
            },
            "outputs": {
                "enhanced_records": self.enhanced_records,
                "eval_cases": self.eval_cases,
                "dpo_pairs": self.dpo_pairs,
            },
            "timing": {
                "start": self.start_time,
                "end": self.end_time,
                "duration_seconds": self.duration_seconds,
            },
        }

