"""
Branch Types for Conversation Worm.

Defines dataclasses for:
- Configuration
- Synthetic branches
- 5D coordinates
- Processing results
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


# =============================================================================
# 5D COORDINATES
# =============================================================================

@dataclass
class DLMCoordinate:
    """5D coordinate in the Dialogue Latent Manifold.
    
    Dimensions:
        x (depth): Conversation depth (turn count from root)
        y (sibling_order): Position among siblings (regenerations)
        z (homogeneity): Semantic similarity to parent
        t (temporal): Normalized temporal position in session
        n (complexity): Content complexity score
    """
    
    x: float = 0.0  # depth
    y: float = 0.0  # sibling_order
    z: float = 0.0  # homogeneity
    t: float = 0.0  # temporal
    n: float = 0.0  # complexity
    
    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.t, self.n]
    
    def to_dict(self) -> dict:
        return {
            "depth": self.x,
            "sibling_order": self.y,
            "homogeneity": self.z,
            "temporal": self.t,
            "complexity": self.n,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DLMCoordinate":
        return cls(
            x=data.get("depth", data.get("x", 0.0)),
            y=data.get("sibling_order", data.get("y", 0.0)),
            z=data.get("homogeneity", data.get("z", 0.0)),
            t=data.get("temporal", data.get("t", 0.0)),
            n=data.get("complexity", data.get("n", 0.0)),
        )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ConversationWormConfig:
    """Configuration for Conversation Worm."""
    
    # Phase-question policy mapping
    phase_question_policies: dict = field(default_factory=lambda: {
        0: "questions_if_required",  # Opening
        1: "questions_if_required",  # Context
        2: "no_questions",           # Solution
        3: "no_questions",           # Refinement
        4: "no_questions",           # Synthesis
        5: "no_questions",           # Conclusion
    })
    
    # Branch generation
    max_branches_per_friction: int = 3
    min_quality_threshold: float = 0.6
    paraphrase_count: int = 2
    extension_max_turns: int = 3
    
    # Validation
    require_no_questions_above_phase: int = 2
    
    # Processing
    max_concurrent_conversations: int = 5
    
    # API settings
    model: str = "gpt-5.2"
    temperature_paraphrase: float = 0.7
    temperature_ideal: float = 0.3
    temperature_extension: float = 0.5
    max_tokens: int = 4096


# =============================================================================
# PATH NODE (Minimal TPO compatibility)
# =============================================================================

@dataclass
class PathNode:
    """A node in a conversation path."""
    
    turn_id: str = ""
    role: str = ""  # "user" or "assistant"
    content: str = ""
    
    # Position
    depth: int = 0
    sibling_order: int = 0
    
    # Coordinates
    homogeneity: float = 0.0
    temporal: float = 0.0
    complexity: float = 0.0
    
    # Metadata
    phase_id: int = 2
    created_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "depth": self.depth,
            "sibling_order": self.sibling_order,
            "homogeneity": self.homogeneity,
            "temporal": self.temporal,
            "complexity": self.complexity,
            "phase_id": self.phase_id,
        }


@dataclass
class ConversationPath:
    """A path through a conversation DAG."""
    
    conversation_id: str = ""
    nodes: list[PathNode] = field(default_factory=list)
    quality_score: float = 0.0
    has_friction: bool = False
    
    def to_messages(self) -> list[dict]:
        """Convert to list of message dicts."""
        return [
            {"role": node.role, "content": node.content}
            for node in self.nodes
        ]


# =============================================================================
# SYNTHETIC BRANCH
# =============================================================================

@dataclass
class SyntheticBranch:
    """A synthetic branch generated from conversation."""
    
    branch_type: str  # paraphrase, ideal_response, extension, contrast
    original_conversation_id: str
    parent_node_id: str
    
    # Content
    messages: list[dict] = field(default_factory=list)
    
    # Coordinates
    coordinates: DLMCoordinate = field(default_factory=DLMCoordinate)
    
    # Labels
    phase_id: int = 2
    question_policy: str = "no_questions"
    directive_completeness: float = 0.8
    
    # Quality
    source: str = "convo_worm"
    is_gold: bool = False
    
    # Metadata
    branch_id: str = field(default_factory=lambda: f"branch_{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "branch_type": self.branch_type,
            "original_conversation_id": self.original_conversation_id,
            "parent_node_id": self.parent_node_id,
            "messages": self.messages,
            "coordinates": self.coordinates.to_dict(),
            "phase_id": self.phase_id,
            "question_policy": self.question_policy,
            "directive_completeness": self.directive_completeness,
            "source": self.source,
            "is_gold": self.is_gold,
            "created_at": self.created_at,
        }


# =============================================================================
# PROCESSING RESULTS
# =============================================================================

@dataclass
class BranchResult:
    """Result of processing a conversation for branches."""
    
    conversation_id: str
    
    # Generated branches
    paraphrase_branches: list[SyntheticBranch] = field(default_factory=list)
    ideal_branches: list[SyntheticBranch] = field(default_factory=list)
    extension_branches: list[SyntheticBranch] = field(default_factory=list)
    contrast_branches: list[SyntheticBranch] = field(default_factory=list)
    
    # Stats
    friction_points_found: int = 0
    total_branches: int = 0
    
    # Errors
    success: bool = True
    error: Optional[str] = None
    
    @property
    def all_branches(self) -> list[SyntheticBranch]:
        return (
            self.paraphrase_branches +
            self.ideal_branches +
            self.extension_branches +
            self.contrast_branches
        )
    
    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "friction_points_found": self.friction_points_found,
            "total_branches": self.total_branches,
            "branches_by_type": {
                "paraphrase": len(self.paraphrase_branches),
                "ideal_response": len(self.ideal_branches),
                "extension": len(self.extension_branches),
                "contrast": len(self.contrast_branches),
            },
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ConversationWormStats:
    """Statistics from Conversation Worm execution."""
    
    conversations_processed: int = 0
    friction_points_found: int = 0
    
    # Branch counts
    paraphrase_branches: int = 0
    ideal_branches: int = 0
    extension_branches: int = 0
    contrast_branches: int = 0
    
    # Output counts
    sft_records: int = 0
    dpo_pairs: int = 0
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    
    @property
    def total_branches(self) -> int:
        return (
            self.paraphrase_branches +
            self.ideal_branches +
            self.extension_branches +
            self.contrast_branches
        )
    
    def to_dict(self) -> dict:
        return {
            "conversations_processed": self.conversations_processed,
            "friction_points_found": self.friction_points_found,
            "branches": {
                "paraphrase": self.paraphrase_branches,
                "ideal_response": self.ideal_branches,
                "extension": self.extension_branches,
                "contrast": self.contrast_branches,
                "total": self.total_branches,
            },
            "outputs": {
                "sft_records": self.sft_records,
                "dpo_pairs": self.dpo_pairs,
            },
            "timing": {
                "start": self.start_time,
                "end": self.end_time,
                "duration_seconds": self.duration_seconds,
            },
        }

