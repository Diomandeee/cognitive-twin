"""
Unified data types for the V3 ingestion pipeline.

These types represent the canonical schema that all data sources
are normalized to before V3 processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path


class SourceProvider(str, Enum):
    """Data source providers."""
    CLAUDE = "claude"
    CODEX = "codex"
    CURSOR = "cursor"
    OPENAI = "openai"
    PROMPT_LOGGER = "prompt_logger"
    SUPABASE = "supabase"
    UNKNOWN = "unknown"


class TurnRole(str, Enum):
    """Conversation turn roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool/function call in a conversation."""
    tool_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class GitContext:
    """Git repository context at time of conversation."""
    repo_name: Optional[str] = None
    repo_root: Optional[str] = None
    branch: Optional[str] = None
    remote: Optional[str] = None
    commit_sha: Optional[str] = None
    commit_message: Optional[str] = None
    is_dirty: bool = False
    uncommitted_files: List[str] = field(default_factory=list)
    staged_files: List[str] = field(default_factory=list)


@dataclass
class TurnMetadata:
    """Metadata associated with a conversation turn."""
    cwd: Optional[str] = None
    git_context: Optional[GitContext] = None
    file_diffs: int = 0
    affected_files: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedTurn:
    """
    A single turn in a conversation, normalized to unified schema.
    
    This is the atomic unit of conversation data that flows through
    the V3 pipeline.
    """
    role: TurnRole
    content: str
    
    # Source tracking
    source: SourceProvider
    original_id: Optional[str] = None
    
    # Timing
    timestamp: Optional[datetime] = None
    
    # Tool interactions
    tool_calls: List[ToolCall] = field(default_factory=list)
    
    # Rich metadata
    metadata: TurnMetadata = field(default_factory=TurnMetadata)
    
    def __post_init__(self):
        if isinstance(self.role, str):
            self.role = TurnRole(self.role)
        if isinstance(self.source, str):
            self.source = SourceProvider(self.source)
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
    
    @property
    def content_length(self) -> int:
        return len(self.content) if self.content else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "source": self.source.value,
            "original_id": self.original_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tool_calls": [
                {
                    "tool_id": tc.tool_id,
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                }
                for tc in self.tool_calls
            ],
            "metadata": {
                "cwd": self.metadata.cwd,
                "file_diffs": self.metadata.file_diffs,
                "affected_files": self.metadata.affected_files,
                "model_used": self.metadata.model_used,
            },
        }


@dataclass
class UnifiedConversation:
    """
    A complete conversation normalized to unified schema.
    
    Contains all turns plus conversation-level metadata.
    """
    conversation_id: str
    source_provider: SourceProvider
    turns: List[UnifiedTurn]
    
    # Context
    project_path: Optional[str] = None
    project_context: Optional[str] = None
    plan_references: List[str] = field(default_factory=list)
    
    # Timing
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Session tracking
    session_id: Optional[str] = None
    parent_conversation_id: Optional[str] = None
    
    # Quality indicators
    is_complete: bool = True
    has_errors: bool = False
    error_message: Optional[str] = None
    
    # Raw source data (for debugging)
    raw_source: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if isinstance(self.source_provider, str):
            self.source_provider = SourceProvider(self.source_provider)
    
    @property
    def turn_count(self) -> int:
        return len(self.turns)
    
    @property
    def user_turns(self) -> List[UnifiedTurn]:
        return [t for t in self.turns if t.role == TurnRole.USER]
    
    @property
    def assistant_turns(self) -> List[UnifiedTurn]:
        return [t for t in self.turns if t.role == TurnRole.ASSISTANT]
    
    @property
    def total_content_length(self) -> int:
        return sum(t.content_length for t in self.turns)
    
    @property
    def content_hash(self) -> str:
        """Generate a hash of conversation content for deduplication."""
        import hashlib
        content = "".join(f"{t.role.value}:{t.content}" for t in self.turns)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "source_provider": self.source_provider.value,
            "turns": [t.to_dict() for t in self.turns],
            "project_path": self.project_path,
            "project_context": self.project_context,
            "plan_references": self.plan_references,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "content_hash": self.content_hash,
        }
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to simple messages format for training."""
        return [
            {"role": t.role.value, "content": t.content}
            for t in self.turns
        ]


@dataclass
class ExtractorConfig:
    """Configuration for data extractors."""
    # Paths
    base_path: Optional[Path] = None
    
    # Filtering
    min_turns: int = 2
    max_turns: int = 100
    min_content_length: int = 10
    
    # Source filtering
    include_sources: Optional[List[SourceProvider]] = None
    exclude_sources: Optional[List[SourceProvider]] = None
    
    # Time filtering
    after_date: Optional[datetime] = None
    before_date: Optional[datetime] = None
    
    # Processing
    include_system_turns: bool = False
    include_tool_results: bool = True
    max_conversations: Optional[int] = None
    
    # Debugging
    verbose: bool = False


@dataclass
class ExtractionResult:
    """Result of a data extraction operation."""
    conversations: List[UnifiedConversation]
    source: SourceProvider
    
    # Statistics
    total_found: int = 0
    total_extracted: int = 0
    total_skipped: int = 0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_found == 0:
            return 0.0
        return self.total_extracted / self.total_found
    
    @property
    def duration_seconds(self) -> float:
        if not self.started_at or not self.completed_at:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class NormalizationResult:
    """Result of schema normalization."""
    conversations: List[UnifiedConversation]
    
    # Statistics
    total_input: int = 0
    total_normalized: int = 0
    total_invalid: int = 0
    
    # Field coverage
    field_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Errors
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class DeduplicationResult:
    """Result of deduplication."""
    conversations: List[UnifiedConversation]
    
    # Statistics
    total_input: int = 0
    total_unique: int = 0
    total_duplicates: int = 0
    
    # Duplicate clusters
    duplicate_groups: List[List[str]] = field(default_factory=list)
    
    # Methods used
    exact_duplicates: int = 0
    semantic_duplicates: int = 0
    session_overlaps: int = 0

