"""
Schema Normalizer

Normalizes conversations from all sources to a unified schema,
ensuring consistent structure for downstream V3 processing.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .types import (
    NormalizationResult,
    SourceProvider,
    TurnRole,
    UnifiedConversation,
    UnifiedTurn,
)

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for normalization."""
    # Content cleaning
    strip_whitespace: bool = True
    normalize_newlines: bool = True
    remove_empty_turns: bool = True
    merge_consecutive_turns: bool = False
    
    # Content limits
    max_turn_length: int = 100000
    min_turn_length: int = 1
    
    # Role normalization
    normalize_roles: bool = True
    
    # Timestamp handling
    infer_timestamps: bool = True
    
    # Validation
    require_alternating_roles: bool = False
    require_user_first: bool = False


class SchemaNormalizer:
    """
    Normalize conversations to unified schema.
    
    Handles:
    - Content cleaning and normalization
    - Role standardization
    - Timestamp inference
    - Structure validation
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        
    def normalize_all(
        self, 
        conversations: List[UnifiedConversation]
    ) -> NormalizationResult:
        """
        Normalize all conversations.
        
        Args:
            conversations: List of conversations to normalize
            
        Returns:
            NormalizationResult with normalized conversations
        """
        normalized: List[UnifiedConversation] = []
        validation_errors: List[str] = []
        field_counts: Dict[str, int] = {}
        total_fields = 0
        
        for conv in conversations:
            try:
                norm_conv = self.normalize_conversation(conv)
                if norm_conv and self._validate_conversation(norm_conv):
                    normalized.append(norm_conv)
                    self._count_fields(norm_conv, field_counts)
                    total_fields += 1
                else:
                    validation_errors.append(
                        f"Conversation {conv.conversation_id} failed validation"
                    )
            except Exception as e:
                validation_errors.append(
                    f"Failed to normalize {conv.conversation_id}: {e}"
                )
        
        # Calculate field coverage
        field_coverage = {}
        if total_fields > 0:
            for field, count in field_counts.items():
                field_coverage[field] = count / total_fields
        
        return NormalizationResult(
            conversations=normalized,
            total_input=len(conversations),
            total_normalized=len(normalized),
            total_invalid=len(conversations) - len(normalized),
            field_coverage=field_coverage,
            validation_errors=validation_errors,
        )
    
    def normalize_conversation(
        self, 
        conversation: UnifiedConversation
    ) -> Optional[UnifiedConversation]:
        """Normalize a single conversation."""
        # Normalize turns
        normalized_turns: List[UnifiedTurn] = []
        
        for turn in conversation.turns:
            norm_turn = self.normalize_turn(turn)
            if norm_turn:
                normalized_turns.append(norm_turn)
        
        if not normalized_turns:
            return None
        
        # Merge consecutive turns if configured
        if self.config.merge_consecutive_turns:
            normalized_turns = self._merge_consecutive(normalized_turns)
        
        # Infer timestamps if configured
        if self.config.infer_timestamps:
            self._infer_timestamps(normalized_turns, conversation.created_at)
        
        return UnifiedConversation(
            conversation_id=conversation.conversation_id,
            source_provider=conversation.source_provider,
            turns=normalized_turns,
            project_path=conversation.project_path,
            project_context=self._normalize_text(conversation.project_context),
            plan_references=conversation.plan_references,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            session_id=conversation.session_id,
            parent_conversation_id=conversation.parent_conversation_id,
            is_complete=True,
            has_errors=False,
        )
    
    def normalize_turn(self, turn: UnifiedTurn) -> Optional[UnifiedTurn]:
        """Normalize a single turn."""
        # Normalize content
        content = self._normalize_text(turn.content)
        
        # Check minimum length
        if not content or len(content) < self.config.min_turn_length:
            if self.config.remove_empty_turns:
                return None
            content = content or ""
        
        # Truncate if too long
        if len(content) > self.config.max_turn_length:
            content = content[:self.config.max_turn_length] + "..."
        
        # Normalize role
        role = turn.role
        if self.config.normalize_roles:
            role = self._normalize_role(turn.role)
        
        return UnifiedTurn(
            role=role,
            content=content,
            source=turn.source,
            original_id=turn.original_id,
            timestamp=turn.timestamp,
            tool_calls=turn.tool_calls,
            metadata=turn.metadata,
        )
    
    def _normalize_text(self, text: Optional[str]) -> str:
        """Normalize text content."""
        if not text:
            return ""
        
        result = text
        
        if self.config.strip_whitespace:
            result = result.strip()
        
        if self.config.normalize_newlines:
            # Normalize different newline styles
            result = result.replace("\r\n", "\n")
            result = result.replace("\r", "\n")
            # Collapse multiple blank lines
            result = re.sub(r"\n{3,}", "\n\n", result)
        
        return result
    
    def _normalize_role(self, role: TurnRole) -> TurnRole:
        """Normalize role to standard values."""
        # Tool responses are treated as assistant turns for training
        if role == TurnRole.TOOL:
            return TurnRole.ASSISTANT
        return role
    
    def _merge_consecutive(
        self, 
        turns: List[UnifiedTurn]
    ) -> List[UnifiedTurn]:
        """Merge consecutive turns from the same role."""
        if not turns:
            return []
        
        merged: List[UnifiedTurn] = []
        current: Optional[UnifiedTurn] = None
        
        for turn in turns:
            if current is None:
                current = turn
            elif current.role == turn.role:
                # Merge content
                merged_content = f"{current.content}\n\n{turn.content}"
                current = UnifiedTurn(
                    role=current.role,
                    content=merged_content,
                    source=current.source,
                    original_id=current.original_id,
                    timestamp=current.timestamp,
                    tool_calls=current.tool_calls + turn.tool_calls,
                    metadata=current.metadata,
                )
            else:
                merged.append(current)
                current = turn
        
        if current:
            merged.append(current)
        
        return merged
    
    def _infer_timestamps(
        self, 
        turns: List[UnifiedTurn],
        conversation_start: Optional[datetime],
    ):
        """Infer missing timestamps based on order."""
        if not turns:
            return
        
        # Find existing timestamps
        existing_times = [t.timestamp for t in turns if t.timestamp]
        
        if not existing_times and conversation_start:
            # Use conversation start as base
            base_time = conversation_start
        elif existing_times:
            base_time = min(existing_times)
        else:
            # No time info available
            return
        
        # Fill in missing timestamps
        last_time = base_time
        for i, turn in enumerate(turns):
            if turn.timestamp:
                last_time = turn.timestamp
            else:
                # Assume 30 seconds between turns
                from datetime import timedelta
                inferred_time = last_time + timedelta(seconds=30 * (i + 1))
                turn.timestamp = inferred_time
    
    def _validate_conversation(
        self, 
        conversation: UnifiedConversation
    ) -> bool:
        """Validate conversation structure."""
        if not conversation.turns:
            return False
        
        if self.config.require_user_first:
            if conversation.turns[0].role != TurnRole.USER:
                return False
        
        if self.config.require_alternating_roles:
            for i in range(1, len(conversation.turns)):
                if conversation.turns[i].role == conversation.turns[i-1].role:
                    return False
        
        return True
    
    def _count_fields(
        self, 
        conversation: UnifiedConversation,
        counts: Dict[str, int],
    ):
        """Count populated fields for coverage metrics."""
        fields = [
            ("project_path", conversation.project_path),
            ("project_context", conversation.project_context),
            ("created_at", conversation.created_at),
            ("session_id", conversation.session_id),
        ]
        
        for field_name, value in fields:
            if value:
                counts[field_name] = counts.get(field_name, 0) + 1
        
        # Check turn-level fields
        has_timestamps = any(t.timestamp for t in conversation.turns)
        has_tool_calls = any(t.tool_calls for t in conversation.turns)
        has_metadata = any(t.metadata.cwd for t in conversation.turns)
        
        if has_timestamps:
            counts["turn_timestamps"] = counts.get("turn_timestamps", 0) + 1
        if has_tool_calls:
            counts["turn_tool_calls"] = counts.get("turn_tool_calls", 0) + 1
        if has_metadata:
            counts["turn_metadata"] = counts.get("turn_metadata", 0) + 1


def normalize_conversations(
    conversations: List[UnifiedConversation],
    merge_consecutive: bool = False,
) -> NormalizationResult:
    """
    Normalize conversations with minimal configuration.
    
    Args:
        conversations: List of conversations to normalize
        merge_consecutive: Whether to merge consecutive same-role turns
        
    Returns:
        NormalizationResult with normalized conversations
    """
    config = NormalizationConfig(merge_consecutive_turns=merge_consecutive)
    normalizer = SchemaNormalizer(config)
    return normalizer.normalize_all(conversations)

