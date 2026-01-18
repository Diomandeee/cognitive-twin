"""
Content Deduplicator

Removes duplicate conversations using multiple strategies:
- Exact content hash matching
- Semantic similarity (optional, requires embeddings)
- Session overlap detection
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .types import (
    DeduplicationResult,
    UnifiedConversation,
)

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    # Exact matching
    use_exact_hash: bool = True
    hash_first_n_turns: Optional[int] = None  # None = all turns
    
    # Semantic matching (requires embeddings)
    use_semantic_similarity: bool = False
    semantic_threshold: float = 0.95
    embedding_model: str = "text-embedding-3-small"
    
    # Session overlap
    use_session_overlap: bool = True
    session_overlap_threshold: float = 0.8
    
    # Preference for duplicates
    prefer_longer: bool = True
    prefer_newer: bool = True
    prefer_source_order: List[str] = None  # e.g., ["claude", "cursor", "openai"]


class ContentDeduplicator:
    """
    Remove duplicate conversations using multiple strategies.
    
    Strategies:
    1. Exact hash: Content hash of concatenated turns
    2. Semantic: Embedding cosine similarity (optional)
    3. Session overlap: Detect overlapping sessions
    """
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self._embeddings_cache: Dict[str, List[float]] = {}
        
    def deduplicate(
        self, 
        conversations: List[UnifiedConversation]
    ) -> DeduplicationResult:
        """
        Remove duplicates from conversation list.
        
        Args:
            conversations: List of conversations to deduplicate
            
        Returns:
            DeduplicationResult with unique conversations
        """
        total_input = len(conversations)
        duplicate_groups: List[List[str]] = []
        
        # Track duplicates
        exact_dupes = 0
        semantic_dupes = 0
        session_dupes = 0
        
        # Start with all conversations
        remaining = list(conversations)
        
        # Phase 1: Exact hash deduplication
        if self.config.use_exact_hash:
            remaining, exact_groups = self._dedupe_by_hash(remaining)
            duplicate_groups.extend(exact_groups)
            exact_dupes = sum(len(g) - 1 for g in exact_groups)
            logger.info(f"Exact hash dedup: {exact_dupes} duplicates removed")
        
        # Phase 2: Session overlap detection
        if self.config.use_session_overlap:
            remaining, session_groups = self._dedupe_by_session_overlap(remaining)
            duplicate_groups.extend(session_groups)
            session_dupes = sum(len(g) - 1 for g in session_groups)
            logger.info(f"Session overlap dedup: {session_dupes} duplicates removed")
        
        # Phase 3: Semantic similarity (expensive, optional)
        if self.config.use_semantic_similarity:
            try:
                remaining, semantic_groups = self._dedupe_by_semantic(remaining)
                duplicate_groups.extend(semantic_groups)
                semantic_dupes = sum(len(g) - 1 for g in semantic_groups)
                logger.info(f"Semantic dedup: {semantic_dupes} duplicates removed")
            except Exception as e:
                logger.warning(f"Semantic deduplication failed: {e}")
        
        total_duplicates = exact_dupes + semantic_dupes + session_dupes
        
        return DeduplicationResult(
            conversations=remaining,
            total_input=total_input,
            total_unique=len(remaining),
            total_duplicates=total_duplicates,
            duplicate_groups=duplicate_groups,
            exact_duplicates=exact_dupes,
            semantic_duplicates=semantic_dupes,
            session_overlaps=session_dupes,
        )
    
    def _dedupe_by_hash(
        self, 
        conversations: List[UnifiedConversation]
    ) -> Tuple[List[UnifiedConversation], List[List[str]]]:
        """Remove exact duplicates by content hash."""
        hash_to_convos: Dict[str, List[UnifiedConversation]] = defaultdict(list)
        
        for conv in conversations:
            content_hash = self._compute_hash(conv)
            hash_to_convos[content_hash].append(conv)
        
        unique: List[UnifiedConversation] = []
        groups: List[List[str]] = []
        
        for hash_val, convos in hash_to_convos.items():
            # Pick the best representative
            best = self._pick_best(convos)
            unique.append(best)
            
            if len(convos) > 1:
                groups.append([c.conversation_id for c in convos])
        
        return unique, groups
    
    def _dedupe_by_session_overlap(
        self, 
        conversations: List[UnifiedConversation]
    ) -> Tuple[List[UnifiedConversation], List[List[str]]]:
        """Remove conversations with high session overlap."""
        # Group by session ID
        session_to_convos: Dict[str, List[UnifiedConversation]] = defaultdict(list)
        no_session: List[UnifiedConversation] = []
        
        for conv in conversations:
            if conv.session_id:
                session_to_convos[conv.session_id].append(conv)
            else:
                no_session.append(conv)
        
        unique: List[UnifiedConversation] = list(no_session)
        groups: List[List[str]] = []
        
        for session_id, convos in session_to_convos.items():
            if len(convos) == 1:
                unique.append(convos[0])
            else:
                # Check for overlaps within session
                merged = self._merge_session_overlaps(convos)
                unique.extend(merged)
                
                if len(convos) > len(merged):
                    groups.append([c.conversation_id for c in convos])
        
        return unique, groups
    
    def _dedupe_by_semantic(
        self, 
        conversations: List[UnifiedConversation]
    ) -> Tuple[List[UnifiedConversation], List[List[str]]]:
        """Remove semantically similar conversations."""
        # This would require embeddings - simplified implementation
        # Full implementation would use an embedding model
        
        logger.warning("Semantic deduplication requires embeddings - skipping")
        return conversations, []
    
    def _compute_hash(self, conv: UnifiedConversation) -> str:
        """Compute content hash for a conversation."""
        turns = conv.turns
        
        if self.config.hash_first_n_turns:
            turns = turns[:self.config.hash_first_n_turns]
        
        # Build content string
        content_parts = []
        for turn in turns:
            # Normalize content for hashing
            normalized = turn.content.strip().lower()
            content_parts.append(f"{turn.role.value}:{normalized}")
        
        content = "\n".join(content_parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _merge_session_overlaps(
        self, 
        convos: List[UnifiedConversation]
    ) -> List[UnifiedConversation]:
        """Merge overlapping conversations within a session."""
        if len(convos) <= 1:
            return convos
        
        # Sort by creation time or length
        sorted_convos = sorted(
            convos,
            key=lambda c: (c.created_at or c.turn_count, c.turn_count),
            reverse=True,
        )
        
        # Check for content overlap
        merged: List[UnifiedConversation] = []
        used_content: Set[str] = set()
        
        for conv in sorted_convos:
            # Get content signature
            signature = self._get_content_signature(conv)
            
            # Check if this is a subset of already seen content
            is_subset = False
            for seen in used_content:
                overlap = self._compute_overlap(signature, seen)
                if overlap > self.config.session_overlap_threshold:
                    is_subset = True
                    break
            
            if not is_subset:
                merged.append(conv)
                used_content.add(signature)
        
        return merged
    
    def _get_content_signature(self, conv: UnifiedConversation) -> str:
        """Get a content signature for overlap detection."""
        # Use first and last few turns as signature
        turns = conv.turns
        if len(turns) <= 4:
            sig_turns = turns
        else:
            sig_turns = turns[:2] + turns[-2:]
        
        parts = [f"{t.role.value}:{t.content[:100]}" for t in sig_turns]
        return "||".join(parts)
    
    def _compute_overlap(self, sig1: str, sig2: str) -> float:
        """Compute overlap ratio between two signatures."""
        # Simple character-level Jaccard similarity
        set1 = set(sig1.split("||"))
        set2 = set(sig2.split("||"))
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _pick_best(
        self, 
        convos: List[UnifiedConversation]
    ) -> UnifiedConversation:
        """Pick the best representative from duplicates."""
        if len(convos) == 1:
            return convos[0]
        
        # Sort by preference criteria
        def sort_key(conv: UnifiedConversation):
            score = 0
            
            if self.config.prefer_longer:
                score += conv.turn_count * 10
                score += conv.total_content_length
            
            if self.config.prefer_newer and conv.created_at:
                score += conv.created_at.timestamp() / 1e9  # Normalize
            
            if self.config.prefer_source_order:
                try:
                    idx = self.config.prefer_source_order.index(conv.source_provider.value)
                    score -= idx * 1000  # Lower index = higher score
                except ValueError:
                    pass
            
            return score
        
        sorted_convos = sorted(convos, key=sort_key, reverse=True)
        return sorted_convos[0]


def deduplicate_conversations(
    conversations: List[UnifiedConversation],
    use_semantic: bool = False,
) -> DeduplicationResult:
    """
    Deduplicate conversations with minimal configuration.
    
    Args:
        conversations: List of conversations to deduplicate
        use_semantic: Whether to use semantic similarity (expensive)
        
    Returns:
        DeduplicationResult with unique conversations
    """
    config = DeduplicationConfig(use_semantic_similarity=use_semantic)
    deduplicator = ContentDeduplicator(config)
    return deduplicator.deduplicate(conversations)

