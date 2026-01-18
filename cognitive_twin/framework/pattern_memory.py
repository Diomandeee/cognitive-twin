"""Pattern Memory for CognitiveTwin.

This module provides persistent storage and retrieval of reasoning patterns
learned from user conversations. Supports:
    - Hierarchical pattern organization
    - Approximate nearest neighbor search
    - Temporal decay for pattern importance
    - Consolidation of similar patterns
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cognitive_twin.framework.config import PatternMemoryConfig, MemoryRetrievalMode
from cognitive_twin.framework.reasoning_encoder import ReasoningPattern, PatternType
from cognitive_twin._compat import TrajectoryCoordinate5D

logger = logging.getLogger(__name__)


@dataclass
class StoredPattern:
    """A pattern stored in memory.

    Attributes:
        id: Unique pattern identifier.
        embedding: Pattern embedding vector.
        pattern_type: Type of reasoning pattern.
        coordinate: Trajectory coordinate where pattern occurred.
        importance: Current importance score (decays over time).
        access_count: Number of times pattern was accessed.
        created_at: Creation timestamp.
        last_accessed: Last access timestamp.
        metadata: Additional pattern metadata.
        cluster_id: Hierarchical cluster assignment.
    """

    id: str
    embedding: Tensor
    pattern_type: PatternType
    coordinate: Optional[TrajectoryCoordinate5D]
    importance: float = 1.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    cluster_id: Optional[int] = None

    def decay(self, rate: float, current_time: float) -> None:
        """Apply temporal decay to importance.

        Args:
            rate: Decay rate per second.
            current_time: Current timestamp.
        """
        time_delta = current_time - self.last_accessed
        self.importance *= math.exp(-rate * time_delta)

    def access(self) -> None:
        """Record an access to this pattern."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Boost importance on access
        self.importance = min(1.0, self.importance * 1.1)


@dataclass
class PatternQuery:
    """Query for pattern retrieval.

    Attributes:
        embedding: Query embedding.
        pattern_types: Filter by pattern types (None = all).
        coordinate: Optional trajectory coordinate for spatial filtering.
        min_importance: Minimum importance threshold.
        max_results: Maximum number of results.
        temporal_window: Only patterns from this time window (seconds).
    """

    embedding: Tensor
    pattern_types: Optional[List[PatternType]] = None
    coordinate: Optional[TrajectoryCoordinate5D] = None
    min_importance: float = 0.1
    max_results: int = 10
    temporal_window: Optional[float] = None


class PatternIndex(nn.Module):
    """Approximate nearest neighbor index for patterns.

    Uses learned hash functions for fast retrieval.
    """

    def __init__(
        self,
        pattern_dim: int,
        num_hash_tables: int = 8,
        hash_dim: int = 16,
    ) -> None:
        """Initialize pattern index.

        Args:
            pattern_dim: Dimension of pattern embeddings.
            num_hash_tables: Number of hash tables for LSH.
            hash_dim: Dimension of hash codes.
        """
        super().__init__()
        self.pattern_dim = pattern_dim
        self.num_hash_tables = num_hash_tables
        self.hash_dim = hash_dim

        # Learnable hash functions
        self.hash_projections = nn.ModuleList([
            nn.Linear(pattern_dim, hash_dim)
            for _ in range(num_hash_tables)
        ])

        # Hash tables (pattern_id -> bucket)
        self.hash_tables: List[Dict[int, List[str]]] = [
            defaultdict(list) for _ in range(num_hash_tables)
        ]

    def compute_hash(self, embedding: Tensor, table_idx: int) -> int:
        """Compute hash code for an embedding.

        Args:
            embedding: Pattern embedding [pattern_dim].
            table_idx: Which hash table to use.

        Returns:
            Integer hash code.
        """
        with torch.no_grad():
            projection = self.hash_projections[table_idx](embedding)
            # Binary quantization
            binary = (projection > 0).int()
            # Convert to integer
            hash_code = sum(b.item() * (2 ** i) for i, b in enumerate(binary))
            return hash_code

    def insert(self, pattern_id: str, embedding: Tensor) -> None:
        """Insert a pattern into the index.

        Args:
            pattern_id: Unique pattern identifier.
            embedding: Pattern embedding.
        """
        for table_idx, table in enumerate(self.hash_tables):
            hash_code = self.compute_hash(embedding, table_idx)
            table[hash_code].append(pattern_id)

    def query(self, embedding: Tensor, max_candidates: int = 100) -> List[str]:
        """Query the index for candidate patterns.

        Args:
            embedding: Query embedding.
            max_candidates: Maximum candidates to return.

        Returns:
            List of candidate pattern IDs.
        """
        candidates = set()
        for table_idx, table in enumerate(self.hash_tables):
            hash_code = self.compute_hash(embedding, table_idx)
            candidates.update(table.get(hash_code, []))

            if len(candidates) >= max_candidates:
                break

        return list(candidates)[:max_candidates]

    def remove(self, pattern_id: str, embedding: Tensor) -> None:
        """Remove a pattern from the index.

        Args:
            pattern_id: Pattern to remove.
            embedding: Pattern's embedding (for hash lookup).
        """
        for table_idx, table in enumerate(self.hash_tables):
            hash_code = self.compute_hash(embedding, table_idx)
            if pattern_id in table.get(hash_code, []):
                table[hash_code].remove(pattern_id)


class HierarchicalClusterer(nn.Module):
    """Hierarchical clustering for pattern organization.

    Organizes patterns into a hierarchy for efficient browsing
    and consolidation.
    """

    def __init__(
        self,
        pattern_dim: int,
        num_levels: int = 3,
        branching_factor: int = 10,
    ) -> None:
        """Initialize hierarchical clusterer.

        Args:
            pattern_dim: Dimension of pattern embeddings.
            num_levels: Number of hierarchy levels.
            branching_factor: Children per node.
        """
        super().__init__()
        self.pattern_dim = pattern_dim
        self.num_levels = num_levels
        self.branching_factor = branching_factor

        # Cluster centroids at each level
        # Level 0 has branching_factor clusters
        # Level 1 has branching_factor^2, etc.
        self.centroids: List[Tensor] = []
        for level in range(num_levels):
            num_clusters = branching_factor ** (level + 1)
            centroid = nn.Parameter(torch.randn(num_clusters, pattern_dim) * 0.1)
            self.register_parameter(f"centroids_{level}", centroid)
            self.centroids.append(centroid)

    def assign_cluster(self, embedding: Tensor) -> Tuple[int, List[int]]:
        """Assign embedding to cluster hierarchy.

        Args:
            embedding: Pattern embedding [pattern_dim].

        Returns:
            Tuple of (leaf_cluster_id, path_through_hierarchy).
        """
        path = []
        current_idx = 0

        for level, centroids in enumerate(self.centroids):
            # Get relevant centroids at this level
            start_idx = current_idx * self.branching_factor
            end_idx = start_idx + self.branching_factor

            if end_idx > len(centroids):
                break

            level_centroids = centroids[start_idx:end_idx]

            # Find nearest centroid
            distances = torch.norm(level_centroids - embedding.unsqueeze(0), dim=-1)
            nearest = distances.argmin().item()

            path.append(start_idx + nearest)
            current_idx = start_idx + nearest

        leaf_id = path[-1] if path else 0
        return leaf_id, path

    def get_cluster_patterns(
        self,
        patterns: Dict[str, StoredPattern],
        cluster_id: int,
    ) -> List[StoredPattern]:
        """Get all patterns in a cluster.

        Args:
            patterns: All stored patterns.
            cluster_id: Cluster to retrieve.

        Returns:
            Patterns in the specified cluster.
        """
        return [p for p in patterns.values() if p.cluster_id == cluster_id]

    def update_centroids(
        self,
        patterns: Dict[str, StoredPattern],
        learning_rate: float = 0.1,
    ) -> None:
        """Update centroids based on current patterns.

        Args:
            patterns: Current stored patterns.
            learning_rate: Update rate for centroids.
        """
        for level, centroids in enumerate(self.centroids):
            # Group patterns by cluster at this level
            cluster_embeddings = defaultdict(list)
            for pattern in patterns.values():
                if pattern.cluster_id is not None:
                    # Find which centroid this pattern belongs to at this level
                    _, path = self.assign_cluster(pattern.embedding)
                    if level < len(path):
                        cluster_embeddings[path[level]].append(pattern.embedding)

            # Update each centroid
            for cluster_idx, embeddings in cluster_embeddings.items():
                if cluster_idx < len(centroids) and embeddings:
                    stacked = torch.stack(embeddings)
                    mean = stacked.mean(dim=0)
                    with torch.no_grad():
                        centroids[cluster_idx] = (
                            (1 - learning_rate) * centroids[cluster_idx]
                            + learning_rate * mean
                        )


class PatternMemory(nn.Module):
    """Persistent memory for reasoning patterns.

    Provides storage, retrieval, and consolidation of patterns
    learned from user conversations.

    Attributes:
        config: Pattern memory configuration.
        patterns: Stored patterns by ID.
        index: Approximate nearest neighbor index.
        clusterer: Hierarchical pattern organizer.
    """

    def __init__(self, config: PatternMemoryConfig) -> None:
        """Initialize pattern memory.

        Args:
            config: Pattern memory configuration.
        """
        super().__init__()
        self.config = config

        # Pattern storage
        self.patterns: Dict[str, StoredPattern] = {}
        self._next_id = 0

        # Approximate index
        self.index = PatternIndex(
            pattern_dim=config.pattern_dim,
            num_hash_tables=8,
            hash_dim=16,
        )

        # Hierarchical organization
        if config.use_hierarchical:
            self.clusterer = HierarchicalClusterer(
                pattern_dim=config.pattern_dim,
                num_levels=config.hierarchy_levels,
            )
        else:
            self.clusterer = None

        # Consolidation counter
        self._operations_since_consolidation = 0

        logger.info(
            f"Initialized PatternMemory: max_patterns={config.max_patterns}, "
            f"mode={config.retrieval_mode}"
        )

    def _generate_id(self) -> str:
        """Generate unique pattern ID."""
        self._next_id += 1
        return f"pattern_{self._next_id}"

    def store(
        self,
        pattern: ReasoningPattern,
        auto_consolidate: bool = True,
    ) -> str:
        """Store a reasoning pattern.

        Args:
            pattern: Pattern to store.
            auto_consolidate: Whether to trigger consolidation if needed.

        Returns:
            Stored pattern ID.
        """
        pattern_id = self._generate_id()

        # Create stored pattern
        stored = StoredPattern(
            id=pattern_id,
            embedding=pattern.embedding,
            pattern_type=pattern.pattern_type,
            coordinate=pattern.coordinate,
            importance=pattern.strength,
            metadata=pattern.metadata,
        )

        # Assign to cluster if hierarchical
        if self.clusterer is not None:
            cluster_id, _ = self.clusterer.assign_cluster(pattern.embedding)
            stored.cluster_id = cluster_id

        # Store pattern
        self.patterns[pattern_id] = stored

        # Add to index
        self.index.insert(pattern_id, pattern.embedding)

        # Check capacity
        if len(self.patterns) > self.config.max_patterns:
            self._evict_least_important()

        # Auto-consolidation
        self._operations_since_consolidation += 1
        if auto_consolidate and self._operations_since_consolidation >= self.config.consolidation_interval:
            self.consolidate()
            self._operations_since_consolidation = 0

        return pattern_id

    def retrieve(self, query: PatternQuery) -> List[Tuple[StoredPattern, float]]:
        """Retrieve patterns matching a query.

        Args:
            query: Pattern query specification.

        Returns:
            List of (pattern, similarity) tuples sorted by relevance.
        """
        current_time = time.time()

        # Apply decay to all patterns
        for pattern in self.patterns.values():
            pattern.decay(self.config.decay_rate, current_time)

        # Get candidates based on retrieval mode
        if self.config.retrieval_mode == MemoryRetrievalMode.APPROXIMATE:
            candidate_ids = self.index.query(
                query.embedding,
                max_candidates=query.max_results * 3,
            )
        elif self.config.retrieval_mode == MemoryRetrievalMode.EXACT:
            candidate_ids = list(self.patterns.keys())
        else:  # HYBRID
            # Start with ANN, then refine
            candidate_ids = self.index.query(
                query.embedding,
                max_candidates=query.max_results * 5,
            )
            if len(candidate_ids) < query.max_results:
                candidate_ids = list(self.patterns.keys())

        # Filter and score candidates
        results = []
        for pattern_id in candidate_ids:
            if pattern_id not in self.patterns:
                continue

            pattern = self.patterns[pattern_id]

            # Apply filters
            if pattern.importance < query.min_importance:
                continue

            if query.pattern_types and pattern.pattern_type not in query.pattern_types:
                continue

            if query.temporal_window:
                age = current_time - pattern.created_at
                if age > query.temporal_window:
                    continue

            # Compute similarity
            similarity = F.cosine_similarity(
                query.embedding.unsqueeze(0),
                pattern.embedding.unsqueeze(0),
            ).item()

            # Boost by coordinate proximity if available
            if query.coordinate and pattern.coordinate:
                coord_distance = query.coordinate.weighted_distance(pattern.coordinate)
                coord_boost = math.exp(-coord_distance)
                similarity = 0.7 * similarity + 0.3 * coord_boost

            # Boost by importance
            final_score = similarity * (0.5 + 0.5 * pattern.importance)

            if final_score >= self.config.similarity_threshold:
                results.append((pattern, final_score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Record access for top results
        for pattern, _ in results[:query.max_results]:
            pattern.access()

        return results[:query.max_results]

    def _evict_least_important(self) -> None:
        """Evict least important pattern when at capacity."""
        if not self.patterns:
            return

        # Find least important
        least_important_id = min(
            self.patterns.keys(),
            key=lambda k: self.patterns[k].importance,
        )
        pattern = self.patterns[least_important_id]

        # Remove from index
        self.index.remove(least_important_id, pattern.embedding)

        # Remove from storage
        del self.patterns[least_important_id]

        logger.debug(f"Evicted pattern {least_important_id} (importance={pattern.importance:.3f})")

    def consolidate(self) -> int:
        """Consolidate similar patterns to reduce redundancy.

        Returns:
            Number of patterns consolidated.
        """
        if len(self.patterns) < 10:
            return 0

        consolidated = 0
        patterns_to_merge = []

        # Find similar patterns
        pattern_list = list(self.patterns.values())
        for i, p1 in enumerate(pattern_list):
            for p2 in pattern_list[i + 1:]:
                if p1.pattern_type != p2.pattern_type:
                    continue

                similarity = F.cosine_similarity(
                    p1.embedding.unsqueeze(0),
                    p2.embedding.unsqueeze(0),
                ).item()

                if similarity > 0.95:  # Very similar
                    patterns_to_merge.append((p1, p2))

        # Merge similar patterns
        merged_ids = set()
        for p1, p2 in patterns_to_merge:
            if p1.id in merged_ids or p2.id in merged_ids:
                continue

            # Keep the more important one, merge info
            if p1.importance >= p2.importance:
                keep, remove = p1, p2
            else:
                keep, remove = p2, p1

            # Update kept pattern
            keep.access_count += remove.access_count
            keep.importance = max(keep.importance, remove.importance)

            # Average embeddings
            keep.embedding = (keep.embedding + remove.embedding) / 2

            # Remove duplicate
            self.index.remove(remove.id, remove.embedding)
            del self.patterns[remove.id]
            merged_ids.add(remove.id)
            consolidated += 1

        # Update cluster centroids
        if self.clusterer is not None:
            self.clusterer.update_centroids(self.patterns)

        logger.info(f"Consolidated {consolidated} patterns, {len(self.patterns)} remaining")
        return consolidated

    def get_by_type(self, pattern_type: PatternType) -> List[StoredPattern]:
        """Get all patterns of a specific type.

        Args:
            pattern_type: Type to filter by.

        Returns:
            Patterns of the specified type.
        """
        return [p for p in self.patterns.values() if p.pattern_type == pattern_type]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary of memory statistics.
        """
        if not self.patterns:
            return {"num_patterns": 0}

        importances = [p.importance for p in self.patterns.values()]
        type_counts = defaultdict(int)
        for p in self.patterns.values():
            type_counts[p.pattern_type.value] += 1

        return {
            "num_patterns": len(self.patterns),
            "mean_importance": sum(importances) / len(importances),
            "min_importance": min(importances),
            "max_importance": max(importances),
            "type_distribution": dict(type_counts),
            "total_accesses": sum(p.access_count for p in self.patterns.values()),
        }

    def save_state(self, path: str) -> None:
        """Save memory state to disk.

        Args:
            path: Path to save state.
        """
        state = {
            "patterns": {
                pid: {
                    "embedding": p.embedding.cpu(),
                    "pattern_type": p.pattern_type.value,
                    "importance": p.importance,
                    "access_count": p.access_count,
                    "created_at": p.created_at,
                    "last_accessed": p.last_accessed,
                    "cluster_id": p.cluster_id,
                    "metadata": p.metadata,
                }
                for pid, p in self.patterns.items()
            },
            "next_id": self._next_id,
            "operations_since_consolidation": self._operations_since_consolidation,
        }
        torch.save(state, path)
        logger.info(f"Saved pattern memory to {path}")

    def load_state(self, path: str) -> None:
        """Load memory state from disk.

        Args:
            path: Path to load state from.
        """
        state = torch.load(path)

        self.patterns = {}
        for pid, data in state["patterns"].items():
            pattern = StoredPattern(
                id=pid,
                embedding=data["embedding"],
                pattern_type=PatternType(data["pattern_type"]),
                coordinate=None,  # Coordinates not serialized
                importance=data["importance"],
                access_count=data["access_count"],
                created_at=data["created_at"],
                last_accessed=data["last_accessed"],
                cluster_id=data["cluster_id"],
                metadata=data["metadata"],
            )
            self.patterns[pid] = pattern
            self.index.insert(pid, pattern.embedding)

        self._next_id = state["next_id"]
        self._operations_since_consolidation = state["operations_since_consolidation"]

        logger.info(f"Loaded {len(self.patterns)} patterns from {path}")


def create_pattern_memory(**kwargs) -> PatternMemory:
    """Factory function to create pattern memory.

    Args:
        **kwargs: Configuration arguments.

    Returns:
        PatternMemory instance.
    """
    config = PatternMemoryConfig(**kwargs)
    return PatternMemory(config)
