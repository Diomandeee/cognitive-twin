"""Global Style Signature for Hybrid CognitiveTwin.

This module provides an evolving global style signature that accumulates
across all training sessions. Uses exponential moving average for smooth
evolution across time and interactions.

The GlobalStyleSignature:
    - Blends new style observations into a persistent signature
    - Tracks confidence based on sample count
    - Maintains history snapshots for temporal analysis
    - Persists to checkpoint files for continuity across sessions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SignatureSnapshot:
    """A point-in-time snapshot of the global signature.

    Used for tracking signature evolution over time.

    Attributes:
        signature: The signature vector at this point.
        timestamp: When this snapshot was taken.
        confidence: Confidence level at snapshot time.
        update_count: Number of updates at snapshot time.
        metadata: Additional snapshot metadata.
    """

    signature: Tensor
    timestamp: datetime
    confidence: float
    update_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "signature": self.signature.tolist(),
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "update_count": self.update_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignatureSnapshot":
        """Reconstruct from dictionary."""
        return cls(
            signature=torch.tensor(data["signature"], dtype=torch.float32),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            confidence=data["confidence"],
            update_count=data["update_count"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class StyleSignatureVector:
    """A style signature with metadata.

    Attributes:
        vector: The style embedding vector.
        confidence: Confidence in this signature (0-1).
        update_count: Number of updates that contributed.
        last_updated: When the signature was last updated.
    """

    vector: Tensor
    confidence: float = 0.0
    update_count: int = 0
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "vector": self.vector.tolist(),
            "confidence": self.confidence,
            "update_count": self.update_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class GlobalStyleSignature:
    """Evolving global style signature that accumulates across all training.

    Uses exponential moving average for smooth evolution. The signature
    represents the user's unique style across all projects and sessions.

    Attributes:
        dim: Dimension of the style embedding.
        momentum: EMA momentum factor (higher = more stable).
        signature: Current signature vector.
        confidence: Confidence in the signature (0-1).
        update_count: Number of blending updates.
        history: List of historical snapshots.
    """

    def __init__(
        self,
        dim: int = 256,
        momentum: float = 0.8,
        snapshot_interval: int = 10,
        max_history: int = 100,
    ) -> None:
        """Initialize global style signature.

        Args:
            dim: Dimension of style embedding.
            momentum: EMA momentum (0.8 = 80% old, 20% new for faster adaptation).
            snapshot_interval: How often to take snapshots.
            max_history: Maximum history snapshots to keep.
        """
        self.dim = dim
        self.momentum = momentum
        self.snapshot_interval = snapshot_interval
        self.max_history = max_history

        # Core signature state
        self.signature = torch.zeros(dim)
        self.confidence = 0.0
        self.update_count = 0
        self.last_updated: Optional[datetime] = None

        # History for temporal analysis
        self.history: List[SignatureSnapshot] = []

        # Component weights (if using decomposed style)
        self.component_weights: Optional[Tensor] = None

        logger.info(f"Initialized GlobalStyleSignature (dim={dim}, momentum={momentum})")

    def blend(
        self,
        new_signature: Tensor,
        weight: float = 0.1,
        source: Optional[str] = None,
    ) -> None:
        """Blend a new signature observation into the global signature.

        Uses exponential moving average with the configured momentum.
        Lower weight means the new observation has less impact.

        Args:
            new_signature: New style signature to blend in.
            weight: Importance weight for this observation (0-1).
            source: Optional source identifier for logging.
        """
        if new_signature.shape[0] != self.dim:
            raise ValueError(
                f"Signature dimension mismatch: expected {self.dim}, got {new_signature.shape[0]}"
            )

        # Normalize input
        new_norm = F.normalize(new_signature.detach(), dim=0)

        # EMA update: signature = momentum * old + (1 - momentum) * weight * new
        self.signature = (
            self.momentum * self.signature
            + (1 - self.momentum) * weight * new_norm
        )

        # Re-normalize to unit sphere
        self.signature = F.normalize(self.signature, dim=0)

        # Update tracking
        self.update_count += 1
        self.last_updated = datetime.utcnow()

        # Confidence grows with more observations (asymptotic to 1.0)
        # ~50% confidence at 50 updates, ~90% at 200 updates
        self.confidence = min(1.0, self.update_count / (self.update_count + 100))

        # Take snapshot at intervals
        if self.update_count % self.snapshot_interval == 0:
            self._take_snapshot(source)

        logger.debug(
            f"Blended signature (update={self.update_count}, "
            f"confidence={self.confidence:.3f}, source={source})"
        )

    def blend_batch(
        self,
        signatures: Tensor,
        weights: Optional[Tensor] = None,
    ) -> None:
        """Blend a batch of signatures at once.

        More efficient than individual blends for batch training.

        Args:
            signatures: Batch of signatures [batch, dim].
            weights: Optional per-signature weights [batch].
        """
        if signatures.shape[1] != self.dim:
            raise ValueError(
                f"Signature dimension mismatch: expected {self.dim}, got {signatures.shape[1]}"
            )

        batch_size = signatures.shape[0]

        if weights is None:
            weights = torch.ones(batch_size) / batch_size

        # Normalize weights
        weights = weights / weights.sum()

        # Compute weighted average of new signatures
        signatures_norm = F.normalize(signatures.detach(), dim=1)
        weighted_new = (signatures_norm * weights.unsqueeze(1)).sum(dim=0)

        # Blend with EMA
        self.signature = (
            self.momentum * self.signature
            + (1 - self.momentum) * weighted_new
        )
        self.signature = F.normalize(self.signature, dim=0)

        # Update tracking
        self.update_count += batch_size
        self.last_updated = datetime.utcnow()
        self.confidence = min(1.0, self.update_count / (self.update_count + 100))

        logger.debug(f"Blended batch of {batch_size} signatures")

    def current(self) -> StyleSignatureVector:
        """Get the current style signature with metadata.

        Returns:
            StyleSignatureVector with current state.
        """
        return StyleSignatureVector(
            vector=self.signature.clone(),
            confidence=self.confidence,
            update_count=self.update_count,
            last_updated=self.last_updated,
        )

    def similarity(self, other: Tensor) -> float:
        """Compute cosine similarity with another signature.

        Args:
            other: Another signature to compare.

        Returns:
            Cosine similarity in [-1, 1].
        """
        if other.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.shape[0]}")

        other_norm = F.normalize(other.detach(), dim=0)
        return F.cosine_similarity(
            self.signature.unsqueeze(0),
            other_norm.unsqueeze(0),
        ).item()

    def _take_snapshot(self, source: Optional[str] = None) -> None:
        """Take a snapshot of current state for history."""
        snapshot = SignatureSnapshot(
            signature=self.signature.clone(),
            timestamp=datetime.utcnow(),
            confidence=self.confidence,
            update_count=self.update_count,
            metadata={"source": source} if source else {},
        )
        self.history.append(snapshot)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_evolution(self) -> List[Dict[str, Any]]:
        """Get signature evolution over time.

        Returns:
            List of snapshot summaries ordered by time.
        """
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "confidence": s.confidence,
                "update_count": s.update_count,
            }
            for s in self.history
        ]

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary containing all state needed to restore.
        """
        return {
            "dim": self.dim,
            "momentum": self.momentum,
            "snapshot_interval": self.snapshot_interval,
            "max_history": self.max_history,
            "signature": self.signature.tolist(),
            "confidence": self.confidence,
            "update_count": self.update_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "history": [s.to_dict() for s in self.history],
            "component_weights": (
                self.component_weights.tolist() if self.component_weights is not None else None
            ),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from dictionary.

        Args:
            state: State dictionary from state_dict().
        """
        self.dim = state["dim"]
        self.momentum = state["momentum"]
        self.snapshot_interval = state.get("snapshot_interval", 10)
        self.max_history = state.get("max_history", 100)
        self.signature = torch.tensor(state["signature"], dtype=torch.float32)
        self.confidence = state["confidence"]
        self.update_count = state["update_count"]
        self.last_updated = (
            datetime.fromisoformat(state["last_updated"])
            if state.get("last_updated")
            else None
        )
        self.history = [
            SignatureSnapshot.from_dict(s)
            for s in state.get("history", [])
        ]
        if state.get("component_weights"):
            self.component_weights = torch.tensor(
                state["component_weights"], dtype=torch.float32
            )

        logger.info(
            f"Loaded GlobalStyleSignature (updates={self.update_count}, "
            f"confidence={self.confidence:.3f})"
        )

    def save(self, path: Path) -> None:
        """Save signature to checkpoint file.

        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Saved GlobalStyleSignature to {path}")

    @classmethod
    def load(cls, path: Path) -> "GlobalStyleSignature":
        """Load signature from checkpoint file.

        Args:
            path: Path to checkpoint file.

        Returns:
            Loaded GlobalStyleSignature.
        """
        state = torch.load(path, weights_only=True)
        instance = cls(
            dim=state["dim"],
            momentum=state["momentum"],
        )
        instance.load_state_dict(state)
        return instance

    def reset(self) -> None:
        """Reset the signature to initial state."""
        self.signature = torch.zeros(self.dim)
        self.confidence = 0.0
        self.update_count = 0
        self.last_updated = None
        self.history = []
        logger.info("Reset GlobalStyleSignature")

    def __repr__(self) -> str:
        return (
            f"GlobalStyleSignature(dim={self.dim}, "
            f"updates={self.update_count}, "
            f"confidence={self.confidence:.3f})"
        )

