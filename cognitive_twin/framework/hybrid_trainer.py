"""Hybrid CognitiveTwin Trainer.

Unified trainer supporting all training modes for the CognitiveTwin:
1. Batch on prompt/response pairs (from claude_prompts)
2. Batch on memory_turns (unified knowledge fabric)
3. Real-time style extraction (inference only)
4. Incremental learning per interaction

This trainer integrates with:
- Orbit orchestrator for session-end triggers
- Scheduled training (Celery/APScheduler)
- Threshold-based training triggers
- Manual API triggers

The trainer maintains a GlobalStyleSignature that evolves across all training.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from cognitive_twin.framework.config import CognitiveTwinConfig
from cognitive_twin.framework.global_signature import (
    GlobalStyleSignature,
    StyleSignatureVector,
)
from cognitive_twin.framework.trainer import (
    CognitiveTwinTrainer,
    TrainingMetrics,
    TrainingState,
    CognitiveTwinDataset,
    collate_conversation_batch,
)
from cognitive_twin.framework.twin import CognitiveTwin
from cognitive_twin.framework.style_projector import StyleSignature

logger = logging.getLogger(__name__)


class TrainingMode(str, Enum):
    """Training mode for the hybrid trainer."""

    BATCH_PROMPTS = "batch_prompts"      # Historical prompt/response pairs
    BATCH_TURNS = "batch_turns"          # Unified memory_turns
    REALTIME_STYLE = "realtime_style"    # Inference-only style extraction
    INCREMENTAL = "incremental"          # Update after each interaction


class TriggerType(str, Enum):
    """Training trigger types."""

    SESSION_END = "session_end"    # Orbit session boundary
    SCHEDULED = "scheduled"        # Cron-like schedule
    THRESHOLD = "threshold"        # After N interactions
    MANUAL = "manual"              # API/UI triggered


@dataclass
class HybridConfig:
    """Configuration for hybrid training.

    Attributes:
        enable_incremental: Whether to run incremental learning.
        incremental_weight: Weight for incremental blending (0-1).
        threshold_count: Trigger batch training after N incremental updates.
        signature_checkpoint_path: Path to save/load global signature.
        style_dim: Dimension of style embeddings.
        batch_epochs: Number of epochs for batch training.
        min_batch_size: Minimum samples required for batch training.
        auto_save_signature: Auto-save signature after training.
    """

    enable_incremental: bool = True
    incremental_weight: float = 0.1
    threshold_count: int = 50
    signature_checkpoint_path: Path = Path("checkpoints/cognitive_twin/global_signature.pt")
    style_dim: int = 256
    batch_epochs: int = 3
    min_batch_size: int = 10
    auto_save_signature: bool = True


@dataclass
class PromptResponsePair:
    """A prompt/response pair for training.

    Attributes:
        prompt_id: Unique identifier for the prompt.
        prompt_text: User's prompt text.
        response_text: Assistant's response text.
        embedding: Pre-computed embedding (optional).
        timestamp: When the prompt was submitted.
        metadata: Additional metadata.
    """

    prompt_id: str
    prompt_text: str
    response_text: str
    embedding: Optional[Tensor] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def combined_text(self) -> str:
        """Get combined prompt + response text."""
        return f"User: {self.prompt_text}\n\nAssistant: {self.response_text}"

    def to_turn_dict(self) -> Dict[str, Any]:
        """Convert to turn dictionary format."""
        return {
            "content": self.combined_text,
            "role": "assistant",
            "conversation_id": self.prompt_id,
            "turn_index": 0,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class MemoryTurn:
    """A memory turn from the unified knowledge fabric.

    Attributes:
        turn_id: Unique turn identifier.
        content: Turn content text.
        role: Role (user/assistant).
        embedding: Pre-computed embedding.
        trajectory_coords: 5D trajectory coordinates.
        conversation_id: Parent conversation ID.
        metadata: Additional metadata.
    """

    turn_id: str
    content: str
    role: str
    embedding: Tensor
    trajectory_coords: Optional[Tensor] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "content": self.content,
            "role": self.role,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "trajectory_coords": (
                self.trajectory_coords.tolist()
                if self.trajectory_coords is not None
                else None
            ),
            "conversation_id": self.conversation_id,
            "metadata": self.metadata,
        }


@dataclass
class TrainingEvent:
    """Record of a training event.

    Attributes:
        event_id: Unique event identifier.
        trigger: What triggered the training.
        mode: Training mode used.
        started_at: When training started.
        completed_at: When training completed.
        metrics: Training metrics.
        samples_count: Number of samples trained on.
        success: Whether training succeeded.
        error: Error message if failed.
    """

    event_id: str
    trigger: TriggerType
    mode: TrainingMode
    started_at: datetime
    completed_at: Optional[datetime] = None
    metrics: Optional[TrainingMetrics] = None
    samples_count: int = 0
    success: bool = True
    error: Optional[str] = None


class HybridCognitiveTwinTrainer:
    """Unified trainer supporting all training modes.

    The hybrid trainer manages:
    - Batch training on prompts and memory turns
    - Real-time style extraction during inference
    - Incremental learning after each interaction
    - Global signature evolution across all training

    It integrates with external triggers:
    - Orbit session-end events
    - Scheduled training tasks
    - Threshold-based triggers
    - Manual API calls
    """

    def __init__(
        self,
        twin: CognitiveTwin,
        config: HybridConfig,
        embedding_fn: Optional[Callable[[str], Tensor]] = None,
    ) -> None:
        """Initialize the hybrid trainer.

        Args:
            twin: CognitiveTwin model to train.
            config: Hybrid training configuration.
            embedding_fn: Optional function to compute embeddings.
        """
        self.twin = twin
        self.config = config
        self.embedding_fn = embedding_fn

        # Global style signature
        self.global_signature = GlobalStyleSignature(
            dim=config.style_dim,
            momentum=0.9,
        )

        # Try to load existing signature
        if config.signature_checkpoint_path.exists():
            try:
                self.global_signature = GlobalStyleSignature.load(
                    config.signature_checkpoint_path
                )
                logger.info(f"Loaded existing global signature: {self.global_signature}")
            except Exception as e:
                logger.warning(f"Failed to load signature, starting fresh: {e}")

        # Incremental learning state
        self.interaction_counter = 0
        self.pending_batch: List[MemoryTurn] = []

        # Training history
        self.training_events: List[TrainingEvent] = []
        self.last_scheduled_train: Optional[datetime] = None

        # Device
        self.device = next(twin.parameters()).device

        logger.info(
            f"Initialized HybridCognitiveTwinTrainer:\n"
            f"  - Device: {self.device}\n"
            f"  - Incremental: {config.enable_incremental}\n"
            f"  - Threshold: {config.threshold_count}\n"
            f"  - Style dim: {config.style_dim}"
        )

    # ========== Training Modes ==========

    async def batch_train_prompts(
        self,
        prompts: List[PromptResponsePair],
        epochs: int = 3,
    ) -> TrainingMetrics:
        """Train on prompt/response pairs from claude_prompts.

        Converts prompts to turn format and trains the CognitiveTwin.

        Args:
            prompts: List of prompt/response pairs.
            epochs: Number of training epochs.

        Returns:
            Training metrics.
        """
        if len(prompts) < self.config.min_batch_size:
            logger.warning(
                f"Insufficient prompts for batch training: "
                f"{len(prompts)} < {self.config.min_batch_size}"
            )
            return TrainingMetrics(num_samples=0)

        logger.info(f"Starting batch training on {len(prompts)} prompts")

        # Compute embeddings if needed
        embeddings = await self._ensure_embeddings(
            [p.combined_text for p in prompts],
            [p.embedding for p in prompts],
        )

        # Convert to turn dictionaries
        turns = [p.to_turn_dict() for p in prompts]
        for i, turn in enumerate(turns):
            turn["embedding"] = embeddings[i].tolist()

        # Train using existing trainer infrastructure
        metrics = self._train_on_turns(turns, epochs)

        # Update global signature with learned style
        if metrics.num_samples > 0:
            self._update_global_from_batch(embeddings)

        # Record event
        self._record_training_event(
            trigger=TriggerType.MANUAL,
            mode=TrainingMode.BATCH_PROMPTS,
            metrics=metrics,
            samples_count=len(prompts),
        )

        return metrics

    async def batch_train_turns(
        self,
        turns: List[MemoryTurn],
        epochs: int = 3,
    ) -> TrainingMetrics:
        """Train on unified memory_turns with trajectory context.

        Uses the full trajectory coordinates for contextual learning.

        Args:
            turns: List of memory turns.
            epochs: Number of training epochs.

        Returns:
            Training metrics.
        """
        if len(turns) < self.config.min_batch_size:
            logger.warning(
                f"Insufficient turns for batch training: "
                f"{len(turns)} < {self.config.min_batch_size}"
            )
            return TrainingMetrics(num_samples=0)

        logger.info(f"Starting batch training on {len(turns)} memory turns")

        # Stack embeddings and coordinates
        embeddings = torch.stack([t.embedding for t in turns])
        coordinates = [t.trajectory_coords for t in turns if t.trajectory_coords is not None]

        # Convert to turn dictionaries
        turn_dicts = [t.to_dict() for t in turns]

        # Train
        metrics = self._train_on_turns(turn_dicts, epochs)

        # Update global signature
        if metrics.num_samples > 0:
            self._update_global_from_batch(embeddings)

        # Record event
        self._record_training_event(
            trigger=TriggerType.MANUAL,
            mode=TrainingMode.BATCH_TURNS,
            metrics=metrics,
            samples_count=len(turns),
        )

        return metrics

    def realtime_style_extract(
        self,
        text: str,
        embedding: Optional[Tensor] = None,
    ) -> StyleSignatureVector:
        """Extract style without training (inference only).

        Fast path for real-time style analysis during generation.

        Args:
            text: Text to analyze.
            embedding: Pre-computed embedding (optional).

        Returns:
            Style signature for the input.
        """
        self.twin.eval()

        with torch.no_grad():
            # Get or compute embedding
            if embedding is None:
                if self.embedding_fn is None:
                    raise ValueError("No embedding provided and no embedding_fn configured")
                embedding = self.embedding_fn(text)

            embedding = embedding.to(self.device)

            # Project to style space
            style_embedding = self.twin.style_projector(embedding.unsqueeze(0))

            return StyleSignatureVector(
                vector=style_embedding.embedding.squeeze(0),
                confidence=0.8,  # Single-sample confidence
                update_count=1,
            )

    async def incremental_learn(
        self,
        turn: MemoryTurn,
    ) -> None:
        """Update global signature after single interaction.

        Lightweight update for real-time learning. Accumulates
        interactions and triggers batch training at threshold.

        Args:
            turn: Memory turn to learn from.
        """
        if not self.config.enable_incremental:
            return

        # Extract style (inference only)
        style = self.realtime_style_extract(turn.content, turn.embedding)

        # Blend into global signature
        self.global_signature.blend(
            style.vector,
            weight=self.config.incremental_weight,
            source="incremental",
        )

        # Track for potential batch training
        self.pending_batch.append(turn)
        self.interaction_counter += 1

        logger.debug(
            f"Incremental learn: interaction={self.interaction_counter}, "
            f"pending={len(self.pending_batch)}"
        )

        # Check threshold trigger
        if self.interaction_counter >= self.config.threshold_count:
            await self._trigger_threshold_training()

    # ========== Training Triggers ==========

    async def on_session_end(self, session_id: str) -> Optional[TrainingMetrics]:
        """Triggered by Orbit when a session ends.

        Fetches all turns from the session and triggers batch training.

        Args:
            session_id: Orbit session ID that ended.

        Returns:
            Training metrics if training was triggered.
        """
        logger.info(f"Session end trigger: {session_id}")

        # Fetch session turns (would integrate with Supabase)
        turns = await self._fetch_session_turns(session_id)

        if not turns:
            logger.info(f"No turns found for session {session_id}")
            return None

        # Train on session turns
        metrics = await self.batch_train_turns(turns)

        # Persist signature
        self.persist_global_signature()

        # Record event
        self._record_training_event(
            trigger=TriggerType.SESSION_END,
            mode=TrainingMode.BATCH_TURNS,
            metrics=metrics,
            samples_count=len(turns),
        )

        return metrics

    async def on_schedule(self) -> Optional[TrainingMetrics]:
        """Triggered by scheduler (cron/celery).

        Trains on recent turns since last scheduled training.

        Returns:
            Training metrics if training was triggered.
        """
        logger.info("Scheduled training trigger")

        # Fetch recent turns
        turns = await self._fetch_recent_turns(since=self.last_scheduled_train)

        if not turns:
            logger.info("No new turns for scheduled training")
            return None

        # Train
        metrics = await self.batch_train_turns(turns)

        # Update last scheduled time
        self.last_scheduled_train = datetime.utcnow()

        # Persist signature
        self.persist_global_signature()

        # Record event
        self._record_training_event(
            trigger=TriggerType.SCHEDULED,
            mode=TrainingMode.BATCH_TURNS,
            metrics=metrics,
            samples_count=len(turns),
        )

        return metrics

    async def on_manual_trigger(
        self,
        mode: TrainingMode,
        source_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrainingMetrics]:
        """Triggered via API or UI.

        Allows explicit control over training mode and data source.

        Args:
            mode: Training mode to use.
            source_filter: Optional filter for data selection.

        Returns:
            Training metrics if training was triggered.
        """
        logger.info(f"Manual training trigger: mode={mode}, filter={source_filter}")

        if mode == TrainingMode.BATCH_PROMPTS:
            prompts = await self._fetch_prompts(source_filter)
            if prompts:
                return await self.batch_train_prompts(prompts)

        elif mode == TrainingMode.BATCH_TURNS:
            turns = await self._fetch_turns(source_filter)
            if turns:
                return await self.batch_train_turns(turns)

        elif mode == TrainingMode.REALTIME_STYLE:
            logger.info("Real-time style mode is inference-only, no training")
            return None

        elif mode == TrainingMode.INCREMENTAL:
            # Process pending batch
            if self.pending_batch:
                return await self.batch_train_turns(self.pending_batch)

        return None

    async def _trigger_threshold_training(self) -> None:
        """Internal trigger when threshold is reached."""
        logger.info(f"Threshold trigger: {self.interaction_counter} interactions")

        if self.pending_batch:
            metrics = await self.batch_train_turns(self.pending_batch)

            # Record event
            self._record_training_event(
                trigger=TriggerType.THRESHOLD,
                mode=TrainingMode.BATCH_TURNS,
                metrics=metrics,
                samples_count=len(self.pending_batch),
            )

            # Clear pending batch
            self.pending_batch = []

        # Reset counter
        self.interaction_counter = 0

        # Persist signature
        self.persist_global_signature()

    # ========== Global Signature Management ==========

    def get_global_signature(self) -> StyleSignatureVector:
        """Get current global style signature.

        Returns:
            Current style signature with metadata.
        """
        return self.global_signature.current()

    def persist_global_signature(self) -> None:
        """Save signature to checkpoint."""
        if self.config.auto_save_signature:
            self.global_signature.save(self.config.signature_checkpoint_path)

    def reset_global_signature(self) -> None:
        """Reset the global signature to initial state."""
        self.global_signature.reset()
        logger.info("Reset global signature")

    # ========== Internal Methods ==========

    def _train_on_turns(
        self,
        turns: List[Dict[str, Any]],
        epochs: int,
    ) -> TrainingMetrics:
        """Internal training on turn dictionaries."""
        from torch.utils.data import DataLoader

        self.twin.train()

        # Create dataset
        dataset = CognitiveTwinDataset(
            turns=turns,
            embed_dim=768,  # TODO: Make configurable
            max_context_length=32,
        )

        if len(dataset) == 0:
            return TrainingMetrics(num_samples=0)

        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=min(32, len(dataset)),
            shuffle=True,
            collate_fn=lambda x: collate_conversation_batch(x, embed_dim=768),
        )

        # Simple training loop
        total_metrics = TrainingMetrics()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in loader:
                batch = batch.to(self.device)

                # Forward pass
                output = self.twin.learn_from_conversation(
                    turns=batch.content,
                    embeddings=batch.response_embeddings,
                    coordinates=batch.response_coords,
                    context_embeddings=batch.context_embeddings,
                    context_mask=batch.context_mask,
                )

                epoch_loss += output.metrics.get("total_loss", 0.0)
                num_batches += 1

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        total_metrics.num_samples = len(turns)
        return total_metrics

    def _update_global_from_batch(self, embeddings: Tensor) -> None:
        """Update global signature from batch of embeddings."""
        self.twin.eval()

        with torch.no_grad():
            embeddings = embeddings.to(self.device)

            # Project all to style space
            style_embeddings = self.twin.style_projector(embeddings)

            # Blend batch into global
            self.global_signature.blend_batch(style_embeddings.embedding)

    async def _ensure_embeddings(
        self,
        texts: List[str],
        existing: List[Optional[Tensor]],
    ) -> Tensor:
        """Ensure all texts have embeddings."""
        embeddings = []

        for i, (text, emb) in enumerate(zip(texts, existing)):
            if emb is not None:
                embeddings.append(emb)
            elif self.embedding_fn is not None:
                embeddings.append(self.embedding_fn(text))
            else:
                # Zero embedding fallback
                embeddings.append(torch.zeros(768))

        return torch.stack(embeddings)

    async def _fetch_session_turns(self, session_id: str) -> List[MemoryTurn]:
        """Fetch turns for a session from Supabase.

        Queries memory_turns where metadata contains the orbit_session_id.

        Args:
            session_id: Orbit session UUID.

        Returns:
            List of MemoryTurn objects for the session.
        """
        logger.debug(f"Fetching turns for session: {session_id}")

        try:
            from cognitive_twin._compat import SupabaseClient

            client = SupabaseClient()

            # Query turns with matching session in metadata
            # Note: Supabase JSON query syntax for metadata->>'orbit_session_id'
            result = (
                client.client.table("memory_turns")
                .select(
                    "id, content_text, role, embedding, conversation_id, "
                    "trajectory_depth, trajectory_sibling_order, "
                    "trajectory_homogeneity, trajectory_temporal, trajectory_complexity, "
                    "metadata"
                )
                .eq("metadata->>orbit_session_id", session_id)
                .not_.is_("embedding", "null")
                .order("created_at")
                .execute()
            )

            turns = []
            for row in result.data or []:
                embedding = row.get("embedding")
                if embedding is None:
                    continue

                # Build trajectory coordinates tensor
                coords = torch.tensor([
                    row.get("trajectory_depth", 0) or 0,
                    row.get("trajectory_sibling_order", 0) or 0,
                    row.get("trajectory_homogeneity", 0.5) or 0.5,
                    row.get("trajectory_temporal", 0.5) or 0.5,
                    row.get("trajectory_complexity", 1) or 1,
                ], dtype=torch.float32)

                turns.append(MemoryTurn(
                    turn_id=row["id"],
                    content=row.get("content_text", ""),
                    role=row.get("role", "assistant"),
                    embedding=torch.tensor(embedding, dtype=torch.float32),
                    trajectory_coords=coords,
                    conversation_id=row.get("conversation_id"),
                    metadata=row.get("metadata", {}),
                ))

            logger.info(f"Fetched {len(turns)} turns for session {session_id}")
            return turns

        except Exception as e:
            logger.error(f"Failed to fetch session turns: {e}")
            return []

    async def _fetch_recent_turns(
        self,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[MemoryTurn]:
        """Fetch recent turns since a timestamp.

        Args:
            since: Fetch turns created after this timestamp.
            limit: Maximum number of turns to fetch.

        Returns:
            List of MemoryTurn objects.
        """
        logger.debug(f"Fetching turns since: {since}")

        try:
            from cognitive_twin._compat import SupabaseClient

            client = SupabaseClient()

            # Build query
            query = (
                client.client.table("memory_turns")
                .select(
                    "id, content_text, role, embedding, conversation_id, "
                    "trajectory_depth, trajectory_sibling_order, "
                    "trajectory_homogeneity, trajectory_temporal, trajectory_complexity, "
                    "metadata, created_at"
                )
                .not_.is_("embedding", "null")
            )

            # Apply timestamp filter
            if since is not None:
                query = query.gte("created_at", since.isoformat())

            # Order and limit
            result = query.order("created_at", desc=True).limit(limit).execute()

            turns = []
            for row in result.data or []:
                embedding = row.get("embedding")
                if embedding is None:
                    continue

                coords = torch.tensor([
                    row.get("trajectory_depth", 0) or 0,
                    row.get("trajectory_sibling_order", 0) or 0,
                    row.get("trajectory_homogeneity", 0.5) or 0.5,
                    row.get("trajectory_temporal", 0.5) or 0.5,
                    row.get("trajectory_complexity", 1) or 1,
                ], dtype=torch.float32)

                turns.append(MemoryTurn(
                    turn_id=row["id"],
                    content=row.get("content_text", ""),
                    role=row.get("role", "assistant"),
                    embedding=torch.tensor(embedding, dtype=torch.float32),
                    trajectory_coords=coords,
                    conversation_id=row.get("conversation_id"),
                    metadata=row.get("metadata", {}),
                ))

            logger.info(f"Fetched {len(turns)} recent turns")
            return turns

        except Exception as e:
            logger.error(f"Failed to fetch recent turns: {e}")
            return []

    async def _fetch_prompts(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 500,
    ) -> List[PromptResponsePair]:
        """Fetch prompts from claude_prompts table.

        Args:
            filter: Optional filter dictionary with keys:
                - project_id: Filter by Orbit project
                - session_id: Filter by Orbit session
                - since: Filter by timestamp
            limit: Maximum number of prompts to fetch.

        Returns:
            List of PromptResponsePair objects.
        """
        logger.debug(f"Fetching prompts with filter: {filter}")
        filter = filter or {}

        try:
            from cognitive_twin._compat import SupabaseClient

            client = SupabaseClient()

            # Build query
            query = (
                client.client.table("claude_prompts")
                .select("id, prompt_text, response_text, created_at, metadata")
            )

            # Apply filters
            if "project_id" in filter:
                query = query.eq("metadata->>orbit_project_id", filter["project_id"])

            if "session_id" in filter:
                query = query.eq("metadata->>orbit_session_id", filter["session_id"])

            if "since" in filter and filter["since"] is not None:
                since = filter["since"]
                if isinstance(since, datetime):
                    since = since.isoformat()
                query = query.gte("created_at", since)

            # Order and limit
            result = query.order("created_at", desc=True).limit(limit).execute()

            prompts = []
            for row in result.data or []:
                prompts.append(PromptResponsePair(
                    prompt_id=row["id"],
                    prompt_text=row.get("prompt_text", ""),
                    response_text=row.get("response_text", ""),
                    timestamp=(
                        datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                        if row.get("created_at")
                        else None
                    ),
                    metadata=row.get("metadata", {}),
                ))

            logger.info(f"Fetched {len(prompts)} prompts")
            return prompts

        except Exception as e:
            logger.error(f"Failed to fetch prompts: {e}")
            return []

    async def _fetch_turns(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
    ) -> List[MemoryTurn]:
        """Fetch turns from memory_turns with flexible filtering.

        Args:
            filter: Optional filter dictionary with keys:
                - conversation_id: Filter by conversation
                - role: Filter by role (user/assistant)
                - phase: Filter by trajectory phase
                - min_salience: Minimum salience score
                - project_id: Filter by Orbit project (in metadata)
            limit: Maximum number of turns to fetch.

        Returns:
            List of MemoryTurn objects.
        """
        logger.debug(f"Fetching turns with filter: {filter}")
        filter = filter or {}

        try:
            from cognitive_twin._compat import SupabaseClient

            client = SupabaseClient()

            # Build query
            query = (
                client.client.table("memory_turns")
                .select(
                    "id, content_text, role, embedding, conversation_id, "
                    "trajectory_depth, trajectory_sibling_order, "
                    "trajectory_homogeneity, trajectory_temporal, trajectory_complexity, "
                    "salience_score, trajectory_phase, metadata"
                )
                .not_.is_("embedding", "null")
            )

            # Apply filters
            if "conversation_id" in filter:
                query = query.eq("conversation_id", filter["conversation_id"])

            if "role" in filter:
                query = query.eq("role", filter["role"])

            if "phase" in filter:
                query = query.eq("trajectory_phase", filter["phase"])

            if "min_salience" in filter:
                query = query.gte("salience_score", filter["min_salience"])

            if "project_id" in filter:
                query = query.eq("metadata->>orbit_project_id", filter["project_id"])

            # Order and limit
            result = query.order("created_at", desc=True).limit(limit).execute()

            turns = []
            for row in result.data or []:
                embedding = row.get("embedding")
                if embedding is None:
                    continue

                coords = torch.tensor([
                    row.get("trajectory_depth", 0) or 0,
                    row.get("trajectory_sibling_order", 0) or 0,
                    row.get("trajectory_homogeneity", 0.5) or 0.5,
                    row.get("trajectory_temporal", 0.5) or 0.5,
                    row.get("trajectory_complexity", 1) or 1,
                ], dtype=torch.float32)

                turns.append(MemoryTurn(
                    turn_id=row["id"],
                    content=row.get("content_text", ""),
                    role=row.get("role", "assistant"),
                    embedding=torch.tensor(embedding, dtype=torch.float32),
                    trajectory_coords=coords,
                    conversation_id=row.get("conversation_id"),
                    metadata=row.get("metadata", {}),
                ))

            logger.info(f"Fetched {len(turns)} turns with filter")
            return turns

        except Exception as e:
            logger.error(f"Failed to fetch turns: {e}")
            return []

    def _record_training_event(
        self,
        trigger: TriggerType,
        mode: TrainingMode,
        metrics: Optional[TrainingMetrics],
        samples_count: int,
    ) -> None:
        """Record a training event in history."""
        import uuid

        event = TrainingEvent(
            event_id=str(uuid.uuid4())[:8],
            trigger=trigger,
            mode=mode,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            metrics=metrics,
            samples_count=samples_count,
            success=True,
        )
        self.training_events.append(event)

        # Keep last 100 events
        if len(self.training_events) > 100:
            self.training_events = self.training_events[-100:]

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training event history.

        Returns:
            List of training events.
        """
        return [
            {
                "event_id": e.event_id,
                "trigger": e.trigger.value,
                "mode": e.mode.value,
                "started_at": e.started_at.isoformat(),
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "samples_count": e.samples_count,
                "success": e.success,
            }
            for e in self.training_events
        ]

    def __repr__(self) -> str:
        return (
            f"HybridCognitiveTwinTrainer(\n"
            f"  interactions={self.interaction_counter},\n"
            f"  pending_batch={len(self.pending_batch)},\n"
            f"  global_signature={self.global_signature}\n"
            f")"
        )


def create_hybrid_trainer(
    config: Optional[CognitiveTwinConfig] = None,
    hybrid_config: Optional[HybridConfig] = None,
    checkpoint_path: Optional[Path] = None,
) -> HybridCognitiveTwinTrainer:
    """Factory function to create a hybrid trainer.

    Args:
        config: CognitiveTwin configuration.
        hybrid_config: Hybrid training configuration.
        checkpoint_path: Path to load twin checkpoint from.

    Returns:
        Configured HybridCognitiveTwinTrainer.
    """
    from cognitive_twin.framework.twin import create_cognitive_twin

    if config is None:
        config = CognitiveTwinConfig.balanced()

    if hybrid_config is None:
        hybrid_config = HybridConfig()

    # Create or load twin
    twin = create_cognitive_twin(config)

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        twin.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded twin from {checkpoint_path}")

    return HybridCognitiveTwinTrainer(twin=twin, config=hybrid_config)

