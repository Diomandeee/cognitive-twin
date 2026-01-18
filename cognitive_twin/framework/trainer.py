"""CognitiveTwin Trainer.

Training pipeline for learning user reasoning patterns from conversation history.
Leverages Supabase data layer with 100K+ embedded turns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, Dataset

from cognitive_twin.framework.config import CognitiveTwinConfig
from cognitive_twin.framework.twin import CognitiveTwin, TwinMode
from cognitive_twin.framework.reasoning_encoder import PatternType, ReasoningPattern
from cognitive_twin.framework.style_projector import StyleEmbedding
from cognitive_twin._compat import TrajectoryCoordinate5D

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from a training step or epoch."""

    total_loss: float = 0.0
    pattern_loss: float = 0.0
    style_loss: float = 0.0
    reconstruction_loss: float = 0.0
    contrastive_loss: float = 0.0
    kl_loss: float = 0.0
    orthogonality_loss: float = 0.0
    num_patterns_extracted: int = 0
    num_samples: int = 0
    step_time: float = 0.0

    def __add__(self, other: "TrainingMetrics") -> "TrainingMetrics":
        """Accumulate metrics."""
        return TrainingMetrics(
            total_loss=self.total_loss + other.total_loss,
            pattern_loss=self.pattern_loss + other.pattern_loss,
            style_loss=self.style_loss + other.style_loss,
            reconstruction_loss=self.reconstruction_loss + other.reconstruction_loss,
            contrastive_loss=self.contrastive_loss + other.contrastive_loss,
            kl_loss=self.kl_loss + other.kl_loss,
            orthogonality_loss=self.orthogonality_loss + other.orthogonality_loss,
            num_patterns_extracted=self.num_patterns_extracted + other.num_patterns_extracted,
            num_samples=self.num_samples + other.num_samples,
            step_time=self.step_time + other.step_time,
        )

    def average(self) -> "TrainingMetrics":
        """Return averaged metrics."""
        if self.num_samples == 0:
            return self
        n = self.num_samples
        return TrainingMetrics(
            total_loss=self.total_loss / n,
            pattern_loss=self.pattern_loss / n,
            style_loss=self.style_loss / n,
            reconstruction_loss=self.reconstruction_loss / n,
            contrastive_loss=self.contrastive_loss / n,
            kl_loss=self.kl_loss / n,
            orthogonality_loss=self.orthogonality_loss / n,
            num_patterns_extracted=self.num_patterns_extracted,
            num_samples=n,
            step_time=self.step_time,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "loss/total": self.total_loss,
            "loss/pattern": self.pattern_loss,
            "loss/style": self.style_loss,
            "loss/reconstruction": self.reconstruction_loss,
            "loss/contrastive": self.contrastive_loss,
            "loss/kl": self.kl_loss,
            "loss/orthogonality": self.orthogonality_loss,
            "patterns/extracted": self.num_patterns_extracted,
            "training/samples": self.num_samples,
            "training/step_time_ms": self.step_time * 1000,
        }


@dataclass
class TrainingState:
    """Current training state for checkpointing."""

    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    patterns_learned: int = 0
    conversations_processed: int = 0
    training_started: Optional[datetime] = None
    last_checkpoint: Optional[datetime] = None


@dataclass
class ConversationBatch:
    """A batch of conversation data for training."""

    # Response embeddings [batch, embed_dim]
    response_embeddings: Tensor

    # Context embeddings [batch, max_context_len, embed_dim]
    context_embeddings: Tensor

    # Context mask [batch, max_context_len]
    context_mask: Tensor

    # Response coordinates [batch, 5]
    response_coords: Tensor

    # Context coordinates [batch, max_context_len, 5]
    context_coords: Tensor

    # Conversation IDs for grouping
    conversation_ids: List[str]

    # Turn indices within conversations
    turn_indices: List[int]

    # Roles (user/assistant)
    roles: List[str]

    # Optional: content text for analysis
    content: Optional[List[str]] = None

    # Multi-source temporal features for evolution learning
    absolute_temporal: Optional[Tensor] = None  # [batch, 1] position across all data
    era_onehot: Optional[Tensor] = None  # [batch, 3] one-hot era (early, middle, recent)
    data_sources: Optional[List[str]] = None  # Source identifiers

    def to(self, device: torch.device) -> "ConversationBatch":
        """Move batch to device."""
        return ConversationBatch(
            response_embeddings=self.response_embeddings.to(device),
            context_embeddings=self.context_embeddings.to(device),
            context_mask=self.context_mask.to(device),
            response_coords=self.response_coords.to(device),
            context_coords=self.context_coords.to(device),
            conversation_ids=self.conversation_ids,
            turn_indices=self.turn_indices,
            roles=self.roles,
            content=self.content,
            absolute_temporal=self.absolute_temporal.to(device) if self.absolute_temporal is not None else None,
            era_onehot=self.era_onehot.to(device) if self.era_onehot is not None else None,
            data_sources=self.data_sources,
        )


class CognitiveTwinDataset(Dataset):
    """Dataset for CognitiveTwin training from Supabase data."""

    def __init__(
        self,
        turns: List[Dict[str, Any]],
        embed_dim: int = 768,
        max_context_length: int = 32,
        min_context_length: int = 2,
    ) -> None:
        """Initialize dataset.

        Args:
            turns: List of turn records from Supabase.
            embed_dim: Embedding dimension.
            max_context_length: Maximum context turns to include.
            min_context_length: Minimum context required.
        """
        self.embed_dim = embed_dim
        self.max_context_length = max_context_length
        self.min_context_length = min_context_length

        # Group turns by conversation
        self.conversations: Dict[str, List[Dict]] = {}
        for turn in turns:
            conv_id = turn.get("conversation_id", "unknown")
            if conv_id not in self.conversations:
                self.conversations[conv_id] = []
            self.conversations[conv_id].append(turn)

        # Sort turns within each conversation by turn_index
        for conv_id in self.conversations:
            self.conversations[conv_id].sort(
                key=lambda t: t.get("turn_index", 0)
            )

        # Build training samples (response, context) pairs
        self.samples: List[Tuple[str, int]] = []  # (conv_id, turn_idx)
        for conv_id, conv_turns in self.conversations.items():
            # Each turn after min_context can be a response
            for i in range(self.min_context_length, len(conv_turns)):
                # Only use turns that have embeddings
                if conv_turns[i].get("embedding") is not None:
                    self.samples.append((conv_id, i))

        logger.info(
            f"Created dataset with {len(self.samples)} samples "
            f"from {len(self.conversations)} conversations"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample."""
        conv_id, turn_idx = self.samples[idx]
        conv_turns = self.conversations[conv_id]

        # Response is the current turn
        response_turn = conv_turns[turn_idx]

        # Context is previous turns (up to max_context_length)
        context_start = max(0, turn_idx - self.max_context_length)
        context_turns = conv_turns[context_start:turn_idx]

        # Extract embeddings
        response_embedding = self._get_embedding(response_turn)
        context_embeddings = [
            self._get_embedding(t) for t in context_turns
            if t.get("embedding") is not None
        ]

        # Extract coordinates
        response_coord = self._get_coordinate(response_turn)
        context_coords = [self._get_coordinate(t) for t in context_turns]

        # Extract temporal metadata (for multi-source evolution learning)
        temporal_meta = self._get_temporal_metadata(response_turn)

        return {
            "response_embedding": response_embedding,
            "context_embeddings": context_embeddings,
            "response_coord": response_coord,
            "context_coords": context_coords,
            "conversation_id": conv_id,
            "turn_index": turn_idx,
            "role": response_turn.get("role", "assistant"),
            "content": response_turn.get("content", ""),
            # Temporal features
            "absolute_temporal": temporal_meta["absolute_temporal"],
            "era": temporal_meta["era"],
            "data_source": temporal_meta["data_source"],
        }

    def _get_embedding(self, turn: Dict) -> Tensor:
        """Extract embedding from turn record."""
        embedding = turn.get("embedding")
        if embedding is None:
            return torch.zeros(self.embed_dim)
        if isinstance(embedding, list):
            return torch.tensor(embedding, dtype=torch.float32)
        return embedding

    def _get_coordinate(self, turn: Dict) -> Tensor:
        """Extract 5D coordinate from turn record."""
        # Try to get from trajectory columns
        depth = turn.get("depth", 0)
        max_depth = turn.get("max_depth", 1)
        sibling_order = turn.get("sibling_order", 0)
        num_siblings = turn.get("num_siblings", 1)
        homogeneity = turn.get("homogeneity", 0.5)
        temporal = turn.get("temporal", 0.5)
        complexity = turn.get("complexity", 1)

        return torch.tensor([
            depth / max(max_depth, 1),  # x: normalized depth
            sibling_order / max(num_siblings, 1),  # y: sibling position
            homogeneity,  # z: semantic similarity
            temporal,  # t: normalized time
            float(complexity),  # n: content parts
        ], dtype=torch.float32)

    def _get_temporal_metadata(self, turn: Dict) -> Dict[str, Any]:
        """Extract temporal metadata for multi-source evolution learning.

        Args:
            turn: Turn record from Supabase.

        Returns:
            Dictionary with absolute_temporal, era, and data_source.
        """
        # Get metadata dict (may be nested from Supabase JSONB)
        metadata = turn.get("metadata", {}) or {}

        # Try direct columns first, then fall back to metadata
        absolute_temporal = turn.get("absolute_temporal")
        if absolute_temporal is None:
            absolute_temporal = metadata.get("absolute_temporal", 0.0)

        era = turn.get("era")
        if era is None:
            era = metadata.get("era")

        data_source = turn.get("data_source")
        if data_source is None:
            data_source = metadata.get("data_source")

        return {
            "absolute_temporal": float(absolute_temporal) if absolute_temporal is not None else 0.0,
            "era": era,
            "data_source": data_source,
        }


def collate_conversation_batch(
    samples: List[Dict[str, Any]],
    embed_dim: int = 768,
    max_context_length: int = 32,
) -> ConversationBatch:
    """Collate samples into a batch with padding.

    Args:
        samples: List of samples from dataset.
        embed_dim: Embedding dimension.
        max_context_length: Maximum context length for padding.

    Returns:
        Collated ConversationBatch.
    """
    batch_size = len(samples)

    # Find max context length in this batch
    max_ctx_len = min(
        max(len(s["context_embeddings"]) for s in samples),
        max_context_length,
    )
    max_ctx_len = max(max_ctx_len, 1)  # At least 1

    # Initialize tensors
    response_embeddings = torch.zeros(batch_size, embed_dim)
    context_embeddings = torch.zeros(batch_size, max_ctx_len, embed_dim)
    context_mask = torch.zeros(batch_size, max_ctx_len, dtype=torch.bool)
    response_coords = torch.zeros(batch_size, 5)
    context_coords = torch.zeros(batch_size, max_ctx_len, 5)

    # Temporal features tensors
    absolute_temporal = torch.zeros(batch_size, 1)
    era_onehot = torch.zeros(batch_size, 3)  # [early, middle, recent]

    conversation_ids = []
    turn_indices = []
    roles = []
    content = []
    data_sources = []

    # Era mapping for one-hot encoding
    era_to_idx = {"early": 0, "middle": 1, "recent": 2}

    for i, sample in enumerate(samples):
        # Response embedding
        response_embeddings[i] = sample["response_embedding"]

        # Context embeddings with padding
        ctx_embs = sample["context_embeddings"]
        ctx_len = min(len(ctx_embs), max_ctx_len)
        for j in range(ctx_len):
            context_embeddings[i, j] = ctx_embs[-(ctx_len - j)]  # Most recent last
            context_mask[i, j] = True

        # Coordinates
        response_coords[i] = sample["response_coord"]
        ctx_crds = sample["context_coords"]
        for j in range(min(len(ctx_crds), max_ctx_len)):
            context_coords[i, j] = ctx_crds[-(min(len(ctx_crds), max_ctx_len) - j)]

        # Temporal features
        abs_temp = sample.get("absolute_temporal", 0.0)
        absolute_temporal[i, 0] = abs_temp if abs_temp is not None else 0.0

        era = sample.get("era")
        if era and era in era_to_idx:
            era_onehot[i, era_to_idx[era]] = 1.0

        # Metadata
        conversation_ids.append(sample["conversation_id"])
        turn_indices.append(sample["turn_index"])
        roles.append(sample["role"])
        content.append(sample.get("content", ""))
        data_sources.append(sample.get("data_source"))

    # Check if any temporal data is present
    has_temporal = (absolute_temporal.abs().sum() > 0) or (era_onehot.sum() > 0)

    return ConversationBatch(
        response_embeddings=response_embeddings,
        context_embeddings=context_embeddings,
        context_mask=context_mask,
        response_coords=response_coords,
        context_coords=context_coords,
        conversation_ids=conversation_ids,
        turn_indices=turn_indices,
        roles=roles,
        content=content,
        absolute_temporal=absolute_temporal if has_temporal else None,
        era_onehot=era_onehot if has_temporal else None,
        data_sources=data_sources if has_temporal else None,
    )


class CognitiveTwinLoss(nn.Module):
    """Combined loss function for CognitiveTwin training.

    Combines:
    - Pattern classification loss
    - Style reconstruction/KL loss
    - Contrastive loss for embeddings
    - Orthogonality loss for style components
    - Temporal consistency loss (temporally close samples should have similar styles)
    - Temporal discrimination loss (distinguish between eras)
    """

    def __init__(
        self,
        pattern_weight: float = 0.3,
        style_weight: float = 0.3,
        reconstruction_weight: float = 0.2,
        contrastive_weight: float = 0.1,
        kl_weight: float = 0.05,
        orthogonality_weight: float = 0.05,
        contrastive_temperature: float = 0.07,
        temporal_consistency_weight: float = 0.1,
        temporal_discrimination_weight: float = 0.05,
    ) -> None:
        """Initialize loss function.

        Args:
            pattern_weight: Weight for pattern classification loss.
            style_weight: Weight for style consistency loss.
            reconstruction_weight: Weight for embedding reconstruction.
            contrastive_weight: Weight for contrastive loss.
            kl_weight: Weight for KL divergence loss.
            orthogonality_weight: Weight for component orthogonality.
            contrastive_temperature: Temperature for contrastive loss.
            temporal_consistency_weight: Weight for temporal smoothness loss.
            temporal_discrimination_weight: Weight for era discrimination loss.
        """
        super().__init__()
        self.pattern_weight = pattern_weight
        self.style_weight = style_weight
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.kl_weight = kl_weight
        self.orthogonality_weight = orthogonality_weight
        self.temperature = contrastive_temperature
        self.temporal_consistency_weight = temporal_consistency_weight
        self.temporal_discrimination_weight = temporal_discrimination_weight

    def forward(
        self,
        pattern_logits: Tensor,
        pattern_targets: Optional[Tensor],
        style_embedding: StyleEmbedding,
        reconstructed_embedding: Optional[Tensor],
        original_embedding: Tensor,
        conversation_labels: Optional[Tensor] = None,
        orthogonality_loss: Optional[Tensor] = None,
        absolute_temporal: Optional[Tensor] = None,
        era_onehot: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute combined loss.

        Args:
            pattern_logits: Predicted pattern logits [batch, num_patterns].
            pattern_targets: Target pattern labels [batch] (optional).
            style_embedding: StyleEmbedding with optional variational params.
            reconstructed_embedding: Reconstructed embedding (optional).
            original_embedding: Original input embedding.
            conversation_labels: Labels for contrastive loss (optional).
            orthogonality_loss: Pre-computed orthogonality loss (optional).
            absolute_temporal: Absolute temporal positions [batch, 1] (optional).
            era_onehot: Era one-hot encoding [batch, 3] (optional).

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=original_embedding.device)

        # Pattern classification loss
        if pattern_targets is not None and pattern_logits is not None:
            pattern_loss = F.cross_entropy(pattern_logits, pattern_targets)
            losses["pattern"] = pattern_loss
            total_loss = total_loss + self.pattern_weight * pattern_loss

        # Style consistency loss (MSE between consecutive styles)
        if style_embedding.embedding.shape[0] > 1:
            style_consistency = F.mse_loss(
                style_embedding.embedding[:-1],
                style_embedding.embedding[1:],
            )
            losses["style"] = style_consistency
            total_loss = total_loss + self.style_weight * style_consistency

        # Reconstruction loss
        if reconstructed_embedding is not None:
            reconstruction_loss = F.mse_loss(
                reconstructed_embedding,
                original_embedding,
            )
            losses["reconstruction"] = reconstruction_loss
            total_loss = total_loss + self.reconstruction_weight * reconstruction_loss

        # KL divergence loss (for variational style)
        if style_embedding.is_variational:
            kl_loss = -0.5 * torch.sum(
                1 + style_embedding.log_var
                - style_embedding.mean.pow(2)
                - style_embedding.log_var.exp(),
                dim=-1,
            ).mean()
            losses["kl"] = kl_loss
            total_loss = total_loss + self.kl_weight * kl_loss

        # Contrastive loss (group by conversation)
        if conversation_labels is not None:
            contrastive_loss = self._contrastive_loss(
                style_embedding.embedding,
                conversation_labels,
            )
            losses["contrastive"] = contrastive_loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        # Orthogonality loss
        if orthogonality_loss is not None:
            losses["orthogonality"] = orthogonality_loss
            total_loss = total_loss + self.orthogonality_weight * orthogonality_loss

        # Temporal consistency loss (temporally close samples should have similar styles)
        if absolute_temporal is not None and self.temporal_consistency_weight > 0:
            temporal_consistency = self._temporal_consistency_loss(
                style_embedding.embedding,
                absolute_temporal,
            )
            losses["temporal_consistency"] = temporal_consistency
            total_loss = total_loss + self.temporal_consistency_weight * temporal_consistency

        # Temporal discrimination loss (distinguish between eras)
        if era_onehot is not None and self.temporal_discrimination_weight > 0:
            temporal_discrimination = self._temporal_discrimination_loss(
                style_embedding.embedding,
                era_onehot,
            )
            losses["temporal_discrimination"] = temporal_discrimination
            total_loss = total_loss + self.temporal_discrimination_weight * temporal_discrimination

        return total_loss, losses

    def _contrastive_loss(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute contrastive loss for style embeddings.

        Args:
            embeddings: Style embeddings [batch, style_dim].
            labels: Conversation labels for positive pairs [batch].

        Returns:
            Contrastive loss.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create positive pair mask
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()

        # Remove self-similarity
        self_mask = torch.eye(embeddings.shape[0], device=embeddings.device)
        positive_mask = positive_mask - self_mask

        # Compute loss
        exp_sim = torch.exp(similarity)
        exp_sim = exp_sim * (1 - self_mask)

        pos_sim = (exp_sim * positive_mask).sum(dim=-1)
        all_sim = exp_sim.sum(dim=-1)

        # Avoid log(0)
        loss = -torch.log(pos_sim / (all_sim + 1e-10) + 1e-10)

        # Only compute for samples with positive pairs
        valid_mask = positive_mask.sum(dim=-1) > 0
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        return torch.tensor(0.0, device=embeddings.device)

    def _temporal_consistency_loss(
        self,
        embeddings: Tensor,
        temporal_positions: Tensor,
        sigma: float = 0.1,
    ) -> Tensor:
        """Compute temporal consistency loss.

        Encourages temporally close samples to have similar style embeddings.
        Uses a Gaussian kernel to weight style differences by temporal distance.

        Args:
            embeddings: Style embeddings [batch, style_dim].
            temporal_positions: Absolute temporal positions [batch, 1].
            sigma: Bandwidth for Gaussian kernel (controls how "close" is defined).

        Returns:
            Temporal consistency loss.
        """
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        norm_emb = F.normalize(embeddings, dim=-1)

        # Compute pairwise temporal distances [batch, batch]
        temporal_diff = temporal_positions - temporal_positions.t()
        temporal_dist = temporal_diff.pow(2)

        # Gaussian kernel weights: closer in time = higher weight
        temporal_weights = torch.exp(-temporal_dist / (2 * sigma ** 2))

        # Remove self-comparisons
        self_mask = torch.eye(embeddings.shape[0], device=embeddings.device)
        temporal_weights = temporal_weights * (1 - self_mask)

        # Compute pairwise style distances (1 - cosine similarity)
        style_similarity = torch.mm(norm_emb, norm_emb.t())
        style_dist = 1 - style_similarity

        # Weighted average: penalize large style differences for temporally close samples
        weighted_dist = (style_dist * temporal_weights).sum()
        weight_sum = temporal_weights.sum()

        if weight_sum > 0:
            return weighted_dist / weight_sum
        return torch.tensor(0.0, device=embeddings.device)

    def _temporal_discrimination_loss(
        self,
        embeddings: Tensor,
        era_onehot: Tensor,
    ) -> Tensor:
        """Compute temporal discrimination loss.

        Encourages the model to learn era-specific style patterns.
        Uses contrastive-like loss where samples from same era are pulled together.

        Args:
            embeddings: Style embeddings [batch, style_dim].
            era_onehot: Era one-hot encoding [batch, 3].

        Returns:
            Temporal discrimination loss.
        """
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Check if we have era labels (non-zero one-hot)
        has_era = era_onehot.sum(dim=-1) > 0
        if has_era.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Convert one-hot to labels
        era_labels = era_onehot.argmax(dim=-1)

        # Normalize embeddings
        norm_emb = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        similarity = torch.mm(norm_emb, norm_emb.t()) / self.temperature

        # Create same-era mask
        era_labels_expanded = era_labels.view(-1, 1)
        same_era_mask = (era_labels_expanded == era_labels_expanded.t()).float()

        # Only consider samples with valid era labels
        valid_mask = has_era.float().view(-1, 1) * has_era.float().view(1, -1)
        same_era_mask = same_era_mask * valid_mask

        # Remove self-similarity
        self_mask = torch.eye(embeddings.shape[0], device=embeddings.device)
        same_era_mask = same_era_mask - self_mask * same_era_mask

        # Compute cross-era mask (samples from different eras)
        diff_era_mask = valid_mask - same_era_mask - (self_mask * valid_mask)
        diff_era_mask = diff_era_mask.clamp(min=0)

        # InfoNCE-style loss: maximize same-era similarity vs different-era
        exp_sim = torch.exp(similarity)

        # Numerator: sum of same-era similarities
        same_era_sim = (exp_sim * same_era_mask).sum(dim=-1)

        # Denominator: sum of all valid similarities (excluding self)
        all_valid_sim = (exp_sim * (valid_mask - self_mask * valid_mask).clamp(min=0)).sum(dim=-1)

        # Avoid log(0)
        loss = -torch.log((same_era_sim + 1e-10) / (all_valid_sim + 1e-10))

        # Only compute for samples with same-era pairs
        has_same_era_pairs = same_era_mask.sum(dim=-1) > 0
        if has_same_era_pairs.sum() > 0:
            return loss[has_same_era_pairs].mean()
        return torch.tensor(0.0, device=embeddings.device)


class CognitiveTwinTrainer:
    """Trainer for CognitiveTwin model.

    Handles:
    - Loading data from Supabase
    - Training loop with mixed precision
    - Checkpointing and resumption
    - Logging and metrics
    """

    def __init__(
        self,
        model: CognitiveTwin,
        config: CognitiveTwinConfig,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: CognitiveTwin model to train.
            config: Training configuration.
            train_data: Training data (turns from Supabase).
            val_data: Optional validation data.
            checkpoint_dir: Directory for checkpoints.
            callbacks: Optional callback functions.
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints/cognitive_twin")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = callbacks or []

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Create datasets
        self.train_dataset = CognitiveTwinDataset(
            turns=train_data,
            embed_dim=config.reasoning_encoder.embed_dim,
            max_context_length=config.reasoning_encoder.max_context_length,
        )
        self.val_dataset = None
        if val_data:
            self.val_dataset = CognitiveTwinDataset(
                turns=val_data,
                embed_dim=config.reasoning_encoder.embed_dim,
                max_context_length=config.reasoning_encoder.max_context_length,
            )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=lambda x: collate_conversation_batch(
                x,
                embed_dim=config.reasoning_encoder.embed_dim,
                max_context_length=config.reasoning_encoder.max_context_length,
            ),
            pin_memory=True,
        )
        self.val_loader = None
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=lambda x: collate_conversation_batch(
                    x,
                    embed_dim=config.reasoning_encoder.embed_dim,
                    max_context_length=config.reasoning_encoder.max_context_length,
                ),
                pin_memory=True,
            )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
        )

        # Loss function
        self.criterion = CognitiveTwinLoss(
            pattern_weight=0.3,
            style_weight=0.3,
            reconstruction_weight=0.2,
            contrastive_weight=0.1,
            kl_weight=0.05,
            orthogonality_weight=0.05,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

        # Training state
        self.state = TrainingState(training_started=datetime.now())

        logger.info(
            f"Initialized CognitiveTwinTrainer:\n"
            f"  - Device: {self.device}\n"
            f"  - Train samples: {len(self.train_dataset)}\n"
            f"  - Val samples: {len(self.val_dataset) if self.val_dataset else 0}\n"
            f"  - Batch size: {config.batch_size}\n"
            f"  - Epochs: {config.num_epochs}\n"
            f"  - Learning rate: {config.learning_rate}"
        )

    def train(self) -> Dict[str, Any]:
        """Run full training loop.

        Returns:
            Training results and final metrics.
        """
        logger.info("Starting CognitiveTwin training...")

        best_val_loss = float("inf")
        results = {
            "train_metrics": [],
            "val_metrics": [],
            "best_epoch": 0,
        }

        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            logger.info(f"\n{'='*60}\nEpoch {epoch + 1}/{self.config.num_epochs}\n{'='*60}")

            # Training epoch
            train_metrics = self._train_epoch()
            results["train_metrics"].append(train_metrics.to_dict())
            logger.info(f"Train loss: {train_metrics.total_loss:.4f}")

            # Validation epoch
            if self.val_loader:
                val_metrics = self._validate_epoch()
                results["val_metrics"].append(val_metrics.to_dict())
                logger.info(f"Val loss: {val_metrics.total_loss:.4f}")

                # Check for best model
                if val_metrics.total_loss < best_val_loss:
                    best_val_loss = val_metrics.total_loss
                    results["best_epoch"] = epoch
                    self._save_checkpoint("best")
                    logger.info(f"New best model saved (loss: {best_val_loss:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}")

            # Callbacks
            for callback in self.callbacks:
                callback(self, epoch, train_metrics)

        # Final checkpoint
        self._save_checkpoint("final")

        # Summary
        results["total_patterns_learned"] = self.state.patterns_learned
        results["total_conversations"] = self.state.conversations_processed

        logger.info(
            f"\nTraining complete!\n"
            f"  - Best epoch: {results['best_epoch'] + 1}\n"
            f"  - Best val loss: {best_val_loss:.4f}\n"
            f"  - Patterns learned: {self.state.patterns_learned}\n"
            f"  - Conversations processed: {self.state.conversations_processed}"
        )

        return results

    def _train_epoch(self) -> TrainingMetrics:
        """Run one training epoch.

        Returns:
            Aggregated training metrics.
        """
        self.model.train()
        epoch_metrics = TrainingMetrics()

        for batch_idx, batch in enumerate(self.train_loader):
            step_start = time.time()
            batch = batch.to(self.device)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                metrics = self._train_step(batch)

            epoch_metrics = epoch_metrics + metrics
            epoch_metrics.step_time += time.time() - step_start

            # Logging
            if (batch_idx + 1) % self.config.log_every == 0:
                avg = epoch_metrics.average()
                logger.info(
                    f"Step {batch_idx + 1}/{len(self.train_loader)}: "
                    f"loss={avg.total_loss:.4f}, "
                    f"patterns={epoch_metrics.num_patterns_extracted}"
                )

            self.state.global_step += 1

        return epoch_metrics.average()

    def _train_step(self, batch: ConversationBatch) -> TrainingMetrics:
        """Execute single training step.

        Args:
            batch: Training batch.

        Returns:
            Step metrics.
        """
        self.optimizer.zero_grad()

        # Get model outputs
        output = self.model.learn_from_conversation(
            turns=batch.content,
            embeddings=batch.response_embeddings,
            coordinates=batch.response_coords,
            context_embeddings=batch.context_embeddings,
            context_mask=batch.context_mask,
        )

        # Compute losses
        total_loss, loss_components = self.criterion(
            pattern_logits=output.pattern_logits,
            pattern_targets=None,  # Unsupervised for now
            style_embedding=output.style_embedding,
            reconstructed_embedding=output.reconstructed_embeddings,
            original_embedding=batch.response_embeddings,
            conversation_labels=self._get_conversation_labels(batch),
            orthogonality_loss=output.orthogonality_loss,
            # Temporal features for evolution learning
            absolute_temporal=batch.absolute_temporal,
            era_onehot=batch.era_onehot,
        )

        # Backward pass
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.scheduler.step()

        # Update state
        self.state.patterns_learned += len(output.patterns_extracted) if output.patterns_extracted else 0
        self.state.conversations_processed += len(set(batch.conversation_ids))

        # Build metrics
        return TrainingMetrics(
            total_loss=total_loss.item(),
            pattern_loss=loss_components.get("pattern", torch.tensor(0)).item(),
            style_loss=loss_components.get("style", torch.tensor(0)).item(),
            reconstruction_loss=loss_components.get("reconstruction", torch.tensor(0)).item(),
            contrastive_loss=loss_components.get("contrastive", torch.tensor(0)).item(),
            kl_loss=loss_components.get("kl", torch.tensor(0)).item(),
            orthogonality_loss=loss_components.get("orthogonality", torch.tensor(0)).item(),
            num_patterns_extracted=len(output.patterns_extracted) if output.patterns_extracted else 0,
            num_samples=batch.response_embeddings.shape[0],
        )

    def _validate_epoch(self) -> TrainingMetrics:
        """Run validation epoch.

        Returns:
            Validation metrics.
        """
        self.model.eval()
        epoch_metrics = TrainingMetrics()

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                output = self.model.learn_from_conversation(
                    turns=batch.content,
                    embeddings=batch.response_embeddings,
                    coordinates=batch.response_coords,
                    context_embeddings=batch.context_embeddings,
                    context_mask=batch.context_mask,
                )

                total_loss, loss_components = self.criterion(
                    pattern_logits=output.pattern_logits,
                    pattern_targets=None,
                    style_embedding=output.style_embedding,
                    reconstructed_embedding=output.reconstructed_embeddings,
                    original_embedding=batch.response_embeddings,
                    conversation_labels=self._get_conversation_labels(batch),
                    orthogonality_loss=output.orthogonality_loss,
                    # Temporal features for evolution learning
                    absolute_temporal=batch.absolute_temporal,
                    era_onehot=batch.era_onehot,
                )

                epoch_metrics = epoch_metrics + TrainingMetrics(
                    total_loss=total_loss.item(),
                    pattern_loss=loss_components.get("pattern", torch.tensor(0)).item(),
                    style_loss=loss_components.get("style", torch.tensor(0)).item(),
                    reconstruction_loss=loss_components.get("reconstruction", torch.tensor(0)).item(),
                    num_samples=batch.response_embeddings.shape[0],
                )

        return epoch_metrics.average()

    def _get_conversation_labels(self, batch: ConversationBatch) -> Tensor:
        """Convert conversation IDs to integer labels for contrastive loss.

        Args:
            batch: Current batch.

        Returns:
            Integer labels tensor.
        """
        unique_ids = list(set(batch.conversation_ids))
        id_to_label = {cid: i for i, cid in enumerate(unique_ids)}
        labels = [id_to_label[cid] for cid in batch.conversation_ids]
        return torch.tensor(labels, device=self.device)

    def _save_checkpoint(self, name: str) -> Path:
        """Save training checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_loss": self.state.best_loss,
                "patterns_learned": self.state.patterns_learned,
                "conversations_processed": self.state.conversations_processed,
            },
            "config": self.config,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.state.last_checkpoint = datetime.now()

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        state = checkpoint["training_state"]
        self.state.epoch = state["epoch"]
        self.state.global_step = state["global_step"]
        self.state.best_loss = state["best_loss"]
        self.state.patterns_learned = state["patterns_learned"]
        self.state.conversations_processed = state["conversations_processed"]

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Loaded checkpoint from {checkpoint_path}\n"
            f"  - Epoch: {self.state.epoch}\n"
            f"  - Global step: {self.state.global_step}"
        )


async def load_training_data_from_supabase(
    supabase_url: str,
    supabase_key: str,
    limit: Optional[int] = None,
    min_embedding_count: int = 10,
) -> Tuple[List[Dict], List[Dict]]:
    """Load training data from Supabase.

    Args:
        supabase_url: Supabase project URL.
        supabase_key: Supabase anon key.
        limit: Optional limit on conversations.
        min_embedding_count: Minimum turns with embeddings per conversation.

    Returns:
        Tuple of (train_data, val_data).
    """
    from supabase import create_client

    client = create_client(supabase_url, supabase_key)

    # Get conversations with trajectory data
    query = client.table("memory_turns")\
        .select("*, memory_conversations(title, id)")\
        .not_.is_("embedding", "null")

    if limit:
        query = query.limit(limit)

    result = query.execute()
    turns = result.data

    logger.info(f"Loaded {len(turns)} turns from Supabase")

    # Split 90/10 train/val
    split_idx = int(len(turns) * 0.9)
    train_turns = turns[:split_idx]
    val_turns = turns[split_idx:]

    return train_turns, val_turns


def create_trainer_from_config(
    config: CognitiveTwinConfig,
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    checkpoint_dir: Optional[Path] = None,
) -> CognitiveTwinTrainer:
    """Factory function to create trainer from config.

    Args:
        config: CognitiveTwin configuration.
        train_data: Training data.
        val_data: Optional validation data.
        checkpoint_dir: Checkpoint directory.

    Returns:
        Configured CognitiveTwinTrainer.
    """
    from cognitive_twin.framework.twin import create_cognitive_twin

    model = create_cognitive_twin(config)

    return CognitiveTwinTrainer(
        model=model,
        config=config,
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=checkpoint_dir,
    )
