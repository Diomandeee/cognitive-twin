"""Reasoning Pattern Encoder for CognitiveTwin.

This module encodes how the user reasons about problems by learning
the attention patterns that produce their responses. Uses inverse attention
to infer: "Given response R, what context C was attended to?"

Key insight: R ≈ sum(alpha_i * C_i) where alpha are learned attention weights.

The encoder captures:
    - What the user focuses on when responding
    - How attention varies with trajectory position
    - Recurring reasoning patterns across conversations
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cognitive_twin.framework.config import ReasoningEncoderConfig, AttentionMode
from cognitive_twin._compat import (
    TrajectoryCoordinate5D,
    InverseAttentionMechanism as InverseAttention,
    InverseAttentionConfig,
    CoordinateBias,
)

# MultiScaleInverseAttention is only available with rag_plusplus
try:
    from rag_plusplus.ml.attention.inverse import MultiScaleInverseAttention
except ImportError:
    MultiScaleInverseAttention = None  # type: ignore

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of reasoning patterns detected."""

    ANALYTICAL = "analytical"  # Breaking down complex problems
    CREATIVE = "creative"  # Novel connections and ideas
    CORRECTIVE = "corrective"  # Error correction and refinement
    EXPLORATORY = "exploratory"  # Branching and exploration
    CONSOLIDATIVE = "consolidative"  # Synthesis and summarization
    METHODOLOGICAL = "methodological"  # Systematic approaches
    REFERENTIAL = "referential"  # Building on prior knowledge


@dataclass
class ReasoningPattern:
    """A detected reasoning pattern.

    Attributes:
        pattern_type: Type of reasoning pattern.
        attention_weights: Inferred attention distribution over context.
        attention_entropy: Entropy of attention (focused vs diffuse).
        coordinate: Trajectory position where pattern occurred.
        context_indices: Which context elements were attended.
        strength: Pattern strength [0, 1].
        embedding: Pattern embedding for storage/retrieval.
        metadata: Additional pattern metadata.
    """

    pattern_type: PatternType
    attention_weights: Tensor
    attention_entropy: float
    coordinate: Optional[TrajectoryCoordinate5D]
    context_indices: List[int]
    strength: float
    embedding: Optional[Tensor] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def is_focused(self) -> bool:
        """Whether attention is focused (low entropy)."""
        return self.attention_entropy < 1.0

    @property
    def top_attended(self) -> List[int]:
        """Indices of top-5 attended context elements."""
        if self.attention_weights.dim() == 1:
            values, indices = torch.topk(self.attention_weights, min(5, len(self.attention_weights)))
            return indices.tolist()
        return self.context_indices[:5]


class CoordinateAwareAttention(nn.Module):
    """Attention biased by trajectory coordinates.

    Combines semantic attention with spatial proximity in trajectory space.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        coord_weight: float = 0.3,
        coord_weights: Tuple[float, float, float, float, float] = (0.25, 0.15, 0.30, 0.20, 0.10),
    ) -> None:
        """Initialize coordinate-aware attention.

        Args:
            embed_dim: Embedding dimension.
            hidden_dim: Hidden projection dimension.
            coord_weight: Weight for coordinate bias.
            coord_weights: DLM weights for coordinate distance.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.coord_weight = coord_weight
        self.coord_weights = coord_weights

        # Coordinate embedding (5D -> hidden_dim)
        self.coord_embed = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Combine semantic and coordinate attention
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_embed: Tensor,
        context_embed: Tensor,
        query_coord: Optional[Tensor] = None,
        context_coords: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute coordinate-biased attention scores.

        Args:
            query_embed: Query embeddings [batch, query_len, embed_dim].
            context_embed: Context embeddings [batch, context_len, embed_dim].
            query_coord: Query coordinates [batch, query_len, 5].
            context_coords: Context coordinates [batch, context_len, 5].

        Returns:
            Attention bias [batch, query_len, context_len].
        """
        batch_size, query_len, _ = query_embed.shape
        _, context_len, _ = context_embed.shape

        # Base semantic attention
        semantic_scores = torch.bmm(query_embed, context_embed.transpose(1, 2))
        semantic_scores = semantic_scores / math.sqrt(self.embed_dim)

        if query_coord is None or context_coords is None:
            return semantic_scores

        # Coordinate-based bias
        # Expand for pairwise: [batch, query, 1, 5] - [batch, 1, context, 5]
        query_coord_exp = query_coord.unsqueeze(2)  # [batch, query, 1, 5]
        context_coord_exp = context_coords.unsqueeze(1)  # [batch, 1, context, 5]

        # Compute coordinate distance
        coord_diff = query_coord_exp - context_coord_exp  # [batch, query, context, 5]

        # Embed coordinate differences
        coord_features = self.coord_embed(coord_diff)  # [batch, query, context, hidden]

        # Expand semantic for combination
        query_exp = query_embed.unsqueeze(2).expand(-1, -1, context_len, -1)
        context_exp = context_embed.unsqueeze(1).expand(-1, query_len, -1, -1)
        semantic_cat = torch.cat([query_exp, context_exp], dim=-1)

        # Project semantic to hidden
        semantic_proj = nn.Linear(self.embed_dim * 2, self.hidden_dim).to(query_embed.device)(semantic_cat)

        # Combine semantic and coordinate
        combined = torch.cat([semantic_proj, coord_features], dim=-1)
        coord_bias = self.combine(combined).squeeze(-1)  # [batch, query, context]

        # Blend semantic and coordinate-biased scores
        final_scores = (1 - self.coord_weight) * semantic_scores + self.coord_weight * coord_bias

        return final_scores


class ReasoningPatternEncoder(nn.Module):
    """Encodes user reasoning patterns via inverse attention.

    This module learns to infer what the user attended to when generating
    each response, capturing their unique reasoning style.

    Attributes:
        config: Encoder configuration.
        inverse_attention: Multi-scale inverse attention mechanism.
        pattern_classifier: Classifies detected patterns.
        pattern_embedder: Creates pattern embeddings for storage.
    """

    def __init__(self, config: ReasoningEncoderConfig) -> None:
        """Initialize reasoning pattern encoder.

        Args:
            config: Encoder configuration.
        """
        super().__init__()
        self.config = config

        # Inverse attention for learning what was attended
        inv_attn_config = InverseAttentionConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_attention_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            temperature=config.temperature,
            use_gumbel=config.use_gumbel,
            sparsity_regularization=config.sparsity_weight,
        )
        self.inverse_attention = InverseAttention(inv_attn_config)

        # Coordinate-aware attention bias
        if config.use_coordinate_bias:
            self.coord_attention = CoordinateAwareAttention(
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                coord_weight=config.coordinate_weight,
                coord_weights=config.coord_weights,
            )
        else:
            self.coord_attention = None

        # Pattern type classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(config.embed_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, len(PatternType)),
        )

        # Pattern embedder for memory storage
        self.pattern_embedder = nn.Sequential(
            nn.Linear(config.embed_dim * 2 + 5, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

        # Attention aggregation for pattern detection
        self.attention_aggregator = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        logger.info(
            f"Initialized ReasoningPatternEncoder: embed_dim={config.embed_dim}, "
            f"heads={config.num_attention_heads}, mode={config.attention_mode}"
        )

    def forward(
        self,
        response: Tensor,
        context: Tensor,
        response_coords: Optional[Tensor] = None,
        context_coords: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode reasoning patterns from response and context.

        Args:
            response: Response embeddings [batch, response_len, embed_dim].
            context: Context embeddings [batch, context_len, embed_dim].
            response_coords: Response coordinates [batch, response_len, 5].
            context_coords: Context coordinates [batch, context_len, 5].
            mask: Context mask [batch, context_len].

        Returns:
            Tuple of:
                - Inferred attention weights [batch, response_len, context_len]
                - Pattern type logits [batch, response_len, num_types]
                - Pattern embeddings [batch, response_len, embed_dim]
        """
        batch_size, response_len, _ = response.shape
        _, context_len, _ = context.shape

        # Get inverse attention weights
        attention_weights, reconstructed = self.inverse_attention(
            response, context, mask=mask, return_reconstruction=True
        )

        # Apply coordinate bias if available
        if self.coord_attention is not None and response_coords is not None:
            coord_bias = self.coord_attention(
                response, context, response_coords, context_coords
            )
            # Soft blend with inverse attention
            attention_weights = 0.7 * attention_weights + 0.3 * F.softmax(coord_bias, dim=-1)
            attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        # Aggregate attended context for pattern classification
        attended_context = torch.bmm(attention_weights, context)  # [batch, response, embed]
        aggregated = self.attention_aggregator(attended_context)

        # Classify pattern type
        classifier_input = torch.cat([response, aggregated], dim=-1)
        pattern_logits = self.pattern_classifier(classifier_input)

        # Create pattern embeddings
        # Include response, attended context, and coordinates
        if response_coords is not None:
            embedder_input = torch.cat([
                response,
                attended_context,
                response_coords,
            ], dim=-1)
        else:
            # Pad with zeros if no coordinates
            zero_coords = torch.zeros(batch_size, response_len, 5, device=response.device)
            embedder_input = torch.cat([
                response,
                attended_context,
                zero_coords,
            ], dim=-1)

        pattern_embeddings = self.pattern_embedder(embedder_input)

        return attention_weights, pattern_logits, pattern_embeddings

    def extract_patterns(
        self,
        response: Tensor,
        context: Tensor,
        response_coords: Optional[List[TrajectoryCoordinate5D]] = None,
        context_coords: Optional[List[TrajectoryCoordinate5D]] = None,
        threshold: float = 0.5,
    ) -> List[ReasoningPattern]:
        """Extract reasoning patterns from a response.

        Higher-level API that returns structured ReasoningPattern objects.

        Args:
            response: Response embeddings [1, response_len, embed_dim].
            context: Context embeddings [1, context_len, embed_dim].
            response_coords: List of response coordinates.
            context_coords: List of context coordinates.
            threshold: Minimum pattern strength for inclusion.

        Returns:
            List of detected ReasoningPattern objects.
        """
        self.eval()
        with torch.no_grad():
            # Convert coordinates to tensors if provided
            resp_coord_tensor = None
            ctx_coord_tensor = None

            if response_coords is not None:
                resp_coord_tensor = torch.tensor(
                    [[c.as_tuple for c in response_coords]],
                    device=response.device,
                    dtype=response.dtype,
                )

            if context_coords is not None:
                ctx_coord_tensor = torch.tensor(
                    [[c.as_tuple for c in context_coords]],
                    device=context.device,
                    dtype=context.dtype,
                )

            # Forward pass
            attention_weights, pattern_logits, pattern_embeddings = self.forward(
                response, context, resp_coord_tensor, ctx_coord_tensor
            )

            # Process each response position
            patterns = []
            attention_weights = attention_weights[0]  # Remove batch dim
            pattern_logits = pattern_logits[0]
            pattern_embeddings = pattern_embeddings[0]

            for i in range(attention_weights.shape[0]):
                attn = attention_weights[i]
                logits = pattern_logits[i]
                embedding = pattern_embeddings[i]

                # Get pattern type
                pattern_probs = F.softmax(logits, dim=-1)
                pattern_type_idx = pattern_probs.argmax().item()
                pattern_strength = pattern_probs[pattern_type_idx].item()

                if pattern_strength < threshold:
                    continue

                # Compute attention entropy
                entropy = -(attn * (attn + 1e-10).log()).sum().item()

                # Get attended context indices (above mean attention)
                mean_attn = attn.mean().item()
                context_indices = (attn > mean_attn).nonzero().squeeze(-1).tolist()
                if isinstance(context_indices, int):
                    context_indices = [context_indices]

                pattern = ReasoningPattern(
                    pattern_type=list(PatternType)[pattern_type_idx],
                    attention_weights=attn,
                    attention_entropy=entropy,
                    coordinate=response_coords[i] if response_coords else None,
                    context_indices=context_indices,
                    strength=pattern_strength,
                    embedding=embedding,
                    metadata={
                        "position": i,
                        "all_probs": pattern_probs.tolist(),
                    },
                )
                patterns.append(pattern)

            return patterns

    def compute_loss(
        self,
        response: Tensor,
        context: Tensor,
        target_patterns: Optional[Tensor] = None,
        response_coords: Optional[Tensor] = None,
        context_coords: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute training loss for the encoder.

        Args:
            response: Response embeddings.
            context: Context embeddings.
            target_patterns: Target pattern labels (if supervised).
            response_coords: Response coordinates.
            context_coords: Context coordinates.
            mask: Context mask.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        # Forward pass
        attention_weights, pattern_logits, pattern_embeddings = self.forward(
            response, context, response_coords, context_coords, mask
        )

        losses = {}

        # Reconstruction loss (from inverse attention)
        recon_loss = self.inverse_attention.reconstruction_loss(response, context, mask)
        losses["reconstruction"] = recon_loss

        # Pattern classification loss (if supervised)
        if target_patterns is not None:
            pattern_loss = F.cross_entropy(
                pattern_logits.view(-1, len(PatternType)),
                target_patterns.view(-1),
            )
            losses["pattern_classification"] = pattern_loss

        # Attention sparsity loss (encourage focused attention)
        entropy = -(attention_weights * (attention_weights + 1e-10).log()).sum(dim=-1).mean()
        sparsity_loss = self.config.sparsity_weight * entropy
        losses["sparsity"] = sparsity_loss

        # Pattern embedding consistency (similar patterns should have similar embeddings)
        if pattern_embeddings.shape[0] > 1:
            # Compute pairwise similarity
            norm_embeddings = F.normalize(pattern_embeddings.view(-1, self.config.embed_dim), dim=-1)
            similarity = torch.mm(norm_embeddings, norm_embeddings.t())

            # Same pattern types should be similar
            if target_patterns is not None:
                flat_patterns = target_patterns.view(-1)
                same_pattern = (flat_patterns.unsqueeze(0) == flat_patterns.unsqueeze(1)).float()
                consistency_loss = F.mse_loss(similarity, same_pattern)
                losses["consistency"] = 0.1 * consistency_loss

        # Total loss
        total_loss = sum(losses.values())

        return total_loss, losses


def create_reasoning_encoder(
    embed_dim: int = 768,
    num_heads: int = 4,
    **kwargs,
) -> ReasoningPatternEncoder:
    """Factory function to create reasoning encoder.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        **kwargs: Additional config arguments.

    Returns:
        ReasoningPatternEncoder instance.
    """
    config = ReasoningEncoderConfig(
        embed_dim=embed_dim,
        num_attention_heads=num_heads,
        **kwargs,
    )
    return ReasoningPatternEncoder(config)
