"""Style Projector for CognitiveTwin.

This module projects embeddings into a user-specific style space,
capturing the unique characteristics of how the user communicates:
    - Vocabulary preferences
    - Sentence structure patterns
    - Technical depth tendencies
    - Exploration vs consolidation style

Uses a variational approach to learn a probabilistic style space
that can be sampled for generation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cognitive_twin.framework.config import StyleProjectorConfig

logger = logging.getLogger(__name__)


@dataclass
class StyleEmbedding:
    """A projected style embedding.

    Attributes:
        embedding: The style embedding vector.
        mean: Mean of variational distribution (if variational).
        log_var: Log variance of variational distribution.
        components: Individual style component activations.
        metadata: Additional style metadata.
    """

    embedding: Tensor
    mean: Optional[Tensor] = None
    log_var: Optional[Tensor] = None
    components: Optional[Tensor] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def is_variational(self) -> bool:
        """Whether this is a variational embedding."""
        return self.mean is not None and self.log_var is not None

    def sample(self, temperature: float = 1.0) -> Tensor:
        """Sample from the variational distribution.

        Args:
            temperature: Sampling temperature (higher = more random).

        Returns:
            Sampled style embedding.
        """
        if not self.is_variational:
            return self.embedding

        std = torch.exp(0.5 * self.log_var) * temperature
        eps = torch.randn_like(std)
        return self.mean + eps * std


@dataclass
class StyleSignature:
    """A user's style signature aggregated from many embeddings.

    Attributes:
        mean_style: Average style embedding.
        style_variance: Variance across style embeddings.
        component_weights: Learned weights for style components.
        characteristic_patterns: Most distinctive style patterns.
        num_samples: Number of samples used to compute signature.
    """

    mean_style: Tensor
    style_variance: Tensor
    component_weights: Tensor
    characteristic_patterns: List[Tensor] = field(default_factory=list)
    num_samples: int = 0

    def similarity(self, other: "StyleSignature") -> float:
        """Compute similarity to another style signature.

        Args:
            other: Another style signature.

        Returns:
            Cosine similarity in [-1, 1].
        """
        return F.cosine_similarity(
            self.mean_style.unsqueeze(0),
            other.mean_style.unsqueeze(0),
        ).item()


class StyleComponentBank(nn.Module):
    """Bank of orthogonal style components.

    Learns a set of orthogonal style directions that can be
    combined to represent any user's style.
    """

    def __init__(
        self,
        style_dim: int,
        num_components: int,
    ) -> None:
        """Initialize style component bank.

        Args:
            style_dim: Dimension of style space.
            num_components: Number of style components.
        """
        super().__init__()
        self.style_dim = style_dim
        self.num_components = num_components

        # Learnable component vectors (initialized orthogonally)
        self.components = nn.Parameter(torch.zeros(num_components, style_dim))
        nn.init.orthogonal_(self.components)

        # Component importance weights
        self.component_importance = nn.Parameter(torch.ones(num_components))

    def forward(self, style_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        """Project style embedding onto components.

        Args:
            style_embedding: Style embeddings [batch, style_dim].

        Returns:
            Tuple of:
                - Component activations [batch, num_components]
                - Reconstructed embedding [batch, style_dim]
        """
        # Normalize components for stable projection
        normalized_components = F.normalize(self.components, dim=-1)

        # Project onto components
        activations = torch.mm(style_embedding, normalized_components.t())

        # Apply importance weighting
        weighted_activations = activations * F.softmax(self.component_importance, dim=0)

        # Reconstruct from components
        reconstructed = torch.mm(weighted_activations, normalized_components)

        return activations, reconstructed

    def orthogonality_loss(self) -> Tensor:
        """Compute loss to encourage orthogonal components.

        Returns:
            Orthogonality regularization loss.
        """
        normalized = F.normalize(self.components, dim=-1)
        similarity = torch.mm(normalized, normalized.t())

        # Off-diagonal elements should be 0
        identity = torch.eye(self.num_components, device=similarity.device)
        off_diagonal = similarity - identity

        return (off_diagonal ** 2).mean()


class VariationalStyleEncoder(nn.Module):
    """Variational encoder for style space.

    Learns a probabilistic mapping from content embeddings to
    style space, enabling sampling and interpolation.
    """

    def __init__(
        self,
        embed_dim: int,
        style_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        """Initialize variational encoder.

        Args:
            embed_dim: Input embedding dimension.
            style_dim: Output style dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Mean and variance heads
        self.mean_head = nn.Linear(hidden_dim, style_dim)
        self.log_var_head = nn.Linear(hidden_dim, style_dim)

    def forward(
        self,
        x: Tensor,
        sample: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode input to variational style space.

        Args:
            x: Input embeddings [batch, embed_dim].
            sample: Whether to sample from distribution.

        Returns:
            Tuple of (sample/mean, mean, log_var).
        """
        hidden = self.encoder(x)
        mean = self.mean_head(hidden)
        log_var = self.log_var_head(hidden)

        if sample:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean

        return z, mean, log_var

    def kl_divergence(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Compute KL divergence from prior.

        Args:
            mean: Latent mean.
            log_var: Latent log variance.

        Returns:
            KL divergence loss.
        """
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).mean()


class ContrastiveStyleLoss(nn.Module):
    """Contrastive learning for style separation.

    Encourages embeddings from the same user to be similar
    and embeddings from different contexts to be distinct.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialize contrastive loss.

        Args:
            temperature: Softmax temperature.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute contrastive loss.

        Args:
            embeddings: Style embeddings [batch, style_dim].
            labels: Class labels for positive pairs [batch].

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
        exp_sim = exp_sim * (1 - self_mask)  # Remove self

        # For each sample, positive pairs vs all negatives
        pos_sim = (exp_sim * positive_mask).sum(dim=-1)
        all_sim = exp_sim.sum(dim=-1)

        # Avoid log(0)
        loss = -torch.log(pos_sim / (all_sim + 1e-10) + 1e-10)

        return loss.mean()


class StyleProjector(nn.Module):
    """Projects embeddings into user-specific style space.

    Combines variational encoding with orthogonal style components
    to learn a rich, interpretable style representation.

    Attributes:
        config: Style projector configuration.
        encoder: Variational style encoder.
        component_bank: Bank of style components.
        contrastive_loss: Contrastive learning module.
    """

    def __init__(self, config: StyleProjectorConfig) -> None:
        """Initialize style projector.

        Args:
            config: Style projector configuration.
        """
        super().__init__()
        self.config = config

        # Variational encoder
        if config.use_variational:
            self.encoder = VariationalStyleEncoder(
                embed_dim=config.embed_dim,
                style_dim=config.style_dim,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(config.embed_dim, config.style_dim * 2),
                nn.LayerNorm(config.style_dim * 2),
                nn.GELU(),
                nn.Linear(config.style_dim * 2, config.style_dim),
            )

        # Style component bank
        self.component_bank = StyleComponentBank(
            style_dim=config.style_dim,
            num_components=config.num_style_components,
        )

        # Contrastive loss
        if config.use_contrastive:
            self.contrastive_loss = ContrastiveStyleLoss(
                temperature=config.contrastive_temperature,
            )
        else:
            self.contrastive_loss = None

        # Style consistency projection (for temporal consistency)
        self.consistency_proj = nn.Sequential(
            nn.Linear(config.style_dim * 2, config.style_dim),
            nn.Tanh(),
        )

        # Output projection
        self.output_proj = nn.Linear(config.style_dim, config.style_dim)

        logger.info(
            f"Initialized StyleProjector: embed_dim={config.embed_dim}, "
            f"style_dim={config.style_dim}, components={config.num_style_components}"
        )

    def forward(
        self,
        embeddings: Tensor,
        return_components: bool = False,
    ) -> Union[StyleEmbedding, Tuple[StyleEmbedding, Tensor]]:
        """Project embeddings to style space.

        Args:
            embeddings: Input embeddings [batch, embed_dim].
            return_components: Whether to return component activations.

        Returns:
            StyleEmbedding (and optionally component activations).
        """
        batch_size = embeddings.shape[0]

        # Encode to style space
        if self.config.use_variational:
            z, mean, log_var = self.encoder(embeddings)
        else:
            z = self.encoder(embeddings)
            mean = None
            log_var = None

        # Project onto style components
        activations, reconstructed = self.component_bank(z)

        # Blend original and component-reconstructed
        style_embedding = 0.7 * z + 0.3 * reconstructed
        style_embedding = self.output_proj(style_embedding)

        result = StyleEmbedding(
            embedding=style_embedding,
            mean=mean,
            log_var=log_var,
            components=activations,
        )

        if return_components:
            return result, activations
        return result

    def project_batch(
        self,
        embeddings: Tensor,
        aggregate: bool = False,
    ) -> Union[List[StyleEmbedding], StyleSignature]:
        """Project a batch of embeddings.

        Args:
            embeddings: Batch of embeddings [batch, embed_dim].
            aggregate: Whether to aggregate into a StyleSignature.

        Returns:
            List of StyleEmbeddings or aggregated StyleSignature.
        """
        result, activations = self.forward(embeddings, return_components=True)

        if aggregate:
            # Compute signature statistics
            mean_style = result.embedding.mean(dim=0)
            style_variance = result.embedding.var(dim=0)
            component_weights = activations.mean(dim=0)

            return StyleSignature(
                mean_style=mean_style,
                style_variance=style_variance,
                component_weights=component_weights,
                num_samples=embeddings.shape[0],
            )

        # Return individual embeddings
        return [
            StyleEmbedding(
                embedding=result.embedding[i],
                mean=result.mean[i] if result.mean is not None else None,
                log_var=result.log_var[i] if result.log_var is not None else None,
                components=activations[i],
            )
            for i in range(embeddings.shape[0])
        ]

    def compute_signature(
        self,
        embeddings: Tensor,
        num_characteristic: int = 5,
    ) -> StyleSignature:
        """Compute a comprehensive style signature.

        Args:
            embeddings: User's embeddings [num_samples, embed_dim].
            num_characteristic: Number of characteristic patterns to keep.

        Returns:
            StyleSignature capturing user's style.
        """
        self.eval()
        with torch.no_grad():
            # Project all embeddings
            result, activations = self.forward(embeddings, return_components=True)

            # Compute statistics
            mean_style = result.embedding.mean(dim=0)
            style_variance = result.embedding.var(dim=0)
            component_weights = activations.mean(dim=0)

            # Find characteristic patterns (most distinctive from mean)
            distances = torch.norm(result.embedding - mean_style.unsqueeze(0), dim=-1)
            _, extreme_indices = torch.topk(distances, min(num_characteristic, len(distances)))
            characteristic_patterns = [result.embedding[i] for i in extreme_indices]

            return StyleSignature(
                mean_style=mean_style,
                style_variance=style_variance,
                component_weights=component_weights,
                characteristic_patterns=characteristic_patterns,
                num_samples=embeddings.shape[0],
            )

    def style_consistency_loss(
        self,
        style_t: Tensor,
        style_t1: Tensor,
    ) -> Tensor:
        """Compute temporal style consistency loss.

        Encourages style to be consistent across adjacent turns.

        Args:
            style_t: Style at time t [batch, style_dim].
            style_t1: Style at time t+1 [batch, style_dim].

        Returns:
            Consistency loss.
        """
        # Predict next style from current
        combined = torch.cat([style_t, style_t1], dim=-1)
        predicted_change = self.consistency_proj(combined)

        # The predicted change should be small
        actual_change = style_t1 - style_t

        return F.mse_loss(predicted_change, actual_change)

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Optional[Tensor] = None,
        adjacent_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute training loss for style projector.

        Args:
            embeddings: Input embeddings.
            labels: Labels for contrastive learning.
            adjacent_embeddings: Temporally adjacent embeddings for consistency.

        Returns:
            Tuple of (total_loss, loss_components).
        """
        losses = {}

        # Project embeddings
        result, activations = self.forward(embeddings, return_components=True)

        # KL divergence loss (if variational)
        if self.config.use_variational and result.is_variational:
            kl_loss = self.encoder.kl_divergence(result.mean, result.log_var)
            losses["kl_divergence"] = self.config.kl_weight * kl_loss

        # Orthogonality loss for components
        ortho_loss = self.component_bank.orthogonality_loss()
        losses["orthogonality"] = 0.1 * ortho_loss

        # Contrastive loss
        if self.contrastive_loss is not None and labels is not None:
            contrastive = self.contrastive_loss(result.embedding, labels)
            losses["contrastive"] = contrastive

        # Style consistency loss
        if adjacent_embeddings is not None:
            adj_result = self.forward(adjacent_embeddings)
            consistency = self.style_consistency_loss(
                result.embedding, adj_result.embedding
            )
            losses["consistency"] = self.config.style_consistency_weight * consistency

        total_loss = sum(losses.values())
        return total_loss, losses

    def interpolate_styles(
        self,
        style1: StyleEmbedding,
        style2: StyleEmbedding,
        alpha: float,
    ) -> StyleEmbedding:
        """Interpolate between two styles.

        Args:
            style1: First style.
            style2: Second style.
            alpha: Interpolation factor [0, 1].

        Returns:
            Interpolated StyleEmbedding.
        """
        embedding = (1 - alpha) * style1.embedding + alpha * style2.embedding

        components = None
        if style1.components is not None and style2.components is not None:
            components = (1 - alpha) * style1.components + alpha * style2.components

        return StyleEmbedding(
            embedding=embedding,
            components=components,
        )


def create_style_projector(
    embed_dim: int = 768,
    style_dim: int = 256,
    **kwargs,
) -> StyleProjector:
    """Factory function to create style projector.

    Args:
        embed_dim: Input embedding dimension.
        style_dim: Style embedding dimension.
        **kwargs: Additional config arguments.

    Returns:
        StyleProjector instance.
    """
    config = StyleProjectorConfig(
        embed_dim=embed_dim,
        style_dim=style_dim,
        **kwargs,
    )
    return StyleProjector(config)
