"""Prompt Generator for CognitiveTwin.

This module generates prompts and continuations in the user's style,
enabling autonomous prompting that mimics how the user would continue
a conversation or project.

Key capabilities:
    - Style-conditioned generation
    - Trajectory-guided continuations
    - Pattern-based prompt templating
    - Diversity sampling
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cognitive_twin.framework.config import PromptGeneratorConfig, StyleTransferMode
from cognitive_twin.framework.style_projector import StyleEmbedding, StyleSignature
from cognitive_twin.framework.pattern_memory import StoredPattern, PatternQuery
from cognitive_twin._compat import TrajectoryCoordinate5D

logger = logging.getLogger(__name__)


@dataclass
class ContinuationContext:
    """Context for generating a continuation.

    Attributes:
        conversation_history: Recent conversation turns (embeddings).
        current_topic: Current topic embedding.
        trajectory_position: Current position in trajectory space.
        active_patterns: Recently activated reasoning patterns.
        style_signature: User's style signature.
        project_context: Optional project/task context.
    """

    conversation_history: List[Tensor]
    current_topic: Optional[Tensor] = None
    trajectory_position: Optional[TrajectoryCoordinate5D] = None
    active_patterns: List[StoredPattern] = field(default_factory=list)
    style_signature: Optional[StyleSignature] = None
    project_context: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedPrompt:
    """A generated prompt with metadata.

    Attributes:
        embedding: Generated prompt embedding.
        style_contribution: How much style influenced generation.
        trajectory_contribution: How much trajectory guided generation.
        pattern_sources: Patterns that contributed to generation.
        confidence: Generation confidence score.
        diversity_score: How different from recent prompts.
        metadata: Additional generation metadata.
    """

    embedding: Tensor
    style_contribution: float
    trajectory_contribution: float
    pattern_sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    diversity_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """Whether this is a high-confidence generation."""
        return self.confidence > 0.7


class TrajectoryGuidance(nn.Module):
    """Guides generation based on trajectory coordinates.

    Predicts the likely next trajectory position and uses it
    to bias generation.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        """Initialize trajectory guidance.

        Args:
            embed_dim: Embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        # Predict next trajectory position
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(embed_dim + 5, hidden_dim),  # embed + current coords
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),  # next coords
            nn.Sigmoid(),  # Bound to [0, 1] for most dims
        )

        # Convert trajectory to generation guidance
        self.trajectory_to_guidance = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(
        self,
        context_embedding: Tensor,
        current_coords: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute trajectory guidance for generation.

        Args:
            context_embedding: Current context embedding [batch, embed_dim].
            current_coords: Current trajectory coordinates [batch, 5].

        Returns:
            Tuple of (predicted_coords, guidance_vector).
        """
        # Predict next position
        combined = torch.cat([context_embedding, current_coords], dim=-1)
        predicted_coords = self.trajectory_predictor(combined)

        # Convert to guidance
        guidance = self.trajectory_to_guidance(predicted_coords)

        return predicted_coords, guidance


class StyleConditioner(nn.Module):
    """Conditions generation on user style.

    Applies style embedding to guide the generation process.
    """

    def __init__(
        self,
        embed_dim: int,
        style_dim: int,
        mode: StyleTransferMode = StyleTransferMode.ADAPTIVE,
    ) -> None:
        """Initialize style conditioner.

        Args:
            embed_dim: Main embedding dimension.
            style_dim: Style embedding dimension.
            mode: How to apply style.
        """
        super().__init__()
        self.mode = mode

        # Style projection
        self.style_proj = nn.Linear(style_dim, embed_dim)

        # Adaptive gating
        if mode == StyleTransferMode.ADAPTIVE:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim + style_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Sigmoid(),
            )

        # Style blending weights
        self.blend_weight = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        content: Tensor,
        style: Tensor,
    ) -> Tuple[Tensor, float]:
        """Apply style conditioning to content.

        Args:
            content: Content embedding [batch, embed_dim].
            style: Style embedding [batch, style_dim].

        Returns:
            Tuple of (styled_content, style_contribution).
        """
        # Project style to content space
        style_proj = self.style_proj(style)

        if self.mode == StyleTransferMode.DIRECT:
            # Direct addition with learned weight
            weight = torch.sigmoid(self.blend_weight).item()
            styled = content + weight * style_proj
            return styled, weight

        elif self.mode == StyleTransferMode.ADAPTIVE:
            # Adaptive gating based on content
            gate_input = torch.cat([content, style], dim=-1)
            gate = self.gate(gate_input)
            styled = content + gate * style_proj
            return styled, gate.mean().item()

        else:  # INTERPOLATED
            # Interpolate between content and style
            weight = torch.sigmoid(self.blend_weight)
            styled = (1 - weight) * content + weight * style_proj
            return styled, weight.item()


class PatternIntegrator(nn.Module):
    """Integrates retrieved patterns into generation.

    Uses attention over patterns to create a pattern-informed
    generation context.
    """

    def __init__(
        self,
        embed_dim: int,
        max_patterns: int = 10,
    ) -> None:
        """Initialize pattern integrator.

        Args:
            embed_dim: Embedding dimension.
            max_patterns: Maximum patterns to consider.
        """
        super().__init__()
        self.max_patterns = max_patterns

        # Pattern attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        context: Tensor,
        pattern_embeddings: Tensor,
        pattern_importances: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Integrate patterns into context.

        Args:
            context: Context embedding [batch, embed_dim].
            pattern_embeddings: Pattern embeddings [batch, num_patterns, embed_dim].
            pattern_importances: Pattern importance weights [batch, num_patterns].

        Returns:
            Tuple of (integrated_context, attention_weights).
        """
        batch_size = context.shape[0]

        # Compute attention
        query = self.query_proj(context).unsqueeze(1)  # [batch, 1, embed_dim]
        keys = self.key_proj(pattern_embeddings)  # [batch, patterns, embed_dim]
        values = self.value_proj(pattern_embeddings)

        # Scaled dot-product attention
        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(context.shape[-1])

        # Optionally weight by importance
        if pattern_importances is not None:
            scores = scores + pattern_importances.unsqueeze(1).log()

        attention = F.softmax(scores, dim=-1)

        # Aggregate patterns
        aggregated = torch.bmm(attention, values).squeeze(1)
        output = self.output_proj(aggregated)

        return output, attention.squeeze(1)


class DiversitySampler(nn.Module):
    """Encourages diverse prompt generation.

    Tracks recent generations and promotes diversity.
    """

    def __init__(
        self,
        embed_dim: int,
        history_size: int = 10,
    ) -> None:
        """Initialize diversity sampler.

        Args:
            embed_dim: Embedding dimension.
            history_size: Number of recent generations to track.
        """
        super().__init__()
        self.history_size = history_size

        # Recent generation history
        self.register_buffer(
            "history",
            torch.zeros(history_size, embed_dim),
        )
        self.history_ptr = 0

        # Diversity projection
        self.diversity_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
        )

    def compute_diversity(self, embedding: Tensor) -> float:
        """Compute diversity score for an embedding.

        Args:
            embedding: Candidate embedding [embed_dim].

        Returns:
            Diversity score (higher = more diverse).
        """
        if self.history.abs().sum() == 0:
            return 1.0  # No history yet

        # Compute similarity to history
        similarities = F.cosine_similarity(
            embedding.unsqueeze(0),
            self.history,
            dim=-1,
        )

        # Diversity is inverse of max similarity
        max_sim = similarities.max().item()
        return 1.0 - max_sim

    def update_history(self, embedding: Tensor) -> None:
        """Add embedding to history.

        Args:
            embedding: New embedding to add.
        """
        with torch.no_grad():
            self.history[self.history_ptr] = embedding
            self.history_ptr = (self.history_ptr + 1) % self.history_size

    def diversify(self, embedding: Tensor, strength: float = 0.1) -> Tensor:
        """Push embedding away from recent history.

        Args:
            embedding: Embedding to diversify.
            strength: Diversification strength.

        Returns:
            Diversified embedding.
        """
        if self.history.abs().sum() == 0:
            return embedding

        # Compute mean of history
        history_mean = self.history.mean(dim=0)

        # Push away from mean
        direction = embedding - history_mean
        direction = F.normalize(direction, dim=-1)

        diversified = embedding + strength * direction
        return diversified


class PromptGenerator(nn.Module):
    """Generates prompts in user's style.

    Combines style conditioning, trajectory guidance, and pattern
    integration to generate prompts that mimic how the user would
    continue a conversation.

    Attributes:
        config: Generator configuration.
        style_conditioner: Applies user style.
        trajectory_guidance: Guides by trajectory.
        pattern_integrator: Integrates reasoning patterns.
        diversity_sampler: Promotes diverse outputs.
    """

    def __init__(self, config: PromptGeneratorConfig) -> None:
        """Initialize prompt generator.

        Args:
            config: Generator configuration.
        """
        super().__init__()
        self.config = config

        # Core components
        self.style_conditioner = StyleConditioner(
            embed_dim=config.embed_dim,
            style_dim=config.style_dim,
            mode=config.style_transfer_mode,
        )

        if config.use_trajectory_guidance:
            self.trajectory_guidance = TrajectoryGuidance(
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
            )
        else:
            self.trajectory_guidance = None

        self.pattern_integrator = PatternIntegrator(
            embed_dim=config.embed_dim,
        )

        self.diversity_sampler = DiversitySampler(
            embed_dim=config.embed_dim,
        )

        # Context aggregation
        self.context_aggregator = nn.Sequential(
            nn.Linear(config.embed_dim * 3, config.hidden_dim),  # content, style, patterns
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

        # Generation head
        self.generation_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"Initialized PromptGenerator: embed_dim={config.embed_dim}, "
            f"style_mode={config.style_transfer_mode}"
        )

    def forward(
        self,
        context: ContinuationContext,
        num_samples: int = 1,
        temperature: float = None,
    ) -> List[GeneratedPrompt]:
        """Generate prompts for a continuation context.

        Args:
            context: Continuation context.
            num_samples: Number of prompts to generate.
            temperature: Generation temperature (overrides config).

        Returns:
            List of generated prompts.
        """
        temperature = temperature or self.config.temperature
        prompts = []

        # Aggregate conversation history
        if context.conversation_history:
            history_tensor = torch.stack(context.conversation_history)
            content_embedding = history_tensor.mean(dim=0).unsqueeze(0)
        else:
            content_embedding = torch.zeros(1, self.config.embed_dim)

        # Get style embedding
        if context.style_signature is not None:
            style_embedding = context.style_signature.mean_style.unsqueeze(0)
        else:
            style_embedding = torch.zeros(1, self.config.style_dim)

        # Get pattern embeddings
        if context.active_patterns:
            pattern_embeds = torch.stack([p.embedding for p in context.active_patterns])
            pattern_embeds = pattern_embeds.unsqueeze(0)  # Add batch dim
            pattern_importances = torch.tensor(
                [p.importance for p in context.active_patterns]
            ).unsqueeze(0)
        else:
            pattern_embeds = torch.zeros(1, 1, self.config.embed_dim)
            pattern_importances = None

        # Apply style conditioning
        styled_content, style_contrib = self.style_conditioner(
            content_embedding, style_embedding
        )

        # Integrate patterns
        pattern_context, pattern_attention = self.pattern_integrator(
            content_embedding, pattern_embeds, pattern_importances
        )

        # Apply trajectory guidance
        traj_contrib = 0.0
        if self.trajectory_guidance is not None and context.trajectory_position is not None:
            coords_tensor = torch.tensor(
                context.trajectory_position.as_tuple,
                dtype=torch.float32,
            ).unsqueeze(0)
            _, traj_guidance = self.trajectory_guidance(content_embedding, coords_tensor)
            styled_content = styled_content + self.config.trajectory_weight * traj_guidance
            traj_contrib = self.config.trajectory_weight

        # Generate samples
        for _ in range(num_samples):
            # Aggregate all context
            aggregator_input = torch.cat([
                styled_content,
                pattern_context,
                content_embedding,
            ], dim=-1)

            aggregated = self.context_aggregator(aggregator_input)

            # Add noise for diversity
            if temperature > 0:
                noise = torch.randn_like(aggregated) * temperature * 0.1
                aggregated = aggregated + noise

            # Generate
            generated = self.generation_head(aggregated)

            # Apply diversity
            diversity = self.diversity_sampler.compute_diversity(generated.squeeze(0))
            if self.config.diversity_weight > 0:
                generated = self.diversity_sampler.diversify(
                    generated.squeeze(0),
                    strength=self.config.diversity_weight,
                ).unsqueeze(0)

            # Estimate confidence
            confidence = self.confidence_head(generated).item()

            # Get pattern sources
            pattern_sources = []
            if context.active_patterns and pattern_attention is not None:
                top_patterns = pattern_attention.squeeze().topk(
                    min(3, len(context.active_patterns))
                )
                for idx in top_patterns.indices:
                    pattern_sources.append(context.active_patterns[idx.item()].id)

            prompt = GeneratedPrompt(
                embedding=generated.squeeze(0),
                style_contribution=style_contrib,
                trajectory_contribution=traj_contrib,
                pattern_sources=pattern_sources,
                confidence=confidence,
                diversity_score=diversity,
            )
            prompts.append(prompt)

            # Update diversity history
            self.diversity_sampler.update_history(generated.squeeze(0))

        return prompts

    def generate_continuation(
        self,
        context: ContinuationContext,
        target_pattern_type: Optional[str] = None,
    ) -> GeneratedPrompt:
        """Generate a single continuation prompt.

        Higher-level API for common use case.

        Args:
            context: Continuation context.
            target_pattern_type: Optionally target a specific pattern type.

        Returns:
            Best generated prompt.
        """
        # Generate multiple samples
        prompts = self.forward(context, num_samples=3)

        # Select best by confidence and diversity
        scored = [
            (p, p.confidence * 0.7 + p.diversity_score * 0.3)
            for p in prompts
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0]

    def compute_loss(
        self,
        context: ContinuationContext,
        target_embedding: Tensor,
        target_pattern_type: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute training loss for generator.

        Args:
            context: Input context.
            target_embedding: Target prompt embedding.
            target_pattern_type: Optional target pattern type.

        Returns:
            Tuple of (total_loss, loss_components).
        """
        losses = {}

        # Generate
        prompts = self.forward(context, num_samples=1, temperature=0)
        generated = prompts[0].embedding

        # Reconstruction loss
        recon_loss = F.mse_loss(generated, target_embedding)
        losses["reconstruction"] = recon_loss

        # Style consistency loss
        if context.style_signature is not None:
            style_sim = F.cosine_similarity(
                generated.unsqueeze(0),
                context.style_signature.mean_style.unsqueeze(0),
            )
            style_loss = 1 - style_sim.mean()
            losses["style_consistency"] = 0.1 * style_loss

        # Diversity loss (encourage not repeating)
        diversity = self.diversity_sampler.compute_diversity(generated)
        diversity_loss = -torch.tensor(diversity)  # Maximize diversity
        losses["diversity"] = self.config.diversity_weight * diversity_loss

        total_loss = sum(losses.values())
        return total_loss, losses


def create_prompt_generator(**kwargs) -> PromptGenerator:
    """Factory function to create prompt generator.

    Args:
        **kwargs: Configuration arguments.

    Returns:
        PromptGenerator instance.
    """
    config = PromptGeneratorConfig(**kwargs)
    return PromptGenerator(config)
