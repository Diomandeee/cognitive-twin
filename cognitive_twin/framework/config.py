"""Configuration for CognitiveTwin architecture.

This module defines all configuration dataclasses for the CognitiveTwin
system, including configs for each subcomponent.

Presets:
    - CognitiveTwinConfig.fast(): Optimized for inference speed
    - CognitiveTwinConfig.accurate(): Optimized for pattern quality
    - CognitiveTwinConfig.balanced(): Balance of speed and quality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class AttentionMode(str, Enum):
    """Mode for inverse attention computation."""

    SINGLE_SCALE = "single_scale"
    MULTI_SCALE = "multi_scale"
    ADAPTIVE = "adaptive"


class MemoryRetrievalMode(str, Enum):
    """Mode for pattern memory retrieval."""

    EXACT = "exact"  # Exact nearest neighbor
    APPROXIMATE = "approximate"  # ANN for speed
    HYBRID = "hybrid"  # Combination


class StyleTransferMode(str, Enum):
    """Mode for style transfer in generation."""

    DIRECT = "direct"  # Direct style embedding
    ADAPTIVE = "adaptive"  # Context-adaptive
    INTERPOLATED = "interpolated"  # Interpolate between styles


@dataclass
class ReasoningEncoderConfig:
    """Configuration for ReasoningPatternEncoder.

    Attributes:
        embed_dim: Embedding dimension (matches Gemini 768d or projected).
        hidden_dim: Hidden dimension for projection networks.
        num_attention_heads: Number of inverse attention heads.
        attention_mode: Single-scale, multi-scale, or adaptive.
        temperature: Softmax temperature for attention.
        use_gumbel: Use Gumbel-softmax for discrete attention.
        sparsity_weight: Weight for attention sparsity regularization.
        dropout: Dropout probability.
        use_coordinate_bias: Whether to bias attention by trajectory coords.
        coordinate_weight: Weight for coordinate distance in attention.
    """

    embed_dim: int = 768
    hidden_dim: int = 1024
    num_attention_heads: int = 4
    attention_mode: AttentionMode = AttentionMode.MULTI_SCALE
    temperature: float = 1.0
    use_gumbel: bool = False
    sparsity_weight: float = 0.01
    dropout: float = 0.1
    use_coordinate_bias: bool = True
    coordinate_weight: float = 0.3

    # DLM coordinate weights for attention bias
    coord_weights: Tuple[float, float, float, float, float] = (
        0.25,  # depth
        0.15,  # sibling_order
        0.30,  # homogeneity
        0.20,  # temporal
        0.10,  # complexity
    )


@dataclass
class StyleProjectorConfig:
    """Configuration for StyleProjector.

    Attributes:
        embed_dim: Input embedding dimension.
        style_dim: Style embedding dimension.
        num_style_components: Number of orthogonal style components.
        use_variational: Use VAE-style variational projection.
        kl_weight: Weight for KL divergence in variational mode.
        style_consistency_weight: Weight for style consistency loss.
        use_contrastive: Use contrastive learning for style separation.
        contrastive_temperature: Temperature for contrastive loss.
    """

    embed_dim: int = 768
    style_dim: int = 256
    num_style_components: int = 8
    use_variational: bool = True
    kl_weight: float = 0.001
    style_consistency_weight: float = 0.1
    use_contrastive: bool = True
    contrastive_temperature: float = 0.07


@dataclass
class PatternMemoryConfig:
    """Configuration for PatternMemory.

    Attributes:
        max_patterns: Maximum number of patterns to store.
        pattern_dim: Dimension of pattern embeddings.
        retrieval_mode: How to retrieve patterns.
        num_retrieval_results: Number of patterns to retrieve.
        similarity_threshold: Minimum similarity for retrieval.
        decay_rate: Temporal decay rate for pattern importance.
        consolidation_interval: How often to consolidate patterns.
        use_hierarchical: Use hierarchical pattern organization.
        hierarchy_levels: Number of hierarchy levels.
    """

    max_patterns: int = 10000
    pattern_dim: int = 768
    retrieval_mode: MemoryRetrievalMode = MemoryRetrievalMode.HYBRID
    num_retrieval_results: int = 10
    similarity_threshold: float = 0.7
    decay_rate: float = 0.01
    consolidation_interval: int = 100
    use_hierarchical: bool = True
    hierarchy_levels: int = 3


@dataclass
class PromptGeneratorConfig:
    """Configuration for PromptGenerator.

    Attributes:
        embed_dim: Embedding dimension.
        hidden_dim: Hidden dimension for generation networks.
        style_dim: Style embedding dimension.
        max_prompt_length: Maximum generated prompt length.
        style_transfer_mode: How to apply style.
        temperature: Generation temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        use_trajectory_guidance: Guide generation with trajectory coords.
        trajectory_weight: Weight for trajectory guidance.
        diversity_weight: Weight for promoting diverse outputs.
    """

    embed_dim: int = 768
    hidden_dim: int = 1024
    style_dim: int = 256
    max_prompt_length: int = 512
    style_transfer_mode: StyleTransferMode = StyleTransferMode.ADAPTIVE
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    use_trajectory_guidance: bool = True
    trajectory_weight: float = 0.2
    diversity_weight: float = 0.1


@dataclass
class CognitiveTwinConfig:
    """Master configuration for CognitiveTwin.

    Combines all subcomponent configurations with global settings.

    Attributes:
        reasoning_encoder: ReasoningPatternEncoder configuration.
        style_projector: StyleProjector configuration.
        pattern_memory: PatternMemory configuration.
        prompt_generator: PromptGenerator configuration.
        use_rust_acceleration: Use Rust for performance-critical ops.
        batch_size: Default batch size for processing.
        device: Device for computation ('cpu', 'cuda', 'mps').
        dtype: Data type ('float32', 'float16', 'bfloat16').
        checkpoint_dir: Directory for saving checkpoints.
        log_level: Logging level.
    """

    # Subcomponent configs
    reasoning_encoder: ReasoningEncoderConfig = field(
        default_factory=ReasoningEncoderConfig
    )
    style_projector: StyleProjectorConfig = field(
        default_factory=StyleProjectorConfig
    )
    pattern_memory: PatternMemoryConfig = field(
        default_factory=PatternMemoryConfig
    )
    prompt_generator: PromptGeneratorConfig = field(
        default_factory=PromptGeneratorConfig
    )

    # Global settings
    use_rust_acceleration: bool = True
    batch_size: int = 32
    device: str = "cpu"
    dtype: str = "float32"
    checkpoint_dir: Optional[str] = None
    log_level: str = "INFO"

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Knowledge transfer settings
    knowledge_transfer_patterns: List[str] = field(
        default_factory=lambda: [
            r"as (we|I) discussed (before|earlier|previously)",
            r"building on (the|our) previous",
            r"similar to what we did (for|with|in)",
            r"remember when we",
            r"like (in|from) the .* (project|conversation|discussion)",
        ]
    )
    detect_methodology_transfer: bool = True
    detect_correction_patterns: bool = True

    @classmethod
    def fast(cls) -> "CognitiveTwinConfig":
        """Create config optimized for inference speed.

        Returns:
            CognitiveTwinConfig with fast settings.
        """
        return cls(
            reasoning_encoder=ReasoningEncoderConfig(
                num_attention_heads=2,
                attention_mode=AttentionMode.SINGLE_SCALE,
                hidden_dim=512,
            ),
            style_projector=StyleProjectorConfig(
                style_dim=128,
                num_style_components=4,
                use_variational=False,
            ),
            pattern_memory=PatternMemoryConfig(
                max_patterns=5000,
                retrieval_mode=MemoryRetrievalMode.APPROXIMATE,
                num_retrieval_results=5,
            ),
            prompt_generator=PromptGeneratorConfig(
                hidden_dim=512,
                style_dim=128,
            ),
            batch_size=64,
        )

    @classmethod
    def accurate(cls) -> "CognitiveTwinConfig":
        """Create config optimized for pattern quality.

        Returns:
            CognitiveTwinConfig with accuracy-focused settings.
        """
        return cls(
            reasoning_encoder=ReasoningEncoderConfig(
                num_attention_heads=8,
                attention_mode=AttentionMode.ADAPTIVE,
                hidden_dim=1536,
                use_gumbel=True,
            ),
            style_projector=StyleProjectorConfig(
                style_dim=512,
                num_style_components=16,
                use_variational=True,
                use_contrastive=True,
            ),
            pattern_memory=PatternMemoryConfig(
                max_patterns=50000,
                retrieval_mode=MemoryRetrievalMode.EXACT,
                num_retrieval_results=20,
                use_hierarchical=True,
            ),
            prompt_generator=PromptGeneratorConfig(
                hidden_dim=1536,
                style_dim=512,
                use_trajectory_guidance=True,
            ),
            batch_size=16,
        )

    @classmethod
    def balanced(cls) -> "CognitiveTwinConfig":
        """Create balanced config for general use.

        Returns:
            CognitiveTwinConfig with balanced settings.
        """
        return cls()  # Defaults are balanced

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Check dimension consistency
        if self.style_projector.embed_dim != self.reasoning_encoder.embed_dim:
            raise ValueError(
                f"Style projector embed_dim ({self.style_projector.embed_dim}) "
                f"must match reasoning encoder ({self.reasoning_encoder.embed_dim})"
            )

        if self.prompt_generator.style_dim != self.style_projector.style_dim:
            raise ValueError(
                f"Prompt generator style_dim ({self.prompt_generator.style_dim}) "
                f"must match style projector ({self.style_projector.style_dim})"
            )

        # Check reasonable ranges
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
