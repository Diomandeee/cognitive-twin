"""CognitiveTwin - Main Digital Twin Module.

This module provides the main CognitiveTwin class that orchestrates
all components to create a digital twin of the user's reasoning patterns.

The CognitiveTwin can:
    - Learn from conversation history
    - Generate prompts in the user's style
    - Continue projects autonomously
    - Adapt to the user's exploration patterns
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from cognitive_twin.framework.style_projector import StyleEmbedding

import torch
import torch.nn as nn
from torch import Tensor

from cognitive_twin.framework.config import CognitiveTwinConfig
from cognitive_twin.framework.reasoning_encoder import (
    ReasoningPatternEncoder,
    ReasoningPattern,
    PatternType,
)
from cognitive_twin.framework.style_projector import (
    StyleProjector,
    StyleEmbedding,
    StyleSignature,
)
from cognitive_twin.framework.pattern_memory import (
    PatternMemory,
    StoredPattern,
    PatternQuery,
)
from cognitive_twin.framework.prompt_generator import (
    PromptGenerator,
    GeneratedPrompt,
    ContinuationContext,
)
from cognitive_twin._compat import TrajectoryCoordinate5D

logger = logging.getLogger(__name__)


class TwinMode(str, Enum):
    """Operating mode for the twin."""

    LEARNING = "learning"  # Actively learning from new data
    INFERENCE = "inference"  # Generating prompts/continuations
    ANALYSIS = "analysis"  # Analyzing patterns


@dataclass
class TwinState:
    """Current state of the CognitiveTwin.

    Attributes:
        mode: Current operating mode.
        style_signature: Learned style signature.
        num_patterns_learned: Total patterns learned.
        num_conversations_processed: Conversations processed.
        last_trajectory_position: Last known position in trajectory space.
        active_patterns: Currently active patterns.
        confidence: Overall twin confidence.
        metadata: Additional state metadata.
    """

    mode: TwinMode = TwinMode.INFERENCE
    style_signature: Optional[StyleSignature] = None
    num_patterns_learned: int = 0
    num_conversations_processed: int = 0
    last_trajectory_position: Optional[TrajectoryCoordinate5D] = None
    active_patterns: List[StoredPattern] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Whether the twin is ready for generation."""
        return (
            self.style_signature is not None
            and self.num_patterns_learned >= 10
            and self.confidence > 0.3
        )


@dataclass
class TwinOutput:
    """Output from CognitiveTwin operations.

    Attributes:
        prompts: Generated prompts.
        patterns: Detected reasoning patterns.
        analysis: Analysis results.
        state: Updated twin state.
        metrics: Performance metrics.
        # Training-specific outputs
        pattern_logits: Pattern classification logits for loss.
        style_embedding: Style embedding for loss computation.
        reconstructed_embeddings: Reconstructed embeddings.
        patterns_extracted: Patterns extracted this step.
        orthogonality_loss: Style component orthogonality loss.
    """

    prompts: List[GeneratedPrompt] = field(default_factory=list)
    patterns: List[ReasoningPattern] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    state: Optional[TwinState] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    # Training-specific outputs (optional - only during training)
    pattern_logits: Optional[Tensor] = None
    style_embedding: Optional["StyleEmbedding"] = None
    reconstructed_embeddings: Optional[Tensor] = None
    patterns_extracted: Optional[List[ReasoningPattern]] = None
    orthogonality_loss: Optional[Tensor] = None


class CognitiveTwin(nn.Module):
    """Digital Twin for Learning User Reasoning Patterns.

    The CognitiveTwin creates a personalized model of how the user
    thinks, prompts, and explores topics. It can then generate
    prompts and continuations in the user's style.

    Core capabilities:
        - Learn reasoning patterns via inverse attention
        - Build a style signature from conversation history
        - Store and retrieve patterns efficiently
        - Generate style-consistent prompts
        - Guide generation by trajectory position

    Attributes:
        config: Twin configuration.
        reasoning_encoder: Learns reasoning patterns.
        style_projector: Projects to style space.
        pattern_memory: Stores learned patterns.
        prompt_generator: Generates prompts.
        state: Current twin state.
    """

    def __init__(self, config: CognitiveTwinConfig) -> None:
        """Initialize CognitiveTwin.

        Args:
            config: Twin configuration.
        """
        super().__init__()
        self.config = config
        config.validate()

        # Initialize components
        self.reasoning_encoder = ReasoningPatternEncoder(config.reasoning_encoder)
        self.style_projector = StyleProjector(config.style_projector)
        self.pattern_memory = PatternMemory(config.pattern_memory)
        self.prompt_generator = PromptGenerator(config.prompt_generator)

        # Twin state
        self.state = TwinState()

        # Device
        self.device = torch.device(config.device)

        # Statistics
        self._training_steps = 0
        self._generation_count = 0

        logger.info(
            f"Initialized CognitiveTwin: device={config.device}, "
            f"embed_dim={config.reasoning_encoder.embed_dim}"
        )

    def to_device(self, device: str) -> "CognitiveTwin":
        """Move twin to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.to(self.device)
        return self

    def learn_from_conversation(
        self,
        turns: Union[List[Dict[str, Any]], List[str]],
        embeddings: Tensor,
        coordinates: Optional[Union[List[TrajectoryCoordinate5D], Tensor]] = None,
        context_embeddings: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> TwinOutput:
        """Learn from a conversation.

        Extracts reasoning patterns, updates style signature, and
        stores patterns in memory.

        Supports two modes:
        1. Single conversation: turns as List[Dict], embeddings as [num_turns, dim]
        2. Batched training: turns as List[str], embeddings as [batch, dim],
           context_embeddings as [batch, ctx_len, dim], context_mask as [batch, ctx_len]

        Args:
            turns: List of conversation turns with 'role' and 'content', or text content.
            embeddings: Turn embeddings [num_turns, embed_dim] or [batch, embed_dim].
            coordinates: Optional trajectory coordinates as list or tensor [batch, 5].
            context_embeddings: Optional context for batched training [batch, ctx_len, dim].
            context_mask: Optional mask for context [batch, ctx_len].

        Returns:
            TwinOutput with learned patterns and training tensors.
        """
        self.state.mode = TwinMode.LEARNING
        self.train()

        output = TwinOutput()
        embeddings = embeddings.to(self.device)

        # Check if batched training mode (context_embeddings provided)
        if context_embeddings is not None:
            return self._learn_batched(
                turns, embeddings, coordinates, context_embeddings, context_mask
            )

        # Original single-conversation mode
        # Separate user and assistant turns
        user_indices = [i for i, t in enumerate(turns) if isinstance(t, dict) and t.get("role") == "user"]
        assistant_indices = [i for i, t in enumerate(turns) if isinstance(t, dict) and t.get("role") == "assistant"]

        # Learn from assistant responses (how user would respond)
        for i, asst_idx in enumerate(assistant_indices):
            # Context is everything before this turn
            if asst_idx == 0:
                continue

            context_indices = list(range(asst_idx))
            context = embeddings[context_indices].unsqueeze(0)
            response = embeddings[asst_idx:asst_idx + 1].unsqueeze(0)

            # Get coordinates if available
            resp_coords = None
            ctx_coords = None
            if coordinates is not None and not isinstance(coordinates, Tensor):
                resp_coords = [coordinates[asst_idx]]
                ctx_coords = [coordinates[j] for j in context_indices]

            # Extract patterns
            patterns = self.reasoning_encoder.extract_patterns(
                response, context, resp_coords, ctx_coords
            )

            # Store patterns
            for pattern in patterns:
                pattern_id = self.pattern_memory.store(pattern)
                output.patterns.append(pattern)

            self.state.num_patterns_learned += len(patterns)

        # Update style signature from user turns
        if user_indices:
            user_embeddings = embeddings[user_indices]
            new_signature = self.style_projector.compute_signature(user_embeddings)

            if self.state.style_signature is None:
                self.state.style_signature = new_signature
            else:
                # Blend with existing signature
                self._blend_signatures(new_signature)

        # Update state
        self.state.num_conversations_processed += 1
        if coordinates is not None and not isinstance(coordinates, Tensor):
            self.state.last_trajectory_position = coordinates[-1]

        # Update confidence
        self._update_confidence()

        output.state = self.state
        output.metrics = {
            "patterns_extracted": len(output.patterns),
            "total_patterns": self.state.num_patterns_learned,
            "confidence": self.state.confidence,
        }

        logger.info(
            f"Learned from conversation: {len(output.patterns)} patterns, "
            f"confidence={self.state.confidence:.3f}"
        )

        return output

    def _learn_batched(
        self,
        turns: List[str],
        embeddings: Tensor,
        coordinates: Optional[Tensor],
        context_embeddings: Tensor,
        context_mask: Optional[Tensor],
    ) -> TwinOutput:
        """Batched training forward pass.

        Args:
            turns: Text content (for reference).
            embeddings: Response embeddings [batch, embed_dim].
            coordinates: Coordinates tensor [batch, 5].
            context_embeddings: Context [batch, ctx_len, embed_dim].
            context_mask: Context mask [batch, ctx_len].

        Returns:
            TwinOutput with training tensors.
        """
        output = TwinOutput()
        batch_size = embeddings.shape[0]

        # Move to device
        context_embeddings = context_embeddings.to(self.device)
        if context_mask is not None:
            context_mask = context_mask.to(self.device)
        if coordinates is not None:
            coordinates = coordinates.to(self.device)

        # Get coordinates for attention bias
        resp_coords = None
        ctx_coords = None
        if coordinates is not None:
            # Response coordinates [batch, 1, 5] to match response shape
            resp_coords = coordinates.unsqueeze(1)
            # Broadcast to context length
            ctx_len = context_embeddings.shape[1]
            # For now, use response coords for all context (simplified)
            ctx_coords = coordinates.unsqueeze(1).expand(-1, ctx_len, -1)

        # Forward through reasoning encoder (get logits for loss)
        attention_weights, pattern_logits, pattern_embeddings = self.reasoning_encoder(
            response=embeddings.unsqueeze(1),  # [batch, 1, dim]
            context=context_embeddings,
            response_coords=resp_coords,
            context_coords=ctx_coords,
            mask=context_mask,
        )

        # Forward through style projector (get style for loss)
        style_result, style_activations = self.style_projector(
            embeddings, return_components=True
        )

        # Compute orthogonality loss from style component bank
        ortho_loss = self.style_projector.component_bank.orthogonality_loss()

        # Extract patterns (for logging, not differentiable)
        patterns_extracted = []
        with torch.no_grad():
            for i in range(min(batch_size, 5)):  # Sample first 5 for efficiency
                # Get predicted pattern type
                pattern_idx = pattern_logits[i].argmax().item()
                pattern_type = list(PatternType)[pattern_idx % len(PatternType)]

                # Compute attention entropy
                attn = attention_weights[i].squeeze()  # [context_len]
                if attn.dim() == 0:
                    attn = attn.unsqueeze(0)
                entropy = -(attn * (attn + 1e-10).log()).sum().item()

                patterns_extracted.append(
                    ReasoningPattern(
                        pattern_type=pattern_type,
                        attention_weights=attn.cpu(),
                        attention_entropy=entropy,
                        coordinate=None,
                        context_indices=list(range(attn.shape[-1])),
                        strength=pattern_logits[i].softmax(dim=-1).max().item(),
                        embedding=pattern_embeddings[i].squeeze().cpu() if pattern_embeddings is not None else None,
                    )
                )

        # Populate output for training
        output.pattern_logits = pattern_logits
        output.style_embedding = style_result
        output.reconstructed_embeddings = None  # Could add decoder if needed
        output.patterns_extracted = patterns_extracted
        output.orthogonality_loss = ortho_loss

        # Update counters (non-training)
        self.state.num_patterns_learned += len(patterns_extracted)

        output.state = self.state
        output.metrics = {
            "batch_size": batch_size,
            "patterns_sampled": len(patterns_extracted),
        }

        return output

    def _blend_signatures(self, new_signature: StyleSignature) -> None:
        """Blend new signature with existing.

        Args:
            new_signature: New signature to blend in.
        """
        if self.state.style_signature is None:
            self.state.style_signature = new_signature
            return

        old = self.state.style_signature
        n_old = old.num_samples
        n_new = new_signature.num_samples

        # Weighted average
        weight_old = n_old / (n_old + n_new)
        weight_new = n_new / (n_old + n_new)

        self.state.style_signature = StyleSignature(
            mean_style=weight_old * old.mean_style + weight_new * new_signature.mean_style,
            style_variance=weight_old * old.style_variance + weight_new * new_signature.style_variance,
            component_weights=weight_old * old.component_weights + weight_new * new_signature.component_weights,
            num_samples=n_old + n_new,
        )

    def _update_confidence(self) -> None:
        """Update overall twin confidence."""
        # Factors: patterns, style, conversations
        pattern_factor = min(1.0, self.state.num_patterns_learned / 100)
        style_factor = 1.0 if self.state.style_signature is not None else 0.0
        conv_factor = min(1.0, self.state.num_conversations_processed / 10)

        self.state.confidence = 0.4 * pattern_factor + 0.4 * style_factor + 0.2 * conv_factor

    def generate_prompt(
        self,
        context: Optional[str] = None,
        context_embeddings: Optional[Tensor] = None,
        trajectory_position: Optional[TrajectoryCoordinate5D] = None,
        num_samples: int = 1,
        temperature: float = 0.8,
    ) -> TwinOutput:
        """Generate prompts in user's style.

        Args:
            context: Text context for generation.
            context_embeddings: Pre-computed context embeddings.
            trajectory_position: Current trajectory position.
            num_samples: Number of prompts to generate.
            temperature: Generation temperature.

        Returns:
            TwinOutput with generated prompts.
        """
        if not self.state.is_ready:
            logger.warning("Twin not ready for generation")
            return TwinOutput(
                analysis={"error": "Twin not ready", "confidence": self.state.confidence}
            )

        self.state.mode = TwinMode.INFERENCE
        self.eval()

        output = TwinOutput()

        # Prepare context
        if context_embeddings is not None:
            history = [e for e in context_embeddings]
        else:
            history = []

        # Retrieve relevant patterns
        if history:
            query = PatternQuery(
                embedding=history[-1] if history else torch.zeros(self.config.reasoning_encoder.embed_dim),
                max_results=10,
            )
            retrieved = self.pattern_memory.retrieve(query)
            active_patterns = [p for p, _ in retrieved]
        else:
            active_patterns = []

        # Build continuation context
        cont_context = ContinuationContext(
            conversation_history=history,
            trajectory_position=trajectory_position or self.state.last_trajectory_position,
            active_patterns=active_patterns,
            style_signature=self.state.style_signature,
        )

        # Generate
        with torch.no_grad():
            prompts = self.prompt_generator(
                cont_context,
                num_samples=num_samples,
                temperature=temperature,
            )

        output.prompts = prompts
        output.state = self.state
        output.metrics = {
            "num_generated": len(prompts),
            "mean_confidence": sum(p.confidence for p in prompts) / len(prompts) if prompts else 0,
            "mean_diversity": sum(p.diversity_score for p in prompts) / len(prompts) if prompts else 0,
        }

        self._generation_count += len(prompts)

        return output

    def continue_project(
        self,
        project_context: Dict[str, Any],
        recent_embeddings: Tensor,
        trajectory_position: Optional[TrajectoryCoordinate5D] = None,
    ) -> TwinOutput:
        """Generate continuation for a project.

        Higher-level API for autonomous project continuation.

        Args:
            project_context: Project context (files, goals, etc.).
            recent_embeddings: Recent conversation embeddings.
            trajectory_position: Current trajectory position.

        Returns:
            TwinOutput with continuation prompts.
        """
        output = self.generate_prompt(
            context_embeddings=recent_embeddings,
            trajectory_position=trajectory_position,
            num_samples=3,
            temperature=0.7,
        )

        # Rank by confidence and select best
        if output.prompts:
            best_prompt = max(output.prompts, key=lambda p: p.confidence)
            output.prompts = [best_prompt]

        output.analysis = {
            "project_context": project_context,
            "continuation_type": "autonomous",
        }

        return output

    def analyze_patterns(
        self,
        pattern_types: Optional[List[PatternType]] = None,
    ) -> TwinOutput:
        """Analyze learned reasoning patterns.

        Args:
            pattern_types: Filter by specific types.

        Returns:
            TwinOutput with analysis results.
        """
        self.state.mode = TwinMode.ANALYSIS
        output = TwinOutput()

        # Get memory statistics
        memory_stats = self.pattern_memory.get_statistics()

        # Analyze pattern distribution
        if pattern_types:
            patterns_by_type = {
                pt: self.pattern_memory.get_by_type(pt)
                for pt in pattern_types
            }
        else:
            patterns_by_type = {
                pt: self.pattern_memory.get_by_type(pt)
                for pt in PatternType
            }

        # Compute pattern insights
        analysis = {
            "memory_stats": memory_stats,
            "patterns_by_type": {
                pt.value: len(patterns) for pt, patterns in patterns_by_type.items()
            },
            "dominant_pattern_type": max(
                patterns_by_type.items(),
                key=lambda x: len(x[1]),
                default=(None, []),
            )[0],
            "style_signature_exists": self.state.style_signature is not None,
            "confidence": self.state.confidence,
        }

        output.analysis = analysis
        output.state = self.state

        return output

    def save(self, path: Union[str, Path]) -> None:
        """Save twin state and models.

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save({
            "reasoning_encoder": self.reasoning_encoder.state_dict(),
            "style_projector": self.style_projector.state_dict(),
            "prompt_generator": self.prompt_generator.state_dict(),
            "config": self.config,
        }, path / "model.pt")

        # Save pattern memory
        self.pattern_memory.save_state(str(path / "patterns.pt"))

        # Save state
        state_dict = {
            "mode": self.state.mode.value,
            "num_patterns_learned": self.state.num_patterns_learned,
            "num_conversations_processed": self.state.num_conversations_processed,
            "confidence": self.state.confidence,
            "training_steps": self._training_steps,
            "generation_count": self._generation_count,
        }

        if self.state.style_signature is not None:
            state_dict["style_signature"] = {
                "mean_style": self.state.style_signature.mean_style.cpu(),
                "style_variance": self.state.style_signature.style_variance.cpu(),
                "component_weights": self.state.style_signature.component_weights.cpu(),
                "num_samples": self.state.style_signature.num_samples,
            }

        torch.save(state_dict, path / "state.pt")

        logger.info(f"Saved CognitiveTwin to {path}")

    def load(self, path: Union[str, Path]) -> "CognitiveTwin":
        """Load twin state and models.

        Args:
            path: Directory to load from.

        Returns:
            Self for chaining.
        """
        path = Path(path)

        # Load model weights
        model_state = torch.load(path / "model.pt", map_location=self.device)
        self.reasoning_encoder.load_state_dict(model_state["reasoning_encoder"])
        self.style_projector.load_state_dict(model_state["style_projector"])
        self.prompt_generator.load_state_dict(model_state["prompt_generator"])

        # Load pattern memory
        self.pattern_memory.load_state(str(path / "patterns.pt"))

        # Load state
        state_dict = torch.load(path / "state.pt", map_location=self.device)
        self.state.mode = TwinMode(state_dict["mode"])
        self.state.num_patterns_learned = state_dict["num_patterns_learned"]
        self.state.num_conversations_processed = state_dict["num_conversations_processed"]
        self.state.confidence = state_dict["confidence"]
        self._training_steps = state_dict["training_steps"]
        self._generation_count = state_dict["generation_count"]

        if "style_signature" in state_dict:
            sig = state_dict["style_signature"]
            self.state.style_signature = StyleSignature(
                mean_style=sig["mean_style"].to(self.device),
                style_variance=sig["style_variance"].to(self.device),
                component_weights=sig["component_weights"].to(self.device),
                num_samples=sig["num_samples"],
            )

        logger.info(f"Loaded CognitiveTwin from {path}")
        return self

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive twin statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "state": {
                "mode": self.state.mode.value,
                "is_ready": self.state.is_ready,
                "confidence": self.state.confidence,
                "num_patterns": self.state.num_patterns_learned,
                "num_conversations": self.state.num_conversations_processed,
            },
            "memory": self.pattern_memory.get_statistics(),
            "generation": {
                "total_generated": self._generation_count,
            },
            "training": {
                "steps": self._training_steps,
            },
        }


def create_cognitive_twin(
    preset: str = "balanced",
    **kwargs,
) -> CognitiveTwin:
    """Factory function to create a CognitiveTwin.

    Args:
        preset: Configuration preset ('fast', 'accurate', 'balanced').
        **kwargs: Override configuration values.

    Returns:
        CognitiveTwin instance.
    """
    if preset == "fast":
        config = CognitiveTwinConfig.fast()
    elif preset == "accurate":
        config = CognitiveTwinConfig.accurate()
    else:
        config = CognitiveTwinConfig.balanced()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return CognitiveTwin(config)
