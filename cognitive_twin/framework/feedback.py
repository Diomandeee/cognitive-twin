"""Feedback Learning for CognitiveTwin.

This module implements feedback-based learning to refine CognitiveTwin
using explicit user signals (thumbs up/down) and implicit signals
(regeneration, edits, time-to-accept).

Core Components:
    - FeedbackLearner: Processes feedback signals for model updates
    - FeedbackBuffer: Stores recent feedback for batch updates
    - RewardModel: Learned reward function from feedback data
    - PreferenceOptimizer: Applies preference optimization (DPO-style)

Feedback Types:
    1. Explicit: thumbs_up, thumbs_down, rating (1-5)
    2. Implicit: regeneration, edit_distance, time_to_accept
    3. Behavioral: follow-up questions, topic switches

Usage:
    >>> from cognitive_twin.framework.feedback import FeedbackLearner, FeedbackConfig
    >>> learner = FeedbackLearner(config=FeedbackConfig())
    >>> learner.record_feedback(prompt_id, response_id, signal="thumbs_up")
    >>> # Batch update when buffer is full
    >>> if learner.should_update():
    ...     learner.update_model(twin)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# Feedback Types and Configuration
# =============================================================================

class FeedbackSignal(str, Enum):
    """Types of feedback signals."""
    
    # Explicit
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 scale
    
    # Implicit
    REGENERATE = "regenerate"
    EDIT = "edit"
    ACCEPT_IMMEDIATE = "accept_immediate"  # < 5 seconds
    ACCEPT_SLOW = "accept_slow"  # > 30 seconds
    
    # Behavioral
    FOLLOW_UP = "follow_up"  # Continued the conversation
    TOPIC_SWITCH = "topic_switch"  # Abrupt topic change
    ABANDON = "abandon"  # Left conversation


@dataclass
class FeedbackConfig:
    """Configuration for feedback learning.
    
    Attributes:
        buffer_size: Maximum feedback samples to store.
        update_threshold: Trigger update after N feedback samples.
        learning_rate: Learning rate for feedback updates.
        reward_weight: Weight for reward-based updates.
        preference_weight: Weight for preference optimization.
        decay_factor: Discount older feedback.
        implicit_signal_weight: Weight for implicit vs explicit signals.
        positive_reward: Reward value for positive feedback.
        negative_reward: Reward value for negative feedback.
        neutral_threshold: Threshold for neutral signal.
    """
    
    buffer_size: int = 1000
    update_threshold: int = 50
    learning_rate: float = 1e-5
    reward_weight: float = 0.3
    preference_weight: float = 0.5
    contrastive_weight: float = 0.2
    decay_factor: float = 0.95
    implicit_signal_weight: float = 0.5
    positive_reward: float = 1.0
    negative_reward: float = -1.0
    neutral_threshold: float = 0.3
    min_pairs_for_update: int = 10


@dataclass
class FeedbackSample:
    """Single feedback sample.
    
    Attributes:
        prompt_id: Unique prompt identifier.
        response_id: Unique response identifier.
        signal: Feedback signal type.
        value: Numeric value (for ratings, edit distance, etc.).
        embedding: Response embedding at feedback time.
        style_embedding: Style embedding at feedback time.
        timestamp: When feedback was received.
        context_embedding: Context used for response.
        metadata: Additional metadata.
    """
    
    prompt_id: str
    response_id: str
    signal: FeedbackSignal
    value: float = 1.0
    embedding: Optional[Tensor] = None
    style_embedding: Optional[Tensor] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context_embedding: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_reward(self, config: FeedbackConfig) -> float:
        """Compute reward from feedback signal."""
        if self.signal == FeedbackSignal.THUMBS_UP:
            return config.positive_reward
        elif self.signal == FeedbackSignal.THUMBS_DOWN:
            return config.negative_reward
        elif self.signal == FeedbackSignal.RATING:
            # Normalize 1-5 to -1 to 1
            return (self.value - 3) / 2
        elif self.signal == FeedbackSignal.REGENERATE:
            return config.negative_reward * config.implicit_signal_weight
        elif self.signal == FeedbackSignal.EDIT:
            # Larger edits = more negative
            edit_penalty = min(1.0, self.value / 100)  # Normalize edit distance
            return config.negative_reward * edit_penalty * config.implicit_signal_weight
        elif self.signal == FeedbackSignal.ACCEPT_IMMEDIATE:
            return config.positive_reward * config.implicit_signal_weight
        elif self.signal == FeedbackSignal.ACCEPT_SLOW:
            return config.neutral_threshold * config.implicit_signal_weight
        elif self.signal == FeedbackSignal.FOLLOW_UP:
            return config.positive_reward * 0.5 * config.implicit_signal_weight
        elif self.signal == FeedbackSignal.TOPIC_SWITCH:
            return 0.0  # Neutral
        elif self.signal == FeedbackSignal.ABANDON:
            return config.negative_reward * 0.5 * config.implicit_signal_weight
        return 0.0


@dataclass
class PreferencePair:
    """Preference pair for DPO-style training.
    
    Attributes:
        prompt_embedding: Shared prompt embedding.
        chosen_embedding: Embedding of preferred response.
        rejected_embedding: Embedding of rejected response.
        chosen_reward: Reward of chosen response.
        rejected_reward: Reward of rejected response.
        margin: Preference margin (chosen_reward - rejected_reward).
    """
    
    prompt_embedding: Tensor
    chosen_embedding: Tensor
    rejected_embedding: Tensor
    chosen_reward: float
    rejected_reward: float
    
    @property
    def margin(self) -> float:
        return self.chosen_reward - self.rejected_reward


# =============================================================================
# Feedback Buffer
# =============================================================================

class FeedbackBuffer:
    """Buffer for storing recent feedback samples.
    
    Maintains a rolling window of feedback samples and can generate
    preference pairs for training.
    """
    
    def __init__(self, config: FeedbackConfig):
        """Initialize buffer.
        
        Args:
            config: Feedback configuration.
        """
        self.config = config
        self._buffer: deque[FeedbackSample] = deque(maxlen=config.buffer_size)
        self._positive: List[FeedbackSample] = []
        self._negative: List[FeedbackSample] = []
        
    def add(self, sample: FeedbackSample) -> None:
        """Add feedback sample to buffer.
        
        Args:
            sample: Feedback sample to add.
        """
        self._buffer.append(sample)
        
        # Categorize for preference pairs
        reward = sample.get_reward(self.config)
        if reward > self.config.neutral_threshold:
            self._positive.append(sample)
        elif reward < -self.config.neutral_threshold:
            self._negative.append(sample)
            
        # Trim category lists
        max_per_category = self.config.buffer_size // 2
        if len(self._positive) > max_per_category:
            self._positive = self._positive[-max_per_category:]
        if len(self._negative) > max_per_category:
            self._negative = self._negative[-max_per_category:]
    
    def get_samples(self, n: Optional[int] = None) -> List[FeedbackSample]:
        """Get recent samples.
        
        Args:
            n: Number of samples to get (None for all).
            
        Returns:
            List of feedback samples.
        """
        samples = list(self._buffer)
        if n is not None:
            samples = samples[-n:]
        return samples
    
    def get_preference_pairs(self, max_pairs: Optional[int] = None) -> List[PreferencePair]:
        """Generate preference pairs from positive/negative samples.
        
        Pairs each positive sample with a negative sample for contrastive learning.
        
        Args:
            max_pairs: Maximum number of pairs to generate.
            
        Returns:
            List of preference pairs.
        """
        pairs = []
        
        # Filter samples that have embeddings
        positive_with_emb = [s for s in self._positive if s.embedding is not None]
        negative_with_emb = [s for s in self._negative if s.embedding is not None]
        
        if not positive_with_emb or not negative_with_emb:
            return pairs
        
        # Create pairs
        import itertools
        for pos, neg in itertools.product(positive_with_emb, negative_with_emb):
            if pos.prompt_id == neg.prompt_id:
                # Same prompt - perfect pair
                pair = PreferencePair(
                    prompt_embedding=pos.context_embedding or torch.zeros_like(pos.embedding),
                    chosen_embedding=pos.embedding,
                    rejected_embedding=neg.embedding,
                    chosen_reward=pos.get_reward(self.config),
                    rejected_reward=neg.get_reward(self.config),
                )
                pairs.append(pair)
            
            if max_pairs and len(pairs) >= max_pairs:
                break
        
        # If not enough same-prompt pairs, create cross-prompt pairs
        if len(pairs) < (max_pairs or self.config.min_pairs_for_update):
            for pos in positive_with_emb:
                for neg in negative_with_emb:
                    if pos.prompt_id != neg.prompt_id:
                        pair = PreferencePair(
                            prompt_embedding=pos.context_embedding or torch.zeros_like(pos.embedding),
                            chosen_embedding=pos.embedding,
                            rejected_embedding=neg.embedding,
                            chosen_reward=pos.get_reward(self.config),
                            rejected_reward=neg.get_reward(self.config),
                        )
                        pairs.append(pair)
                        
                        if max_pairs and len(pairs) >= max_pairs:
                            break
                if max_pairs and len(pairs) >= max_pairs:
                    break
        
        return pairs
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def clear(self) -> None:
        """Clear all samples."""
        self._buffer.clear()
        self._positive.clear()
        self._negative.clear()


# =============================================================================
# Reward Model
# =============================================================================

class RewardModel(nn.Module):
    """Learned reward model from feedback data.
    
    Predicts reward score for a given (prompt, response) pair based on
    learned patterns from feedback data.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize reward model.
        
        Args:
            embed_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Combine prompt and response
        self.combine = nn.Linear(embed_dim * 2, hidden_dim)
        
        # Hidden layers
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.hidden = nn.Sequential(*layers)
        
        # Output reward score
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        prompt_embedding: Tensor,
        response_embedding: Tensor,
    ) -> Tensor:
        """Predict reward for prompt-response pair.
        
        Args:
            prompt_embedding: Prompt embedding [batch, embed_dim].
            response_embedding: Response embedding [batch, embed_dim].
            
        Returns:
            Reward score [batch, 1].
        """
        # Combine embeddings
        combined = torch.cat([prompt_embedding, response_embedding], dim=-1)
        x = self.combine(combined)
        x = F.relu(x)
        
        # Hidden layers
        x = self.hidden(x)
        
        # Output
        reward = self.output(x)
        
        return reward


# =============================================================================
# Preference Optimizer (DPO-style)
# =============================================================================

class PreferenceOptimizer(nn.Module):
    """Direct Preference Optimization for CognitiveTwin.
    
    Implements a simplified DPO-style loss for learning from preference pairs.
    Instead of operating on log-probabilities (like standard DPO), we operate
    on style embeddings to learn which patterns are preferred.
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """Initialize preference optimizer.
        
        Args:
            beta: Temperature for preference learning.
            label_smoothing: Label smoothing for cross-entropy.
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        chosen_embeddings: Tensor,
        rejected_embeddings: Tensor,
        reference_chosen: Optional[Tensor] = None,
        reference_rejected: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute preference loss.
        
        Args:
            chosen_embeddings: Embeddings of preferred responses [batch, dim].
            rejected_embeddings: Embeddings of rejected responses [batch, dim].
            reference_chosen: Reference model embeddings for chosen (optional).
            reference_rejected: Reference model embeddings for rejected (optional).
            
        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size = chosen_embeddings.shape[0]
        
        # Compute similarity scores (as proxy for log-prob)
        if reference_chosen is not None and reference_rejected is not None:
            # With reference model, compute relative difference
            chosen_score = F.cosine_similarity(
                chosen_embeddings, reference_chosen, dim=-1
            )
            rejected_score = F.cosine_similarity(
                rejected_embeddings, reference_rejected, dim=-1
            )
        else:
            # Without reference, use embedding norms as proxy
            chosen_score = chosen_embeddings.norm(dim=-1)
            rejected_score = rejected_embeddings.norm(dim=-1)
        
        # DPO-style loss
        # log sigmoid(beta * (chosen_score - rejected_score))
        logits = self.beta * (chosen_score - rejected_score)
        
        # We want chosen to be higher than rejected
        loss = -F.logsigmoid(logits).mean()
        
        # Metrics
        with torch.no_grad():
            accuracy = (chosen_score > rejected_score).float().mean()
            chosen_mean = chosen_score.mean()
            rejected_mean = rejected_score.mean()
            margin = (chosen_score - rejected_score).mean()
        
        metrics = {
            "preference_loss": loss,
            "preference_accuracy": accuracy,
            "chosen_score_mean": chosen_mean,
            "rejected_score_mean": rejected_mean,
            "preference_margin": margin,
        }
        
        return loss, metrics


# =============================================================================
# Enhanced Contrastive Loss
# =============================================================================

class EnhancedContrastiveLoss(nn.Module):
    """Enhanced contrastive loss with hard negative mining.
    
    Improvements over basic contrastive:
    1. Hard negative mining - focus on hard examples
    2. Margin-based ranking - maintain minimum margin
    3. Multi-positive support - group positives together
    4. Temperature annealing - adapt temperature over training
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 0.5,
        margin: float = 0.2,
        use_hard_negatives: bool = True,
        num_hard_negatives: int = 3,
    ):
        """Initialize enhanced contrastive loss.
        
        Args:
            temperature: Temperature for softmax scaling.
            hard_negative_weight: Weight for hard negative loss.
            margin: Minimum margin for ranking loss.
            use_hard_negatives: Whether to use hard negative mining.
            num_hard_negatives: Number of hard negatives to sample.
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin
        self.use_hard_negatives = use_hard_negatives
        self.num_hard_negatives = num_hard_negatives
    
    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        rewards: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute enhanced contrastive loss.
        
        Args:
            embeddings: Embeddings [batch, dim].
            labels: Class/group labels [batch].
            rewards: Optional reward values for weighting [batch].
            
        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove self-similarity
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - eye_mask
        
        # Standard InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * (1 - eye_mask)  # Remove self
        
        # Positive similarities
        pos_sim = (exp_sim * positive_mask).sum(dim=-1)
        
        # All similarities (denominator)
        all_sim = exp_sim.sum(dim=-1)
        
        # Basic contrastive loss
        basic_loss = -torch.log(pos_sim / (all_sim + 1e-10) + 1e-10)
        basic_loss = basic_loss[positive_mask.sum(dim=-1) > 0].mean()
        
        metrics = {"basic_contrastive": basic_loss}
        total_loss = basic_loss
        
        # Hard negative mining
        if self.use_hard_negatives:
            hard_neg_loss = self._hard_negative_loss(
                sim_matrix, positive_mask, negative_mask
            )
            metrics["hard_negative"] = hard_neg_loss
            total_loss = total_loss + self.hard_negative_weight * hard_neg_loss
        
        # Margin ranking loss
        if self.margin > 0:
            margin_loss = self._margin_ranking_loss(
                sim_matrix, positive_mask, negative_mask
            )
            metrics["margin_ranking"] = margin_loss
            total_loss = total_loss + 0.1 * margin_loss
        
        # Reward weighting
        if rewards is not None:
            reward_weight = torch.abs(rewards).clamp(min=0.1)
            total_loss = (total_loss * reward_weight).mean()
        
        metrics["total_contrastive"] = total_loss
        
        return total_loss, metrics
    
    def _hard_negative_loss(
        self,
        sim_matrix: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
    ) -> Tensor:
        """Compute loss focusing on hard negatives.
        
        Hard negatives are negative samples with high similarity.
        """
        batch_size = sim_matrix.shape[0]
        
        # Find hardest negatives (highest similarity among negatives)
        negative_sim = sim_matrix * negative_mask
        negative_sim[negative_mask == 0] = float('-inf')
        
        # Get top-k hard negatives per sample
        k = min(self.num_hard_negatives, batch_size - 1)
        hard_neg_sim, _ = torch.topk(negative_sim, k, dim=-1)
        
        # Get positive similarities
        positive_sim = sim_matrix * positive_mask
        positive_sim[positive_mask == 0] = float('-inf')
        pos_mean_sim = positive_sim.max(dim=-1).values
        
        # Loss: positives should be higher than hard negatives
        # margin_loss = max(0, hard_neg_sim - pos_sim + margin)
        hard_neg_mean = hard_neg_sim.mean(dim=-1)
        
        loss = F.relu(hard_neg_mean - pos_mean_sim + self.margin)
        
        # Only for samples with positives
        has_positive = positive_mask.sum(dim=-1) > 0
        if has_positive.sum() > 0:
            return loss[has_positive].mean()
        return torch.tensor(0.0, device=sim_matrix.device)
    
    def _margin_ranking_loss(
        self,
        sim_matrix: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
    ) -> Tensor:
        """Compute margin ranking loss."""
        # Get max positive and max negative for each sample
        pos_sim = sim_matrix.masked_fill(positive_mask == 0, float('-inf'))
        neg_sim = sim_matrix.masked_fill(negative_mask == 0, float('-inf'))
        
        max_pos = pos_sim.max(dim=-1).values
        max_neg = neg_sim.max(dim=-1).values
        
        # Margin loss
        loss = F.relu(max_neg - max_pos + self.margin)
        
        # Only for valid samples
        valid = (max_pos > float('-inf')) & (max_neg > float('-inf'))
        if valid.sum() > 0:
            return loss[valid].mean()
        return torch.tensor(0.0, device=sim_matrix.device)


# =============================================================================
# Feedback Learner
# =============================================================================

class FeedbackLearner:
    """Main class for feedback-based learning.
    
    Coordinates feedback collection, buffer management, and model updates.
    """
    
    def __init__(
        self,
        config: FeedbackConfig,
        embed_dim: int = 768,
        device: str = "cpu",
    ):
        """Initialize feedback learner.
        
        Args:
            config: Feedback configuration.
            embed_dim: Embedding dimension.
            device: Torch device.
        """
        self.config = config
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        
        # Buffer
        self.buffer = FeedbackBuffer(config)
        
        # Models
        self.reward_model = RewardModel(embed_dim=embed_dim).to(self.device)
        self.preference_optimizer = PreferenceOptimizer()
        self.contrastive_loss = EnhancedContrastiveLoss()
        
        # Optimizer for reward model
        self.reward_optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=config.learning_rate,
        )
        
        # Stats
        self.update_count = 0
        self.total_feedback = 0
        
        logger.info(f"FeedbackLearner initialized: embed_dim={embed_dim}")
    
    def record_feedback(
        self,
        prompt_id: str,
        response_id: str,
        signal: Union[FeedbackSignal, str],
        value: float = 1.0,
        embedding: Optional[Tensor] = None,
        style_embedding: Optional[Tensor] = None,
        context_embedding: Optional[Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackSample:
        """Record a feedback signal.
        
        Args:
            prompt_id: Prompt identifier.
            response_id: Response identifier.
            signal: Feedback signal type.
            value: Numeric value for ratings/edits.
            embedding: Response embedding.
            style_embedding: Style embedding.
            context_embedding: Context embedding.
            metadata: Additional metadata.
            
        Returns:
            Created feedback sample.
        """
        if isinstance(signal, str):
            signal = FeedbackSignal(signal)
        
        sample = FeedbackSample(
            prompt_id=prompt_id,
            response_id=response_id,
            signal=signal,
            value=value,
            embedding=embedding,
            style_embedding=style_embedding,
            context_embedding=context_embedding,
            metadata=metadata or {},
        )
        
        self.buffer.add(sample)
        self.total_feedback += 1
        
        logger.debug(f"Recorded feedback: {signal.value} for {prompt_id}")
        
        return sample
    
    def should_update(self) -> bool:
        """Check if model should be updated.
        
        Returns:
            True if update threshold reached.
        """
        return len(self.buffer) >= self.config.update_threshold
    
    def update_reward_model(self) -> Dict[str, float]:
        """Update reward model from buffer.
        
        Returns:
            Training metrics.
        """
        samples = self.buffer.get_samples()
        if len(samples) < self.config.min_pairs_for_update:
            return {"status": "not_enough_samples"}
        
        # Prepare training data
        prompt_embeddings = []
        response_embeddings = []
        rewards = []
        
        for sample in samples:
            if sample.embedding is not None and sample.context_embedding is not None:
                prompt_embeddings.append(sample.context_embedding)
                response_embeddings.append(sample.embedding)
                rewards.append(sample.get_reward(self.config))
        
        if len(prompt_embeddings) < self.config.min_pairs_for_update:
            return {"status": "not_enough_embeddings"}
        
        # Stack tensors
        prompt_emb = torch.stack(prompt_embeddings).to(self.device)
        response_emb = torch.stack(response_embeddings).to(self.device)
        reward_targets = torch.tensor(rewards, device=self.device).unsqueeze(-1)
        
        # Train reward model
        self.reward_model.train()
        self.reward_optimizer.zero_grad()
        
        predicted_rewards = self.reward_model(prompt_emb, response_emb)
        loss = F.mse_loss(predicted_rewards, reward_targets)
        
        loss.backward()
        self.reward_optimizer.step()
        
        self.update_count += 1
        
        metrics = {
            "reward_loss": loss.item(),
            "num_samples": len(prompt_embeddings),
            "update_count": self.update_count,
        }
        
        logger.info(f"Updated reward model: loss={loss.item():.4f}")
        
        return metrics
    
    def compute_feedback_loss(
        self,
        style_embeddings: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute feedback-based loss for training.
        
        Args:
            style_embeddings: Current style embeddings [batch, dim].
            labels: Optional group labels for contrastive.
            
        Returns:
            Tuple of (loss, metrics).
        """
        total_loss = torch.tensor(0.0, device=style_embeddings.device)
        metrics = {}
        
        # Get preference pairs
        pairs = self.buffer.get_preference_pairs(max_pairs=32)
        
        if len(pairs) >= self.config.min_pairs_for_update:
            # Stack pair embeddings
            chosen = torch.stack([p.chosen_embedding for p in pairs]).to(self.device)
            rejected = torch.stack([p.rejected_embedding for p in pairs]).to(self.device)
            
            # Preference loss
            pref_loss, pref_metrics = self.preference_optimizer(chosen, rejected)
            total_loss = total_loss + self.config.preference_weight * pref_loss
            metrics.update(pref_metrics)
        
        # Contrastive loss if labels provided
        if labels is not None and len(style_embeddings) > 1:
            # Get rewards for weighting
            samples = self.buffer.get_samples(len(style_embeddings))
            rewards = None
            if len(samples) == len(style_embeddings):
                rewards = torch.tensor(
                    [s.get_reward(self.config) for s in samples],
                    device=self.device,
                )
            
            cont_loss, cont_metrics = self.contrastive_loss(
                style_embeddings, labels, rewards
            )
            total_loss = total_loss + self.config.contrastive_weight * cont_loss
            metrics.update(cont_metrics)
        
        metrics["total_feedback_loss"] = total_loss
        
        return total_loss, metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback learning statistics.
        
        Returns:
            Statistics dictionary.
        """
        samples = self.buffer.get_samples()
        positive = len(self.buffer._positive)
        negative = len(self.buffer._negative)
        
        return {
            "total_feedback": self.total_feedback,
            "buffer_size": len(self.buffer),
            "positive_samples": positive,
            "negative_samples": negative,
            "update_count": self.update_count,
            "preference_pairs": len(self.buffer.get_preference_pairs()),
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_feedback_learner(
    embed_dim: int = 768,
    buffer_size: int = 1000,
    update_threshold: int = 50,
    device: str = "cpu",
    **kwargs,
) -> FeedbackLearner:
    """Create a feedback learner with configuration.
    
    Args:
        embed_dim: Embedding dimension.
        buffer_size: Feedback buffer size.
        update_threshold: Samples before update.
        device: Torch device.
        **kwargs: Additional config options.
        
    Returns:
        Configured FeedbackLearner.
    """
    config = FeedbackConfig(
        buffer_size=buffer_size,
        update_threshold=update_threshold,
        **kwargs,
    )
    
    return FeedbackLearner(
        config=config,
        embed_dim=embed_dim,
        device=device,
    )

