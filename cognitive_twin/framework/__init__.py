"""CognitiveTwin - Digital Twin for Learning User Reasoning Patterns.

This module provides a specialized architecture for creating a digital twin
that learns and replicates the user's reasoning patterns, prompting style,
and exploration behavior across conversations.

Core Concepts:
    - **Inverse Attention Learning**: Learn what the user attends to when
      generating responses (R = sum(alpha_i * C_i))
    - **Trajectory-Aware Modeling**: Capture branching behavior, topic
      exploration patterns, and depth preferences using 5D coordinates
    - **Knowledge Transfer Detection**: Identify how knowledge flows across
      conversations (explicit references, pattern replication, methodology)
    - **Style Projection**: Project responses into user's unique style space
    - **Pattern Memory**: Store and retrieve recurring reasoning patterns

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     CognitiveTwin                                │
    │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
    │  │ Reasoning   │  │ Style        │  │ Pattern                 │ │
    │  │ Encoder     │──│ Projector    │──│ Memory                  │ │
    │  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
    │         │                │                      │               │
    │         ▼                ▼                      ▼               │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │              Prompt Generator                               ││
    │  │  (generates prompts/continuations in user's style)          ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘

Key Components:
    - CognitiveTwin: Main orchestrator combining all components
    - ReasoningPatternEncoder: Encodes reasoning patterns via inverse attention
    - StyleProjector: Projects to user's unique style space
    - PatternMemory: Stores recurring patterns with trajectory context
    - PromptGenerator: Generates prompts mimicking user style

Training Data Sources:
    - 100K+ conversation turns from Supabase memory_turns
    - 5D trajectory coordinates (depth, sibling, homogeneity, temporal, complexity)
    - Knowledge transfer patterns across conversations
    - User feedback signals (thumbs_up/thumbs_down)

Example:
    >>> from cognitive_twin.framework import CognitiveTwin, CognitiveTwinConfig
    >>> from rag_plusplus.db.client import SupabaseClient
    >>>
    >>> # Initialize with default config
    >>> config = CognitiveTwinConfig()
    >>> twin = CognitiveTwin(config)
    >>>
    >>> # Load from training data
    >>> twin.load_from_supabase(SupabaseClient())
    >>>
    >>> # Generate continuation prompt
    >>> prompt = twin.generate_prompt(
    ...     context="We were discussing the implementation of IRCP...",
    ...     style_temperature=0.8,
    ... )

Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Version
__version__ = "1.0.0"

# Imports
from cognitive_twin.framework.config import (
    CognitiveTwinConfig,
    ReasoningEncoderConfig,
    StyleProjectorConfig,
    PatternMemoryConfig,
    PromptGeneratorConfig,
)
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
from cognitive_twin.framework.twin import (
    CognitiveTwin,
    TwinState,
    TwinOutput,
    TwinMode,
    create_cognitive_twin,
)
from cognitive_twin.framework.trainer import (
    CognitiveTwinTrainer,
    CognitiveTwinDataset,
    CognitiveTwinLoss,
    ConversationBatch,
    TrainingMetrics,
    TrainingState,
    collate_conversation_batch,
    create_trainer_from_config,
    load_training_data_from_supabase,
)
from cognitive_twin.framework.global_signature import (
    GlobalStyleSignature,
    StyleSignatureVector,
    SignatureSnapshot,
)
from cognitive_twin.framework.hybrid_trainer import (
    HybridCognitiveTwinTrainer,
    HybridConfig,
    TrainingMode,
    TriggerType,
    PromptResponsePair,
    MemoryTurn,
    create_hybrid_trainer,
)
from cognitive_twin.framework.feedback import (
    FeedbackLearner,
    FeedbackConfig,
    FeedbackSignal,
    FeedbackSample,
    FeedbackBuffer,
    RewardModel,
    PreferenceOptimizer,
    EnhancedContrastiveLoss,
    PreferencePair,
    create_feedback_learner,
)
from cognitive_twin.framework.multi_source import (
    MultiSourceFusion,
    MultiSourceConfig,
    SourceType,
    SourceBatch,
    SourceEncoder,
    FusedOutput,
    AttentionFusion,
    GatedFusion,
    CrossModalContrastiveLoss,
    MultiSourceDataset,
    create_multi_source_fusion,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "CognitiveTwinConfig",
    "ReasoningEncoderConfig",
    "StyleProjectorConfig",
    "PatternMemoryConfig",
    "PromptGeneratorConfig",
    # Reasoning Encoder
    "ReasoningPatternEncoder",
    "ReasoningPattern",
    "PatternType",
    # Style Projector
    "StyleProjector",
    "StyleEmbedding",
    "StyleSignature",
    # Pattern Memory
    "PatternMemory",
    "StoredPattern",
    "PatternQuery",
    # Prompt Generator
    "PromptGenerator",
    "GeneratedPrompt",
    "ContinuationContext",
    # Main Twin
    "CognitiveTwin",
    "TwinState",
    "TwinOutput",
    "TwinMode",
    "create_cognitive_twin",
    # Trainer
    "CognitiveTwinTrainer",
    "CognitiveTwinDataset",
    "CognitiveTwinLoss",
    "ConversationBatch",
    "TrainingMetrics",
    "TrainingState",
    "collate_conversation_batch",
    "create_trainer_from_config",
    "load_training_data_from_supabase",
    # Global Signature
    "GlobalStyleSignature",
    "StyleSignatureVector",
    "SignatureSnapshot",
    # Hybrid Trainer
    "HybridCognitiveTwinTrainer",
    "HybridConfig",
    "TrainingMode",
    "TriggerType",
    "PromptResponsePair",
    "MemoryTurn",
    "create_hybrid_trainer",
    # Feedback Learning
    "FeedbackLearner",
    "FeedbackConfig",
    "FeedbackSignal",
    "FeedbackSample",
    "FeedbackBuffer",
    "RewardModel",
    "PreferenceOptimizer",
    "EnhancedContrastiveLoss",
    "PreferencePair",
    "create_feedback_learner",
    # Multi-Source Fusion
    "MultiSourceFusion",
    "MultiSourceConfig",
    "SourceType",
    "SourceBatch",
    "SourceEncoder",
    "FusedOutput",
    "AttentionFusion",
    "GatedFusion",
    "CrossModalContrastiveLoss",
    "MultiSourceDataset",
    "create_multi_source_fusion",
]

logger.debug("CognitiveTwin module initialized (version %s)", __version__)
