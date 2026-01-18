"""Compatibility layer for optional rag_plusplus dependencies.

This module provides stubs and fallbacks when rag_plusplus is not installed,
allowing cognitive-twin to function in standalone mode with reduced features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Track whether rag_plusplus is available
RAG_PLUSPLUS_AVAILABLE = False

try:
    from rag_plusplus.ml.core.coordinates import TrajectoryCoordinate5D as _RagPlusPlusCoord5D
    from rag_plusplus.ml.core.exceptions import (
        CoordinateDimensionError,
        CoordinateDistanceError,
        CoordinateRangeError,
    )
    RAG_PLUSPLUS_AVAILABLE = True
    TrajectoryCoordinate5D = _RagPlusPlusCoord5D
    logger.debug("rag_plusplus.ml available - using native TrajectoryCoordinate5D")
except ImportError:
    logger.debug("rag_plusplus.ml not available - using stub TrajectoryCoordinate5D")

    # Stub exceptions
    class CoordinateDimensionError(Exception):
        """Stub exception for coordinate dimension errors."""
        pass

    class CoordinateDistanceError(Exception):
        """Stub exception for coordinate distance errors."""
        pass

    class CoordinateRangeError(Exception):
        """Stub exception for coordinate range errors."""
        pass

    @dataclass
    class TrajectoryCoordinate5D:
        """Stub implementation of 5D trajectory coordinates.

        This is a minimal stub for when rag_plusplus is not installed.
        For full functionality, install rag_plusplus with:
            pip install rag-plusplus[ml]

        Coordinate Dimensions:
            x (depth): Normalized tree depth [0, 1]
            y (sibling_order): Position among siblings [0, 1]
            z (homogeneity): Semantic similarity to parent [0, 1]
            t (temporal): Normalized timestamp [0, 1]
            n (complexity): Content component count [1, inf)
        """
        x: float = 0.0  # depth
        y: float = 0.0  # sibling_order
        z: float = 0.0  # homogeneity
        t: float = 0.0  # temporal
        n: float = 1.0  # complexity

        def __post_init__(self):
            """Validate coordinate ranges."""
            pass  # Stub doesn't validate

        def validate(self) -> None:
            """Validate that coordinates are in valid ranges."""
            if not (0.0 <= self.x <= 1.0):
                raise CoordinateRangeError(f"x must be in [0, 1], got {self.x}")
            if not (0.0 <= self.y <= 1.0):
                raise CoordinateRangeError(f"y must be in [0, 1], got {self.y}")
            if not (0.0 <= self.z <= 1.0):
                raise CoordinateRangeError(f"z must be in [0, 1], got {self.z}")
            if not (0.0 <= self.t <= 1.0):
                raise CoordinateRangeError(f"t must be in [0, 1], got {self.t}")
            if self.n < 1.0:
                raise CoordinateRangeError(f"n must be >= 1, got {self.n}")

        def to_tuple(self) -> tuple:
            """Return coordinates as tuple (x, y, z, t, n)."""
            return (self.x, self.y, self.z, self.t, self.n)

        def to_dict(self) -> Dict[str, float]:
            """Return coordinates as dictionary."""
            return {
                "x": self.x, "y": self.y, "z": self.z,
                "t": self.t, "n": self.n
            }

        def euclidean_distance(self, other: "TrajectoryCoordinate5D") -> float:
            """Compute Euclidean distance to another coordinate."""
            import math
            return math.sqrt(
                (self.x - other.x) ** 2 +
                (self.y - other.y) ** 2 +
                (self.z - other.z) ** 2 +
                (self.t - other.t) ** 2 +
                (self.n - other.n) ** 2
            )

        def weighted_distance(
            self,
            other: "TrajectoryCoordinate5D",
            weights: Optional[Dict[str, float]] = None
        ) -> float:
            """Compute weighted Euclidean distance."""
            import math
            if weights is None:
                weights = {"x": 1.0, "y": 1.0, "z": 1.0, "t": 1.0, "n": 0.5}
            return math.sqrt(
                weights.get("x", 1.0) * (self.x - other.x) ** 2 +
                weights.get("y", 1.0) * (self.y - other.y) ** 2 +
                weights.get("z", 1.0) * (self.z - other.z) ** 2 +
                weights.get("t", 1.0) * (self.t - other.t) ** 2 +
                weights.get("n", 0.5) * (self.n - other.n) ** 2
            )

        @classmethod
        def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryCoordinate5D":
            """Create coordinate from dictionary."""
            return cls(
                x=float(d.get("x", 0.0)),
                y=float(d.get("y", 0.0)),
                z=float(d.get("z", 0.0)),
                t=float(d.get("t", 0.0)),
                n=float(d.get("n", 1.0)),
            )


# Attention module stubs
try:
    from rag_plusplus.ml.attention.inverse import (
        InverseAttentionMechanism,
        InverseAttentionConfig,
        AttentionWeights,
    )
    from rag_plusplus.ml.attention.bias import CoordinateBias
except ImportError:
    logger.debug("rag_plusplus.ml.attention not available - using stubs")

    @dataclass
    class InverseAttentionConfig:
        """Stub config for inverse attention."""
        num_heads: int = 8
        hidden_dim: int = 768
        dropout: float = 0.1

    @dataclass
    class AttentionWeights:
        """Stub for attention weights."""
        weights: Any = None
        context_indices: Any = None

    class InverseAttentionMechanism:
        """Stub for inverse attention mechanism."""
        def __init__(self, config: Optional[InverseAttentionConfig] = None):
            self.config = config or InverseAttentionConfig()

        def forward(self, *args, **kwargs):
            raise NotImplementedError(
                "InverseAttentionMechanism requires rag_plusplus. "
                "Install with: pip install rag-plusplus[ml]"
            )

    class CoordinateBias:
        """Stub for coordinate bias."""
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise NotImplementedError(
                "CoordinateBias requires rag_plusplus. "
                "Install with: pip install rag-plusplus[ml]"
            )


# FunctionGemma stubs
try:
    from rag_plusplus.ml.functiongemma.data_format import (
        ToolSchema,
        ToolLayer,
        FunctionCall,
    )
    from rag_plusplus.ml.functiongemma.runtime import (
        FunctionGemmaInference,
        FunctionGemmaRuntime,
        RuntimeConfig,
        RuntimeBackend,
        GenerationResult,
    )
except ImportError:
    logger.debug("rag_plusplus.ml.functiongemma not available - using stubs")

    @dataclass
    class ToolSchema:
        """Stub for tool schema."""
        name: str = ""
        description: str = ""
        parameters: Dict[str, Any] = field(default_factory=dict)
        layer: str = ""  # FunctionGemma layer (L1_INTERPRET, L2_STRUCTURE, L3_EXECUTE)

    class ToolLayer:
        """Stub for tool layer enum."""
        # Original layers
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        # FunctionGemma layers
        L1_INTERPRET = "L1_INTERPRET"
        L2_STRUCTURE = "L2_STRUCTURE"
        L3_EXECUTE = "L3_EXECUTE"

    @dataclass
    class FunctionCall:
        """Stub for function call."""
        name: str = ""
        arguments: Dict[str, Any] = field(default_factory=dict)

    class RuntimeBackend:
        """Stub for runtime backend enum."""
        LOCAL = "local"
        VERTEX = "vertex"
        AUTO = "auto"

    @dataclass
    class RuntimeConfig:
        """Stub for runtime config."""
        backend: str = "local"
        model_path: Optional[str] = None
        temperature: float = 0.0

    @dataclass
    class GenerationResult:
        """Stub for generation result."""
        text: str = ""
        function_call: Optional[FunctionCall] = None
        success: bool = False
        error: Optional[str] = None

    class FunctionGemmaInference:
        """Stub for FunctionGemma inference."""
        def __init__(self, *args, **kwargs):
            pass

        def parse(self, *args, **kwargs):
            raise NotImplementedError(
                "FunctionGemmaInference requires rag_plusplus. "
                "Install with: pip install rag-plusplus[ml]"
            )

    class FunctionGemmaRuntime:
        """Stub for FunctionGemma runtime."""
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *args, **kwargs):
            raise NotImplementedError(
                "FunctionGemmaRuntime requires rag_plusplus. "
                "Install with: pip install rag-plusplus[ml]"
            )


# Database client stubs
try:
    from rag_plusplus.db import SupabaseClient
except ImportError:
    logger.debug("rag_plusplus.db not available - using stub")

    class SupabaseClient:
        """Stub for Supabase client."""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "SupabaseClient requires rag_plusplus. "
                "Install with: pip install rag-plusplus[memory]"
            )


# Service stubs
try:
    from rag_plusplus.service.embedding import EmbedderService
except ImportError:
    logger.debug("rag_plusplus.service not available - using stub")

    class EmbedderService:
        """Stub for embedder service."""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "EmbedderService requires rag_plusplus. "
                "Install with: pip install rag-plusplus[service]"
            )


__all__ = [
    "RAG_PLUSPLUS_AVAILABLE",
    # Coordinates
    "TrajectoryCoordinate5D",
    "CoordinateDimensionError",
    "CoordinateDistanceError",
    "CoordinateRangeError",
    # Attention
    "InverseAttentionMechanism",
    "InverseAttentionConfig",
    "AttentionWeights",
    "CoordinateBias",
    # FunctionGemma
    "ToolSchema",
    "ToolLayer",
    "FunctionCall",
    "FunctionGemmaInference",
    "FunctionGemmaRuntime",
    "RuntimeConfig",
    "RuntimeBackend",
    "GenerationResult",
    # Database
    "SupabaseClient",
    # Service
    "EmbedderService",
]
