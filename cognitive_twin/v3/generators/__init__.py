"""
CognitiveTwin V3 Generators Package

Uses GPT-5-mini for preferred response generation and batch processing
to efficiently leverage the 400K context window.

Key components:
- V2Generator: Uses GPT-5-mini for preferred, base Llama for dispreferred
- DPOFactory: Creates DPO pairs for various failure modes
- BatchGenerator: Batches prompts for efficient generation
"""

from .v2_generator import V2Generator, GeneratorConfig
from .dpo_factory import DPOFactory, DPOPairType
from .batch_generator import (
    BatchGenerator,
    BatchConfig,
    BatchPromptFormatter,
    BatchResponseParser,
    BatchPrompt,
    BatchResult,
)

__all__ = [
    "V2Generator",
    "GeneratorConfig",
    "DPOFactory",
    "DPOPairType",
    "BatchGenerator",
    "BatchConfig",
    "BatchPromptFormatter",
    "BatchResponseParser",
    "BatchPrompt",
    "BatchResult",
]

