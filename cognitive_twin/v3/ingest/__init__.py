"""
CognitiveTwin V3 Data Ingestion Package

Unified data extraction from multiple sources:
- Prompt Logger (MCP + files)
- Claude JSON exports
- OpenAI JSON exports
- Claude Plans (markdown)

All data is normalized to a unified schema before processing.
"""

from .types import (
    UnifiedTurn,
    UnifiedConversation,
    SourceProvider,
    TurnRole,
    ExtractorConfig,
    ExtractionResult,
    NormalizationResult,
    DeduplicationResult,
)
from .prompt_logger import PromptLoggerExtractor
from .claude_json import ClaudeJSONParser
from .openai_json import OpenAIJSONParser
from .normalizer import SchemaNormalizer
from .deduplicator import ContentDeduplicator

__all__ = [
    # Types
    "UnifiedTurn",
    "UnifiedConversation",
    "SourceProvider",
    "TurnRole",
    "ExtractorConfig",
    "ExtractionResult",
    "NormalizationResult",
    "DeduplicationResult",
    # Extractors
    "PromptLoggerExtractor",
    "ClaudeJSONParser",
    "OpenAIJSONParser",
    # Processors
    "SchemaNormalizer",
    "ContentDeduplicator",
]

