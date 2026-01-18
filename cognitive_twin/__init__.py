"""
CognitiveTwin - User pattern learning with trajectory-aware DPO training.

A sophisticated system for learning user communication patterns through:
- Corpus surgery (data cleaning and validation)
- WORMS (trajectory generators for synthetic data)
- Dataset building with preference pairs
- Comprehensive evaluation suite

Components:
- v3: Main implementation (current version)
- framework: Supporting framework (config, trainer, twin)
"""

__version__ = "3.0.0"

from cognitive_twin.v3 import pipeline, schema
from cognitive_twin.framework import config, twin, trainer

__all__ = [
    "pipeline",
    "schema",
    "config",
    "twin",
    "trainer",
    "__version__",
]
