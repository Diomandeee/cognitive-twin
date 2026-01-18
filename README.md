# CognitiveTwin

User pattern learning with trajectory-aware DPO (Direct Preference Optimization) training.

## Overview

CognitiveTwin is a sophisticated system for learning user communication patterns through:

- **Corpus Surgery**: Data cleaning, validation, and quality filtering
- **WORMS**: Trajectory generators for synthetic training data
  - Conversation Worm: Dialogue trajectory generation
  - Repo Worm: Code repository analysis
  - Task Worm: Task execution patterns
  - DPO Generator: Preference pair generation
- **Dataset Building**: Preference pair labeling and export
- **Evaluation Suite**: Comprehensive testing framework

## Installation

```bash
pip install cognitive-twin

# With training dependencies
pip install cognitive-twin[training]
```

## Quick Start

```python
from cognitive_twin.v3 import pipeline, schema
from cognitive_twin.framework import config

# Initialize pipeline
cfg = config.CognitiveTwinConfig(
    model_name="your-base-model",
    output_dir="./output"
)

# Run corpus surgery
pipeline.run_corpus_surgery(cfg)

# Generate training data
pipeline.generate_dpo_pairs(cfg)

# Train
pipeline.train(cfg)
```

## Components

### v3/ - Main Implementation

- `corpus_surgery/` - Data cleaning and validation
- `dataset/` - Dataset generation and labeling
- `eval/` - Evaluation framework
- `generators/` - Batch and DPO generators
- `ingest/` - Data ingestion (Claude, OpenAI, Supabase)
- `worms/` - Trajectory generators
- `pipeline.py` - Main orchestrator
- `schema.py` - Type definitions

### framework/ - Supporting Infrastructure

- `config.py` - Configuration management
- `twin.py` - Core twin abstraction
- `trainer.py` - Training loop

## Documentation

See the `docs/` directory for detailed documentation:

- `00_OVERVIEW.md` - System overview
- `01_CORPUS_SURGERY.md` - Data cleaning pipeline
- `02_REPO_WORM.md` - Repository analysis
- `03_CONVERSATION_WORM.md` - Dialogue generation
- `04_ENHANCER_AGENT.md` - Quality enhancement
- `05_DATASET_BUILDER.md` - Dataset construction
- `06_TRAINING_PIPELINE.md` - Training guide
- `07_EVALUATION_SUITE.md` - Evaluation metrics
- `08_API_INTEGRATION.md` - API usage

## License

MIT
