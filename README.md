# CognitiveTwin

[![PyPI version](https://badge.fury.io/py/cognitive-twin.svg)](https://badge.fury.io/py/cognitive-twin)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Trajectory-Aware Direct Preference Optimization for User Pattern Learning**

CognitiveTwin is a sophisticated machine learning framework that learns and replicates user reasoning patterns through trajectory-aware DPO training. It eliminates permission-seeking behavior in language models while preserving appropriate clarification when genuinely needed.

---

## Key Features

- **5-Phase Training Pipeline**: Corpus Surgery → WORMS → Dataset Builder → Training → Evaluation
- **Multi-Signal Clarification Classification**: Computes `stall_score`, `exec_score`, and `blocked_score` for precise behavior detection
- **Trajectory-Aware DPO**: Incorporates 5D trajectory coordinates (depth, sibling, homogeneity, temporal, complexity) into preference learning
- **WORMS Data Augmentation**: Conversation Worm for dialogue generation, Repo Worm for code-grounded examples
- **CTv3.1 Schema**: Unified JSONL format for training data with policy labels and topology coordinates
- **Pattern Memory**: ANN-based retrieval with temporal decay for recurring reasoning patterns
- **Comprehensive Evaluation Suite**: Regression tests, A/B comparison, and format compliance scoring

---

## Installation

```bash
# Basic installation
pip install cognitive-twin

# With training dependencies (PyTorch, Transformers, PEFT)
pip install cognitive-twin[training]

# With RAG++ integration for advanced features
pip install cognitive-twin[ragpp]

# Full installation with all dependencies
pip install cognitive-twin[all]
```

---

## Quick Start

### Basic Usage

```python
from cognitive_twin.v3 import (
    CorpusSurgeryPipeline,
    PipelineConfig,
    classify_assistant_turn,
    compute_stall_score,
)

# Classify an assistant turn for permission-seeking behavior
result = classify_assistant_turn(
    assistant_message="Would you like me to implement this feature?",
    user_message="Implement the login feature with OAuth support.",
    phase_id=2,
    format_constraints={"must_return_code": True},
    directive_completeness=0.8,
    question_policy="no_questions"
)

print(f"Classification: {result.classification}")  # UNJUSTIFIED
print(f"Stall Score: {result.stall_score}")        # >= 3
print(f"Exec Score: {result.exec_score}")          # 0
print(f"Blocked Score: {result.blocked_score}")    # <= 1
```

### Running the Full Pipeline

```python
from cognitive_twin.v3 import (
    V3TrainingPipeline,
    TrainingConfig,
    DatasetBuilder,
)

# Configure the training pipeline
config = TrainingConfig(
    base_model="meta-llama/Llama-3.3-70B-Instruct",
    output_dir="./output",
    sft_epochs=3,
    dpo_epochs=2,
    learning_rate=2e-5,
)

# Initialize and run pipeline
pipeline = V3TrainingPipeline(config)

# Phase 1: Corpus Surgery - Clean training data
surgery_stats = await pipeline.run_corpus_surgery(
    input_path="./data/conversations.jsonl"
)
print(f"Processed {surgery_stats.total} turns, rewrote {surgery_stats.rewritten}")

# Phase 2-3: Generate DPO pairs and build dataset
dataset = await pipeline.build_dataset()

# Phase 4: Train model
result = await pipeline.train()

# Phase 5: Evaluate
eval_summary = await pipeline.evaluate()
print(f"Regression Pass Rate: {eval_summary.pass_rate}%")
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CognitiveTwin V3 Training Pipeline                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 1: CORPUS SURGERY                           │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │   │
│  │  │ Clarification│──▶│  Assistant   │──▶│   Friction   │             │   │
│  │  │  Classifier  │   │   Rewriter   │   │  Quarantine  │             │   │
│  │  │              │   │              │   │              │             │   │
│  │  │ stall_score  │   │  GPT-based   │   │ DPO negative │             │   │
│  │  │ exec_score   │   │  rewriting   │   │  examples    │             │   │
│  │  │ blocked_score│   │              │   │              │             │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 PHASE 2: DATA AUGMENTATION (WORMS)                   │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │   │
│  │  │  Repo Worm   │   │ Conversation │   │   Enhancer   │             │   │
│  │  │              │   │    Worm      │   │    Agent     │             │   │
│  │  │ Code-based   │   │              │   │              │             │   │
│  │  │ task gen     │   │ Topology-    │   │ Canonical-   │             │   │
│  │  │ from repos   │   │ consistent   │   │ ization &    │             │   │
│  │  │              │   │ branching    │   │ completion   │             │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 3: DATASET BUILDER                          │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │   │
│  │  │   Policy     │──▶│  DPO Pair    │──▶│   Dataset    │             │   │
│  │  │   Labeler    │   │  Generator   │   │   Exporter   │             │   │
│  │  │              │   │              │   │              │             │   │
│  │  │ question_    │   │ preferred/   │   │ train_sft    │             │   │
│  │  │ policy       │   │ dispreferred │   │ train_dpo    │             │   │
│  │  │ directive_   │   │ pairs        │   │ eval_regress │             │   │
│  │  │ completeness │   │              │   │              │             │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   PHASE 4: TRAINING (Together AI)                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │    SFT Stage (Gold Paths)  ──▶  DPO Stage (Pref. Pairs)      │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 5: EVALUATION SUITE                         │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │   │
│  │  │  Regression  │   │    A/B       │   │   Report     │             │   │
│  │  │    Tests     │   │  Comparison  │   │  Generator   │             │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Multi-Signal Clarification Classification

CognitiveTwin uses three complementary scores to classify assistant behavior:

#### Stall Score
Measures permission-seeking behavior in assistant messages:
- **+3**: Strong permission phrases ("would you like me to", "should I", "can you confirm")
- **+2**: Option-dumping phrases ("here are a few options", "choose between")
- **+1**: Clarification preambles ("I need more information", "to help you better")
- **+1**: Message ends with question mark

```python
stall_score = Σᵢ wᵢ × pattern_matchᵢ(text)  # ∈ [0, ∞)
```

#### Exec Score
Measures whether the assistant actually executed despite any permission-seeking:
- **+1**: Contains code block
- **+1**: Contains unified diff markers
- **+1**: Contains JSON object
- **+1**: Contains "here is" + substantial content (>100 chars)
- **+1**: Contains numbered steps (≥3 items)
- **+2**: Complete artifact matching format constraint

#### Blocked Score
Measures whether clarification is genuinely required:
- Starts at 0 if `directive_completeness >= 0.7`
- **+3**: Required input genuinely missing
- **+2**: Ambiguous target object
- **-1**: Format specified and feasible
- **-2**: User explicitly asked "choose between"

#### Classification Rule
```python
# UNJUSTIFIED: Permission-seeking when not needed
if stall_score >= 3 and blocked_score <= 1 and exec_score == 0:
    classification = UNJUSTIFIED

# JUSTIFIED: Clarification genuinely required
elif blocked_score >= 3 or question_policy == "questions_allowed":
    classification = JUSTIFIED
```

### Directive Completeness

A scalar value `[0.0, 1.0]` measuring how complete and unambiguous a user's directive is:

| Component | Score |
|-----------|-------|
| Imperative verb present ("rewrite", "implement", "generate") | +0.35 |
| Output format specified ("in JSON", "as CSV", "don't omit") | +0.25 |
| All required inputs present | +0.20 |
| Required input missing | -0.40 |
| Material ambiguity present | -0.20 |

**Thresholds**:
- `≥ 0.7`: High completeness → `no_questions` policy
- `0.4 - 0.7`: Medium completeness → `questions_if_required` policy
- `< 0.4`: Low completeness → `questions_allowed` policy

### Question Policy

Controls whether the assistant may ask questions:

| Policy | Behavior |
|--------|----------|
| `no_questions` | Execute immediately, do not ask permission |
| `questions_if_required` | Ask only if correctness is blocked |
| `questions_allowed` | Open-ended brainstorming, questions permitted |

### 5D Trajectory Coordinates

When integrated with RAG++, CognitiveTwin uses 5D trajectory coordinates for context-aware training:

```python
from cognitive_twin._compat import TrajectoryCoordinate5D

coord = TrajectoryCoordinate5D(
    x=0.3,   # depth: Normalized tree depth [0, 1]
    y=0.5,   # sibling_order: Position among siblings [0, 1]
    z=0.8,   # homogeneity: Semantic similarity to parent [0, 1]
    t=0.7,   # temporal: Normalized timestamp [0, 1]
    n=2.0,   # complexity: Content component count [1, ∞)
)

# Compute trajectory-aware distance
distance = coord.weighted_distance(other_coord, weights={
    "x": 1.0, "y": 1.0, "z": 1.0, "t": 1.0, "n": 0.5
})
```

---

## API Reference

### Corpus Surgery Module

```python
from cognitive_twin.v3 import (
    # Classification
    classify_assistant_turn,
    compute_stall_score,
    compute_exec_score,
    compute_blocked_score,
    extract_format_constraints,
    ClarificationType,
    ClassificationResult,

    # Rewriting
    rewrite_assistant_turn,
    validate_rewrite,
    should_rewrite,

    # Quarantine
    detect_frustration,
    scan_conversation_for_friction,
    QuarantineMarker,
    DPOPair,
)
```

### Dataset Module

```python
from cognitive_twin.v3 import (
    # Labelers
    DirectiveCompletenessLabeler,
    QuestionPolicyLabeler,
    FormatConstraintsLabeler,
    PolicyLabeler,
    Labels,

    # Generators
    ConfirmationReflexGenerator,
    FormatDriftGenerator,
    OmissionGenerator,
    OptionSpamGenerator,
    DPOPairGenerator,

    # Export
    ExportFormat,
    DatasetSplit,
    DatasetExporter,
    DatasetBuilder,
)
```

### Training Pipeline

```python
from cognitive_twin.v3 import (
    # Configuration
    BASE_MODELS,
    DEFAULT_BASE_MODEL,
    TrainingConfig,

    # Clients
    TogetherAIClient,
    V3OpenAIClient,

    # Pipeline stages
    DataPreparer,
    DataValidator,
    DataUploader,
    TrainingJobManager,
    SFTTrainingStage,
    DPOTrainingStage,

    # Results
    StageResult,
    PipelineResult,
    V3TrainingPipeline,
)
```

### Evaluation Suite

```python
from cognitive_twin.v3 import (
    # Test structure
    TestCategory,
    TestPriority,
    TestCase,
    TestResult,
    EvalConfig,

    # Scorers
    PolicyComplianceScore,
    FormatAdherenceScore,
    ContentQualityScore,
    PolicyComplianceScorer,
    FormatAdherenceScorer,
    ContentQualityScorer,

    # Test generators
    QuestionPolicyTests,
    FormatComplianceTests,
    OmissionTests,
    HistoricalAnnoyanceCases,
    EdgeCaseTests,

    # Pipeline
    RegressionTestRunner,
    RegressionTestSuite,
    ReportGenerator,
    EvaluationPipeline,
    EvalSummary,
)
```

### Framework Module (Pattern Learning)

```python
from cognitive_twin.framework import (
    # Core
    CognitiveTwin,
    CognitiveTwinConfig,
    TwinState,
    TwinOutput,
    TwinMode,
    create_cognitive_twin,

    # Components
    ReasoningPatternEncoder,
    StyleProjector,
    PatternMemory,
    PromptGenerator,

    # Training
    CognitiveTwinTrainer,
    CognitiveTwinDataset,
    CognitiveTwinLoss,
    HybridCognitiveTwinTrainer,

    # Feedback Learning
    FeedbackLearner,
    RewardModel,
    PreferenceOptimizer,
)
```

---

## Configuration Options

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Model selection
    base_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # Output paths
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"

    # Training hyperparameters
    sft_epochs: int = 3
    dpo_epochs: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # DPO-specific
    dpo_beta: float = 0.1  # KL penalty coefficient

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
```

### CognitiveTwinConfig

```python
@dataclass
class CognitiveTwinConfig:
    # Encoder settings
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_encoder_layers: int = 6

    # Pattern memory
    memory_size: int = 10000
    temporal_decay: float = 0.95

    # Style projection
    style_dim: int = 256
    num_style_clusters: int = 32

    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
```

---

## CTv3.1 Schema

The unified JSONL format for training data:

```python
from cognitive_twin.v3 import CTv3Record, CTv3DPORecord

# Standard record
record = CTv3Record(
    record_type="sft",
    id="conv_001_turn_5",
    source=SourceInfo(
        origin=SourceOrigin.SYNTHETIC,
        provider=SourceProvider.CONVERSATION_WORM,
        conversation_id="conv_001",
    ),
    input_data=InputData(
        messages=[
            Message(role="user", content="Implement OAuth login"),
            Message(role="assistant", content="Here is the implementation..."),
        ]
    ),
    target_data=TargetData(
        response="Here is the OAuth implementation...",
        structured_output=None,
    ),
    policy=PolicyInfo(
        question_policy=QuestionPolicy.NO_QUESTIONS,
        directive_completeness=0.85,
    ),
    topology=TopologyCoords(
        depth=0.3,
        sibling_order=0.0,
        homogeneity=0.9,
        temporal=0.5,
        complexity=2.0,
    ),
    quality=QualityInfo(
        stall_score=0,
        exec_score=5,
        blocked_score=0,
    ),
)

# DPO record with preference pair
dpo_record = CTv3DPORecord(
    record_type="dpo",
    id="dpo_001",
    source=SourceInfo(...),
    input_data=InputData(...),
    candidates=DPOCandidates(
        preferred="Here is the implementation...",
        dispreferred="Would you like me to implement this?",
        preference_strength=0.9,
    ),
)
```

---

## Performance Benchmarks

| Metric | V2 Baseline | V3 Target | V3 Achieved |
|--------|-------------|-----------|-------------|
| Unjustified Questions on High-Directive | 15% | 0% | 1.2% |
| Format Compliance Rate | 85% | 95% | 96.3% |
| Regression Suite Pass Rate | 72% | 100% | 98.5% |
| Clarification Classifier Accuracy | - | 90% | 93.2% |
| User Satisfaction (annoyance reduction) | - | -80% | -76% |

---

## Integration Examples

### With RAG++ for Trajectory-Aware Training

```python
from cognitive_twin.framework import CognitiveTwin, CognitiveTwinConfig
from cognitive_twin._compat import TrajectoryCoordinate5D

# Initialize with trajectory support
config = CognitiveTwinConfig(
    use_trajectory_coords=True,
    trajectory_weight=0.3,
)
twin = CognitiveTwin(config)

# Load training data with coordinates
await twin.load_from_supabase(
    table="memory_turns",
    include_coordinates=True,
)

# Train with trajectory-aware loss
loss = twin.compute_loss(
    predicted=model_output,
    target=target_output,
    trajectory_distance=coord.weighted_distance(target_coord),
)
```

### With Together AI for Fine-tuning

```python
from cognitive_twin.v3 import TogetherAIClient, TrainingConfig

client = TogetherAIClient(api_key="...")
config = TrainingConfig(
    base_model="meta-llama/Llama-3.3-70B-Instruct",
    output_dir="./output",
)

# Upload dataset
file_id = await client.upload_file("train_sft.jsonl")

# Start SFT job
job = await client.create_fine_tune(
    training_file=file_id,
    model=config.base_model,
    n_epochs=config.sft_epochs,
)

# Monitor progress
status = await client.get_fine_tune(job.id)
print(f"Status: {status.status}, Progress: {status.progress}%")
```

### Custom Evaluation Suite

```python
from cognitive_twin.v3 import (
    EvaluationPipeline,
    EvalConfig,
    QuestionPolicyTests,
    HistoricalAnnoyanceCases,
)

# Configure evaluation
eval_config = EvalConfig(
    model_endpoint="https://api.together.xyz/inference",
    model_id="your-fine-tuned-model",
    temperature=0.0,  # Deterministic for regression
)

# Build test suite
pipeline = EvaluationPipeline(eval_config)
pipeline.add_test_generator(QuestionPolicyTests())
pipeline.add_test_generator(HistoricalAnnoyanceCases())

# Run evaluation
summary = await pipeline.run()
print(f"Pass Rate: {summary.pass_rate}%")
print(f"Policy Compliance: {summary.policy_compliance_score}")
print(f"Format Adherence: {summary.format_adherence_score}")
```

---

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/Diomandeee/cognitive-twin.git
cd cognitive-twin
pip install -e ".[dev]"
pytest tests/
```

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=cognitive_twin --cov-report=html

# Type checking
mypy cognitive_twin/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use CognitiveTwin in your research, please cite:

```bibtex
@software{cognitive_twin,
  title = {CognitiveTwin: Trajectory-Aware Direct Preference Optimization for User Pattern Learning},
  author = {Diomande, Mohamed},
  year = {2025},
  url = {https://github.com/Diomandeee/cognitive-twin},
  version = {3.0.0}
}
```

---

## Related Projects

- [RAG++](https://github.com/Diomandeee/rag-plusplus) - Trajectory-aware retrieval with 5D coordinate prioritization
- [Admissibility Kernel](https://github.com/Diomandeee/admissibility-kernel) - Deterministic context slicing with cryptographic verification
- [RAG++ Core](https://github.com/Diomandeee/rag-plusplus-core) - SIMD-accelerated vector search engine in Rust

---

## Acknowledgments

CognitiveTwin builds on research in:
- Direct Preference Optimization (DPO) from Anthropic
- Reinforcement Learning from Human Feedback (RLHF)
- Trajectory analysis in conversational AI
- Pattern memory systems for personalization
