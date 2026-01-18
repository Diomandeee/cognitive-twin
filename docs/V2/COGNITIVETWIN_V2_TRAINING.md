# CognitiveTwin V2 Training Pipeline

## Overview

CognitiveTwin V2 is a trajectory-aware, style-learning model that uses **output-level trajectory injection** instead of per-layer injection. This architectural decision avoids issues with complex position embedding schemes (like RoPE) while still enabling trajectory-aware learning.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CognitiveTwin V2                          │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Base Model (Frozen with LoRA)               │  │
│  │  • Llama 3.2 3B / Mistral 7B / Gemma 3               │  │
│  │  • Native forward() handles RoPE internally           │  │
│  │  • Phase 1: LoRA fine-tuning on Together AI           │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           TrajectoryAdapter (Trainable)               │  │
│  │  • 5D coordinate embedding                            │  │
│  │  • Phase-aware gating                                 │  │
│  │  • Homogeneity modulation                             │  │
│  │  • Gated residual connection                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│           ┌───────────────┼───────────────┐                  │
│           ▼               ▼               ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ StyleHead   │  │ PatternHead │  │ PlanningHead│          │
│  │ 256-dim sig │  │ 12 patterns │  │ 8 steps     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Training Phases

### Phase 1: Base Model LoRA Training (Together AI)

Fine-tune the base model with LoRA adapters to learn vocabulary and style patterns.

**Duration:** 10-30 minutes
**Cost:** ~$0.05-0.50 depending on data size

```bash
# Set environment variables
export TOGETHER_API_KEY="your-api-key"
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"

# Run Phase 1 training
python scripts/together_ai_training.py \
    --model llama-3.2-3b \
    --epochs 3 \
    --suffix cognitivetwin

# Monitor job
python scripts/together_ai_training.py --check-job <job-id>

# Wait for completion
python scripts/together_ai_training.py --wait-job <job-id>

# Download model for Phase 2
python scripts/together_ai_training.py --download <job-id> --output-dir ./models/cognitivetwin-lora
```

### Phase 2: Custom Component Training (Local)

Train the custom heads (TrajectoryAdapter, StyleHead, PatternHead, PlanningHead) on top of the frozen base model.

**Duration:** 1-2 hours (GPU) or 4-8 hours (CPU)
**Requirements:** GPU with 16GB+ VRAM recommended

```bash
# Prepare trajectory-aware training data
python scripts/train_custom_heads.py --prepare-data

# Train custom components
python scripts/train_custom_heads.py \
    --base-model ./models/cognitivetwin-lora \
    --data-file ./data/trajectory_training.jsonl \
    --output-dir ./models/cognitivetwin-v2 \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4
```

## Data Formats

### Phase 1: Standard Chat Format

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Phase 2: Trajectory-Aware Format

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "trajectory": {
    "coordinates": [0.5, 0.33, 0.75, 0.5, 0.6],
    "phase_id": 2,
    "homogeneity": 0.75,
    "turn_index": 5
  },
  "conversation_id": "uuid"
}
```

**Trajectory Coordinates (5D):**
1. **depth**: Conversation depth (0-1, normalized position)
2. **sibling_order**: Role ordering (0=system, 0.33=user, 0.67=assistant)
3. **homogeneity**: Topic coherence with previous turn (0-1)
4. **temporal**: Relative time position (0-1)
5. **complexity**: Content complexity score (0-1)

**Phase IDs:**
- 0: Opening (first 10%)
- 1: Context gathering (10-25%)
- 2: Problem exploration (25-50%)
- 3: Solution development (50-75%)
- 4: Refinement (75-90%)
- 5: Closing (last 10%)

## Supported Base Models

| Model | Together AI ID | Hidden Size | Context | Recommended |
|-------|----------------|-------------|---------|-------------|
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | 3072 | 131K | ✅ Best |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B-Instruct` | 2048 | 131K | Fast |
| Llama 3.1 8B | `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference` | 4096 | 32K | More capable |
| Gemma 3 4B | `google/gemma-3-4b-it` | 2560 | 131K | Efficient |
| Qwen 3 4B | `Qwen/Qwen3-4B` | 2560 | 32K | Multilingual |

## Key Design Decisions

### Why Output-Level Injection?

The original T5Gemma-2 approach used per-layer trajectory injection, which required manually iterating through transformer layers. This broke the model's internal position embedding mechanism (RoPE), causing the error:

```
TypeError: cannot unpack non-iterable NoneType object
cos, sin = position_embeddings
```

**Solution:** Apply trajectory adaptation at the OUTPUT level, after the base model's forward pass. This:
- Lets the base model handle RoPE internally
- Simplifies the architecture
- Enables easy base model swapping
- Maintains trajectory-awareness through output fusion

### Why Two-Phase Training?

1. **Stability:** Base model learns vocabulary/style first
2. **Efficiency:** LoRA training on Together AI is fast and cheap
3. **Flexibility:** Custom heads can be retrained without touching base
4. **Resource Optimization:** Phase 1 on cloud, Phase 2 locally

### Why Llama 3.2 3B?

- **Decoder-only:** Simpler architecture than encoder-decoder models
- **Standard RoPE:** Well-documented, widely supported
- **3B parameters:** Fits on consumer GPUs with LoRA
- **131K context:** Handles long conversations
- **Excellent instruction following:** Great for chat tasks

## Files Created

| File | Purpose |
|------|---------|
| `scripts/together_ai_training.py` | Phase 1: Together AI LoRA training |
| `scripts/train_custom_heads.py` | Phase 2: Custom component training |
| `rag_plusplus/ml/cognitivetwin_v2.py` | CognitiveTwinV2 model definition |
| `rag_plusplus/ml/data/trajectory_dataset.py` | Trajectory-aware dataset utilities |

## Usage Example

```python
from rag_plusplus.ml.cognitivetwin_v2 import create_cognitivetwin_v2

# Load the complete model
model = create_cognitivetwin_v2(
    base_model_path="./models/cognitivetwin-lora",
    load_base_model=True,
    device="cuda",
)

# Load custom components trained in Phase 2
model.load_custom_components("./models/cognitivetwin-v2/best")

# Extract style signature
style = model.extract_style(
    input_ids=tokenizer.encode("Hello, how are you?", return_tensors="pt"),
)

# Full forward pass with trajectory
output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    trajectory_coords=torch.tensor([[0.5, 0.33, 0.8, 0.5, 0.6]]),
    phase_id=torch.tensor([2]),
    homogeneity=torch.tensor([[0.8]]),
)

print(f"Style signature: {output.style_signature.shape}")  # [batch, 256]
print(f"Pattern logits: {output.pattern_logits.shape}")    # [batch, 12]
print(f"Plan steps: {output.plan_steps.shape}")            # [batch, 4, hidden]
```

## Troubleshooting

### Phase 1: Together AI Issues

**"TOGETHER_API_KEY not found"**
```bash
export TOGETHER_API_KEY="your-api-key"
# Or add to .env file
```

**"Model not supported for fine-tuning"**
Use exact model IDs from the supported list. Note: Turbo variants may not support serverless LoRA inference.

### Phase 2: Training Issues

**CUDA out of memory**
- Reduce batch size: `--batch-size 2`
- Increase gradient accumulation: `--gradient-accumulation 8`
- Use CPU (slower): `--device cpu`

**"Base model not set"**
Ensure the Phase 1 model was downloaded:
```bash
python scripts/together_ai_training.py --download <job-id>
```

## Cost Estimates

| Phase | Resource | Estimated Cost |
|-------|----------|----------------|
| Phase 1 | Together AI (50K tokens) | $0.05-0.10 |
| Phase 2 | GPU (2 hours) | $2-5 (cloud) or free (local) |
| **Total** | | **~$2-6** |

Compare to full custom training on Vertex AI with T5Gemma-2: **$50+** (due to complex architecture issues).

