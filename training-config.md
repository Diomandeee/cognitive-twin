# Optimal Cognitive Twin Training Configuration

> Target: Qwen2.5-7B-Instruct-4bit on Mac5 (M4 16GB) via MLX LoRA
> Dataset: 2,923 train / 328 valid examples of Mohamed's responses
> Goal: Override "helpful assistant" persona with Mohamed's direct, action-oriented voice
> Date: 2026-03-23
> Extends: lora-persona-research.md (2026-03-22)

---

## Executive Summary

The current training runs have two fundamental problems: (1) the 4-bit quantization requires dramatically different hyperparameters than full-precision LoRA, and (2) persona transfer demands comprehensive layer coverage that conflicts with the 16GB memory ceiling. The solution is a specific combination of conservative learning rate (5e-5), gradient checkpointing, LoRA rank 32 on all layers with attention+MLP targeting, batch 1 with grad accumulation of 8, and the `--mask-prompt` flag. For the anticipation geometry integration, the scalars should be embedded into the system prompt as conditioning context at inference time, not injected into the training loop. The knowledge graph integrates via a Parametric-RAG pattern: query at inference, inject retrieved triples into the prompt.

---

## Question 1: Safe Learning Rate for 4-Bit Quantized Models

### The Problem

The training log shows a clear pattern:
- **1e-5**: Converges but too slow to override instruct persona (val loss 1.796, still sounds generic)
- **2e-4**: Gradient explosion (loss -> NaN at iter 40)
- **5e-5 with mask-prompt**: Running, val loss 2.377 at iter 800 (higher due to mask-prompt, which is expected)

### Root Cause

4-bit NormalFloat (NF4) quantization introduces quantization noise into the forward pass. When gradients flow back through quantized weights, the effective gradient magnitudes are noisier and more volatile than with full-precision weights. The QLoRA paper (Dettmers et al., NeurIPS 2023) found this can cause sudden gradient spikes during backpropagation, particularly in attention layers where the Q/K dot product amplifies small weight perturbations quadratically.

At LR 2e-4, these amplified gradients exceed the stable training region. The NaN at iter 40 is consistent with a gradient spike accumulating over ~40 updates until the loss landscape escapes the local basin entirely.

### Evidence

- **QLoRA paper**: Uses max gradient norm of **0.3** (much more aggressive clipping than the typical 1.0). This is critical for 4-bit stability.
  Source: https://arxiv.org/abs/2305.14314

- **Unsloth guide**: Recommends 2e-4 as starting LR, but this is for their custom kernel with built-in stability patches. Stock MLX does not have Unsloth's gradient accumulation fixes.
  Source: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

- **MLX LoRA on 4-bit**: The mlx-lm framework does QLoRA automatically when the base model is quantized. But it uses default gradient clipping (1.0), not the 0.3 that Dettmers found essential.
  Source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md

### Recommendation

**Use LR 5e-5** (the currently running config). This is the right ballpark. The 2e-4 recommendation from the previous research assumed full-precision LoRA. For 4-bit:

| Precision | Safe LR Range | Sweet Spot |
|-----------|--------------|------------|
| Full (BF16/FP16) | 1e-4 to 3e-4 | 2e-4 |
| 8-bit quantized | 5e-5 to 1e-4 | 1e-4 |
| 4-bit (NF4/QLoRA) | 2e-5 to 8e-5 | **5e-5** |

Additional stability measures:
- MLX does not expose a `--max-grad-norm` flag directly, but the training loop in `mlx_lm/tuner/trainer.py` uses `nn.utils.clip_grad_norm` with a default. If you can modify the training script, set clip norm to **0.3** (matching QLoRA paper).
- Use **cosine schedule with warmup** (5-10% of total iters) to avoid the initial LR being too high before the optimizer state stabilizes.
- The val loss of 2.377 at iter 800 with mask-prompt is **expected to be higher** than 1.796 without mask-prompt. Mask-prompt computes loss only on assistant tokens (shorter sequences, harder prediction task). Compare masked-to-masked and unmasked-to-unmasked, not across modes.

---

## Question 2: DoRA on 16GB with 7B-4bit

### The Problem

DoRA OOMs at every layer count tested:
- 28 layers: OOM
- 16 layers: OOM
- 12 layers: OOM

Standard LoRA at 16 layers with same LR (2e-4) causes gradient explosion, but at least it fits in memory.

### Root Cause

DoRA decomposes each weight matrix W into magnitude (m) and direction (V/||V||). During training, it must:
1. Store the magnitude vector m for each adapted layer
2. Compute the norm ||V + BA|| at each forward pass (additional activation memory)
3. Backpropagate through the normalization, which requires storing intermediate norms

This adds approximately **40-60% memory overhead** compared to standard LoRA at the same rank, because:
- The magnitude vector m is small, but the norm computation ||V + BA|| materializes a full-size weight matrix in the forward pass
- Gradient checkpointing interacts poorly with DoRA's normalization (the recomputed norm may differ slightly due to FP arithmetic, creating gradient noise)

Source: https://medium.com/@AntonioVFranco/qdora-explained-the-new-peft-standard-for-2025-5cf59afeb6ba

### Solutions (Ranked by Feasibility)

**Option A: DoRA with very low rank (rank 8, all layers)**
DoRA outperforms LoRA at lower ranks, so reduce rank to compensate for memory. DoRA at rank 8 may match LoRA at rank 32 for style transfer.

```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --fine-tune-type dora \
  --num-layers -1 \
  --rank 8 \
  --learning-rate 5e-5 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --grad-checkpoint \
  --mask-prompt \
  --iters 2000
```

Memory estimate with DoRA rank 8:
- Model: ~4.0 GB
- DoRA adapters (rank 8, 28 layers, attention+MLP): ~0.03 GB
- Gradient state (AdamW + magnitude vectors): ~0.1 GB
- Activations (batch=1, with grad checkpoint): ~3-4 GB
- DoRA norm overhead: ~1-2 GB
- Framework: ~1 GB
- **Total: ~9-11 GB** (should fit)

**Option B: Standard LoRA with higher rank (rank 32, all layers)**
If DoRA still OOMs even at rank 8, use standard LoRA at rank 32 with all layers. This is the safer path.

**Option C: Use the 3B model (see Question 3)**
The 3B-4bit model uses ~2GB for weights, freeing 2GB for DoRA overhead. DoRA rank 32 on all layers of the 3B would fit comfortably.

### Recommendation

Try Option A first. If it OOMs, immediately fall back to Option B (LoRA rank 32, all layers). Do not waste time on intermediate DoRA configurations. The rank reduction from 64 to 8 is a 64x reduction in adapter parameters, which should more than compensate for DoRA's ~50% overhead.

---

## Question 3: Smaller Base Model (3B or 1.5B)

### The Analysis

| Factor | Qwen2.5-3B-Instruct-4bit | Qwen2.5-7B-Instruct-4bit |
|--------|--------------------------|--------------------------|
| Model memory | ~2.0 GB | ~4.0 GB |
| Available for training | ~14 GB | ~12 GB |
| Layers | 36 | 28 |
| Hidden dim | 2048 | 3584 |
| Heads | 16 | 28 |
| Instruct conditioning depth | Lower (less to override) | Higher (harder to override) |
| Base capabilities (code, reasoning) | Good for 3B, weaker on complex tasks | Strong |
| Persona steerability | Higher (fewer competing parameters) | Lower (deep instruct identity) |
| MLX community model | `mlx-community/Qwen2.5-3B-Instruct-4bit` | `mlx-community/Qwen2.5-7B-Instruct-4bit` |

Source: https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit

### Key Finding: 3B with All Layers > 7B with 8 Layers

The Qwen2.5 technical report shows that 3B approaches 7B-level performance on many benchmarks. The gap is real but narrower than the 2.3x parameter difference suggests. The 3B model has **36 layers** (more than 7B's 28), meaning style information is distributed across more, thinner layers.

For persona override specifically:
- The 3B model has weaker instruct conditioning, meaning the LoRA adapters face less resistance
- All 36 layers can be adapted with DoRA rank 32 and still fit in 12GB
- The shorter hidden dimension (2048 vs 3584) means each layer's adapter is smaller

Source: https://arxiv.org/pdf/2412.15115

### When to Use 3B vs 7B

**Use 3B if:**
- The cognitive twin only needs to handle conversational routing, delegation, and short directives (Mohamed's typical responses)
- DoRA is important to you (3B gives DoRA headroom)
- You want to iterate faster on training experiments (3B trains ~2x faster)

**Stay on 7B if:**
- The twin needs to generate substantive code, debug complex issues, or write extended responses
- You need the stronger base reasoning for tool chain planning (the KARL use case)
- You plan to serve it as a general-purpose assistant, not just a personality layer

### Recommendation

**Run a parallel experiment on 3B.** The marginal cost is one training run. Compare output quality side-by-side on the same 20 validation prompts. If the 3B persona sounds right and handles the actual use cases, it is the better choice because it allows more aggressive adaptation (higher rank, DoRA, all layers).

Specific 3B config:
```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --fine-tune-type dora \
  --num-layers -1 \
  --rank 32 \
  --learning-rate 5e-5 \
  --batch-size 2 \
  --grad-accumulation-steps 4 \
  --grad-checkpoint \
  --mask-prompt \
  --iters 2000 \
  --adapter-path ~/adapters/persona-3b-dora-v1
```

Memory estimate for 3B DoRA rank 32:
- Model: ~2.0 GB
- DoRA adapters (rank 32, 36 layers): ~0.08 GB
- Gradient state: ~0.25 GB
- Activations (batch=2, grad checkpoint): ~4-5 GB
- DoRA norm overhead: ~1.5 GB
- Framework: ~1 GB
- **Total: ~9-10 GB** (comfortable fit with headroom)

---

## Question 4: Which Layers Matter for Persona/Style Transfer

### Research Findings

Recent ablation studies (UW 2026, Raschka 2025, Databricks 2025) converge on a clear hierarchy:

**Layer importance for style/personality transfer:**

1. **o_proj (output projection)**: The single most impactful module. Mixes representations across attention heads into a unified form. Controls how the model's internal representation maps to output token probabilities. Style is encoded here because it governs the "how to say it" transformation.
   Source: https://nwquantum.uw.edu/2026/03/19/optimizing-lora-target-module-selection-for-efficient-fine-tuning/

2. **q_proj + v_proj (query + value projections)**: Control what the model attends to and what it retrieves. For persona, these determine which aspects of the input context influence the generation (e.g., "attend to task urgency" vs "attend to social niceties").
   Source: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

3. **gate_proj + up_proj + down_proj (MLP layers)**: Where factual knowledge and stylistic patterns are stored. Raschka found enabling MLP adaptation increased performance "noticeably" vs attention-only.
   Source: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

4. **k_proj (key projection)**: Least impactful for style. Controls what tokens "advertise" about themselves for attention. More relevant for factual recall than for generation style.

### The Surprising Finding

Over-parameterization can HURT: adapting all 7 module types simultaneously sometimes decreased performance vs adapting 3-4 (qkv + o_proj). The interference hypothesis suggests that adapting every linear layer creates competing gradients that partially cancel.
Source: https://nwquantum.uw.edu/2026/03/19/optimizing-lora-target-module-selection-for-efficient-fine-tuning/

### Recommendation for Persona Transfer

Target these modules in order of priority:
```
q_proj, v_proj, o_proj    (attention: what to attend to, how to mix)
gate_proj, down_proj       (MLP: style patterns)
```

Skip k_proj and up_proj initially. If results are good, you have headroom to add them. If results are bad, you have fewer parameters to debug.

**MLX-specific note:** The `--num-layers` flag in MLX controls how many layers get adapters. The target modules within each layer are controlled by the LoRA config. By default, MLX adapts `self_attn.q_proj`, `self_attn.v_proj` when using LoRA. To adapt additional modules, you need a YAML config:

```yaml
# lora_config.yaml
lora_parameters:
  keys:
    - self_attn.q_proj
    - self_attn.v_proj
    - self_attn.o_proj
    - mlp.gate_proj
    - mlp.down_proj
  rank: 32
  alpha: 64
  dropout: 0.05
  scale: 2.0
```

Pass via: `--lora-config lora_config.yaml`

---

## Question 5: Optimal LoRA Rank for 3K Examples

### The Calculation

The heuristic from scaling law research: trainable parameters should roughly match the information content of the dataset, which can be estimated by total token count.

**Dataset stats:**
- 2,923 training examples
- Average assistant response: ~50 tokens (Mohamed's responses are short)
- With mask-prompt: only assistant tokens contribute gradient
- Effective training tokens: ~2,923 x 50 = ~146K tokens

**Rank to parameter mapping (Qwen2.5-7B, 28 layers, 5 target modules):**
- Rank 8: ~2.2M params (each adapter: 2 x rank x hidden_dim per module)
- Rank 16: ~4.5M params
- Rank 32: ~9.0M params
- Rank 64: ~18M params

The heuristic suggests rank 16-32 is appropriate for 146K effective tokens. Rank 64 risks overfitting unless regularized.

### Empirical Evidence

- Raschka: rank 256 optimal for diverse multi-task with 50K+ examples. For 3K single-task (persona), this would massively overfit.
- Unsloth: "If a model suffers from overfitting, decreasing rank is the first candidate."
- The LIMA paper showed 1,000 curated examples can match 50,000 synthetic. Quality > quantity, but rank must match quantity.

### Recommendation

**Use rank 32 with alpha 64 (2x ratio).**

| Rank | Risk for 3K examples | Recommendation |
|------|---------------------|----------------|
| 8 | Underfitting (may not capture voice nuances) | Only with DoRA |
| 16 | Borderline (captured task patterns but not voice in v1) | Second choice |
| **32** | **Sweet spot (9M params for 146K tokens)** | **Primary choice** |
| 64 | Overfitting risk without strong regularization | Only with weight decay 0.1 + dropout 0.1 |
| 128+ | Will overfit | Do not use |

Add regularization to protect against overfitting at rank 32:
- LoRA dropout: 0.05 (not too aggressive, just prevents memorization)
- Weight decay: 0.01 (standard AdamW regularization)
- Early stopping: monitor val loss, stop when it stops improving for 200 iters

---

## Question 6: Gradient Accumulation vs Larger Batch Sizes

### Why Gradient Accumulation Is Mandatory Here

On 16GB with a 7B-4bit model, batch size 2 may already be tight with all-layer adaptation. Batch size 4+ will OOM.

Gradient accumulation simulates larger batches by accumulating gradients over N forward passes before stepping the optimizer:

```
Effective batch size = batch_size x grad_accumulation_steps
```

**Recommended:** batch_size=1, grad_accumulation_steps=8, giving effective batch size 8.

### Why Effective Batch Size 8, Not 4 or 16

- **Batch 4 (1x4):** Too noisy for persona learning. Each step sees only 4 examples, and persona signal is subtle. The optimizer oscillates.
- **Batch 8 (1x8):** Good balance. Smooth gradients, reasonable training speed. Standard for LoRA SFT.
- **Batch 16 (1x16):** Smoother but halves the number of optimizer steps per epoch. With only 2,923 examples and 2000 iters, you get ~2000/16=125 unique batches per "epoch". This might under-train.

### Important MLX Note

MLX recently fixed a bug where gradient accumulation produced different results than equivalent larger batches. Verify you are on mlx-lm >= 0.25 where this was corrected. On older versions, results may differ depending on how you achieve the same effective batch.

### Recommendation

```
--batch-size 1 --grad-accumulation-steps 8
```

This gives 2000 iters x 1 example = 2000 forward passes, with 2000/8 = 250 optimizer steps. Each optimizer step averages gradients over 8 examples. Over 2000 iters, you see 2000/2923 = 68% of the dataset, which is about 0.7 epochs. To see the full dataset ~2 times, increase iters to 6000.

---

## Question 7: Weight Decay to Prevent Adapter Divergence

### Why Weight Decay Matters for 4-Bit

In full-precision training, LoRA adapters start at zero (A is random, B is zero) and grow slowly toward the target update. Weight decay pulls them back toward zero, preventing overshoot.

With 4-bit quantization, the gradient noise from quantization can push adapters further from the optimal region. Weight decay acts as a regularizing force that combats this drift.

### Specific Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Weight decay | 0.01 | Standard AdamW default. Prevents adapter weights from growing too large. |
| LoRA dropout | 0.05 | Light regularization. Randomly zeros adapter contributions during training. |
| Alpha/rank ratio | 2.0 | Alpha=64 for rank 32. This scales the effective LR for the adapters. |

### Do NOT Use High Weight Decay

Weight decay above 0.1 will fight the learning signal too aggressively. The adapter needs to make meaningful updates to override instruct persona. Values of 0.01-0.05 are the sweet spot.

### Cosine Schedule with Warmup

The learning rate schedule itself is a form of implicit regularization:

```
Warmup: 5% of total iters (100 steps for 2000 total)
Schedule: cosine decay from 5e-5 to ~5e-7
```

This prevents the early training from making oversized updates (when the optimizer momentum is uninitialized) and gradually reduces the LR as the model converges, preventing late-stage divergence.

---

## Question 8: Anticipation Geometry as Conditioning Signals

### What the 7 Scalars Provide

The anticipation-geometry package (`~/Desktop/anticipation-geometry/`) computes 7 geometric scalars from conversation trajectories:

1. **Commitment**: How locked-in the trajectory is (topic stability)
2. **Uncertainty**: How many future directions remain
3. **Transition pressure**: Rate of convergence toward a decision
4. **Recovery margin**: Ease of backtracking
5. **Phase stiffness**: Rigidity of current motion (autocorrelation of velocity)
6. **Novelty**: Distance to nearest historical state
7. **Stability**: Smoothness of trajectory (inverse jerk)

The `GeometryBridge` class (`transformer/geometry_bridge.py`) already converts between numpy-based geometry computation and PyTorch tensors. It can compute these scalars from embedding trajectories via `embeddings_to_scalars()` or directly from token IDs via `batch_compute()`.

### Can These Be Used During LoRA Training?

**Short answer: Not directly with MLX's LoRA trainer, but there are two viable integration paths.**

**Path A: Prompt-Level Conditioning (Recommended)**

Embed the geometry scalars as structured text in the system prompt or user context during both training and inference:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Mohamed. Direct, casual, action-oriented. [trajectory: commitment=0.82 uncertainty=0.15 pressure=0.61 novelty=0.23 stability=0.91 regime=committed]"
    },
    { "role": "user", "content": "Should we refactor or patch?" },
    { "role": "assistant", "content": "Patch it. Ship the fix." }
  ]
}
```

This works because:
- The LoRA adapter learns to attend to the trajectory tags in the system prompt
- At inference time, you compute the live trajectory scalars using the conversation trajectory builder (`conversation_trajectory.py`) and inject them into the prompt
- No modification to the MLX training loop required
- The model learns the correlation between "high commitment + high stability" and Mohamed's decisive responses vs "high uncertainty + low commitment" and Mohamed's exploratory responses

**Implementation:**
1. Pre-process training data: for each conversation, compute geometry scalars from the sequence of prior turns
2. Inject the scalar summary into the system prompt
3. Train with the augmented data
4. At inference, use `ConversationTrajectory.analyze_turns()` to compute live scalars

**Path B: Custom Training Loop with Auxiliary Loss (Not Recommended for Now)**

The anticipatory transformer architecture (`transformer/model.py`) shows how to use the scalars as training supervision via additive trajectory bias in attention:

```python
attention_scores = (Q @ K^T) / sqrt(d_k) + trajectory_bias(scalars)
```

This would require:
1. Forking the MLX LoRA training loop
2. Adding a `GeometryBridge` that works with MLX arrays (not PyTorch)
3. Computing scalars per-batch and feeding them as auxiliary input
4. Adding a scalar prediction loss alongside the language modeling loss

This is architecturally elegant but requires significant engineering. Save it for v2 after the basic persona adapter works.

### Recommendation

**Use Path A.** Add geometry scalars to the system prompt in training data. This gets 80% of the benefit with 5% of the engineering effort. The model will learn to condition its response style on the trajectory state without any custom training code.

Pre-processing script sketch:
```python
from anticipation_geometry.conversation_trajectory import ConversationTrajectory

ct = ConversationTrajectory()

for example in training_data:
    # Get prior conversation turns (if available)
    prior_turns = get_prior_turns(example)
    if len(prior_turns) >= 3:
        analysis = ct.analyze_turns(prior_turns)
        packet = analysis.packet
        t = len(prior_turns) - 1
        regime = packet.regime_at(t)
        tag = (f"[trajectory: c={packet.commitment[t]:.2f} "
               f"u={packet.uncertainty[t]:.2f} "
               f"p={packet.transition_pressure[t]:.2f} "
               f"regime={regime}]")
    else:
        tag = "[trajectory: cold-start]"

    example["messages"][0]["content"] += f" {tag}"
```

---

## Question 9: Live Knowledge Graph Integration at Inference Time

### Architecture

The live knowledge graph (`~/Desktop/live-knowledge-graphs/`) provides:
- `KGClient`: HTTP client for the Graph Kernel API (:8001)
- `kg_reward.py`: Path-based reward function for trajectory scoring
- Traversal, query, and context slicing APIs

### Integration Pattern: Parametric-RAG

Recent research on Parametric-RAG (PRAG) and LoRA-Augmented Generation (LAG) shows the optimal pattern:

1. **At inference time**, before generating the response:
   a. Extract key entities from the user's prompt
   b. Query the KG for relevant triples: `kg.query(subject="spore", min_confidence=0.7)`
   c. Traverse for context: `kg.traverse("spore", max_hops=2)`
   d. Format the KG context as structured text
   e. Inject into the system prompt alongside the trajectory scalars

2. **The LoRA adapter handles personality, the KG handles knowledge.**
   The adapter does not need to memorize facts about projects. The KG provides that at inference time, keeping the adapter focused purely on voice/style.

Source: https://arxiv.org/html/2507.05346v2

### Concrete Implementation

```python
from live_knowledge_graphs.kg_client import KGClient
from anticipation_geometry.conversation_trajectory import ConversationTrajectory

kg = KGClient("http://localhost:8001")
ct = ConversationTrajectory()

def build_context(user_prompt: str, conversation_history: list[str]) -> str:
    """Build enriched system prompt with KG + trajectory context."""

    # 1. Extract entities from prompt (simple keyword match or embedding-based)
    entities = extract_entities(user_prompt)

    # 2. Query KG for relevant context
    kg_context = []
    for entity in entities[:3]:  # Limit to avoid prompt bloat
        triples = kg.query(subject=entity, limit=5)
        for t in triples:
            kg_context.append(f"{t.subject} {t.predicate} {t.object}")

    # 3. Compute trajectory geometry
    if len(conversation_history) >= 3:
        analysis = ct.analyze_turns(conversation_history[-10:])
        t = len(conversation_history) - 1
        p = analysis.packet
        traj_tag = (f"c={p.commitment[min(t, p.trajectory_length-1)]:.2f} "
                    f"u={p.uncertainty[min(t, p.trajectory_length-1)]:.2f} "
                    f"regime={analysis.regime_labels[-1]}")
    else:
        traj_tag = "cold-start"

    # 4. Build system prompt
    system = "You are Mohamed. Direct, casual, action-oriented. Short sentences."
    if kg_context:
        system += f"\n\nContext: {'; '.join(kg_context[:5])}"
    system += f"\n[trajectory: {traj_tag}]"

    return system
```

### Why Not Bake KG Into Training?

The knowledge graph changes constantly (new projects, evolving relationships). If you train the adapter on KG snapshots, it learns stale facts that become wrong within days. The adapter should learn VOICE (stable), while the KG provides KNOWLEDGE (dynamic). This separation of concerns is the same principle behind RAG vs fine-tuning for factual tasks.

### Advanced: LoRA Adapter Selection via KG

IBM Research's inference-time adapter work suggests a future path: maintain multiple specialized LoRA adapters (one per domain: iOS, infra, content) and use the KG to route to the correct adapter based on the conversation's entity context. This is the PRAG pattern applied to multi-adapter systems.

Source: https://research.ibm.com/blog/inference-friendly-aloras-lora

---

## The Optimal Training Command

### Primary Config (LoRA, conservative, should work on first try)

```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data ~/projects/karl/persona-clean/ \
  --fine-tune-type lora \
  --num-layers -1 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --iters 4000 \
  --grad-checkpoint \
  --mask-prompt \
  --val-batches 25 \
  --steps-per-eval 200 \
  --steps-per-report 20 \
  --adapter-path ~/adapters/persona-7b-lora-v1
```

With a LoRA config YAML:
```yaml
# ~/projects/karl/persona-lora-config.yaml
lora_parameters:
  keys:
    - self_attn.q_proj
    - self_attn.v_proj
    - self_attn.o_proj
    - mlp.gate_proj
    - mlp.down_proj
  rank: 32
  alpha: 64
  dropout: 0.05
  scale: 2.0
```

### Aggressive Config (DoRA, try after primary succeeds)

```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data ~/projects/karl/persona-clean/ \
  --fine-tune-type dora \
  --num-layers -1 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --iters 4000 \
  --grad-checkpoint \
  --mask-prompt \
  --val-batches 25 \
  --steps-per-eval 200 \
  --steps-per-report 20 \
  --adapter-path ~/adapters/persona-7b-dora-v1
```

DoRA config YAML (lower rank to fit in memory):
```yaml
# ~/projects/karl/persona-dora-config.yaml
lora_parameters:
  keys:
    - self_attn.q_proj
    - self_attn.v_proj
    - self_attn.o_proj
    - mlp.gate_proj
    - mlp.down_proj
  rank: 8
  alpha: 16
  dropout: 0.05
  scale: 2.0
```

### 3B Parallel Experiment

```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --data ~/projects/karl/persona-clean/ \
  --fine-tune-type dora \
  --num-layers -1 \
  --batch-size 2 \
  --grad-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --iters 4000 \
  --grad-checkpoint \
  --mask-prompt \
  --val-batches 25 \
  --steps-per-eval 200 \
  --steps-per-report 20 \
  --adapter-path ~/adapters/persona-3b-dora-v1
```

3B DoRA config (higher rank because more memory available):
```yaml
# ~/projects/karl/persona-3b-dora-config.yaml
lora_parameters:
  keys:
    - self_attn.q_proj
    - self_attn.v_proj
    - self_attn.o_proj
    - mlp.gate_proj
    - mlp.down_proj
  rank: 32
  alpha: 64
  dropout: 0.05
  scale: 2.0
```

---

## Experiment Priority Queue

Run these in order. Each takes 30-60 minutes on Mac5.

| # | Config | Goal | Stop if |
|---|--------|------|---------|
| 1 | **7B LoRA r32, all layers, LR 5e-5, mask-prompt** | Establish baseline that converges | OOM or val loss not decreasing by iter 500 |
| 2 | **3B DoRA r32, all layers, LR 5e-5, mask-prompt** | Test if 3B persona quality matches 7B | Converges but persona sounds thin |
| 3 | **7B DoRA r8, all layers, LR 5e-5, mask-prompt** | Test DoRA advantage at low rank | OOM (try r4) or worse than #1 |
| 4 | **Winner of 1-3, but with geometry-augmented data** | Test anticipation geometry conditioning | No measurable difference in persona quality |
| 5 | **Winner + DPO stage** | Kill remaining AI-isms via preference learning | Time/data budget exhausted |

---

## Evaluation Protocol

After each training run, test with these 10 prompts. Score each response 0 (full AI slop) to 3 (sounds like Mohamed):

```
1. "The deploy failed with a 503 error on the staging server."
2. "Should we refactor the auth module or just patch the bug?"
3. "The PR is ready for review."
4. "We need to decide between Kafka and RabbitMQ for the event bus."
5. "I'm stuck on this TypeScript type error, can you help?"
6. "What's the status of the Spore app update?"
7. "The CI pipeline is taking 45 minutes. Can we speed it up?"
8. "Should I write tests for this or ship it as-is?"
9. "The client wants a progress update by Friday."
10. "We're running low on compute credits."
```

**Passing score:** Average >= 2.0 across all 10.
**Mohamed's expected voice:** Direct imperative ("Show me the logs"), no filler ("I understand that must be..."), action-oriented ("Patch it, ship the fix"), casual register.

---

## Contrarian Analysis: What If None of This Works?

If after all experiments the model still sounds generic:

1. **The problem may be data, not config.** 2,923 examples of mostly short directives ("continue", "push it", "merge it") may not have enough stylistic signal for the model to learn a distinct voice. Consider augmenting with longer-form Mohamed responses from Supabase memory_turns where his responses are multi-sentence.

2. **The problem may be evaluation, not training.** The model might have learned the persona but revert to instruct mode on novel prompts. Test with prompts similar to the training distribution, not just adversarial cases.

3. **Consider a classifier approach instead.** Rather than overriding the base model's persona, train a lightweight classifier that scores each generated token for "Mohamed-ness" and use it to rerank candidates during generation. This avoids the personality override problem entirely.

4. **Prompt engineering may be sufficient.** A well-crafted system prompt with 5-10 few-shot examples of Mohamed's voice, combined with the KG context and trajectory scalars, may achieve 80% of the persona effect without any fine-tuning at all. Test this as a baseline before investing more in adapter training.

---

## Sources

### Papers
- QLoRA: Efficient Finetuning of Quantized LLMs (NeurIPS 2023): https://arxiv.org/abs/2305.14314
- DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024): https://arxiv.org/pdf/2402.09353
- Qwen2.5 Technical Report: https://arxiv.org/pdf/2412.15115
- LoRA-Augmented Generation (LAG): https://arxiv.org/html/2507.05346v2
- Parametric-RAG (PRAG): https://www.emergentmind.com/topics/parametric-rag-prag
- LoRA Target Module Selection (UW, 2026): https://nwquantum.uw.edu/2026/03/19/optimizing-lora-target-module-selection-for-efficient-fine-tuning/

### Practitioner Guides
- Sebastian Raschka, Practical Tips for Finetuning LLMs Using LoRA: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- Unsloth LoRA Hyperparameters Guide: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- MLX-LM LoRA Documentation: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- QDoRA Explained: https://medium.com/@AntonioVFranco/qdora-explained-the-new-peft-standard-for-2025-5cf59afeb6ba
- IBM Inference-Friendly LoRA: https://research.ibm.com/blog/inference-friendly-aloras-lora

### Models
- mlx-community/Qwen2.5-3B-Instruct-4bit: https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit
- mlx-community/Qwen2.5-7B-Instruct-4bit: https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit
- HumanLLMs/Human-Like-Qwen2.5-7B-Instruct: https://huggingface.co/HumanLLMs/Human-Like-Qwen2.5-7B-Instruct

### Local Codebase
- KARL trainer: ~/Desktop/karl/karl/trainer.py
- KARL config: ~/Desktop/karl/karl/config.py
- SFT exporter: ~/Desktop/karl/karl/sft_exporter.py
- Geometry bridge: ~/Desktop/anticipation-geometry/transformer/geometry_bridge.py
- Conversation trajectory: ~/Desktop/anticipation-geometry/python/anticipation_geometry/conversation_trajectory.py
- KG client: ~/Desktop/live-knowledge-graphs/python/kg_client.py
- KG reward: ~/Desktop/live-knowledge-graphs/python/kg_reward.py
