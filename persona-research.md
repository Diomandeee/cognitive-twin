# LoRA Persona Override: Research and Recommendations

> Target: Make Qwen2.5-7B-Instruct-4bit fully adopt Mohamed's communication style.
> Hardware: Mac5 (M4 16GB), MLX mlx_lm LoRA trainer.
> Data: 3,126 training examples in ChatML format.
> Date: 2026-03-22

---

## Diagnosis: Why the Current Config Fails

The current configuration has **four compounding problems** that prevent persona override:

| Parameter | Current | Problem |
|-----------|---------|---------|
| LoRA rank | 16 | Too low for style transfer. Only captures task patterns, not voice. |
| Num layers | 8 / 28 | 71% of the model is frozen. MLP layers (where style lives) are mostly untouched. |
| Learning rate | 1e-5 | Too conservative. The adapter barely nudges the base model's distribution. |
| System prompt | ~456 chars avg, up to 2055 chars | Massive context window consumed by tool call history. Dilutes the persona signal with noise. |

The base Qwen2.5-7B-Instruct has deep instruct conditioning baked through all 28 transformer layers. With rank 16 on only 8 layers, you are trying to override a 7-billion-parameter personality with ~4M trainable parameters touching 28% of the network. The instruct persona dominates by default.

---

## Finding 1: Rank Must Be 64+ for Style Transfer

**Evidence:**

- Sebastian Raschka's experiments (hundreds of LoRA runs): rank 256 "significantly improved performance" over rank 8. With r=256 and alpha=512, results matched full fine-tuning. Ranks above 256 (512, 1024) failed to converge.
  Source: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

- Lightning AI's experiments: optimal at r=256, alpha=512. Higher ranks diverged.
  Source: https://lightning.ai/pages/community/lora-insights/

- P-Tailor (personality LoRA paper, EMNLP): used effective rank 256 distributed across 16 experts. Found "continuous improvement with higher rank" for personality control.
  Source: https://arxiv.org/html/2406.12548v1

- Unsloth guide: recommends r=16-32 as baseline, up to r=128 for complex tasks. Alpha = 2x rank.
  Source: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

**Why rank 16 fails for persona:** LoRA rank determines the expressiveness of the weight update. Task adaptation (Q&A format, tool use) requires low rank because the semantic change is narrow. Persona/style transfer requires modifying HOW the model generates across ALL topics, which is a much higher-dimensional change. Rank 16 can learn "respond in JSON" but cannot learn "respond like Mohamed."

**Recommendation:** Use rank 64 with alpha 128 (2x ratio). This gives ~20M trainable parameters on 28 layers -- enough to meaningfully shift the style distribution without overfitting on 3K examples.

Do NOT go to rank 256. With only 3,126 examples and 16GB RAM, rank 64 is the sweet spot. Rank 256 would risk overfitting and may exceed memory during backprop on all 28 layers.

---

## Finding 2: ALL 28 Layers Must Be Adapted

**Evidence:**

- QLoRA paper's core finding: "The most important thing you can do to make LoRA fine-tuning effective is to train all layers of the network. Then, and only then, were they able to achieve the quality of full-parameter fine-tuning."
  Source: https://huggingface.co/docs/peft/en/developer_guides/lora

- LoRA Without Regret: Restricting to attention-only caused "5-15% underperformance on downstream metrics" even at high ranks. Must include Q, K, V, O projections AND MLP (up_proj, down_proj, gate_proj).
  Source: https://www.ikangai.com/lora-without-regret-a-practitioners-guide-to-reliable-fine-tuning/

- Raschka: enabling all layers (attention + MLP + projections) improved performance "noticeably" vs attention-only.
  Source: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

- Layer function research: early layers encode syntax/structure, middle layers encode semantics, late layers encode style/generation patterns. To override persona, you need ALL depths.
  Source: multiple transformer interpretability papers (ICLR 2024, NAACL 2024)

**Why 8 layers fails for persona:** With 8/28 layers, only the last 8 layers get adapters. The model's voice is formed across ALL layers. The first 20 frozen layers still push the output toward "helpful AI assistant." It is like trying to change someone's accent by only modifying how they pronounce the last few words of each sentence.

**Recommendation:** Set `--num-layers -1` (all layers) in mlx_lm. This is the single most impactful change.

**Memory concern:** On Mac5 (M4 16GB), Qwen2.5-7B-4bit takes ~4GB for the model. With rank 64 on all 28 layers, adapter params are ~20M (80MB). Gradient state is the bottleneck, but `--grad-checkpoint` trades compute for memory. With batch size 1 and grad accumulation of 4, this should fit in 16GB.

---

## Finding 3: Learning Rate Should Be 2e-4, Not 1e-5

**Evidence:**

- Unsloth: standard LoRA LR is 2e-4. Only use 5e-6 for DPO/RL.
  Source: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

- LoRA Without Regret: "Use approximately 10x full fine-tuning rate (1e-4 to 5e-4 range). This factor remained remarkably consistent across configurations."
  Source: https://www.ikangai.com/lora-without-regret-a-practitioners-guide-to-reliable-fine-tuning/

- HumanLLMs/Human-Like-Qwen2.5-7B-Instruct (the directly relevant project): used LR 0.0002 (2e-4) with cosine scheduler.
  Source: https://huggingface.co/HumanLLMs/Human-Like-Qwen2.5-7B-Instruct

- Qwen3 SFT+DPO guide: SFT stage uses LR 1e-4.
  Source: https://blog.ivan.digital/finetuning-qwen3-with-lora-done-right-94d6343e1814

**Why 1e-5 fails:** At 1e-5, the gradient updates are 20x smaller than recommended. The adapter weights barely move from initialization. After 1000 iterations, the model has learned almost nothing -- the base instruct personality overwhelms the tiny perturbations.

**Recommendation:** Use 2e-4 with cosine schedule. This is aggressive enough to override the instruct personality but standard for LoRA SFT.

---

## Finding 4: System Prompt Must Be Shortened Dramatically

**Current problem:** The system prompt averages 456 characters and maxes out at 2055 characters. Nearly half the examples (1,490/3,126) have system prompts over 500 chars. These system prompts are packed with tool call history that is irrelevant noise for persona learning.

The model sees:
```
<|im_start|>system
You are Mohamed's cognitive twin. Respond as Mohamed would based on the context of what just happened.

Recent tool calls:
- (earlier: Editx32, Readx7, TaskCreatex3, Writex3, Taskx1)
- TaskUpdate:  -> Updated task #1 status
- TaskUpdate:  -> Updated task #2 status
[... 400 more chars of tool call history ...]
<|im_end|>
```

The model is spending its limited capacity learning the correlation between tool call patterns and Mohamed's responses, instead of learning Mohamed's voice. Worse, the tool call context varies wildly between examples, making the persona signal noisy.

**Research support:**
- Neurosymbolic LoRA paper (2025): system prompt changes interact with LoRA training. Joint system+question prompt updates gave 3.5-4.5% accuracy boosts. Shorter, more targeted prompts produce better alignment.
  Source: https://arxiv.org/html/2601.12711

- HumanLLMs used minimal system prompts with ChatML format, not elaborate tool call histories.
  Source: https://huggingface.co/HumanLLMs/Human-Like-Qwen2.5-7B-Instruct

**Recommendation:** Strip system prompts to a MAXIMUM of one short line:

```
You are Mohamed. Be direct, casual, action-oriented. Short responses.
```

Or even better: **remove the system prompt entirely** and let the LoRA learn the persona purely from the (user, assistant) pairs. The system prompt at inference time can add context, but during training it should not dilute the signal.

Alternative: keep a minimal system prompt but put ALL the persona description there. Remove the tool call history completely.

---

## Finding 5: Training Data Format Needs Optimization

The current data has the right voice (verified by sampling):
- "Yeah keep going. Don't stop until completion."
- "commit it and push"
- "Give me the full picture. What's done and what's left."
- "Yes. 3 retries with exponential backoff. After that, alert and move on."

But there are format issues:

1. **518 responses under 20 characters** (16.5%): responses like "continuw" (sic), "Continue.", "launch it" are so short they may not carry enough gradient signal per example.

2. **Typos preserved**: "continuw" -- decide if this is intentional (authentic voice) or noise.

3. **Some responses are meta** rather than persona: "[Request interrupted by user for tool use]" should be removed.

**Recommendation:**
- Remove examples where assistant response is less than 10 characters (too little signal)
- Remove meta-responses (interrupted, empty, etc.)
- Keep typos if intentional (Mohamed's authentic texting style)
- Target 2,500-3,000 clean examples after filtering

---

## Finding 6: Two-Stage SFT + DPO Is the Best Approach

**Evidence:**

HumanLLMs/Human-Like-Qwen2.5-7B-Instruct used exactly this pattern:
1. Stage 1: LoRA SFT to learn the basic style patterns
2. Stage 2: DPO with preferred (Mohamed-style) vs rejected (generic AI-style) pairs

They achieved persona transfer with minimal benchmark degradation (-0.2% average).
Source: https://huggingface.co/HumanLLMs/Human-Like-Qwen2.5-7B-Instruct

Qwen3 fine-tuning guide recommends KL-anchored SFT (KL=0.05) then DPO with beta sweep.
Source: https://blog.ivan.digital/finetuning-qwen3-with-lora-done-right-94d6343e1814

**Why this works:** SFT teaches the model Mohamed's response patterns. DPO teaches the model to PREFER Mohamed's style over the AI assistant style. The DPO stage is what kills the "Great job!" / "Let's break down..." tendencies.

**Recommendation:**
- Phase 1: LoRA SFT with the current 3,126 examples (cleaned)
- Phase 2: Generate DPO pairs by running the fused SFT model + the base model on the same prompts. Mohamed's real responses are "chosen," base model's AI-slop responses are "rejected."
- MLX supports DPO via `--train-mode dpo` natively

---

## Finding 7: DoRA > LoRA for Style Transfer

**Evidence:**

- DoRA (ICML 2024 Oral, NVlabs): decomposes weights into magnitude + direction. Trains magnitude vector separately from directional LoRA. Achieves better results than LoRA at LOWER rank, with zero inference overhead after merging.
  Source: https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/

- "DoRA is more robust than LoRA, and even with a lower number of ranks it shows higher performance."
  Source: https://github.com/NVlabs/DoRA

- MLX supports DoRA natively: `--fine-tune-type dora`
  Source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md

**Why DoRA is better for persona:** Standard LoRA only learns an additive direction update. DoRA can also scale weight importance (magnitude), which matters for style: some attention heads need to be scaled UP (those handling tone) while others need directional shifts (those handling content). DoRA provides both levers.

**Recommendation:** Use `--fine-tune-type dora` instead of default lora. Zero downside, strictly better.

---

## Finding 8: 7B Is Fine with 3K Examples (No Need to Downsize)

**Evidence:**

- Small models (1B-3B) are MORE tunable per example, but 7B with 3K examples is well within normal range. LoRA Without Regret: r=32 matched full fine-tuning up to 50K examples on 7B. We have 3K, so rank 64 is safe from overfitting.
  Source: https://www.ikangai.com/lora-without-regret-a-practitioners-guide-to-reliable-fine-tuning/

- HumanLLMs used 10,884 samples on Qwen2.5-7B and got good results.
  Source: https://huggingface.co/HumanLLMs/Human-Like-Qwen2.5-7B-Instruct

- Smaller models (1B, 3B) would lose the base capabilities we need (code understanding, tool use, instruction following). The goal is to change HOW the model speaks, not to reduce WHAT it can do.

**Recommendation:** Stay on Qwen2.5-7B-Instruct-4bit. Consider switching to Qwen2.5-7B-Instruct (non-quantized) if memory allows -- quantization introduces noise that can interfere with fine-tuning gradients. But 4-bit is acceptable if memory is the constraint.

---

## Finding 9: Use `--mask-prompt` to Focus Learning

MLX's `--mask-prompt` flag (also called loss masking) masks the loss on the system and user turns, so the model only learns to predict the ASSISTANT tokens. Without this, the model wastes gradient signal learning to predict tool call histories and user messages.

This is critical for persona: we want 100% of the gradient signal focused on predicting Mohamed's responses.

**Recommendation:** Add `--mask-prompt` to the training command.

---

## The Exact Training Command

### Phase 1: Aggressive SFT (DoRA, all layers, rank 64)

```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data /path/to/cleaned-data/ \
  --fine-tune-type dora \
  --num-layers -1 \
  --rank 64 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --iters 2000 \
  --grad-checkpoint \
  --mask-prompt \
  --adapter-path /path/to/adapters-persona-v1 \
  --val-batches 25 \
  --steps-per-eval 100 \
  --steps-per-report 10
```

**Hyperparameter justification:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--fine-tune-type dora` | DoRA | Magnitude + direction decomposition. Strictly better than LoRA for style tasks. Zero inference overhead. |
| `--num-layers -1` | All 28 layers | QLoRA paper: "all layers required to match full fine-tuning." Style lives across all depths. |
| `--rank 64` | 64 | Raschka: r=256 optimal for diverse tasks. We use 64 (not 256) because: (a) 3K examples is moderate, (b) 16GB memory constraint, (c) persona is high-dimensional but not as diverse as multi-task. |
| `--learning-rate 2e-4` | 0.0002 | Standard LoRA LR per Unsloth, LoRA Without Regret, and HumanLLMs. 20x higher than the failed 1e-5. |
| `--batch-size 1` | 1 | Memory constraint: M4 16GB. Effective batch = 4 via grad accumulation. |
| `--grad-accumulation-steps 4` | 4 | Effective batch size of 4. Larger batches (32-128) are ideal but memory-constrained. |
| `--iters 2000` | 2000 | ~2.5 epochs over 3126 examples (3126/1 batch / 1 = 3126 steps per epoch, but with shuffling, 2000 iters covers most data). Lightning AI found 2 epochs sometimes worse than 1 -- monitor validation loss. |
| `--grad-checkpoint` | On | Trades compute for memory. Essential at rank 64 on all layers with 16GB. |
| `--mask-prompt` | On | Only learn on assistant tokens. Focuses 100% of gradient on Mohamed's voice. |

### Phase 2: DPO (Optional but Recommended)

After fusing the SFT adapter, generate DPO data:

1. Run the fused SFT model on 500-1000 of the same prompts
2. Also run the BASE model (no adapter) on the same prompts
3. Format as DPO pairs: `{"chosen": Mohamed_real_response, "rejected": base_model_response}`
4. Train DPO:

```bash
python3 -m mlx_lm lora \
  --model /path/to/fused-sft-model/ \
  --data /path/to/dpo-pairs/ \
  --train-mode dpo \
  --fine-tune-type dora \
  --num-layers -1 \
  --rank 32 \
  --learning-rate 5e-6 \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --iters 500 \
  --grad-checkpoint \
  --beta 0.1 \
  --adapter-path /path/to/adapters-persona-dpo-v1
```

Note: DPO uses much lower LR (5e-6) and fewer iterations. Beta=0.1 is the default DPO temperature.

---

## Data Preprocessing Script

Before training, clean the data:

```python
#!/usr/bin/env python3
"""Clean training data for persona LoRA."""
import json

INPUT = "autocontinue-v3/train.jsonl"
OUTPUT = "persona-clean/train.jsonl"
EVAL_INPUT = "autocontinue-v3/valid.jsonl"
EVAL_OUTPUT = "persona-clean/valid.jsonl"

SYSTEM_PROMPT = "You are Mohamed. Direct, casual, action-oriented. Short responses. Never say 'Great job', 'Let me help', or 'I understand'. No filler."

# Responses to filter out
SKIP_PATTERNS = [
    "[Request interrupted",
    "[No response",
    "No response requested",
]

def clean_file(inp, out):
    kept = 0
    skipped = 0
    with open(inp) as f, open(out, "w") as fout:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages", [])

            # Find assistant response
            asst = None
            user = None
            for m in msgs:
                if m["role"] == "assistant":
                    asst = m["content"]
                if m["role"] == "user":
                    user = m["content"]

            if not asst or len(asst.strip()) < 10:
                skipped += 1
                continue

            if any(pat in asst for pat in SKIP_PATTERNS):
                skipped += 1
                continue

            # Rebuild with minimal system prompt
            clean = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": asst},
                ]
            }
            fout.write(json.dumps(clean) + "\n")
            kept += 1

    print(f"{inp}: kept {kept}, skipped {skipped}")

clean_file(INPUT, OUTPUT)
clean_file(EVAL_INPUT, EVAL_OUTPUT)
```

---

## Memory Budget Estimate (Mac5, M4 16GB)

| Component | Memory |
|-----------|--------|
| Qwen2.5-7B-4bit model weights | ~4.0 GB |
| DoRA adapters (rank 64, all 28 layers, both attention + MLP) | ~0.1 GB |
| Gradient state (AdamW, 2 momentum buffers per adapter param) | ~0.3 GB |
| Activations (batch=1, seq_len=2048, with grad checkpoint) | ~3-5 GB |
| MLX framework overhead | ~1 GB |
| **Total estimated** | **~8.4-10.4 GB** |
| **Headroom** | **~5.6-7.6 GB** |

This should fit. If it OOMs, reduce rank to 48 or reduce max sequence length.

---

## Validation Strategy

After training, test with these prompts (the model should NOT produce generic AI responses):

```
User: "The deploy failed with a 503 error on the staging server."
GOOD: "Show me the logs. What changed since last deploy?"
BAD: "I understand that must be frustrating! Let me help you troubleshoot..."

User: "Should we refactor the auth module or just patch the bug?"
GOOD: "Patch it. Ship the fix, then refactor next sprint."
BAD: "Great question! Let's break down the pros and cons..."

User: "The PR is ready for review."
GOOD: "Merge it."
BAD: "Awesome work! I'd be happy to review it for you!"
```

---

## Contrarian Check: What Could Go Wrong?

1. **Overfitting risk with rank 64 on 3K examples:** Mitigated by DoRA (more parameter-efficient than LoRA at same rank), monitoring validation loss, and stopping early if val loss diverges.

2. **Losing base capabilities:** The instruct model's code/reasoning ability could degrade. Mitigated by: (a) LoRA only modifies a small fraction of total params, (b) `--mask-prompt` focuses learning on output style not input understanding, (c) DPO stage specifically penalizes losing useful content.

3. **DoRA may not be supported well in mlx_lm:** It is listed as a first-class option (`--fine-tune-type dora`). If it causes issues, fall back to standard LoRA with same params.

4. **System prompt stripping loses context:** At inference time, we still USE the full system prompt with tool call context. We only strip it during TRAINING to focus the gradient signal. The model learns voice from training, uses context from inference prompts.

---

## Summary: Changes from Failed Config

| Parameter | Failed Config | New Config | Impact |
|-----------|--------------|------------|--------|
| Fine-tune type | lora | **dora** | Better style capture via magnitude decomposition |
| Rank | 16 | **64** | 4x more expressive capacity for voice patterns |
| Num layers | 8 | **-1 (all 28)** | Full network adaptation instead of 28% |
| Learning rate | 1e-5 | **2e-4** | 20x stronger gradient signal |
| Batch size | 2 | **1 (+ 4 grad accum)** | Same effective batch, less memory |
| Iterations | 1000 | **2000** | More exposure to full dataset |
| System prompt | ~456 chars avg | **~80 chars fixed** | Removes noise, focuses on persona |
| Mask prompt | off (assumed) | **on** | Only learn assistant outputs |
| Grad checkpoint | on | **on** | Same -- keeps memory manageable |

---

## Sources

### Papers and Research
- DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024 Oral): https://arxiv.org/pdf/2402.09353
- P-Tailor: Personality LoRA Experts (EMNLP 2024): https://arxiv.org/html/2406.12548v1
- Fusian: Multi-LoRA Personality Control (2026): https://arxiv.org/html/2603.15405
- Neeko: Multi-Character Role-Playing Agent (EMNLP 2024): https://arxiv.org/html/2402.13717v1
- Neurosymbolic LoRA (2025): https://arxiv.org/html/2601.12711
- OpenCharacter: Synthetic Persona Training (2025): https://arxiv.org/html/2501.15427v1

### Practitioner Guides
- Raschka, Practical Tips for Finetuning LLMs Using LoRA: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- Lightning AI, LoRA Insights from Hundreds of Experiments: https://lightning.ai/pages/community/lora-insights/
- LoRA Without Regret: https://www.ikangai.com/lora-without-regret-a-practitioners-guide-to-reliable-fine-tuning/
- Unsloth LoRA Hyperparameters Guide: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- DoRA from Scratch (Raschka): https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch

### Directly Relevant Projects
- HumanLLMs/Human-Like-Qwen2.5-7B-Instruct: https://huggingface.co/HumanLLMs/Human-Like-Qwen2.5-7B-Instruct
- NVlabs DoRA Implementation: https://github.com/NVlabs/DoRA
- MLX-LM LoRA Documentation: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- UW LoRA Target Module Selection (2026): https://nwquantum.uw.edu/2026/03/19/optimizing-lora-target-module-selection-for-efficient-fine-tuning/

### Qwen-Specific
- Qwen3 SFT+DPO Pipeline Guide: https://blog.ivan.digital/finetuning-qwen3-with-lora-done-right-94d6343e1814
- NVIDIA NeMo Qwen2/2.5 Guide: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/qwen2.html
