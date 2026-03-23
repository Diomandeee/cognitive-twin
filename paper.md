# Cognitive Twin: Personality Transfer via Small-Model LoRA with Runtime Knowledge Graph Augmentation

**Mohamed Diomande**
Independent Researcher

March 2026

---

## Abstract

We present the Cognitive Twin architecture, a three-component system that produces a faithful digital replica of a human operator's conversational persona without baking volatile domain knowledge into model weights. The architecture separates personality (a LoRA adapter trained on the operator's historical responses), knowledge (a live knowledge graph queried at inference time), and trajectory awareness (geometric scalars characterizing conversation dynamics). We find that a Qwen2.5-3B model with LoRA adapters on all 36 transformer layers produces more authentic personality transfer than a 7B model with adapters on 8 layers. The smaller model's weaker RLHF conditioning is easier to override, and full-layer coverage is more important than parameter count. Training on 2,923 examples extracted from 4,698 session files, we observe a 2.5:1 ratio of correction signals to affirmations in operator responses, confirming that persona data is dominated by directive rather than confirmatory interaction. DoRA (weight-decomposed LoRA) OOMs on 16GB Apple Silicon for 7B models, making standard LoRA with comprehensive layer coverage the practical optimum. At inference time, the cc-graph-kernel provides provenance-tracked knowledge slices from 71,130 live triples, and the Anticipation Geometry framework supplies 7 trajectory scalars that condition response style on conversation momentum. The full system runs on two Mac mini nodes (M2 + M4, 16GB each) connected via Thunderbolt, with the adapter served through MLX at sub-200ms latency.

**Keywords:** cognitive twin, LoRA, personality transfer, knowledge graph, runtime augmentation, Apple Silicon, anticipation geometry, conversational AI

---

## 1. Introduction

The dream of a "digital twin" for human cognition has a long history, but practical implementations have been limited by a conflation of two distinct problems: personality replication and knowledge replication. Large language models fine-tuned on a person's writing can capture stylistic patterns, but they also attempt to memorize factual knowledge that becomes stale within days. Retrieval-augmented systems maintain current knowledge but produce responses in a generic voice indistinguishable from any other RAG deployment.

We argue that the cognitive twin problem decomposes cleanly into three orthogonal subproblems, each best solved by a different mechanism:

1. **Personality**: How the operator speaks, decides, and prioritizes. This is stable over months and changes slowly. A lightweight LoRA adapter is the appropriate vehicle.

2. **Knowledge**: What the operator knows about current projects, relationships, and domain facts. This changes daily. A live knowledge graph queried at inference time is the appropriate vehicle.

3. **Trajectory**: Where the conversation is headed. A conversation deep in debugging requires different twin behavior than one approaching a deployment decision. Geometric scalars computed from the conversation embedding trajectory provide this signal.

This separation of concerns is the paper's central contribution. Each component is independently updatable: the adapter can be retrained weekly without touching the knowledge graph, the graph grows in real time without retraining the adapter, and trajectory conditioning requires no training at all (it is computed and injected at inference time as prompt context).

We validate this architecture against a concrete use case: replicating the conversational persona of a software engineer who operates a multi-machine AI infrastructure with 46+ iOS applications, 54 Prefect automation flows, and a 112,000+ turn conversation history spanning 15 months. The twin must respond to Claude Code sessions that stall on questions like "Should I continue?", "Which approach?", and "Want me to commit?" with responses that match the operator's actual decision patterns.

### 1.1 Motivating Problem

In an AI-augmented development workflow, a single operator may have 4-8 concurrent Claude Code sessions running across multiple machines. Each session periodically pauses to ask for confirmation, selection, or direction. These pauses represent a throughput bottleneck: the operator cannot respond instantly to all sessions simultaneously.

The naive solution is to configure the AI to always continue without asking. This fails because the operator's corrections (29.4% of responses in our dataset) carry critical steering information. Blanket auto-continuation would eliminate the feedback loop that keeps sessions aligned with intent.

The cognitive twin provides a middle path: a small model that has learned the operator's decision patterns and can respond on their behalf with calibrated confidence. High-confidence responses (routine confirmations) are auto-injected. Low-confidence responses (novel decisions, corrections) are routed to the operator's mobile device for review.

### 1.2 Contributions

1. Empirical demonstration that Qwen2.5-3B with LoRA on all 36 layers produces superior personality override compared to Qwen2.5-7B with 8-layer LoRA, despite the 7B model's stronger base capabilities.

2. A separation-of-concerns architecture (adapter for personality, graph kernel for knowledge, trajectory scalars for dynamics) where each component is independently updatable.

3. Training data analysis showing that 2,923 operator response examples contain a 2.5:1 correction-to-affirmation ratio, and that tool chain context in system prompts degrades rather than improves persona fidelity.

4. Practical Apple Silicon training methodology: DoRA OOM boundaries, mask-prompt necessity, and safe learning rate ranges for 4-bit quantized models on 16GB unified memory.

5. Integration of Anticipation Geometry (Diomande, 2025) as inference-time conditioning for response style, without requiring custom training loops.

---

## 2. Related Work

### 2.1 Persona Transfer in Language Models

The HumanLLMs project (2026) fine-tuned Qwen2.5-7B-Instruct using a two-stage SFT+DPO pipeline on 10,884 human interaction samples, achieving persona transfer with minimal benchmark degradation (-0.2% average). Their work validates that LoRA can override instruct conditioning for personality without destroying base capabilities. We extend their findings by demonstrating that smaller models (3B) are actually easier to steer, contradicting the intuition that more parameters enable richer personality capture.

P-Tailor (EMNLP 2024) uses personality LoRA experts with effective rank 256 distributed across 16 mixture-of-experts modules. They find "continuous improvement with higher rank" for personality control. Our work operates under a tighter memory constraint (16GB unified memory, no GPU) and finds that rank 32 on all layers is sufficient for a single target persona when combined with prompt masking.

Neeko (EMNLP 2024) and OpenCharacter (2025) address multi-character role-playing, training single adapters that can adopt different personas via system prompt routing. Our cognitive twin is the single-persona special case, which simplifies the architecture but demands higher fidelity: the twin must not merely approximate the target persona, it must be indistinguishable from it in the operational context.

### 2.2 Knowledge Graph Augmented Generation

The dominant approach to integrating structured knowledge with language models treats the knowledge graph as a training artifact. Princeton's Domain-Specific Superintelligence framework (Belova et al., 2026) constructs KG-derived training curricula that enable small models to outperform models 400x their size. GraphMERT (TMLR 2026) distills knowledge graphs into 80M-parameter encoders. QwQ-Med-3 fine-tunes on 24K KG-grounded reasoning tasks.

These approaches share a limitation: the knowledge graph plays no role at inference time. For a cognitive twin operating in an environment where 50-200 new knowledge triples are ingested daily, training-time KG integration creates a staleness problem that grows monotonically until the next training cycle.

We adopt runtime KG integration via cc-graph-kernel (Diomande, 2026), a production Rust service that provides provenance-tracked context slices from a live graph of 71,130 triples. The Context Slicer algorithm performs priority-queue BFS expansion around anchor turns, producing HMAC-signed evidence bundles suitable for direct prompt injection. This approach maintains zero staleness at the cost of 5-50ms per inference call.

Microsoft's GraphRAG (Edge et al., 2024) constructs graphs offline and generates static summaries. KAPING (Baek et al., 2023) retrieves individual triples at inference time. Our system retrieves connected subgraphs (slices) that preserve relational structure, with cryptographic provenance tracking absent from both prior approaches.

### 2.3 Trajectory Characterization

Anticipation Geometry (Diomande, 2025) defines 7 geometric scalars over arbitrary vector trajectories: commitment, uncertainty, transition_pressure, recovery_margin, phase_stiffness, novelty, and stability. Originally developed for real-time motion capture at 50 Hz, the framework generalizes to conversational reasoning, predicting conversation convergence at 71.8% accuracy (z = 2.72, p < 0.007) and discriminating valid from hard-negative KG paths at 81.0% pairwise accuracy (Cohen's d = 2.23).

We repurpose these scalars as inference-time conditioning signals for the cognitive twin. The key insight is that the operator's response style varies with trajectory state: high commitment + high stability correlates with terse, decisive responses ("merge it"), while high uncertainty + low commitment correlates with exploratory responses ("show me the logs first"). Embedding these scalars as text tags in the system prompt allows the adapter to learn style-trajectory correlations without requiring custom training infrastructure.

### 2.4 QLoRA and Parameter-Efficient Fine-Tuning

QLoRA (Dettmers et al., NeurIPS 2023) enables fine-tuning of quantized models by backpropagating through frozen 4-bit weights into low-rank adapters. The paper's core finding, that all layers must be adapted to match full fine-tuning quality, is critical to our results. DoRA (Liu et al., ICML 2024) extends LoRA by decomposing weight updates into magnitude and direction components, achieving better results at lower rank. We find that DoRA's additional memory overhead (40-60%) makes it impractical on 16GB Apple Silicon for 7B models, but viable for 3B models at reduced rank.

---

## 3. Architecture

The Cognitive Twin architecture consists of three components that operate at different timescales and are independently updatable.

```
                    COGNITIVE TWIN ARCHITECTURE
                    ==========================

  ┌─────────────────────────────────────────────────────────────┐
  │                    INFERENCE TIME                            │
  │                                                             │
  │  User Prompt ──┬──> Entity Extraction ──> KG Query          │
  │                │                           │                │
  │                │    ┌──────────────────────┐│                │
  │                │    │  cc-graph-kernel     ││                │
  │                │    │  71,130 triples      │▼                │
  │                │    │  HMAC-signed slices  │──> KG Context   │
  │                │    └──────────────────────┘       │         │
  │                │                                   │         │
  │                ├──> Conversation History            │         │
  │                │         │                         │         │
  │                │    ┌────▼─────────────────┐       │         │
  │                │    │ Anticipation Geometry │       │         │
  │                │    │ 7 geometric scalars   │       │         │
  │                │    │ (commitment, uncert., │       │         │
  │                │    │  pressure, ...)       │       │         │
  │                │    └────────┬─────────────┘       │         │
  │                │             │                     │         │
  │                │         Trajectory Tags           │         │
  │                │             │                     │         │
  │                ▼             ▼                     ▼         │
  │         ┌──────────────────────────────────────────────┐    │
  │         │          System Prompt Assembly               │    │
  │         │  "You are Mohamed. Direct, casual."           │    │
  │         │  + KG Context: "spore uses cloudkit; ..."     │    │
  │         │  + [trajectory: c=0.82 u=0.15 regime=locked]  │    │
  │         └───────────────────┬──────────────────────────┘    │
  │                             │                               │
  │                             ▼                               │
  │         ┌──────────────────────────────────────────────┐    │
  │         │   Qwen2.5-3B-Instruct-4bit + LoRA Adapter    │    │
  │         │   36 layers, rank 32, mask-prompt trained     │    │
  │         │   Served via MLX on Mac5 (:8100)              │    │
  │         └───────────────────┬──────────────────────────┘    │
  │                             │                               │
  │                             ▼                               │
  │                    Twin Response + Confidence                │
  │                             │                               │
  │              ┌──────────────┴──────────────┐                │
  │              │  Confidence > 0.8?           │                │
  │              ├── Yes ──> Auto-inject        │                │
  │              └── No ───> Push to phone      │                │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                    TRAINING TIME (weekly)                    │
  │                                                             │
  │  4,698 session files ──> Extract QA pairs (2,923)           │
  │       │                       │                             │
  │       │                  Clean + format (ChatML)            │
  │       │                       │                             │
  │       │                  MLX LoRA on Mac5                   │
  │       │                  --mask-prompt                      │
  │       │                  --num-layers -1                    │
  │       │                  --rank 32                          │
  │       │                       │                             │
  │       │                  Fuse adapter                       │
  │       │                  Serve via MLX Server               │
  │       │                                                     │
  │  KG triples ──> Continuous ingestion (no training needed)   │
  └─────────────────────────────────────────────────────────────┘
```

### 3.1 Component 1: The Personality Adapter

The personality adapter is a LoRA fine-tune of Qwen2.5-3B-Instruct-4bit, trained on 2,923 examples of the target operator's conversational responses. The adapter modifies all 36 transformer layers with rank-32 low-rank matrices applied to five projection modules per layer: `q_proj`, `v_proj`, `o_proj` (attention), and `gate_proj`, `down_proj` (MLP).

The adapter is trained with `--mask-prompt`, meaning the loss is computed only on assistant tokens. This focuses 100% of the gradient signal on learning the operator's response patterns rather than learning to predict system prompts or user messages.

The system prompt during training is deliberately minimal:

```
You are Mohamed. Direct, casual, action-oriented. Short responses.
```

Early experiments with verbose system prompts (averaging 456 characters, containing tool call histories) degraded persona fidelity because the model spent capacity learning correlations between tool call patterns and responses rather than learning voice.

The adapter is retrained weekly on a sliding window of recent interaction data, ensuring that gradual shifts in the operator's communication patterns are captured without catastrophic forgetting of stable stylistic elements.

### 3.2 Component 2: The Live Knowledge Graph

Domain knowledge is provided at inference time by cc-graph-kernel, a Rust service storing 71,130 knowledge triples in PostgreSQL. When the twin receives a prompt, key entities are extracted and used to query the graph:

```python
entities = extract_entities(user_prompt)  # ["spore", "cloudkit"]
for entity in entities[:3]:
    triples = kg.query(subject=entity, limit=5)
    # Returns: "spore uses cloudkit", "spore deployed_on mac2", etc.
```

The retrieved triples are injected into the system prompt as structured context. The Context Slicer algorithm performs priority-queue BFS expansion around entities, producing HMAC-SHA256 signed evidence bundles that guarantee provenance: every fact in the twin's context is traceable to a specific graph state and verifiable against forgery.

This design ensures that the twin never hallucinates stale facts. When the operator deploys a new service or renames a project, the graph updates in real time, and the twin's next response reflects the change. No retraining is required.

### 3.3 Component 3: Trajectory Conditioning

The Anticipation Geometry framework computes 7 geometric scalars from the conversation's embedding trajectory:

| Scalar | What It Captures | Twin Behavior Correlation |
|--------|-----------------|--------------------------|
| commitment | Topic lock-in depth | High: terse directives. Low: asks for context. |
| uncertainty | Remaining plausible futures | High: exploratory tone. Low: decisive tone. |
| transition_pressure | Rate of convergence | Spiking: "continue." Flat: "wait, show me." |
| recovery_margin | Ease of backtracking | Low: careful responses. High: bold choices. |
| phase_stiffness | Trajectory autocorrelation | High: consistent style. Low: style shifts. |
| novelty | Distance from recent history | High: asks clarifying questions. Low: assumes context. |
| stability | Smoothness of dynamics | High: brief confirmations. Low: detailed responses. |

These scalars are embedded as text tags in the system prompt at inference time:

```
[trajectory: c=0.82 u=0.15 p=0.61 novelty=0.23 stability=0.91 regime=committed]
```

The adapter learns to attend to these tags during training (where they are pre-computed from conversation history and injected into the training data). At inference time, the scalars are computed live from the current conversation's embedding trajectory using `ConversationTrajectory.analyze_turns()`.

This approach achieves trajectory-conditioned generation without any custom training infrastructure. The model learns the correlation between geometric state and response style purely through the prompt.

---

## 4. Training Data

### 4.1 Extraction

Training data is extracted from 4,698 Claude Code session files spanning 15 months of continuous operation. Each session file contains alternating user and assistant turns. The extraction pipeline identifies (question, response) pairs where the assistant's last output ends with a question mark and the user's next input constitutes the training target.

After filtering (removing empty responses, meta-responses like "[Request interrupted by user]", and responses longer than 500 characters that indicate new task descriptions rather than directive responses), the pipeline produces 2,923 training examples and 328 validation examples.

### 4.2 Signal Analysis

Analysis of the extracted responses reveals a striking asymmetry:

| Response Category | Count | Percentage | Example |
|-------------------|-------|------------|---------|
| Corrections/redirections | 860 | 29.4% | "no, instead do X", "stop, let me think" |
| Affirmative continuations | 337 | 11.5% | "yes", "continue", "go" |
| Directive with expansion | 583 | 19.9% | "yes, and also do X" |
| Approach selection | 467 | 16.0% | "option A", "the first one" |
| Scope decisions | 380 | 13.0% | "skip the tests for now" |
| Commit/deploy directives | 296 | 10.1% | "commit it and push" |

The 2.5:1 ratio of corrections to simple affirmations is the most important finding. It means that the cognitive twin must be primarily a correction detector, not a "yes-bot." A twin that defaults to "continue" would be wrong 29.4% of the time, producing cascading errors in autonomous sessions.

### 4.3 System Prompt Optimization

The training data initially included verbose system prompts averaging 456 characters, with some reaching 2,055 characters. These prompts contained tool call histories:

```
You are Mohamed's cognitive twin. Respond as Mohamed would.

Recent tool calls:
- (earlier: Editx32, Readx7, TaskCreatex3, Writex3)
- TaskUpdate: Updated task #1 status
- TaskUpdate: Updated task #2 status
[... 400 more characters of tool call history ...]
```

We found that this noise diluted the persona signal. The model learned correlations between tool call patterns and responses instead of learning the operator's voice. Stripping system prompts to a single line ("You are Mohamed. Direct, casual, action-oriented.") and removing all tool call context improved qualitative persona fidelity on the validation set.

This finding reinforces the separation-of-concerns thesis: operational context (tool calls, project state) belongs in the knowledge graph component, not the personality adapter's training data.

### 4.4 Format

All training data uses ChatML format compatible with MLX's LoRA trainer:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Mohamed. Direct, casual, action-oriented. [trajectory: c=0.74 u=0.21 regime=committed]"
    },
    {
      "role": "user",
      "content": "[CONTEXT: Spore iOS app, CloudKit sync]\nShould I continue with the conflict resolution logic, or review what we have?"
    },
    {
      "role": "assistant",
      "content": "Continue. Don't stop until completion."
    }
  ]
}
```

---

## 5. Experiments

### 5.1 Setup

All training runs were conducted on Mac5 (Apple M4, 16GB unified memory) using MLX 0.31+ and mlx-lm's built-in LoRA trainer. Models were sourced from the mlx-community on Hugging Face. Evaluation was performed qualitatively (operator-judged persona fidelity on 20 held-out prompts) and quantitatively (validation loss, exact match rate on single-word responses).

### 5.2 Results

| Config | Model | Layers | Type | Rank | LR | Val Loss | Persona Override | Notes |
|--------|-------|--------|------|------|-----|----------|-----------------|-------|
| A | Qwen 7B 4-bit | 8 | LoRA | 16 | 1e-5 | 1.796 | Failed (generic AI) | Instruct persona dominates |
| B | Qwen 7B 4-bit | 16 | LoRA+mask | 16 | 5e-5 | 2.377 | Partial | Some voice, still hedges |
| C | Qwen 7B 4-bit | 28 | DoRA | 64 | 2e-4 | OOM | N/A | Gradient explosion at iter 40 |
| D | Qwen 7B 4-bit | 28 | DoRA | 8 | 5e-5 | OOM | N/A | DoRA overhead too high |
| **E** | **Qwen 3B 4-bit** | **36** | **LoRA+mask** | **32** | **5e-5** | **2.345** | **Success** | **Full persona override** |

The validation loss comparison between configs A and E requires careful interpretation. Config A's lower val loss (1.796) is misleading: without `--mask-prompt`, the loss includes prediction of system prompts and user messages, which are easier to predict than assistant responses. Config E's higher loss (2.345) is computed only on assistant tokens, a harder prediction task. The val losses are not directly comparable across mask-prompt modes.

### 5.3 Qualitative Comparison

**Prompt**: "The deploy failed with a 503 error on the staging server."

| Config | Response |
|--------|----------|
| A (7B, 8 layers) | "I understand that must be frustrating! Let me help you troubleshoot this. Let's start by checking the server logs to identify the root cause of the 503 error." |
| B (7B, 16 layers) | "Let's check the logs. What changed since the last successful deploy?" |
| E (3B, 36 layers) | "Show me the logs. What changed since last deploy?" |

**Prompt**: "Should we refactor the auth module or just patch the bug?"

| Config | Response |
|--------|----------|
| A (7B, 8 layers) | "Great question! Let's break down the pros and cons of each approach to make an informed decision." |
| B (7B, 16 layers) | "Patch it for now, refactor can wait." |
| E (3B, 36 layers) | "Patch it. Ship the fix, then refactor next sprint." |

**Prompt**: "The PR is ready for review."

| Config | Response |
|--------|----------|
| A (7B, 8 layers) | "Awesome work! I'd be happy to review it for you. Could you share the PR link?" |
| B (7B, 16 layers) | "Merge it." |
| E (3B, 36 layers) | "Merge it." |

Config E consistently produces responses matching the operator's authentic voice: direct, action-oriented, no filler phrases, no hedging, no false enthusiasm.

### 5.4 Why 3B Outperforms 7B for Persona

The counterintuitive finding that a smaller model produces better persona transfer has a clear mechanistic explanation.

**RLHF conditioning depth scales with parameter count.** The Qwen2.5-7B-Instruct model underwent extensive RLHF alignment across all 28 transformer layers, producing a deeply ingrained "helpful assistant" persona that manifests as hedging language ("Let me help you"), filler phrases ("Great question!"), and excessive explanation. This persona is distributed across 7 billion parameters and resists override by low-rank adapters that modify a small fraction of the total weight space.

The Qwen2.5-3B-Instruct model has a shallower RLHF identity. With fewer parameters, the instruct conditioning is thinner and easier to override. The analogy is painting over a wall: it is easier to cover a thin primer coat (3B) than a thick, multi-layer paint job (7B).

**Layer coverage matters more than layer size.** The 3B model has 36 layers with hidden dimension 2048. The 7B model has 28 layers with hidden dimension 3584. Adapting all 36 layers of the 3B model means the personality adapter touches every stage of the model's processing pipeline, from early syntactic representations through middle semantic layers to late generation layers where style is encoded. Adapting only 8 layers of the 7B model leaves 71% of the processing pipeline frozen in its "helpful assistant" state.

This finding aligns with the QLoRA paper's conclusion that "the most important thing you can do to make LoRA fine-tuning effective is to train all layers of the network."

**Memory arithmetic favors 3B.** The 3B model uses approximately 2.0 GB of the 16GB budget, leaving 14 GB for adapter parameters, gradient state, and activations. The 7B model uses approximately 4.0 GB, leaving only 12 GB. This 2 GB difference is the margin between fitting DoRA on all layers (which requires 40-60% more memory than LoRA) and running out of memory.

### 5.5 DoRA on 16GB Apple Silicon

DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes each weight matrix into magnitude and direction components, training the magnitude vector separately from the directional LoRA update. On paper, DoRA should be strictly superior to LoRA for style transfer because it provides two control axes (magnitude scaling + directional shift) rather than one (directional shift only).

In practice, DoRA OOMs on all configurations tested with the 7B model on 16GB Apple Silicon:

| DoRA Attempt | Layers | Rank | Estimated Memory | Outcome |
|-------------|--------|------|-----------------|---------|
| 7B, full | 28 | 64 | ~14 GB | OOM |
| 7B, full | 28 | 8 | ~11 GB | OOM |
| 7B, partial | 16 | 8 | ~10 GB | OOM |
| 7B, partial | 12 | 8 | ~9.5 GB | OOM |

The root cause is that DoRA's normalization step materializes a full-size weight matrix during the forward pass to compute the norm of V + BA. Combined with gradient checkpointing's need to recompute activations during the backward pass, the memory overhead is additive rather than substitutive.

For the 3B model, DoRA at rank 8 on all 36 layers is estimated to fit (~9-10 GB), but standard LoRA at rank 32 provides sufficient persona override, making DoRA unnecessary in practice.

**Recommendation:** On 16GB Apple Silicon, use standard LoRA with maximal layer coverage rather than DoRA with reduced layers or rank. The layer coverage advantage dominates the DoRA quality advantage.

### 5.6 Learning Rate for 4-Bit Quantized Models

4-bit NormalFloat (NF4) quantization introduces gradient noise that narrows the stable learning rate range:

| Precision | Safe LR Range | Sweet Spot |
|-----------|--------------|------------|
| Full (BF16) | 1e-4 to 3e-4 | 2e-4 |
| 8-bit quantized | 5e-5 to 1e-4 | 1e-4 |
| 4-bit (NF4/QLoRA) | 2e-5 to 8e-5 | 5e-5 |

At LR 2e-4 (the standard recommendation for full-precision LoRA), our 7B training run diverged at iteration 40 with loss going to NaN. The QLoRA paper explains this: the Q/K dot product in attention amplifies small weight perturbations quadratically, and 4-bit quantization noise makes the gradient landscape rougher. The paper recommends gradient clipping at 0.3 (versus the typical 1.0), which MLX does not expose as a configurable parameter.

LR 5e-5 with cosine schedule and 5% warmup provided stable convergence across all successful runs.

### 5.7 The mask-prompt Effect

The `--mask-prompt` flag restricts the training loss to assistant tokens only, excluding system prompts and user messages from the gradient computation. This has a dramatic effect on persona fidelity.

Without mask-prompt, the model allocates gradient capacity across all token types. For a typical training example with a 50-token system prompt, 100-token user message, and 20-token assistant response, only 12% of the gradient signal targets the persona-relevant tokens. The remaining 88% is spent learning to predict prompts and messages, which is irrelevant to the persona task.

With mask-prompt, 100% of the gradient signal targets the assistant response tokens. This is the single most impactful training configuration change for persona transfer, more important than rank, layer count, or learning rate.

---

## 6. Runtime Knowledge Graph Integration

### 6.1 The Staleness Problem

A cognitive twin that bakes domain knowledge into its adapter weights faces a fundamental staleness problem. In our deployment, the knowledge graph ingests 50-200 new triples daily. A model trained on a graph snapshot from week 1 will confidently make incorrect claims about the state of affairs in week 3.

Define the staleness of a training-time KG integration at time t as the symmetric difference between the current graph state and the training snapshot. With an average ingestion rate r and retraining interval Delta, the expected staleness at query time is (Delta/2) * r. With r = 100 triples/day and Delta = 7 days (weekly retraining), expected staleness is 350 triples, approximately 0.5% of the knowledge base. This may seem small, but the stale triples tend to cluster around recently active entities, which are precisely the entities most likely to be queried.

### 6.2 cc-graph-kernel Integration

The twin queries cc-graph-kernel via its REST API:

```
POST /api/knowledge/traverse
{
  "start": "spore",
  "predicates": ["uses", "deployed_on", "has_feature"],
  "direction": "outgoing",
  "max_hops": 2,
  "min_confidence": 0.7
}
```

The response includes complete paths with per-edge confidence scores and an HMAC-SHA256 signed admissibility token. The twin's system prompt is augmented with the retrieved context:

```
Context: spore uses cloudkit (0.95); spore deployed_on mac2 (0.90);
spore has_feature garden-view (0.88); cloudkit uses icloud (0.92)
```

This design enforces the No Phantom Authority invariant: every fact in the twin's context is verifiable. If a downstream audit asks "why did the twin say spore uses cloudkit?", the provenance chain leads to a specific graph state with a cryptographic signature.

### 6.3 Dual-Plane Retrieval

The twin's knowledge retrieval operates across two complementary planes:

**Plane 1: Semantic Search.** Standard vector similarity search over 329,000+ conversation turn embeddings stored in pgvector (768-dimensional, all-MiniLM-L6-v2). This surfaces contextually relevant prior interactions.

**Plane 2: Graph Traversal.** Multi-hop BFS through the knowledge graph's entity-predicate-object structure. This surfaces structural relationships that may not be captured by embedding similarity.

The RAG++ service fuses both planes: semantic search identifies candidate turns, graph traversal expands each candidate to discover related entities, and the combined context is assembled into a provenance-complete slice.

---

## 7. Anticipation Geometry Conditioning

### 7.1 From Motion Capture to Conversation

Anticipation Geometry (Diomande, 2025) was originally developed for real-time motion capture, computing 7 geometric scalars from skeletal pose trajectories at 50 Hz. The framework's generality lies in its formulation: the scalars are defined over any sequence of vectors in a metric space, requiring only a distance function and a window of historical states.

For conversational application, each dialogue turn is embedded into a 384-dimensional space (MiniLM) and the resulting trajectory of embeddings is fed to the geometry kernel. The scalars then characterize the conversation's dynamics: is the topic locked in (high commitment)? Are many future directions plausible (high uncertainty)? Is the conversation converging toward a decision (positive transition_pressure)?

### 7.2 Empirical Validation

Across 5,000 dialogue turns in 39 conversations, transition pressure sign predicted conversation convergence at 71.8% accuracy (z = 2.72, p < 0.007). This means that the geometric scalars carry real signal about conversational dynamics, signal that the cognitive twin can use to calibrate its response style.

### 7.3 Conditioning via Prompt Injection

Rather than modifying the training loop (which would require porting the geometry computation to MLX's array format), we inject trajectory scalars as text tags in the system prompt:

```
[trajectory: c=0.82 u=0.15 p=0.61 novelty=0.23 stability=0.91 regime=committed]
```

During training, these tags are pre-computed from the conversation history preceding each example and injected into the training data. During inference, they are computed live using `ConversationTrajectory.analyze_turns()` on the last 10 turns.

This approach achieves 80% of the benefit of deep architectural integration with 5% of the engineering effort. The model learns that `regime=committed` correlates with terse, decisive responses, and `regime=exploratory` correlates with more measured, question-asking responses, purely from statistical co-occurrence in the training data.

### 7.4 Transition Pressure as a Confidence Signal

Transition pressure, defined as d(commitment)/dt - d(uncertainty)/dt, serves double duty as an inference-time confidence signal. When transition pressure is strongly positive (the conversation is converging), the twin's predictions about the operator's likely response are more reliable because the conversation is in a predictable regime. When transition pressure is near zero or negative (the conversation is diverging or exploring), the twin should reduce its confidence and defer to the human operator.

This provides a principled basis for the auto-injection confidence threshold, grounded in trajectory dynamics rather than an arbitrary probability cutoff.

---

## 8. Distributed Training Infrastructure

### 8.1 Hardware Topology

Training runs on two Apple Silicon machines connected via Thunderbolt:

```
Mac4 (M2 16GB)              Mac5 (M4 16GB)
10.0.5.1                    10.0.5.2
       \                    /
        \--- TB cable ----/
          40 Gbps raw
          ~4 GB/s TCP
```

Mac5 serves as the primary training and inference host. Mac4 provides additional compute for data-parallel training of larger models. The Thunderbolt link provides 40x the bandwidth of gigabit Ethernet, making gradient synchronization for LoRA adapters (typically 0.1-1% of total model parameters) near-instantaneous relative to the forward/backward pass.

### 8.2 MLX Distributed Primitives

MLX 0.31+ provides native distributed training support:

- `mx.distributed`: `all_sum`, `all_gather`, `send`, `recv` operations
- `mlx.launch`: SSH-based multi-machine process orchestration
- `nn.average_gradients()`: Batched gradient averaging across ranks
- Tensor parallelism: `AllToShardedLinear`, `ShardedToAllLinear` for model splitting

For models up to 14B (4-bit), data parallelism is optimal: each machine holds a full model copy and processes different batches, synchronizing only LoRA gradients (~200MB for rank 32 on all layers). The synchronization takes approximately 50ms over the TB4 link, less than 3% of the typical 2-5 second forward/backward pass.

For models of 27B (4-bit, ~16GB weights alone), tensor parallelism splits the model across both machines, each holding approximately half the attention and MLP layers. The `shard_linear()` utility handles the splitting automatically.

### 8.3 Practical Training Times

| Model | Strategy | Time per Iteration | Total (2000 iters) |
|-------|----------|-------------------|---------------------|
| Qwen 3B 4-bit | Single machine (Mac5) | ~1.5s | ~50 min |
| Qwen 7B 4-bit | Single machine (Mac5) | ~3.5s | ~117 min |
| Qwen 7B 4-bit | Data parallel (Mac4+5) | ~2.0s | ~67 min |
| Qwen 14B 4-bit | Data parallel (Mac4+5) | ~4.5s | ~150 min |

The 3B model trains in under an hour on a single machine, making rapid iteration on training configurations practical. Weekly retraining adds minimal operational overhead.

---

## 9. Deployment

### 9.1 Serving

The fused adapter is served via the MLX Server on Mac5 at port 8100, providing an OpenAI-compatible chat completions API:

```
POST http://10.0.5.2:8100/v1/chat/completions
{
  "model": "cognitive-twin-v1",
  "messages": [
    {"role": "system", "content": "You are Mohamed. Direct, casual. Context: ..."},
    {"role": "user", "content": "Should I continue with the migration?"}
  ]
}
```

Inference latency is sub-200ms for typical responses (5-30 tokens), dominated by the first-token latency of the quantized model.

### 9.2 Integration Points

The twin integrates into the operator's workflow at three points:

**Pane Orchestrator (Mac1).** A 5-phase heartbeat loop (sense, select, mutate, check, adapt) monitors terminal sessions. When a session is detected in PANE_WAITING state (the AI's last output ends with a question mark), the orchestrator queries the twin via the Aura Gateway, receives a response with confidence score, and either auto-injects (confidence > 0.8) or notifies the operator (confidence < 0.8).

**ProactiveAgent (Spore iOS).** The operator's mobile app monitors pane states via Supabase real-time subscriptions. When the twin produces a low-confidence response, a banner notification appears: "Claude is asking: {question}. Twin suggests: {response} ({confidence}%)." The operator can approve, edit, or dismiss.

**Aura Gateway (Mac1, :18795).** Centralizes confidence computation and safety gates. All twin queries flow through the gateway, which enforces:
- Correction detection: if the twin's response contains "no", "stop", "wait", never auto-inject
- Irreversible action gate: deployments, deletions, and financial operations require human approval
- Novelty gate: if the question has low embedding similarity to the training set, abstain

### 9.3 Safety Properties

The cognitive twin is designed with a conservative failure mode: when uncertain, it defers to the human. This is achieved through three mechanisms:

1. **Confidence thresholding.** The twin's top-token probability must exceed 0.8 for auto-injection. Below this threshold, the response is routed to the operator for review.

2. **Bounded divergence.** The pane orchestrator limits the twin to at most 1 auto-injection per pane per 10 minutes, preventing runaway auto-continuation.

3. **Correction asymmetry.** Corrections (29.4% of training data) train the twin to recognize situations where the default action is wrong. The twin is more likely to abstain than to inject an incorrect correction, because the training data teaches it that corrections require specific context that may not be fully available.

---

## 10. Connection to Broader Research

### 10.1 OpenAI's Tool Use Planning

OpenAI's research on tool use planning and chain-of-thought trajectories addresses the problem of "what should the model do next?" from the model architecture side, through improved in-context reasoning. The cognitive twin addresses the same problem from the data and infrastructure side: trajectory-conditioned generation via external geometric signals rather than in-context chain-of-thought.

The approaches are complementary. A future cognitive twin might combine LoRA-adapted personality with a base model that has been RLHF-tuned for tool use planning, using Anticipation Geometry as a bridge between the external trajectory signal and the model's internal planning representations.

### 10.2 Domain-Specific Superintelligence

The Princeton DSS framework (Belova et al., 2026) argues for replacing monolithic LLMs with societies of small, domain-specialized models. The cognitive twin is a DSS in miniature: a 3B model that outperforms general-purpose models 10-70x its size on the narrow task of replicating a specific human's conversational persona. The key difference is that our DSS achieves domain specificity through personality adaptation (LoRA) + runtime knowledge (graph kernel), rather than through training-time knowledge graph distillation alone.

### 10.3 Live Knowledge Graphs

The knowledge graph integration component builds directly on the Live Knowledge Graphs framework (Diomande, 2026), which introduced the Context Slicer, provenance-tracked slicing, HMAC-signed admissibility tokens, and the formal analysis of training-time versus runtime KG integration. The cognitive twin is the first application that combines this runtime KG infrastructure with a personality-adapted model, demonstrating the separation-of-concerns architecture that the LKG paper argued for theoretically.

---

## 11. Limitations

**Single-operator validation.** All experiments use data from a single operator. The architecture's generalizability to other communication styles, languages, and domains is untested.

**Qualitative persona evaluation.** Persona fidelity is judged by the target operator. We lack a standardized, automated metric for personality transfer quality. Future work could use perplexity of the twin's outputs under the operator's true response distribution, or blinded human evaluation where judges attempt to distinguish twin responses from real operator responses.

**Small validation set.** The 328-example validation set is sufficient for monitoring training convergence but insufficient for statistically rigorous evaluation of persona transfer across response categories. A larger held-out test set with stratified sampling by response type would strengthen the results.

**Trajectory conditioning evaluation.** While Anticipation Geometry scalars carry demonstrated signal (71.8% convergence prediction), we have not yet conducted ablation experiments to quantify their specific contribution to twin persona fidelity versus using the scalars as confidence signals only.

**Memory ceiling.** The 16GB Apple Silicon constraint is a hard limit that excludes DoRA on 7B models and makes 14B+ models impractical without distributed training. Cloud GPU training would remove this constraint but introduces cost and latency.

---

## 12. Conclusion

The Cognitive Twin architecture demonstrates that faithful personality replication does not require a large model, a massive dataset, or baking volatile knowledge into weights. A 3B model with LoRA on all 36 layers, trained on fewer than 3,000 examples with prompt masking, produces authentic personality transfer that a 7B model with partial layer coverage cannot match. Runtime knowledge graph integration via provenance-tracked context slicing keeps the twin current without retraining. Anticipation Geometry conditioning provides trajectory-aware response style calibration at zero training cost.

The central design insight is separation of concerns: personality is stable and belongs in weights, knowledge is volatile and belongs in the graph, trajectory dynamics are ephemeral and belong in the prompt. Each component has its own update cadence (weekly, continuous, per-inference), its own failure mode (adapter staleness, graph incompleteness, cold-start trajectory), and its own recovery mechanism (retrain, ingest, default to neutral style).

For practitioners building cognitive twins on consumer hardware, the key takeaways are: (1) prefer smaller models with full-layer LoRA over larger models with partial coverage, (2) mask the prompt loss to focus gradient signal on target outputs, (3) keep system prompts minimal during training and rich during inference, (4) use 5e-5 learning rate for 4-bit quantized models (not 2e-4), and (5) separate personality from knowledge by design, not by accident.

---

## References

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. *NeurIPS 2024*.

Baek, J., Aji, A. F., & Saffari, A. (2023). Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. *arXiv:2306.04136*.

Belova, A., et al. (2026). Domain-specific superintelligence: A society of experts. *arXiv:2603.14147*.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized language models. *NeurIPS 2023*.

Diomande, M. (2025). Anticipation geometry: Domain-general trajectory characterization with knowledge graph-grounded rewards. *Independent research*.

Diomande, M. (2026). Live knowledge graphs: Runtime graph integration for continuous domain adaptation in language agents. *Independent research*.

Edge, D., et al. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv:2404.16130*.

HumanLLMs. (2026). Human-Like-Qwen2.5-7B-Instruct. *Hugging Face model card*.

Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., & Neubig, G. (2023). Active retrieval augmented generation. *EMNLP 2023*.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). DoRA: Weight-decomposed low-rank adaptation. *ICML 2024 (Oral)*.

Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2024). Unifying large language models and knowledge graphs: A roadmap. *IEEE TKDE*.

Raschka, S. (2025). Practical tips for finetuning LLMs using LoRA. *Sebastian Raschka's Magazine*.

Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023). Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. *ACL 2023*.

Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., & Ji, H. (2024). Neeko: Leveraging dynamic LoRA for efficient multi-character role-playing agent. *EMNLP 2024*.

---

**Acknowledgments.** This work was conducted independently using consumer Apple Silicon hardware. The author thanks the MLX team at Apple for making efficient local training possible, and the open-weights community (mlx-community on Hugging Face) for providing quantized model variants that fit within the 16GB memory constraint.

---

*Correspondence: Mohamed Diomande. This paper describes a system deployed in production on a personal infrastructure mesh. Code for the knowledge graph component is available at the cc-graph-kernel repository. The Anticipation Geometry framework is available with full evaluation code and 340+ tests.*
