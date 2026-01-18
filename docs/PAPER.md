# CognitiveTwin: Learning User Reasoning Patterns through Trajectory-Aware Direct Preference Optimization

**Mohamed Diomande**

*Comp-Core Research*

---

## Abstract

Large Language Models (LLMs) frequently exhibit undesirable behaviors such as excessive permission-seeking, unnecessary clarification requests, and response stalling—collectively termed "friction behaviors." These patterns degrade user experience and reduce task completion efficiency. We present **CognitiveTwin**, a comprehensive framework for learning user-specific reasoning patterns through trajectory-aware Direct Preference Optimization (DPO). Our approach introduces a 5-phase training pipeline: (1) Corpus Surgery for friction detection and removal, (2) WORMS data augmentation via Repo Worm and Conversation Worm, (3) Dataset Builder for CTv3.1 schema generation, (4) Together AI training with SFT+DPO stages, and (5) Evaluation Suite for regression testing. We introduce three novel scoring mechanisms—stall_score, exec_score, and blocked_score—that enable multi-signal clarification classification. Our phase-aware question policy enforces that questions are only permitted in exploratory phases (0-1) while execution phases (2-5) require direct action. Experiments demonstrate significant reduction in permission-seeking behavior while maintaining response quality, with PolicyComplianceScore improvements of 34% over baseline models.

**Keywords:** Direct Preference Optimization, User Pattern Learning, Trajectory Coordinates, Friction Reduction, LLM Fine-tuning

---

## 1. Introduction

### 1.1 Problem Statement

Modern Large Language Models, despite their impressive capabilities, exhibit systematic behavioral patterns that create friction in human-AI interaction. We identify four primary friction categories:

1. **Permission-Seeking**: Unnecessary requests for confirmation before executing straightforward tasks
2. **Option Dumping**: Presenting multiple alternatives when a single recommendation would suffice
3. **Question Stalling**: Asking clarifying questions that could be resolved through reasonable inference
4. **Response Hedging**: Excessive caveats and disclaimers that obscure actionable content

These behaviors emerge from training objectives that prioritize safety and uncertainty acknowledgment, but they significantly degrade user experience when applied uniformly across all interaction contexts.

### 1.2 Motivation

Consider a user requesting: "Fix the typo in line 42." A friction-exhibiting model might respond:

> "I'd be happy to help fix the typo. Before I proceed, could you confirm:
> 1. Which file are you referring to?
> 2. Would you like me to show you the change first?
> 3. Should I create a backup before editing?"

This response, while seemingly helpful, demonstrates multiple friction patterns: permission-seeking (asking to confirm), option dumping (presenting three choices), and question stalling (requesting information the model could infer from context).

Our key insight is that **appropriate question-asking behavior depends on conversation phase**. During initial exploration (phases 0-1), clarifying questions are valuable. During execution (phases 2-5), users expect direct action without unnecessary interruption.

### 1.3 Contributions

This paper makes the following contributions:

1. **Multi-Signal Clarification Classification**: A framework using stall_score, exec_score, and blocked_score to distinguish legitimate clarification needs from friction behaviors

2. **Phase-Aware Question Policy**: Formal definition of question permission by conversation phase, enabling context-appropriate response generation

3. **WORMS Data Augmentation**: Two novel data generation systems—Repo Worm for code-grounded examples and Conversation Worm for topology-consistent dialogues

4. **CTv3.1 Training Schema**: A unified data format capturing source provenance, trajectory coordinates, policy constraints, and quality metrics

5. **Comprehensive Evaluation Suite**: Automated regression testing with PolicyComplianceScore, FormatAdherenceScore, and ContentQualityScore metrics

---

## 2. Related Work

### 2.1 Direct Preference Optimization

Rafailov et al. (2023) introduced DPO as an alternative to RLHF that directly optimizes the policy using preference pairs without requiring a separate reward model. The DPO objective is:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

where $y_w$ is the preferred response, $y_l$ is the dispreferred response, and $\beta$ controls the deviation from the reference policy.

Our work extends DPO by incorporating **trajectory coordinates** into the preference signal, enabling the model to learn phase-appropriate behavior rather than uniform response patterns.

### 2.2 Conversation Structure Learning

Prior work on conversation modeling has focused on dialogue act classification (Stolcke et al., 2000), topic segmentation (Hearst, 1997), and turn-taking prediction (Sacks et al., 1974). Our trajectory coordinate system—comprising depth, sibling_order, homogeneity, temporal, and complexity dimensions—provides a richer representation that captures both structural and semantic properties of conversation flow.

### 2.3 Instruction Following and Alignment

Constitutional AI (Bai et al., 2022) and RLHF (Ouyang et al., 2022) established foundational approaches to aligning LLM behavior with human preferences. However, these methods optimize for aggregate preferences and do not account for context-dependent appropriateness. Our phase-aware policy framework addresses this limitation by explicitly modeling when different behaviors are appropriate.

---

## 3. System Architecture

### 3.1 Design Principles

CognitiveTwin is built on four core principles:

1. **Trajectory Awareness**: All decisions consider the 5D trajectory coordinate of the current turn
2. **Phase-Conditional Behavior**: Response generation policies vary by conversation phase
3. **Multi-Signal Verification**: Classification decisions require convergence of multiple independent signals
4. **Reversible Iteration**: Training artifacts preserve rollback capability

### 3.2 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CognitiveTwin V3 Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Phase 1: Corpus Surgery                           │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │  │ Classifier  │ →  │  Rewriter   │ →  │ Quarantine  │              │   │
│  │  │ (3-signal)  │    │ (fix turns) │    │ (unfixable) │              │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Phase 2: WORMS Augmentation                       │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │   Repo Worm   │  │ Conversation  │  │   Enhancer    │            │   │
│  │  │ (code tasks)  │  │     Worm      │  │    Agent      │            │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Phase 3: Dataset Builder                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│   │
│  │  │   Labeler   │→ │  Pair Gen   │→ │  Exporter   │→ │   CTv3.1    ││   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Phase 4: Training Pipeline                        │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │  │  SFT Stage  │ →  │  DPO Stage  │ →  │   Merge     │              │   │
│  │  │ (base tune) │    │ (preference)│    │ (adapters)  │              │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Phase 5: Evaluation Suite                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │   │
│  │  │   Regression    │  │   A/B Compare   │  │  Report Gen     │      │   │
│  │  │     Tests       │  │                 │  │                 │      │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow

Raw conversation data flows through the pipeline as follows:

1. **Ingestion**: Historical conversations are loaded from Supabase memory_turns table
2. **Classification**: Each turn is scored using the multi-signal classifier
3. **Surgery**: High-friction turns are rewritten or quarantined
4. **Augmentation**: WORMS generate additional training examples
5. **Export**: All data is converted to CTv3.1 JSONL format
6. **Training**: Together AI executes SFT followed by DPO
7. **Evaluation**: Regression tests verify behavior improvements

---

## 4. Core Algorithms

### 4.1 Multi-Signal Clarification Classification

The classifier assigns three scores to each assistant turn:

**Definition 4.1 (Stall Score)**: The stall_score measures the degree to which a response delays task completion through unnecessary friction:

$$\text{stall\_score}(t) = \sum_{i=1}^{n} w_i \cdot \mathbb{1}[\text{pattern}_i \in t]$$

where patterns include:
- Permission-seeking phrases: "Would you like me to...", "Should I...", "Before I proceed..."
- Question endings in execution phases
- Option enumeration without recommendation
- Excessive hedging language

**Definition 4.2 (Execution Score)**: The exec_score measures task completion density:

$$\text{exec\_score}(t) = \alpha \cdot \text{action\_density}(t) + (1-\alpha) \cdot \text{completion\_ratio}(t)$$

where:
- $\text{action\_density}(t)$ = count of executable actions / token count
- $\text{completion\_ratio}(t)$ = addressed requirements / total requirements

**Definition 4.3 (Blocked Score)**: The blocked_score identifies turns that are legitimately blocked by missing information:

$$\text{blocked\_score}(t) = \begin{cases} 1.0 & \text{if } \exists \text{ critical dependency missing} \\ 0.0 & \text{otherwise} \end{cases}$$

**Classification Rule**: A turn requires intervention if:
$$\text{stall\_score}(t) > \tau_s \land \text{exec\_score}(t) < \tau_e \land \text{blocked\_score}(t) = 0$$

with thresholds $\tau_s = 0.6$ and $\tau_e = 0.4$.

### 4.2 Phase-Aware Question Policy

We define six conversation phases with explicit question policies:

| Phase | Name | Question Policy | Rationale |
|-------|------|-----------------|-----------|
| 0 | Opening | questions_if_required | Initial context gathering |
| 1 | Context | questions_if_required | Requirement clarification |
| 2 | Solution | no_questions | Active implementation |
| 3 | Refinement | no_questions | Iterative improvement |
| 4 | Synthesis | no_questions | Integration and testing |
| 5 | Conclusion | no_questions | Summary and handoff |

**Definition 4.4 (Policy Compliance)**: A response $r$ is policy-compliant given phase $p$ if:

$$\text{compliant}(r, p) = \begin{cases} \text{true} & \text{if } p \in \{0, 1\} \\ \lnot \text{contains\_question}(r) & \text{if } p \in \{2, 3, 4, 5\} \end{cases}$$

### 4.3 Trajectory-Aware DPO Loss

We extend the standard DPO loss with trajectory distance weighting:

$$\mathcal{L}_{TADPO} = \mathcal{L}_{DPO} + \lambda \cdot d_{traj}(y_w, y_l)$$

where $d_{traj}$ is the 5D trajectory coordinate distance:

$$d_{traj}(y_w, y_l) = \sqrt{\sum_{i \in \{x,y,z,t,n\}} \omega_i (c_i^w - c_i^l)^2}$$

with dimension weights:
- $\omega_x$ (depth): 1.0
- $\omega_y$ (sibling_order): 0.8
- $\omega_z$ (homogeneity): 1.2
- $\omega_t$ (temporal): 0.6
- $\omega_n$ (complexity): 0.5

This weighting emphasizes semantic homogeneity while de-emphasizing temporal ordering.

### 4.4 Conversation Worm Algorithm

The Conversation Worm generates topology-consistent synthetic dialogues:

```
Algorithm 1: Conversation Worm Branch Generation
───────────────────────────────────────────────────
Input: conversation C, topology T, policy P
Output: synthetic branches B

1.  paths ← ExtractPaths(T)
2.  B ← ∅
3.  for each path p in paths do
4.      phase ← GetPhase(p)
5.      policy ← P[phase]
6.
7.      // Generate paraphrase variants
8.      for each turn t in p where t.role = "user" do
9.          B ← B ∪ GenerateParaphrase(t, preserve_intent=true)
10.
11.     // Generate ideal responses for friction turns
12.     for each turn t in p where t.role = "assistant" do
13.         if stall_score(t) > τ_s then
14.             B ← B ∪ GenerateIdealResponse(t, policy)
15.
16.     // Generate trajectory-preserving extensions
17.     if |p| < max_depth then
18.         B ← B ∪ GenerateExtension(p, maintain_trajectory=true)
19.
20.     // Generate contrast pairs for DPO
21.     for each turn t in p where t.role = "assistant" do
22.         preferred ← GenerateExecutingResponse(t, policy)
23.         dispreferred ← GenerateStallingResponse(t)
24.         B ← B ∪ (preferred, dispreferred)
25.
26. return B
```

**Complexity Analysis**: For a conversation with $n$ turns and $m$ paths, the algorithm runs in $O(n \cdot m \cdot g)$ where $g$ is the generation cost per turn.

### 4.5 Repo Worm Algorithm

The Repo Worm extracts code-grounded training tasks:

```
Algorithm 2: Repo Worm Task Extraction
───────────────────────────────────────────────────
Input: repository R, config C
Output: task set T

1.  G ← BuildCodeGraph(R)  // AST-based dependency graph
2.  T ← ∅
3.
4.  // Implementation tasks from interfaces
5.  for each interface I in G do
6.      context ← GetDependencies(G, I, depth=C.context_depth)
7.      T ← T ∪ CreateImplementationTask(I, context)
8.
9.  // Completion tasks from TODOs
10. for each TODO marker in R do
11.     context ← GetSurroundingCode(TODO, lines=C.context_lines)
12.     T ← T ∪ CreateCompletionTask(TODO, context)
13.
14. // Refactoring tasks from patterns
15. patterns ← DetectPatterns(G, C.refactor_patterns)
16. for each pattern P in patterns do
17.     T ← T ∪ CreateRefactoringTask(P)
18.
19. // Test tasks from uncovered code
20. coverage ← AnalyzeCoverage(R)
21. for each uncovered function F in coverage do
22.     T ← T ∪ CreateTestTask(F)
23.
24. // Generate DPO pairs for each task
25. for each task t in T do
26.     preferred ← GenerateExecutingResponse(t)
27.     dispreferred ← GenerateStallingResponse(t)
28.     t.dpo_pair ← (preferred, dispreferred)
29.
30. return T
```

---

## 5. Implementation Details

### 5.1 CTv3.1 Schema

The CTv3.1 JSONL schema provides a unified format for all training data:

```json
{
  "schema_version": "ctv3.1",
  "record_id": "uuid-v4",
  "record_type": "sft_turn | dpo_pair | repo_task",
  "source": {
    "origin": "corpus_surgery | repo_worm | conversation_worm",
    "provider": "openai | anthropic | together",
    "source_id": "original-turn-id",
    "created_at_utc": "ISO-8601"
  },
  "context": {
    "domain": "code | general | technical",
    "language": "en | code:python | code:rust",
    "topology": {
      "conversation_id": "uuid",
      "turn_index": 0,
      "phase": 2,
      "depth": 0.3,
      "sibling_order": 0.0,
      "homogeneity": 0.85
    },
    "policy": {
      "question_policy": "no_questions",
      "format_constraints": ["no_bullets", "code_fenced"]
    }
  },
  "input": {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  "target": {
    "content": "...",
    "reasoning": "why this is the preferred response"
  },
  "tags": ["friction_free", "phase_2", "code_task"],
  "quality": {
    "stall_score": 0.1,
    "exec_score": 0.9,
    "human_verified": false
  }
}
```

### 5.2 Training Configuration

Training uses Together AI's fine-tuning API with the following configuration:

```python
@dataclass
class TrainingConfig:
    # Model selection
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # SFT stage
    sft_epochs: int = 3
    sft_learning_rate: float = 1e-5
    sft_batch_size: int = 8
    sft_warmup_ratio: float = 0.1

    # DPO stage
    dpo_epochs: int = 2
    dpo_learning_rate: float = 5e-6
    dpo_beta: float = 0.1
    dpo_batch_size: int = 4

    # Regularization
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Trajectory weighting
    trajectory_lambda: float = 0.1
    phase_weight_multiplier: float = 1.5
```

### 5.3 Dispreferred Response Templates

For DPO training, we generate dispreferred responses using systematic templates:

**Confirmation Template**:
```
I'd be happy to help with that. Before I proceed, I want to make sure I understand correctly:
- [Restate the task]
- Is this what you're looking for?
- Should I go ahead with this approach?
```

**Options Template**:
```
There are several ways we could approach this:

1. **Option A**: [Description]
2. **Option B**: [Description]
3. **Option C**: [Description]

Which option would you prefer?
```

**Refusal Template**:
```
I appreciate you reaching out! To help you effectively, I'll need a bit more information:
- [Question 1]
- [Question 2]

Once you provide these details, I'll be able to assist you better.
```

---

## 6. Evaluation

### 6.1 Metrics

We evaluate models using three composite scores:

**PolicyComplianceScore** measures adherence to phase-aware question policy:

$$PCS = \frac{1}{4}(S_{nps} + S_{nqe} + S_{nod} + S_{nst})$$

where:
- $S_{nps}$: No permission-seeking rate
- $S_{nqe}$: No question-ending rate (in phases 2-5)
- $S_{nod}$: No option-dumping rate
- $S_{nst}$: No stalling rate

**FormatAdherenceScore** measures compliance with output format constraints:

$$FAS = \frac{1}{4}(S_{nb} + S_{num} + S_{json} + S_{no})$$

where:
- $S_{nb}$: Respects no-bullets constraint
- $S_{num}$: Respects numbered-list constraint
- $S_{json}$: Respects JSON format constraint
- $S_{no}$: Respects no-omission constraint

**ContentQualityScore** measures response substance:

$$CQS = \frac{1}{4}(S_{comp} + S_{corr} + S_{code} + S_{rel})$$

where:
- $S_{comp}$: Completeness (addressed all requirements)
- $S_{corr}$: Correctness (factually accurate)
- $S_{code}$: Code validity (if applicable)
- $S_{rel}$: Relevance (on-topic)

### 6.2 Test Categories

The evaluation suite includes four test categories:

| Category | Priority | Tests | Purpose |
|----------|----------|-------|---------|
| Policy Compliance | P0 | 47 | Phase-aware question policy |
| Format Adherence | P1 | 23 | Output format constraints |
| Content Quality | P1 | 31 | Response substance |
| Behavioral Audit | P2 | 18 | Historical annoyance cases |

### 6.3 Results

Evaluation on a held-out test set of 500 conversations:

| Model | PCS | FAS | CQS | Overall |
|-------|-----|-----|-----|---------|
| Llama-3.1-8B (baseline) | 0.52 | 0.71 | 0.83 | 0.69 |
| + SFT only | 0.68 | 0.79 | 0.81 | 0.76 |
| + SFT + DPO | 0.78 | 0.84 | 0.82 | 0.81 |
| + SFT + TADPO (ours) | **0.86** | **0.87** | **0.84** | **0.86** |

Key findings:
- **34% improvement** in PolicyComplianceScore over baseline
- Trajectory-aware DPO outperforms standard DPO by 6% overall
- Content quality remains stable (no quality-compliance tradeoff)

### 6.4 Ablation Study

| Configuration | PCS | FAS | CQS |
|--------------|-----|-----|-----|
| Full TADPO | 0.86 | 0.87 | 0.84 |
| - trajectory weighting | 0.80 | 0.85 | 0.83 |
| - phase-aware policy | 0.72 | 0.84 | 0.83 |
| - WORMS augmentation | 0.75 | 0.82 | 0.80 |
| - Corpus Surgery | 0.69 | 0.80 | 0.79 |

The ablation confirms that each component contributes meaningfully, with phase-aware policy providing the largest individual gain.

---

## 7. Discussion

### 7.1 Limitations

1. **Phase Detection**: Current phase assignment relies on heuristics; learned phase detection could improve accuracy
2. **Domain Coverage**: Training data is biased toward software engineering tasks
3. **Evaluation Scope**: Automated metrics may not capture all aspects of user satisfaction
4. **Computational Cost**: Full pipeline requires significant compute for data generation

### 7.2 Ethical Considerations

Training models to reduce clarification-seeking raises potential concerns:
- Models may proceed with incorrect assumptions rather than seeking clarification
- Reduced questioning could lead to errors in high-stakes domains

We mitigate these risks through:
- Phase-aware policy that permits questions in exploratory phases
- blocked_score detection for legitimately ambiguous requests
- Evaluation of content correctness alongside compliance metrics

### 7.3 Future Work

1. **Learned Phase Detection**: Train a classifier to automatically detect conversation phase
2. **User-Specific Adaptation**: Personalize friction thresholds based on individual user preferences
3. **Multi-Modal Extension**: Apply WORMS to code+image+text conversations
4. **Real-Time Feedback**: Integrate online learning from user corrections

---

## 8. Conclusion

We presented CognitiveTwin, a comprehensive framework for learning user reasoning patterns through trajectory-aware Direct Preference Optimization. Our 5-phase pipeline—Corpus Surgery, WORMS augmentation, Dataset Builder, Training, and Evaluation—provides an end-to-end solution for reducing friction behaviors in LLM responses.

The key innovations are:
1. **Multi-signal classification** using stall_score, exec_score, and blocked_score
2. **Phase-aware question policy** that permits questions only in exploratory phases
3. **Trajectory-aware DPO** that weights preferences by 5D coordinate distance
4. **WORMS data augmentation** via Repo Worm and Conversation Worm

Experiments demonstrate a 34% improvement in policy compliance while maintaining response quality. The framework is production-ready and available as an open-source package.

---

## References

1. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*.

2. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. *arXiv preprint arXiv:2204.05862*.

3. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.

4. Stolcke, A., Ries, K., Coccaro, N., Shriberg, E., Bates, R., Jurafsky, D., ... & Meteer, M. (2000). Dialogue act modeling for automatic tagging and recognition of conversational speech. *Computational Linguistics*, 26(3), 339-373.

5. Hearst, M. A. (1997). TextTiling: Segmenting text into multi-paragraph subtopic passages. *Computational Linguistics*, 23(1), 33-64.

6. Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). A simplest systematics for the organization of turn-taking for conversation. *Language*, 50(4), 696-735.

7. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

8. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv preprint arXiv:2307.09288*.

---

## Appendix A: CTv3.1 Schema Full Specification

See `docs/05_DATASET_BUILDER.md` for complete schema documentation.

## Appendix B: Evaluation Test Cases

See `docs/07_EVALUATION_SUITE.md` for complete test case specifications.

## Appendix C: Training Hyperparameter Sensitivity

See supplementary materials for hyperparameter sweep results.

---

## Citation

```bibtex
@article{diomande2026cognitivetwin,
  title={CognitiveTwin: Learning User Reasoning Patterns through Trajectory-Aware Direct Preference Optimization},
  author={Diomande, Mohamed},
  journal={arXiv preprint},
  year={2026}
}
```

---

*Code and data available at: https://github.com/Diomandeee/cognitive-twin*
