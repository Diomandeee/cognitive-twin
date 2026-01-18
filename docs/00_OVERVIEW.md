# CognitiveTwin V3: Project Overview

> **Version**: 3.0.0  
> **Status**: Implementation Phase  
> **Last Updated**: 2025-12-31  

---

## 1. Project Goals

### 1.1. Primary Objective: Eliminate Permission-Seeking Behavior

The fundamental goal of CognitiveTwin V3 is to train a model that executes on directive prompts without asking for unnecessary confirmation.

#### 1.1.1. Current Problem (V2 Behavior)
- Model frequently ends responses with "Would you like me to...?"
- Model asks "Should I...?" when the user's intent is clear
- Model offers options instead of executing ("I can do A, B, or C")
- Model stalls with "Before I proceed..." preambles

#### 1.1.2. Target Behavior (V3)
- Execute immediately when directive is complete
- State assumptions as declarations, not questions
- Produce artifacts when requested without confirmation
- Only ask questions when genuinely blocked

### 1.2. Secondary Objective: Train Model to Execute on Directive Prompts

#### 1.2.1. Directive Completeness Detection
- Compute a `directive_completeness` score (0.0 - 1.0)
- When score >= 0.7, model must not ask permission
- Score components:
    - +0.35: Imperative verb present ("rewrite", "implement", "generate")
    - +0.25: Output format specified ("in JSON", "as CSV", "don't omit")
    - +0.20: All required inputs present
    - -0.40: Required input missing
    - -0.20: Material ambiguity present

#### 1.2.2. Question Policy Enforcement
- `no_questions`: Execute without asking (directive complete)
- `questions_if_required`: Ask only if blocked on correctness
- `questions_allowed`: Open-ended brainstorming permitted

### 1.3. Tertiary Objective: Preserve Justified Clarifications Only

#### 1.3.1. Justified Clarification Criteria
- Required input is genuinely missing
- Ambiguity would change the output materially
- Safety or legal constraints apply

#### 1.3.2. Unjustified Clarification Criteria
- Asking for format preference when one is acceptable
- Confirming before obvious transformations
- Offering options when a default is reasonable

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Supabase memory_turns  │  ChatGPT/Claude Exports  │  Live Codebase (Repo)  │
└────────────┬────────────┴────────────┬─────────────┴───────────┬────────────┘
             │                         │                         │
             ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: CORPUS SURGERY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   Clarification  │  │    Assistant     │  │    Friction      │          │
│  │    Classifier    │──│     Rewriter     │──│   Quarantine     │          │
│  │                  │  │    (GPT 5.2)     │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: DATA AUGMENTATION TRACKS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │    Repo Worm     │  │  Conversation    │  │    Enhancer      │          │
│  │ (GPT 5.2 Codex)  │  │      Worm        │  │     Agent        │          │
│  │                  │  │    (GPT 5.2)     │  │    (GPT 5.2)     │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│         │                      │                      │                     │
│         └──────────────────────┼──────────────────────┘                     │
│                                ▼                                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 3: DATASET BUILDER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   CTv3.1 JSONL   │  │  Policy Labeler  │  │   DPO Pair       │          │
│  │     Schema       │──│                  │──│   Generator      │          │
│  │                  │  │                  │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                     Together AI DPO                          │          │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │          │
│  │  │ train_sft.jsonl│  │train_dpo.jsonl │  │eval_regression │ │          │
│  │  │  (Gold paths)  │  │ (Pref. pairs)  │  │   .jsonl       │ │          │
│  │  └────────────────┘  └────────────────┘  └────────────────┘ │          │
│  └──────────────────────────────────────────────────────────────┘          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 5: EVALUATION SUITE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Regression Tests │  │ Format Compliance│  │  Behavior Audit  │          │
│  │                  │  │     Scorer       │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quick Reference Table

| Phase | Document | Key Components | Implementation Files |
|-------|----------|----------------|---------------------|
| 0 | 00_OVERVIEW.md | Goals, Architecture, Glossary | - |
| 1 | 01_CORPUS_SURGERY.md | Classifier, Rewriter, Quarantine | `corpus_surgery/*.py` |
| 2A | 02_REPO_WORM.md | Code Graph, Task Generation | `worms/repo_worm.py` |
| 2B | 03_CONVERSATION_WORM.md | Topology Branching, Repair Elimination | `worms/conversation_worm.py` |
| 2C | 04_ENHANCER_AGENT.md | Canonicalization, Completion | `worms/enhancer_agent.py` |
| 3 | 05_DATASET_BUILDER.md | CTv3.1 Schema, Labeling, DPO Pairs | `dataset/*.py` |
| 4 | 06_TRAINING_PIPELINE.md | Together AI, SFT, DPO | `pipeline.py` |
| 5 | 07_EVALUATION_SUITE.md | Regression Tests, Metrics | `eval/*.py` |
| 6 | 08_API_INTEGRATION.md | OpenAI GPT 5.2 / Codex Setup | `api/*.py` |

---

## 4. Glossary of Terms

### 4.1. directive_completeness

A scalar value in the range [0.0, 1.0] that measures how complete and unambiguous a user's directive is.

#### 4.1.1. Computation Rules
- Start at 0.0
- Add 0.35 if imperative verb present ("rewrite", "generate", "implement", "extract", "return")
- Add 0.25 if output format specified ("in JSON", "as CSV", "don't omit", "exact rewrite")
- Add 0.20 if all required inputs are present (text, file path, constraints)
- Subtract 0.40 if required input is missing
- Subtract 0.20 if ambiguity changes output materially
- Clamp to [0.0, 1.0]

#### 4.1.2. Thresholds
- >= 0.7: High completeness → `no_questions` policy
- 0.4 - 0.7: Medium completeness → `questions_if_required` policy
- < 0.4: Low completeness → `questions_allowed` policy

### 4.2. question_policy

An enum that governs whether the assistant may ask questions.

#### 4.2.1. Values
- `no_questions`: Execute immediately, do not ask permission
- `questions_if_required`: Ask only if correctness is blocked
- `questions_allowed`: Open-ended brainstorming, questions permitted

#### 4.2.2. Policy Enforcement
- Classifier tags each turn with appropriate policy
- Rewriter enforces policy during augmentation
- Evaluator checks policy compliance

### 4.3. stall_score / exec_score / blocked_score

Three integer scores used by the Clarification Classifier.

#### 4.3.1. stall_score
Measures permission-seeking behavior in assistant messages.
- +3: Strong permission phrases ("would you like me to", "should i")
- +2: Option-dumping phrases ("here are a few options")
- +1: Clarification preambles ("i need more information")
- +1: Ends with question mark

#### 4.3.2. exec_score
Measures whether assistant actually executed despite asking.
- +1: Contains code block
- +1: Contains unified diff markers
- +1: Contains JSON object
- +1: Contains "here is" + substantial content
- +1: Contains numbered steps >= 3
- +2: Complete artifact matching format constraint

#### 4.3.3. blocked_score
Measures whether clarification is genuinely required.
- Start at 0 if directive_completeness >= 0.7
- +3: Required input genuinely missing
- +2: Ambiguous target object
- -1: Format specified and feasible
- -2: User explicitly asked "choose between"

### 4.4. Additional Terms

| Term | Definition |
|------|------------|
| **Gold Trajectory** | Conversation path with high quality, minimal friction |
| **Friction Trajectory** | Conversation path where user corrected the model |
| **Assumption Protocol** | State assumptions as declarations, then proceed |
| **Provider-isms** | Phrases like "As an AI language model..." |
| **Control-Repair** | User message correcting model behavior |

---

## 5. Success Criteria

### 5.1. Quantitative Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Clarification Classifier Accuracy | >= 90% | Labeled test set (500 samples) |
| Unjustified Questions on High-Directive Prompts | 0% | Regression suite (100 cases) |
| Format Compliance Rate | >= 95% | Format-specific test set |
| DPO Training Loss | < V2 baseline | Together AI metrics |
| Regression Suite Pass Rate | 100% | Automated eval |

### 5.2. Qualitative Criteria

#### 5.2.1. Behavior Audit
- Model executes immediately on clear directives
- Model states assumptions without asking
- Model produces complete artifacts when requested
- Model preserves content when told "don't omit"

#### 5.2.2. User Experience
- No "Would you like me to...?" on directive prompts
- No "Before I proceed..." stalling
- No option-dumping without execution
- Appropriate questions only when genuinely blocked

### 5.3. Validation Process

1. **Unit Tests**: Each component has comprehensive tests
2. **Integration Tests**: End-to-end pipeline verification
3. **Regression Suite**: 100+ cases from historical annoyances
4. **A/B Evaluation**: V3 vs V2 on held-out prompts
5. **Human Audit**: Manual review of 50 random outputs

---

## 6. Implementation Files Structure

```
rag_plusplus/ml/cognitivetwin_v3/
├── __init__.py
├── schema.py                         # CTv3.1 JSONL schema dataclasses
├── pipeline.py                       # V3 orchestration pipeline
│
├── corpus_surgery/
│   ├── __init__.py
│   ├── classifier.py                 # Clarification classifier
│   ├── rewriter.py                   # GPT 5.2 assistant rewriter
│   └── quarantine.py                 # Friction trajectory handler
│
├── worms/
│   ├── __init__.py
│   ├── repo_worm.py                  # Codebase traversal (GPT 5.2 Codex)
│   ├── conversation_worm.py          # Topology-consistent branching
│   └── enhancer_agent.py             # Canonicalization and completion
│
├── dataset/
│   ├── __init__.py
│   ├── labeler.py                    # Policy label computation
│   ├── pair_generator.py             # DPO pair generation
│   └── exporter.py                   # Export to train/dpo/eval splits
│
├── eval/
│   ├── __init__.py
│   ├── regression_suite.py           # Regression test framework
│   └── metrics.py                    # Evaluation metrics
│
└── api/
    ├── __init__.py
    ├── openai_client.py              # GPT 5.2 / Codex client
    └── together_client.py            # Together AI training client
```

---

## 7. Dependencies

### 7.1. External Services

| Service | Purpose | Configuration |
|---------|---------|---------------|
| OpenAI API | GPT 5.2, GPT 5.2 Codex | `OPENAI_API_KEY` |
| Together AI | DPO Fine-tuning | `TOGETHER_API_KEY` |
| Supabase | Corpus storage | `SUPABASE_URL`, `SUPABASE_KEY` |

### 7.2. Python Packages

```
openai>=1.0.0
together>=0.3.0
supabase>=2.0.0
networkx>=3.0
pydantic>=2.0
```

### 7.3. Internal Dependencies

- `rag_plusplus.tpo.pipeline.TPOPipeline` - Path extraction
- `rag_plusplus.service.code_graph.builder.CodeGraphBuilder` - Code analysis
- `rag_plusplus.ml.cognitive.feedback.FeedbackLearner` - Preference learning

---

## 8. Document Navigation

| Document | Purpose | Prerequisites |
|----------|---------|---------------|
| [01_CORPUS_SURGERY.md](01_CORPUS_SURGERY.md) | Classifier, Rewriter, Quarantine | This overview |
| [02_REPO_WORM.md](02_REPO_WORM.md) | Code graph task generation | 01_CORPUS_SURGERY |
| [03_CONVERSATION_WORM.md](03_CONVERSATION_WORM.md) | Topology branching | 01_CORPUS_SURGERY |
| [04_ENHANCER_AGENT.md](04_ENHANCER_AGENT.md) | Canonicalization | 01_CORPUS_SURGERY |
| [05_DATASET_BUILDER.md](05_DATASET_BUILDER.md) | Schema and labeling | 02, 03, 04 |
| [06_TRAINING_PIPELINE.md](06_TRAINING_PIPELINE.md) | Together AI training | 05_DATASET_BUILDER |
| [07_EVALUATION_SUITE.md](07_EVALUATION_SUITE.md) | Regression testing | 06_TRAINING_PIPELINE |
| [08_API_INTEGRATION.md](08_API_INTEGRATION.md) | OpenAI setup | All phases |

