"""
Corpus Surgery: Clean training data by removing unjustified permission-seeking.

Components:
    - Classifier: Detect unjustified clarifications using stall/exec/blocked scoring
    - Rewriter: Use GPT 5.2 to rewrite bad turns into direct execution
    - Quarantine: Isolate friction trajectories for DPO/eval use
    - Pipeline: Orchestrate the full corpus surgery process
    - FunctionGemma Scorer: Parse directives into tool calls for verification
"""

from .types import (
    ClarificationType,
    ClassificationResult,
    QuarantineMarker,
    DPOPair,
    EvalCase,
    ValidationResult,
    FormatConstraints,
    ProcessedTurn,
    ProcessedConversation,
    ParsabilityInfo,
)

from .classifier import (
    classify_assistant_turn,
    classify_with_functiongemma,
    classify_with_functiongemma_sync,
    compute_stall_score,
    compute_exec_score,
    compute_blocked_score,
    normalize_for_matching,
    extract_format_constraints,
    compute_directive_completeness,
)

from .functiongemma_scorer import (
    FunctionGemmaDirectiveScorer,
    ParsabilityResult,
    compute_parsability_for_message,
    compute_parsability_sync,
)

from .rewriter import (
    rewrite_assistant_turn,
    validate_rewrite,
    should_rewrite,
)

from .quarantine import (
    detect_frustration,
    scan_conversation_for_friction,
    create_dpo_pair_from_quarantine,
    create_eval_case_from_quarantine,
)

from .pipeline import (
    CorpusSurgeryPipeline,
    PipelineConfig,
    PipelineStats,
    run_pipeline_sync,
)

__all__ = [
    # Types
    "ClarificationType",
    "ClassificationResult",
    "QuarantineMarker",
    "DPOPair",
    "EvalCase",
    "ValidationResult",
    "FormatConstraints",
    "ProcessedTurn",
    "ProcessedConversation",
    "ParsabilityInfo",
    # Classifier functions
    "classify_assistant_turn",
    "classify_with_functiongemma",
    "classify_with_functiongemma_sync",
    "compute_stall_score",
    "compute_exec_score",
    "compute_blocked_score",
    "normalize_for_matching",
    "extract_format_constraints",
    "compute_directive_completeness",
    # FunctionGemma scorer
    "FunctionGemmaDirectiveScorer",
    "ParsabilityResult",
    "compute_parsability_for_message",
    "compute_parsability_sync",
    # Rewriter functions
    "rewrite_assistant_turn",
    "validate_rewrite",
    "should_rewrite",
    # Quarantine functions
    "detect_frustration",
    "scan_conversation_for_friction",
    "create_dpo_pair_from_quarantine",
    "create_eval_case_from_quarantine",
    # Pipeline
    "CorpusSurgeryPipeline",
    "PipelineConfig",
    "PipelineStats",
    "run_pipeline_sync",
]

