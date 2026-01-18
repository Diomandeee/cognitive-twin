"""
CognitiveTwin V3: Data augmentation and training pipeline.

This module implements the V3 training pipeline focused on eliminating
permission-seeking behavior through corpus surgery, data augmentation,
and DPO training.

Modules:
    corpus_surgery: Classifier, rewriter, and quarantine for training data cleanup
    api: OpenAI GPT 5.2 / Codex client integration
    worms: Repo Worm, Conversation Worm, Enhancer Agent
    schema: CTv3.1 schema definitions
    dataset: Dataset builder, labeler, pair generator, exporter
"""

from .corpus_surgery import (
    # Types
    ClarificationType,
    ClassificationResult,
    QuarantineMarker,
    DPOPair,
    EvalCase,
    FormatConstraints,
    ProcessedTurn,
    ProcessedConversation,
    # Classifier
    classify_assistant_turn,
    compute_stall_score,
    compute_exec_score,
    compute_blocked_score,
    extract_format_constraints,
    # Rewriter
    rewrite_assistant_turn,
    validate_rewrite,
    should_rewrite,
    # Quarantine
    detect_frustration,
    scan_conversation_for_friction,
    # Pipeline
    CorpusSurgeryPipeline,
    PipelineConfig,
    PipelineStats,
    run_pipeline_sync,
)

from .api import V3OpenAIClient

# Schema exports
from .schema import (
    SCHEMA_VERSION,
    RecordType,
    SourceOrigin,
    SourceProvider,
    Domain,
    QuestionPolicy,
    TaskType,
    PromptClass,
    ReviewStatus,
    FailureMode,
    DPOPairType,
    SourceInfo,
    TopologyCoords,
    PolicyInfo,
    ContextInfo,
    Message,
    Attachment,
    InputData,
    StructuredOutput,
    TargetData,
    RepoTaskInfo,
    TagInfo,
    QualityInfo,
    CTv3Record,
    DPOCandidates,
    CTv3DPORecord,
    EvalChecks,
    CTv3EvalRecord,
)

# Dataset exports
from .dataset import (
    DirectiveCompletenessLabeler,
    QuestionPolicyLabeler,
    FormatConstraintsLabeler,
    PolicyLabeler,
    Labels,
    ConfirmationReflexGenerator,
    FormatDriftGenerator,
    OmissionGenerator,
    OptionSpamGenerator,
    DPOPairGenerator,
    ExportFormat,
    DatasetSplit,
    DatasetExporter,
    DatasetBuilder,
)

# Pipeline exports
from .pipeline import (
    BASE_MODELS,
    DEFAULT_BASE_MODEL,
    TrainingConfig,
    TogetherAIClient,
    DataPreparer,
    DataValidator,
    DataUploader,
    TrainingJobManager,
    JobMonitor,
    JobController,
    StageResult,
    SFTTrainingStage,
    DPOTrainingStage,
    ModelInference,
    RegressionTester,
    ABComparison,
    PipelineResult,
    V3TrainingPipeline,
)

# Evaluation Suite exports
from .eval import (
    # Enums
    TestCategory,
    TestPriority,
    # Configuration
    EvalConfig,
    # Test structures
    TestCase,
    TestResult,
    # Scores
    PolicyComplianceScore,
    FormatAdherenceScore,
    ContentQualityScore,
    # Summary
    EvalSummary,
    # Test case generators
    QuestionPolicyTests,
    FormatComplianceTests,
    OmissionTests,
    HistoricalAnnoyanceCases,
    EdgeCaseTests,
    # Scorers
    PolicyComplianceScorer,
    FormatAdherenceScorer,
    ContentQualityScorer,
    # Components
    RegressionTestRunner,
    RegressionTestSuite,
    ReportGenerator,
    EvaluationPipeline,
)

__all__ = [
    # Types
    "ClarificationType",
    "ClassificationResult",
    "QuarantineMarker",
    "DPOPair",
    "EvalCase",
    "FormatConstraints",
    "ProcessedTurn",
    "ProcessedConversation",
    # Classifier
    "classify_assistant_turn",
    "compute_stall_score",
    "compute_exec_score",
    "compute_blocked_score",
    "extract_format_constraints",
    # Rewriter
    "rewrite_assistant_turn",
    "validate_rewrite",
    "should_rewrite",
    # Quarantine
    "detect_frustration",
    "scan_conversation_for_friction",
    # Pipeline
    "CorpusSurgeryPipeline",
    "PipelineConfig",
    "PipelineStats",
    "run_pipeline_sync",
    # API
    "V3OpenAIClient",
    # Schema
    "SCHEMA_VERSION",
    "RecordType",
    "SourceOrigin",
    "SourceProvider",
    "Domain",
    "QuestionPolicy",
    "TaskType",
    "PromptClass",
    "ReviewStatus",
    "FailureMode",
    "DPOPairType",
    "SourceInfo",
    "TopologyCoords",
    "PolicyInfo",
    "ContextInfo",
    "Message",
    "Attachment",
    "InputData",
    "StructuredOutput",
    "TargetData",
    "RepoTaskInfo",
    "TagInfo",
    "QualityInfo",
    "CTv3Record",
    "DPOCandidates",
    "CTv3DPORecord",
    "EvalChecks",
    "CTv3EvalRecord",
    # Dataset
    "DirectiveCompletenessLabeler",
    "QuestionPolicyLabeler",
    "FormatConstraintsLabeler",
    "PolicyLabeler",
    "Labels",
    "ConfirmationReflexGenerator",
    "FormatDriftGenerator",
    "OmissionGenerator",
    "OptionSpamGenerator",
    "DPOPairGenerator",
    "ExportFormat",
    "DatasetSplit",
    "DatasetExporter",
    "DatasetBuilder",
    # Pipeline
    "BASE_MODELS",
    "DEFAULT_BASE_MODEL",
    "TrainingConfig",
    "TogetherAIClient",
    "DataPreparer",
    "DataValidator",
    "DataUploader",
    "TrainingJobManager",
    "JobMonitor",
    "JobController",
    "StageResult",
    "SFTTrainingStage",
    "DPOTrainingStage",
    "ModelInference",
    "RegressionTester",
    "ABComparison",
    "PipelineResult",
    "V3TrainingPipeline",
    # Evaluation Suite
    "TestCategory",
    "TestPriority",
    "EvalConfig",
    "TestCase",
    "TestResult",
    "PolicyComplianceScore",
    "FormatAdherenceScore",
    "ContentQualityScore",
    "EvalSummary",
    "QuestionPolicyTests",
    "FormatComplianceTests",
    "OmissionTests",
    "HistoricalAnnoyanceCases",
    "EdgeCaseTests",
    "PolicyComplianceScorer",
    "FormatAdherenceScorer",
    "ContentQualityScorer",
    "RegressionTestRunner",
    "RegressionTestSuite",
    "ReportGenerator",
    "EvaluationPipeline",
]

__version__ = "3.0.0"

