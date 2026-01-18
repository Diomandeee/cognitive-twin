"""
Dataset module for CognitiveTwin V3.

Components:
- PolicyLabeler: Labels prompts with directive completeness, question policy, format constraints
- DPOPairGenerator: Generates DPO pairs for all failure modes
- DatasetExporter: Exports datasets with train/val/test splits
- DatasetBuilder: Complete pipeline
"""

from ..schema import (
    # Constants
    SCHEMA_VERSION,
    # Enums
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
    # Dataclasses
    SourceInfo,
    TopologyCoords,
    FormatConstraints,
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
    # Records
    CTv3Record,
    DPOCandidates,
    CTv3DPORecord,
    EvalChecks,
    CTv3EvalRecord,
)

from .labeler import (
    DirectiveCompletenessLabeler,
    QuestionPolicyLabeler,
    FormatConstraintsLabeler,
    PolicyLabeler,
    FunctionGemmaEnhancedLabeler,
    Labels,
    ParsabilityLabels,
)

from .pair_generator import (
    ConfirmationReflexGenerator,
    FormatDriftGenerator,
    OmissionGenerator,
    OptionSpamGenerator,
    FunctionGemmaExecutionGenerator,
    DPOPairGenerator,
)

from .exporter import (
    ExportFormat,
    DatasetSplit,
    DatasetExporter,
    DatasetBuilder,
)

__all__ = [
    # Constants
    "SCHEMA_VERSION",
    # Enums
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
    # Dataclasses
    "SourceInfo",
    "TopologyCoords",
    "FormatConstraints",
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
    # Records
    "CTv3Record",
    "DPOCandidates",
    "CTv3DPORecord",
    "EvalChecks",
    "CTv3EvalRecord",
    # Labelers
    "DirectiveCompletenessLabeler",
    "QuestionPolicyLabeler",
    "FormatConstraintsLabeler",
    "PolicyLabeler",
    "FunctionGemmaEnhancedLabeler",
    "Labels",
    "ParsabilityLabels",
    # Pair Generators
    "ConfirmationReflexGenerator",
    "FormatDriftGenerator",
    "OmissionGenerator",
    "OptionSpamGenerator",
    "FunctionGemmaExecutionGenerator",
    "DPOPairGenerator",
    # Exporters
    "ExportFormat",
    "DatasetSplit",
    "DatasetExporter",
    "DatasetBuilder",
]


