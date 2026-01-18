"""
CognitiveTwin V3 Evaluation Suite.

Provides comprehensive regression testing and evaluation framework including:
- Policy compliance checking (no permission-seeking, etc.)
- Format adherence validation
- Content quality scoring
- Behavioral audits
- Comparison against baseline models

Usage:
    from cognitive_twin.v3.eval import (
        EvaluationPipeline,
        EvalConfig,
        RegressionTestSuite,
    )
    
    # Configure evaluation
    config = EvalConfig(
        model_id="your-model-id",
        baseline_model_id="baseline-model-id",  # Optional
    )
    
    # Run evaluation
    pipeline = EvaluationPipeline(config)
    summary = await pipeline.run()
    
    # Generate report
    pipeline.generate_report(summary, "eval_results/")
"""

from .types import (
    # Enums
    TestCategory,
    TestPriority,
    PolicyType,
    FormatConstraint,
    # Dataclasses
    EvalConfig,
    Message,
    ExpectedBehavior,
    TestCase,
    FailureDetail,
    ScoreBreakdown,
    TestResult,
    CategoryMetrics,
    EvalSummary,
)

from .test_cases import (
    QuestionPolicyTests,
    FormatComplianceTests,
    OmissionTests,
    HistoricalAnnoyanceCases,
    EdgeCaseTests,
)

from .scorers import (
    PolicyComplianceScore,
    PolicyComplianceScorer,
    FormatAdherenceScore,
    FormatAdherenceScorer,
    ContentQualityScore,
    ContentQualityScorer,
)

from .runner import RegressionTestRunner

from .suite import RegressionTestSuite

from .reporter import ReportGenerator

from .pipeline import EvaluationPipeline

__all__ = [
    # Enums
    "TestCategory",
    "TestPriority",
    "PolicyType",
    "FormatConstraint",
    # Dataclasses
    "EvalConfig",
    "Message",
    "ExpectedBehavior",
    "TestCase",
    "FailureDetail",
    "ScoreBreakdown",
    "TestResult",
    "CategoryMetrics",
    "EvalSummary",
    # Test case generators
    "QuestionPolicyTests",
    "FormatComplianceTests",
    "OmissionTests",
    "HistoricalAnnoyanceCases",
    "EdgeCaseTests",
    # Score dataclasses
    "PolicyComplianceScore",
    "FormatAdherenceScore",
    "ContentQualityScore",
    # Scorers
    "PolicyComplianceScorer",
    "FormatAdherenceScorer",
    "ContentQualityScorer",
    # Runner and suite
    "RegressionTestRunner",
    "RegressionTestSuite",
    # Reporting
    "ReportGenerator",
    # Pipeline
    "EvaluationPipeline",
]
