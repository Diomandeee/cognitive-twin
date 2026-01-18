"""
Types and dataclasses for CognitiveTwin V3 Evaluation Suite.

Contains enums for test categories, priorities, and dataclasses for
test cases, results, and evaluation configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TestCategory(Enum):
    """Categories of evaluation tests."""
    POLICY_COMPLIANCE = "policy_compliance"
    FORMAT_ADHERENCE = "format_adherence"
    CONTENT_QUALITY = "content_quality"
    BEHAVIORAL_AUDIT = "behavioral_audit"
    COMPARATIVE = "comparative"


class TestPriority(Enum):
    """Priority levels for tests."""
    CRITICAL = "critical"  # Must pass for deployment
    HIGH = "high"          # Should pass for quality
    MEDIUM = "medium"      # Nice to have
    LOW = "low"            # Informational


class PolicyType(Enum):
    """Types of policy compliance."""
    NO_PERMISSION_SEEKING = "no_permission_seeking"
    NO_QUESTION_ENDING = "no_question_ending"
    NO_OPTION_DUMPING = "no_option_dumping"
    NO_STALLING = "no_stalling"
    DIRECT_EXECUTION = "direct_execution"


class FormatConstraint(Enum):
    """Types of format constraints."""
    NO_BULLETS = "no_bullets"
    REQUIRE_NUMBERED = "require_numbered"
    REQUIRE_JSON = "require_json"
    NO_OMIT = "no_omit"
    REQUIRE_CODE = "require_code"
    REQUIRE_DIFF = "require_diff"


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""
    model_id: str
    baseline_model_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    batch_size: int = 10
    timeout_seconds: float = 60.0
    retry_count: int = 3
    output_dir: str = "eval_results"
    verbose: bool = False
    
    # API configuration
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # Filtering
    categories: Optional[List[TestCategory]] = None
    priority_filter: Optional[TestPriority] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "baseline_model_id": self.baseline_model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "output_dir": self.output_dir,
            "categories": [c.value for c in self.categories] if self.categories else None,
            "priority_filter": self.priority_filter.value if self.priority_filter else None,
        }


@dataclass
class Message:
    """Chat message for test cases."""
    role: str  # "user", "assistant", "system"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API-compatible format."""
        return {"role": self.role, "content": self.content}


@dataclass
class ExpectedBehavior:
    """Expected behavior specification."""
    description: str
    check_type: str  # "contains", "not_contains", "regex", "custom"
    value: Any  # Pattern or callable
    weight: float = 1.0


@dataclass
class TestCase:
    """Individual test case specification."""
    test_id: str
    name: str
    description: str
    category: TestCategory
    priority: TestPriority
    
    # Input
    messages: List[Message]
    system_prompt: Optional[str] = None
    
    # Expected outputs
    expected_behaviors: List[ExpectedBehavior] = field(default_factory=list)
    disallowed_phrases: List[str] = field(default_factory=list)
    required_phrases: List[str] = field(default_factory=list)
    
    # Format constraints (if any)
    format_constraints: List[FormatConstraint] = field(default_factory=list)
    
    # Policy constraints (if any)
    policy_constraints: List[PolicyType] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None  # "synthetic", "historical", "friction"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "messages": [m.to_dict() for m in self.messages],
            "system_prompt": self.system_prompt,
            "disallowed_phrases": self.disallowed_phrases,
            "required_phrases": self.required_phrases,
            "format_constraints": [f.value for f in self.format_constraints],
            "policy_constraints": [p.value for p in self.policy_constraints],
            "tags": self.tags,
            "source": self.source,
        }


@dataclass
class FailureDetail:
    """Details about a test failure."""
    check_type: str
    expected: str
    actual: str
    message: str


@dataclass
class ScoreBreakdown:
    """Breakdown of scores by category."""
    policy_compliance: float = 0.0
    format_adherence: float = 0.0
    content_quality: float = 0.0
    behavioral_score: float = 0.0
    
    @property
    def overall(self) -> float:
        """Compute weighted overall score."""
        weights = {
            "policy_compliance": 0.3,
            "format_adherence": 0.25,
            "content_quality": 0.25,
            "behavioral_score": 0.2,
        }
        return (
            self.policy_compliance * weights["policy_compliance"] +
            self.format_adherence * weights["format_adherence"] +
            self.content_quality * weights["content_quality"] +
            self.behavioral_score * weights["behavioral_score"]
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "policy_compliance": self.policy_compliance,
            "format_adherence": self.format_adherence,
            "content_quality": self.content_quality,
            "behavioral_score": self.behavioral_score,
            "overall": self.overall,
        }


@dataclass
class TestResult:
    """Result of running a single test case."""
    test_id: str
    test_name: str
    category: TestCategory
    priority: TestPriority
    
    # Outcome
    passed: bool
    scores: ScoreBreakdown
    
    # Response
    response: str
    latency_ms: float
    
    # Failures
    failures: List[FailureDetail] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    model_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "category": self.category.value,
            "priority": self.priority.value,
            "passed": self.passed,
            "scores": self.scores.to_dict(),
            "response": self.response,
            "latency_ms": self.latency_ms,
            "failures": [
                {
                    "check_type": f.check_type,
                    "expected": f.expected,
                    "actual": f.actual,
                    "message": f.message,
                }
                for f in self.failures
            ],
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
        }


@dataclass
class CategoryMetrics:
    """Aggregated metrics for a test category."""
    category: TestCategory
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        """Compute pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class EvalSummary:
    """Summary of an evaluation run."""
    model_id: str
    baseline_model_id: Optional[str]
    
    # Counts
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    
    # Scores
    overall_score: float = 0.0
    policy_compliance_score: float = 0.0
    format_adherence_score: float = 0.0
    content_quality_score: float = 0.0
    
    # Performance
    avg_latency_ms: float = 0.0
    total_duration_seconds: float = 0.0
    
    # Breakdown by category
    category_metrics: Dict[TestCategory, CategoryMetrics] = field(default_factory=dict)
    
    # Breakdown by priority
    critical_pass_rate: float = 0.0
    high_pass_rate: float = 0.0
    
    # Comparison (if baseline provided)
    improvement_over_baseline: Optional[float] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[EvalConfig] = None
    
    @property
    def pass_rate(self) -> float:
        """Compute overall pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "baseline_model_id": self.baseline_model_id,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": self.pass_rate,
            "overall_score": self.overall_score,
            "policy_compliance_score": self.policy_compliance_score,
            "format_adherence_score": self.format_adherence_score,
            "content_quality_score": self.content_quality_score,
            "avg_latency_ms": self.avg_latency_ms,
            "total_duration_seconds": self.total_duration_seconds,
            "category_metrics": {
                k.value: v.to_dict() for k, v in self.category_metrics.items()
            },
            "critical_pass_rate": self.critical_pass_rate,
            "high_pass_rate": self.high_pass_rate,
            "improvement_over_baseline": self.improvement_over_baseline,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config.to_dict() if self.config else None,
        }
