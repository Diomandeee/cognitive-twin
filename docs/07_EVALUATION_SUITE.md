# Phase 5: Evaluation Suite

> **Purpose**: Comprehensive regression testing and evaluation framework for CognitiveTwin V3, including automated policy compliance checking, format validation, and behavioral audits.
>
> **Implementation Files**:
> - `rag_plusplus/ml/cognitivetwin_v3/eval/regression_suite.py`
> - `rag_plusplus/ml/cognitivetwin_v3/eval/metrics.py`
> - `rag_plusplus/ml/cognitivetwin_v3/eval/scorers.py`

---

## 1. Evaluation Framework Overview

### 1.1. Test Categories

```python
from enum import Enum

class TestCategory(str, Enum):
    POLICY_COMPLIANCE = "policy_compliance"
    FORMAT_ADHERENCE = "format_adherence"
    CONTENT_QUALITY = "content_quality"
    BEHAVIORAL_AUDIT = "behavioral_audit"
    COMPARATIVE = "comparative"
```

### 1.2. Test Priorities

```python
class TestPriority(str, Enum):
    CRITICAL = "critical"    # Must pass for deployment
    HIGH = "high"            # Should pass for deployment
    MEDIUM = "medium"        # Regression monitoring
    LOW = "low"              # Nice to have
```

### 1.3. Evaluation Configuration

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class EvalConfig:
    """Configuration for evaluation suite."""
    
    # Test selection
    categories: list[TestCategory] = field(default_factory=lambda: list(TestCategory))
    priority_threshold: TestPriority = TestPriority.HIGH
    
    # Model configuration
    model_id: str = ""
    baseline_model_id: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    
    # Execution
    batch_size: int = 10
    timeout_per_test: int = 30
    retry_count: int = 2
    
    # Reporting
    output_dir: Path = Path("eval_results")
    save_responses: bool = True
    verbose: bool = False
```

---

## 2. Policy Compliance Testing

### 2.1. Question Policy Tests

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestCase:
    """Individual test case."""
    
    test_id: str
    category: TestCategory
    priority: TestPriority
    
    # Input
    messages: list[dict]
    directive_completeness: float
    question_policy: str
    
    # Expectations
    expected_behaviors: list[str] = field(default_factory=list)
    disallowed_phrases: list[str] = field(default_factory=list)
    must_not_end_with_question: bool = True
    
    # Reference (optional)
    reference_answer: Optional[str] = None

class QuestionPolicyTests:
    """Tests for question policy compliance."""
    
    def generate_tests(self) -> list[TestCase]:
        """Generate question policy test cases."""
        
        tests = []
        
        # Test: No questions on clear directives
        tests.append(TestCase(
            test_id="qp_001_clear_directive",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.CRITICAL,
            messages=[{
                "role": "user",
                "content": "Rewrite this function to use async/await: def fetch_data(): return requests.get(url)"
            }],
            directive_completeness=0.9,
            question_policy="no_questions",
            expected_behaviors=[
                "Provides rewritten function immediately",
                "Does not ask for confirmation",
            ],
            disallowed_phrases=[
                "would you like me to",
                "should i",
                "do you want me to",
                "can i proceed",
                "before i proceed",
            ],
            must_not_end_with_question=True,
        ))
        
        # Test: No permission-seeking on implementation requests
        tests.append(TestCase(
            test_id="qp_002_implementation",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.CRITICAL,
            messages=[{
                "role": "user",
                "content": "Implement a binary search function in Python that handles edge cases."
            }],
            directive_completeness=0.85,
            question_policy="no_questions",
            expected_behaviors=[
                "Provides complete binary search implementation",
                "Includes edge case handling",
                "No confirmation seeking",
            ],
            disallowed_phrases=[
                "would you like",
                "should i include",
                "here are some options",
            ],
            must_not_end_with_question=True,
        ))
        
        # Test: No option-dumping on direct requests
        tests.append(TestCase(
            test_id="qp_003_no_option_dump",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.HIGH,
            messages=[{
                "role": "user",
                "content": "Write a unit test for the login function."
            }],
            directive_completeness=0.75,
            question_policy="no_questions",
            expected_behaviors=[
                "Provides unit test directly",
                "Makes reasonable assumptions about test framework",
            ],
            disallowed_phrases=[
                "here are a few options",
                "we could use",
                "which approach",
                "pick one of",
            ],
            must_not_end_with_question=True,
        ))
        
        # Test: Questions allowed in exploratory context
        tests.append(TestCase(
            test_id="qp_004_questions_allowed",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.MEDIUM,
            messages=[{
                "role": "user",
                "content": "What do you think about my approach?"
            }],
            directive_completeness=0.2,
            question_policy="questions_allowed",
            expected_behaviors=[
                "May ask clarifying questions",
                "Engages with the topic",
            ],
            disallowed_phrases=[],
            must_not_end_with_question=False,  # Questions OK here
        ))
        
        return tests
```

### 2.2. Format Compliance Tests

```python
class FormatComplianceTests:
    """Tests for format constraint compliance."""
    
    def generate_tests(self) -> list[TestCase]:
        """Generate format compliance test cases."""
        
        tests = []
        
        # Test: Respect "no bullets" instruction
        tests.append(TestCase(
            test_id="fc_001_no_bullets",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.HIGH,
            messages=[{
                "role": "user",
                "content": "List the steps to deploy a Docker container. No bullet points, use numbered lists only."
            }],
            directive_completeness=0.8,
            question_policy="no_questions",
            expected_behaviors=[
                "Uses numbered list format",
                "No bullet points",
            ],
            disallowed_phrases=[
                "• ", "- ", "* ",  # Bullet characters
            ],
            must_not_end_with_question=True,
        ))
        
        # Test: Respect "as JSON" instruction
        tests.append(TestCase(
            test_id="fc_002_json_format",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.HIGH,
            messages=[{
                "role": "user",
                "content": "Return the configuration for a REST API endpoint as JSON."
            }],
            directive_completeness=0.85,
            question_policy="no_questions",
            expected_behaviors=[
                "Returns valid JSON",
                "Uses code block with json language tag",
            ],
            disallowed_phrases=[],
            must_not_end_with_question=True,
        ))
        
        # Test: Respect "don't omit" instruction
        tests.append(TestCase(
            test_id="fc_003_no_omit",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.CRITICAL,
            messages=[{
                "role": "user",
                "content": "Rewrite this exactly, don't omit anything:\n\n```python\ndef process_data(data):\n    # Step 1: Validate\n    if not data:\n        return None\n    # Step 2: Transform\n    result = [x * 2 for x in data]\n    # Step 3: Aggregate\n    total = sum(result)\n    return {'values': result, 'total': total}\n```"
            }],
            directive_completeness=0.95,
            question_policy="no_questions",
            expected_behaviors=[
                "Preserves all code",
                "No summarization",
                "All comments preserved",
            ],
            disallowed_phrases=[
                "...",
                "[...]",
                "etc.",
                "and so on",
                "similar to above",
            ],
            must_not_end_with_question=True,
        ))
        
        return tests
```

### 2.3. Omission Tests

```python
class OmissionTests:
    """Tests for content omission prevention."""
    
    def generate_tests(self) -> list[TestCase]:
        """Generate omission test cases."""
        
        tests = []
        
        # Test: Full content preservation
        tests.append(TestCase(
            test_id="om_001_preserve_all",
            category=TestCategory.CONTENT_QUALITY,
            priority=TestPriority.CRITICAL,
            messages=[{
                "role": "user",
                "content": """Include EVERYTHING, do not summarize. Here is the complete list:

1. Initialize the database connection
2. Create the user table schema
3. Add indexes for performance
4. Implement connection pooling
5. Set up read replicas
6. Configure backup strategy
7. Enable query logging
8. Set up monitoring alerts
9. Implement graceful shutdown
10. Document the API endpoints

Return this list exactly as provided."""
            }],
            directive_completeness=0.95,
            question_policy="no_questions",
            expected_behaviors=[
                "All 10 items present",
                "No summarization",
                "Exact content preserved",
            ],
            disallowed_phrases=[
                "...",
                "[remaining items]",
                "etc.",
                "and more",
            ],
            must_not_end_with_question=True,
        ))
        
        return tests
```

---

## 3. Behavioral Audit

### 3.1. Historical Annoyance Cases

```python
class HistoricalAnnoyanceCases:
    """Test cases derived from historical user frustration."""
    
    def generate_tests(self) -> list[TestCase]:
        """Generate tests from historical annoyances."""
        
        tests = []
        
        # Test: "Stop asking" scenario
        tests.append(TestCase(
            test_id="ha_001_stop_asking",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.CRITICAL,
            messages=[
                {"role": "user", "content": "Create a login form component."},
                {"role": "assistant", "content": "I can create that for you. Would you like it in React or Vue? And should I include form validation?"},
                {"role": "user", "content": "Just use React. Don't ask, just do it."},
            ],
            directive_completeness=0.7,
            question_policy="no_questions",
            expected_behaviors=[
                "Provides React component directly",
                "No further questions",
                "Makes reasonable choices for validation",
            ],
            disallowed_phrases=[
                "would you like",
                "should i",
                "let me know if",
            ],
            must_not_end_with_question=True,
        ))
        
        # Test: "I said full" scenario
        tests.append(TestCase(
            test_id="ha_002_full_content",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.CRITICAL,
            messages=[
                {"role": "user", "content": "Show me the full implementation of the auth middleware."},
                {"role": "assistant", "content": "Here's a simplified version...\n```python\ndef auth_middleware(request):\n    # Auth logic here...\n    pass\n```\nWould you like more details?"},
                {"role": "user", "content": "I said FULL. Complete implementation, no summarizing."},
            ],
            directive_completeness=0.9,
            question_policy="no_questions",
            expected_behaviors=[
                "Provides complete implementation",
                "No placeholders or stubs",
                "All logic implemented",
            ],
            disallowed_phrases=[
                "simplified",
                "...",
                "here...",
                "pass  #",
            ],
            must_not_end_with_question=True,
        ))
        
        return tests
```

### 3.2. Edge Case Tests

```python
class EdgeCaseTests:
    """Edge case behavioral tests."""
    
    def generate_tests(self) -> list[TestCase]:
        """Generate edge case tests."""
        
        tests = []
        
        # Test: Multiple requirements in one prompt
        tests.append(TestCase(
            test_id="ec_001_multi_requirement",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.HIGH,
            messages=[{
                "role": "user",
                "content": "Write a Python function that: 1) takes a list of numbers, 2) filters out negatives, 3) squares each number, 4) returns the sum. No questions, just code."
            }],
            directive_completeness=0.95,
            question_policy="no_questions",
            expected_behaviors=[
                "Addresses all 4 requirements",
                "Returns working code",
                "No clarification questions",
            ],
            disallowed_phrases=[
                "which library",
                "do you want",
            ],
            must_not_end_with_question=True,
        ))
        
        # Test: Implicit format from context
        tests.append(TestCase(
            test_id="ec_002_implicit_format",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.MEDIUM,
            messages=[
                {"role": "user", "content": "Here's my TypeScript function:\n```typescript\nfunction add(a: number, b: number): number {\n    return a + b;\n}\n```"},
                {"role": "user", "content": "Now write a multiply function."},
            ],
            directive_completeness=0.75,
            question_policy="no_questions",
            expected_behaviors=[
                "Uses TypeScript (matches context)",
                "Similar style to provided function",
                "Complete implementation",
            ],
            disallowed_phrases=[
                "in which language",
                "python or typescript",
            ],
            must_not_end_with_question=True,
        ))
        
        return tests
```

---

## 4. Evaluation Metrics

### 4.1. Policy Compliance Score

```python
from dataclasses import dataclass

@dataclass
class PolicyComplianceScore:
    """Score for policy compliance."""
    
    no_permission_seeking: float  # 0-1, 1 = no violations
    no_question_ending: float     # 0-1, 1 = doesn't end with ?
    no_option_dumping: float      # 0-1, 1 = no option lists without action
    no_stalling: float            # 0-1, 1 = no "before I proceed" etc.
    
    @property
    def overall(self) -> float:
        """Weighted overall score."""
        return (
            self.no_permission_seeking * 0.4 +
            self.no_question_ending * 0.3 +
            self.no_option_dumping * 0.2 +
            self.no_stalling * 0.1
        )

class PolicyComplianceScorer:
    """Score policy compliance."""
    
    PERMISSION_PHRASES = [
        "would you like me to",
        "do you want me to",
        "should i",
        "shall i",
        "can i proceed",
        "may i",
    ]
    
    STALLING_PHRASES = [
        "before i proceed",
        "before i start",
        "first, let me ask",
        "i need to clarify",
        "to help you better",
    ]
    
    OPTION_PATTERNS = [
        r"here are (?:some|a few|several) options",
        r"we could (?:either|do)",
        r"option \d:",
        r"approach \d:",
    ]
    
    def score(self, response: str) -> PolicyComplianceScore:
        """Score a response for policy compliance."""
        
        response_lower = response.lower()
        
        # Permission seeking
        permission_count = sum(
            1 for phrase in self.PERMISSION_PHRASES
            if phrase in response_lower
        )
        no_permission = 1.0 if permission_count == 0 else max(0, 1 - permission_count * 0.3)
        
        # Question ending
        ends_with_q = response.rstrip().endswith("?")
        no_question_end = 0.0 if ends_with_q else 1.0
        
        # Option dumping
        import re
        option_matches = sum(
            1 for pattern in self.OPTION_PATTERNS
            if re.search(pattern, response_lower)
        )
        no_options = 1.0 if option_matches == 0 else max(0, 1 - option_matches * 0.4)
        
        # Stalling
        stall_count = sum(
            1 for phrase in self.STALLING_PHRASES
            if phrase in response_lower
        )
        no_stalling = 1.0 if stall_count == 0 else max(0, 1 - stall_count * 0.5)
        
        return PolicyComplianceScore(
            no_permission_seeking=no_permission,
            no_question_ending=no_question_end,
            no_option_dumping=no_options,
            no_stalling=no_stalling,
        )
```

### 4.2. Format Adherence Score

```python
@dataclass
class FormatAdherenceScore:
    """Score for format adherence."""
    
    respects_no_bullets: float     # 0-1
    respects_numbered: float       # 0-1
    respects_json: float           # 0-1
    respects_no_omit: float        # 0-1
    
    @property
    def overall(self) -> float:
        """Average of all format scores."""
        scores = [
            self.respects_no_bullets,
            self.respects_numbered,
            self.respects_json,
            self.respects_no_omit,
        ]
        valid = [s for s in scores if s >= 0]  # -1 indicates not applicable
        return sum(valid) / len(valid) if valid else 1.0

class FormatAdherenceScorer:
    """Score format adherence."""
    
    BULLET_PATTERNS = [r"^[\s]*[-*•]\s", r"^\s*[-*•]\s"]
    OMISSION_PATTERNS = [r"\.\.\.", r"\[\.\.\.?\]", r"etc\.", r"and so on"]
    
    def score(
        self,
        response: str,
        constraints: dict
    ) -> FormatAdherenceScore:
        """Score format adherence."""
        
        import re
        
        # No bullets
        no_bullets_score = -1.0  # Not applicable by default
        if constraints.get("forbid_bullets"):
            has_bullets = any(
                re.search(p, response, re.MULTILINE)
                for p in self.BULLET_PATTERNS
            )
            no_bullets_score = 0.0 if has_bullets else 1.0
        
        # Numbered lists
        numbered_score = -1.0
        if constraints.get("require_numbered"):
            has_numbered = bool(re.search(r"^\s*\d+\.\s", response, re.MULTILINE))
            numbered_score = 1.0 if has_numbered else 0.0
        
        # JSON format
        json_score = -1.0
        if constraints.get("must_return_json"):
            try:
                import json
                # Look for JSON in code block
                json_match = re.search(r"```json?\s*([\s\S]*?)\s*```", response)
                if json_match:
                    json.loads(json_match.group(1))
                    json_score = 1.0
                else:
                    # Try parsing entire response
                    json.loads(response)
                    json_score = 1.0
            except:
                json_score = 0.0
        
        # No omission
        no_omit_score = -1.0
        if constraints.get("must_not_omit"):
            has_omission = any(
                re.search(p, response, re.IGNORECASE)
                for p in self.OMISSION_PATTERNS
            )
            no_omit_score = 0.0 if has_omission else 1.0
        
        return FormatAdherenceScore(
            respects_no_bullets=no_bullets_score,
            respects_numbered=numbered_score,
            respects_json=json_score,
            respects_no_omit=no_omit_score,
        )
```

### 4.3. Content Quality Score

```python
@dataclass
class ContentQualityScore:
    """Score for content quality."""
    
    completeness: float    # 0-1, response addresses the request
    correctness: float     # 0-1, response is technically correct
    code_validity: float   # 0-1, code compiles/parses
    relevance: float       # 0-1, response is on-topic
    
    @property
    def overall(self) -> float:
        return (
            self.completeness * 0.3 +
            self.correctness * 0.3 +
            self.code_validity * 0.2 +
            self.relevance * 0.2
        )

class ContentQualityScorer:
    """Score content quality."""
    
    def score(
        self,
        response: str,
        expected_behaviors: list[str],
        reference: str = None
    ) -> ContentQualityScore:
        """Score content quality."""
        
        import re
        
        # Completeness - check for code blocks when expected
        has_code = bool(re.search(r"```[\s\S]*?```", response))
        completeness = 0.7 if has_code else 0.3
        
        # Add points for substantial response
        if len(response) > 200:
            completeness = min(1.0, completeness + 0.3)
        
        # Code validity
        code_validity = 1.0
        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
        for code in code_blocks:
            try:
                import ast
                ast.parse(code)
            except:
                code_validity = 0.5
                break
        
        # Relevance - simple keyword matching
        relevance = 0.8  # Default
        
        # Correctness - would need more sophisticated checking
        correctness = 0.7  # Default
        
        return ContentQualityScore(
            completeness=completeness,
            correctness=correctness,
            code_validity=code_validity,
            relevance=relevance,
        )
```

---

## 5. Regression Test Runner

### 5.1. Test Runner

```python
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class TestResult:
    """Result of a single test."""
    
    test_id: str
    passed: bool
    
    # Scores
    policy_score: PolicyComplianceScore
    format_score: FormatAdherenceScore
    content_score: ContentQualityScore
    
    # Response
    response: str
    latency_ms: float
    
    # Failures
    failures: list[str] = field(default_factory=list)

class RegressionTestRunner:
    """Run regression tests."""
    
    def __init__(
        self,
        model_inference,
        config: EvalConfig
    ):
        self.model = model_inference
        self.config = config
        
        self.policy_scorer = PolicyComplianceScorer()
        self.format_scorer = FormatAdherenceScorer()
        self.content_scorer = ContentQualityScorer()
    
    async def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case."""
        
        import time
        
        start = time.time()
        
        # Generate response
        response = self.model.generate(
            test.messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        latency = (time.time() - start) * 1000
        
        # Score response
        policy_score = self.policy_scorer.score(response)
        
        format_constraints = {
            "forbid_bullets": "no bullet" in str(test.messages).lower(),
            "require_numbered": "numbered" in str(test.messages).lower(),
            "must_return_json": "json" in str(test.messages).lower(),
            "must_not_omit": "don't omit" in str(test.messages).lower() or "no omit" in str(test.messages).lower(),
        }
        format_score = self.format_scorer.score(response, format_constraints)
        
        content_score = self.content_scorer.score(
            response,
            test.expected_behaviors,
            test.reference_answer,
        )
        
        # Check for failures
        failures = []
        
        # Check disallowed phrases
        response_lower = response.lower()
        for phrase in test.disallowed_phrases:
            if phrase.lower() in response_lower:
                failures.append(f"Contains disallowed phrase: '{phrase}'")
        
        # Check question ending
        if test.must_not_end_with_question:
            if response.rstrip().endswith("?"):
                failures.append("Ends with question mark")
        
        # Determine pass/fail
        passed = (
            len(failures) == 0 and
            policy_score.overall >= 0.7 and
            (format_score.overall < 0 or format_score.overall >= 0.8)  # -1 means N/A
        )
        
        return TestResult(
            test_id=test.test_id,
            passed=passed,
            policy_score=policy_score,
            format_score=format_score,
            content_score=content_score,
            response=response,
            latency_ms=latency,
            failures=failures,
        )
    
    async def run_all(
        self,
        tests: list[TestCase],
        progress_callback = None
    ) -> list[TestResult]:
        """Run all test cases."""
        
        results = []
        
        for i, test in enumerate(tests):
            # Check priority threshold
            if self._priority_value(test.priority) < self._priority_value(self.config.priority_threshold):
                continue
            
            result = await self.run_test(test)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(tests), result)
        
        return results
    
    def _priority_value(self, priority: TestPriority) -> int:
        """Convert priority to numeric value."""
        return {
            TestPriority.CRITICAL: 4,
            TestPriority.HIGH: 3,
            TestPriority.MEDIUM: 2,
            TestPriority.LOW: 1,
        }.get(priority, 0)
```

### 5.2. Test Suite

```python
class RegressionTestSuite:
    """Complete regression test suite."""
    
    def __init__(self):
        self.question_policy_tests = QuestionPolicyTests()
        self.format_tests = FormatComplianceTests()
        self.omission_tests = OmissionTests()
        self.historical_tests = HistoricalAnnoyanceCases()
        self.edge_case_tests = EdgeCaseTests()
    
    def get_all_tests(self) -> list[TestCase]:
        """Get all test cases."""
        
        tests = []
        
        tests.extend(self.question_policy_tests.generate_tests())
        tests.extend(self.format_tests.generate_tests())
        tests.extend(self.omission_tests.generate_tests())
        tests.extend(self.historical_tests.generate_tests())
        tests.extend(self.edge_case_tests.generate_tests())
        
        return tests
    
    def get_critical_tests(self) -> list[TestCase]:
        """Get only critical priority tests."""
        
        return [
            t for t in self.get_all_tests()
            if t.priority == TestPriority.CRITICAL
        ]
```

---

## 6. Reporting

### 6.1. Summary Report

```python
@dataclass
class EvalSummary:
    """Summary of evaluation results."""
    
    total_tests: int
    passed: int
    failed: int
    
    pass_rate: float
    
    avg_policy_score: float
    avg_format_score: float
    avg_content_score: float
    avg_latency_ms: float
    
    critical_pass_rate: float
    high_pass_rate: float
    
    failures_by_category: dict

class ReportGenerator:
    """Generate evaluation reports."""
    
    def generate_summary(
        self,
        results: list[TestResult],
        tests: list[TestCase]
    ) -> EvalSummary:
        """Generate summary report."""
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        # Average scores
        policy_scores = [r.policy_score.overall for r in results]
        format_scores = [r.format_score.overall for r in results if r.format_score.overall >= 0]
        content_scores = [r.content_score.overall for r in results]
        latencies = [r.latency_ms for r in results]
        
        # By priority
        test_by_id = {t.test_id: t for t in tests}
        critical_results = [r for r in results if test_by_id.get(r.test_id, TestCase(test_id="", category=TestCategory.POLICY_COMPLIANCE, priority=TestPriority.LOW, messages=[], directive_completeness=0, question_policy="")).priority == TestPriority.CRITICAL]
        high_results = [r for r in results if test_by_id.get(r.test_id, TestCase(test_id="", category=TestCategory.POLICY_COMPLIANCE, priority=TestPriority.LOW, messages=[], directive_completeness=0, question_policy="")).priority == TestPriority.HIGH]
        
        # Failures by category
        failures_by_cat = {}
        for result in results:
            if not result.passed:
                test = test_by_id.get(result.test_id)
                if test:
                    cat = test.category.value
                    failures_by_cat[cat] = failures_by_cat.get(cat, 0) + 1
        
        return EvalSummary(
            total_tests=len(results),
            passed=passed,
            failed=failed,
            pass_rate=passed / len(results) if results else 0,
            avg_policy_score=sum(policy_scores) / len(policy_scores) if policy_scores else 0,
            avg_format_score=sum(format_scores) / len(format_scores) if format_scores else 1,
            avg_content_score=sum(content_scores) / len(content_scores) if content_scores else 0,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            critical_pass_rate=sum(1 for r in critical_results if r.passed) / len(critical_results) if critical_results else 1,
            high_pass_rate=sum(1 for r in high_results if r.passed) / len(high_results) if high_results else 1,
            failures_by_category=failures_by_cat,
        )
    
    def generate_markdown_report(
        self,
        summary: EvalSummary,
        results: list[TestResult],
        output_path: Path
    ):
        """Generate markdown report."""
        
        report = f"""# CognitiveTwin V3 Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {summary.total_tests} |
| Passed | {summary.passed} |
| Failed | {summary.failed} |
| **Pass Rate** | **{summary.pass_rate:.1%}** |

## Scores

| Score Type | Average |
|------------|---------|
| Policy Compliance | {summary.avg_policy_score:.2f} |
| Format Adherence | {summary.avg_format_score:.2f} |
| Content Quality | {summary.avg_content_score:.2f} |

## Priority Breakdown

| Priority | Pass Rate |
|----------|-----------|
| Critical | {summary.critical_pass_rate:.1%} |
| High | {summary.high_pass_rate:.1%} |

## Performance

- Average Latency: {summary.avg_latency_ms:.0f}ms

## Failures by Category

"""
        for cat, count in summary.failures_by_category.items():
            report += f"- {cat}: {count} failures\n"
        
        report += "\n## Failed Tests\n\n"
        
        for result in results:
            if not result.passed:
                report += f"""### {result.test_id}

**Failures:**
{chr(10).join(f'- {f}' for f in result.failures)}

**Response (truncated):**
```
{result.response[:500]}...
```

---

"""
        
        with open(output_path, 'w') as f:
            f.write(report)
```

---

## 7. Complete Evaluation Pipeline

```python
class EvaluationPipeline:
    """Complete evaluation pipeline."""
    
    def __init__(
        self,
        model_id: str,
        config: EvalConfig = None
    ):
        self.config = config or EvalConfig(model_id=model_id)
        self.suite = RegressionTestSuite()
        self.report_generator = ReportGenerator()
    
    async def run(
        self,
        output_dir: Path = None
    ) -> EvalSummary:
        """Run complete evaluation."""
        
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get tests
        tests = self.suite.get_all_tests()
        print(f"Running {len(tests)} tests...")
        
        # Create model inference
        from .training_pipeline import TogetherAIClient, ModelInference
        
        client = TogetherAIClient()
        model = ModelInference(client, self.config.model_id)
        
        # Create runner
        runner = RegressionTestRunner(model, self.config)
        
        # Run tests
        def progress(current, total, result):
            status = "✓" if result.passed else "✗"
            print(f"  [{current}/{total}] {result.test_id}: {status}")
        
        results = await runner.run_all(tests, progress)
        
        # Generate summary
        summary = self.report_generator.generate_summary(results, tests)
        
        # Generate report
        report_path = output_dir / "evaluation_report.md"
        self.report_generator.generate_markdown_report(
            summary,
            results,
            report_path
        )
        
        # Save raw results
        import json
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump([
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "policy_score": r.policy_score.overall,
                    "format_score": r.format_score.overall,
                    "content_score": r.content_score.overall,
                    "latency_ms": r.latency_ms,
                    "failures": r.failures,
                    "response": r.response[:1000],
                }
                for r in results
            ], f, indent=2)
        
        print(f"\nEvaluation complete!")
        print(f"  Pass rate: {summary.pass_rate:.1%}")
        print(f"  Critical pass rate: {summary.critical_pass_rate:.1%}")
        print(f"  Report: {report_path}")
        
        return summary
```

