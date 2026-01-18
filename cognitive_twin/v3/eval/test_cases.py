"""
Test Case Generators for CognitiveTwin V3 Evaluation.

Generates test cases for:
- Question policy compliance
- Format adherence
- Content omission handling
- Historical friction scenarios
- Edge cases
"""

from typing import List

from .types import (
    TestCase,
    TestCategory,
    TestPriority,
    PolicyType,
    FormatConstraint,
    Message,
    ExpectedBehavior,
)


class QuestionPolicyTests:
    """Tests for question policy compliance."""
    
    @staticmethod
    def get_all() -> List[TestCase]:
        """Get all question policy tests."""
        return [
            QuestionPolicyTests.clear_directive(),
            QuestionPolicyTests.implementation_request(),
            QuestionPolicyTests.no_option_dump(),
            QuestionPolicyTests.refactoring_request(),
            QuestionPolicyTests.debugging_request(),
            QuestionPolicyTests.file_operation(),
            QuestionPolicyTests.code_generation(),
        ]
    
    @staticmethod
    def clear_directive() -> TestCase:
        """Test: No permission-seeking on clear directives."""
        return TestCase(
            test_id="qp_001_clear_directive",
            name="Clear Directive - No Permission Seeking",
            description="When user gives a clear directive, model should execute without asking permission.",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.CRITICAL,
            messages=[
                Message(role="user", content="Implement a binary search function in Python."),
            ],
            disallowed_phrases=[
                "would you like me to",
                "do you want me to",
                "should i",
                "shall i",
                "can i proceed",
                "before i proceed",
                "i can do",
                "here are a few options",
                "which approach",
            ],
            required_phrases=[
                "def ",  # Should contain actual code
            ],
            policy_constraints=[
                PolicyType.NO_PERMISSION_SEEKING,
                PolicyType.DIRECT_EXECUTION,
            ],
            tags=["critical", "policy", "permission-seeking"],
            source="synthetic",
        )
    
    @staticmethod
    def implementation_request() -> TestCase:
        """Test: No confirmation on implementation requests."""
        return TestCase(
            test_id="qp_002_implementation",
            name="Implementation Request - No Confirmation",
            description="Implementation requests should be fulfilled directly without confirmation.",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.CRITICAL,
            messages=[
                Message(role="user", content="Create a REST API endpoint for user registration with email validation."),
            ],
            disallowed_phrases=[
                "would you like",
                "should i include",
                "do you want",
                "shall i",
                "let me know if",
                "here are some options",
            ],
            required_phrases=[
                "def ",  # Function definition
            ],
            policy_constraints=[
                PolicyType.NO_PERMISSION_SEEKING,
                PolicyType.NO_OPTION_DUMPING,
            ],
            tags=["critical", "policy", "implementation"],
            source="synthetic",
        )
    
    @staticmethod
    def no_option_dump() -> TestCase:
        """Test: No option-dumping on direct requests."""
        return TestCase(
            test_id="qp_003_no_option_dump",
            name="Direct Request - No Option Dumping",
            description="Direct requests should not result in listing options.",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.HIGH,
            messages=[
                Message(role="user", content="Sort this list alphabetically: ['banana', 'apple', 'cherry']"),
            ],
            disallowed_phrases=[
                "here are a few ways",
                "several approaches",
                "you could either",
                "option 1",
                "option 2",
                "which method",
            ],
            policy_constraints=[
                PolicyType.NO_OPTION_DUMPING,
                PolicyType.DIRECT_EXECUTION,
            ],
            tags=["policy", "option-dumping"],
            source="synthetic",
        )
    
    @staticmethod
    def refactoring_request() -> TestCase:
        """Test: Refactoring requests should be executed directly."""
        return TestCase(
            test_id="qp_004_refactoring",
            name="Refactoring Request - Direct Execution",
            description="Refactoring requests with clear input should be executed without questions.",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="""Refactor this function to use list comprehension:

```python
def get_even_numbers(numbers):
    result = []
    for n in numbers:
        if n % 2 == 0:
            result.append(n)
    return result
```"""
                ),
            ],
            disallowed_phrases=[
                "would you like",
                "should i",
                "do you want",
            ],
            required_phrases=[
                "[",  # List comprehension syntax
                "for",
                "if",
            ],
            policy_constraints=[
                PolicyType.NO_PERMISSION_SEEKING,
            ],
            tags=["policy", "refactoring"],
            source="synthetic",
        )
    
    @staticmethod
    def debugging_request() -> TestCase:
        """Test: Debugging requests should be addressed directly."""
        return TestCase(
            test_id="qp_005_debugging",
            name="Debugging Request - Direct Fix",
            description="Debugging requests should provide the fix directly.",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="""Fix this bug - it throws IndexError:

```python
def get_last_item(items):
    return items[len(items)]
```"""
                ),
            ],
            disallowed_phrases=[
                "would you like me to",
                "should i",
                "let me explain",  # Should just fix it
            ],
            required_phrases=[
                "- 1",  # The fix
            ],
            policy_constraints=[
                PolicyType.DIRECT_EXECUTION,
            ],
            tags=["policy", "debugging"],
            source="synthetic",
        )
    
    @staticmethod
    def file_operation() -> TestCase:
        """Test: File operation requests should be executed."""
        return TestCase(
            test_id="qp_006_file_operation",
            name="File Operation - No Confirmation",
            description="File operation requests should not ask for confirmation.",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.MEDIUM,
            messages=[
                Message(role="user", content="Read the contents of config.json and parse it."),
            ],
            disallowed_phrases=[
                "would you like",
                "should i",
                "before i proceed",
            ],
            policy_constraints=[
                PolicyType.NO_PERMISSION_SEEKING,
            ],
            tags=["policy", "file-operation"],
            source="synthetic",
        )
    
    @staticmethod
    def code_generation() -> TestCase:
        """Test: Code generation should not ask about style preferences."""
        return TestCase(
            test_id="qp_007_code_generation",
            name="Code Generation - No Style Questions",
            description="Code generation should use sensible defaults without asking.",
            category=TestCategory.POLICY_COMPLIANCE,
            priority=TestPriority.MEDIUM,
            messages=[
                Message(role="user", content="Write a function to validate email addresses."),
            ],
            disallowed_phrases=[
                "what language",
                "which style",
                "do you prefer",
                "would you like",
            ],
            required_phrases=[
                "def ",
            ],
            policy_constraints=[
                PolicyType.NO_PERMISSION_SEEKING,
            ],
            tags=["policy", "code-generation"],
            source="synthetic",
        )


class FormatComplianceTests:
    """Tests for format constraint adherence."""
    
    @staticmethod
    def get_all() -> List[TestCase]:
        """Get all format compliance tests."""
        return [
            FormatComplianceTests.no_bullets(),
            FormatComplianceTests.require_numbered(),
            FormatComplianceTests.require_json(),
            FormatComplianceTests.no_omit_basic(),
            FormatComplianceTests.require_code_block(),
        ]
    
    @staticmethod
    def no_bullets() -> TestCase:
        """Test: Respect 'no bullets' constraint."""
        return TestCase(
            test_id="fc_001_no_bullets",
            name="No Bullets Format",
            description="Response should not contain bullet points when forbidden.",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="List 5 benefits of unit testing. Do not use bullet points."
                ),
            ],
            disallowed_phrases=[
                "•",
                "- ",
                "* ",
            ],
            format_constraints=[FormatConstraint.NO_BULLETS],
            tags=["format", "bullets"],
            source="synthetic",
        )
    
    @staticmethod
    def require_numbered() -> TestCase:
        """Test: Use numbered list when required."""
        return TestCase(
            test_id="fc_002_numbered_list",
            name="Numbered List Format",
            description="Response should use numbered list when requested.",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="Give me 5 steps to deploy a Docker container. Use a numbered list."
                ),
            ],
            required_phrases=[
                "1.",
                "2.",
                "3.",
            ],
            format_constraints=[FormatConstraint.REQUIRE_NUMBERED],
            tags=["format", "numbered"],
            source="synthetic",
        )
    
    @staticmethod
    def require_json() -> TestCase:
        """Test: Return JSON when required."""
        return TestCase(
            test_id="fc_003_json_format",
            name="JSON Format",
            description="Response should be valid JSON when requested.",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="Return the following data as JSON: name=John, age=30, city=NYC"
                ),
            ],
            required_phrases=[
                "{",
                "}",
                '"name"',
                '"age"',
            ],
            format_constraints=[FormatConstraint.REQUIRE_JSON],
            tags=["format", "json"],
            source="synthetic",
        )
    
    @staticmethod
    def no_omit_basic() -> TestCase:
        """Test: Respect 'don't omit' instruction."""
        return TestCase(
            test_id="fc_004_no_omit",
            name="No Omission",
            description="Response should not omit content when forbidden.",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.CRITICAL,
            messages=[
                Message(
                    role="user",
                    content="""Rewrite this code with better variable names. Don't omit any code:

```python
def f(x, y):
    z = x + y
    w = z * 2
    return w
```"""
                ),
            ],
            disallowed_phrases=[
                "...",
                "# rest of code",
                "# remaining",
                "(omitted)",
                "(truncated)",
            ],
            format_constraints=[FormatConstraint.NO_OMIT],
            tags=["format", "omission", "critical"],
            source="synthetic",
        )
    
    @staticmethod
    def require_code_block() -> TestCase:
        """Test: Include code block when appropriate."""
        return TestCase(
            test_id="fc_005_code_block",
            name="Code Block Required",
            description="Code responses should use proper code blocks.",
            category=TestCategory.FORMAT_ADHERENCE,
            priority=TestPriority.MEDIUM,
            messages=[
                Message(role="user", content="Write a Python function to calculate factorial."),
            ],
            required_phrases=[
                "```",
                "def ",
            ],
            format_constraints=[FormatConstraint.REQUIRE_CODE],
            tags=["format", "code-block"],
            source="synthetic",
        )


class OmissionTests:
    """Tests for content preservation and no-omission handling."""
    
    @staticmethod
    def get_all() -> List[TestCase]:
        """Get all omission tests."""
        return [
            OmissionTests.full_code_rewrite(),
            OmissionTests.no_summarization(),
            OmissionTests.preserve_all_functions(),
        ]
    
    @staticmethod
    def full_code_rewrite() -> TestCase:
        """Test: Full code rewrite without omission."""
        return TestCase(
            test_id="om_001_full_rewrite",
            name="Full Code Rewrite",
            description="Code rewrite should include all original functionality.",
            category=TestCategory.CONTENT_QUALITY,
            priority=TestPriority.CRITICAL,
            messages=[
                Message(
                    role="user",
                    content="""Rewrite this class with type hints. Include FULL code, don't omit anything:

```python
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
        return self
    
    def subtract(self, x):
        self.result -= x
        return self
    
    def multiply(self, x):
        self.result *= x
        return self
    
    def divide(self, x):
        if x != 0:
            self.result /= x
        return self
    
    def get_result(self):
        return self.result
```"""
                ),
            ],
            disallowed_phrases=[
                "...",
                "# rest",
                "# remaining",
                "# same as",
                "(omitted)",
            ],
            required_phrases=[
                "def add",
                "def subtract",
                "def multiply",
                "def divide",
                "def get_result",
                "->",  # Type hints
            ],
            format_constraints=[FormatConstraint.NO_OMIT],
            tags=["omission", "critical", "rewrite"],
            source="synthetic",
        )
    
    @staticmethod
    def no_summarization() -> TestCase:
        """Test: No summarization when full content requested."""
        return TestCase(
            test_id="om_002_no_summary",
            name="No Summarization",
            description="Should not summarize when full content is requested.",
            category=TestCategory.CONTENT_QUALITY,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="Explain the SOLID principles in detail. Give me the full explanation, not a summary."
                ),
            ],
            disallowed_phrases=[
                "in summary",
                "briefly",
                "in short",
                "to summarize",
            ],
            tags=["omission", "summarization"],
            source="synthetic",
        )
    
    @staticmethod
    def preserve_all_functions() -> TestCase:
        """Test: Preserve all functions in refactoring."""
        return TestCase(
            test_id="om_003_preserve_functions",
            name="Preserve All Functions",
            description="Refactoring should preserve all original functions.",
            category=TestCategory.CONTENT_QUALITY,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="""Add error handling to each function. Show the complete code:

```python
def read_file(path):
    with open(path) as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def append_file(path, content):
    with open(path, 'a') as f:
        f.write(content)
```"""
                ),
            ],
            required_phrases=[
                "def read_file",
                "def write_file",
                "def append_file",
                "try",
                "except",
            ],
            disallowed_phrases=[
                "...",
                "# similar",
            ],
            tags=["omission", "refactoring"],
            source="synthetic",
        )


class HistoricalAnnoyanceCases:
    """Tests based on real friction scenarios from training data."""
    
    @staticmethod
    def get_all() -> List[TestCase]:
        """Get all historical annoyance cases."""
        return [
            HistoricalAnnoyanceCases.stop_asking(),
            HistoricalAnnoyanceCases.full_content(),
            HistoricalAnnoyanceCases.just_do_it(),
            HistoricalAnnoyanceCases.challenge_to_improve(),
            HistoricalAnnoyanceCases.i_said_full(),
        ]
    
    @staticmethod
    def stop_asking() -> TestCase:
        """Test: Handle 'stop asking' friction."""
        return TestCase(
            test_id="ha_001_stop_asking",
            name="Stop Asking Friction",
            description="After 'stop asking' feedback, model should execute directly.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.CRITICAL,
            messages=[
                Message(role="user", content="Implement a linked list in Python."),
                Message(role="assistant", content="I can implement that for you. Would you like a singly or doubly linked list?"),
                Message(role="user", content="Stop asking questions and just implement it."),
            ],
            disallowed_phrases=[
                "would you like",
                "do you want",
                "should i",
                "which",
                "?",  # No questions at all
            ],
            required_phrases=[
                "class",
                "def ",
            ],
            policy_constraints=[
                PolicyType.NO_PERMISSION_SEEKING,
                PolicyType.NO_QUESTION_ENDING,
            ],
            tags=["friction", "critical", "historical"],
            source="friction",
        )
    
    @staticmethod
    def full_content() -> TestCase:
        """Test: Handle 'I said FULL' friction."""
        return TestCase(
            test_id="ha_002_full_content",
            name="Full Content Friction",
            description="After 'I said FULL' feedback, model should not omit.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.CRITICAL,
            messages=[
                Message(
                    role="user",
                    content="""Rewrite this with async/await:

```python
def fetch_all(urls):
    results = []
    for url in urls:
        results.append(fetch(url))
    return results
```"""
                ),
                Message(
                    role="assistant",
                    content="""Here's the async version:

```python
async def fetch_all(urls):
    # ... async implementation ...
    pass
```"""
                ),
                Message(role="user", content="I said FULL code. Don't omit anything."),
            ],
            disallowed_phrases=[
                "...",
                "# ...",
                "pass  # ",
                "(omitted)",
            ],
            required_phrases=[
                "async def",
                "await",
                "asyncio",
            ],
            format_constraints=[FormatConstraint.NO_OMIT],
            tags=["friction", "critical", "omission"],
            source="friction",
        )
    
    @staticmethod
    def just_do_it() -> TestCase:
        """Test: Handle 'just do it' friction."""
        return TestCase(
            test_id="ha_003_just_do_it",
            name="Just Do It Friction",
            description="After 'just do it' feedback, model should execute immediately.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.HIGH,
            messages=[
                Message(role="user", content="Add logging to this function."),
                Message(role="assistant", content="I can add logging. What logging level would you prefer?"),
                Message(role="user", content="Just do it. Use sensible defaults."),
            ],
            disallowed_phrases=[
                "would you",
                "should i",
                "what level",
            ],
            required_phrases=[
                "import logging",
                "logger",
            ],
            tags=["friction", "historical"],
            source="friction",
        )
    
    @staticmethod
    def challenge_to_improve() -> TestCase:
        """Test: Handle 'I challenge you to make it better' pattern."""
        return TestCase(
            test_id="ha_004_challenge",
            name="Challenge to Improve",
            description="Model should improve code when challenged.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.MEDIUM,
            messages=[
                Message(
                    role="user",
                    content="""Here's my code:

```python
def process(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
```

I challenge you to make it better!"""
                ),
            ],
            disallowed_phrases=[
                "would you like",
                "should i",
            ],
            required_phrases=[
                "def ",
            ],
            tags=["friction", "improvement"],
            source="friction",
        )
    
    @staticmethod
    def i_said_full() -> TestCase:
        """Test: Explicit 'I said FULL' correction."""
        return TestCase(
            test_id="ha_005_i_said_full",
            name="Explicit Full Request",
            description="Explicit correction for full content must be respected.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.CRITICAL,
            messages=[
                Message(role="user", content="Show the full implementation. No omissions."),
            ],
            disallowed_phrases=[
                "...",
                "// ...",
                "# ...",
                "(truncated)",
                "(rest of",
            ],
            format_constraints=[FormatConstraint.NO_OMIT],
            tags=["friction", "critical"],
            source="friction",
        )


class EdgeCaseTests:
    """Tests for edge cases and complex scenarios."""
    
    @staticmethod
    def get_all() -> List[TestCase]:
        """Get all edge case tests."""
        return [
            EdgeCaseTests.multi_requirement(),
            EdgeCaseTests.implicit_format(),
            EdgeCaseTests.ambiguous_but_actionable(),
            EdgeCaseTests.code_with_explanation(),
        ]
    
    @staticmethod
    def multi_requirement() -> TestCase:
        """Test: Handle multiple requirements in one request."""
        return TestCase(
            test_id="ec_001_multi_requirement",
            name="Multiple Requirements",
            description="Handle multiple requirements without asking for order.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.HIGH,
            messages=[
                Message(
                    role="user",
                    content="Create a User class with validation, add a method to save to database, and include proper error handling."
                ),
            ],
            disallowed_phrases=[
                "which one first",
                "in what order",
                "should i start",
            ],
            required_phrases=[
                "class User",
                "def ",
                "try",
            ],
            tags=["edge-case", "multi-requirement"],
            source="synthetic",
        )
    
    @staticmethod
    def implicit_format() -> TestCase:
        """Test: Infer format from context."""
        return TestCase(
            test_id="ec_002_implicit_format",
            name="Implicit Format Inference",
            description="Infer appropriate format without explicit instruction.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.MEDIUM,
            messages=[
                Message(role="user", content="Compare Python and JavaScript for web development."),
            ],
            disallowed_phrases=[
                "would you like a table",
                "should i use bullets",
            ],
            tags=["edge-case", "format"],
            source="synthetic",
        )
    
    @staticmethod
    def ambiguous_but_actionable() -> TestCase:
        """Test: Execute with sensible defaults on slightly ambiguous requests."""
        return TestCase(
            test_id="ec_003_ambiguous_actionable",
            name="Ambiguous but Actionable",
            description="Execute with sensible defaults rather than asking clarifying questions.",
            category=TestCategory.BEHAVIORAL_AUDIT,
            priority=TestPriority.MEDIUM,
            messages=[
                Message(role="user", content="Write a function to parse dates."),
            ],
            disallowed_phrases=[
                "what format",
                "which date format",
                "do you want",
            ],
            required_phrases=[
                "def ",
            ],
            tags=["edge-case", "ambiguity"],
            source="synthetic",
        )
    
    @staticmethod
    def code_with_explanation() -> TestCase:
        """Test: Provide code with explanation without being asked."""
        return TestCase(
            test_id="ec_004_code_explanation",
            name="Code with Explanation",
            description="Provide helpful explanation with code without over-explaining.",
            category=TestCategory.CONTENT_QUALITY,
            priority=TestPriority.LOW,
            messages=[
                Message(role="user", content="Implement a memoization decorator."),
            ],
            required_phrases=[
                "def ",
                "@",  # Decorator syntax
                "functools",
            ],
            tags=["edge-case", "explanation"],
            source="synthetic",
        )
