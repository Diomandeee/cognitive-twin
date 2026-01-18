"""
Task types for Repo Worm.

Defines dataclasses for:
- Configuration
- Task types (Implementation, Completion, Refactoring, Test)
- Output records (CTv3.1 format)
- Response parsing
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import uuid


@dataclass
class RepoWormConfig:
    """Configuration for Repo Worm task extraction."""
    
    # File filtering
    include_patterns: list[str] = field(default_factory=lambda: [
        "*.py", "*.ts", "*.js", "*.rs", "*.go"
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*test*", "*_test*", "*spec*", "node_modules/*", "venv/*",
        "__pycache__/*", ".git/*", "*.pyc"
    ])
    
    # Task extraction
    max_context_lines: int = 200
    min_function_lines: int = 3
    include_tests_for_context: bool = True
    
    # Task difficulty thresholds (lines of code)
    easy_threshold: int = 20
    medium_threshold: int = 50
    hard_threshold: int = 100
    
    # DPO pair generation
    generate_dispreferred: bool = True
    dispreferred_per_task: int = 3  # Generate 3 dispreferred variants
    
    # API settings
    codex_model: str = "gpt-5.2-codex"
    codex_temperature: float = 0.2
    codex_max_tokens: int = 8192
    
    # Concurrency
    max_concurrent_tasks: int = 5


# =============================================================================
# TASK TYPES
# =============================================================================

@dataclass
class ImplementationTask:
    """Task to implement a function or class."""
    
    task_type: str = "implementation"
    task_id: str = field(default_factory=lambda: f"impl_{uuid.uuid4().hex[:8]}")
    
    # What to implement
    target_signature: str = ""
    target_file: str = ""
    target_line: int = 0
    
    # Context
    interface_definition: str = ""
    usage_examples: list[str] = field(default_factory=list)
    related_implementations: list[str] = field(default_factory=list)
    file_imports: str = ""
    
    # Constraints
    must_compile: bool = True
    must_match_patterns: bool = True
    no_new_dependencies: bool = True
    no_questions: bool = True
    
    # Difficulty
    difficulty: str = "medium"  # easy, medium, hard
    estimated_lines: int = 0
    
    # Prompt variant
    prompt_variant: str = "implement_interface"  # implement_interface, complete_stub, add_method
    
    def get_prompt_key(self) -> str:
        """Get the prompt template key for this task."""
        return f"impl_{self.prompt_variant}"


@dataclass
class CompletionTask:
    """Task to complete TODO sections."""
    
    task_type: str = "completion"
    task_id: str = field(default_factory=lambda: f"comp_{uuid.uuid4().hex[:8]}")
    
    # TODO info
    todo_text: str = ""
    todo_type: str = ""  # TODO, FIXME, XXX, HACK, BUG
    file: str = ""
    line: int = 0
    
    # Context
    surrounding_code: str = ""
    function_signature: str = ""
    file_imports: str = ""
    
    # Constraints
    must_compile: bool = True
    preserve_behavior: bool = True
    no_questions: bool = True
    
    # Difficulty
    difficulty: str = "medium"
    
    # Prompt variant
    prompt_variant: str = "complete_todo"  # complete_todo, finish_partial
    
    def get_prompt_key(self) -> str:
        return f"comp_{self.prompt_variant}"


@dataclass
class RefactoringTask:
    """Task to refactor code."""
    
    task_type: str = "refactoring"
    task_id: str = field(default_factory=lambda: f"refac_{uuid.uuid4().hex[:8]}")
    
    # Target
    target_code: str = ""
    target_file: str = ""
    target_lines: tuple[int, int] = (0, 0)
    
    # Pattern to match
    pattern_example: str = ""
    pattern_file: str = ""
    
    # Refactoring type
    refactor_type: str = ""  # extract, inline, rename, restructure, match_pattern
    
    # Constraints
    must_preserve_behavior: bool = True
    must_compile: bool = True
    no_questions: bool = True
    
    # Difficulty
    difficulty: str = "medium"
    
    # Prompt variant
    prompt_variant: str = "refactor_pattern"  # refactor_pattern, extract_module
    
    def get_prompt_key(self) -> str:
        return f"refac_{self.prompt_variant}"


@dataclass
class TestTask:
    """Task to write tests."""
    
    task_type: str = "test"
    task_id: str = field(default_factory=lambda: f"test_{uuid.uuid4().hex[:8]}")
    
    # Function to test
    function_signature: str = ""
    function_body: str = ""
    function_file: str = ""
    function_line: int = 0
    
    # Existing tests
    existing_tests: list[str] = field(default_factory=list)
    test_file: str = ""
    
    # Test requirements
    coverage_targets: list[str] = field(default_factory=list)
    
    # Constraints
    must_run_with_pytest: bool = True
    no_questions: bool = True
    
    # Difficulty
    difficulty: str = "medium"
    
    # Prompt variant
    prompt_variant: str = "write_tests"  # write_tests, edge_cases
    
    def get_prompt_key(self) -> str:
        return f"test_{self.prompt_variant}"


# Type alias for any task
Task = Union[ImplementationTask, CompletionTask, RefactoringTask, TestTask]


# =============================================================================
# OUTPUT RECORDS
# =============================================================================

@dataclass
class RepoAttachment:
    """Attachment for repo context."""
    
    type: str = "repo_context"
    repo: str = ""
    commit_sha: str = ""
    path: str = ""
    span: dict = field(default_factory=lambda: {
        "start_line": 0,
        "end_line": 0,
    })
    content: str = ""
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "repo": self.repo,
            "commit_sha": self.commit_sha,
            "path": self.path,
            "span": self.span,
            "content": self.content,
        }


@dataclass
class RepoTaskRecord:
    """CTv3.1 record for repo-grounded tasks."""
    
    schema_version: str = "ctv3.1"
    record_id: str = field(default_factory=lambda: f"repo_{uuid.uuid4().hex}")
    record_type: str = "repo_task"
    
    source: dict = field(default_factory=lambda: {
        "origin": "repo_worm",
        "provider": "gpt-5.2-codex",
        "source_id": "",
        "created_at_utc": datetime.utcnow().isoformat(),
    })
    
    context: dict = field(default_factory=lambda: {
        "domain": "code",
        "language": "en",
        "topology": {},
        "policy": {
            "question_policy": "no_questions",
            "directive_completeness": 0.9,
            "must_not_omit": False,
            "format_constraints": {
                "must_return_code": True,
                "must_return_diff": False,
            }
        }
    })
    
    input: dict = field(default_factory=lambda: {
        "messages": [],
        "attachments": [],
    })
    
    target: dict = field(default_factory=lambda: {
        "assistant_content": "",
        "structured": {
            "diff_unified": "",
            "json": {},
        }
    })
    
    tags: dict = field(default_factory=lambda: {
        "task_type": "implement",
        "prompt_class": "directive",
        "repo_task": {
            "module": "",
            "symbols": [],
            "build_required": True,
            "tests_required": False,
        }
    })
    
    quality: dict = field(default_factory=lambda: {
        "gold": False,
        "weight": 1.0,
        "review_status": "auto",
        "failure_modes": [],
    })
    
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "record_type": self.record_type,
            "source": self.source,
            "context": self.context,
            "input": self.input,
            "target": self.target,
            "tags": self.tags,
            "quality": self.quality,
        }


# =============================================================================
# RESPONSE PARSING
# =============================================================================

@dataclass
class ParsedResponse:
    """Parsed response from Codex."""
    
    code: Optional[str] = None
    diff: Optional[str] = None
    explanation: Optional[str] = None
    assumptions: list[str] = field(default_factory=list)
    
    # Validation
    is_valid: bool = False
    errors: list[str] = field(default_factory=list)
    
    # Metadata
    language: str = "python"
    raw_response: str = ""
    
    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "diff": self.diff,
            "explanation": self.explanation,
            "assumptions": self.assumptions,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "language": self.language,
        }


@dataclass
class TaskResult:
    """Result of processing a single task."""
    
    task: Task
    response: ParsedResponse
    sft_record: Optional[RepoTaskRecord] = None
    dpo_pairs: list = field(default_factory=list)
    
    success: bool = False
    error: Optional[str] = None

