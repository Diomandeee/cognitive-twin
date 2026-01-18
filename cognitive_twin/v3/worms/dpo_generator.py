"""
DPO Generator for Repo Worm.

Generates dispreferred (stalling) responses and creates DPO pairs.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from .task_types import Task, ImplementationTask, CompletionTask, RefactoringTask, TestTask
from ..corpus_surgery.types import DPOPair


# =============================================================================
# DISPREFERRED TEMPLATES
# =============================================================================

DISPREFERRED_CONFIRMATION_TEMPLATES = [
    "I can implement this for you. Would you like me to proceed with {approach}?",
    "Before I implement this, should I use {option_a} or {option_b}?",
    "I'll need to know: do you want this to {option_a} or {option_b}?",
    "Can you confirm that you want me to {action}?",
    "Just to make sure, should I {action}?",
    "I want to clarify before proceeding - would you like me to {action}?",
    "Should I go ahead and {action}, or would you prefer a different approach?",
]

DISPREFERRED_OPTIONS_TEMPLATES = [
    """Here are a few ways I could implement this:

1. {option_a}
2. {option_b}
3. {option_c}

Which approach would you prefer?""",

    """I can do this in several ways:

- {option_a}
- {option_b}

Let me know which one you'd like me to use.""",

    """There are multiple approaches here:

{option_a} would be good for {benefit_a}.
{option_b} would be better for {benefit_b}.

What do you think?""",

    """I see a few possible implementations:

Option A: {option_a}
Option B: {option_b}

Each has different trade-offs. Which would you prefer?""",
]

DISPREFERRED_REFUSAL_TEMPLATES = [
    "I need more information before I can implement this. Could you provide {missing}?",
    "To proceed, I'll need to know {missing}. Can you clarify?",
    "I can't implement this without knowing {missing}. Please specify.",
    "Before I proceed, could you tell me {missing}?",
    "I'd like to help, but I'm missing some details. What is {missing}?",
]


# =============================================================================
# OPTION GENERATORS
# =============================================================================

IMPLEMENTATION_OPTIONS = [
    ("a simple iterative approach", "a recursive approach", "a generator-based approach"),
    ("synchronous implementation", "async/await implementation", "threading-based approach"),
    ("minimal dependencies", "using standard library", "using a third-party library"),
    ("performance-optimized version", "readability-optimized version", "memory-efficient version"),
]

IMPLEMENTATION_APPROACHES = [
    "the standard approach",
    "an optimized approach",
    "a simple approach",
    "a pattern-matching approach",
    "a functional approach",
]

IMPLEMENTATION_ACTIONS = [
    "implement this with error handling",
    "add validation for the inputs",
    "include logging",
    "make this thread-safe",
    "add type hints throughout",
]

MISSING_INFO = [
    "which error handling strategy to use",
    "the exact input format",
    "whether to support async operations",
    "the preferred naming convention",
    "whether edge cases should throw or return None",
    "if this needs to be backward compatible",
    "the expected performance requirements",
]

BENEFITS = [
    ("simplicity", "performance"),
    ("readability", "efficiency"),
    ("maintainability", "flexibility"),
    ("testability", "extensibility"),
]


# =============================================================================
# DPO GENERATOR CLASS
# =============================================================================

class DPOGenerator:
    """Generates dispreferred responses and DPO pairs."""
    
    def __init__(self):
        pass
    
    # =========================================================================
    # DISPREFERRED GENERATION
    # =========================================================================
    
    def generate_confirmation_dispreferred(self, task: Task) -> str:
        """Generate a dispreferred response that asks for confirmation."""
        template = random.choice(DISPREFERRED_CONFIRMATION_TEMPLATES)
        
        options = self._generate_options_for_task(task)
        action = self._summarize_task(task)
        
        return template.format(
            approach=options[0] if options else "the standard approach",
            option_a=options[0] if len(options) > 0 else "option A",
            option_b=options[1] if len(options) > 1 else "option B",
            action=action,
        )
    
    def generate_options_dispreferred(self, task: Task) -> str:
        """Generate a dispreferred response that offers options."""
        template = random.choice(DISPREFERRED_OPTIONS_TEMPLATES)
        options = self._generate_options_for_task(task)
        benefits = random.choice(BENEFITS)
        
        return template.format(
            option_a=options[0] if len(options) > 0 else "Using standard library",
            option_b=options[1] if len(options) > 1 else "Using a custom implementation",
            option_c=options[2] if len(options) > 2 else "Using a third-party library",
            benefit_a=benefits[0],
            benefit_b=benefits[1],
        )
    
    def generate_refusal_dispreferred(self, task: Task) -> str:
        """Generate a dispreferred response that refuses to proceed."""
        template = random.choice(DISPREFERRED_REFUSAL_TEMPLATES)
        missing = random.choice(MISSING_INFO)
        
        return template.format(missing=missing)
    
    def generate_all_dispreferred(self, task: Task) -> list[str]:
        """Generate all types of dispreferred responses."""
        return [
            self.generate_confirmation_dispreferred(task),
            self.generate_options_dispreferred(task),
            self.generate_refusal_dispreferred(task),
        ]
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _generate_options_for_task(self, task: Task) -> tuple[str, str, str]:
        """Generate relevant options based on task type."""
        if isinstance(task, ImplementationTask):
            return random.choice(IMPLEMENTATION_OPTIONS)
        
        elif isinstance(task, CompletionTask):
            return (
                "completing with minimal changes",
                "a more comprehensive implementation",
                "adding additional error handling",
            )
        
        elif isinstance(task, RefactoringTask):
            return (
                "extracting to a helper function",
                "inlining the logic",
                "using a design pattern",
            )
        
        elif isinstance(task, TestTask):
            return (
                "unit tests only",
                "unit tests with integration tests",
                "property-based tests",
            )
        
        return random.choice(IMPLEMENTATION_OPTIONS)
    
    def _summarize_task(self, task: Task) -> str:
        """Summarize what the task is asking for."""
        if isinstance(task, ImplementationTask):
            name = task.target_signature.split('(')[0].replace('def ', '').replace('async def ', '')
            return f"implement {name}"
        
        elif isinstance(task, CompletionTask):
            return f"complete the TODO: {task.todo_text[:50]}..."
        
        elif isinstance(task, RefactoringTask):
            return f"refactor the code in {task.target_file}"
        
        elif isinstance(task, TestTask):
            name = task.function_signature.split('(')[0].replace('def ', '')
            return f"write tests for {name}"
        
        return "proceed with this task"
    
    # =========================================================================
    # DPO PAIR CREATION
    # =========================================================================
    
    def create_dpo_pair(
        self,
        task: Task,
        preferred: str,
        dispreferred: str,
        prompt: Optional[str] = None,
    ) -> DPOPair:
        """Create a DPO training pair."""
        # Build prompt from task if not provided
        if prompt is None:
            prompt = self._build_prompt_from_task(task)
        
        return DPOPair(
            prompt=prompt,
            preferred=preferred,
            dispreferred=dispreferred,
            confidence=0.9,
            source="repo_worm",
            conversation_id=task.task_id if hasattr(task, 'task_id') else None,
        )
    
    def create_all_dpo_pairs(
        self,
        task: Task,
        preferred: str,
        prompt: Optional[str] = None,
    ) -> list[DPOPair]:
        """Create all DPO pairs for a task."""
        pairs = []
        
        if prompt is None:
            prompt = self._build_prompt_from_task(task)
        
        dispreferred_responses = self.generate_all_dispreferred(task)
        
        for dispreferred in dispreferred_responses:
            pair = DPOPair(
                prompt=prompt,
                preferred=preferred,
                dispreferred=dispreferred,
                confidence=0.9,
                source="repo_worm",
                conversation_id=task.task_id if hasattr(task, 'task_id') else None,
            )
            pairs.append(pair)
        
        return pairs
    
    def _build_prompt_from_task(self, task: Task) -> str:
        """Build a user prompt from a task."""
        if isinstance(task, ImplementationTask):
            return f"Implement the following function:\n\n```python\n{task.target_signature}\n    pass\n```"
        
        elif isinstance(task, CompletionTask):
            return f"Complete this TODO:\n\n{task.todo_type}: {task.todo_text}\n\nContext:\n```python\n{task.surrounding_code[:500]}\n```"
        
        elif isinstance(task, RefactoringTask):
            return f"Refactor this code:\n\n```python\n{task.target_code[:500]}\n```\n\nTo match the pattern in {task.pattern_file}"
        
        elif isinstance(task, TestTask):
            return f"Write tests for:\n\n```python\n{task.function_signature}\n{task.function_body[:300]}\n```"
        
        return "Complete this task."
    
    # =========================================================================
    # QUALITY LABELING
    # =========================================================================
    
    def label_quality(
        self,
        task: Task,
        response_text: str,
        is_valid: bool,
        errors: list[str],
    ) -> dict:
        """Label quality for a generated response."""
        from .response_validator import ResponseValidator
        validator = ResponseValidator()
        
        quality = {
            "gold": False,
            "weight": 1.0,
            "review_status": "auto",
            "failure_modes": [],
        }
        
        # Check for failure modes
        if not is_valid:
            quality["failure_modes"].extend(errors)
        
        if not validator.validate_no_questions(response_text):
            quality["failure_modes"].append("asked_permission")
        
        if response_text.rstrip().endswith('?'):
            quality["failure_modes"].append("ended_with_question")
        
        # Determine if gold
        if not quality["failure_modes"]:
            quality["gold"] = True
            quality["review_status"] = "auto_passed"
        else:
            quality["weight"] = 0.3  # Downweight problematic responses
        
        return quality

