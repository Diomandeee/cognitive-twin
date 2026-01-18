"""
Prompt Templates for Repo Worm.

Contains all prompt templates for GPT 5.2 Codex code generation tasks.
"""

from typing import Union
from .task_types import (
    ImplementationTask,
    CompletionTask,
    RefactoringTask,
    TestTask,
    Task,
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

CODEX_SYSTEM_PROMPT = """You are a code generation assistant for CognitiveTwin V3.

RULES:
1. Generate complete, working code that compiles
2. Follow existing patterns and conventions in the codebase
3. Do NOT ask clarifying questions - make reasonable assumptions
4. State assumptions briefly at the start if needed
5. Include proper error handling and edge cases
6. Match the existing code style exactly

OUTPUT FORMAT:
- Provide code in markdown code blocks
- Include brief comments explaining complex logic
- For diffs, use unified diff format

ASSUMPTION PROTOCOL:
If you need to assume something:
- State it in a comment: # Assumption: ...
- Then proceed with implementation
- NO questions
"""


# =============================================================================
# IMPLEMENTATION PROMPTS
# =============================================================================

def format_implement_interface_prompt(task: ImplementationTask) -> str:
    """Format prompt for interface implementation."""
    usage_section = ""
    if task.usage_examples:
        examples = '\n\n'.join(f'```python\n{ex}\n```' for ex in task.usage_examples[:3])
        usage_section = f"\nUSAGE EXAMPLES:\n{examples}"
    
    related_section = ""
    if task.related_implementations:
        related = '\n\n'.join(f'```python\n{impl}\n```' for impl in task.related_implementations[:2])
        related_section = f"\nRELATED IMPLEMENTATIONS:\n{related}"
    
    return f"""Implement the following function according to its interface:

SIGNATURE:
```python
{task.target_signature}
```

INTERFACE/PROTOCOL:
```python
{task.interface_definition}
```
{usage_section}
{related_section}

CONSTRAINTS:
- Must compile without errors
- Must match existing code patterns
- Must not introduce new dependencies
- Do NOT ask clarifying questions - make reasonable assumptions

Provide the complete implementation:"""


def format_complete_stub_prompt(task: ImplementationTask) -> str:
    """Format prompt for stub completion."""
    related_section = ""
    if task.related_implementations:
        related = '\n\n'.join(f'```python\n{impl}\n```' for impl in task.related_implementations[:2])
        related_section = f"\nRELATED IMPLEMENTATIONS FOR REFERENCE:\n{related}"
    
    return f"""Complete this function stub:

FILE: {task.target_file}
LINE: {task.target_line}

```python
{task.target_signature}
    pass  # TODO: Implement
```
{related_section}

IMPORTS AVAILABLE:
```python
{task.file_imports}
```

CONSTRAINTS:
- Follow the patterns shown in related implementations
- Make reasonable assumptions for any unknowns
- Do NOT ask questions - just implement

Provide the complete function body:"""


def format_add_method_prompt(task: ImplementationTask) -> str:
    """Format prompt for adding missing method."""
    usage_section = ""
    if task.usage_examples:
        examples = '\n\n'.join(f'```python\n{ex}\n```' for ex in task.usage_examples[:3])
        usage_section = f"\nUSAGE:\n{examples}"
    
    return f"""Add the missing method to this class:

CLASS CONTEXT:
```python
{task.interface_definition}
```

MISSING METHOD: {task.target_signature}

The method is referenced but not implemented. Based on the class context
and how the method is used, implement it.
{usage_section}

CONSTRAINTS:
- Must integrate with existing class methods
- Follow existing code style
- Do NOT ask for clarification

Provide the method implementation:"""


# =============================================================================
# COMPLETION PROMPTS
# =============================================================================

def format_complete_todo_prompt(task: CompletionTask) -> str:
    """Format prompt for TODO completion."""
    return f"""Complete this TODO:

FILE: {task.file}
LINE: {task.line}
TODO: {task.todo_text}

SURROUNDING CODE:
```python
{task.surrounding_code}
```

IMPORTS AVAILABLE:
```python
{task.file_imports}
```

CONSTRAINTS:
- Complete the TODO as specified
- Preserve existing behavior
- Follow the existing code style
- Do NOT ask questions - implement based on the TODO description

Provide the implementation that replaces the TODO:"""


def format_finish_partial_prompt(task: CompletionTask) -> str:
    """Format prompt for finishing partial implementation."""
    return f"""Finish this partial implementation:

```python
{task.surrounding_code}
```

The code above is incomplete. Based on the function signature and 
existing logic, complete the implementation.

FUNCTION: {task.function_signature}

CONSTRAINTS:
- Complete all unfinished logic
- Handle edge cases appropriately
- Follow existing patterns
- Do NOT ask for clarification

Provide the completed code:"""


# =============================================================================
# REFACTORING PROMPTS
# =============================================================================

def format_refactor_pattern_prompt(task: RefactoringTask) -> str:
    """Format prompt for pattern-matching refactor."""
    return f"""Refactor this code to match the pattern used elsewhere:

TARGET CODE (to refactor):
```python
{task.target_code}
```

PATTERN TO MATCH (from {task.pattern_file}):
```python
{task.pattern_example}
```

REFACTORING GOAL:
Make the target code follow the same patterns and conventions as the example.

CONSTRAINTS:
- Preserve the original behavior
- Match the style/patterns exactly
- Code must compile
- Do NOT ask questions

Provide the refactored code:"""


def format_extract_module_prompt(task: RefactoringTask) -> str:
    """Format prompt for module extraction."""
    return f"""Extract this code into a separate module:

CODE TO EXTRACT:
```python
{task.target_code}
```

CURRENT FILE: {task.target_file}

Create a new module with this code, and update the original file 
to import from it.

CONSTRAINTS:
- New module should be self-contained
- Original file should import and use the new module
- All tests should still pass
- Do NOT ask for module name - choose an appropriate one

Provide:
1. The new module code
2. The updated import statement for the original file"""


# =============================================================================
# TEST PROMPTS
# =============================================================================

def format_write_tests_prompt(task: TestTask) -> str:
    """Format prompt for test writing."""
    existing_section = ""
    if task.existing_tests:
        tests = '\n\n'.join(f'```python\n{t}\n```' for t in task.existing_tests[:2])
        existing_section = f"\nEXISTING TESTS FOR REFERENCE:\n{tests}"
    
    return f"""Write comprehensive tests for this function:

FUNCTION:
```python
{task.function_signature}
{task.function_body}
```
{existing_section}

REQUIREMENTS:
- Test normal operation
- Test edge cases
- Test error conditions
- Follow existing test patterns

CONSTRAINTS:
- Tests must be runnable with pytest
- Use existing fixtures if applicable
- Do NOT ask what to test - cover all reasonable cases

Provide the test code:"""


def format_edge_case_prompt(task: TestTask) -> str:
    """Format prompt for edge case coverage."""
    existing_section = ""
    if task.existing_tests:
        tests = '\n\n'.join(f'```python\n{t}\n```' for t in task.existing_tests)
        existing_section = f"\nEXISTING TESTS:\n{tests}"
    
    return f"""Add edge case tests for this function:

FUNCTION:
```python
{task.function_signature}
{task.function_body}
```
{existing_section}

The existing tests cover basic cases. Add tests for:
- Empty inputs
- Boundary values
- Invalid inputs
- Concurrent access (if applicable)
- Resource exhaustion (if applicable)

CONSTRAINTS:
- Focus on edge cases not already covered
- Follow existing test style
- Do NOT ask which cases to add

Provide the additional test code:"""


# =============================================================================
# PROMPT DISPATCHER
# =============================================================================

PROMPT_FORMATTERS = {
    # Implementation
    "impl_implement_interface": format_implement_interface_prompt,
    "impl_complete_stub": format_complete_stub_prompt,
    "impl_add_method": format_add_method_prompt,
    # Completion
    "comp_complete_todo": format_complete_todo_prompt,
    "comp_finish_partial": format_finish_partial_prompt,
    # Refactoring
    "refac_refactor_pattern": format_refactor_pattern_prompt,
    "refac_extract_module": format_extract_module_prompt,
    # Test
    "test_write_tests": format_write_tests_prompt,
    "test_edge_cases": format_edge_case_prompt,
}


def format_prompt(task: Task) -> str:
    """Format the appropriate prompt for a task."""
    key = task.get_prompt_key()
    formatter = PROMPT_FORMATTERS.get(key)
    
    if formatter is None:
        raise ValueError(f"Unknown prompt key: {key}")
    
    return formatter(task)


def prepare_context_window(
    task: Task,
    imports: str = "",
    interface: str = "",
    examples: list[str] = None,
    max_context_tokens: int = 16000,
) -> str:
    """Prepare context window for Codex."""
    context_parts = []
    
    # Add file header
    if hasattr(task, 'target_file') and task.target_file:
        context_parts.append(f"# File: {task.target_file}")
    elif hasattr(task, 'file') and task.file:
        context_parts.append(f"# File: {task.file}")
    
    # Add imports
    if imports:
        context_parts.append(imports)
    elif hasattr(task, 'file_imports') and task.file_imports:
        context_parts.append(task.file_imports)
    
    # Add interface
    if interface:
        context_parts.append(f"\n# Interface/Protocol:\n{interface}")
    elif hasattr(task, 'interface_definition') and task.interface_definition:
        context_parts.append(f"\n# Interface/Protocol:\n{task.interface_definition}")
    
    # Add examples
    if examples:
        context_parts.append("\n# Usage examples:")
        for ex in examples[:3]:
            context_parts.append(ex)
    elif hasattr(task, 'usage_examples') and task.usage_examples:
        context_parts.append("\n# Usage examples:")
        for ex in task.usage_examples[:3]:
            context_parts.append(ex)
    
    return '\n'.join(context_parts)

