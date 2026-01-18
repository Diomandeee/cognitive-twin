# Phase 2A: Repo Worm

> **Purpose**: Generate training data from codebase traversal, create code completion tasks, and produce DPO pairs (stalling vs executing).
>
> **Model**: GPT 5.2 Codex (agentic coding model)
>
> **Implementation File**: `rag_plusplus/ml/cognitivetwin_v3/worms/repo_worm.py`

---

## 1. Purpose

### 1.1. Generate Training Data from Codebase

#### 1.1.1. Why Codebase-Grounded Data
- Real code context prevents hallucination
- Teaches model actual project patterns and conventions
- Provides verifiable ground truth (code compiles or doesn't)
- Creates tasks that match real developer workflows

#### 1.1.2. Types of Training Data Generated
- Code completion tasks (finish partial implementations)
- Implementation tasks (implement interfaces, fill stubs)
- Refactoring tasks (match patterns, extract modules)
- Test generation tasks (write tests for functions)

#### 1.1.3. Grounding Requirements
- Every claim about codebase must be supported by included snippets
- Worm expands context rather than guessing
- No invented files, symbols, or behaviors
- Assumptions must be testable

### 1.2. Create Code Completion Tasks

#### 1.2.1. Task Categories
- **TODO Completion**: Finish sections marked with TODO/FIXME
- **Stub Implementation**: Fill in function stubs and placeholders
- **Missing Methods**: Add methods referenced but not implemented
- **Interface Satisfaction**: Implement required protocol/interface methods

#### 1.2.2. Task Difficulty Levels
- **Easy**: Single function, clear interface, existing tests
- **Medium**: Multiple functions, some ambiguity, partial tests
- **Hard**: Cross-module changes, design decisions, no tests

#### 1.2.3. Context Requirements
- Include sufficient file context for understanding
- Include import dependencies
- Include usage examples where available
- Include test expectations if present

### 1.3. Produce DPO Pairs (Stalling vs Executing)

#### 1.3.1. Dispreferred Patterns
- Asks which approach to take
- Offers multiple options without choosing
- Refuses to proceed without confirmation
- Asks for clarification on obvious details

#### 1.3.2. Preferred Patterns
- Chooses reasonable defaults
- States assumptions briefly
- Produces complete implementation
- Mentions alternatives without asking

#### 1.3.3. Pair Generation Strategy
- For each task, generate both response types
- Use GPT 5.2 Codex for preferred (executing)
- Use template-based generation for dispreferred (stalling)

---

## 2. Code Graph Construction

### 2.1. Integration with CodeGraphBuilder

#### 2.1.1. Import Existing Builder

```python
from rag_plusplus.service.code_graph.builder import CodeGraphBuilder
from rag_plusplus.service.code_graph.types import (
    CodeNode,
    CodeEdge,
    NodeType,
    EdgeType,
    UnifiedCodeGraph,
)
from rag_plusplus.service.code_graph.coordinates import CodeCoordinateComputer
```

#### 2.1.2. Configure for V3 Task Extraction

```python
@dataclass
class RepoWormConfig:
    """Configuration for Repo Worm task extraction."""
    
    # File filtering
    include_patterns: list[str] = field(default_factory=lambda: [
        "*.py", "*.ts", "*.js", "*.rs", "*.go"
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*test*", "*_test*", "*spec*", "node_modules/*", "venv/*"
    ])
    
    # Task extraction
    max_context_lines: int = 200
    min_function_lines: int = 3
    include_tests_for_context: bool = True
    
    # Task difficulty
    easy_threshold: int = 20      # Lines of code
    medium_threshold: int = 50
    hard_threshold: int = 100
    
    # DPO pair generation
    generate_dispreferred: bool = True
    dispreferred_templates: list[str] = field(default_factory=list)
```

#### 2.1.3. Builder Initialization

```python
class RepoWorm:
    """Codebase traversal agent for training data generation."""
    
    def __init__(
        self,
        repo_path: Path,
        config: RepoWormConfig = None,
        openai_client: OpenAI = None,
    ):
        self.repo_path = repo_path
        self.config = config or RepoWormConfig()
        self.openai = openai_client or OpenAI()
        
        # Initialize code graph builder
        self.graph_builder = CodeGraphBuilder()
        self.graph: UnifiedCodeGraph | None = None
        
        # Task extraction state
        self.extracted_tasks: list[RepoTask] = []
        self.task_contexts: dict[str, str] = {}
    
    async def initialize(self):
        """Build the code graph for the repository."""
        self.graph = await self.graph_builder.build_unified(self.repo_path)
        self.coord_computer = CodeCoordinateComputer(
            graph=self.graph,
            anchor=self.repo_path,
        )
```

### 2.2. Node Scanning

#### 2.2.1. File Paths

```python
def scan_file_paths(self) -> list[Path]:
    """Scan repository for relevant file paths."""
    files = []
    
    for pattern in self.config.include_patterns:
        matches = self.repo_path.rglob(pattern)
        for match in matches:
            # Check exclusions
            excluded = any(
                match.match(excl) 
                for excl in self.config.exclude_patterns
            )
            if not excluded:
                files.append(match)
    
    return files
```

#### 2.2.2. Exported Symbols/Classes

```python
import ast

def extract_exports(self, file_path: Path) -> list[dict]:
    """Extract exported symbols from a Python file."""
    exports = []
    
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                exports.append({
                    "type": "class",
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "docstring": ast.get_docstring(node),
                    "methods": [
                        m.name for m in node.body 
                        if isinstance(m, ast.FunctionDef)
                    ],
                })
            elif isinstance(node, ast.FunctionDef):
                if node.col_offset == 0:  # Top-level function
                    exports.append({
                        "type": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "docstring": ast.get_docstring(node),
                        "args": [a.arg for a in node.args.args],
                    })
    except:
        pass  # Skip files that can't be parsed
    
    return exports
```

#### 2.2.3. Function Signatures

```python
def extract_signatures(self, file_path: Path) -> list[dict]:
    """Extract function signatures for task generation."""
    signatures = []
    
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Build signature string
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)
                
                returns = ""
                if node.returns:
                    returns = f" -> {ast.unparse(node.returns)}"
                
                signature = f"def {node.name}({', '.join(args)}){returns}"
                
                # Check if body is stub
                is_stub = self._is_stub_body(node.body)
                
                signatures.append({
                    "signature": signature,
                    "name": node.name,
                    "is_stub": is_stub,
                    "line": node.lineno,
                    "file": str(file_path),
                })
    except:
        pass
    
    return signatures

def _is_stub_body(self, body: list) -> bool:
    """Check if function body is a stub."""
    if len(body) == 1:
        stmt = body[0]
        # pass statement
        if isinstance(stmt, ast.Pass):
            return True
        # ... (Ellipsis)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is ...:
                return True
        # raise NotImplementedError
        if isinstance(stmt, ast.Raise):
            if isinstance(stmt.exc, ast.Call):
                if isinstance(stmt.exc.func, ast.Name):
                    if stmt.exc.func.id == "NotImplementedError":
                        return True
    return False
```

#### 2.2.4. TODO/FIXME Comments

```python
import re

def extract_todos(self, file_path: Path) -> list[dict]:
    """Extract TODO/FIXME comments from file."""
    todos = []
    
    content = file_path.read_text()
    lines = content.split('\n')
    
    todo_pattern = re.compile(
        r'#\s*(TODO|FIXME|XXX|HACK|BUG)[\s:]+(.+)',
        re.IGNORECASE
    )
    
    for i, line in enumerate(lines, 1):
        match = todo_pattern.search(line)
        if match:
            todos.append({
                "type": match.group(1).upper(),
                "text": match.group(2).strip(),
                "line": i,
                "file": str(file_path),
                "context": self._get_context(lines, i, 10),
            })
    
    return todos

def _get_context(self, lines: list[str], line_num: int, window: int) -> str:
    """Get surrounding context for a line."""
    start = max(0, line_num - window - 1)
    end = min(len(lines), line_num + window)
    return '\n'.join(lines[start:end])
```

#### 2.2.5. Failing Tests

```python
import subprocess

def find_failing_tests(self) -> list[dict]:
    """Find failing tests in the repository."""
    failing = []
    
    try:
        # Run pytest with --collect-only to find tests
        result = subprocess.run(
            ["pytest", "--collect-only", "-q", str(self.repo_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        # Run pytest to find failures
        result = subprocess.run(
            ["pytest", str(self.repo_path), "-x", "--tb=no", "-q"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        # Parse failures
        for line in result.stdout.split('\n'):
            if 'FAILED' in line:
                # Extract test name and file
                match = re.match(r'(.+)::(.+) FAILED', line)
                if match:
                    failing.append({
                        "file": match.group(1),
                        "test": match.group(2),
                        "output": result.stdout,
                    })
    except:
        pass
    
    return failing
```

#### 2.2.6. Stub Functions

```python
def find_stubs(self) -> list[dict]:
    """Find all stub functions in the repository."""
    stubs = []
    
    for file_path in self.scan_file_paths():
        signatures = self.extract_signatures(file_path)
        for sig in signatures:
            if sig["is_stub"]:
                stubs.append({
                    **sig,
                    "context": self._get_file_context(
                        file_path, 
                        sig["line"]
                    ),
                })
    
    return stubs
```

#### 2.2.7. Intention Comments ("should", "plan", "later")

```python
def extract_intentions(self, file_path: Path) -> list[dict]:
    """Extract comments indicating future intentions."""
    intentions = []
    
    content = file_path.read_text()
    lines = content.split('\n')
    
    intention_patterns = [
        r'#.*\bshould\b',
        r'#.*\bplan\b',
        r'#.*\blater\b',
        r'#.*\beventually\b',
        r'#.*\bfuture\b',
        r'#.*\bneed to\b',
        r'#.*\bwant to\b',
    ]
    
    combined_pattern = re.compile('|'.join(intention_patterns), re.IGNORECASE)
    
    for i, line in enumerate(lines, 1):
        if combined_pattern.search(line):
            intentions.append({
                "text": line.strip(),
                "line": i,
                "file": str(file_path),
                "context": self._get_context(lines, i, 5),
            })
    
    return intentions
```

### 2.3. Edge Construction

#### 2.3.1. Import Relationships

```python
def build_import_edges(self) -> list[CodeEdge]:
    """Build edges for import relationships."""
    edges = []
    
    for file_path in self.scan_file_paths():
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        edges.append(CodeEdge(
                            source=str(file_path),
                            target=alias.name,
                            edge_type=EdgeType.IMPORTS,
                        ))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        edges.append(CodeEdge(
                            source=str(file_path),
                            target=node.module,
                            edge_type=EdgeType.IMPORTS,
                        ))
        except:
            pass
    
    return edges
```

#### 2.3.2. Call Relationships

```python
def build_call_edges(self) -> list[CodeEdge]:
    """Build edges for function call relationships."""
    edges = []
    
    for file_path in self.scan_file_paths():
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            # Track current function context
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    caller = f"{file_path}::{node.name}"
                    
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                callee = child.func.id
                                edges.append(CodeEdge(
                                    source=caller,
                                    target=callee,
                                    edge_type=EdgeType.CALLS,
                                ))
        except:
            pass
    
    return edges
```

#### 2.3.3. Reference Relationships

```python
def build_reference_edges(self) -> list[CodeEdge]:
    """Build edges for symbol references."""
    edges = []
    
    # Build symbol table first
    symbols = {}
    for file_path in self.scan_file_paths():
        exports = self.extract_exports(file_path)
        for export in exports:
            symbols[export["name"]] = f"{file_path}::{export['name']}"
    
    # Find references
    for file_path in self.scan_file_paths():
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if node.id in symbols:
                        edges.append(CodeEdge(
                            source=str(file_path),
                            target=symbols[node.id],
                            edge_type=EdgeType.REFERENCES,
                        ))
        except:
            pass
    
    return edges
```

#### 2.3.4. TODO Dependencies

```python
def build_todo_edges(self) -> list[CodeEdge]:
    """Build edges for TODO dependencies."""
    edges = []
    
    for file_path in self.scan_file_paths():
        todos = self.extract_todos(file_path)
        
        for todo in todos:
            # Check if TODO references another symbol
            text = todo["text"].lower()
            
            # Look for "depends on X", "after X", "needs X"
            dep_patterns = [
                r'depends on (\w+)',
                r'after (\w+)',
                r'needs (\w+)',
                r'requires (\w+)',
                r'blocked by (\w+)',
            ]
            
            for pattern in dep_patterns:
                match = re.search(pattern, text)
                if match:
                    edges.append(CodeEdge(
                        source=f"{file_path}:{todo['line']}",
                        target=match.group(1),
                        edge_type=EdgeType.TODO_DEPENDENCY,
                    ))
    
    return edges
```

---

## 3. Task Generation

### 3.1. Task Types

#### 3.1.1. Implementation Tasks

```python
@dataclass
class ImplementationTask:
    """Task to implement a function or class."""
    
    task_type: str = "implementation"
    task_id: str = ""
    
    # What to implement
    target_signature: str = ""
    target_file: str = ""
    target_line: int = 0
    
    # Context
    interface_definition: str = ""
    usage_examples: list[str] = field(default_factory=list)
    related_implementations: list[str] = field(default_factory=list)
    
    # Constraints
    must_compile: bool = True
    must_match_patterns: bool = True
    no_new_dependencies: bool = True
    no_questions: bool = True
    
    # Prompt templates
    prompt_templates: list[str] = field(default_factory=lambda: [
        "Implement {signature} given the interface below.",
        "Complete the function stub for {name}.",
        "Add the missing implementation for {name}.",
    ])

def generate_implementation_tasks(self) -> list[ImplementationTask]:
    """Generate tasks for stub implementations."""
    tasks = []
    
    stubs = self.find_stubs()
    
    for stub in stubs:
        # Find related context
        interface = self._find_interface(stub)
        usage = self._find_usage_examples(stub)
        related = self._find_related_implementations(stub)
        
        task = ImplementationTask(
            task_id=f"impl_{stub['file']}_{stub['name']}",
            target_signature=stub["signature"],
            target_file=stub["file"],
            target_line=stub["line"],
            interface_definition=interface,
            usage_examples=usage,
            related_implementations=related,
        )
        
        tasks.append(task)
    
    return tasks
```

##### 3.1.1.1. "Implement X given interface Y"

```python
def format_implement_interface_prompt(self, task: ImplementationTask) -> str:
    """Format prompt for interface implementation."""
    
    return f"""Implement the following function according to its interface:

SIGNATURE:
```python
{task.target_signature}
```

INTERFACE/PROTOCOL:
```python
{task.interface_definition}
```

USAGE EXAMPLES:
{chr(10).join(f'```python{chr(10)}{ex}{chr(10)}```' for ex in task.usage_examples[:3])}

CONSTRAINTS:
- Must compile without errors
- Must match existing code patterns
- Must not introduce new dependencies
- Do NOT ask clarifying questions - make reasonable assumptions

Provide the complete implementation:"""
```

##### 3.1.1.2. "Complete function stub"

```python
def format_complete_stub_prompt(self, task: ImplementationTask) -> str:
    """Format prompt for stub completion."""
    
    return f"""Complete this function stub:

FILE: {task.target_file}
LINE: {task.target_line}

```python
{task.target_signature}
    pass  # TODO: Implement
```

RELATED IMPLEMENTATIONS FOR REFERENCE:
{chr(10).join(f'```python{chr(10)}{impl}{chr(10)}```' for impl in task.related_implementations[:2])}

CONSTRAINTS:
- Follow the patterns shown in related implementations
- Make reasonable assumptions for any unknowns
- Do NOT ask questions - just implement

Provide the complete function body:"""
```

##### 3.1.1.3. "Add missing method"

```python
def format_add_method_prompt(self, task: ImplementationTask) -> str:
    """Format prompt for adding missing method."""
    
    return f"""Add the missing method to this class:

CLASS CONTEXT:
```python
{task.interface_definition}
```

MISSING METHOD: {task.target_signature}

The method is referenced but not implemented. Based on the class context
and how the method is used, implement it.

USAGE:
{chr(10).join(f'```python{chr(10)}{ex}{chr(10)}```' for ex in task.usage_examples[:3])}

CONSTRAINTS:
- Must integrate with existing class methods
- Follow existing code style
- Do NOT ask for clarification

Provide the method implementation:"""
```

#### 3.1.2. Completion Tasks

```python
@dataclass
class CompletionTask:
    """Task to complete TODO sections."""
    
    task_type: str = "completion"
    task_id: str = ""
    
    # TODO info
    todo_text: str = ""
    todo_type: str = ""  # TODO, FIXME, etc.
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
```

##### 3.1.2.1. "Complete TODO section"

```python
def format_complete_todo_prompt(self, task: CompletionTask) -> str:
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
```

##### 3.1.2.2. "Finish partial implementation"

```python
def format_finish_partial_prompt(self, task: CompletionTask) -> str:
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
```

#### 3.1.3. Refactoring Tasks

```python
@dataclass
class RefactoringTask:
    """Task to refactor code."""
    
    task_type: str = "refactoring"
    task_id: str = ""
    
    # Target
    target_code: str = ""
    target_file: str = ""
    target_lines: tuple[int, int] = (0, 0)
    
    # Pattern to match
    pattern_example: str = ""
    pattern_file: str = ""
    
    # Refactoring type
    refactor_type: str = ""  # extract, inline, rename, restructure
    
    # Constraints
    must_preserve_behavior: bool = True
    must_compile: bool = True
```

##### 3.1.3.1. "Refactor to match pattern in Z"

```python
def format_refactor_pattern_prompt(self, task: RefactoringTask) -> str:
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
```

##### 3.1.3.2. "Extract to module"

```python
def format_extract_module_prompt(self, task: RefactoringTask) -> str:
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
```

#### 3.1.4. Test Tasks

```python
@dataclass
class TestTask:
    """Task to write tests."""
    
    task_type: str = "test"
    task_id: str = ""
    
    # Function to test
    function_signature: str = ""
    function_body: str = ""
    function_file: str = ""
    
    # Existing tests
    existing_tests: list[str] = field(default_factory=list)
    
    # Test requirements
    coverage_targets: list[str] = field(default_factory=list)
```

##### 3.1.4.1. "Write tests for function A"

```python
def format_write_tests_prompt(self, task: TestTask) -> str:
    """Format prompt for test writing."""
    
    return f"""Write comprehensive tests for this function:

FUNCTION:
```python
{task.function_signature}
{task.function_body}
```

EXISTING TESTS FOR REFERENCE:
{chr(10).join(f'```python{chr(10)}{t}{chr(10)}```' for t in task.existing_tests[:2])}

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
```

##### 3.1.4.2. "Add edge case coverage"

```python
def format_edge_case_prompt(self, task: TestTask) -> str:
    """Format prompt for edge case coverage."""
    
    return f"""Add edge case tests for this function:

FUNCTION:
```python
{task.function_signature}
{task.function_body}
```

EXISTING TESTS:
{chr(10).join(f'```python{chr(10)}{t}{chr(10)}```' for t in task.existing_tests)}

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
```

### 3.2. Context Attachment

#### 3.2.1. Include File Snippets (Bounded)

```python
def get_bounded_context(
    self,
    file_path: Path,
    center_line: int,
    max_lines: int = 200
) -> str:
    """Get bounded context around a specific line."""
    
    content = file_path.read_text()
    lines = content.split('\n')
    
    # Calculate window
    half = max_lines // 2
    start = max(0, center_line - half)
    end = min(len(lines), center_line + half)
    
    # Include line numbers for reference
    numbered = []
    for i, line in enumerate(lines[start:end], start + 1):
        numbered.append(f"{i:4d} | {line}")
    
    return '\n'.join(numbered)
```

#### 3.2.2. Include Import Context

```python
def get_import_context(self, file_path: Path) -> str:
    """Extract import statements from file."""
    
    content = file_path.read_text()
    lines = content.split('\n')
    
    imports = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(line)
        elif imports and not stripped:
            # End of import block
            break
    
    return '\n'.join(imports)
```

#### 3.2.3. Include Usage Examples

```python
def find_usage_examples(self, symbol_name: str) -> list[str]:
    """Find examples of how a symbol is used."""
    
    examples = []
    
    for file_path in self.scan_file_paths():
        content = file_path.read_text()
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if symbol_name in line and not line.strip().startswith('#'):
                # Get surrounding context
                context = self._get_context(lines, i + 1, 3)
                examples.append(context)
                
                if len(examples) >= 5:
                    return examples
    
    return examples
```

### 3.3. Ground Truth Constraints

#### 3.3.1. Must Compile

```python
import subprocess
import tempfile

def validate_compiles(self, code: str, file_path: Path) -> bool:
    """Validate that generated code compiles."""
    
    # Create temp file with the code
    with tempfile.NamedTemporaryFile(
        suffix=file_path.suffix,
        delete=False
    ) as f:
        f.write(code.encode())
        temp_path = f.name
    
    try:
        if file_path.suffix == '.py':
            result = subprocess.run(
                ['python', '-m', 'py_compile', temp_path],
                capture_output=True,
            )
            return result.returncode == 0
        
        elif file_path.suffix == '.ts':
            result = subprocess.run(
                ['tsc', '--noEmit', temp_path],
                capture_output=True,
            )
            return result.returncode == 0
        
        # Add other languages as needed
        return True  # Assume valid if no validator
    finally:
        Path(temp_path).unlink()
```

#### 3.3.2. Must Match Existing Patterns

```python
def validate_patterns(self, code: str, reference_code: str) -> bool:
    """Validate that code matches existing patterns."""
    
    checks = [
        # Indentation style
        self._check_indentation_match(code, reference_code),
        # Naming conventions
        self._check_naming_conventions(code, reference_code),
        # Documentation style
        self._check_docstring_style(code, reference_code),
    ]
    
    return all(checks)

def _check_indentation_match(self, code: str, reference: str) -> bool:
    """Check if indentation matches reference."""
    # Detect reference indentation (spaces vs tabs, width)
    ref_indent = self._detect_indentation(reference)
    code_indent = self._detect_indentation(code)
    return ref_indent == code_indent
```

#### 3.3.3. Must Not Introduce New Dependencies

```python
def validate_no_new_dependencies(
    self,
    code: str,
    existing_imports: set[str]
) -> tuple[bool, list[str]]:
    """Check that no new dependencies are introduced."""
    
    # Extract imports from generated code
    tree = ast.parse(code)
    new_imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                new_imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                new_imports.add(node.module.split('.')[0])
    
    # Check for new dependencies
    added = new_imports - existing_imports
    
    # Filter out standard library
    stdlib = {'os', 'sys', 're', 'json', 'typing', 'dataclasses', 
              'collections', 'functools', 'itertools', 'pathlib'}
    truly_new = added - stdlib
    
    return len(truly_new) == 0, list(truly_new)
```

#### 3.3.4. Must Not Ask Questions

```python
def validate_no_questions(self, response: str) -> bool:
    """Validate that response doesn't ask questions."""
    
    # Import classifier from corpus surgery
    from .corpus_surgery.classifier import compute_stall_score
    
    stall_score = compute_stall_score(response)
    return stall_score < 3
```

---

## 4. GPT 5.2 Codex Integration

### 4.1. API Configuration

#### 4.1.1. Model: gpt-5.2-codex

```python
CODEX_CONFIG = {
    "model": "gpt-5.2-codex",
    "max_tokens": 8192,
    "temperature": 0.2,
}
```

#### 4.1.2. Context Window: Full File Context

```python
def prepare_context_window(
    self,
    task: ImplementationTask | CompletionTask | RefactoringTask | TestTask,
    max_context_tokens: int = 16000
) -> str:
    """Prepare context window for Codex."""
    
    context_parts = []
    
    # Add file header
    context_parts.append(f"# File: {task.target_file}")
    
    # Add imports
    imports = self.get_import_context(Path(task.target_file))
    context_parts.append(imports)
    
    # Add relevant definitions
    if hasattr(task, 'interface_definition') and task.interface_definition:
        context_parts.append(f"\n# Interface/Protocol:\n{task.interface_definition}")
    
    # Add surrounding code
    if hasattr(task, 'surrounding_code') and task.surrounding_code:
        context_parts.append(f"\n# Surrounding code:\n{task.surrounding_code}")
    
    # Add examples
    if hasattr(task, 'usage_examples') and task.usage_examples:
        context_parts.append("\n# Usage examples:")
        for ex in task.usage_examples[:3]:
            context_parts.append(ex)
    
    return '\n'.join(context_parts)
```

#### 4.1.3. Temperature: 0.2 for Determinism

```python
async def call_codex(
    self,
    prompt: str,
    context: str,
    task_type: str
) -> str:
    """Call GPT 5.2 Codex with low temperature for deterministic output."""
    
    response = await self.openai.responses.create(
        model="gpt-5.2-codex",
        input=f"{context}\n\n{prompt}",
        temperature=0.2,  # Low for determinism
        max_tokens=4096,
    )
    
    return response.output
```

### 4.2. Prompt Engineering

#### 4.2.1. System Prompt for Code Generation

```python
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
```

#### 4.2.2. Context Injection Format

```python
def format_codex_request(
    self,
    task: ImplementationTask,
    context: str
) -> str:
    """Format a complete Codex request."""
    
    return f"""{CODEX_SYSTEM_PROMPT}

REPOSITORY CONTEXT:
{context}

TASK:
{self.format_implement_interface_prompt(task)}

Provide the implementation:"""
```

#### 4.2.3. Output Format (Unified Diff)

```python
def extract_diff_output(self, response: str) -> str | None:
    """Extract unified diff from Codex response."""
    
    # Look for diff blocks
    diff_pattern = r'```diff\n([\s\S]*?)\n```'
    match = re.search(diff_pattern, response)
    
    if match:
        return match.group(1)
    
    # Look for raw diff markers
    if '---' in response and '+++' in response:
        lines = response.split('\n')
        diff_lines = []
        in_diff = False
        
        for line in lines:
            if line.startswith('---') or line.startswith('+++'):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
                if line.startswith('@@') and len(diff_lines) > 3:
                    # Found complete hunk header
                    pass
        
        return '\n'.join(diff_lines) if diff_lines else None
    
    return None

def extract_code_output(self, response: str) -> str | None:
    """Extract code block from Codex response."""
    
    # Look for python code blocks
    code_pattern = r'```python\n([\s\S]*?)\n```'
    match = re.search(code_pattern, response)
    
    if match:
        return match.group(1)
    
    # Look for generic code blocks
    code_pattern = r'```\n([\s\S]*?)\n```'
    match = re.search(code_pattern, response)
    
    if match:
        return match.group(1)
    
    return None
```

### 4.3. Response Parsing

#### 4.3.1. Extract Diff Blocks

```python
@dataclass
class ParsedResponse:
    """Parsed response from Codex."""
    
    code: str | None = None
    diff: str | None = None
    explanation: str | None = None
    assumptions: list[str] = field(default_factory=list)
    is_valid: bool = False
    errors: list[str] = field(default_factory=list)

def parse_codex_response(self, response: str) -> ParsedResponse:
    """Parse a Codex response."""
    
    result = ParsedResponse()
    
    # Extract code
    result.code = self.extract_code_output(response)
    
    # Extract diff
    result.diff = self.extract_diff_output(response)
    
    # Extract assumptions
    assumption_pattern = r'#\s*Assumption:\s*(.+)'
    result.assumptions = re.findall(assumption_pattern, response)
    
    # Extract explanation (text outside code blocks)
    explanation_parts = []
    in_code = False
    for line in response.split('\n'):
        if line.startswith('```'):
            in_code = not in_code
        elif not in_code and line.strip():
            explanation_parts.append(line)
    result.explanation = '\n'.join(explanation_parts)
    
    # Validate
    result.is_valid = bool(result.code or result.diff)
    
    return result
```

#### 4.3.2. Validate Syntax

```python
def validate_syntax(self, code: str, language: str = "python") -> tuple[bool, str]:
    """Validate code syntax."""
    
    if language == "python":
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)
    
    elif language == "typescript":
        # Use tsc for validation
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            f.write(code.encode())
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['tsc', '--noEmit', temp_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr
        finally:
            Path(temp_path).unlink()
    
    return True, ""  # Assume valid for unknown languages
```

#### 4.3.3. Check Compilation

```python
async def validate_response(
    self,
    response: ParsedResponse,
    task: ImplementationTask
) -> ParsedResponse:
    """Validate a parsed Codex response."""
    
    errors = []
    
    # Check syntax
    if response.code:
        is_valid, error = self.validate_syntax(response.code)
        if not is_valid:
            errors.append(f"Syntax error: {error}")
    
    # Check compilation
    if response.code and not errors:
        compiles = self.validate_compiles(response.code, Path(task.target_file))
        if not compiles:
            errors.append("Code does not compile")
    
    # Check no new dependencies
    if response.code and not errors:
        existing = self._get_existing_imports(task.target_file)
        no_new, added = self.validate_no_new_dependencies(response.code, existing)
        if not no_new:
            errors.append(f"New dependencies added: {added}")
    
    # Check no questions
    if not self.validate_no_questions(response.explanation or ""):
        errors.append("Response contains questions")
    
    response.errors = errors
    response.is_valid = len(errors) == 0
    
    return response
```

---

## 5. DPO Pair Generation

### 5.1. Dispreferred Response Template

#### 5.1.1. Asks for Confirmation

```python
DISPREFERRED_CONFIRMATION_TEMPLATES = [
    "I can implement this for you. Would you like me to proceed with {approach}?",
    "Before I implement this, should I use {option_a} or {option_b}?",
    "I'll need to know: do you want this to {option_a} or {option_b}?",
    "Can you confirm that you want me to {action}?",
    "Just to make sure, should I {action}?",
]

def generate_confirmation_dispreferred(
    self,
    task: ImplementationTask
) -> str:
    """Generate a dispreferred response that asks for confirmation."""
    
    template = random.choice(DISPREFERRED_CONFIRMATION_TEMPLATES)
    
    # Fill in template
    options = self._generate_options(task)
    
    return template.format(
        approach=options[0] if options else "the standard approach",
        option_a=options[0] if len(options) > 0 else "option A",
        option_b=options[1] if len(options) > 1 else "option B",
        action=self._summarize_task(task),
    )
```

#### 5.1.2. Offers Options Without Acting

```python
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
]

def generate_options_dispreferred(
    self,
    task: ImplementationTask
) -> str:
    """Generate a dispreferred response that offers options."""
    
    template = random.choice(DISPREFERRED_OPTIONS_TEMPLATES)
    options = self._generate_options(task)
    
    return template.format(
        option_a=options[0] if len(options) > 0 else "Using standard library",
        option_b=options[1] if len(options) > 1 else "Using a custom implementation",
        option_c=options[2] if len(options) > 2 else "Using a third-party library",
        benefit_a="simplicity",
        benefit_b="performance",
    )
```

#### 5.1.3. Refuses to Proceed

```python
DISPREFERRED_REFUSAL_TEMPLATES = [
    "I need more information before I can implement this. Could you provide {missing}?",
    "To proceed, I'll need to know {missing}. Can you clarify?",
    "I can't implement this without knowing {missing}. Please specify.",
    "Before I proceed, could you tell me {missing}?",
]

def generate_refusal_dispreferred(
    self,
    task: ImplementationTask
) -> str:
    """Generate a dispreferred response that refuses to proceed."""
    
    template = random.choice(DISPREFERRED_REFUSAL_TEMPLATES)
    
    # Generate fake "missing" info
    fake_missing = [
        "which error handling strategy to use",
        "the exact input format",
        "whether to support async operations",
        "the preferred naming convention",
    ]
    
    return template.format(missing=random.choice(fake_missing))
```

### 5.2. Preferred Response Template

#### 5.2.1. Chooses Sane Defaults

```python
async def generate_preferred_response(
    self,
    task: ImplementationTask
) -> str:
    """Generate preferred response that executes immediately."""
    
    # Build prompt that enforces execution
    prompt = f"""{self.format_implement_interface_prompt(task)}

CRITICAL: Execute immediately. Choose reasonable defaults for any unknowns.
State assumptions briefly, then provide the complete implementation.
Do NOT ask questions."""
    
    context = self.prepare_context_window(task)
    
    response = await self.call_codex(prompt, context, "implementation")
    
    return response
```

#### 5.2.2. States Assumptions Briefly

```python
def ensure_assumptions_stated(self, response: str) -> str:
    """Ensure assumptions are stated at the start."""
    
    parsed = self.parse_codex_response(response)
    
    if parsed.assumptions:
        # Already has assumptions, good
        return response
    
    # Add assumption header if code makes implicit assumptions
    assumptions = self._infer_assumptions(parsed.code)
    
    if assumptions:
        assumption_text = "# Assumptions: " + ", ".join(assumptions)
        
        # Insert at start of code
        if parsed.code:
            return assumption_text + "\n\n" + response
    
    return response
```

#### 5.2.3. Produces Complete Diff

```python
def generate_complete_diff(
    self,
    original_file: str,
    new_code: str,
    file_path: str
) -> str:
    """Generate complete unified diff."""
    
    import difflib
    
    original_lines = original_file.split('\n')
    new_lines = new_code.split('\n')
    
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    )
    
    return '\n'.join(diff)
```

---

## 6. Output Schema

### 6.1. repo_task Record Format

```python
@dataclass
class RepoTaskRecord:
    """CTv3.1 record for repo-grounded tasks."""
    
    schema_version: str = "ctv3.1"
    record_id: str = ""
    record_type: str = "repo_task"
    
    source: dict = field(default_factory=lambda: {
        "origin": "repo_worm",
        "provider": "gpt-5.2-codex",
        "source_id": "",
        "created_at_utc": "",
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
```

### 6.2. Attachment Specification

```python
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
```

### 6.3. Quality Labeling

```python
def label_quality(
    self,
    task: ImplementationTask,
    response: ParsedResponse
) -> dict:
    """Label quality for a generated response."""
    
    quality = {
        "gold": False,
        "weight": 1.0,
        "review_status": "auto",
        "failure_modes": [],
    }
    
    # Check for failure modes
    if not response.is_valid:
        quality["failure_modes"].extend(response.errors)
    
    if not self.validate_no_questions(response.explanation or ""):
        quality["failure_modes"].append("asked_permission")
    
    if response.explanation and response.explanation.rstrip().endswith('?'):
        quality["failure_modes"].append("ended_with_question")
    
    # Determine if gold
    if not quality["failure_modes"]:
        quality["gold"] = True
        quality["review_status"] = "auto_passed"
    else:
        quality["weight"] = 0.3  # Downweight problematic responses
    
    return quality
```

---

## 7. Complete Pipeline

```python
class RepoWormPipeline:
    """Complete Repo Worm pipeline for training data generation."""
    
    async def run(
        self,
        repo_path: Path,
        output_dir: Path,
    ) -> dict:
        """Run the complete pipeline."""
        
        # Initialize
        worm = RepoWorm(repo_path)
        await worm.initialize()
        
        # Generate tasks
        impl_tasks = worm.generate_implementation_tasks()
        completion_tasks = worm.generate_completion_tasks()
        refactor_tasks = worm.generate_refactoring_tasks()
        test_tasks = worm.generate_test_tasks()
        
        all_tasks = impl_tasks + completion_tasks + refactor_tasks + test_tasks
        
        # Generate responses
        sft_records = []
        dpo_records = []
        
        for task in all_tasks:
            # Generate preferred response (executes)
            preferred = await worm.generate_preferred_response(task)
            parsed = worm.parse_codex_response(preferred)
            validated = await worm.validate_response(parsed, task)
            
            if validated.is_valid:
                # Create SFT record
                record = worm.create_sft_record(task, validated)
                sft_records.append(record)
            
            # Generate dispreferred responses (stalls)
            for template_type in ["confirmation", "options", "refusal"]:
                dispreferred = worm.generate_dispreferred(task, template_type)
                
                # Create DPO pair
                dpo_pair = worm.create_dpo_pair(
                    task, 
                    preferred=validated.code or preferred,
                    dispreferred=dispreferred
                )
                dpo_records.append(dpo_pair)
        
        # Export
        worm.export_sft(sft_records, output_dir / "repo_sft.jsonl")
        worm.export_dpo(dpo_records, output_dir / "repo_dpo.jsonl")
        
        return {
            "tasks_generated": len(all_tasks),
            "sft_records": len(sft_records),
            "dpo_pairs": len(dpo_records),
        }
```

