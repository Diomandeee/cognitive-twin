"""
Task Generator for Repo Worm.

Generates tasks from code analysis:
- Implementation tasks from stubs
- Completion tasks from TODOs
- Refactoring tasks from pattern mismatches
- Test tasks from untested functions
"""

from pathlib import Path
from typing import Optional

from .task_types import (
    RepoWormConfig,
    ImplementationTask,
    CompletionTask,
    RefactoringTask,
    TestTask,
    Task,
)
from .code_scanner import CodeScanner


class TaskGenerator:
    """Generates training tasks from code analysis."""
    
    def __init__(
        self,
        repo_path: Path,
        config: Optional[RepoWormConfig] = None,
        scanner: Optional[CodeScanner] = None,
    ):
        self.repo_path = Path(repo_path)
        self.config = config or RepoWormConfig()
        self.scanner = scanner or CodeScanner(repo_path, config)
    
    # =========================================================================
    # IMPLEMENTATION TASKS
    # =========================================================================
    
    def generate_implementation_tasks(self) -> list[ImplementationTask]:
        """Generate tasks for stub implementations."""
        tasks = []
        
        stubs = self.scanner.find_stubs()
        
        for stub in stubs:
            # Find related context
            interface = self._find_interface(stub)
            usage = self.scanner.find_usage_examples(stub["name"])
            related = self._find_related_implementations(stub)
            imports = stub.get("imports", "")
            
            # Determine difficulty
            difficulty = self._compute_difficulty(stub)
            
            task = ImplementationTask(
                target_signature=stub["signature"],
                target_file=stub["file"],
                target_line=stub["line"],
                interface_definition=interface,
                usage_examples=usage,
                related_implementations=related,
                file_imports=imports,
                difficulty=difficulty,
                estimated_lines=stub.get("line_count", 10),
                prompt_variant="complete_stub",
            )
            
            tasks.append(task)
        
        return tasks
    
    def _find_interface(self, stub: dict) -> str:
        """Find interface/protocol definition for a stub."""
        # Look for class containing this method
        file_path = Path(stub["file"])
        
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            # Find the class this method belongs to
            stub_line = stub["line"]
            for i in range(stub_line - 1, -1, -1):
                line = lines[i]
                if line.strip().startswith("class "):
                    # Found class definition, extract it
                    class_start = i
                    # Find class end (next non-indented line or end of file)
                    class_lines = [lines[i]]
                    base_indent = len(line) - len(line.lstrip())
                    
                    for j in range(i + 1, min(i + 50, len(lines))):  # Limit to 50 lines
                        if lines[j].strip() and not lines[j].startswith(' ' * (base_indent + 1)):
                            break
                        class_lines.append(lines[j])
                    
                    return '\n'.join(class_lines)
            
        except:
            pass
        
        return stub.get("context", "")
    
    def _find_related_implementations(self, stub: dict) -> list[str]:
        """Find similar implemented functions for reference."""
        related = []
        similar = self.scanner.find_similar_functions(stub)
        
        for sig in similar:
            if not sig["is_stub"]:
                # Get the full function body
                try:
                    file_path = Path(sig["file"])
                    content = file_path.read_text()
                    lines = content.split('\n')
                    
                    start = sig["line"] - 1
                    end = min(sig["end_line"], len(lines))
                    
                    func_body = '\n'.join(lines[start:end])
                    related.append(func_body)
                except:
                    pass
        
        return related[:3]  # Limit to 3
    
    def _compute_difficulty(self, item: dict) -> str:
        """Compute task difficulty based on line count."""
        lines = item.get("line_count", item.get("estimated_lines", 10))
        
        if lines <= self.config.easy_threshold:
            return "easy"
        elif lines <= self.config.medium_threshold:
            return "medium"
        else:
            return "hard"
    
    # =========================================================================
    # COMPLETION TASKS
    # =========================================================================
    
    def generate_completion_tasks(self) -> list[CompletionTask]:
        """Generate tasks for TODO completion."""
        tasks = []
        
        for file_path in self.scanner.scan_file_paths():
            if not file_path.suffix == '.py':
                continue
            
            todos = self.scanner.extract_todos(file_path)
            
            for todo in todos:
                # Get function signature if TODO is inside a function
                func_sig = ""
                if todo.get("function"):
                    func_sig = self._get_function_signature(
                        file_path, 
                        todo["function"]
                    )
                
                imports = self.scanner.get_import_context(file_path)
                
                task = CompletionTask(
                    todo_text=todo["text"],
                    todo_type=todo["type"],
                    file=str(file_path),
                    line=todo["line"],
                    surrounding_code=todo["context"],
                    function_signature=func_sig,
                    file_imports=imports,
                    difficulty=self._estimate_todo_difficulty(todo),
                    prompt_variant="complete_todo",
                )
                
                tasks.append(task)
        
        return tasks
    
    def _get_function_signature(self, file_path: Path, func_name: str) -> str:
        """Get the signature of a function by name."""
        signatures = self.scanner.extract_signatures(file_path)
        
        for sig in signatures:
            if sig["name"] == func_name:
                return sig["signature"]
        
        return ""
    
    def _estimate_todo_difficulty(self, todo: dict) -> str:
        """Estimate difficulty based on TODO text."""
        text = todo["text"].lower()
        
        # Easy keywords
        if any(k in text for k in ["add", "fix", "simple", "minor"]):
            return "easy"
        
        # Hard keywords
        if any(k in text for k in ["refactor", "redesign", "complex", "overhaul"]):
            return "hard"
        
        return "medium"
    
    # =========================================================================
    # REFACTORING TASKS
    # =========================================================================
    
    def generate_refactoring_tasks(self) -> list[RefactoringTask]:
        """Generate refactoring tasks from pattern mismatches."""
        tasks = []
        
        # Find functions that could be refactored to match patterns
        for file_path in self.scanner.scan_file_paths():
            if not file_path.suffix == '.py':
                continue
            
            signatures = self.scanner.extract_signatures(file_path)
            
            for sig in signatures:
                if sig["is_stub"]:
                    continue
                
                # Find similar functions to use as pattern
                similar = self.scanner.find_similar_functions(sig)
                
                if similar:
                    pattern_func = similar[0]
                    
                    # Only create task if pattern is significantly different
                    if self._patterns_differ(sig, pattern_func):
                        task = RefactoringTask(
                            target_code=self._get_function_body(sig),
                            target_file=sig["file"],
                            target_lines=(sig["line"], sig["end_line"]),
                            pattern_example=self._get_function_body(pattern_func),
                            pattern_file=pattern_func["file"],
                            refactor_type="match_pattern",
                            difficulty="medium",
                            prompt_variant="refactor_pattern",
                        )
                        tasks.append(task)
        
        return tasks[:20]  # Limit refactoring tasks
    
    def _patterns_differ(self, func1: dict, func2: dict) -> bool:
        """Check if two functions have significantly different patterns."""
        # Simple heuristic: different line counts
        lines1 = func1.get("line_count", 0)
        lines2 = func2.get("line_count", 0)
        
        if lines1 == 0 or lines2 == 0:
            return False
        
        ratio = max(lines1, lines2) / min(lines1, lines2)
        return ratio > 1.5
    
    def _get_function_body(self, sig: dict) -> str:
        """Get the full body of a function."""
        try:
            file_path = Path(sig["file"])
            content = file_path.read_text()
            lines = content.split('\n')
            
            start = sig["line"] - 1
            end = min(sig["end_line"], len(lines))
            
            return '\n'.join(lines[start:end])
        except:
            return ""
    
    # =========================================================================
    # TEST TASKS
    # =========================================================================
    
    def generate_test_tasks(self) -> list[TestTask]:
        """Generate tasks for writing tests."""
        tasks = []
        
        untested = self.scanner.find_untested_functions()
        
        for func in untested:
            if func["is_stub"]:
                continue
            
            # Get function body
            func_body = self._get_function_body(func)
            
            # Find existing tests in similar files
            existing_tests = self._find_example_tests(func)
            
            task = TestTask(
                function_signature=func["signature"],
                function_body=func_body,
                function_file=func["file"],
                function_line=func["line"],
                existing_tests=existing_tests,
                difficulty=self._compute_difficulty(func),
                prompt_variant="write_tests",
            )
            
            tasks.append(task)
        
        return tasks[:30]  # Limit test tasks
    
    def _find_example_tests(self, func: dict) -> list[str]:
        """Find example tests from the repository."""
        examples = []
        
        for file_path in self.scanner.scan_file_paths():
            if 'test' not in file_path.name.lower():
                continue
            
            try:
                content = file_path.read_text()
                
                # Find test functions
                signatures = self.scanner.extract_signatures(file_path)
                
                for sig in signatures:
                    if sig["name"].startswith("test_"):
                        body = self._get_function_body(sig)
                        if len(body) < 500:  # Only short tests
                            examples.append(body)
                        
                        if len(examples) >= 3:
                            return examples
            except:
                continue
        
        return examples
    
    # =========================================================================
    # ALL TASKS
    # =========================================================================
    
    def generate_all_tasks(self) -> list[Task]:
        """Generate all types of tasks."""
        all_tasks = []
        
        # Implementation tasks
        impl_tasks = self.generate_implementation_tasks()
        all_tasks.extend(impl_tasks)
        
        # Completion tasks
        comp_tasks = self.generate_completion_tasks()
        all_tasks.extend(comp_tasks)
        
        # Refactoring tasks
        refac_tasks = self.generate_refactoring_tasks()
        all_tasks.extend(refac_tasks)
        
        # Test tasks
        test_tasks = self.generate_test_tasks()
        all_tasks.extend(test_tasks)
        
        return all_tasks
    
    # =========================================================================
    # CONTEXT HELPERS
    # =========================================================================
    
    def get_bounded_context(
        self,
        file_path: Path,
        center_line: int,
        max_lines: int = 200,
    ) -> str:
        """Get bounded context around a specific line."""
        try:
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
        except:
            return ""
    
    def get_import_context(self, file_path: Path) -> str:
        """Get import context from a file."""
        return self.scanner.get_import_context(file_path)
    
    def find_usage_examples(self, symbol_name: str) -> list[str]:
        """Find usage examples for a symbol."""
        return self.scanner.find_usage_examples(symbol_name)

