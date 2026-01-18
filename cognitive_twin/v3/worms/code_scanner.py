"""
Code Scanner for Repo Worm.

AST-based code analysis for extracting:
- Exported symbols and classes
- Function signatures
- TODO/FIXME comments
- Stub functions
- Intention comments
"""

import ast
import re
from pathlib import Path
from typing import Optional
from fnmatch import fnmatch

from .task_types import RepoWormConfig


class CodeScanner:
    """AST-based code scanner for Python files."""
    
    def __init__(self, repo_path: Path, config: Optional[RepoWormConfig] = None):
        self.repo_path = Path(repo_path)
        self.config = config or RepoWormConfig()
    
    # =========================================================================
    # FILE SCANNING
    # =========================================================================
    
    def scan_file_paths(self) -> list[Path]:
        """Scan repository for relevant file paths."""
        files = []
        
        for pattern in self.config.include_patterns:
            # Handle glob pattern
            if pattern.startswith("*."):
                ext = pattern[1:]  # Get extension including dot
                matches = self.repo_path.rglob(f"*{ext}")
            else:
                matches = self.repo_path.rglob(pattern)
            
            for match in matches:
                if self._should_include(match):
                    files.append(match)
        
        return files
    
    def _should_include(self, path: Path) -> bool:
        """Check if path should be included based on exclude patterns."""
        rel_path = str(path.relative_to(self.repo_path))
        
        for pattern in self.config.exclude_patterns:
            if fnmatch(rel_path, pattern):
                return False
            if fnmatch(path.name, pattern):
                return False
        
        return True
    
    # =========================================================================
    # EXPORT EXTRACTION
    # =========================================================================
    
    def extract_exports(self, file_path: Path) -> list[dict]:
        """Extract exported symbols from a Python file."""
        exports = []
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    
                    exports.append({
                        "type": "class",
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno or node.lineno,
                        "docstring": ast.get_docstring(node),
                        "methods": methods,
                        "file": str(file_path),
                    })
                    
                elif isinstance(node, ast.FunctionDef):
                    exports.append({
                        "type": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno or node.lineno,
                        "docstring": ast.get_docstring(node),
                        "args": [a.arg for a in node.args.args],
                        "file": str(file_path),
                    })
                    
                elif isinstance(node, ast.AsyncFunctionDef):
                    exports.append({
                        "type": "async_function",
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno or node.lineno,
                        "docstring": ast.get_docstring(node),
                        "args": [a.arg for a in node.args.args],
                        "file": str(file_path),
                    })
        except (SyntaxError, UnicodeDecodeError):
            pass  # Skip files that can't be parsed
        
        return exports
    
    # =========================================================================
    # SIGNATURE EXTRACTION
    # =========================================================================
    
    def extract_signatures(self, file_path: Path) -> list[dict]:
        """Extract function signatures for task generation."""
        signatures = []
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Build signature string
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            try:
                                arg_str += f": {ast.unparse(arg.annotation)}"
                            except:
                                pass
                        args.append(arg_str)
                    
                    returns = ""
                    if node.returns:
                        try:
                            returns = f" -> {ast.unparse(node.returns)}"
                        except:
                            pass
                    
                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    signature = f"{prefix} {node.name}({', '.join(args)}){returns}"
                    
                    # Check if body is stub
                    is_stub = self._is_stub_body(node.body)
                    
                    # Calculate line count
                    end_line = node.end_lineno or node.lineno
                    line_count = end_line - node.lineno + 1
                    
                    signatures.append({
                        "signature": signature,
                        "name": node.name,
                        "is_stub": is_stub,
                        "line": node.lineno,
                        "end_line": end_line,
                        "line_count": line_count,
                        "file": str(file_path),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "docstring": ast.get_docstring(node),
                    })
        except (SyntaxError, UnicodeDecodeError):
            pass
        
        return signatures
    
    def _is_stub_body(self, body: list) -> bool:
        """Check if function body is a stub."""
        if not body:
            return True
        
        # Filter out docstrings
        non_docstring_body = []
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    continue  # Skip docstring
            non_docstring_body.append(stmt)
        
        if not non_docstring_body:
            return True
        
        if len(non_docstring_body) == 1:
            stmt = non_docstring_body[0]
            
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
    
    # =========================================================================
    # TODO EXTRACTION
    # =========================================================================
    
    def extract_todos(self, file_path: Path) -> list[dict]:
        """Extract TODO/FIXME comments from file."""
        todos = []
        
        try:
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
                        "function": self._find_enclosing_function(content, i),
                    })
        except (UnicodeDecodeError, IOError):
            pass
        
        return todos
    
    def _get_context(self, lines: list[str], line_num: int, window: int) -> str:
        """Get surrounding context for a line."""
        start = max(0, line_num - window - 1)
        end = min(len(lines), line_num + window)
        return '\n'.join(lines[start:end])
    
    def _find_enclosing_function(self, content: str, line_num: int) -> Optional[str]:
        """Find the function that encloses a given line."""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end_line = node.end_lineno or node.lineno
                    if node.lineno <= line_num <= end_line:
                        return node.name
        except:
            pass
        
        return None
    
    # =========================================================================
    # INTENTION EXTRACTION
    # =========================================================================
    
    def extract_intentions(self, file_path: Path) -> list[dict]:
        """Extract comments indicating future intentions."""
        intentions = []
        
        try:
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
                r'#.*\bwill\b.*\bimplement',
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
        except (UnicodeDecodeError, IOError):
            pass
        
        return intentions
    
    # =========================================================================
    # STUB FINDING
    # =========================================================================
    
    def find_stubs(self) -> list[dict]:
        """Find all stub functions in the repository."""
        stubs = []
        
        for file_path in self.scan_file_paths():
            if not file_path.suffix == '.py':
                continue
            
            signatures = self.extract_signatures(file_path)
            for sig in signatures:
                if sig["is_stub"]:
                    stubs.append({
                        **sig,
                        "context": self._get_file_context(file_path, sig["line"]),
                        "imports": self.get_import_context(file_path),
                    })
        
        return stubs
    
    def _get_file_context(self, file_path: Path, center_line: int) -> str:
        """Get context from a file around a specific line."""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            return self._get_context(lines, center_line, self.config.max_context_lines // 2)
        except:
            return ""
    
    # =========================================================================
    # IMPORT EXTRACTION
    # =========================================================================
    
    def get_import_context(self, file_path: Path) -> str:
        """Extract import statements from file."""
        try:
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
                elif imports and not stripped.startswith('#') and stripped:
                    # Non-import, non-empty line after imports
                    break
            
            return '\n'.join(imports)
        except:
            return ""
    
    # =========================================================================
    # USAGE FINDING
    # =========================================================================
    
    def find_usage_examples(self, symbol_name: str, max_examples: int = 5) -> list[str]:
        """Find examples of how a symbol is used."""
        examples = []
        
        for file_path in self.scan_file_paths():
            if not file_path.suffix == '.py':
                continue
            
            try:
                content = file_path.read_text()
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    if symbol_name in line and not line.strip().startswith('#'):
                        # Get surrounding context
                        context = self._get_context(lines, i + 1, 3)
                        examples.append(context)
                        
                        if len(examples) >= max_examples:
                            return examples
            except:
                continue
        
        return examples
    
    # =========================================================================
    # TEST DISCOVERY
    # =========================================================================
    
    def find_untested_functions(self) -> list[dict]:
        """Find functions that don't have corresponding tests."""
        # Get all non-test functions
        all_functions = []
        test_functions = set()
        
        for file_path in self.scan_file_paths():
            if not file_path.suffix == '.py':
                continue
            
            is_test_file = 'test' in file_path.name.lower()
            
            signatures = self.extract_signatures(file_path)
            for sig in signatures:
                if is_test_file:
                    # Track what functions are being tested
                    if sig["name"].startswith("test_"):
                        # Extract function name from test name
                        tested_func = sig["name"][5:]  # Remove "test_"
                        test_functions.add(tested_func)
                else:
                    if not sig["name"].startswith("_"):  # Skip private
                        all_functions.append(sig)
        
        # Find untested functions
        untested = []
        for func in all_functions:
            if func["name"] not in test_functions:
                untested.append(func)
        
        return untested
    
    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================
    
    def find_similar_functions(self, target_signature: dict) -> list[dict]:
        """Find functions similar to the target for pattern matching."""
        similar = []
        target_name = target_signature.get("name", "")
        target_args = len(target_signature.get("args", []))
        
        for file_path in self.scan_file_paths():
            if not file_path.suffix == '.py':
                continue
            
            signatures = self.extract_signatures(file_path)
            for sig in signatures:
                # Skip the target itself
                if sig["file"] == target_signature.get("file") and sig["name"] == target_name:
                    continue
                
                # Check for similarity
                sig_args = len(sig.get("args", []))
                name_similar = self._name_similarity(target_name, sig["name"]) > 0.5
                args_similar = abs(sig_args - target_args) <= 2
                
                if name_similar or args_similar:
                    similar.append(sig)
        
        return similar[:5]  # Return top 5
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Compute simple name similarity."""
        # Split on underscores and compare words
        words1 = set(name1.lower().split('_'))
        words2 = set(name2.lower().split('_'))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

