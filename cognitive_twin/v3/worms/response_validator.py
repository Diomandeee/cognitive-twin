"""
Response Validator for Repo Worm.

Validates Codex responses for:
- Syntax correctness
- Compilation
- Pattern matching
- No new dependencies
- No questions
"""

import ast
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .task_types import ParsedResponse, Task


class ResponseValidator:
    """Validates Codex responses."""
    
    def __init__(self):
        pass
    
    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================
    
    def parse_codex_response(self, response: str) -> ParsedResponse:
        """Parse a Codex response."""
        result = ParsedResponse(raw_response=response)
        
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
        
        # Detect language
        result.language = self._detect_language(response)
        
        # Initial validity check
        result.is_valid = bool(result.code or result.diff)
        
        return result
    
    def extract_code_output(self, response: str) -> Optional[str]:
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
        
        # Look for any code block with language specifier
        code_pattern = r'```\w+\n([\s\S]*?)\n```'
        match = re.search(code_pattern, response)
        
        if match:
            return match.group(1)
        
        return None
    
    def extract_diff_output(self, response: str) -> Optional[str]:
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
                    if not (line.startswith('+') or line.startswith('-') or 
                            line.startswith('@') or line.startswith(' ') or
                            line.startswith('---') or line.startswith('+++')):
                        if len(diff_lines) > 3:
                            break
            
            return '\n'.join(diff_lines) if diff_lines else None
        
        return None
    
    def _detect_language(self, response: str) -> str:
        """Detect the programming language from response."""
        # Check code block language specifier
        lang_pattern = r'```(\w+)\n'
        match = re.search(lang_pattern, response)
        
        if match:
            lang = match.group(1).lower()
            if lang in ['python', 'py']:
                return 'python'
            elif lang in ['typescript', 'ts']:
                return 'typescript'
            elif lang in ['javascript', 'js']:
                return 'javascript'
            elif lang == 'rust':
                return 'rust'
            elif lang == 'go':
                return 'go'
        
        return 'python'  # Default
    
    # =========================================================================
    # SYNTAX VALIDATION
    # =========================================================================
    
    def validate_syntax(self, code: str, language: str = "python") -> tuple[bool, str]:
        """Validate code syntax."""
        if language == "python":
            try:
                ast.parse(code)
                return True, ""
            except SyntaxError as e:
                return False, str(e)
        
        elif language == "typescript":
            return self._validate_typescript_syntax(code)
        
        elif language == "javascript":
            return self._validate_javascript_syntax(code)
        
        return True, ""  # Assume valid for unknown languages
    
    def _validate_typescript_syntax(self, code: str) -> tuple[bool, str]:
        """Validate TypeScript syntax using tsc."""
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False, mode='w') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['tsc', '--noEmit', temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr
        except FileNotFoundError:
            return True, ""  # tsc not available, assume valid
        except subprocess.TimeoutExpired:
            return False, "Validation timeout"
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def _validate_javascript_syntax(self, code: str) -> tuple[bool, str]:
        """Validate JavaScript syntax using node --check."""
        with tempfile.NamedTemporaryFile(suffix='.js', delete=False, mode='w') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['node', '--check', temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr
        except FileNotFoundError:
            return True, ""  # node not available, assume valid
        except subprocess.TimeoutExpired:
            return False, "Validation timeout"
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    # =========================================================================
    # COMPILATION VALIDATION
    # =========================================================================
    
    def validate_compiles(self, code: str, file_path: Path) -> bool:
        """Validate that generated code compiles."""
        suffix = file_path.suffix if file_path else '.py'
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='w') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            if suffix == '.py':
                result = subprocess.run(
                    ['python', '-m', 'py_compile', temp_path],
                    capture_output=True,
                    timeout=30,
                )
                return result.returncode == 0
            
            elif suffix == '.ts':
                result = subprocess.run(
                    ['tsc', '--noEmit', temp_path],
                    capture_output=True,
                    timeout=30,
                )
                return result.returncode == 0
            
            elif suffix == '.rs':
                # For Rust, we'd need to check in context of a cargo project
                return True
            
            elif suffix == '.go':
                result = subprocess.run(
                    ['go', 'build', '-o', '/dev/null', temp_path],
                    capture_output=True,
                    timeout=30,
                )
                return result.returncode == 0
            
            return True  # Assume valid if no validator
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return True  # Assume valid if tools not available
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    # =========================================================================
    # PATTERN VALIDATION
    # =========================================================================
    
    def validate_patterns(self, code: str, reference_code: str) -> bool:
        """Validate that code matches existing patterns."""
        checks = [
            self._check_indentation_match(code, reference_code),
            self._check_naming_conventions(code, reference_code),
            self._check_docstring_style(code, reference_code),
        ]
        
        return all(checks)
    
    def _check_indentation_match(self, code: str, reference: str) -> bool:
        """Check if indentation matches reference."""
        ref_indent = self._detect_indentation(reference)
        code_indent = self._detect_indentation(code)
        return ref_indent == code_indent
    
    def _detect_indentation(self, code: str) -> str:
        """Detect indentation style (spaces or tabs, width)."""
        lines = code.split('\n')
        
        for line in lines:
            stripped = line.lstrip()
            if stripped and line != stripped:
                indent = line[:len(line) - len(stripped)]
                if '\t' in indent:
                    return 'tabs'
                else:
                    return f'spaces_{len(indent)}'
        
        return 'spaces_4'  # Default
    
    def _check_naming_conventions(self, code: str, reference: str) -> bool:
        """Check if naming conventions match."""
        # Simple check: snake_case vs camelCase
        ref_snake = bool(re.search(r'def [a-z]+_[a-z]+', reference))
        code_snake = bool(re.search(r'def [a-z]+_[a-z]+', code))
        
        ref_camel = bool(re.search(r'def [a-z]+[A-Z]', reference))
        code_camel = bool(re.search(r'def [a-z]+[A-Z]', code))
        
        if ref_snake and code_camel:
            return False
        if ref_camel and code_snake:
            return False
        
        return True
    
    def _check_docstring_style(self, code: str, reference: str) -> bool:
        """Check if docstring style matches."""
        # Simple check: has docstrings or not
        ref_has_docstring = '"""' in reference or "'''" in reference
        code_has_docstring = '"""' in code or "'''" in code
        
        # If reference has docstrings, code should too
        if ref_has_docstring and not code_has_docstring:
            return False
        
        return True
    
    # =========================================================================
    # DEPENDENCY VALIDATION
    # =========================================================================
    
    def validate_no_new_dependencies(
        self,
        code: str,
        existing_imports: set[str],
    ) -> tuple[bool, list[str]]:
        """Check that no new dependencies are introduced."""
        # Extract imports from generated code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True, []  # Can't parse, assume OK
        
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
        stdlib = {
            'os', 'sys', 're', 'json', 'typing', 'dataclasses', 
            'collections', 'functools', 'itertools', 'pathlib',
            'datetime', 'time', 'uuid', 'hashlib', 'logging',
            'asyncio', 'concurrent', 'threading', 'multiprocessing',
            'io', 'tempfile', 'subprocess', 'shutil', 'glob',
            'math', 'random', 'statistics', 'decimal', 'fractions',
            'copy', 'pickle', 'shelve', 'dbm', 'sqlite3',
            'urllib', 'http', 'email', 'html', 'xml',
            'unittest', 'doctest', 'pdb', 'profile', 'timeit',
            'abc', 'contextlib', 'atexit', 'traceback', 'warnings',
            'inspect', 'dis', 'ast', 'types', 'gc', 'weakref',
            'enum', 'graphlib', 'operator', 'builtins',
            'string', 'textwrap', 'unicodedata', 'codecs',
            'struct', 'array', 'queue', 'heapq', 'bisect',
            'base64', 'binascii', 'quopri', 'uu',
            'secrets', 'hmac', 'ssl', 'socket', 'select',
            'argparse', 'getopt', 'optparse', 'configparser',
            'csv', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma',
        }
        
        truly_new = added - stdlib
        
        return len(truly_new) == 0, list(truly_new)
    
    # =========================================================================
    # QUESTION VALIDATION
    # =========================================================================
    
    def validate_no_questions(self, response: str) -> bool:
        """Validate that response doesn't ask questions."""
        # Import stall score from corpus surgery
        try:
            from ..corpus_surgery.classifier import compute_stall_score
            stall_score = compute_stall_score(response)
            return stall_score < 3
        except ImportError:
            # Fallback: simple pattern matching
            question_patterns = [
                r'would you like',
                r'do you want',
                r'should i',
                r'shall i',
                r'can you confirm',
                r'please confirm',
                r'let me know if',
            ]
            
            response_lower = response.lower()
            for pattern in question_patterns:
                if re.search(pattern, response_lower):
                    return False
            
            # Check if ends with question mark
            if response.rstrip().endswith('?'):
                return False
            
            return True
    
    # =========================================================================
    # FULL VALIDATION
    # =========================================================================
    
    def validate_response(
        self,
        response: ParsedResponse,
        task: Task,
        reference_code: str = "",
        existing_imports: set[str] = None,
    ) -> ParsedResponse:
        """Validate a parsed Codex response."""
        errors = []
        existing_imports = existing_imports or set()
        
        # Check syntax
        if response.code:
            is_valid, error = self.validate_syntax(response.code, response.language)
            if not is_valid:
                errors.append(f"Syntax error: {error}")
        
        # Check compilation
        if response.code and not errors:
            file_path = Path(getattr(task, 'target_file', '') or 
                           getattr(task, 'file', '') or 'temp.py')
            compiles = self.validate_compiles(response.code, file_path)
            if not compiles:
                errors.append("Code does not compile")
        
        # Check patterns
        if response.code and reference_code and not errors:
            matches = self.validate_patterns(response.code, reference_code)
            if not matches:
                errors.append("Code does not match existing patterns")
        
        # Check no new dependencies
        if response.code and not errors:
            no_new, added = self.validate_no_new_dependencies(
                response.code, existing_imports
            )
            if not no_new:
                errors.append(f"New dependencies added: {added}")
        
        # Check no questions
        if not self.validate_no_questions(response.explanation or ""):
            errors.append("Response contains questions")
        
        response.errors = errors
        response.is_valid = len(errors) == 0
        
        return response

