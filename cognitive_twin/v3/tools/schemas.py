"""
V3-specific tool schema definitions for FunctionGemma integration.

These tool schemas map to the task types and domains used in CognitiveTwin V3
training data. Each tool represents a common directive pattern that the model
should learn to execute immediately without asking permission.

Tool Categories:
- Code: implement, refactor, fix, test, debug
- Content: rewrite, summarize, extract, transform
- Planning: plan, define, roadmap
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from cognitive_twin._compat import ToolSchema, ToolLayer


# =============================================================================
# V3 Tool Schema Extension
# =============================================================================


class V3Domain(str, Enum):
    """Domain classification for V3 tools."""
    
    CODE = "code"
    CONTENT = "content"
    PLANNING = "planning"
    RESEARCH = "research"
    OPS = "ops"


@dataclass
class V3ToolSchema:
    """Extended tool schema with V3-specific metadata."""
    
    schema: ToolSchema
    domain: V3Domain
    task_type: str  # Maps to V3 TaskType enum
    required_params: Set[str] = field(default_factory=set)
    optional_params: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        return self.schema.name
    
    @property
    def description(self) -> str:
        return self.schema.description
    
    def get_required_param_names(self) -> Set[str]:
        """Get names of required parameters."""
        return self.required_params
    
    def compute_completeness(self, provided_args: Dict[str, Any]) -> float:
        """
        Compute how complete the arguments are.
        
        Returns 1.0 if all required params are present, 0.0 if none.
        """
        if not self.required_params:
            return 1.0
        
        provided = set(provided_args.keys())
        matched = provided & self.required_params
        return len(matched) / len(self.required_params)


# =============================================================================
# CODE TOOLS
# =============================================================================


IMPLEMENT_FUNCTION = V3ToolSchema(
    schema=ToolSchema(
        name="implement_function",
        description="Implement a function or method based on the given specification",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the function to implement"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (python, typescript, rust, etc.)"
                },
                "signature": {
                    "type": "string",
                    "description": "Function signature if specified"
                },
                "description": {
                    "type": "string",
                    "description": "What the function should do"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any constraints or requirements"
                }
            },
            "required": ["name", "description"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.CODE,
    task_type="implement",
    required_params={"name", "description"},
    optional_params={"language", "signature", "constraints"},
    examples=[
        "Implement a binary search function",
        "Create a function that validates email addresses",
        "Write a function to parse JSON with error handling",
    ]
)


REFACTOR_CODE = V3ToolSchema(
    schema=ToolSchema(
        name="refactor_code",
        description="Refactor existing code to improve quality, readability, or performance",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to refactor"
                },
                "goal": {
                    "type": "string",
                    "description": "What aspect to improve (readability, performance, etc.)"
                },
                "preserve_behavior": {
                    "type": "boolean",
                    "description": "Whether to preserve exact behavior",
                    "default": True
                },
                "style_guide": {
                    "type": "string",
                    "description": "Style guide to follow (PEP8, Google, etc.)"
                }
            },
            "required": ["code", "goal"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.CODE,
    task_type="refactor",
    required_params={"code", "goal"},
    optional_params={"preserve_behavior", "style_guide"},
    examples=[
        "Refactor this function to use async/await",
        "Clean up this code to follow PEP8",
        "Optimize this loop for better performance",
    ]
)


FIX_BUG = V3ToolSchema(
    schema=ToolSchema(
        name="fix_bug",
        description="Fix a bug in the provided code",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code containing the bug"
                },
                "error_message": {
                    "type": "string",
                    "description": "Error message or symptoms"
                },
                "expected_behavior": {
                    "type": "string",
                    "description": "What the code should do"
                },
                "actual_behavior": {
                    "type": "string",
                    "description": "What the code actually does"
                }
            },
            "required": ["code"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.CODE,
    task_type="debug",
    required_params={"code"},
    optional_params={"error_message", "expected_behavior", "actual_behavior"},
    examples=[
        "Fix the bug in this function that causes infinite loop",
        "Debug this code - it returns None unexpectedly",
        "Fix the TypeError in this code",
    ]
)


WRITE_TESTS = V3ToolSchema(
    schema=ToolSchema(
        name="write_tests",
        description="Write unit tests for the provided code",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to test"
                },
                "framework": {
                    "type": "string",
                    "description": "Testing framework (pytest, jest, etc.)"
                },
                "coverage": {
                    "type": "string",
                    "enum": ["minimal", "standard", "comprehensive"],
                    "description": "Level of test coverage",
                    "default": "standard"
                },
                "include_edge_cases": {
                    "type": "boolean",
                    "description": "Whether to include edge case tests",
                    "default": True
                }
            },
            "required": ["code"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.CODE,
    task_type="implement",
    required_params={"code"},
    optional_params={"framework", "coverage", "include_edge_cases"},
    examples=[
        "Write pytest tests for this function",
        "Create comprehensive unit tests for this class",
        "Add edge case tests for the validation logic",
    ]
)


DEBUG_CODE = V3ToolSchema(
    schema=ToolSchema(
        name="debug_code",
        description="Analyze code to identify and explain issues",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to debug"
                },
                "symptoms": {
                    "type": "string",
                    "description": "Description of the problem"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about the environment"
                }
            },
            "required": ["code"]
        },
        layer=ToolLayer.L1_INTERPRET,
    ),
    domain=V3Domain.CODE,
    task_type="debug",
    required_params={"code"},
    optional_params={"symptoms", "context"},
    examples=[
        "Debug this code - it hangs after 10 iterations",
        "Analyze why this function is slow",
        "Find the memory leak in this code",
    ]
)


# =============================================================================
# CONTENT TOOLS
# =============================================================================


REWRITE_TEXT = V3ToolSchema(
    schema=ToolSchema(
        name="rewrite_text",
        description="Rewrite text with specified modifications",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to rewrite"
                },
                "style": {
                    "type": "string",
                    "description": "Target style (formal, casual, technical, etc.)"
                },
                "preserve_meaning": {
                    "type": "boolean",
                    "description": "Whether to preserve exact meaning",
                    "default": True
                },
                "length": {
                    "type": "string",
                    "enum": ["shorter", "same", "longer"],
                    "description": "Target length relative to original"
                }
            },
            "required": ["text"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.CONTENT,
    task_type="rewrite",
    required_params={"text"},
    optional_params={"style", "preserve_meaning", "length"},
    examples=[
        "Rewrite this in a more formal tone",
        "Make this text more concise",
        "Rewrite for a technical audience",
    ]
)


SUMMARIZE = V3ToolSchema(
    schema=ToolSchema(
        name="summarize",
        description="Summarize the provided content",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to summarize"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length in words or sentences"
                },
                "format": {
                    "type": "string",
                    "enum": ["paragraph", "bullets", "numbered"],
                    "description": "Output format"
                },
                "focus": {
                    "type": "string",
                    "description": "What aspects to focus on"
                }
            },
            "required": ["content"]
        },
        layer=ToolLayer.L1_INTERPRET,
    ),
    domain=V3Domain.CONTENT,
    task_type="explain",
    required_params={"content"},
    optional_params={"max_length", "format", "focus"},
    examples=[
        "Summarize this article in 3 sentences",
        "Give me the key points as bullets",
        "Summarize focusing on technical details",
    ]
)


EXTRACT_DATA = V3ToolSchema(
    schema=ToolSchema(
        name="extract_data",
        description="Extract structured data from unstructured text",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract from"
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to extract"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv", "table"],
                    "description": "Output format",
                    "default": "json"
                }
            },
            "required": ["text", "fields"]
        },
        layer=ToolLayer.L1_INTERPRET,
    ),
    domain=V3Domain.CONTENT,
    task_type="rewrite",
    required_params={"text", "fields"},
    optional_params={"output_format"},
    examples=[
        "Extract names and emails from this text",
        "Parse the dates and amounts into JSON",
        "Extract all URLs as a list",
    ]
)


TRANSFORM_FORMAT = V3ToolSchema(
    schema=ToolSchema(
        name="transform_format",
        description="Transform content from one format to another",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to transform"
                },
                "from_format": {
                    "type": "string",
                    "description": "Source format"
                },
                "to_format": {
                    "type": "string",
                    "description": "Target format"
                },
                "options": {
                    "type": "object",
                    "description": "Format-specific options"
                }
            },
            "required": ["content", "to_format"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.CONTENT,
    task_type="rewrite",
    required_params={"content", "to_format"},
    optional_params={"from_format", "options"},
    examples=[
        "Convert this JSON to YAML",
        "Transform this markdown to HTML",
        "Convert CSV to JSON array",
    ]
)


# =============================================================================
# PLANNING TOOLS
# =============================================================================


CREATE_PLAN = V3ToolSchema(
    schema=ToolSchema(
        name="create_plan",
        description="Create a plan for accomplishing a goal",
        parameters={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal to plan for"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any constraints to consider"
                },
                "timeline": {
                    "type": "string",
                    "description": "Desired timeline"
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["high-level", "detailed", "comprehensive"],
                    "default": "detailed"
                }
            },
            "required": ["goal"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.PLANNING,
    task_type="design",
    required_params={"goal"},
    optional_params={"constraints", "timeline", "detail_level"},
    examples=[
        "Create a plan to migrate the database",
        "Plan the API refactoring project",
        "Design a testing strategy for the new feature",
    ]
)


DEFINE_STEPS = V3ToolSchema(
    schema=ToolSchema(
        name="define_steps",
        description="Define step-by-step instructions for a task",
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to break down"
                },
                "audience": {
                    "type": "string",
                    "description": "Who will follow these steps"
                },
                "include_verification": {
                    "type": "boolean",
                    "description": "Include verification steps",
                    "default": True
                }
            },
            "required": ["task"]
        },
        layer=ToolLayer.L1_INTERPRET,
    ),
    domain=V3Domain.PLANNING,
    task_type="design",
    required_params={"task"},
    optional_params={"audience", "include_verification"},
    examples=[
        "Define steps to set up the development environment",
        "Break down the deployment process into steps",
        "List the steps to debug a production issue",
    ]
)


GENERATE_ROADMAP = V3ToolSchema(
    schema=ToolSchema(
        name="generate_roadmap",
        description="Generate a project roadmap with milestones",
        parameters={
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project description"
                },
                "duration": {
                    "type": "string",
                    "description": "Total duration"
                },
                "milestones": {
                    "type": "integer",
                    "description": "Number of milestones to include"
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Known dependencies"
                }
            },
            "required": ["project"]
        },
        layer=ToolLayer.L2_STRUCTURE,
    ),
    domain=V3Domain.PLANNING,
    task_type="design",
    required_params={"project"},
    optional_params={"duration", "milestones", "dependencies"},
    examples=[
        "Create a roadmap for the Q1 feature release",
        "Generate a 6-month product roadmap",
        "Plan the migration project timeline",
    ]
)


# =============================================================================
# TOOL REGISTRY
# =============================================================================


class ToolSchemaRegistry:
    """Registry of all V3 tool schemas."""
    
    def __init__(self):
        self._tools: Dict[str, V3ToolSchema] = {}
        self._by_domain: Dict[V3Domain, List[V3ToolSchema]] = {
            domain: [] for domain in V3Domain
        }
        self._register_defaults()
    
    def _register_defaults(self):
        """Register all default tools."""
        default_tools = [
            # Code
            IMPLEMENT_FUNCTION,
            REFACTOR_CODE,
            FIX_BUG,
            WRITE_TESTS,
            DEBUG_CODE,
            # Content
            REWRITE_TEXT,
            SUMMARIZE,
            EXTRACT_DATA,
            TRANSFORM_FORMAT,
            # Planning
            CREATE_PLAN,
            DEFINE_STEPS,
            GENERATE_ROADMAP,
        ]
        
        for tool in default_tools:
            self.register(tool)
    
    def register(self, tool: V3ToolSchema) -> None:
        """Register a tool schema."""
        self._tools[tool.name] = tool
        self._by_domain[tool.domain].append(tool)
    
    def get(self, name: str) -> Optional[V3ToolSchema]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def get_by_domain(self, domain: V3Domain) -> List[V3ToolSchema]:
        """Get all tools for a domain."""
        return self._by_domain.get(domain, [])
    
    def get_all(self) -> List[V3ToolSchema]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas (for FunctionGemma)."""
        return [tool.schema for tool in self._tools.values()]
    
    def get_schemas_by_domain(self, domain: V3Domain) -> List[ToolSchema]:
        """Get tool schemas for a specific domain."""
        return [tool.schema for tool in self._by_domain.get(domain, [])]
    
    def find_best_match(self, directive: str) -> Optional[V3ToolSchema]:
        """
        Find the best matching tool for a directive based on keywords.
        
        This is a simple heuristic matcher - FunctionGemma does the real work.
        """
        directive_lower = directive.lower()
        
        # Simple keyword matching
        keyword_map = {
            "implement": IMPLEMENT_FUNCTION,
            "create function": IMPLEMENT_FUNCTION,
            "write function": IMPLEMENT_FUNCTION,
            "refactor": REFACTOR_CODE,
            "clean up": REFACTOR_CODE,
            "fix bug": FIX_BUG,
            "debug": DEBUG_CODE,
            "test": WRITE_TESTS,
            "unit test": WRITE_TESTS,
            "rewrite": REWRITE_TEXT,
            "summarize": SUMMARIZE,
            "summary": SUMMARIZE,
            "extract": EXTRACT_DATA,
            "convert": TRANSFORM_FORMAT,
            "transform": TRANSFORM_FORMAT,
            "plan": CREATE_PLAN,
            "steps": DEFINE_STEPS,
            "how to": DEFINE_STEPS,
            "roadmap": GENERATE_ROADMAP,
        }
        
        for keyword, tool in keyword_map.items():
            if keyword in directive_lower:
                return tool
        
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global registry instance
_registry = ToolSchemaRegistry()


def get_all_tools() -> List[V3ToolSchema]:
    """Get all registered tools."""
    return _registry.get_all()


def get_tools_by_domain(domain: V3Domain) -> List[V3ToolSchema]:
    """Get tools for a specific domain."""
    return _registry.get_by_domain(domain)


def get_tool_by_name(name: str) -> Optional[V3ToolSchema]:
    """Get a tool by name."""
    return _registry.get(name)


def get_all_schemas() -> List[ToolSchema]:
    """Get all tool schemas for FunctionGemma."""
    return _registry.get_schemas()




