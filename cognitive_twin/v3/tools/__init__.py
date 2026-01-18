"""
CognitiveTwin V3 Tool Schemas.

This module provides tool schema definitions for FunctionGemma integration
with the CognitiveTwin V3 training pipeline.

Tool Categories:
- Code tools: implement_function, refactor_code, fix_bug, write_tests
- Content tools: rewrite_text, summarize, extract_data
- Planning tools: create_plan, define_steps, generate_roadmap
"""

from .schemas import (
    # Tool schemas
    ToolSchemaRegistry,
    V3ToolSchema,
    # Code tools
    IMPLEMENT_FUNCTION,
    REFACTOR_CODE,
    FIX_BUG,
    WRITE_TESTS,
    DEBUG_CODE,
    # Content tools
    REWRITE_TEXT,
    SUMMARIZE,
    EXTRACT_DATA,
    TRANSFORM_FORMAT,
    # Planning tools
    CREATE_PLAN,
    DEFINE_STEPS,
    GENERATE_ROADMAP,
    # Utility functions
    get_all_tools,
    get_tools_by_domain,
    get_tool_by_name,
)

__all__ = [
    "ToolSchemaRegistry",
    "V3ToolSchema",
    # Code tools
    "IMPLEMENT_FUNCTION",
    "REFACTOR_CODE",
    "FIX_BUG",
    "WRITE_TESTS",
    "DEBUG_CODE",
    # Content tools
    "REWRITE_TEXT",
    "SUMMARIZE",
    "EXTRACT_DATA",
    "TRANSFORM_FORMAT",
    # Planning tools
    "CREATE_PLAN",
    "DEFINE_STEPS",
    "GENERATE_ROADMAP",
    # Utility functions
    "get_all_tools",
    "get_tools_by_domain",
    "get_tool_by_name",
]




