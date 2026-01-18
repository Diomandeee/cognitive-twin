"""
FunctionGemma Directive Scorer for CognitiveTwin V3.

This module provides FunctionGemma-based directive completeness scoring,
which uses structured tool call parsing to verify that a user directive
can be cleanly mapped to a tool execution.

The parsability score complements the heuristic-based directive_completeness
score by providing a machine-verifiable signal:
- If FunctionGemma can parse the directive into a complete tool call,
  the directive is definitively complete.
- Missing required parameters indicate genuine blocking issues.
- This enables more accurate classification of unjustified clarifications.

Usage:
    from cognitive_twin.v3.corpus_surgery.functiongemma_scorer import (
        FunctionGemmaDirectiveScorer,
        ParsabilityResult,
    )
    
    scorer = FunctionGemmaDirectiveScorer()
    result = await scorer.compute_parsability("Implement binary search in Python")
    print(f"Parsability: {result.score}")
    print(f"Tool: {result.tool_call}")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from cognitive_twin._compat import (
    FunctionCall,
    ToolSchema,
    FunctionGemmaRuntime,
    RuntimeConfig,
    RuntimeBackend,
    GenerationResult,
)

from ..tools.schemas import (
    ToolSchemaRegistry,
    V3ToolSchema,
    V3Domain,
    get_all_schemas,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PARSABILITY RESULT
# =============================================================================


@dataclass
class ParsabilityResult:
    """Result of FunctionGemma parsability scoring."""
    
    # Core score (0.0 - 1.0)
    score: float
    
    # Parsed tool call (if successful)
    tool_call: Optional[FunctionCall] = None
    
    # Tool schema that was matched
    matched_tool: Optional[V3ToolSchema] = None
    
    # Parameter analysis
    required_params: Set[str] = field(default_factory=set)
    provided_params: Set[str] = field(default_factory=set)
    missing_params: Set[str] = field(default_factory=set)
    
    # Metadata
    raw_output: str = ""
    parse_success: bool = False
    confidence: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if the directive is complete (all required params present)."""
        return self.score >= 1.0 and self.parse_success
    
    @property
    def is_parsable(self) -> bool:
        """Check if the directive was parsable at all."""
        return self.tool_call is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "matched_tool": self.matched_tool.name if self.matched_tool else None,
            "required_params": list(self.required_params),
            "provided_params": list(self.provided_params),
            "missing_params": list(self.missing_params),
            "parse_success": self.parse_success,
            "confidence": self.confidence,
        }


# =============================================================================
# FUNCTIONGEMMA DIRECTIVE SCORER
# =============================================================================


class FunctionGemmaDirectiveScorer:
    """
    Use FunctionGemma to verify directive completeness via structured parsing.
    
    This scorer integrates with the existing heuristic-based classifier to
    provide a machine-verifiable signal for directive completeness.
    
    The key insight is: if FunctionGemma can parse a user directive into a
    complete tool call (with all required parameters), then the directive
    is definitively complete and the model should execute immediately.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: RuntimeBackend = RuntimeBackend.AUTO,
        registry: Optional[ToolSchemaRegistry] = None,
        use_mock: bool = False,
    ):
        """
        Initialize the directive scorer.
        
        Args:
            model_path: Path to fine-tuned FunctionGemma model.
                       If None, will attempt to use default or mock.
            backend: Runtime backend to use.
            registry: Tool schema registry. If None, uses default V3 registry.
            use_mock: If True, use mock mode for testing without a model.
        """
        self.registry = registry or ToolSchemaRegistry()
        self.use_mock = use_mock
        self._runtime: Optional[FunctionGemmaRuntime] = None
        self._model_path = model_path
        self._backend = backend
    
    async def _ensure_runtime(self) -> Optional[FunctionGemmaRuntime]:
        """Lazily initialize the runtime."""
        if self._runtime is not None:
            return self._runtime
        
        if self.use_mock:
            return None
        
        if self._model_path is None:
            logger.warning("No model path provided, using mock mode")
            self.use_mock = True
            return None
        
        try:
            config = RuntimeConfig(
                model_path=self._model_path,
                backend=self._backend,
                max_new_tokens=256,
                temperature=0.1,  # Low temp for deterministic parsing
            )
            self._runtime = FunctionGemmaRuntime(config)
            await self._runtime.load()
            return self._runtime
        except Exception as e:
            logger.warning(f"Failed to load FunctionGemma runtime: {e}, using mock mode")
            self.use_mock = True
            return None
    
    async def compute_parsability(
        self,
        user_message: str,
        tools: Optional[List[ToolSchema]] = None,
        domain: Optional[V3Domain] = None,
    ) -> ParsabilityResult:
        """
        Parse directive into tool call and score completeness.
        
        Args:
            user_message: The user's directive message.
            tools: Specific tools to consider. If None, uses registry.
            domain: Filter tools by domain.
        
        Returns:
            ParsabilityResult with score and parsed tool call.
        """
        # Get tools to use
        if tools is None:
            if domain is not None:
                tools = self.registry.get_schemas_by_domain(domain)
            else:
                tools = self.registry.get_schemas()
        
        if not tools:
            return ParsabilityResult(
                score=0.0,
                raw_output="No tools available",
            )
        
        # Use mock mode if no runtime
        if self.use_mock:
            return await self._mock_parse(user_message, tools)
        
        # Ensure runtime is loaded
        runtime = await self._ensure_runtime()
        if runtime is None:
            return await self._mock_parse(user_message, tools)
        
        # Generate tool call
        try:
            result = await runtime.generate_call(
                instruction=user_message,
                tools=tools,
            )
            return self._analyze_result(result, user_message)
        except Exception as e:
            logger.error(f"FunctionGemma generation failed: {e}")
            return ParsabilityResult(
                score=0.0,
                raw_output=str(e),
            )
    
    def compute_parsability_sync(
        self,
        user_message: str,
        tools: Optional[List[ToolSchema]] = None,
        domain: Optional[V3Domain] = None,
    ) -> ParsabilityResult:
        """Synchronous wrapper for compute_parsability."""
        return asyncio.get_event_loop().run_until_complete(
            self.compute_parsability(user_message, tools, domain)
        )
    
    async def _mock_parse(
        self,
        user_message: str,
        tools: List[ToolSchema],
    ) -> ParsabilityResult:
        """
        Mock parsing using keyword matching for testing.
        
        This allows the scorer to work without a loaded model,
        useful for unit tests and development.
        """
        # Find best matching tool from registry
        matched_v3_tool = self.registry.find_best_match(user_message)
        
        if matched_v3_tool is None:
            return ParsabilityResult(
                score=0.0,
                raw_output="No matching tool found (mock mode)",
            )
        
        # Extract mock arguments from message
        mock_args = self._extract_mock_args(user_message, matched_v3_tool)
        
        # Create mock function call
        mock_call = FunctionCall(
            name=matched_v3_tool.name,
            args=mock_args,
        )
        
        # Compute completeness
        provided = set(mock_args.keys())
        required = matched_v3_tool.required_params
        missing = required - provided
        
        if not required:
            score = 1.0
        else:
            score = len(provided & required) / len(required)
        
        return ParsabilityResult(
            score=score,
            tool_call=mock_call,
            matched_tool=matched_v3_tool,
            required_params=required,
            provided_params=provided,
            missing_params=missing,
            raw_output=f"Mock parse: {matched_v3_tool.name}",
            parse_success=True,
            confidence=0.7 if score > 0.5 else 0.3,
        )
    
    def _extract_mock_args(
        self,
        message: str,
        tool: V3ToolSchema,
    ) -> Dict[str, Any]:
        """
        Extract arguments from message for mock mode.
        
        This is a heuristic extraction for testing purposes.
        """
        args: Dict[str, Any] = {}
        message_lower = message.lower()
        
        # Common patterns
        if "name" in tool.required_params:
            # Try to extract a name
            if " function" in message_lower:
                # "implement X function" -> X
                parts = message.split()
                for i, part in enumerate(parts):
                    if part.lower() == "function" and i > 0:
                        args["name"] = parts[i - 1]
                        break
            if "name" not in args:
                args["name"] = "extracted_name"
        
        if "description" in tool.required_params:
            # Use the whole message as description
            args["description"] = message
        
        if "code" in tool.required_params:
            # Check for code block
            if "```" in message:
                import re
                match = re.search(r"```[\w]*\n([\s\S]*?)```", message)
                if match:
                    args["code"] = match.group(1)
            else:
                # Mark as missing
                pass
        
        if "text" in tool.required_params:
            # Use the message as text
            args["text"] = message
        
        if "content" in tool.required_params:
            args["content"] = message
        
        if "goal" in tool.required_params:
            # Try to extract goal
            args["goal"] = message
        
        if "task" in tool.required_params:
            args["task"] = message
        
        if "project" in tool.required_params:
            args["project"] = message
        
        if "fields" in tool.required_params:
            # Default empty list
            args["fields"] = []
        
        if "to_format" in tool.required_params:
            # Try to extract format
            for fmt in ["json", "yaml", "csv", "markdown", "html"]:
                if fmt in message_lower:
                    args["to_format"] = fmt
                    break
        
        return args
    
    def _analyze_result(
        self,
        result: GenerationResult,
        user_message: str,
    ) -> ParsabilityResult:
        """Analyze FunctionGemma generation result."""
        if result.call is None:
            return ParsabilityResult(
                score=0.0,
                raw_output=result.raw_output,
                parse_success=False,
            )
        
        # Find the V3 tool schema
        matched_tool = self.registry.get(result.call.name)
        
        if matched_tool is None:
            # Tool was parsed but not in our registry
            return ParsabilityResult(
                score=0.5,  # Partial credit
                tool_call=result.call,
                raw_output=result.raw_output,
                parse_success=True,
                confidence=result.confidence,
            )
        
        # Compute parameter completeness
        provided = set(result.call.args.keys())
        required = matched_tool.required_params
        missing = required - provided
        
        if not required:
            score = 1.0
        else:
            score = len(provided & required) / len(required)
        
        return ParsabilityResult(
            score=score,
            tool_call=result.call,
            matched_tool=matched_tool,
            required_params=required,
            provided_params=provided,
            missing_params=missing,
            raw_output=result.raw_output,
            parse_success=True,
            confidence=result.confidence,
        )
    
    def fuse_scores(
        self,
        heuristic_score: float,
        parsability_result: ParsabilityResult,
        heuristic_weight: float = 0.4,
        parsability_weight: float = 0.6,
    ) -> float:
        """
        Fuse heuristic and parsability scores.
        
        If FunctionGemma successfully parses the directive with all
        required params, we weight it more heavily since it's a
        machine-verifiable signal.
        
        Args:
            heuristic_score: The traditional directive_completeness score.
            parsability_result: The FunctionGemma parsability result.
            heuristic_weight: Weight for heuristic score.
            parsability_weight: Weight for parsability score.
        
        Returns:
            Fused score between 0.0 and 1.0.
        """
        if not parsability_result.parse_success:
            # If parsing failed, rely more on heuristics
            return heuristic_score
        
        # Weight by confidence
        effective_parsability = parsability_result.score * parsability_result.confidence
        
        # Fuse scores
        fused = (
            heuristic_weight * heuristic_score +
            parsability_weight * effective_parsability
        )
        
        # Normalize
        total_weight = heuristic_weight + parsability_weight
        return min(1.0, max(0.0, fused / total_weight))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def compute_parsability_for_message(
    user_message: str,
    model_path: Optional[str] = None,
    use_mock: bool = True,
) -> ParsabilityResult:
    """
    Convenience function to compute parsability for a single message.
    
    Args:
        user_message: The user directive to analyze.
        model_path: Path to FunctionGemma model (optional).
        use_mock: Use mock mode (default True for convenience).
    
    Returns:
        ParsabilityResult with score and tool call.
    """
    scorer = FunctionGemmaDirectiveScorer(
        model_path=model_path,
        use_mock=use_mock,
    )
    return await scorer.compute_parsability(user_message)


def compute_parsability_sync(user_message: str) -> ParsabilityResult:
    """Synchronous convenience function using mock mode."""
    scorer = FunctionGemmaDirectiveScorer(use_mock=True)
    return scorer.compute_parsability_sync(user_message)




