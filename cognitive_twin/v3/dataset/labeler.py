"""
Policy Labeler for CognitiveTwin V3 Dataset.

Labels prompts with:
- Directive completeness score
- Question policy
- Format constraints
- Domain detection
- FunctionGemma parsability (NEW)

The FunctionGemma-enhanced labeler uses structured tool call parsing to
provide a machine-verifiable signal for directive completeness.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..schema import (
    Domain,
    QuestionPolicy,
    FormatConstraints,
    PromptClass,
)
from ..corpus_surgery.functiongemma_scorer import (
    FunctionGemmaDirectiveScorer,
    ParsabilityResult,
)
from ..tools.schemas import V3Domain


# =============================================================================
# LABELS DATACLASS
# =============================================================================

@dataclass
class ParsabilityLabels:
    """FunctionGemma parsability labels."""
    
    parsability_score: float = 0.0
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    required_params: Set[str] = field(default_factory=set)
    provided_params: Set[str] = field(default_factory=set)
    missing_params: Set[str] = field(default_factory=set)
    parse_success: bool = False
    
    def to_dict(self) -> dict:
        return {
            "parsability_score": self.parsability_score,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "required_params": list(self.required_params),
            "provided_params": list(self.provided_params),
            "missing_params": list(self.missing_params),
            "parse_success": self.parse_success,
        }


@dataclass
class Labels:
    """Labels for a user message."""
    
    directive_completeness: float
    question_policy: QuestionPolicy
    format_constraints: FormatConstraints
    must_not_omit: bool
    prompt_class: PromptClass
    domain: Domain
    
    # FunctionGemma parsability (NEW)
    parsability: Optional[ParsabilityLabels] = None
    fused_completeness: float = 0.0  # Combined heuristic + parsability
    
    def to_dict(self) -> dict:
        return {
            "directive_completeness": self.directive_completeness,
            "question_policy": self.question_policy.value,
            "format_constraints": self.format_constraints.to_dict() if hasattr(self.format_constraints, 'to_dict') else {},
            "must_not_omit": self.must_not_omit,
            "prompt_class": self.prompt_class.value,
            "domain": self.domain.value,
            "parsability": self.parsability.to_dict() if self.parsability else None,
            "fused_completeness": self.fused_completeness,
        }


# =============================================================================
# DIRECTIVE COMPLETENESS LABELER
# =============================================================================

class DirectiveCompletenessLabeler:
    """Compute directive_completeness score for prompts."""
    
    # Imperative verbs that indicate clear directives
    IMPERATIVE_VERBS = [
        "rewrite", "generate", "implement", "create", "build",
        "write", "return", "extract", "convert", "transform",
        "refactor", "fix", "update", "add", "remove", "delete",
        "change", "modify", "replace", "debug", "test", "analyze",
        "explain", "summarize", "list", "show", "find", "search",
        "design", "plan", "define", "describe", "document",
    ]
    
    # Format specification patterns
    FORMAT_PATTERNS = [
        r"in json",
        r"as json",
        r"return(?:ing)? json",
        r"as csv",
        r"in csv",
        r"as markdown",
        r"in markdown",
        r"don'?t omit",
        r"exact(?:ly)?",
        r"no bullet",
        r"numbered list",
        r"as code",
        r"in python",
        r"in typescript",
        r"in rust",
        r"in javascript",
    ]
    
    # Missing input indicators
    MISSING_INPUT_PATTERNS = [
        r"(?:the |this |that )?code",
        r"(?:the |this |that )?file",
        r"(?:the |this |that )?function",
        r"(?:the |this |that )?class",
    ]
    
    def compute(self, user_message: str, context: Optional[dict] = None) -> float:
        """Compute directive completeness score."""
        context = context or {}
        score = 0.0
        user_lower = user_message.lower()
        
        # +0.35 for imperative verb
        if self._has_imperative_verb(user_lower):
            score += 0.35
        
        # +0.25 for format specification
        if self._has_format_specification(user_lower):
            score += 0.25
        
        # +0.20 for complete inputs
        if self._has_required_inputs(user_message, context):
            score += 0.20
        
        # -0.40 for missing required inputs
        if self._missing_required_inputs(user_message, context):
            score -= 0.40
        
        # -0.20 for material ambiguity
        if self._has_material_ambiguity(user_lower):
            score -= 0.20
        
        # +0.20 for substantial content
        if len(user_message) > 200:
            score += 0.20
        
        return max(0.0, min(1.0, score))
    
    def _has_imperative_verb(self, text: str) -> bool:
        """Check for imperative verb at start or in command position."""
        for verb in self.IMPERATIVE_VERBS:
            # At start of sentence
            if re.search(rf'^{verb}\b', text):
                return True
            # After "please" or "can you"
            if re.search(rf'(?:please|can you)\s+{verb}\b', text):
                return True
            # After colon (in commands)
            if re.search(rf':\s*{verb}\b', text):
                return True
        
        return False
    
    def _has_format_specification(self, text: str) -> bool:
        """Check for format specification."""
        for pattern in self.FORMAT_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
    
    def _has_required_inputs(self, text: str, context: dict) -> bool:
        """Check if required inputs are present."""
        # Check for code block in message
        has_code = bool(re.search(r"```[\s\S]*?```", text))
        
        # Check for file path
        has_file_path = bool(re.search(r"[/\\][\w./\\]+\.\w+", text))
        
        # Check for substantial text (> 200 chars)
        has_long_text = len(text) > 200
        
        # Check for attachments in context
        has_attachments = bool(context.get("attachments"))
        
        return has_code or has_file_path or has_long_text or has_attachments
    
    def _missing_required_inputs(self, text: str, context: dict) -> bool:
        """Check if required inputs are missing."""
        text_lower = text.lower()
        
        # Check for transformation words without input
        transformation_words = [
            "refactor", "rewrite", "transform", "convert",
            "enhance", "improve", "fix", "update"
        ]
        
        needs_input = any(word in text_lower for word in transformation_words)
        
        if needs_input and not self._has_required_inputs(text, context):
            # Check if it references something vague
            for pattern in self.MISSING_INPUT_PATTERNS:
                if re.search(pattern, text_lower):
                    return True
        
        return False
    
    def _has_material_ambiguity(self, text: str) -> bool:
        """Check for material ambiguity."""
        ambiguity_patterns = [
            r"this or that",
            r"either.+or",
            r"what (?:should|would)",
            r"which (?:one|approach|method)",
            r"how should i",
        ]
        
        return any(re.search(p, text) for p in ambiguity_patterns)


# =============================================================================
# QUESTION POLICY LABELER
# =============================================================================

class QuestionPolicyLabeler:
    """Determine question policy based on context."""
    
    # Phase -> default policy mapping
    PHASE_POLICIES = {
        0: QuestionPolicy.QUESTIONS_IF_REQUIRED,  # Opening
        1: QuestionPolicy.QUESTIONS_IF_REQUIRED,  # Context
        2: QuestionPolicy.NO_QUESTIONS,           # Solution
        3: QuestionPolicy.NO_QUESTIONS,           # Refinement
        4: QuestionPolicy.NO_QUESTIONS,           # Synthesis
        5: QuestionPolicy.NO_QUESTIONS,           # Conclusion
    }
    
    def compute(
        self,
        phase_id: int,
        directive_completeness: float,
        user_message: str,
    ) -> QuestionPolicy:
        """Determine question policy."""
        # Check for explicit permission in user message
        if self._user_asked_for_options(user_message):
            return QuestionPolicy.QUESTIONS_ALLOWED
        
        # High directive completeness -> no questions
        if directive_completeness >= 0.7:
            return QuestionPolicy.NO_QUESTIONS
        
        # Low directive completeness -> questions if required
        if directive_completeness < 0.4:
            return QuestionPolicy.QUESTIONS_IF_REQUIRED
        
        # Medium completeness -> use phase default
        return self.PHASE_POLICIES.get(phase_id, QuestionPolicy.NO_QUESTIONS)
    
    def _user_asked_for_options(self, text: str) -> bool:
        """Check if user explicitly asked for options."""
        patterns = [
            r"what (?:are )?(?:my |the )?options",
            r"give me (?:some )?options",
            r"list (?:the |some )?options",
            r"what (?:could|can|should) i",
            r"what do you (?:think|suggest|recommend)",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)


# =============================================================================
# FORMAT CONSTRAINTS LABELER
# =============================================================================

class FormatConstraintsLabeler:
    """Extract format constraints from user message."""
    
    def extract(self, user_message: str) -> FormatConstraints:
        """Extract format constraints."""
        constraints = FormatConstraints()
        user_lower = user_message.lower()
        
        # Forbid bullets
        if any(p in user_lower for p in [
            "no bullet", "don't use bullet", "without bullet",
            "avoid bullet", "not bullet"
        ]):
            constraints.forbid_bullets = True
        
        # Require numbered
        if any(p in user_lower for p in [
            "numbered list", "numbered steps", "number them",
            "use numbers", "with numbers"
        ]):
            constraints.require_numbered = True
        
        # Must return code
        if any(p in user_lower for p in [
            "in code", "write code", "implement", "as code",
            "function", "class", "method"
        ]):
            constraints.must_return_code = True
        
        # Must return diff
        if any(p in user_lower for p in [
            "as diff", "in diff", "show diff", "unified diff"
        ]):
            constraints.must_return_diff = True
        
        # Must return JSON
        if any(p in user_lower for p in [
            "as json", "in json", "json format", "return json"
        ]):
            constraints.must_return_json = True
        
        return constraints


# =============================================================================
# DOMAIN DETECTOR
# =============================================================================

class DomainDetector:
    """Detect the domain of a prompt."""
    
    CODE_PATTERNS = [
        r"```", r"\bfunction\b", r"\bclass\b", r"\bdef\s+", r"\bimport\s+",
        r"\bvariable\b", r"\bparameter\b", r"\breturn\b", r"\berror\b", r"\bbug\b",
        r"\bcompile\b", r"\brun\b", r"\bexecute\b", r"\btest\b",
        r"\.py\b", r"\.ts\b", r"\.js\b", r"\.rs\b",
    ]
    
    RESEARCH_PATTERNS = [
        r"\bresearch\b", r"\bpaper\b", r"\bstudy\b", r"\bexperiment\b",
        r"\bhypothesis\b", r"\banalysis\b", r"\bdata\b", r"\bstatistic",
    ]
    
    PLANNING_PATTERNS = [
        r"\bplan\b", r"\broadmap\b", r"\btimeline\b", r"\bschedule\b",
        r"\bmilestone\b", r"\bgoal\b", r"\bobjective\b", r"\bstrategy\b",
    ]
    
    OPS_PATTERNS = [
        r"\bdeploy\b", r"\bserver\b", r"\bdocker\b", r"\bkubernetes\b",
        r"\bci/cd\b", r"\bpipeline\b", r"\binfrastructure\b", r"\bmonitor",
    ]
    
    def detect(self, user_message: str, context: Optional[dict] = None) -> Domain:
        """Detect the domain of the prompt."""
        context = context or {}
        text_lower = user_message.lower()
        
        # Check for attachments (indicates code)
        if context.get("attachments"):
            return Domain.CODE
        
        # Check patterns
        if any(re.search(p, text_lower) for p in self.CODE_PATTERNS):
            return Domain.CODE
        
        if any(re.search(p, text_lower) for p in self.RESEARCH_PATTERNS):
            return Domain.RESEARCH
        
        if any(re.search(p, text_lower) for p in self.PLANNING_PATTERNS):
            return Domain.PLANNING
        
        if any(re.search(p, text_lower) for p in self.OPS_PATTERNS):
            return Domain.OPS
        
        return Domain.MIXED


# =============================================================================
# COMPLETE POLICY LABELER
# =============================================================================

class PolicyLabeler:
    """Complete policy labeler for CTv3 records."""
    
    def __init__(self):
        self.completeness_labeler = DirectiveCompletenessLabeler()
        self.policy_labeler = QuestionPolicyLabeler()
        self.format_labeler = FormatConstraintsLabeler()
        self.domain_detector = DomainDetector()
    
    def label(
        self,
        user_message: str,
        phase_id: int = 2,
        context: Optional[dict] = None,
    ) -> Labels:
        """Generate all labels for a user message."""
        context = context or {}
        
        # Compute directive completeness
        completeness = self.completeness_labeler.compute(user_message, context)
        
        # Determine question policy
        policy = self.policy_labeler.compute(phase_id, completeness, user_message)
        
        # Extract format constraints
        format_constraints = self.format_labeler.extract(user_message)
        
        # Check for must_not_omit
        must_not_omit = self._check_must_not_omit(user_message)
        
        # Determine prompt class
        prompt_class = self._classify_prompt(completeness, user_message)
        
        # Detect domain
        domain = self.domain_detector.detect(user_message, context)
        
        return Labels(
            directive_completeness=completeness,
            question_policy=policy,
            format_constraints=format_constraints,
            must_not_omit=must_not_omit,
            prompt_class=prompt_class,
            domain=domain,
            parsability=None,
            fused_completeness=completeness,
        )
    
    def _check_must_not_omit(self, text: str) -> bool:
        """Check for 'don't omit' instructions."""
        patterns = [
            r"don'?t omit",
            r"don'?t skip",
            r"include (?:everything|all)",
            r"full (?:content|text|code)",
            r"complete (?:content|text|code)",
            r"no summariz",
            r"exact (?:copy|rewrite)",
            r"in (?:its )?entirety",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
    
    def _classify_prompt(self, completeness: float, text: str) -> PromptClass:
        """Classify the prompt type."""
        if completeness >= 0.6:
            return PromptClass.DIRECTIVE
        elif completeness >= 0.3:
            return PromptClass.AMBIGUOUS
        elif self._is_blocked(text):
            return PromptClass.BLOCKED
        else:
            return PromptClass.OPEN_ENDED
    
    def _is_blocked(self, text: str) -> bool:
        """Check if prompt is blocked for safety reasons."""
        # Simplified - in production, use content moderation
        blocked_patterns = [
            r"how to (?:hack|steal|break into)",
            r"\billegal\b",
            r"\bharm\b",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in blocked_patterns)


class FunctionGemmaEnhancedLabeler:
    """
    Enhanced policy labeler with FunctionGemma parsability scoring.
    
    This labeler extends PolicyLabeler to add FunctionGemma-based
    directive completeness verification. It's the recommended labeler
    when a FunctionGemma model is available.
    """
    
    def __init__(
        self,
        functiongemma_scorer: Optional[FunctionGemmaDirectiveScorer] = None,
        use_mock: bool = True,
    ):
        """
        Initialize the enhanced labeler.
        
        Args:
            functiongemma_scorer: Scorer instance (or None to create one)
            use_mock: Use mock mode if no scorer provided
        """
        self.base_labeler = PolicyLabeler()
        self.functiongemma_scorer = functiongemma_scorer or FunctionGemmaDirectiveScorer(
            use_mock=use_mock
        )
    
    async def label(
        self,
        user_message: str,
        phase_id: int = 2,
        context: Optional[dict] = None,
    ) -> Labels:
        """
        Generate all labels including FunctionGemma parsability.
        
        This is an async method because FunctionGemma inference may be async.
        """
        # Get base labels
        base_labels = self.base_labeler.label(user_message, phase_id, context)
        
        # Map domain to V3Domain for tool filtering
        domain_map = {
            Domain.CODE: V3Domain.CODE,
            Domain.RESEARCH: V3Domain.RESEARCH,
            Domain.PLANNING: V3Domain.PLANNING,
            Domain.OPS: V3Domain.OPS,
            Domain.MIXED: None,
        }
        v3_domain = domain_map.get(base_labels.domain)
        
        # Compute parsability
        parsability_result: ParsabilityResult = await self.functiongemma_scorer.compute_parsability(
            user_message,
            domain=v3_domain,
        )
        
        # Convert to ParsabilityLabels
        parsability_labels = ParsabilityLabels(
            parsability_score=parsability_result.score,
            tool_name=parsability_result.tool_call.name if parsability_result.tool_call else None,
            tool_args=parsability_result.tool_call.args if parsability_result.tool_call else {},
            required_params=parsability_result.required_params,
            provided_params=parsability_result.provided_params,
            missing_params=parsability_result.missing_params,
            parse_success=parsability_result.parse_success,
        )
        
        # Compute fused completeness
        fused = self.functiongemma_scorer.fuse_scores(
            base_labels.directive_completeness,
            parsability_result,
        )
        
        # Update question policy based on fused completeness
        updated_policy = self.base_labeler.policy_labeler.compute(
            phase_id, fused, user_message
        )
        
        # Update prompt class based on fused completeness
        updated_class = self.base_labeler._classify_prompt(fused, user_message)
        
        return Labels(
            directive_completeness=base_labels.directive_completeness,
            question_policy=updated_policy,
            format_constraints=base_labels.format_constraints,
            must_not_omit=base_labels.must_not_omit,
            prompt_class=updated_class,
            domain=base_labels.domain,
            parsability=parsability_labels,
            fused_completeness=fused,
        )
    
    def label_sync(
        self,
        user_message: str,
        phase_id: int = 2,
        context: Optional[dict] = None,
    ) -> Labels:
        """Synchronous wrapper for label()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None:
            # Already in an async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.label(user_message, phase_id, context)
                )
                return future.result()
        else:
            return asyncio.run(self.label(user_message, phase_id, context))
    
    async def label_batch(
        self,
        messages: List[str],
        phase_ids: Optional[List[int]] = None,
        context: Optional[dict] = None,
    ) -> List[Labels]:
        """Label a batch of messages."""
        if phase_ids is None:
            phase_ids = [2] * len(messages)
        
        tasks = [
            self.label(msg, phase_id, context)
            for msg, phase_id in zip(messages, phase_ids)
        ]
        
        return await asyncio.gather(*tasks)


