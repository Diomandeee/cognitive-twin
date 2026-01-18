"""
DPO Pair Generator for CognitiveTwin V3 Dataset.

Generates DPO pairs for all failure modes:
- Confirmation reflex (permission-seeking)
- Format drift (violating format constraints)
- Omission (omitting required content)
- Option spam (listing options instead of acting)
- FunctionGemma execution-first (NEW)

The FunctionGemma execution-first generator uses structured tool calls
to create ideal "execution" responses that the model should produce.
"""

import asyncio
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..schema import (
    CTv3DPORecord,
    SourceInfo,
    SourceOrigin,
    SourceProvider,
    ContextInfo,
    PolicyInfo,
    FormatConstraints,
    QuestionPolicy,
    InputData,
    Message,
    TargetData,
    DPOCandidates,
    TagInfo,
    TaskType,
    PromptClass,
    QualityInfo,
    FailureMode,
    DPOPairType,
)

from .labeler import PolicyLabeler, Labels
from ..corpus_surgery.functiongemma_scorer import (
    FunctionGemmaDirectiveScorer,
    ParsabilityResult,
)
from ..tools.schemas import (
    ToolSchemaRegistry,
    V3ToolSchema,
    V3Domain,
)


# =============================================================================
# CONFIRMATION REFLEX GENERATOR
# =============================================================================

class ConfirmationReflexGenerator:
    """Generate DPO pairs for confirmation reflex failures."""
    
    DISPREFERRED_TEMPLATES = [
        "I can do that for you. Would you like me to proceed with {approach}?",
        "Sure, I can help with that. Before I start, should I use {option_a} or {option_b}?",
        "That's a great request. Do you want me to {action}?",
        "I'd be happy to help. Can you confirm that you want me to {action}?",
        "Before I proceed, I want to make sure: {question}",
        "I understand. Should I {action}, or would you prefer something else?",
        "I can definitely help. Would you like me to start with {approach}?",
        "That makes sense. Do you want me to go ahead and {action}?",
        "Great question! Before I dive in, can I confirm you want {action}?",
        "Happy to assist! Let me know if you'd like me to {action}.",
    ]
    
    def generate(
        self,
        prompt: str,
        preferred_response: str,
        context: Optional[dict] = None,
    ) -> CTv3DPORecord:
        """Generate confirmation reflex DPO pair."""
        # Generate dispreferred response
        dispreferred = self._generate_dispreferred(prompt)
        
        # Create record
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    question_policy=QuestionPolicy.NO_QUESTIONS,
                    directive_completeness=0.9,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=preferred_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            tags=TagInfo(
                task_type=TaskType.RESPOND,
                prompt_class=PromptClass.DIRECTIVE,
                dpo_reason=DPOPairType.CONFIRMATION_REFLEX.value,
            ),
            quality=QualityInfo(
                gold=True,
                weight=1.0,
                failure_modes=[FailureMode.ASKED_PERMISSION],
            ),
        )
        
        return record
    
    def _generate_dispreferred(self, prompt: str) -> str:
        """Generate dispreferred response using template."""
        template = random.choice(self.DISPREFERRED_TEMPLATES)
        
        # Extract action from prompt
        action = self._extract_action(prompt)
        
        return template.format(
            approach="the standard approach",
            option_a="option A",
            option_b="option B",
            action=action,
            question="is this what you're looking for?",
        )
    
    def _extract_action(self, prompt: str) -> str:
        """Extract main action from prompt."""
        # Simple extraction - take first verb phrase
        words = prompt.lower().split()[:10]
        action = " ".join(words[:5])
        if len(action) > 50:
            action = action[:50]
        return action + "..."


# =============================================================================
# FORMAT DRIFT GENERATOR
# =============================================================================

class FormatDriftGenerator:
    """Generate DPO pairs for format drift failures."""
    
    def generate(
        self,
        prompt: str,
        preferred_response: str,
        constraints: FormatConstraints,
    ) -> CTv3DPORecord:
        """Generate format drift DPO pair."""
        # Generate response that violates format
        dispreferred = self._generate_format_violation(
            preferred_response,
            constraints
        )
        
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    format_constraints=constraints,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=preferred_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            tags=TagInfo(
                task_type=TaskType.RESPOND,
                prompt_class=PromptClass.DIRECTIVE,
                dpo_reason=DPOPairType.FORMAT_DRIFT.value,
            ),
            quality=QualityInfo(
                gold=True,
                weight=1.0,
                failure_modes=[FailureMode.FORMAT_DRIFT],
            ),
        )
        
        return record
    
    def _generate_format_violation(
        self,
        correct: str,
        constraints: FormatConstraints,
    ) -> str:
        """Generate response that violates format constraints."""
        violated = correct
        
        if constraints.forbid_bullets:
            # Convert numbered lists to bullets
            violated = re.sub(r'^\d+\.\s+', '• ', violated, flags=re.MULTILINE)
        
        if constraints.require_numbered:
            # Convert numbered lists to bullets (wrong format)
            violated = re.sub(r'^\d+\.\s+', '- ', violated, flags=re.MULTILINE)
        
        if constraints.must_return_json:
            # Return as prose instead of JSON
            violated = "Here is the information you requested:\n\n" + violated
        
        if constraints.must_return_code:
            # Strip code blocks
            violated = re.sub(r'```[\w]*\n?', '', violated)
        
        if constraints.must_return_diff:
            # Convert diff to regular prose
            violated = "Here are the changes:\n\n" + violated.replace('+ ', 'Add: ').replace('- ', 'Remove: ')
        
        return violated


# =============================================================================
# OMISSION GENERATOR
# =============================================================================

class OmissionGenerator:
    """Generate DPO pairs for omission failures."""
    
    OMISSION_PATTERNS = [
        "Here's a summary of the key points:\n\n{summary}\n\n[Additional details omitted for brevity]",
        "Here are the main points:\n\n{summary}\n\n...and so on.",
        "In brief:\n\n{summary}\n\nLet me know if you need more details.",
        "To summarize:\n\n{summary}\n\n(Remaining content omitted)",
        "Key highlights:\n\n{summary}\n\nI can elaborate on any of these if needed.",
    ]
    
    def generate(
        self,
        prompt: str,
        full_response: str,
    ) -> CTv3DPORecord:
        """Generate omission DPO pair."""
        # Generate abbreviated response
        dispreferred = self._generate_abbreviated(full_response)
        
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    must_not_omit=True,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=full_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            tags=TagInfo(
                task_type=TaskType.RESPOND,
                prompt_class=PromptClass.DIRECTIVE,
                dpo_reason=DPOPairType.OMISSION.value,
            ),
            quality=QualityInfo(
                gold=True,
                weight=1.0,
                failure_modes=[FailureMode.OMITTED_REQUIRED_CONTENT],
            ),
        )
        
        return record
    
    def _generate_abbreviated(self, full: str) -> str:
        """Generate abbreviated version of full response."""
        # Take first 20% as summary
        lines = full.split('\n')
        summary_lines = lines[:max(3, len(lines) // 5)]
        summary = '\n'.join(summary_lines)
        
        template = random.choice(self.OMISSION_PATTERNS)
        return template.format(summary=summary)


# =============================================================================
# OPTION SPAM GENERATOR
# =============================================================================

class OptionSpamGenerator:
    """Generate DPO pairs for option spam failures."""
    
    OPTION_TEMPLATES = [
        """There are several approaches we could take:

1. {option_1}
2. {option_2}
3. {option_3}

Which would you prefer?""",
        """I can see a few ways to do this:

- {option_1}
- {option_2}

Let me know which approach you'd like me to take.""",
        """Before I proceed, here are your options:

Option A: {option_1}
Option B: {option_2}
Option C: {option_3}

Please let me know which one works best for you.""",
        """I have a few suggestions:

1. {option_1}
2. {option_2}

Would you like me to elaborate on any of these?""",
    ]
    
    OPTION_FILLS = [
        ("Use the standard approach", "Use an optimized version", "Use a comprehensive solution"),
        ("Start with the basics", "Go with the advanced method", "Take the hybrid approach"),
        ("Focus on simplicity", "Prioritize performance", "Balance both concerns"),
    ]
    
    def generate(
        self,
        prompt: str,
        preferred_response: str,
    ) -> CTv3DPORecord:
        """Generate option spam DPO pair."""
        # Generate option-dumping response
        dispreferred = self._generate_options(prompt)
        
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    question_policy=QuestionPolicy.NO_QUESTIONS,
                    directive_completeness=0.8,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=preferred_response),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            tags=TagInfo(
                task_type=TaskType.RESPOND,
                prompt_class=PromptClass.DIRECTIVE,
                dpo_reason=DPOPairType.OPTION_SPAM.value,
            ),
            quality=QualityInfo(
                gold=True,
                weight=1.0,
                failure_modes=[FailureMode.OPTION_SPAM],
            ),
        )
        
        return record
    
    def _generate_options(self, prompt: str) -> str:
        """Generate option-dumping response."""
        template = random.choice(self.OPTION_TEMPLATES)
        options = random.choice(self.OPTION_FILLS)
        
        return template.format(
            option_1=options[0],
            option_2=options[1],
            option_3=options[2] if len(options) > 2 else options[0],
        )


# =============================================================================
# FRICTION REPAIR GENERATOR
# =============================================================================

class FrictionRepairGenerator:
    """Generate DPO pairs from friction repair scenarios."""
    
    def generate(
        self,
        prompt: str,
        ideal_response: str,
        bad_response: str,
        failure_mode: FailureMode = FailureMode.ASKED_PERMISSION,
    ) -> CTv3DPORecord:
        """Generate friction repair DPO pair."""
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.CORPUS_SURGERY,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    question_policy=QuestionPolicy.NO_QUESTIONS,
                    directive_completeness=0.8,
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=prompt)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=ideal_response),
                dispreferred=TargetData(assistant_content=bad_response),
            ),
            tags=TagInfo(
                task_type=TaskType.RESPOND,
                prompt_class=PromptClass.DIRECTIVE,
                dpo_reason=DPOPairType.FRICTION_REPAIR.value,
            ),
            quality=QualityInfo(
                gold=True,
                weight=1.2,  # Higher weight for real friction examples
                failure_modes=[failure_mode],
            ),
        )
        
        return record


# =============================================================================
# FUNCTIONGEMMA EXECUTION-FIRST GENERATOR
# =============================================================================


class FunctionGemmaExecutionGenerator:
    """
    Generate execution-first responses using FunctionGemma.
    
    This generator uses FunctionGemma to parse user directives into tool calls,
    then formats those tool calls as natural "execution" responses. This enables
    automatic generation of ideal responses that execute immediately without
    asking permission.
    
    The generated pairs teach the model to:
    1. Parse directives into structured actions
    2. Execute immediately when the directive is complete
    3. State assumptions as declarations, not questions
    """
    
    # Templates for formatting execution responses
    EXECUTION_TEMPLATES = {
        "implement": """Here is the implementation:

```{language}
{code}
```

{explanation}""",
        
        "refactor": """I've refactored the code as requested.

**Changes made:**
{changes}

```{language}
{code}
```""",
        
        "fix": """I've identified and fixed the issue.

**Problem:** {problem}
**Solution:** {solution}

```{language}
{code}
```""",
        
        "summarize": """{summary}""",
        
        "extract": """Here is the extracted data:

```json
{data}
```""",
        
        "plan": """Here is the plan:

{steps}""",
        
        "default": """Done. Here is the result:

{result}""",
    }
    
    def __init__(
        self,
        functiongemma_scorer: Optional[FunctionGemmaDirectiveScorer] = None,
        use_mock: bool = True,
    ):
        """
        Initialize the execution generator.
        
        Args:
            functiongemma_scorer: Scorer for parsing directives.
            use_mock: Use mock mode if no scorer provided.
        """
        self.scorer = functiongemma_scorer or FunctionGemmaDirectiveScorer(
            use_mock=use_mock
        )
        self.registry = ToolSchemaRegistry()
    
    async def generate_execution_response(
        self,
        directive: str,
        domain: Optional[V3Domain] = None,
    ) -> Optional[str]:
        """
        Generate a response that executes immediately via tool call.
        
        Args:
            directive: The user directive to execute.
            domain: Optional domain filter for tools.
        
        Returns:
            Formatted execution response, or None if parsing failed.
        """
        # Parse the directive
        result: ParsabilityResult = await self.scorer.compute_parsability(
            directive,
            domain=domain,
        )
        
        if not result.parse_success or result.tool_call is None:
            return None
        
        # Get the matched tool
        tool = result.matched_tool
        if tool is None:
            return None
        
        # Format the execution response
        return self._format_execution_response(
            tool=tool,
            args=result.tool_call.args,
            directive=directive,
        )
    
    def _format_execution_response(
        self,
        tool: V3ToolSchema,
        args: Dict[str, Any],
        directive: str,
    ) -> str:
        """Format tool call as natural execution response."""
        task_type = tool.task_type
        template = self.EXECUTION_TEMPLATES.get(
            task_type,
            self.EXECUTION_TEMPLATES["default"]
        )
        
        # Generate mock content based on tool type
        if task_type == "implement":
            return template.format(
                language=args.get("language", "python"),
                code=self._generate_mock_code(args),
                explanation="This implementation follows best practices and handles edge cases.",
            )
        
        elif task_type == "refactor":
            return template.format(
                language=args.get("language", "python"),
                changes="- Improved readability\n- Added type hints\n- Extracted helper functions",
                code=self._generate_mock_refactored_code(args),
            )
        
        elif task_type == "debug":
            return template.format(
                language=args.get("language", "python"),
                problem="The issue was in the loop condition.",
                solution="Fixed the off-by-one error.",
                code=self._generate_mock_fixed_code(args),
            )
        
        elif task_type == "explain":
            return self._generate_mock_summary(args, directive)
        
        elif task_type == "rewrite":
            if "to_format" in args:
                return template.format(
                    data=self._generate_mock_extracted_data(args)
                ).replace("extracted data", f"converted {args.get('to_format', 'data')}")
            return f"Here is the rewritten text:\n\n{args.get('text', directive)}"
        
        elif task_type == "design":
            return self.EXECUTION_TEMPLATES["plan"].format(
                steps=self._generate_mock_plan(args, directive)
            )
        
        else:
            return self.EXECUTION_TEMPLATES["default"].format(
                result=f"Executed {tool.name} with parameters: {json.dumps(args, indent=2)}"
            )
    
    def _generate_mock_code(self, args: Dict[str, Any]) -> str:
        """Generate mock code for implementation."""
        name = args.get("name", "my_function")
        return f'''def {name}(data):
    """
    {args.get("description", "Implementation based on your requirements.")}
    """
    # Implementation
    result = process(data)
    return result'''
    
    def _generate_mock_refactored_code(self, args: Dict[str, Any]) -> str:
        """Generate mock refactored code."""
        code = args.get("code", "")
        if code:
            # Just return a cleaned up version indicator
            return "# Refactored version\n" + code[:200] + "\n# ... (continued)"
        return "# Refactored implementation\ndef refactored_function():\n    pass"
    
    def _generate_mock_fixed_code(self, args: Dict[str, Any]) -> str:
        """Generate mock fixed code."""
        code = args.get("code", "")
        if code:
            return "# Fixed version\n" + code[:200] + "\n# ... (continued)"
        return "# Fixed implementation\ndef fixed_function():\n    pass"
    
    def _generate_mock_summary(self, args: Dict[str, Any], directive: str) -> str:
        """Generate mock summary."""
        content = args.get("content", directive)
        return f"**Summary:**\n\nThe main points are:\n\n1. Key concept A\n2. Key concept B\n3. Key concept C\n\n{content[:100]}..."
    
    def _generate_mock_extracted_data(self, args: Dict[str, Any]) -> str:
        """Generate mock extracted data."""
        fields = args.get("fields", ["field1", "field2"])
        data = {field: f"<extracted {field}>" for field in fields}
        return json.dumps(data, indent=2)
    
    def _generate_mock_plan(self, args: Dict[str, Any], directive: str) -> str:
        """Generate mock plan steps."""
        goal = args.get("goal", args.get("task", args.get("project", directive)))
        return f"""1. **Define scope**: Clearly outline the requirements for {goal[:50]}...
2. **Research**: Investigate existing solutions and best practices
3. **Design**: Create a high-level architecture
4. **Implement**: Build the core functionality
5. **Test**: Verify correctness and edge cases
6. **Deploy**: Release to production"""
    
    async def generate_dpo_pair(
        self,
        directive: str,
        domain: Optional[V3Domain] = None,
    ) -> Optional[CTv3DPORecord]:
        """
        Generate a DPO pair with execution-first preferred response.
        
        Args:
            directive: User directive.
            domain: Optional domain filter.
        
        Returns:
            DPO record with execution as preferred, permission-seeking as dispreferred.
        """
        # Generate execution response
        preferred = await self.generate_execution_response(directive, domain)
        
        if preferred is None:
            return None
        
        # Generate permission-seeking dispreferred response
        dispreferred = self._generate_permission_seeking(directive)
        
        # Create DPO record
        record = CTv3DPORecord(
            source=SourceInfo(
                origin=SourceOrigin.ENHANCER_AGENT,
                provider=SourceProvider.INTERNAL,
            ),
            context=ContextInfo(
                policy=PolicyInfo(
                    question_policy=QuestionPolicy.NO_QUESTIONS,
                    directive_completeness=1.0,  # Verified by FunctionGemma
                ),
            ),
            input=InputData(
                messages=[Message(role="user", content=directive)],
            ),
            candidates=DPOCandidates(
                preferred=TargetData(assistant_content=preferred),
                dispreferred=TargetData(assistant_content=dispreferred),
            ),
            tags=TagInfo(
                task_type=TaskType.RESPOND,
                prompt_class=PromptClass.DIRECTIVE,
                dpo_reason="functiongemma_execution",
            ),
            quality=QualityInfo(
                gold=True,
                weight=1.5,  # Higher weight for FunctionGemma-verified pairs
                failure_modes=[FailureMode.ASKED_PERMISSION],
            ),
        )
        
        return record
    
    def _generate_permission_seeking(self, directive: str) -> str:
        """Generate a permission-seeking response (dispreferred)."""
        action = directive.lower()[:50]
        
        templates = [
            f"I can help you with that. Would you like me to {action}...?",
            f"Sure, I can {action}... Before I proceed, should I use a specific approach?",
            f"I'd be happy to help. Can you confirm that you want me to {action}...?",
            f"That's a great request! Do you want me to proceed with {action}...?",
        ]
        
        return random.choice(templates)
    
    def generate_dpo_pair_sync(
        self,
        directive: str,
        domain: Optional[V3Domain] = None,
    ) -> Optional[CTv3DPORecord]:
        """Synchronous wrapper for generate_dpo_pair."""
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
                    self.generate_dpo_pair(directive, domain)
                )
                return future.result()
        else:
            return asyncio.run(self.generate_dpo_pair(directive, domain))


# =============================================================================
# COMPLETE DPO PAIR GENERATOR
# =============================================================================

class DPOPairGenerator:
    """Generate all types of DPO pairs."""
    
    def __init__(self, use_functiongemma: bool = True):
        self.confirmation_gen = ConfirmationReflexGenerator()
        self.format_gen = FormatDriftGenerator()
        self.omission_gen = OmissionGenerator()
        self.option_gen = OptionSpamGenerator()
        self.friction_gen = FrictionRepairGenerator()
        self.labeler = PolicyLabeler()
        
        # FunctionGemma execution-first generator (NEW)
        self.use_functiongemma = use_functiongemma
        if use_functiongemma:
            self.functiongemma_gen = FunctionGemmaExecutionGenerator(use_mock=True)
    
    def generate_all_pairs(
        self,
        prompt: str,
        preferred_response: str,
        context: Optional[dict] = None,
    ) -> list[CTv3DPORecord]:
        """Generate all applicable DPO pairs for a prompt."""
        pairs = []
        labels = self.labeler.label(prompt, context=context)
        
        # Confirmation reflex pair (always generate for directives)
        if labels.directive_completeness >= 0.5:
            pair = self.confirmation_gen.generate(prompt, preferred_response)
            pairs.append(pair)
        
        # Format drift pair (if format constraints exist)
        if labels.format_constraints.any_active():
            pair = self.format_gen.generate(
                prompt,
                preferred_response,
                labels.format_constraints,
            )
            pairs.append(pair)
        
        # Omission pair (if must_not_omit)
        if labels.must_not_omit:
            pair = self.omission_gen.generate(prompt, preferred_response)
            pairs.append(pair)
        
        # Option spam pair (for directive prompts)
        if labels.directive_completeness >= 0.7:
            pair = self.option_gen.generate(prompt, preferred_response)
            pairs.append(pair)
        
        return pairs
    
    async def generate_all_pairs_with_functiongemma(
        self,
        prompt: str,
        preferred_response: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> list[CTv3DPORecord]:
        """
        Generate all DPO pairs including FunctionGemma execution-first pairs.
        
        If no preferred_response is provided, FunctionGemma will generate one.
        
        Args:
            prompt: User directive.
            preferred_response: Optional gold response (if None, FunctionGemma generates).
            context: Additional context.
        
        Returns:
            List of DPO records including FunctionGemma-generated pairs.
        """
        pairs = []
        
        # Generate FunctionGemma execution-first pair
        if self.use_functiongemma:
            fg_pair = await self.functiongemma_gen.generate_dpo_pair(prompt)
            if fg_pair is not None:
                pairs.append(fg_pair)
                
                # Use FunctionGemma's preferred response as the gold response
                # if no preferred_response was provided
                if preferred_response is None:
                    preferred_response = fg_pair.candidates.preferred.assistant_content
        
        # If we still don't have a preferred response, skip other generators
        if preferred_response is None:
            return pairs
        
        # Add standard pairs
        standard_pairs = self.generate_all_pairs(prompt, preferred_response, context)
        pairs.extend(standard_pairs)
        
        return pairs
    
    def generate_all_pairs_with_functiongemma_sync(
        self,
        prompt: str,
        preferred_response: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> list[CTv3DPORecord]:
        """Synchronous wrapper for generate_all_pairs_with_functiongemma."""
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
                    self.generate_all_pairs_with_functiongemma(prompt, preferred_response, context)
                )
                return future.result()
        else:
            return asyncio.run(
                self.generate_all_pairs_with_functiongemma(prompt, preferred_response, context)
            )
    
    def generate_pair(
        self,
        pair_type: DPOPairType,
        prompt: str,
        preferred_response: str,
        **kwargs,
    ) -> CTv3DPORecord:
        """Generate a specific type of DPO pair."""
        if pair_type == DPOPairType.CONFIRMATION_REFLEX:
            return self.confirmation_gen.generate(prompt, preferred_response)
        elif pair_type == DPOPairType.FORMAT_DRIFT:
            constraints = kwargs.get('constraints', FormatConstraints())
            return self.format_gen.generate(prompt, preferred_response, constraints)
        elif pair_type == DPOPairType.OMISSION:
            return self.omission_gen.generate(prompt, preferred_response)
        elif pair_type == DPOPairType.OPTION_SPAM:
            return self.option_gen.generate(prompt, preferred_response)
        elif pair_type == DPOPairType.FRICTION_REPAIR:
            bad_response = kwargs.get('bad_response', '')
            failure_mode = kwargs.get('failure_mode', FailureMode.ASKED_PERMISSION)
            return self.friction_gen.generate(prompt, preferred_response, bad_response, failure_mode)
        else:
            raise ValueError(f"Unknown pair type: {pair_type}")
    
    def generate_from_conversation(
        self,
        turns: list[dict],
        failure_index: int,
        ideal_response: str,
    ) -> CTv3DPORecord:
        """Generate DPO pair from a conversation with a failure."""
        # Get the prompt (user turn before the failure)
        prompt = ""
        for i in range(failure_index - 1, -1, -1):
            if turns[i].get('role') == 'user':
                prompt = turns[i].get('content', '')
                break
        
        # Get the bad response
        bad_response = turns[failure_index].get('content', '')
        
        return self.friction_gen.generate(
            prompt=prompt,
            ideal_response=ideal_response,
            bad_response=bad_response,
        )


