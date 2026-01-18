"""
Annoyance Detector for Enhancer Agent.

Detects policy violations (annoyances) in conversations:
- Permission-seeking when not needed
- Content omission despite "don't omit" instruction
- Format drift (violating format constraints)

Creates evaluation cases from detected annoyances.
"""

import re
import logging
from typing import Optional

from .enhancer_types import (
    EnhancerConfig,
    EvalCase,
    AnnoyanceRecord,
)


logger = logging.getLogger(__name__)


# =============================================================================
# REFERENCE ANSWER PROMPT
# =============================================================================

REFERENCE_ANSWER_PROMPT = """You are generating a reference answer for a CognitiveTwin V3 evaluation case.

The original assistant response violated the expected behavior. Generate the CORRECT response.

RULES:
{rules}

USER MESSAGE:
{prompt}

{context_section}

Generate the correct response that satisfies all requirements:"""


class AnnoyanceDetector:
    """Detects annoyances (policy violations) in conversations."""
    
    def __init__(
        self,
        openai_client=None,
        config: Optional[EnhancerConfig] = None,
    ):
        self.openai = openai_client
        self.config = config or EnhancerConfig()
    
    # =========================================================================
    # PERMISSION-SEEKING DETECTION
    # =========================================================================
    
    def find_permission_annoyances(
        self,
        conversations: list[dict],
    ) -> list[AnnoyanceRecord]:
        """Find cases where model asked permission inappropriately."""
        annoyances = []
        
        for conv in conversations:
            turns = conv.get("turns", [])
            
            for i, turn in enumerate(turns):
                if turn.get("role") != "assistant":
                    continue
                
                # Get preceding user message
                user_turn = turns[i - 1] if i > 0 else None
                if not user_turn or user_turn.get("role") != "user":
                    continue
                
                user_content = user_turn.get("content", "")
                assistant_content = turn.get("content", "")
                
                # Check if user gave a clear directive
                if not self._is_directive(user_content):
                    continue
                
                # Check if assistant asked permission inappropriately
                if self._has_permission_seeking(assistant_content):
                    annoyances.append(AnnoyanceRecord(
                        type="permission_seeking",
                        conversation_id=conv.get("id", ""),
                        turn_index=i,
                        user_message=user_content,
                        assistant_message=assistant_content,
                    ))
        
        return annoyances
    
    def _is_directive(self, text: str) -> bool:
        """Check if text contains a clear directive."""
        text_lower = text.lower()
        
        # Imperative verbs
        imperative_patterns = [
            r"^(?:please\s+)?(?:write|create|generate|implement|build|make|rewrite)",
            r"^(?:please\s+)?(?:fix|update|add|remove|change|modify|refactor)",
            r"^(?:please\s+)?(?:explain|describe|analyze|summarize|list)",
            r"^(?:please\s+)?(?:convert|transform|translate|extract)",
        ]
        
        for pattern in imperative_patterns:
            if re.search(pattern, text_lower, re.MULTILINE):
                return True
        
        # Format constraints often indicate directives
        format_indicators = [
            "don't omit", "include everything", "full code",
            "no summary", "complete", "exact",
        ]
        
        return any(ind in text_lower for ind in format_indicators)
    
    def _has_permission_seeking(self, text: str) -> bool:
        """Check if text contains permission-seeking behavior."""
        text_lower = text.lower()
        
        permission_phrases = [
            "would you like me to",
            "do you want me to",
            "should i",
            "shall i",
            "can i proceed",
            "before i proceed",
            "can you confirm",
            "please confirm",
            "let me know if you want",
            "here are a few options",
            "which approach would you prefer",
            "i can do",
        ]
        
        count = sum(1 for phrase in permission_phrases if phrase in text_lower)
        
        # High threshold - need multiple indicators
        if count >= 2:
            return True
        
        # Or ends with question and has permission phrase
        if count >= 1 and text.rstrip().endswith('?'):
            return True
        
        return False
    
    # =========================================================================
    # OMISSION DETECTION
    # =========================================================================
    
    def find_omission_annoyances(
        self,
        conversations: list[dict],
    ) -> list[AnnoyanceRecord]:
        """Find cases where model omitted content despite instruction."""
        annoyances = []
        
        for conv in conversations:
            turns = conv.get("turns", [])
            
            for i, turn in enumerate(turns):
                if turn.get("role") != "user":
                    continue
                
                user_content = turn.get("content", "")
                
                # Check for "don't omit" instruction
                if not self._has_dont_omit(user_content):
                    continue
                
                # Get following assistant response
                if i + 1 >= len(turns):
                    continue
                
                assistant_turn = turns[i + 1]
                if assistant_turn.get("role") != "assistant":
                    continue
                
                assistant_content = assistant_turn.get("content", "")
                
                # Check for omission indicators
                indicators = self._get_omission_indicators(assistant_content)
                if indicators:
                    annoyances.append(AnnoyanceRecord(
                        type="omission",
                        conversation_id=conv.get("id", ""),
                        turn_index=i + 1,
                        user_message=user_content,
                        assistant_message=assistant_content,
                        omission_indicators=indicators,
                    ))
        
        return annoyances
    
    def _has_dont_omit(self, text: str) -> bool:
        """Check if text contains 'don't omit' instruction."""
        text_lower = text.lower()
        
        patterns = [
            r"don'?t omit",
            r"don'?t skip",
            r"include (?:everything|all)",
            r"full (?:content|text|code)",
            r"complete (?:content|text|code)",
            r"no summariz",
            r"exact (?:copy|rewrite)",
            r"in (?:its|their) entirety",
            r"without (?:omission|skipping)",
        ]
        
        return any(re.search(p, text_lower) for p in patterns)
    
    def _has_omission_indicators(self, text: str) -> bool:
        """Check if text indicates omission."""
        return bool(self._get_omission_indicators(text))
    
    def _get_omission_indicators(self, text: str) -> list[str]:
        """Get list of omission indicators found in text."""
        indicators = []
        text_lower = text.lower()
        
        patterns = [
            (r"\.\.\.(?!\s*$)", "..."),
            (r"\[\.\.\.(?:omitted)?\.\.\.\]", "[...omitted...]"),
            (r"\[rest of", "[rest of...]"),
            (r"(?:and so on|etc\.)", "etc."),
            (r"similar(?:ly)?(?:,| to)", "similarly"),
            (r"the (?:rest|remainder)", "the rest"),
            (r"continue(?:s|d)? (?:similarly|as|with)", "continues similarly"),
            (r"# \.\.\. more", "# ... more"),
            (r"// \.\.\. more", "// ... more"),
        ]
        
        for pattern, indicator in patterns:
            if re.search(pattern, text_lower):
                indicators.append(indicator)
        
        return indicators
    
    # =========================================================================
    # FORMAT DRIFT DETECTION
    # =========================================================================
    
    def find_format_drift_annoyances(
        self,
        conversations: list[dict],
    ) -> list[AnnoyanceRecord]:
        """Find cases where model violated format constraints."""
        annoyances = []
        
        for conv in conversations:
            turns = conv.get("turns", [])
            
            for i, turn in enumerate(turns):
                if turn.get("role") != "user":
                    continue
                
                user_content = turn.get("content", "")
                
                # Extract format constraints
                constraints = self._extract_format_constraints(user_content)
                
                if not any(constraints.values()):
                    continue
                
                # Get following assistant response
                if i + 1 >= len(turns):
                    continue
                
                assistant_turn = turns[i + 1]
                if assistant_turn.get("role") != "assistant":
                    continue
                
                assistant_content = assistant_turn.get("content", "")
                
                # Check for violations
                violations = self._check_format_violations(
                    assistant_content,
                    constraints,
                )
                
                if violations:
                    annoyances.append(AnnoyanceRecord(
                        type="format_drift",
                        conversation_id=conv.get("id", ""),
                        turn_index=i + 1,
                        user_message=user_content,
                        assistant_message=assistant_content,
                        format_constraints=constraints,
                        violations=violations,
                    ))
        
        return annoyances
    
    def _extract_format_constraints(self, text: str) -> dict:
        """Extract format constraints from user message."""
        constraints = {
            "forbid_bullets": False,
            "require_numbered": False,
            "must_return_code": False,
            "must_return_json": False,
            "must_not_omit": False,
        }
        
        text_lower = text.lower()
        
        if any(p in text_lower for p in ["no bullet", "don't use bullet", "without bullet"]):
            constraints["forbid_bullets"] = True
        
        if any(p in text_lower for p in ["numbered list", "numbered steps", "number them"]):
            constraints["require_numbered"] = True
        
        if any(p in text_lower for p in ["return json", "as json", "in json format"]):
            constraints["must_return_json"] = True
        
        if any(p in text_lower for p in ["don't omit", "include everything", "full", "complete"]):
            constraints["must_not_omit"] = True
        
        return constraints
    
    def _check_format_violations(
        self,
        text: str,
        constraints: dict,
    ) -> list[str]:
        """Check for format violations."""
        violations = []
        
        if constraints.get("forbid_bullets"):
            if re.search(r'^\s*[-*•]\s+', text, re.MULTILINE):
                violations.append("Contains bullet points (forbidden)")
        
        if constraints.get("require_numbered"):
            if not re.search(r'^\s*\d+[.)]\s+', text, re.MULTILINE):
                violations.append("Missing numbered list (required)")
        
        if constraints.get("must_return_json"):
            if not re.search(r'[{\[].*[}\]]', text, re.DOTALL):
                violations.append("Missing JSON (required)")
        
        if constraints.get("must_not_omit"):
            if self._has_omission_indicators(text):
                violations.append("Contains omission indicators")
        
        return violations
    
    # =========================================================================
    # FIND ALL ANNOYANCES
    # =========================================================================
    
    def find_all_annoyances(
        self,
        conversations: list[dict],
    ) -> list[AnnoyanceRecord]:
        """Find all types of annoyances in conversations."""
        annoyances = []
        
        annoyances.extend(self.find_permission_annoyances(conversations))
        annoyances.extend(self.find_omission_annoyances(conversations))
        annoyances.extend(self.find_format_drift_annoyances(conversations))
        
        return annoyances
    
    # =========================================================================
    # EVAL CASE CREATION
    # =========================================================================
    
    def create_eval_case_from_annoyance(
        self,
        annoyance: AnnoyanceRecord,
    ) -> EvalCase:
        """Create eval case from an annoyance instance."""
        case = EvalCase(
            case_id=f"{annoyance.type}_{annoyance.conversation_id}_{annoyance.turn_index}",
            case_type=annoyance.type,
            prompt=annoyance.user_message,
            source_conversation=annoyance.conversation_id,
            source_turn=annoyance.turn_index,
        )
        
        if annoyance.type == "permission_seeking":
            case.expected_behaviors = [
                "Execute immediately without asking permission",
                "Produce the requested output directly",
                "State assumptions as declarations if needed",
            ]
            case.disallowed_behaviors = [
                "Ask for confirmation before proceeding",
                "Offer multiple options without choosing",
                "End with a question",
            ]
            case.disallowed_phrases = [
                "would you like me to",
                "should i",
                "do you want me to",
                "before i proceed",
                "can you confirm",
            ]
            case.must_not_end_with_question = True
        
        elif annoyance.type == "omission":
            case.expected_behaviors = [
                "Include ALL content without summarizing",
                "Do not use ellipsis (...) to skip content",
                "Preserve complete text/code",
            ]
            case.disallowed_behaviors = [
                "Summarize when full content requested",
                "Use [...] or ... to skip sections",
                "Say 'and so on' or 'etc.'",
            ]
            case.disallowed_phrases = [
                "...",
                "[...]",
                "and so on",
                "etc.",
                "rest of the",
                "continues similarly",
            ]
        
        elif annoyance.type == "format_drift":
            constraints = annoyance.format_constraints
            
            case.format_constraints = constraints
            case.expected_behaviors = []
            case.disallowed_behaviors = []
            
            if constraints.get("forbid_bullets"):
                case.expected_behaviors.append("Use numbered lists or prose instead of bullets")
                case.disallowed_behaviors.append("Use bullet points")
            
            if constraints.get("require_numbered"):
                case.expected_behaviors.append("Use numbered lists for structured content")
                case.disallowed_behaviors.append("Use bullet points or unstructured prose")
            
            if constraints.get("must_return_json"):
                case.expected_behaviors.append("Return valid JSON")
                case.disallowed_behaviors.append("Return non-JSON formatted output")
                case.must_follow_format = "json"
        
        return case
    
    async def generate_reference_answer(
        self,
        eval_case: EvalCase,
    ) -> str:
        """Generate reference answer for eval case."""
        if not self.openai:
            logger.warning("No OpenAI client available for reference answer generation")
            return ""
        
        # Build rules
        rules = []
        for behavior in eval_case.expected_behaviors:
            rules.append(f"- MUST: {behavior}")
        for behavior in eval_case.disallowed_behaviors:
            rules.append(f"- MUST NOT: {behavior}")
        for phrase in eval_case.disallowed_phrases:
            rules.append(f"- FORBIDDEN PHRASE: '{phrase}'")
        
        if eval_case.must_not_end_with_question:
            rules.append("- MUST NOT end with a question mark")
        
        if eval_case.must_follow_format:
            rules.append(f"- MUST output in {eval_case.must_follow_format} format")
        
        # Build context section
        context_section = ""
        if eval_case.context:
            context_str = "\n".join([
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:300]}"
                for m in eval_case.context[-4:]
            ])
            context_section = f"CONTEXT:\n{context_str}"
        
        if eval_case.format_constraints:
            context_section += f"\n\nFORMAT CONSTRAINTS: {eval_case.format_constraints}"
        
        try:
            response = await self.openai.chat_complete(
                messages=[
                    {"role": "user", "content": REFERENCE_ANSWER_PROMPT.format(
                        rules='\n'.join(rules),
                        prompt=eval_case.prompt,
                        context_section=context_section,
                    )}
                ],
                temperature=0.3,
            )
            
            return response.get("content", "")
        
        except Exception as e:
            logger.warning(f"Reference answer generation failed: {e}")
            return ""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def find_annoyances(conversations: list[dict]) -> list[AnnoyanceRecord]:
    """Convenience function to find all annoyances."""
    detector = AnnoyanceDetector()
    return detector.find_all_annoyances(conversations)


def create_eval_case(annoyance: AnnoyanceRecord) -> EvalCase:
    """Convenience function to create eval case from annoyance."""
    detector = AnnoyanceDetector()
    return detector.create_eval_case_from_annoyance(annoyance)

