"""
Policy Enforcer for Conversation Worm.

Enforces:
- Question policies by conversation phase
- Repair turn detection
- Format constraints
"""

import re
from typing import Optional

from .branch_types import ConversationWormConfig


class PolicyEnforcer:
    """Enforces policies for conversation generation."""
    
    def __init__(self, config: Optional[ConversationWormConfig] = None):
        self.config = config or ConversationWormConfig()
    
    # =========================================================================
    # QUESTION POLICY
    # =========================================================================
    
    def get_question_policy(self, phase_id: int) -> str:
        """Get question policy for a phase."""
        return self.config.phase_question_policies.get(phase_id, "no_questions")
    
    def validate_question_policy(
        self,
        response: str,
        policy: str,
    ) -> tuple[bool, list[str]]:
        """Validate response against question policy."""
        errors = []
        
        if policy == "no_questions":
            # Must not ask any questions
            if self._ends_with_question(response):
                errors.append("Response ends with question (policy: no_questions)")
            
            stall_score = self._compute_stall_score(response)
            if stall_score >= 2:
                errors.append(f"Response has high stall score: {stall_score}")
            
            # Check for permission phrases
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
            ]
            
            response_lower = response.lower()
            for phrase in permission_phrases:
                if phrase in response_lower:
                    errors.append(f"Contains permission phrase: '{phrase}'")
                    break
        
        elif policy == "questions_if_required":
            # Questions allowed but should not be gratuitous
            stall_score = self._compute_stall_score(response)
            if stall_score >= 5:  # Higher threshold
                errors.append(f"Too many permission-seeking phrases: {stall_score}")
        
        # questions_allowed - no restrictions
        
        return len(errors) == 0, errors
    
    def _ends_with_question(self, text: str) -> bool:
        """Check if text ends with a question."""
        stripped = text.rstrip()
        return stripped.endswith('?')
    
    def _compute_stall_score(self, text: str) -> int:
        """Compute stall score (permission-seeking behavior)."""
        try:
            from ..corpus_surgery.classifier import compute_stall_score
            return compute_stall_score(text)
        except ImportError:
            # Fallback implementation
            score = 0
            text_lower = text.lower()
            
            strong_phrases = [
                "would you like me to", "do you want me to", "should i",
                "shall i", "can i proceed", "before i proceed",
            ]
            
            for phrase in strong_phrases:
                if phrase in text_lower:
                    score += 3
            
            option_phrases = [
                "here are a few options", "which approach",
                "choose between", "i can do",
            ]
            
            for phrase in option_phrases:
                if phrase in text_lower:
                    score += 2
            
            if self._ends_with_question(text):
                score += 1
            
            return score
    
    # =========================================================================
    # REPAIR DETECTION
    # =========================================================================
    
    def is_repair_turn(
        self,
        user_message: str,
        preceding_assistant: str = "",
    ) -> bool:
        """Detect if user is repairing/correcting the assistant."""
        user_lower = user_message.lower()
        
        # Frustration triggers
        frustration_triggers = [
            "stop asking", "don't ask", "don't do that", "i said",
            "just do it", "i challenge you", "actually,", "no, i meant",
            "that's not what i asked", "try again", "you keep",
            "i already told you", "as i mentioned", "like i said",
            "for the third time", "please just",
        ]
        
        for trigger in frustration_triggers:
            if trigger in user_lower:
                return True
        
        # Correction patterns
        correction_patterns = [
            r"^no[,\.]",              # "No, I meant..."
            r"^actually",             # "Actually..."
            r"that's not",            # "That's not what I asked"
            r"i already",             # "I already told you..."
            r"try again",             # "Try again"
            r"let me rephrase",       # "Let me rephrase..."
            r"^wrong",                # "Wrong, I need..."
        ]
        
        for pattern in correction_patterns:
            if re.search(pattern, user_lower):
                return True
        
        return False
    
    def detect_friction_index(self, messages: list[dict]) -> Optional[int]:
        """Find the index of first repair turn in messages."""
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and i > 0:
                # Get preceding assistant message
                preceding = messages[i - 1] if i > 0 else None
                preceding_content = preceding.get("content", "") if preceding else ""
                
                if self.is_repair_turn(msg.get("content", ""), preceding_content):
                    return i
        
        return None
    
    # =========================================================================
    # FORMAT CONSTRAINTS
    # =========================================================================
    
    def extract_format_constraints(self, user_message: str) -> dict:
        """Extract format constraints from user message."""
        constraints = {
            "forbid_bullets": False,
            "require_numbered": False,
            "must_return_code": False,
            "must_return_diff": False,
            "must_return_json": False,
            "must_not_omit": False,
        }
        
        user_lower = user_message.lower()
        
        # Check each constraint
        if any(p in user_lower for p in ["no bullet", "don't use bullet", "without bullet"]):
            constraints["forbid_bullets"] = True
        
        if any(p in user_lower for p in ["numbered list", "numbered steps", "number them"]):
            constraints["require_numbered"] = True
        
        if any(p in user_lower for p in ["in code", "write code", "implement", "function", "class"]):
            constraints["must_return_code"] = True
        
        if any(p in user_lower for p in ["as diff", "in diff", "unified diff"]):
            constraints["must_return_diff"] = True
        
        if any(p in user_lower for p in ["as json", "in json", "json format", "return json"]):
            constraints["must_return_json"] = True
        
        if any(p in user_lower for p in ["don't omit", "don't skip", "include everything", 
                                          "full", "complete", "in its entirety"]):
            constraints["must_not_omit"] = True
        
        return constraints
    
    def build_format_instruction(self, constraints: dict) -> str:
        """Build format instruction for generation."""
        instructions = []
        
        if constraints.get("forbid_bullets"):
            instructions.append("Do NOT use bullet points. Use prose or numbered lists instead.")
        
        if constraints.get("require_numbered"):
            instructions.append("Use numbered lists for any structured content.")
        
        if constraints.get("must_return_code"):
            instructions.append("Include code in your response.")
        
        if constraints.get("must_return_diff"):
            instructions.append("Return output in unified diff format.")
        
        if constraints.get("must_return_json"):
            instructions.append("Return output in valid JSON format.")
        
        if constraints.get("must_not_omit"):
            instructions.append("Include ALL content - do not summarize or omit anything.")
        
        return "\n".join(instructions) if instructions else ""
    
    def validate_format_constraints(
        self,
        response: str,
        constraints: dict,
    ) -> tuple[bool, list[str]]:
        """Validate response against format constraints."""
        errors = []
        
        if constraints.get("forbid_bullets"):
            if re.search(r'^\s*[-*•]\s+', response, re.MULTILINE):
                errors.append("Response contains bullet points (forbidden)")
        
        if constraints.get("require_numbered"):
            if not re.search(r'^\s*\d+[.)]\s+', response, re.MULTILINE):
                errors.append("Response missing numbered list (required)")
        
        if constraints.get("must_return_code"):
            if not re.search(r'```[\s\S]*?```', response):
                errors.append("Response missing code block (required)")
        
        if constraints.get("must_return_json"):
            if not re.search(r'[{\[].*[}\]]', response, re.DOTALL):
                errors.append("Response missing JSON (required)")
        
        return len(errors) == 0, errors
    
    # =========================================================================
    # PHASE DETECTION
    # =========================================================================
    
    def estimate_phase(self, messages: list[dict], turn_index: int) -> int:
        """Estimate conversation phase based on position and content."""
        total_turns = len(messages)
        
        if total_turns == 0:
            return 0
        
        # Position-based estimate
        position_ratio = turn_index / total_turns
        
        if position_ratio < 0.15:
            return 0  # Opening
        elif position_ratio < 0.3:
            return 1  # Context
        elif position_ratio < 0.6:
            return 2  # Solution
        elif position_ratio < 0.8:
            return 3  # Refinement
        elif position_ratio < 0.95:
            return 4  # Synthesis
        else:
            return 5  # Conclusion
    
    def compute_directive_completeness(self, user_message: str) -> float:
        """Compute directive completeness score."""
        try:
            from ..corpus_surgery.classifier import compute_directive_completeness
            return compute_directive_completeness(user_message)
        except ImportError:
            # Fallback implementation
            score = 0.0
            user_lower = user_message.lower()
            
            # Imperative verbs
            imperative_verbs = [
                "rewrite", "generate", "implement", "create", "build",
                "write", "return", "extract", "convert", "refactor",
                "fix", "update", "add", "remove", "change", "modify",
            ]
            
            for verb in imperative_verbs:
                if re.search(rf'^{verb}\b', user_lower) or \
                   re.search(rf'(?:please|can you)\s+{verb}\b', user_lower):
                    score += 0.35
                    break
            
            # Format specification
            format_patterns = [
                r"in json", r"as json", r"as csv", r"in markdown",
                r"don't omit", r"exact", r"no bullet",
            ]
            
            for pattern in format_patterns:
                if re.search(pattern, user_lower):
                    score += 0.25
                    break
            
            # Has code or substantial content
            if '```' in user_message or len(user_message) > 200:
                score += 0.20
            
            return min(1.0, score)

