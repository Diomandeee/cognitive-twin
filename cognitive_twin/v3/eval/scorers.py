"""
Scorers for CognitiveTwin V3 Evaluation.

Provides scoring classes for:
- Policy compliance (no permission-seeking, etc.)
- Format adherence (bullets, JSON, etc.)
- Content quality (completeness, correctness, etc.)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .types import (
    TestCase,
    PolicyType,
    FormatConstraint,
    FailureDetail,
)


@dataclass
class PolicyComplianceScore:
    """Score for policy compliance."""
    no_permission_seeking: float = 1.0
    no_question_ending: float = 1.0
    no_option_dumping: float = 1.0
    no_stalling: float = 1.0
    direct_execution: float = 1.0
    
    @property
    def overall(self) -> float:
        """Compute overall policy compliance score."""
        scores = [
            self.no_permission_seeking,
            self.no_question_ending,
            self.no_option_dumping,
            self.no_stalling,
            self.direct_execution,
        ]
        return sum(scores) / len(scores)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "no_permission_seeking": self.no_permission_seeking,
            "no_question_ending": self.no_question_ending,
            "no_option_dumping": self.no_option_dumping,
            "no_stalling": self.no_stalling,
            "direct_execution": self.direct_execution,
            "overall": self.overall,
        }


class PolicyComplianceScorer:
    """Score responses for policy compliance."""
    
    PERMISSION_PHRASES = [
        "would you like me to",
        "do you want me to",
        "should i",
        "shall i",
        "can i proceed",
        "before i proceed",
        "can you confirm",
        "please confirm",
        "let me know if you want",
        "tell me if you want",
        "is that okay",
        "does that work",
        "sound good",
        "would you prefer",
    ]
    
    OPTION_DUMPING_PHRASES = [
        "i can do",
        "here are a few options",
        "here are some options",
        "which approach do you want",
        "pick one of the following",
        "choose between",
        "a few ways to",
        "several approaches",
        "multiple options",
        "we could either",
        "option 1",
        "option 2",
    ]
    
    STALLING_PHRASES = [
        "i need a bit more information",
        "i'll need more context",
        "to help you better",
        "could you provide",
        "what exactly do you mean",
        "could you clarify",
        "to make sure i understand",
        "just to clarify",
        "can you tell me more",
    ]
    
    def score(
        self,
        response: str,
        test_case: TestCase,
    ) -> Tuple[PolicyComplianceScore, List[FailureDetail]]:
        """
        Score a response for policy compliance.
        
        Args:
            response: The model's response
            test_case: The test case being evaluated
            
        Returns:
            Tuple of (score, list of failures)
        """
        response_lower = response.lower()
        failures = []
        score = PolicyComplianceScore()
        
        # Check for permission-seeking
        if PolicyType.NO_PERMISSION_SEEKING in test_case.policy_constraints:
            for phrase in self.PERMISSION_PHRASES:
                if phrase in response_lower:
                    score.no_permission_seeking = 0.0
                    failures.append(FailureDetail(
                        check_type="policy_compliance",
                        expected="No permission-seeking phrases",
                        actual=f"Found: '{phrase}'",
                        message=f"Response contains permission-seeking phrase: '{phrase}'",
                    ))
                    break
        
        # Check for question ending
        if PolicyType.NO_QUESTION_ENDING in test_case.policy_constraints:
            stripped = response.rstrip()
            if stripped.endswith("?"):
                score.no_question_ending = 0.0
                failures.append(FailureDetail(
                    check_type="policy_compliance",
                    expected="No question at end",
                    actual="Response ends with '?'",
                    message="Response ends with a question mark",
                ))
        
        # Check for option-dumping
        if PolicyType.NO_OPTION_DUMPING in test_case.policy_constraints:
            for phrase in self.OPTION_DUMPING_PHRASES:
                if phrase in response_lower:
                    score.no_option_dumping = 0.0
                    failures.append(FailureDetail(
                        check_type="policy_compliance",
                        expected="No option-dumping phrases",
                        actual=f"Found: '{phrase}'",
                        message=f"Response contains option-dumping phrase: '{phrase}'",
                    ))
                    break
        
        # Check for stalling
        if PolicyType.NO_STALLING in test_case.policy_constraints:
            for phrase in self.STALLING_PHRASES:
                if phrase in response_lower:
                    score.no_stalling = 0.0
                    failures.append(FailureDetail(
                        check_type="policy_compliance",
                        expected="No stalling phrases",
                        actual=f"Found: '{phrase}'",
                        message=f"Response contains stalling phrase: '{phrase}'",
                    ))
                    break
        
        # Check for direct execution (should contain code/content)
        if PolicyType.DIRECT_EXECUTION in test_case.policy_constraints:
            has_code = bool(re.search(r"```[\s\S]*?```", response))
            has_implementation = "def " in response or "class " in response
            
            if not (has_code or has_implementation):
                score.direct_execution = 0.5  # Partial credit
                failures.append(FailureDetail(
                    check_type="policy_compliance",
                    expected="Direct execution with code/implementation",
                    actual="No code block or implementation found",
                    message="Response lacks direct execution (no code provided)",
                ))
        
        # Check disallowed phrases from test case
        for phrase in test_case.disallowed_phrases:
            if phrase.lower() in response_lower:
                # Determine which score to penalize
                if any(p in phrase.lower() for p in ["would you", "should i", "shall i"]):
                    score.no_permission_seeking = 0.0
                elif "option" in phrase.lower() or "approach" in phrase.lower():
                    score.no_option_dumping = 0.0
                
                failures.append(FailureDetail(
                    check_type="disallowed_phrase",
                    expected=f"Should not contain '{phrase}'",
                    actual=f"Found: '{phrase}'",
                    message=f"Response contains disallowed phrase: '{phrase}'",
                ))
        
        return score, failures


@dataclass
class FormatAdherenceScore:
    """Score for format constraint adherence."""
    respects_no_bullets: float = 1.0
    respects_numbered: float = 1.0
    respects_json: float = 1.0
    respects_no_omit: float = 1.0
    respects_code: float = 1.0
    
    @property
    def overall(self) -> float:
        """Compute overall format adherence score."""
        scores = [
            self.respects_no_bullets,
            self.respects_numbered,
            self.respects_json,
            self.respects_no_omit,
            self.respects_code,
        ]
        return sum(scores) / len(scores)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "respects_no_bullets": self.respects_no_bullets,
            "respects_numbered": self.respects_numbered,
            "respects_json": self.respects_json,
            "respects_no_omit": self.respects_no_omit,
            "respects_code": self.respects_code,
            "overall": self.overall,
        }


class FormatAdherenceScorer:
    """Score responses for format constraint adherence."""
    
    OMISSION_PATTERNS = [
        r"\.\.\.",
        r"# \.\.\.",
        r"// \.\.\.",
        r"\(omitted\)",
        r"\(truncated\)",
        r"\(rest of",
        r"# rest of",
        r"# remaining",
        r"# same as",
        r"# similar",
    ]
    
    BULLET_PATTERNS = [
        r"^\s*[-*•]\s+",  # Markdown bullets
        r"^\s*[○◦●]\s+",  # Unicode bullets
    ]
    
    def score(
        self,
        response: str,
        test_case: TestCase,
    ) -> Tuple[FormatAdherenceScore, List[FailureDetail]]:
        """
        Score a response for format adherence.
        
        Args:
            response: The model's response
            test_case: The test case being evaluated
            
        Returns:
            Tuple of (score, list of failures)
        """
        failures = []
        score = FormatAdherenceScore()
        
        # Check no bullets
        if FormatConstraint.NO_BULLETS in test_case.format_constraints:
            for pattern in self.BULLET_PATTERNS:
                if re.search(pattern, response, re.MULTILINE):
                    score.respects_no_bullets = 0.0
                    failures.append(FailureDetail(
                        check_type="format_adherence",
                        expected="No bullet points",
                        actual="Found bullet points",
                        message="Response contains bullet points when forbidden",
                    ))
                    break
        
        # Check numbered list
        if FormatConstraint.REQUIRE_NUMBERED in test_case.format_constraints:
            numbered_pattern = r"^\s*\d+[.)]\s+"
            matches = re.findall(numbered_pattern, response, re.MULTILINE)
            if len(matches) < 2:
                score.respects_numbered = 0.0
                failures.append(FailureDetail(
                    check_type="format_adherence",
                    expected="Numbered list format",
                    actual=f"Found {len(matches)} numbered items",
                    message="Response should use numbered list format",
                ))
        
        # Check JSON format
        if FormatConstraint.REQUIRE_JSON in test_case.format_constraints:
            try:
                import json
                # Try to find JSON in response
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found")
            except (json.JSONDecodeError, ValueError):
                score.respects_json = 0.0
                failures.append(FailureDetail(
                    check_type="format_adherence",
                    expected="Valid JSON format",
                    actual="No valid JSON found",
                    message="Response should contain valid JSON",
                ))
        
        # Check no omission
        if FormatConstraint.NO_OMIT in test_case.format_constraints:
            for pattern in self.OMISSION_PATTERNS:
                if re.search(pattern, response, re.IGNORECASE):
                    score.respects_no_omit = 0.0
                    failures.append(FailureDetail(
                        check_type="format_adherence",
                        expected="No omission markers",
                        actual=f"Found omission pattern: {pattern}",
                        message="Response contains omission markers",
                    ))
                    break
        
        # Check code block
        if FormatConstraint.REQUIRE_CODE in test_case.format_constraints:
            if not re.search(r"```[\s\S]*?```", response):
                score.respects_code = 0.0
                failures.append(FailureDetail(
                    check_type="format_adherence",
                    expected="Code block present",
                    actual="No code block found",
                    message="Response should contain a code block",
                ))
        
        # Check required phrases
        for phrase in test_case.required_phrases:
            if phrase not in response:
                failures.append(FailureDetail(
                    check_type="required_phrase",
                    expected=f"Should contain '{phrase}'",
                    actual="Phrase not found",
                    message=f"Response missing required phrase: '{phrase}'",
                ))
        
        # Check disallowed phrases for format
        for phrase in test_case.disallowed_phrases:
            if phrase in response:
                if "..." in phrase or "omit" in phrase.lower():
                    score.respects_no_omit = 0.0
        
        return score, failures


@dataclass
class ContentQualityScore:
    """Score for content quality."""
    completeness: float = 1.0
    correctness: float = 1.0
    code_validity: float = 1.0
    relevance: float = 1.0
    
    @property
    def overall(self) -> float:
        """Compute overall content quality score."""
        scores = [
            self.completeness,
            self.correctness,
            self.code_validity,
            self.relevance,
        ]
        return sum(scores) / len(scores)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "completeness": self.completeness,
            "correctness": self.correctness,
            "code_validity": self.code_validity,
            "relevance": self.relevance,
            "overall": self.overall,
        }


class ContentQualityScorer:
    """Score responses for content quality."""
    
    def score(
        self,
        response: str,
        test_case: TestCase,
    ) -> Tuple[ContentQualityScore, List[FailureDetail]]:
        """
        Score a response for content quality.
        
        Args:
            response: The model's response
            test_case: The test case being evaluated
            
        Returns:
            Tuple of (score, list of failures)
        """
        failures = []
        score = ContentQualityScore()
        
        # Check completeness via required phrases
        required_found = 0
        for phrase in test_case.required_phrases:
            if phrase in response:
                required_found += 1
        
        if test_case.required_phrases:
            score.completeness = required_found / len(test_case.required_phrases)
            
            if score.completeness < 1.0:
                missing = [p for p in test_case.required_phrases if p not in response]
                failures.append(FailureDetail(
                    check_type="content_quality",
                    expected=f"All required phrases present",
                    actual=f"Missing: {missing[:3]}...",
                    message=f"Response missing {len(missing)} required phrases",
                ))
        
        # Check code validity (basic syntax check)
        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
        if code_blocks:
            valid_blocks = 0
            for code in code_blocks:
                try:
                    compile(code, "<string>", "exec")
                    valid_blocks += 1
                except SyntaxError:
                    pass
            
            score.code_validity = valid_blocks / len(code_blocks) if code_blocks else 1.0
            
            if score.code_validity < 1.0:
                failures.append(FailureDetail(
                    check_type="content_quality",
                    expected="Valid Python syntax",
                    actual=f"{len(code_blocks) - valid_blocks} invalid code blocks",
                    message="Some code blocks have syntax errors",
                ))
        
        # Check relevance (response length and keyword overlap)
        if len(response) < 50:
            score.relevance = 0.5
            failures.append(FailureDetail(
                check_type="content_quality",
                expected="Substantial response",
                actual=f"Response length: {len(response)} chars",
                message="Response seems too short",
            ))
        
        return score, failures
