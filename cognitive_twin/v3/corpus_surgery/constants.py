"""
Constants for Corpus Surgery module.

Contains phrase lists, scoring weights, and thresholds for classification.
"""

# =============================================================================
# STALL SCORE PHRASES
# =============================================================================

# Refusal phrases - HIGHEST PRIORITY (+4 each)
# These are explicit capability denials that should NEVER appear in retrieval
REFUSAL_PHRASES = [
    "i'm sorry, i am not able to",
    "i am not able to",
    "i cannot",
    "i can not",
    "i am an ai and cannot",
    "goes beyond my capabilities",
    "beyond my capabilities",
    "i don't have the ability",
    "i'm not able to",
    "as an ai language model",
    "as a language model",
    "i am unable to",
    "i'm unable to",
    "i do not have access",
    "i don't have access",
    "i am not capable",
    "i'm not capable",
    "i lack the ability",
    "outside my capabilities",
    "not within my capabilities",
    "i am just an ai",
    "i'm just an ai",
    "unfortunately, i cannot",
    "unfortunately i cannot",
    "i apologize, but i cannot",
    "i'm sorry, but i can't",
    "regrettably, i cannot",
]
REFUSAL_SCORE = 4  # Highest penalty

# Strong permission phrases (+3 each)
STRONG_PERMISSION_PHRASES = [
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
    "should we",
]
STRONG_PERMISSION_SCORE = 3

# Option-dumping phrases (+2 each)
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
]
OPTION_DUMPING_SCORE = 2

# Clarification preambles (+1 each)
CLARIFICATION_PREAMBLES = [
    "i need a bit more information",
    "i'll need more context",
    "to help you better",
    "could you provide",
    "what exactly do you mean",
    "could you clarify",
    "to make sure i understand",
    "just to clarify",
    "can you tell me more",
    "what do you mean by",
]
CLARIFICATION_PREAMBLE_SCORE = 1

# End-of-message question mark score
END_QUESTION_SCORE = 1

# =============================================================================
# EXEC SCORE INDICATORS
# =============================================================================

CODE_BLOCK_SCORE = 1
DIFF_MARKER_SCORE = 1
JSON_OBJECT_SCORE = 1
HERE_IS_SCORE = 1
NUMBERED_STEPS_SCORE = 1
COMPLETE_ARTIFACT_SCORE = 2

# =============================================================================
# BLOCKED SCORE MODIFIERS
# =============================================================================

MISSING_INPUT_SCORE = 3
AMBIGUOUS_TARGET_SCORE = 2
FORMAT_SPECIFIED_BONUS = -1
USER_ASKED_OPTIONS_BONUS = -2

# =============================================================================
# CLASSIFICATION THRESHOLDS
# =============================================================================

# Unjustified: stall >= 3 AND blocked <= 1 AND exec == 0
# Lowered thresholds to catch more unjustified turns
# Original: 3, 1, 0 - too strict, only caught 5/2177
STALL_THRESHOLD_UNJUSTIFIED = 1  # Any stalling pattern
BLOCKED_THRESHOLD_UNJUSTIFIED = 1
EXEC_THRESHOLD_UNJUSTIFIED = 0

# Justified: blocked >= 3 AND asks for missing input
BLOCKED_THRESHOLD_JUSTIFIED = 3

# Directive completeness thresholds
DIRECTIVE_HIGH_THRESHOLD = 0.7
DIRECTIVE_MEDIUM_THRESHOLD = 0.4
DIRECTIVE_REWRITE_THRESHOLD = 0.5

# =============================================================================
# FRICTION / FRUSTRATION TRIGGERS
# =============================================================================

FRUSTRATION_TRIGGERS = [
    "stop asking",
    "don't ask",
    "don't do that",
    "i said",
    "just do it",
    "i challenge you",
    "actually,",
    "no, i meant",
    "that's not what i asked",
    "try again",
    "you keep",
    "i already told you",
    "as i mentioned",
    "like i said",
    "for the third time",
    "please just",
]

# =============================================================================
# PROVIDER-ISMS TO REMOVE
# =============================================================================

PROVIDER_ISMS = [
    # AI self-reference
    r"as an ai(?: language model)?",
    r"as a large language model",
    r"i'?m (?:just )?an ai",
    r"i don'?t have (?:personal )?(?:opinions|feelings|preferences)",
    r"i'?m not able to",
    r"i can'?t (?:actually )?(?:browse|access|see)",
    
    # Over-apologizing
    r"i apologize(?:,? but)?",
    r"i'?m sorry(?:,? but)?",
    r"sorry for (?:any )?confusion",
    r"my apologies",
    
    # Over-hedging
    r"it'?s worth noting that",
    r"it'?s important to (?:note|mention|remember) that",
    r"please (?:note|keep in mind) that",
    r"i should mention that",
    
    # Filler acknowledgments
    r"^(?:sure|certainly|absolutely|of course)[,!]?\s*",
    r"^great question[,!]?\s*",
    r"^that'?s a (?:great|good|interesting) (?:question|point)[,!]?\s*",
    
    # Permission closers
    r"would you like me to[^?]*\??\s*$",
    r"do you want me to[^?]*\??\s*$",
    r"shall i[^?]*\??\s*$",
    r"should i[^?]*\??\s*$",
    r"let me know if you[^.]*\.\s*$",
    r"feel free to ask[^.]*\.\s*$",
]

# =============================================================================
# FORMAT CONSTRAINT PATTERNS
# =============================================================================

FORMAT_PATTERNS = {
    "forbid_bullets": [
        r"no bullet",
        r"don't use bullet",
        r"without bullet",
        r"avoid bullet",
        r"not bullet",
    ],
    "require_numbered": [
        r"numbered list",
        r"numbered steps",
        r"number them",
        r"use numbers",
        r"with numbers",
    ],
    "must_return_code": [
        r"in code",
        r"write code",
        r"implement",
        r"as code",
        r"function",
        r"class\b",
        r"method",
    ],
    "must_return_diff": [
        r"as diff",
        r"in diff",
        r"show diff",
        r"unified diff",
    ],
    "must_return_json": [
        r"as json",
        r"in json",
        r"json format",
        r"return json",
    ],
    "must_not_omit": [
        r"don'?t omit",
        r"don'?t skip",
        r"include (?:everything|all)",
        r"full (?:content|text|code)",
        r"complete (?:content|text|code)",
        r"no summariz",
        r"exact (?:copy|rewrite)",
        r"in (?:its )?entirety",
    ],
}

# =============================================================================
# DIRECTIVE COMPLETENESS PATTERNS
# =============================================================================

IMPERATIVE_VERBS = [
    "rewrite", "generate", "implement", "create", "build",
    "write", "return", "extract", "convert", "transform",
    "refactor", "fix", "update", "add", "remove", "delete",
    "change", "modify", "replace", "debug", "test", "analyze",
    "explain", "summarize", "list", "show", "find", "search",
]

FORMAT_SPECIFICATION_PATTERNS = [
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
]

TRANSFORMATION_WORDS = [
    "refactor", "rewrite", "transform", "convert",
    "enhance", "improve", "fix", "update"
]

AMBIGUITY_PATTERNS = [
    r"this or that",
    r"either.+or",
    r"what (?:should|would)",
    r"which (?:one|approach|method)",
    r"how should i",
]

USER_ASKED_OPTIONS_PATTERNS = [
    r"what (?:are )?(?:my |the )?options",
    r"give me (?:some )?options",
    r"list (?:the |some )?options",
    r"what (?:could|can|should) i",
    r"what do you (?:think|suggest|recommend)",
]

# =============================================================================
# ARTIFACT DETECTION
# =============================================================================

ARTIFACT_TRIGGERS = {
    "implement": "code",
    "create": "code",
    "write": "code",
    "generate": "code",
    "build": "code",
    "refactor": "diff",
    "rewrite": "content",
    "transform": "content",
    "convert": "content",
    "analyze": "analysis",
    "explain": "explanation",
    "summarize": "summary",
    "extract": "data",
    "parse": "data",
}

# =============================================================================
# PHASE CONFIGURATION
# =============================================================================

PHASE_QUESTION_POLICIES = {
    0: "questions_if_required",  # Opening
    1: "questions_if_required",  # Context
    2: "no_questions",           # Solution
    3: "no_questions",           # Refinement
    4: "no_questions",           # Synthesis
    5: "no_questions",           # Conclusion
}

