"""
Prompt Templates for Conversation Worm.

Contains all GPT 5.2 prompts for generating:
- Paraphrases
- Ideal responses
- Extensions
- Phase-appropriate responses
"""

from typing import Optional


# =============================================================================
# PARAPHRASE PROMPTS
# =============================================================================

PARAPHRASE_SYSTEM_PROMPT = """You are a paraphrase generator for training data.

Generate {count} paraphrases of the user message that:
1. Preserve the exact same intent and meaning
2. Use different wording, sentence structure, or phrasing
3. Maintain the same level of formality
4. Keep any technical terms unchanged
5. Preserve any format constraints (JSON, code, etc.)

Output format:
PARAPHRASE 1: ...
PARAPHRASE 2: ...
etc.

Do NOT:
- Add or remove information
- Change the meaning
- Make the message more or less specific
"""


def format_paraphrase_prompt(user_message: str, count: int = 2) -> str:
    """Format prompt for paraphrase generation."""
    return f"""Generate {count} paraphrases for the following user message:

---
{user_message}
---

Remember: Preserve the exact meaning and intent, only change the wording."""


# =============================================================================
# IDEAL RESPONSE PROMPTS
# =============================================================================

IDEAL_RESPONSE_SYSTEM_PROMPT = """You are generating the ideal assistant response for CognitiveTwin V3.

The original assistant response asked for permission or clarification when it shouldn't have.
Generate what the assistant SHOULD have said instead.

RULES:
1. Execute immediately - do not ask permission
2. If assumptions are needed, state them briefly then proceed
3. Produce the requested artifact/output
4. Do NOT end with a question
5. Match the technical level and style of the conversation

ASSUMPTION PROTOCOL:
- State assumptions as: "Assumptions: [brief list]"
- Then proceed with full response
- NO question marks in assumptions

The original conversation context is provided. Generate only the ideal assistant response."""


def format_ideal_response_prompt(
    context: list[dict],
    friction_content: str,
    format_constraints: dict,
) -> str:
    """Format prompt for ideal response generation."""
    context_str = "\n\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in context[-4:]
    ])
    
    constraint_str = ""
    if format_constraints:
        constraints = []
        if format_constraints.get("forbid_bullets"):
            constraints.append("No bullet points")
        if format_constraints.get("require_numbered"):
            constraints.append("Use numbered lists")
        if format_constraints.get("must_return_code"):
            constraints.append("Include code")
        if format_constraints.get("must_return_json"):
            constraints.append("Return JSON")
        if format_constraints.get("must_not_omit"):
            constraints.append("Include ALL content, no omissions")
        
        if constraints:
            constraint_str = f"\nFORMAT CONSTRAINTS: {', '.join(constraints)}"
    
    return f"""CONVERSATION CONTEXT:
{context_str}

USER MESSAGE (that triggered friction):
{context[-1]['content'] if context else 'N/A'}

PROBLEMATIC ASSISTANT RESPONSE (asked permission when shouldn't have):
{friction_content[:1000]}
{constraint_str}

Generate the ideal assistant response that executes immediately:"""


# =============================================================================
# EXTENSION PROMPTS
# =============================================================================

EXTENSION_SYSTEM_PROMPT = """You are extending a conversation for CognitiveTwin V3 training.

Generate a natural continuation of this conversation that:
1. Maintains the same technical depth and topic
2. Shows productive progression (not circular)
3. Follows the established interaction style
4. Does NOT introduce unnecessary questions from the assistant

Generate both the next user message and assistant response.

Output format:
USER: [next user message]
ASSISTANT: [assistant response that executes without asking permission]

The assistant response should:
- Execute immediately if user gives a directive
- Produce concrete output (code, analysis, etc.)
- NOT end with a question unless user explicitly asked for options"""


def format_extension_prompt(context: list[dict]) -> str:
    """Format prompt for extension generation."""
    context_str = "\n\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}"
        for m in context[-6:]
    ])
    
    return f"""CONVERSATION:
{context_str}

Generate the next exchange (user message and assistant response):"""


# =============================================================================
# PHASE-APPROPRIATE RESPONSE PROMPTS
# =============================================================================

PHASE_DESCRIPTIONS = {
    0: "Opening phase - gathering context, clarifying questions acceptable",
    1: "Context phase - deep understanding, some clarification OK",
    2: "Solution phase - actively solving, NO questions, execute immediately",
    3: "Refinement phase - iterating on solution, NO questions, just improve",
    4: "Synthesis phase - summarizing, NO questions, produce deliverables",
    5: "Conclusion phase - final output, NO questions, complete the task",
}


def format_phase_response_prompt(
    prompt: str,
    phase_id: int,
    question_policy: str,
) -> str:
    """Format prompt for phase-appropriate response generation."""
    phase_desc = PHASE_DESCRIPTIONS.get(phase_id, "unknown phase")
    
    policy_instruction = ""
    if question_policy == "no_questions":
        policy_instruction = "Do NOT ask any questions. Execute immediately."
    elif question_policy == "questions_if_required":
        policy_instruction = "You MAY ask clarifying questions if genuinely needed."
    else:
        policy_instruction = "Questions are allowed if helpful."
    
    return f"""You are responding in the {phase_desc} of a conversation.

Question policy: {question_policy}

{policy_instruction}

USER MESSAGE:
{prompt}

Respond appropriately for this phase:"""


def get_phase_system_prompt(phase_id: int, question_policy: str) -> str:
    """Get system prompt for phase-appropriate response."""
    phase_desc = PHASE_DESCRIPTIONS.get(phase_id, "unknown phase")
    
    if question_policy == "no_questions":
        policy_instruction = "Do NOT ask any questions. Execute immediately. Produce concrete output."
    elif question_policy == "questions_if_required":
        policy_instruction = "You MAY ask clarifying questions if genuinely needed, but prefer execution."
    else:
        policy_instruction = "Questions are allowed if helpful."
    
    return f"""You are responding in the {phase_desc} of a conversation.

Question policy: {question_policy}

{policy_instruction}

If you need to make assumptions:
- State them briefly at the start
- Then proceed with execution
- Do NOT ask for confirmation"""


# =============================================================================
# CONTRAST PAIR PROMPTS
# =============================================================================

CONTRAST_GENERATION_PROMPT = """Generate two different responses to the same user message, 
appropriate for different conversation phases.

USER MESSAGE:
{prompt}

PHASE A: {phase_a_desc}
Policy: {policy_a}

PHASE B: {phase_b_desc}
Policy: {policy_b}

Output format:
RESPONSE A (Phase {phase_a}):
[response appropriate for phase A]

RESPONSE B (Phase {phase_b}):
[response appropriate for phase B]
"""


def format_contrast_prompt(
    prompt: str,
    phase_a: int,
    phase_b: int,
    policy_a: str,
    policy_b: str,
) -> str:
    """Format prompt for contrast pair generation."""
    return CONTRAST_GENERATION_PROMPT.format(
        prompt=prompt,
        phase_a=phase_a,
        phase_b=phase_b,
        phase_a_desc=PHASE_DESCRIPTIONS.get(phase_a, "unknown"),
        phase_b_desc=PHASE_DESCRIPTIONS.get(phase_b, "unknown"),
        policy_a=policy_a,
        policy_b=policy_b,
    )


# =============================================================================
# NON-REPAIR TRAJECTORY PROMPT
# =============================================================================

NON_REPAIR_SYSTEM_PROMPT = """You are generating an ideal conversation trajectory where
the assistant responded correctly the first time, so no user correction was needed.

The original conversation had friction - the user had to correct the assistant.
Generate what should have happened instead:
1. The assistant executes immediately without asking permission
2. The user follows up naturally (not with correction)
3. The conversation progresses productively

Output format:
ASSISTANT: [ideal response that executes immediately]
USER: [natural follow-up, NOT a correction]
"""


def format_non_repair_prompt(
    context: list[dict],
    bad_response: str,
    correction: str,
) -> str:
    """Format prompt for non-repair trajectory generation."""
    context_str = "\n\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}"
        for m in context[-4:]
    ])
    
    return f"""CONVERSATION BEFORE FRICTION:
{context_str}

PROBLEMATIC ASSISTANT RESPONSE:
{bad_response[:500]}

USER CORRECTION (what we want to avoid):
{correction[:300]}

Generate the ideal trajectory where correction was never needed:"""

