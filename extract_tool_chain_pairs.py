#!/usr/bin/env python3
"""V3 extractor: Tool-chain-augmented (question, answer) triples for cognitive twin training.

Key insight: The tool call chain provides the context that makes responses meaningful.
"Continue" after a successful build is different from "continue" after a failed test.

Pipeline:
  1. Parse all per-session JSONL files (raw Claude session format)
  2. For each (assistant_text, user_response) pair, extract:
     - The tool calls from the assistant turn (name + brief result from subsequent tool_result)
     - Project name inferred from cwd/file paths in tool calls
     - Machine name if present
  3. Format as ChatML with dynamic system prompt containing tool chain context
  4. Merge in enhanced synthetic pairs (from v2) with synthetic tool chains added
  5. Filter, dedup, split, save
"""

import json
import glob
import hashlib
import os
import random
import re
import socket
import sys
from collections import Counter
from pathlib import Path

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
SESSION_BASE = os.path.expanduser("~/.claude/projects")
OUTPUT_DIR = os.path.expanduser("~/projects/karl/autocontinue-v3")
V2_ENHANCED_DIR = os.path.expanduser("~/projects/karl/autocontinue-enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_RESPONSE_LEN = 300  # Mohamed's responses are short and direct
MAX_TOOL_CHAIN = 5      # Last N tool calls to include in system prompt
MAX_RESULT_SNIPPET = 80 # Truncate tool results to this length

# Responses that violate Mohamed's style or aren't useful for training
BAD_RESPONSE_PATTERNS = [
    re.compile(r"^i don'?t know", re.IGNORECASE),
    re.compile(r"^not sure", re.IGNORECASE),
    re.compile(r"^hmm+$", re.IGNORECASE),
    re.compile(r"^lol$", re.IGNORECASE),
    re.compile(r"^haha$", re.IGNORECASE),
    re.compile(r"^Previous session context:", re.IGNORECASE),
    re.compile(r"^Changes committed and pushed", re.IGNORECASE),
]

# Patterns that indicate secrets/credentials in the response
SECRET_PATTERNS = [
    re.compile(r"sk-ant-api[a-zA-Z0-9_-]{20,}"),       # Anthropic API keys
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),                  # OpenAI-style API keys
    re.compile(r"AIza[a-zA-Z0-9_-]{30,}"),                # Google API keys
    re.compile(r"ghp_[a-zA-Z0-9]{30,}"),                  # GitHub PATs
    re.compile(r"eyJ[a-zA-Z0-9_-]{50,}"),                 # JWT tokens
    re.compile(r"AKIA[A-Z0-9]{16}"),                       # AWS access keys
    re.compile(r"supabase.*service_role.*eyJ", re.IGNORECASE),
    re.compile(r"password\s*[:=]\s*\S{8,}", re.IGNORECASE),
]

# ── Skip patterns ──────────────────────────────────────────────────────────────
SKIP_PATTERNS = [
    re.compile(r"^<task-notification>"),
    re.compile(r"^<objective>"),
    re.compile(r"^Implement the following plan"),
    re.compile(r"^/[a-z]"),  # slash commands
    re.compile(r"^#\s"),     # markdown headers (pasted plans)
    re.compile(r"^```"),     # code blocks
    re.compile(r"^Read the output file"),
    re.compile(r"^Full transcript available"),
    re.compile(r"^<system"),
    re.compile(r"^<context"),
]

# ── Project detection ──────────────────────────────────────────────────────────
PROJECT_PATTERNS = {
    "Spore": re.compile(r"(?i)(?:Desktop/Spore|spore[\-_]|SporeApp)"),
    "CreativeDirector": re.compile(r"(?i)(?:Desktop/CreativeDirector|CreativeDirector)"),
    "OpenClawHub": re.compile(r"(?i)(?:Desktop/OpenClawHub|OpenClawHub)"),
    "SecuriClaw": re.compile(r"(?i)(?:Desktop/SecuriClaw|SecuriClaw)"),
    "SpeakFlow": re.compile(r"(?i)(?:Desktop/SpeakFlow|SpeakFlow)"),
    "FirstDate": re.compile(r"(?i)(?:Desktop/FirstDate|FirstDate)"),
    "NKoScribe": re.compile(r"(?i)(?:Desktop/NKoScribe|NKoScribe)"),
    "SerenityScribe": re.compile(r"(?i)(?:Serenity[\s_-]?Soother|serenity[\s_-]?scribe)"),
    "KoatjiField": re.compile(r"(?i)(?:Desktop/KoatjiField|koatji)"),
    "NexusPortal": re.compile(r"(?i)(?:nexus[\s_-]?portal)"),
    "FeedHub": re.compile(r"(?i)(?:feed[\s_-]?hub|flows/feed-hub)"),
    "GraphKernel": re.compile(r"(?i)(?:graph[\s_-]?kernel)"),
    "AgentIntelligence": re.compile(r"(?i)(?:agent[\s_-]?intelligence)"),
    "EvolutionWorld": re.compile(r"(?i)(?:evolution[\s_-]?world)"),
    "KARL": re.compile(r"(?i)(?:projects/karl|karl[\s_-]trajectory)"),
    "Epoch": re.compile(r"(?i)(?:epoch/|stacks[\-_]appchain)"),
    "PromptLogger": re.compile(r"(?i)(?:prompt[\s_-]?logger)"),
    "Cortex": re.compile(r"(?i)(?:cortex/)"),
    "MeshInfra": re.compile(r"(?i)(?:monitoring/|docker[\s_-]?compose)"),
}

MACHINE_PATTERNS = {
    "mac1": re.compile(r"(?i)mac[\s_-]?1(?!\d)"),
    "mac2": re.compile(r"(?i)mac[\s_-]?2(?!\d)"),
    "mac3": re.compile(r"(?i)mac[\s_-]?3(?!\d)"),
    "mac4": re.compile(r"(?i)mac[\s_-]?4(?!\d)"),
    "mac5": re.compile(r"(?i)mac[\s_-]?5(?!\d)"),
    "cloud-vm": re.compile(r"(?i)(?:cloud[\s_-]?vm|100\.114\.92\.88)"),
}


# ── Tool result extraction ─────────────────────────────────────────────────────

def extract_tool_result_snippet(content) -> str:
    """Extract a brief snippet from a tool_result content block."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        text = "\n".join(texts)
    else:
        return ""

    text = text.strip()
    if not text:
        return ""

    # Detect success/failure patterns
    lower = text.lower()
    if "build succeeded" in lower or "** build" in lower:
        return "BUILD SUCCEEDED"
    if "build failed" in lower:
        return "BUILD FAILED"
    if "error:" in lower and len(text) < 200:
        # Extract just the error line
        for line in text.split("\n"):
            if "error:" in line.lower():
                return line.strip()[:MAX_RESULT_SNIPPET]
    if "exit code" in lower:
        for line in text.split("\n"):
            if "exit code" in line.lower():
                return line.strip()[:MAX_RESULT_SNIPPET]
    if "** test succeeded" in lower:
        return "TEST SUCCEEDED"
    if "** test failed" in lower:
        return "TEST FAILED"
    if text.startswith("Ripgrep search timed out"):
        return "search timed out"
    if "no matches found" in lower:
        return "no matches"

    # For file listings, just note the count
    lines = text.split("\n")
    if len(lines) > 5 and all("/" in l for l in lines[:3] if l.strip()):
        return f"{len(lines)} files listed"

    # For code/file reads, just note it was read
    if text.startswith("     1"):
        return "file content read"

    # Default: first meaningful line, truncated
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            return line[:MAX_RESULT_SNIPPET]

    return text[:MAX_RESULT_SNIPPET]


def detect_project(text_blob: str) -> str:
    """Detect project name from accumulated text (cwd, file paths, content)."""
    for name, pat in PROJECT_PATTERNS.items():
        if pat.search(text_blob):
            return name
    return ""


def detect_machine(text_blob: str) -> str:
    """Detect machine from text."""
    for name, pat in MACHINE_PATTERNS.items():
        if pat.search(text_blob):
            return name
    # Default: detect from hostname
    hostname = socket.gethostname().lower()
    if "mac1" in hostname or "mohameddiomande" in hostname:
        return "mac1"
    return ""


# ── Session parsing ────────────────────────────────────────────────────────────

def parse_session_v3(jsonl_path: str) -> list[dict]:
    """Parse a session JSONL and extract conversation turns with tool chain context.

    Returns list of dicts:
    {
        "role": "user" | "assistant",
        "content": str,            # Text content only
        "tool_calls": list[dict],  # For assistant turns: [{name, input_snippet, result_snippet}]
        "cwd": str,
        "session_id": str,
    }
    """
    entries = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (IOError, OSError):
        return []

    # First pass: build tool_use_id -> result mapping
    tool_results = {}
    for e in entries:
        if e.get("type") != "user":
            continue
        msg = e.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id:
                    tool_results[tool_use_id] = extract_tool_result_snippet(
                        block.get("content", "")
                    )

    # Second pass: build conversation turns
    turns = []
    seen_content = set()
    session_cwd = ""

    for e in entries:
        entry_type = e.get("type", "")
        if entry_type not in ("user", "assistant"):
            continue

        cwd = e.get("cwd", "")
        if cwd:
            session_cwd = cwd
        session_id = e.get("sessionId", "")

        msg = e.get("message", {})
        if not isinstance(msg, dict):
            continue

        content_raw = msg.get("content", "")

        if entry_type == "assistant":
            # Extract text blocks and tool_use blocks
            text_parts = []
            tool_calls = []

            if isinstance(content_raw, list):
                for block in content_raw:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tc = {
                            "name": block.get("name", "unknown"),
                            "input_snippet": _tool_input_snippet(block.get("input", {})),
                        }
                        # Look up result
                        tu_id = block.get("id", "")
                        if tu_id in tool_results:
                            tc["result_snippet"] = tool_results[tu_id]
                        tool_calls.append(tc)
            elif isinstance(content_raw, str):
                text_parts.append(content_raw)

            text = "\n".join(text_parts).strip()

            # If this is a tool-only turn (no text), merge tool calls into previous
            # assistant turn or create a minimal turn
            if not text and tool_calls:
                if turns and turns[-1]["role"] == "assistant":
                    turns[-1]["tool_calls"].extend(tool_calls)
                    continue
                else:
                    turns.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                        "cwd": session_cwd,
                        "session_id": session_id,
                    })
                    continue

            if not text:
                # Also merge tool calls if present
                if tool_calls and turns and turns[-1]["role"] == "assistant":
                    turns[-1]["tool_calls"].extend(tool_calls)
                continue

            content_hash = hashlib.md5(text.encode()).hexdigest()
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            if turns and turns[-1]["role"] == "assistant":
                turns[-1]["content"] += "\n\n" + text
                turns[-1]["tool_calls"].extend(tool_calls)
            else:
                turns.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                    "cwd": session_cwd,
                    "session_id": session_id,
                })

        elif entry_type == "user":
            # Skip tool_result-only user messages
            if isinstance(content_raw, list):
                has_text = False
                text_parts = []
                for block in content_raw:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text", "").strip()
                        if t:
                            text_parts.append(t)
                            has_text = True
                if not has_text:
                    continue
                text = "\n".join(text_parts).strip()
            elif isinstance(content_raw, str):
                text = content_raw.strip()
            else:
                continue

            if not text:
                continue

            content_hash = hashlib.md5(text.encode()).hexdigest()
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            if turns and turns[-1]["role"] == "user":
                turns[-1]["content"] += "\n\n" + text
            else:
                turns.append({
                    "role": "user",
                    "content": text,
                    "tool_calls": [],
                    "cwd": session_cwd,
                    "session_id": session_id,
                })

    return turns


def _tool_input_snippet(inp: dict) -> str:
    """Create a brief summary of tool input."""
    if not isinstance(inp, dict):
        return ""

    # Bash: show the command
    if "command" in inp:
        cmd = inp["command"]
        if len(cmd) > 100:
            cmd = cmd[:97] + "..."
        return cmd

    # Read: show file path
    if "file_path" in inp:
        return inp["file_path"].split("/")[-1]

    # Write: show file path
    if "file_path" in inp:
        return f"write {inp['file_path'].split('/')[-1]}"

    # Edit: show file + old_string start
    if "old_string" in inp:
        fp = inp.get("file_path", "")
        return f"edit {fp.split('/')[-1]}" if fp else "edit"

    # Glob: show pattern
    if "pattern" in inp:
        return inp["pattern"]

    # Grep: show pattern
    if "pattern" in inp:
        return f"grep {inp['pattern'][:50]}"

    # WebFetch/WebSearch
    if "url" in inp:
        return inp["url"][:60]
    if "query" in inp:
        return inp["query"][:60]

    return ""


# ── Context extraction ─────────────────────────────────────────────────────────

def extract_assistant_question(text: str, max_len: int = 500) -> str:
    """Extract the last meaningful chunk from assistant text (the question/proposal)."""
    text = text.rstrip()
    if not text:
        return ""

    paragraphs = text.split("\n\n")

    # Find last non-empty paragraph
    for para in reversed(paragraphs):
        para = para.strip()
        if para and len(para) > 10:
            if len(para) > max_len:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                result = ""
                for s in reversed(sentences):
                    if len(result) + len(s) + 2 > max_len:
                        break
                    result = s + " " + result if result else s
                return result.strip()
            return para

    lines = text.split("\n")
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:max_len]
    return text[-max_len:].strip()


def format_tool_chain(tool_calls: list[dict]) -> str:
    """Format a tool chain as concise bullet points for the system prompt.

    If there are more than MAX_TOOL_CHAIN calls, summarize the earlier ones
    and show the last few in detail.
    """
    if not tool_calls:
        return ""

    lines = []

    if len(tool_calls) > MAX_TOOL_CHAIN:
        # Summarize the earlier calls by tool name frequency
        earlier = tool_calls[:-MAX_TOOL_CHAIN]
        name_counts = Counter(tc.get("name", "?") for tc in earlier)
        summary_parts = [f"{name}x{count}" for name, count in name_counts.most_common(5)]
        lines.append(f"- (earlier: {', '.join(summary_parts)})")

    # Show last MAX_TOOL_CHAIN calls in detail
    recent = tool_calls[-MAX_TOOL_CHAIN:]
    for tc in recent:
        name = tc.get("name", "unknown")
        inp = tc.get("input_snippet", "")
        result = tc.get("result_snippet", "")

        if result:
            lines.append(f"- {name}: {inp} -> {result}")
        elif inp:
            lines.append(f"- {name}: {inp}")
        else:
            lines.append(f"- {name}")

    return "\n".join(lines)


def redact_secrets(text: str) -> str:
    """Remove any secrets/credentials from text."""
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    # Also catch common patterns: anon keys, service role keys, etc
    text = re.sub(r"eyJ[a-zA-Z0-9_=-]{40,}", "[REDACTED_JWT]", text)
    text = re.sub(r"AIza[a-zA-Z0-9_-]{20,}", "[REDACTED_GAPI]", text)
    text = re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED_KEY]", text)
    text = re.sub(r"ghp_[a-zA-Z0-9]{20,}", "[REDACTED_GH]", text)
    text = re.sub(r"AKIA[A-Z0-9]{12,}", "[REDACTED_AWS]", text)
    return text


def build_system_prompt(
    tool_chain_str: str,
    project: str,
    machine: str,
) -> str:
    """Build a dynamic system prompt with tool chain context."""
    base = (
        "You are Mohamed's cognitive twin. Respond as Mohamed would based on "
        "the context of what just happened."
    )

    parts = [base]

    if tool_chain_str:
        # Redact any secrets that may have leaked into tool output
        tool_chain_str = redact_secrets(tool_chain_str)
        parts.append(f"\nRecent tool calls:\n{tool_chain_str}")

    meta = []
    if project:
        meta.append(f"Active project: {project}")
    if machine:
        meta.append(f"Machine: {machine}")
    if meta:
        parts.append("\n" + "\n".join(meta))

    return "\n".join(parts)


# ── Filtering ──────────────────────────────────────────────────────────────────

def is_valid_response(text: str) -> bool:
    """Check if a user message is a valid training response."""
    text = text.strip()
    if not text or len(text) < 5:
        return False
    if len(text) > MAX_RESPONSE_LEN:
        return False
    for pat in SKIP_PATTERNS:
        if pat.search(text):
            return False
    for pat in BAD_RESPONSE_PATTERNS:
        if pat.search(text):
            return False
    # Skip if just a URL
    if text.startswith("http") and "\n" not in text:
        return False
    # Skip if just a file path
    if text.startswith("/") and "\n" not in text and " " not in text:
        return False
    # Skip em-dash-heavy content (likely AI-generated, not Mohamed)
    if text.count("—") > 2:
        return False
    # Skip responses containing secrets/credentials
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            return False
    return True


def classify_response(answer: str) -> str:
    """Classify the type of response for stats."""
    lower = answer.lower().strip().rstrip(".,!?")

    affirmatives = {
        "yes", "yeah", "yep", "sure", "ok", "okay", "go", "do it",
        "continue", "proceed", "next", "ship it", "push it", "go for it",
        "run it", "build it", "deploy", "commit", "fix it", "keep going",
        "lets go", "let's go", "go ahead", "bump it", "yes do it",
    }
    if lower in affirmatives:
        return "affirmative"

    if lower in ("status", "status report", "what's the status", "show me", "check"):
        return "status_request"

    # Correction / redirect
    if any(w in lower for w in ("no ", "don't", "not that", "instead", "actually", "wait")):
        return "correction"

    if len(answer) < 60:
        return "short_directive"
    if len(answer) < 150:
        return "medium_directive"
    return "long_directive"


def clean_response(text: str) -> str:
    """Clean Mohamed's response text."""
    # Remove XML tags
    text = re.sub(r"<[^>]+>", "", text).strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Apply Mohamed's style: no em dashes
    text = text.replace(" — ", ", ").replace("—", ", ")
    return text


# ── Synthetic pairs with tool chains ──────────────────────────────────────────

def generate_synthetic_pairs() -> list[dict]:
    """Generate synthetic training pairs WITH tool chain context.
    These cover scenarios that may be underrepresented in logs."""

    pairs = []

    # ── Build & Deploy ──
    pairs.extend([
        {
            "tool_chain": "- Bash: xcodebuild archive -scheme Spore -> BUILD SUCCEEDED\n- Bash: xcodebuild -exportArchive -> EXPORT SUCCEEDED",
            "project": "Spore",
            "machine": "mac1",
            "question": "Archive and export succeeded. Want me to upload to TestFlight?",
            "answer": "Yeah upload it. Use the standard altool command with the API key.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Bash: xcodebuild build -scheme Spore -> BUILD FAILED\n- Read: SporeApp.swift",
            "project": "Spore",
            "machine": "mac1",
            "question": "Build failed with a missing import error in SporeApp.swift. Should I fix it?",
            "answer": "Fix the import and rebuild. Don't stop.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Bash: xcodebuild test -> TEST SUCCEEDED\n- Bash: git status",
            "project": "SpeakFlow",
            "machine": "mac1",
            "question": "All tests passed. 14 tests, 0 failures. Want me to commit these changes?",
            "answer": "Commit and push. Then move to the next feature.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Bash: xcodebuild archive -> BUILD SUCCEEDED\n- Bash: xcrun altool --upload-app -> error: Could not authenticate",
            "project": "CreativeDirector",
            "machine": "mac1",
            "question": "Upload failed with authentication error. The API key might have expired. Should I check the issuer ID?",
            "answer": "Check the issuer ID at ~/.config/asc-issuer-id.txt. Make sure you're using 26YT8HF7KR.",
            "category": "medium_directive",
        },
        {
            "tool_chain": "- Bash: docker compose up -d -> started 8 services\n- Bash: curl localhost:8001/health -> ok",
            "project": "MeshInfra",
            "machine": "cloud-vm",
            "question": "All services healthy. Grafana, Prometheus, and Graph Kernel are up. Anything else to check?",
            "answer": "Check the Prefect server too. Make sure flows are registered.",
            "category": "short_directive",
        },

        # ── Code editing flow ──
        {
            "tool_chain": "- Read: GrowthEngine.swift\n- Edit: GrowthEngine.swift\n- Edit: SporeApp.swift",
            "project": "Spore",
            "machine": "mac1",
            "question": "I've updated the growth algorithm and wired it into the app. Want me to build and test?",
            "answer": "Build it. Run on the simulator first.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Glob: **/*.py -> 14 files\n- Read: daemon.py\n- Edit: daemon.py\n- Read: requirements.txt",
            "project": "FeedHub",
            "machine": "mac1",
            "question": "The daemon has a missing import for the new supabase client. I can either add it to requirements.txt or use the existing one. Which approach?",
            "answer": "Use the existing client. Don't add new deps if we already have what we need.",
            "category": "medium_directive",
        },

        # ── Correction scenarios ──
        {
            "tool_chain": "- Write: newfile.swift\n- Bash: xcodegen generate",
            "project": "Spore",
            "machine": "mac1",
            "question": "I created a new wrapper file that re-exports from the existing service. Ready to build?",
            "answer": "No, delete the wrapper. The existing service handles that directly. Never create thin wrappers.",
            "category": "correction",
        },
        {
            "tool_chain": "- Bash: git push origin main",
            "project": "OpenClawHub",
            "machine": "mac1",
            "question": "Pushed to main. Want me to write a summary of the changes?",
            "answer": "No summary needed, I can read the diff. Move to the next thing.",
            "category": "correction",
        },
        {
            "tool_chain": "- Bash: ssh cloud-vm docker compose restart\n- Bash: curl localhost:3001 -> connection refused",
            "project": "MeshInfra",
            "machine": "mac1",
            "question": "Nexus portal isn't responding after restart. Should I check the Docker logs?",
            "answer": "Check the logs. Probably a port conflict or the container didn't come up.",
            "category": "short_directive",
        },

        # ── Multi-step execution ──
        {
            "tool_chain": "- Bash: python3 spore_evolution_daemon.py --deep -> Phase 1 SENSE complete\n- Bash: Phase 2 SCORE complete",
            "project": "Spore",
            "machine": "mac1",
            "question": "Deep evolution daemon completed phases 1-2. Want me to continue with the remaining phases or check the intermediate results?",
            "answer": "Continue through all phases. Check results at the end, not in between.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Read: extract_v3.py\n- Bash: python3 extract_v3.py -> 847 examples extracted",
            "project": "KARL",
            "machine": "mac1",
            "question": "Extracted 847 training examples. Should I run the quality analysis before starting the training?",
            "answer": "Show me the category distribution first. Then we train.",
            "category": "short_directive",
        },

        # ── Deployment decisions ──
        {
            "tool_chain": "- Bash: supabase functions deploy spore-feedback -> deployed\n- Bash: supabase functions deploy spore-plant -> deployed",
            "project": "Spore",
            "machine": "mac1",
            "question": "Both edge functions deployed. Want me to test them with a sample payload?",
            "answer": "Test them. Hit both endpoints with real data, not mock payloads.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Bash: prefect deployment run 'spore-evolution-daemon/deep' -> Scheduled\n- Bash: prefect flow-run ls -> 3 running",
            "project": "FeedHub",
            "machine": "cloud-vm",
            "question": "Deep evolution daemon scheduled. There are 3 flows already running. Should I wait or let it queue?",
            "answer": "Let it queue. Prefect handles concurrency, don't intervene.",
            "category": "short_directive",
        },

        # ── Research/investigation ──
        {
            "tool_chain": "- Grep: 'error' -> 23 matches in 8 files\n- Read: error_handler.py",
            "project": "AgentIntelligence",
            "machine": "cloud-vm",
            "question": "Found 23 error references across 8 files. Most are in error_handler.py. Want me to categorize them?",
            "answer": "Just fix the ones that are actual bugs. Ignore the error handling boilerplate.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Bash: git log --oneline -20\n- Bash: git diff HEAD~3",
            "project": "Spore",
            "machine": "mac1",
            "question": "Last 3 commits were all build fixes. The feature hasn't been tested yet. Want me to run the test suite?",
            "answer": "Run tests. If they pass, commit and move on.",
            "category": "short_directive",
        },

        # ── Continue variants with different contexts ──
        {
            "tool_chain": "- Write: TopicClassifier.swift\n- Write: BehaviorTracker.swift\n- Write: TierBroker.swift",
            "project": "Spore",
            "machine": "mac1",
            "question": "Wave 1 foundation files created. Should I move to Wave 2 core logic?",
            "answer": "Build first. Make sure Wave 1 compiles before moving to Wave 2.",
            "category": "short_directive",
        },
        {
            "tool_chain": "- Bash: npm test -> 66 tests, 66 passing\n- Bash: npm run build -> compiled successfully",
            "project": "NexusPortal",
            "machine": "cloud-vm",
            "question": "All 66 tests passing and build succeeded. Ready to deploy?",
            "answer": "Deploy. Then check it in the browser to make sure the new page renders.",
            "category": "short_directive",
        },

        # ── SSH/infra scenarios ──
        {
            "tool_chain": "- Bash: ssh cloud-vm 'systemctl status prefect' -> active (running)\n- Bash: ssh cloud-vm 'docker ps' -> 12 containers",
            "project": "MeshInfra",
            "machine": "mac1",
            "question": "Prefect is running, 12 Docker containers healthy. Want me to check Prometheus metrics?",
            "answer": "Check Prometheus. Also make sure the Grafana dashboards are loading.",
            "category": "medium_directive",
        },
        {
            "tool_chain": "- Bash: tailscale status -> 5 machines online\n- Bash: ssh mac4 'uptime' -> up 14 days",
            "project": "MeshInfra",
            "machine": "mac1",
            "question": "All 5 machines in the mesh are online. Mac4 has been up 14 days. Want me to check disk space?",
            "answer": "Check disk on Mac4 and cloud-vm. Those fill up fastest.",
            "category": "short_directive",
        },

        # ── Error recovery ──
        {
            "tool_chain": "- Bash: python3 daemon.py -> ModuleNotFoundError: No module named 'prefect'\n- Bash: pip install prefect",
            "project": "FeedHub",
            "machine": "cloud-vm",
            "question": "Prefect wasn't installed in this environment. I've installed it. Should I retry the daemon?",
            "answer": "Retry. Make sure to add prefect to requirements.txt so this doesn't happen again.",
            "category": "medium_directive",
        },
        {
            "tool_chain": "- Bash: xcodebuild build -> error: Signing for 'Spore' requires a development team\n- Read: project.yml",
            "project": "Spore",
            "machine": "mac1",
            "question": "Build failed due to signing. The project.yml is missing the team ID. Should I add it?",
            "answer": "Add the team ID. Use the standard OpenClaw team.",
            "category": "short_directive",
        },

        # ── Nuanced decision-making ──
        {
            "tool_chain": "- Bash: wc -l train.jsonl -> 2498 examples\n- Bash: python3 -c 'import json...' -> avg loss 1.84",
            "project": "KARL",
            "machine": "mac5",
            "question": "Current dataset has 2498 examples with avg loss 1.84. Should we train with this or wait for more data?",
            "answer": "Train with what we have. We can retrain when we get more data. Don't block on perfect.",
            "category": "medium_directive",
        },
        {
            "tool_chain": "- Bash: git stash\n- Bash: git pull origin main -> CONFLICT in SporeApp.swift",
            "project": "Spore",
            "machine": "mac1",
            "question": "There's a merge conflict in SporeApp.swift after pulling main. Should I resolve it or rebase instead?",
            "answer": "Resolve the conflict. Keep both changes if possible. Rebase is overkill here.",
            "category": "medium_directive",
        },
    ])

    return pairs


# ── Main extraction ────────────────────────────────────────────────────────────

def discover_session_files() -> list[str]:
    """Find all session JSONL files across all project dirs."""
    all_files = []
    if not os.path.isdir(SESSION_BASE):
        return all_files

    for d in os.listdir(SESSION_BASE):
        full = os.path.join(SESSION_BASE, d)
        if not os.path.isdir(full):
            continue
        for f in os.listdir(full):
            if f.endswith(".jsonl"):
                all_files.append(os.path.join(full, f))

    return all_files


def extract_from_sessions(session_files: list[str]) -> list[dict]:
    """Extract (tool_chain + question, answer) triples from all session files."""
    all_examples = []
    files_with_examples = 0
    tool_chain_lengths = []

    for i, fpath in enumerate(session_files):
        if i % 300 == 0 and i > 0:
            print(f"  ...{i}/{len(session_files)}, {len(all_examples)} examples so far")

        turns = parse_session_v3(fpath)
        file_examples = []

        # Accumulate tool calls across consecutive assistant turns
        accumulated_tools = []

        for j in range(len(turns) - 1):
            turn = turns[j]

            if turn["role"] == "assistant":
                # Accumulate tool calls
                accumulated_tools.extend(turn.get("tool_calls", []))

                # Check if next turn is user response
                next_turn = turns[j + 1]
                if next_turn["role"] != "user":
                    continue

                user_text = next_turn["content"].strip()
                if not is_valid_response(user_text):
                    accumulated_tools = []  # Reset on skip
                    continue

                # Extract the assistant's question/context
                assistant_text = turn["content"]
                question = extract_assistant_question(assistant_text)
                if not question or len(question) < 10:
                    accumulated_tools = []
                    continue

                # Clean the response
                answer = clean_response(user_text)
                if not answer or len(answer) < 3:
                    accumulated_tools = []
                    continue

                # Build context from accumulated tool calls
                tool_chain_str = format_tool_chain(accumulated_tools)

                # Detect project and machine from accumulated context
                context_blob = turn.get("cwd", "") + " " + question + " " + tool_chain_str
                project = detect_project(context_blob)
                machine = detect_machine(context_blob)
                if not machine:
                    machine = detect_machine(turn.get("cwd", ""))

                system_prompt = build_system_prompt(tool_chain_str, project, machine)

                # Redact secrets from question too
                question = redact_secrets(question)

                file_examples.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "category": classify_response(answer),
                    "tool_chain_len": len(accumulated_tools),
                    "has_tool_chain": len(accumulated_tools) > 0,
                    "project": project,
                    "source": os.path.basename(fpath),
                })

                tool_chain_lengths.append(len(accumulated_tools))
                accumulated_tools = []  # Reset after emitting

            else:
                # User turn that wasn't preceded by assistant, reset tools
                accumulated_tools = []

        if file_examples:
            all_examples.extend(file_examples)
            files_with_examples += 1

    return all_examples, tool_chain_lengths


def load_v2_enhanced() -> list[dict]:
    """Load existing v2 enhanced data to merge in."""
    v2_examples = []
    v2_path = os.path.join(V2_ENHANCED_DIR, "train.jsonl")
    if not os.path.exists(v2_path):
        return v2_examples

    try:
        with open(v2_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d.get("messages", [])
                if len(msgs) >= 3:
                    # Keep the user and assistant content, upgrade system prompt
                    user_content = msgs[1]["content"]
                    assistant_content = msgs[2]["content"]
                    v2_examples.append({
                        "user": user_content,
                        "assistant": assistant_content,
                    })
    except (IOError, json.JSONDecodeError):
        pass

    return v2_examples


def upgrade_v2_with_synthetic_chains(v2_examples: list[dict]) -> list[dict]:
    """Add synthetic tool chain context to v2 examples based on content heuristics."""
    upgraded = []

    # Heuristics for adding tool chains based on content keywords
    chain_templates = {
        "build": "- Bash: xcodebuild build -> BUILD SUCCEEDED",
        "test": "- Bash: xcodebuild test -> TEST SUCCEEDED",
        "deploy": "- Bash: supabase functions deploy -> deployed",
        "commit": "- Bash: git add . && git commit",
        "push": "- Bash: git push origin main -> up to date",
        "fix": "- Read: error.log\n- Edit: source_file",
        "upload": "- Bash: xcrun altool --upload-app -> uploaded",
        "install": "- Bash: pip install -r requirements.txt -> installed",
        "docker": "- Bash: docker compose up -d -> started",
        "ssh": "- Bash: ssh cloud-vm 'systemctl status' -> active",
    }

    for ex in v2_examples:
        # Apply same quality filters to v2 data
        answer = ex["assistant"].strip()
        if not is_valid_response(answer):
            continue

        user_lower = ex["user"].lower()
        assistant_lower = answer.lower()
        combined = user_lower + " " + assistant_lower

        # Pick a matching chain template
        chain = ""
        for keyword, template in chain_templates.items():
            if keyword in combined:
                chain = template
                break

        # Detect project from content
        project = detect_project(combined)

        system_prompt = build_system_prompt(chain, project, "mac1")

        # Redact secrets from user content too
        user_content = redact_secrets(ex["user"])

        upgraded.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "category": classify_response(answer),
            "tool_chain_len": 1 if chain else 0,
            "has_tool_chain": bool(chain),
            "project": project,
            "source": "v2_enhanced",
        })

    return upgraded


def dedup_examples(examples: list[dict]) -> list[dict]:
    """Deduplicate by (question, answer) pair."""
    seen = set()
    unique = []
    for ex in examples:
        msgs = ex["messages"]
        key = (msgs[1]["content"].lower()[:200], msgs[2]["content"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def main():
    print("=" * 60)
    print("KARL V3: Tool-Chain-Augmented Cognitive Twin Dataset")
    print("=" * 60)

    # 1. Discover session files
    session_files = discover_session_files()
    print(f"\nFound {len(session_files)} session files")

    # 2. Extract from sessions
    print("\nExtracting from session files...")
    session_examples, tc_lengths = extract_from_sessions(session_files)
    print(f"  Raw session examples: {len(session_examples)}")

    # 3. Generate synthetic pairs with tool chains
    print("\nGenerating synthetic pairs...")
    synthetic_raw = generate_synthetic_pairs()
    synthetic_examples = []
    for sp in synthetic_raw:
        system_prompt = build_system_prompt(
            sp["tool_chain"], sp["project"], sp["machine"]
        )
        synthetic_examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sp["question"]},
                {"role": "assistant", "content": sp["answer"]},
            ],
            "category": sp["category"],
            "tool_chain_len": sp["tool_chain"].count("\n") + 1 if sp["tool_chain"] else 0,
            "has_tool_chain": bool(sp["tool_chain"]),
            "project": sp["project"],
            "source": "synthetic_v3",
        })
    print(f"  Synthetic pairs: {len(synthetic_examples)}")

    # 4. Load and upgrade v2 enhanced data
    print("\nUpgrading v2 enhanced data with tool chains...")
    v2_raw = load_v2_enhanced()
    v2_upgraded = upgrade_v2_with_synthetic_chains(v2_raw)
    print(f"  V2 upgraded pairs: {len(v2_upgraded)}")

    # 5. Merge all sources
    all_examples = session_examples + synthetic_examples + v2_upgraded
    print(f"\nTotal before dedup: {len(all_examples)}")

    # 6. Deduplicate
    all_examples = dedup_examples(all_examples)
    print(f"Total after dedup: {len(all_examples)}")

    # 7. Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.9)
    train = all_examples[:split_idx]
    valid = all_examples[split_idx:]

    # 8. Write output
    def write_jsonl(examples: list[dict], path: str):
        with open(path, "w") as f:
            for ex in examples:
                # Write only the messages (strip metadata for training)
                record = {"messages": ex["messages"]}
                f.write(json.dumps(record) + "\n")

    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    valid_path = os.path.join(OUTPUT_DIR, "valid.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test.jsonl")

    write_jsonl(train, train_path)
    write_jsonl(valid, valid_path)
    write_jsonl(valid, test_path)  # test = copy of valid

    # 9. Stats
    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train)} examples ({os.path.getsize(train_path)/1024:.1f} KB)")
    print(f"  Valid: {len(valid)} examples ({os.path.getsize(valid_path)/1024:.1f} KB)")
    print(f"  Test:  {len(valid)} examples (copy of valid)")

    # Source distribution
    source_counts = Counter(ex["source"] for ex in all_examples)
    session_count = sum(1 for ex in all_examples if ex["source"] not in ("synthetic_v3", "v2_enhanced"))
    print(f"\nSource distribution:")
    print(f"  Session-extracted: {session_count}")
    print(f"  Synthetic v3: {source_counts.get('synthetic_v3', 0)}")
    print(f"  V2 upgraded: {source_counts.get('v2_enhanced', 0)}")

    # Tool chain stats
    with_chain = sum(1 for ex in all_examples if ex.get("has_tool_chain"))
    avg_chain = (
        sum(ex.get("tool_chain_len", 0) for ex in all_examples if ex.get("has_tool_chain"))
        / max(with_chain, 1)
    )
    print(f"\nTool chain coverage:")
    print(f"  With tool chain: {with_chain}/{len(all_examples)} ({100*with_chain/max(len(all_examples),1):.1f}%)")
    print(f"  Avg chain length (when present): {avg_chain:.1f} tool calls")

    # Response length
    resp_lengths = [len(ex["messages"][2]["content"]) for ex in all_examples]
    avg_resp = sum(resp_lengths) / max(len(resp_lengths), 1)
    print(f"\nResponse length:")
    print(f"  Average: {avg_resp:.0f} chars")
    print(f"  Min: {min(resp_lengths) if resp_lengths else 0}")
    print(f"  Max: {max(resp_lengths) if resp_lengths else 0}")

    # Category distribution
    cats = Counter(ex.get("category", "unknown") for ex in all_examples)
    print(f"\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count} ({100*count/len(all_examples):.1f}%)")

    # Project distribution
    projs = Counter(ex.get("project", "") for ex in all_examples)
    print(f"\nProject distribution (top 10):")
    for proj, count in projs.most_common(10):
        label = proj if proj else "(undetected)"
        print(f"  {label}: {count}")

    # Sample outputs
    print(f"\n{'=' * 60}")
    print("SAMPLES (3 with tool chains, 2 without)")
    print("=" * 60)

    with_tc = [ex for ex in all_examples if ex.get("has_tool_chain")]
    without_tc = [ex for ex in all_examples if not ex.get("has_tool_chain")]

    for label, pool, n in [("WITH tool chain", with_tc, 3), ("WITHOUT tool chain", without_tc, 2)]:
        print(f"\n--- {label} ---")
        for ex in pool[:n]:
            msgs = ex["messages"]
            sys_lines = msgs[0]["content"].split("\n")
            # Show just the tool chain part of system prompt
            tc_part = [l for l in sys_lines if l.startswith("- ")]
            print(f"  Tools: {tc_part[:3] if tc_part else '(none)'}")
            print(f"  Q: {msgs[1]['content'][:120]}")
            print(f"  A: {msgs[2]['content'][:120]}")
            print(f"  Cat: {ex.get('category')}, Proj: {ex.get('project', '?')}")
            print()

    print(f"\nOutput files:")
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
