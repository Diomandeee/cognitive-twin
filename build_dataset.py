#!/usr/bin/env python3
"""Build corrected v7 dataset with NO char cap, stream-of-consciousness messages,
and real trajectory tags from Supabase."""

import json, re, os, random

random.seed(42)

OUTPUT_DIR = os.path.expanduser("~/projects/karl/autocontinue-v7-corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load base v3 data (has tool chain context)
with open(os.path.expanduser("~/projects/karl/autocontinue-v3/train.jsonl")) as f:
    base_train = [json.loads(l) for l in f]
with open(os.path.expanduser("~/projects/karl/autocontinue-v3/valid.jsonl")) as f:
    base_valid = [json.loads(l) for l in f]

print(f"Base v3 train: {len(base_train)}")
print(f"Base v3 valid: {len(base_valid)}")

# 2. Remove the 300 char cap — check if responses were truncated
# The base data already has max 300 chars per response. We can't recover truncated text.
# But we CAN add the stream-of-consciousness messages which are uncapped.

# 3. Load stream-of-consciousness messages from Supabase
with open(os.path.expanduser("~/projects/karl/stream_of_consciousness_combined.json")) as f:
    stream_raw = json.load(f)

# Filter to genuine Mohamed voice (conversational score >= 2, < 3000 chars)
stream = [m for m in stream_raw
          if m["length"] < 3000
          and m.get("conversational_score", m.get("conv", 0)) >= 2]

print(f"Stream-of-consciousness messages: {len(stream)}")

# 4. Convert stream messages to training format
# These are USER messages (Mohamed speaking TO the AI)
# We use them as the RESPONSE (assistant turn) to teach the model Mohamed's extended voice
# The "question" is a generic prompt that would elicit this kind of response

question_templates = [
    "What are your thoughts on how we should approach this?",
    "What's your plan here?",
    "How do you want to handle this?",
    "What should we do next?",
    "Tell me what you're thinking.",
    "How should we proceed?",
    "What's the direction?",
    "Walk me through your thinking.",
    "What do you want to do?",
    "What's the move?",
]

def compute_trajectory_tag(text):
    """Compute trajectory conditioning tag from text characteristics."""
    length = len(text)
    lower = text.lower()

    is_correction = any(w in lower for w in ["no", "don't", "wrong", "stop", "fix", "didn't", "that's not"])
    is_affirmation = any(w in lower for w in ["yes", "yeah", "continue", "go", "do it", "ship"])
    is_exploring = any(w in lower for w in ["what if", "imagine", "perhaps", "considering", "i think", "the idea"])

    if length < 50:
        commitment, uncertainty = "high", "low"
    elif length < 200:
        commitment, uncertainty = "moderate", "moderate"
    elif length < 500:
        commitment, uncertainty = "moderate", "high"
    else:
        commitment, uncertainty = "evolving", "high"

    if is_correction:
        pressure, phase = "negative", "corrective"
    elif is_exploring:
        pressure, phase = "exploratory", "ideating"
    elif is_affirmation:
        pressure, phase = "positive", "directive"
    else:
        pressure, phase = "neutral", "operational"

    return f"[trajectory: commitment={commitment}, uncertainty={uncertainty}, pressure={pressure}, phase={phase}]"

stream_examples = []
for m in stream:
    text = m["text"]
    tag = compute_trajectory_tag(text)

    # Use Supabase trajectory data if available
    depth = m.get("depth")
    complexity = m.get("complexity")
    if depth and depth > 0:
        tag += f" [depth={depth}, complexity={complexity}]"

    system = f"You are Mohamed. Direct, casual, action-oriented. Stream of thought when exploring ideas. {tag}"
    question = random.choice(question_templates)

    example = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": text}
        ]
    }
    stream_examples.append(example)

print(f"Stream examples created: {len(stream_examples)}")

# 5. Add trajectory tags to ALL base examples (they were missing)
for ex in base_train + base_valid:
    resp = ex["messages"][2]["content"]
    tag = compute_trajectory_tag(resp)
    # Replace system prompt with tagged version
    ex["messages"][0]["content"] = f"You are Mohamed. Direct, casual, action-oriented. {tag}"

# 6. Upweight long-form: duplicate based on length
upweighted = []
for ex in stream_examples:
    resp_len = len(ex["messages"][2]["content"])
    if resp_len > 1000:
        dupes = 5
    elif resp_len > 500:
        dupes = 3
    else:
        dupes = 2
    for _ in range(dupes):
        upweighted.append(ex)

print(f"Stream examples after upweighting: {len(upweighted)}")

# 7. Upweight corrections in base data
correction_dupes = []
for ex in base_train:
    resp = ex["messages"][2]["content"].lower()
    if any(w in resp for w in ["no", "don't", "wrong", "stop", "fix", "didn't"]):
        correction_dupes.append(ex)

print(f"Correction dupes added: {len(correction_dupes)}")

# 8. Combine all
all_train = base_train + upweighted + correction_dupes
random.shuffle(all_train)

# Remove any AI-contaminated examples
ai_phrases = ["great job", "hey there", "hey!", "let me help", "let's break down",
              "here's what", "i'd recommend", "happy to help", "let's tackle", "good question"]
all_train = [ex for ex in all_train if not any(p in ex["messages"][2]["content"].lower() for p in ai_phrases)]

# 9. Split: keep separate valid set
all_valid = base_valid + stream_examples[:10]  # Add some stream to valid too
random.shuffle(all_valid)

# 10. Stats
lengths = [len(ex["messages"][2]["content"]) for ex in all_train]
print()
print("=" * 60)
print("V7 CORRECTED DATASET")
print("=" * 60)
print(f"Train: {len(all_train)}")
print(f"Valid: {len(all_valid)}")
print(f"Avg response: {sum(lengths)/len(lengths):.0f} chars")
print(f"Max response: {max(lengths)} chars")
print(f"<50 chars: {sum(1 for l in lengths if l < 50)} ({sum(1 for l in lengths if l < 50)/len(lengths)*100:.1f}%)")
print(f"50-200: {sum(1 for l in lengths if 50 <= l < 200)} ({sum(1 for l in lengths if 50 <= l < 200)/len(lengths)*100:.1f}%)")
print(f"200-500: {sum(1 for l in lengths if 200 <= l < 500)} ({sum(1 for l in lengths if 200 <= l < 500)/len(lengths)*100:.1f}%)")
print(f">500: {sum(1 for l in lengths if l >= 500)} ({sum(1 for l in lengths if l >= 500)/len(lengths)*100:.1f}%)")
print(f">1000: {sum(1 for l in lengths if l >= 1000)} ({sum(1 for l in lengths if l >= 1000)/len(lengths)*100:.1f}%)")

# Trajectory tag distribution
tags = {}
for ex in all_train:
    s = ex["messages"][0]["content"]
    if "corrective" in s: tags["corrective"] = tags.get("corrective", 0) + 1
    elif "directive" in s: tags["directive"] = tags.get("directive", 0) + 1
    elif "ideating" in s: tags["ideating"] = tags.get("ideating", 0) + 1
    elif "operational" in s: tags["operational"] = tags.get("operational", 0) + 1
    else: tags["other"] = tags.get("other", 0) + 1
print()
print("TRAJECTORY TAGS:")
for k, v in sorted(tags.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/len(all_train)*100:.1f}%)")

# Sample a stream-of-consciousness example
print()
print("SAMPLE LONG-FORM:")
long_samples = [ex for ex in all_train if len(ex["messages"][2]["content"]) > 800]
if long_samples:
    s = random.choice(long_samples)
    print(f"  System: {s['messages'][0]['content'][:100]}")
    print(f"  User: {s['messages'][1]['content']}")
    print(f"  Mohamed ({len(s['messages'][2]['content'])} chars): {s['messages'][2]['content'][:300]}...")

# 11. Write
with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w") as f:
    for ex in all_train:
        f.write(json.dumps(ex) + "\n")
with open(os.path.join(OUTPUT_DIR, "valid.jsonl"), "w") as f:
    for ex in all_valid:
        f.write(json.dumps(ex) + "\n")
with open(os.path.join(OUTPUT_DIR, "test.jsonl"), "w") as f:
    for ex in all_valid:
        f.write(json.dumps(ex) + "\n")

print(f"\nWritten to {OUTPUT_DIR}")
