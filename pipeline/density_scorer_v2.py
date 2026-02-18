#!/usr/bin/env python3
"""
Cognitive Twin Corpus Density Scorer V2
Batched scoring — sends multiple turns per request to maximize context utilization.

V2 improvements over V1:
- Batches 10 turns per request (10x fewer API calls)
- Uses ~10K of 32K context per request (vs 1.6K in V1)
- Model sees conversation context, not isolated messages
- Amortizes reasoning overhead across batch
- Retry + checkpoint + parameterized SQL (from V1 fixes)
"""

import sqlite3
import json
import time
import sys
import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request

DB_PATH = os.path.expanduser("~/projects/dream-weaver-engine/memory/kimi_memory.db")
OUTPUT_DIR = os.path.expanduser("~/Desktop/cognitive-twin/output")
MINIMAX_URL = "http://localhost:18080/v1/chat/completions"
MODEL = "MiniMax-M2.5-UD-TQ1_0.gguf"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint_v2.json")
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 8]

BATCH_SCORING_PROMPT = """You are a training data classifier. Score each conversation turn by "personality density" — how much it reveals about the human's unique personality, values, expertise, or decision-making style.

Density scale:
- CORE (9-10): Deep values, strong opinions, unique decision patterns
- ENRICHED (7-8): Preferences, expertise signals, communication style
- ACTIVE (4-6): Useful context but generic
- PRUNED (1-3): Noise, boilerplate, "ok", media-only, system output

You will receive multiple turns. Score EACH one. Output ONLY a JSON array, one object per turn:
[{"id":N,"score":N,"density":"LEVEL","reason":"brief"},...]

Be concise in reasoning. Output valid JSON array only."""


def score_batch(turns: list[tuple]) -> list[dict]:
    """Score a batch of turns in a single request."""
    # Format turns for the prompt
    turn_texts = []
    for turn_id, role, content, channel in turns:
        truncated = content[:3000]  # Allow ~750 tokens per turn, 10 turns = ~7500 tokens
        turn_texts.append(f"[ID:{turn_id}] Role:{role} Channel:{channel}\n{truncated}")
    
    batch_content = "\n---\n".join(turn_texts)
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": BATCH_SCORING_PROMPT},
            {"role": "user", "content": f"Score these {len(turns)} turns:\n\n{batch_content}"}
        ],
        "max_tokens": 2000,  # ~200 tokens per turn in the response
        "temperature": 0.1
    }
    
    data = json.dumps(payload).encode()
    
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                MINIMAX_URL,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            
            content_text = result["choices"][0]["message"].get("content", "")
            timings = result.get("timings", {})
            tok_s = timings.get("predicted_per_second", 0)
            
            # Parse the JSON array response
            try:
                scores = json.loads(content_text.strip())
                if not isinstance(scores, list):
                    scores = [scores]
            except json.JSONDecodeError:
                # Try to find JSON array in the response
                match = re.search(r'\[.*\]', content_text, re.DOTALL)
                if match:
                    try:
                        scores = json.loads(match.group())
                    except json.JSONDecodeError:
                        scores = []
                else:
                    # Try individual objects
                    scores = []
                    for m in re.finditer(r'\{[^{}]+\}', content_text):
                        try:
                            scores.append(json.loads(m.group()))
                        except json.JSONDecodeError:
                            pass
            
            # Map scores back to turn IDs
            results = []
            turn_ids = {t[0] for t in turns}
            scored_ids = set()
            
            for s in scores:
                sid = s.get("id")
                if sid in turn_ids:
                    scored_ids.add(sid)
                    results.append({
                        "id": sid,
                        "score": s.get("score", 0),
                        "density": s.get("density", "UNKNOWN"),
                        "reason": s.get("reason", ""),
                        "tok_s": tok_s,
                        "tokens": result["usage"]["completion_tokens"],
                        "ok": True
                    })
            
            # Fill in any missing turns
            for turn_id, role, content, channel in turns:
                if turn_id not in scored_ids:
                    results.append({
                        "id": turn_id,
                        "score": 0,
                        "density": "ERROR",
                        "reason": f"Not in batch response ({len(scores)} scores returned for {len(turns)} turns)",
                        "tok_s": tok_s,
                        "tokens": 0,
                        "ok": False
                    })
            
            return results
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            # All retries failed — return errors for all turns
            return [{
                "id": t[0],
                "score": 0,
                "density": "ERROR",
                "reason": str(e)[:200],
                "tok_s": 0,
                "tokens": 0,
                "ok": False
            } for t in turns]


def load_checkpoint() -> int:
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            return data.get("last_id", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def save_checkpoint(last_id: int, scored: int, errors: int, distribution: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "last_id": last_id,
            "scored": scored,
            "errors": errors,
            "distribution": distribution,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, f)


def main():
    parser = argparse.ArgumentParser(description="Score corpus density via MiniMax M2.5 (batched V2)")
    parser.add_argument("--batch-size", type=int, default=10, help="Turns per API request")
    parser.add_argument("--offset", type=int, default=0, help="Start from message ID offset")
    parser.add_argument("--limit", type=int, default=0, help="Max turns to process (0=all)")
    parser.add_argument("--min-length", type=int, default=50, help="Skip content shorter than this")
    parser.add_argument("--role", type=str, default="user", help="Filter by role (user/assistant/both)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to kimi_memory.db")
    parser.add_argument("--parallel", type=int, default=2, help="Parallel batch requests")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    cur = db.cursor()

    if args.resume:
        checkpoint_id = load_checkpoint()
        if checkpoint_id > 0:
            args.offset = max(args.offset, checkpoint_id)
            print(f"📎 Resuming from checkpoint: message ID > {checkpoint_id}")

    where_clauses = ["length(content) >= ?"]
    params = [args.min_length]
    if args.role != "both":
        where_clauses.append("role = ?")
        params.append(args.role)
    if args.offset > 0:
        where_clauses.append("id > ?")
        params.append(args.offset)

    where = " AND ".join(where_clauses)
    limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""
    query = f"SELECT id, role, content, channel FROM messages WHERE {where} ORDER BY id {limit_clause}"
    turns = cur.execute(query, params).fetchall()
    total = len(turns)

    api_calls = (total + args.batch_size - 1) // args.batch_size

    print(f"📊 Corpus Density Scorer V2 (Batched)")
    print(f"   Model: MiniMax M2.5 @ localhost:18080")
    print(f"   Turns to score: {total}")
    print(f"   Batch size: {args.batch_size} turns/request")
    print(f"   API calls needed: {api_calls}")
    print(f"   Parallel: {args.parallel}")
    print(f"   Estimated speedup: ~{args.batch_size}x vs V1")
    print()

    if args.dry_run:
        print("🏃 DRY RUN — first batch would be:")
        for t in turns[:args.batch_size]:
            print(f"  [{t[0]}] {t[1]} ({t[3]}): {str(t[2])[:80]}...")
        db.close()
        return

    # Health check
    try:
        req = urllib.request.Request("http://localhost:18080/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            health = json.loads(resp.read())
        if health.get("status") != "ok":
            print("❌ MiniMax not healthy"); sys.exit(1)
    except Exception as e:
        print(f"❌ MiniMax unreachable: {e}"); sys.exit(1)

    print("🟢 MiniMax healthy — starting batched scoring...\n")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"density_v2_{timestamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scored = 0
    errors = 0
    distribution = {"CORE": 0, "ENRICHED": 0, "ACTIVE": 0, "PRUNED": 0, "ERROR": 0}
    start_time = time.time()

    # Create batches
    batches = []
    for i in range(0, total, args.batch_size):
        batches.append(turns[i:i + args.batch_size])

    with open(out_path, "w") as f, ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for chunk_start in range(0, len(batches), args.parallel):
            chunk = batches[chunk_start:chunk_start + args.parallel]
            futures = [pool.submit(score_batch, batch) for batch in chunk]

            for future in futures:
                results = future.result()
                for result in results:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    if result["ok"]:
                        scored += 1
                        d = result.get("density", "ERROR")
                        distribution[d] = distribution.get(d, 0) + 1
                    else:
                        errors += 1
                        distribution["ERROR"] += 1

            # Checkpoint
            last_batch = chunk[-1] if chunk else []
            last_id = last_batch[-1][0] if last_batch else 0
            save_checkpoint(last_id, scored, errors, distribution)

            elapsed = time.time() - start_time
            rate = scored / elapsed if elapsed > 0 else 0
            done = chunk_start + len(chunk)
            pct = done / len(batches) * 100

            print(f"\r  [{pct:5.1f}%] Batches: {done}/{len(batches)} | Scored: {scored} | Errors: {errors} | "
                  f"Rate: {rate:.1f} t/s | "
                  f"C:{distribution['CORE']} E:{distribution['ENRICHED']} "
                  f"A:{distribution['ACTIVE']} P:{distribution['PRUNED']}",
                  end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n\n✅ Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"   Output: {out_path}")
    print(f"   Scored: {scored} | Errors: {errors} ({errors/max(scored+errors,1)*100:.1f}%)")
    print(f"   Rate: {scored/elapsed:.1f} turns/sec (V1 was 0.6/sec)")
    print(f"   Distribution:")
    for k, v in sorted(distribution.items()):
        pct = v / max(scored, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"     {k:10s}: {v:5d} ({pct:5.1f}%) {bar}")

    db.close()


if __name__ == "__main__":
    main()
