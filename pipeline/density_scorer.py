#!/usr/bin/env python3
"""
Cognitive Twin Corpus Density Scorer
Uses MiniMax M2.5 (via local fleet) to classify conversation turns by training density.

Density levels:
  - CORE (9-10): Reveals deep personality, values, decision-making patterns
  - ENRICHED (7-8): Shows preferences, expertise, communication style
  - ACTIVE (4-6): Useful context but not personality-defining
  - PRUNED (1-3): Boilerplate, debug output, system noise

Runs against: kimi_memory.db (39K+ messages)
Model: MiniMax M2.5 @ localhost:18080 (141 tok/s, $0)
"""

import sqlite3
import json
import time
import sys
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request

DB_PATH = os.path.expanduser("~/projects/dream-weaver-engine/memory/kimi_memory.db")
OUTPUT_DIR = os.path.expanduser("~/Desktop/cognitive-twin/output")
MINIMAX_URL = "http://localhost:18080/v1/chat/completions"
MODEL = "MiniMax-M2.5-UD-TQ1_0.gguf"

SCORING_PROMPT = """Classify this conversation turn by personality density. Think briefly, then output ONLY one JSON line.

Density scale:
- CORE (9-10): Deep values, opinions, unique decisions
- ENRICHED (7-8): Preferences, expertise, style  
- ACTIVE (4-6): Useful but generic
- PRUNED (1-3): Noise, debug, "ok", media-only

Output format (STRICTLY one JSON object, nothing else):
{"score":N,"density":"LEVEL","reason":"short"}"""


CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 8]  # seconds


def score_turn(turn_id: int, role: str, content: str, channel: str) -> dict:
    """Score a single turn via MiniMax M2.5 with retry logic."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SCORING_PROMPT},
            {"role": "user", "content": f"Channel: {channel}\nRole: {role}\nContent: {content[:4000]}"}
        ],
        "max_tokens": 500,
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            
            content_text = result["choices"][0]["message"].get("content", "")
            timings = result.get("timings", {})
            
            # Parse the JSON response
            try:
                score_data = json.loads(content_text.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                match = re.search(r'\{[^}]+\}', content_text)
                if match:
                    score_data = json.loads(match.group())
                else:
                    score_data = {"score": 0, "density": "ERROR", "reason": f"Parse failed: {content_text[:100]}"}
            
            return {
                "id": turn_id,
                "score": score_data.get("score", 0),
                "density": score_data.get("density", "UNKNOWN"),
                "reason": score_data.get("reason", ""),
                "tok_s": timings.get("predicted_per_second", 0),
                "tokens": result["usage"]["completion_tokens"],
                "ok": True
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            return {
                "id": turn_id,
                "score": 0,
                "density": "ERROR",
                "reason": str(e)[:200],
                "tok_s": 0,
                "tokens": 0,
                "ok": False
            }


def load_checkpoint() -> int:
    """Load the last processed message ID from checkpoint."""
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            return data.get("last_id", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def save_checkpoint(last_id: int, scored: int, errors: int, distribution: dict):
    """Save progress checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "last_id": last_id,
            "scored": scored,
            "errors": errors,
            "distribution": distribution,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, f)


def main():
    parser = argparse.ArgumentParser(description="Score corpus density via MiniMax M2.5")
    parser.add_argument("--batch-size", type=int, default=100, help="Turns per batch")
    parser.add_argument("--offset", type=int, default=0, help="Start from message ID offset")
    parser.add_argument("--limit", type=int, default=0, help="Max turns to process (0=all)")
    parser.add_argument("--min-length", type=int, default=50, help="Skip content shorter than this")
    parser.add_argument("--role", type=str, default="user", help="Filter by role (user/assistant/both)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel requests (match llama.cpp slots)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to kimi_memory.db")
    parser.add_argument("--output", type=str, default="", help="Custom output path")
    args = parser.parse_args()

    db_path = args.db
    db = sqlite3.connect(db_path)
    cur = db.cursor()

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_id = load_checkpoint()
        if checkpoint_id > 0:
            args.offset = max(args.offset, checkpoint_id)
            print(f"📎 Resuming from checkpoint: message ID > {checkpoint_id}")

    # Build query with parameterized values
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
    
    print(f"📊 Corpus Density Scorer")
    print(f"   Model: MiniMax M2.5 @ localhost:18080")
    print(f"   Turns to score: {total}")
    print(f"   Role filter: {args.role}")
    print(f"   Min length: {args.min_length}")
    print(f"   Parallel: {args.parallel}")
    print()

    if args.dry_run:
        print("🏃 DRY RUN — showing first 5 turns:")
        for t in turns[:5]:
            print(f"  [{t[0]}] {t[1]} ({t[3]}): {str(t[2])[:100]}...")
        print(f"\n  ... and {total - 5} more")
        db.close()
        return

    # Check MiniMax is alive
    try:
        req = urllib.request.Request("http://localhost:18080/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            health = json.loads(resp.read())
        if health.get("status") != "ok":
            print("❌ MiniMax not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"❌ MiniMax unreachable: {e}")
        sys.exit(1)

    print("🟢 MiniMax healthy — starting scoring...\n")

    # Output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"density_scores_{timestamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scored = 0
    errors = 0
    distribution = {"CORE": 0, "ENRICHED": 0, "ACTIVE": 0, "PRUNED": 0, "ERROR": 0}
    start_time = time.time()

    with open(out_path, "w") as f, ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for batch_start in range(0, total, args.batch_size):
            batch = turns[batch_start:batch_start + args.batch_size]
            
            futures = []
            for turn_id, role, content, channel in batch:
                futures.append(pool.submit(score_turn, turn_id, role, content, channel))
            
            for future in futures:
                result = future.result()
                f.write(json.dumps(result) + "\n")
                f.flush()
                
                if result["ok"]:
                    scored += 1
                    density = result.get("density", "ERROR")
                    if density in distribution:
                        distribution[density] += 1
                    else:
                        distribution["ERROR"] += 1
                else:
                    errors += 1
                    distribution["ERROR"] += 1

            elapsed = time.time() - start_time
            rate = scored / elapsed if elapsed > 0 else 0
            pct = (batch_start + len(batch)) / total * 100

            # Save checkpoint every batch
            last_id = batch[-1][0] if batch else 0
            save_checkpoint(last_id, scored, errors, distribution)

            print(f"\r  [{pct:5.1f}%] Scored: {scored} | Errors: {errors} | "
                  f"Rate: {rate:.1f} turns/s | "
                  f"Core:{distribution['CORE']} Enriched:{distribution['ENRICHED']} "
                  f"Active:{distribution['ACTIVE']} Pruned:{distribution['PRUNED']}", 
                  end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n\n✅ Done in {elapsed:.0f}s")
    print(f"   Output: {out_path}")
    print(f"   Scored: {scored} | Errors: {errors}")
    print(f"   Distribution:")
    for k, v in sorted(distribution.items()):
        pct = v / max(scored, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"     {k:10s}: {v:5d} ({pct:5.1f}%) {bar}")

    db.close()


if __name__ == "__main__":
    main()
