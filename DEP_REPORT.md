# DEP Report — Cognitive Twin Pipeline + MiniMax Fleet Integration
**Date:** 2026-02-16
**Auditor:** Claw 🦞
**Scope:** `~/Desktop/cognitive-twin/pipeline/` + `~/.clawdbot/minimax-fleet/`

---

## 1. Structure (Score: 7/10)

### ✅ Strengths
- Clean separation: `pipeline/` for scripts, `output/` for results
- Registry file (`minimax-fleet/registry.json`) tracks instance metadata
- Single-file scorer is appropriately simple for the task

### ⚠️ Issues
- **No `__init__.py` or module structure** — fine for now but limits importability
- **No `requirements.txt`** — script uses only stdlib (good) but should document that
- **No README.md** in `cognitive-twin/` — new contributor can't onboard
- **Output files not gitignored** — JSONL scoring data could be large

### Recommendations
- [ ] Add `README.md` with pipeline overview, usage, and architecture
- [ ] Add `.gitignore` for `output/*.jsonl` (keep structure, ignore data)
- [ ] Add `CLAUDE.md` for sub-agent context

**Structure Score: 7/10**

---

## 2. Compilation / Runtime (Score: 8/10)

### ✅ Strengths
- Pure Python 3, stdlib only (no dependencies to break)
- `urllib.request` instead of `requests` — zero install required
- Parallel execution via `ThreadPoolExecutor` — matches llama.cpp's 4 slots
- Health check before starting — fails fast if MiniMax is down

### ⚠️ Issues
- **SQL injection risk** — `args.role` is interpolated directly into SQL query (`f"role = '{args.role}'"`)
- **No retry logic** — network hiccups or slot contention cause permanent failures
- **No checkpoint/resume** — if the 3.5hr run crashes at 80%, must restart from 0
- **Hardcoded DB path** — breaks if kimi_memory.db moves

### Recommendations
- [ ] **CRITICAL: Parameterize SQL** — use `?` placeholders, not f-strings
- [ ] Add exponential backoff retry (3 attempts per turn)
- [ ] Add checkpoint file — write last processed ID, support `--resume`
- [ ] Make DB path configurable via `--db` argument
- [ ] Add `--output` flag to specify output path

**Compilation Score: 8/10**

---

## 3. Integration (Score: 6/10)

### ✅ Strengths
- Clawdbot gateway properly configured with `minimax-fleet` provider
- `models.mode: "merge"` preserves all existing providers
- Alias `minimax` registered — accessible via `/model minimax`
- SSH tunnel verified and health-checked

### ⚠️ Issues
- **Tunnel is ephemeral** — dies on Mac sleep, SSH disconnect, or network change
- **No auto-reconnect** — if tunnel drops mid-scoring, the run fails silently
- **No monitoring** — nobody alerts when the Vast.ai instance goes down
- **No auto-shutdown** — instance burns $0.77/hr even when idle
- **Clawdbot end-to-end not verified** — only direct API tested, never through gateway

### Recommendations
- [ ] **Create tunnel keepalive script** with autossh or a cron watchdog
- [ ] **Add Vast.ai balance monitor** to heartbeat checks
- [ ] **Add auto-shutdown script** — stop instance after N hours idle
- [ ] **Test `/model minimax` in a live Discord session** — verify full round-trip
- [ ] Add tunnel status to `memory/agent-capacity.json`

**Integration Score: 6/10**

---

## 4. Content / Quality (Score: 7/10)

### ✅ Strengths
- Scoring prompt is well-designed — clear taxonomy, structured output
- JSON output format enables downstream pipeline consumption
- 100% parse success on test batch after prompt tuning
- Distribution looks realistic (50% ACTIVE, 20% ENRICHED, etc.)

### ⚠️ Issues
- **No ground truth validation** — are the scores actually correct?
- **No inter-rater reliability** — should score a subset with Claude and compare
- **Reasoning overhead** — model burns ~70% of tokens on thinking for a simple classification
- **No score calibration** — what makes a 7 vs 8? No reference examples
- **Content truncation at 2000 chars** — long technical messages lose context

### Recommendations
- [ ] **Score 50 turns with Claude Sonnet** → compare against MiniMax scores → measure agreement
- [ ] Add few-shot examples to the prompt (1 per density level)
- [ ] Increase content window to 4000 chars for long-form messages
- [ ] Log reasoning_content for audit trail (optional flag)
- [ ] Create `calibration_set.json` with human-verified reference scores

**Content Score: 7/10**

---

## 5. User Journey (Score: 5/10)

### ✅ Strengths
- CLI interface with clear flags (`--limit`, `--parallel`, `--dry-run`)
- Progress bar with real-time stats during execution
- Final summary with distribution chart

### ⚠️ Issues
- **No way to monitor a running job** besides `tail -f` the log
- **No progress webhook** — long runs should ping Discord
- **No results viewer** — scoring output is raw JSONL, no summary tool
- **No pipeline orchestration** — density scoring is step 1, but steps 2-4 (WORMS, SFT export, training) don't exist yet
- **No dashboard** — should post results to #ct-corpus when done

### Recommendations
- [ ] Add `--notify` flag that posts completion to #ct-corpus
- [ ] Create `analyze_scores.py` — reads JSONL, generates distribution report
- [ ] Add `watch_run.sh` script for monitoring
- [ ] Plan next pipeline stages: WORMS augmentation → SFT export → training
- [ ] Post live progress to #ct-corpus thread every 500 turns

**User Journey Score: 5/10**

---

## 6. Deployment / Operations (Score: 5/10)

### ✅ Strengths
- Vast.ai instance details documented in `registry.json`
- Pipeline runs as a simple background process
- Cost model is clear ($0.77/hr, ~$2.67 for full user corpus)

### ⚠️ Issues
- **No launchd/systemd service** for the SSH tunnel
- **No cost tracking** — balance check is manual
- **No data backup** — scoring output lives only on local disk
- **No CI/CD** — no automated pipeline trigger
- **Instance lifecycle manual** — start/stop via Vast.ai web UI

### Recommendations
- [ ] Create `com.minimax-fleet.tunnel.plist` for persistent tunnel
- [ ] Add `vast_balance_check.sh` to heartbeat
- [ ] Commit output summaries (not full JSONL) to git
- [ ] Create `vast_ctl.sh` — start/stop/status wrapper for the instance
- [ ] Add fleet health to HEARTBEAT.md checks

**Deployment Score: 5/10**

---

## Overall DEP Score: 6.3/10

| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| Structure | 7 | 1.0 | 7.0 |
| Compilation | 8 | 1.5 | 12.0 |
| Integration | 6 | 1.5 | 9.0 |
| Content | 7 | 1.0 | 7.0 |
| User Journey | 5 | 1.0 | 5.0 |
| Deployment | 5 | 1.0 | 5.0 |
| **Total** | | **7.0** | **45.0 / 70 = 64.3%** |

---

## Priority Actions (Ranked)

### 🔴 Critical (do now)
1. **Fix SQL injection** in density_scorer.py — parameterize queries
2. **Add checkpoint/resume** — can't afford losing a 3.5hr run
3. **Create tunnel keepalive** — tunnel death = wasted compute

### 🟡 Important (do this week)
4. **Validate against Claude** — 50-turn calibration set
5. **Add retry logic** — 3 attempts with exponential backoff
6. **Post results to #ct-corpus** when scoring completes
7. **Add auto-shutdown** for Vast.ai instance

### 🟢 Nice to have
8. README.md + CLAUDE.md
9. Results analyzer script
10. Few-shot examples in prompt
11. Fleet health in heartbeat

---

## Architecture Notes

```
┌─────────────┐     SSH Tunnel      ┌──────────────────┐
│   Mac1 Air  │◄───────────────────►│  Vast.ai GPU     │
│  Clawdbot   │   localhost:18080   │  RTX PRO 6000    │
│  Pipeline   │                     │  MiniMax M2.5    │
└──────┬──────┘                     │  141 tok/s       │
       │                            │  $0.77/hr        │
       ▼                            └──────────────────┘
┌──────────────┐
│ kimi_memory  │
│   39K msgs   │──► density_scorer.py ──► scores.jsonl
│   (SQLite)   │         │
└──────────────┘         ▼
                   ┌──────────────┐
                   │  Next Steps: │
                   │  WORMS aug   │
                   │  SFT export  │
                   │  LoRA train  │
                   └──────────────┘
```
