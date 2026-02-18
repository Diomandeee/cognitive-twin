# RAG++ & Graph Kernel — Detailed Improvement Analysis

*Generated: 2026-02-16 21:20 EST*

---

## Current State of Play

### Two Parallel Tracks Running

| Track | Target | Status | ETA |
|-------|--------|--------|-----|
| **Track 1: MiniMax Density Scoring** | kimi_memory.db (9.1K user turns) | 🟢 62% complete, 0 errors | ~1.5h |
| **Track 2: RAG++ & Graph Kernel** | Search quality, entity enrichment | 🔴 Audited, issues documented | Needs work |

---

## TRACK 1: MiniMax Density Scoring — Status

**Running process:** PID 56430, `density_scorer.py --role user --min-length 50 --parallel 4`

**Progress at 62% (5,700/9,155 turns):**

| Tier | Count | % | Description |
|------|-------|---|-------------|
| CORE | 146 | 2.6% | Deep values, unique decisions, strong preferences |
| ENRICHED | 559 | 9.8% | Substantive context, project decisions |
| ACTIVE | 2,983 | 52.3% | Normal productive conversation |
| PRUNED | 1,317 | 23.1% | Low-value, boilerplate, noise |
| ERROR (parse) | 695 | 12.2% | MiniMax output parsing failures |

**Projected final (at current ratios):**
- CORE: ~235 turns
- ENRICHED: ~900 turns  
- **High-value (CORE+ENRICHED): ~1,135 turns** for v9 expansion

**Output:** `~/Desktop/cognitive-twin/output/density_scores_20260216_183513.jsonl` (5,792 lines, 1.2MB)

---

## TRACK 2: RAG++ Deep Dive

### Architecture Overview

```
RAG++ (Python, FastAPI, :8000)
├── Ingestion Layer
│   ├── clawdbot_bridge.py     → Clawdbot sessions → Supabase
│   ├── claude_code_bridge.py  → Claude Code sessions → Supabase
│   ├── embedder.py            → Gemini embedding-001 (768-dim)
│   └── prompt_bridge.py       → Orbit prompt tracking
├── Retrieval Layer
│   ├── query.py               → MemoryRetriever (Supabase pgvector)
│   ├── dual_plane.py          → Raw turns + Semantic artifacts
│   ├── quality.py             → QualityReranker (stall/exec scoring)
│   ├── intent.py              → QueryIntentAnalyzer
│   └── provenance.py          → Slice-conditioned retrieval
├── Generation Layer
│   ├── enhanced.py            → EnhancedGenerationEngine
│   ├── context.py             → Context assembly
│   └── synthesis.py           → Response synthesis
├── ML Layer
│   ├── cognitivetwin_v2.py    → 270M param model (MPS)
│   ├── inference/             → Inference server
│   └── training/              → Continuous learning pipeline
└── Service Layer (API Routes)
    ├── context.py             → /api/context/search (general search)
    ├── rag.py                 → /api/rag/search, /enhanced, /global, /slice
    ├── graph_enrichment.py    → /api/rag/enrich (GK bridge)
    ├── topology.py            → /api/topology/* (3D visualization)
    └── training.py            → /api/training/*
```

### Data Stats (Supabase `memory_turns`)

| Metric | Value |
|--------|-------|
| Total turns | **245,499** |
| With embeddings | **234,731** (95.6%) |
| With salience scores | **245,499** (100%) |
| Embedding model | Gemini embedding-001 (768-dim) |
| Search method | pgvector cosine similarity via `search_memory` RPC |

**Lifecycle Distribution:**

| Tier | Count | % |
|------|-------|---|
| CORE | 1,474 | 0.6% |
| ENRICHED | 24,049 | 9.8% |
| ACTIVE | 106,408 | 43.4% |
| LOW | 20,788 | 8.5% |
| PRUNED | 15,038 | 6.1% |
| (unclassified) | 77,742 | 31.7% |

**Role split:** 62,602 user / 178,201 assistant / ~5K other

### Issues Found (Stress Test Results)

#### Issue 1: Search Returns Irrelevant Old AI Outputs
**Severity: HIGH**

Query: "What is Mo passionate about?"
- Result 1: [0.670] "I have a passion for public speaking..." ← OLD ChatGPT generic output
- Result 2: [0.663] "I am an avid learner..." ← More generic AI text
- Result 3: [0.660] "I am a passionate traveler..." ← Completely wrong

**Root cause:** No lifecycle filtering in default search. PRUNED turns (15K) and old generic ChatGPT outputs rank by pure cosine similarity without quality weighting. The `min_salience` parameter exists but defaults to 0.0.

**Fix:** Set `min_salience` default to 0.3 in `search_context()` route AND exclude `lifecycle_status = 'pruned'` by default.

#### Issue 2: Duplicate Results
**Severity: MEDIUM**

Query: "Why did we choose Supabase?"
- Returns the exact same assistant message 3 times (all [0.707])

**Root cause:** The RAG Bridge is creating duplicate entries during sync. The `search_memory` RPC doesn't deduplicate by content hash.

**Fix:** Add `content_hash` column to `memory_turns`, deduplicate during ingestion, and add DISTINCT filtering in search RPC.

#### Issue 3: Enhanced Search Endpoint Broken
**Severity: MEDIUM**

`GET /api/rag/search/enhanced?q=test` → 422 "Field required"

**Root cause:** Endpoint expects `query` parameter, not `q`. Inconsistent with `/api/context/search` which uses `q`.

**Fix:** Normalize all search endpoints to accept both `q` and `query`.

#### Issue 4: Style Signature Never Computed
**Severity: HIGH**

`GET /api/rag/signature` → `{"signature": null, "confidence": 0.0, "update_count": 0}`

**Root cause:** The `twin/extract-style` endpoint was never called with enough data. The continuous learning pipeline is OFF.

**Fix:** Run style extraction on CORE+ENRICHED turns, enable continuous learning.

#### Issue 5: No Recency Bias in Default Search
**Severity: MEDIUM**

Old 2023 ChatGPT outputs rank higher than 2026 conversations because embeddings don't account for temporal relevance.

**Root cause:** The `recency_boost` parameter exists in enhanced search (default 0.15) but the basic `/api/context/search` doesn't use it.

**Fix:** Add recency scoring to context search, or route all searches through enhanced search.

### Concrete Improvement Plan

**Phase 1 — Quick Wins (can do now, code changes only):**

1. **Default min_salience to 0.3** in `/api/context/search`
   - File: `rag_plusplus/service/routes/context.py` line ~28
   - Change: `min_salience: float = Query(0.0, ...)` → `Query(0.3, ...)`

2. **Exclude PRUNED from default search**
   - File: `rag_plusplus/retrieval/query.py` in `_vector_search()`
   - Add: `lifecycle_status != 'pruned'` filter to Supabase query

3. **Fix enhanced search parameter**
   - File: `rag_plusplus/service/routes/rag.py` line 926
   - Add alias: accept both `q` and `query`

4. **Add deduplication to search results**
   - File: `rag_plusplus/retrieval/query.py` in `search()`
   - Post-process: dedupe by content_hash before returning

**Phase 2 — Salience-Weighted Search (medium effort):**

5. **Composite scoring** in search: `final_score = 0.7 * similarity + 0.2 * salience + 0.1 * recency`
   - File: `rag_plusplus/retrieval/query.py`
   - Or update `search_memory` RPC in Supabase to include salience weighting

6. **Run style extraction** on top 1,474 CORE turns
   - Endpoint: `POST /api/rag/twin/extract-style`
   - Will compute Mo's linguistic fingerprint

7. **Enable continuous learning pipeline**
   - Endpoint: `POST /api/rag/twin/continuous/start`
   - Will keep style signature updated as new turns arrive

**Phase 3 — Wire as Claw's Recall Backend:**

8. **Create a Clawdbot plugin** that calls RAG++ context search before answering recall questions
   - Replace broken `memory_search` tool with RAG++ backend
   - No need for OpenAI API key — uses Gemini embeddings already

---

## TRACK 2b: Graph Kernel Deep Dive

### Architecture

```
Graph Kernel (Rust binary, :8001)
├── Source: ~/Desktop/Comp-Core/packages/admissibility-kernel/
├── Binary: ~/.compcore/bin/graph_kernel_service
├── DB: Supabase PostgreSQL (same as RAG++)
├── Config: DATABASE_URL + KERNEL_HMAC_SECRET
└── Purpose: Context slicing + admissibility verification
```

### Actual API Routes (from Rust source)

| Method | Path | Purpose | Status |
|--------|------|---------|--------|
| POST | /api/slice | Construct context slice around anchor turn | ✅ Exists |
| POST | /api/slice/batch | Batch context slicing | ✅ Exists |
| POST | /api/verify_token | Verify admissibility token | ✅ Exists |
| GET | /api/policies | List registered policies | ✅ Exists |
| POST | /api/policies | Register new policy | ✅ Exists |
| GET | /health | Health check | ✅ Working |
| GET | /health/live | Liveness probe | ✅ Exists |
| GET | /health/ready | Readiness probe | ✅ Exists |
| GET | /health/startup | Startup probe | ✅ Exists |

### The Disconnect

**What RAG++ expects from GK:**
- `GET /api/knowledge?subject=X` → Triple query (entity relationships)
- `POST /api/knowledge/traverse` → Graph traversal
- `GET /api/knowledge/aliases` → Entity alias lookup

**What GK actually provides:**
- `/api/slice` → Context slicing (admissibility-focused)
- `/api/verify_token` → Token verification
- `/api/policies` → Policy management

**These are completely different capabilities.** The GK is an **admissibility kernel** (slice-based context management), not a **knowledge graph** (entity-relationship storage). The RAG++ bridge was written assuming GK would have knowledge graph endpoints that were never implemented.

### The /api/knowledge/batch 422 Mystery

Something is hitting `/api/knowledge/batch` every ~5 minutes, getting 422 each time. This endpoint **doesn't exist in the Rust routes** — it's falling through to a catch-all that returns 422 instead of 404. The caller is likely the Clawdbot ingestion pipeline trying to push extracted triples.

### What GK CAN Do (Currently)

1. **Context Slicing** — Given an anchor turn, construct a context window of related turns
2. **Admissibility Tokens** — HMAC-signed tokens proving a slice was properly constructed
3. **Policy Management** — Register and apply retrieval policies

### What GK NEEDS to Do (for the RAG++ bridge to work)

Option A: **Add knowledge graph endpoints to GK**
- Add triple storage (subject, predicate, object) table to Supabase
- Implement `GET /api/knowledge?subject=X`
- Implement `POST /api/knowledge/traverse`
- Implement `GET /api/knowledge/aliases`
- This makes GK a true knowledge graph + admissibility kernel

Option B: **Update RAG++ bridge to use GK's actual capabilities**
- Replace entity enrichment with slice-based context enrichment
- Use `/api/slice` to get contextually-related turns for any search result
- Use admissibility tokens to filter search results by context coherence
- This leverages what GK already does well

**Recommendation: Option B first (faster), then Option A for full knowledge graph.**

### Concrete GK Improvement Plan

**Phase 1 — Fix the bridge (use what exists):**

1. **Update graph_enrichment.py** to use `/api/slice` instead of `/api/knowledge`
   - For each search result, ask GK to slice around it
   - Use slice turn_ids to fetch related turns as context
   - Attach slice context to enriched results

2. **Stop the 422 spam** — find and fix whatever is calling `/api/knowledge/batch`
   - Check launchd services, cron jobs, or the ingestion pipeline

3. **Wire slice-conditioned search** — RAG++ already has `search_slice()` in query.py
   - Currently not exposed through any route handler efficiently
   - Make it the default for high-precision queries

**Phase 2 — Add knowledge graph to GK (Rust):**

4. **Add knowledge tables** to Supabase:
   ```sql
   CREATE TABLE knowledge_triples (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     subject TEXT NOT NULL,
     predicate TEXT NOT NULL,
     object TEXT NOT NULL,
     source_turn_id UUID REFERENCES memory_turns(id),
     confidence FLOAT DEFAULT 1.0,
     created_at TIMESTAMPTZ DEFAULT now()
   );
   CREATE TABLE entity_aliases (
     canonical TEXT NOT NULL,
     alias TEXT NOT NULL,
     PRIMARY KEY (canonical, alias)
   );
   ```

5. **Add routes to Rust service:**
   - `GET /api/knowledge?subject=X` → query triples
   - `POST /api/knowledge/batch` → batch insert triples (fix the 422)
   - `POST /api/knowledge/traverse` → BFS/DFS traversal
   - `GET /api/knowledge/aliases` → alias resolution

6. **Extract triples from CORE turns** — use the existing entity extraction in graph_enrichment.py to populate the knowledge graph

---

## Unified Improvement Roadmap

### Tonight (Track 1 finishing)
- [ ] MiniMax scoring completes (~1.5h)
- [ ] Generate v9 expansion from CORE+ENRICHED turns

### Quick Wins (1-2 hours of work)
- [ ] Set min_salience default to 0.3
- [ ] Exclude PRUNED from default search
- [ ] Fix enhanced search parameter bug
- [ ] Add result deduplication
- [ ] Find and fix the /api/knowledge/batch caller

### Medium Effort (4-6 hours)
- [ ] Implement composite scoring (similarity + salience + recency)
- [ ] Run style extraction on CORE turns
- [ ] Update graph_enrichment.py to use /api/slice
- [ ] Wire RAG++ as Claw's recall backend (replace memory_search)

### Deep Work (1-2 days)
- [ ] Add knowledge triple tables to Supabase
- [ ] Implement knowledge graph routes in Rust GK
- [ ] Extract and populate knowledge graph from CORE+ENRICHED turns
- [ ] Enable continuous learning pipeline
- [ ] Full integration test: search → enrich → generate

---

*Both tracks are complementary. Track 1 (MiniMax scoring) feeds Track 2 (RAG++ quality) — the scored turns become the quality filter for search results. The Graph Kernel becomes useful once it can provide structural context alongside RAG++'s semantic search.*
