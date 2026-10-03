# CodeFlowLens: Consolidated Specification (v3)

Single source of truth. Supersedes `ProjectPlan.md`, `CodeFlowLens_Plan_v2.md` and `CodeFlowLens_Plan_v3_Addendum.md`.
Target reader: you and any coding agent implementing it. Follow Section 17 (invariants) strictly.

---

## 0. Purpose and scope

**What it does.** Ingest a codebase once (target 1-3 h, unattended, resumable). Afterwards answer, quickly and with citations:
1. What does this function/class do (brief or detailed)?
2. What is the control flow from an entrypoint?
3. How does a feature work across files?
4. Structural questions (who calls X, what breaks if X changes) with no LLM at all.
5. Export a wiki: per-module pages, `ARCHITECTURE.md`, feature pages.

**Hard limits.** 10 GB VRAM, 16 GB RAM for the CodeFlowLens process (Postgres counts against RAM; keep it to about 1-2 GB).

**MVP language:** Python (via `ast`). Other languages later via tree-sitter behind the same adapter interface.

**Non-goals:** code editing, running tests as part of normal operation (dynamic tracing is opt-in), cloud LLMs.

---

## 1. Design principles

1. **Do the expensive thinking once at ingest; keep queries cheap.**
2. **Never let the LLM do what code can do:** call edges, traversal order, Mermaid, line numbers, citations are deterministic.
3. **Bottom-up, context-light prompts:** one code unit + one-line summaries of its dependencies.
4. **Graph first, LLM last:** structural questions are answered from SQL; the LLM narrates.
5. **Everything resumable and invalidation-aware** (per-symbol hashes, ripple only when summaries change).
6. **Fixed pipelines over free-form agents** for a small model; any agent loop is bounded and read-only.
7. **Measure before tuning:** eval set and metrics exist before LLM work starts.

---

## 2. Resource plan and Ollama configuration

### 2.1 Budget (estimates; verify with `cfl doctor`)

| Item | Approx. VRAM |
|---|---|
| Generation model (7B Q4_K_M class) | ~4.7 GB |
| Embedding model (0.6B-class) | ~0.5-1.2 GB |
| KV cache, 7B GQA model: ~56 KiB/token at fp16 | 4k ≈ 0.22 GB, 8k ≈ 0.45 GB, 16k ≈ 0.9 GB |
| Compute buffers / overhead | a few hundred MB |

Conclusion: start at `NUM_CTX=8192`; test 16384 if `/api/ps` shows headroom. Larger context means fewer chunk-and-merge hops. A 14B model at Q4 (~9 GB) does not fit with KV cache plus embedder under 10 GB.

### 2.2 Server environment

```
OLLAMA_NUM_PARALLEL=1          # server-side; client-side serialization alone is not enough
OLLAMA_MAX_LOADED_MODELS=2     # generator + embedder stay resident
OLLAMA_KEEP_ALIVE=-1
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0      # optional; compare quality against f16
```

### 2.3 Rules
- **One fixed `num_ctx` for all generation calls** (changing it reloads the model in Ollama, which is very slow).
- Use the **native Ollama API** (`/api/chat` or `/api/generate`, `/api/embed`). The OpenAI-compatible endpoint does not honor per-request `num_ctx`.
- `num_ctx` covers prompt + output. Always set `num_predict`.
- Ollama silently truncates over-long prompts: enforce budgets in code (Section 8) and compare to the response's `prompt_eval_count`.
- Preflight (`cfl doctor`) must call `/api/ps` and **abort if the model is not fully on GPU** (`size_vram < size`), since CPU spill is roughly 10x slower.
- Concurrency to the LLM is exactly 1 (a lock around every generation call, including chat).

---

## 3. Technology stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| CLI | `typer` (bundles `rich`), `rich` for progress |
| Parsing | `ast` (MVP); later `tree-sitter` + `tree-sitter-language-pack` |
| Graph | `networkx` (SCC condensation, PageRank, Louvain); SQL recursive CTEs for query-time traversal |
| Database | PostgreSQL 15+ with `pgvector` and `pg_trgm`; `psycopg` 3 (sync) |
| HTTP | `httpx` (sync client; streaming for answers) |
| Gitignore handling | `pathspec` |
| Tokens | `tokenizers` with the model's tokenizer file, else calibrated chars/token ratio |
| Validation | `pydantic` |
| Tests | `pytest`, `pytest-mock` |

Dev environment: `docker compose` with the `pgvector/pgvector` image. Postgres tuning for this workload: `shared_buffers` 512 MB-1 GB, small `work_mem`.

Synchronous code is intentional: LLM concurrency is 1, so async adds complexity without throughput.

---

## 4. Directory layout

```
codeflowlens/
├── pyproject.toml
├── docker-compose.yml
├── README.md
├── migrations/                  # numbered .sql files
├── cfl/
│   ├── cli.py
│   ├── config.py                # NUM_CTX, budgets, thresholds, DSN, model tags, EMBED_DIM
│   ├── core/
│   │   ├── client.py            # Ollama: generate/chat/embed, retries, usage logging
│   │   ├── budget.py            # token counting, priority assembly, hard asserts
│   │   ├── db.py                # connection, migrations, repository functions (ALL SQL lives here)
│   │   ├── hashing.py
│   │   └── preflight.py         # /api/ps residency check, benchmark, estimator
│   ├── parser/
│   │   ├── base.py              # LanguageAdapter protocol
│   │   ├── python_adapter.py
│   │   ├── resolver.py          # layered call resolution + confidence
│   │   ├── graph.py             # DiGraph, SCC condensation, layers, PageRank, communities
│   │   └── overlays/            # scip.py, dynamic.py (optional)
│   ├── pipeline/
│   │   ├── scan.py
│   │   ├── symbol_pass.py       # Stage 3 (resumable worker loop)
│   │   ├── modules.py           # Stage 4/5: module tree + features
│   │   └── indexer.py           # Stage 6: lexical/dense views
│   ├── engines/
│   │   ├── graph_queries.py     # callers/callees/path/impact/hubs/dead (no LLM)
│   │   ├── explain.py
│   │   ├── flow.py
│   │   ├── ask.py               # router, retrieval, answer, validation
│   │   ├── chat.py              # session + deterministic compaction
│   │   └── docs_export.py
│   └── prompts/
│       ├── prompts.py
│       └── schemas.py
├── eval/
│   ├── sample_repo/
│   └── questions.yaml
└── tests/
    ├── fakes.py                 # FakeOllama
    └── test_*.py
```

Rule: engines never write SQL; only `core/db.py` does.

---

## 5. Data model (PostgreSQL)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- keys: schema_version, gen_model_tag, embed_model_tag, num_ctx, prompt_version

CREATE TABLE files (
  path TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  language TEXT,
  size_bytes INT,
  token_est INT,
  parsed_at TIMESTAMPTZ,
  summary TEXT,
  summary_ctx_hash TEXT
);

CREATE TABLE symbols (
  id TEXT PRIMARY KEY,                       -- path::Qual.name (+@line if ambiguous)
  file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('function','method','class','nested')),
  qualname TEXT NOT NULL,
  name TEXT NOT NULL,
  parent_id TEXT,
  signature TEXT,
  decorators TEXT,
  docstring TEXT,
  start_line INT NOT NULL,
  end_line INT NOT NULL,
  raw_code TEXT NOT NULL,
  code_hash TEXT NOT NULL,
  token_est INT,
  pagerank DOUBLE PRECISION,
  summary_short TEXT,                        -- one line; used in prompts, retrieval, roll-ups
  summary_json JSONB,                        -- structured summary (schema in Section 13)
  summary_long TEXT,                         -- lazily generated detailed explanation (cached)
  ctx_hash TEXT,                             -- hash(code_hash + callee summary_short hashes + prompt_version + gen_model_tag)
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','done','failed','skipped_trivial')),
  attempts INT NOT NULL DEFAULT 0,
  error TEXT,
  search_text TEXT,                          -- identifier-split text built in Python
  search TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', coalesce(search_text,''))) STORED,
  embed_hash TEXT
);
CREATE INDEX symbols_search_idx ON symbols USING GIN (search);
CREATE INDEX symbols_qualname_trgm ON symbols USING GIN (qualname gin_trgm_ops);
CREATE INDEX symbols_status_idx ON symbols (status);
CREATE INDEX symbols_file_idx ON symbols (file_path);

CREATE TABLE edges (
  caller_id TEXT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
  callee_id TEXT,                            -- NULL when external/unresolved
  callee_expr TEXT NOT NULL,                 -- raw text e.g. self.repo.save
  line INT NOT NULL,
  kind TEXT,                                 -- call | constructor | decorator
  resolution TEXT,                           -- local|self|import|ctor|typed|name-unique|ambiguous|external|scip|dynamic
  source TEXT NOT NULL DEFAULT 'ast' CHECK (source IN ('ast','scip','dynamic')),
  confidence REAL NOT NULL DEFAULT 0.5,
  control_ctx TEXT,                          -- e.g. "if>for>try"
  PRIMARY KEY (caller_id, callee_expr, line, source)
);
CREATE INDEX edges_callee_idx ON edges (callee_id);
CREATE INDEX edges_caller_idx ON edges (caller_id);

-- Content-addressed vectors. Dimension must match the chosen embedding model.
CREATE TABLE embeddings (
  hash TEXT PRIMARY KEY,                     -- hash(embed_text + embed_model_tag)
  vec vector(:EMBED_DIM) NOT NULL            -- templated by the migration runner from config
);

CREATE TABLE module_tree (
  id BIGSERIAL PRIMARY KEY,
  parent_id BIGINT REFERENCES module_tree(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  dir_hint TEXT,
  member_hash TEXT NOT NULL,                 -- hash of member symbol ids; stability key for names
  summary TEXT,
  aggregate_hash TEXT
);
CREATE TABLE module_members (
  module_id BIGINT REFERENCES module_tree(id) ON DELETE CASCADE,
  symbol_id TEXT REFERENCES symbols(id) ON DELETE CASCADE,
  PRIMARY KEY (module_id, symbol_id)
);

CREATE TABLE features (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  entry_symbol_id TEXT REFERENCES symbols(id) ON DELETE SET NULL,
  summary TEXT,
  aggregate_hash TEXT
);
CREATE TABLE feature_members (
  feature_id BIGINT REFERENCES features(id) ON DELETE CASCADE,
  symbol_id TEXT REFERENCES symbols(id) ON DELETE CASCADE,
  role TEXT,
  PRIMARY KEY (feature_id, symbol_id)
);

CREATE TABLE view_status (
  view TEXT PRIMARY KEY,                     -- lexical | dense | graph | modules
  status TEXT NOT NULL,                      -- fresh | stale | failed
  built_at TIMESTAMPTZ,
  config JSONB
);

CREATE TABLE answer_cache (
  query_hash TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  answer TEXT NOT NULL,
  index_version TEXT NOT NULL
);

CREATE TABLE eval_runs (
  run_id TEXT, question_id TEXT, metrics JSONB, config JSONB,
  PRIMARY KEY (run_id, question_id)
);
```

Connection rules: set autocommit per unit of work; **one transaction per symbol summary**; every connection uses UTF-8 and `statement_timeout` for query paths.

### Invalidation rules
- File `sha256` changed: re-parse; match symbols by `code_hash`; unchanged symbols keep their rows (line numbers updated).
- A symbol is re-summarized only if its `ctx_hash` changes. `ctx_hash` includes callees' `summary_short` hashes, so a callee change ripples to callers only when its summary text actually changed.
- Bumping `prompt_version` or `gen_model_tag` in `meta` intentionally invalidates all summaries.
- Module/feature rows regenerate when their `aggregate_hash` (hash of member summaries) changes.
- Embeddings are reused when `hash(embed_text + embed_model_tag)` already exists.

---

## 6. Pipeline stages

```
Stage 0  Preflight     Ollama health, models pulled, fully on GPU, tokens/s benchmark, DB extensions
Stage 1  Scan + Parse  files -> symbols -> raw call sites                      (no LLM, minutes)
Stage 2  Resolve       call sites -> edges (+ optional overlays), SCCs, layers, PageRank
Stage 3  Symbol pass   bottom-up summaries                                      (LLM, hours, resumable)
Stage 4  Module tree   cluster graph, summarize leaves, then parents            (LLM, short)
Stage 5  Features      entrypoints + communities -> named feature cards          (LLM, short)
Stage 6  Index         lexical (tsvector/trgm) + dense (embeddings) views        (embedding model)
```

`cfl build` runs all stages in order; each stage is idempotent and resumable. Stage 1-2 run with the LLM idle; Stage 6 can run per batch or at the end (batch via `/api/embed`).

---

## 7. Parsing and call resolution

### 7.1 Extraction (Python adapter)
Per symbol: kind, qualname, signature (params, annotations, return), decorators, docstring, start/end line, raw code, parent, async flag, **call sites** (callee expression text, line, enclosing control context: `if`/`for`/`while`/`try`/`with`/comprehension). Never crash on syntax errors: record the file as `parse_failed`, continue.

File discovery: respect `.gitignore` (pathspec) plus defaults (`.git`, `node_modules`, `venv`, `dist`, `build`, `__pycache__`); skip binaries, minified/generated files, lockfiles, files over a size cap, and secret-like files (`.env*`, key files).

### 7.2 Layered resolution (record `resolution` and `confidence`)
Default confidences (tune on your repo):

| Layer | Resolution | Confidence |
|---|---|---|
| import-map exact (`from a import b as c`, `import a.b as m`) | `import` | 0.95 |
| same module / local scope | `local` | 0.90 |
| `self.x()` / `cls.x()` to enclosing class (walk bases in index) | `self` | 0.90 |
| constructor `Foo()` to `Foo.__init__` and class | `ctor` | 0.90 |
| attribute on typed name (`x = Foo()`, annotated params) | `typed` | 0.80 |
| unique simple name project-wide | `name-unique` | 0.75 |
| several candidates: store all | `ambiguous` | 0.40 |
| nothing found (stdlib/third-party) | `external` | n/a (callee_id NULL) |

Store unresolved/external/ambiguous edges; do not drop them. Query-time default threshold: `confidence >= 0.6` (configurable).

### 7.3 Graph
Build `DiGraph` from edges above the threshold. Use `strongly_connected_components` then `condensation` (linear time; do **not** use `simple_cycles`). Recursion and mutual recursion become single SCC nodes summarized together. Compute PageRank (importance) and Louvain communities (clusters) from the same graph.

### 7.4 Optional overlays (each may fail without breaking the build)
1. **SCIP overlay** (`scip-python`, Pyright-based; newer `ty-scip` is preview): subprocess with timeout and per-file exclusion list; decode occurrences, upgrade matching edges to confidence 1.0, add missing ones with `source='scip'`. Needs Node.js and Python 3.10+; has known crashes on some large dependencies, so "overlay unavailable" must degrade silently to the AST resolver.
2. **Dynamic overlay** (`cfl trace -- <command>`, opt-in): record actual caller to callee pairs while running tests/entry commands (Python 3.12+ `sys.monitoring`, or `sys.setprofile` earlier), stored as `source='dynamic'`. Executes your code; run in a throwaway environment.

---

## 8. Token budgeting (`core/budget.py`)

- Budget = `NUM_CTX` - template overhead - output reserve (per task: ~300 for one-liners, ~1,200 for detailed).
- Counting: real tokenizer if available; otherwise chars/3.2 for code with a 15% margin. After each call, compare to `prompt_eval_count` and auto-calibrate the ratio; log drift.
- **Priority-based assembly:** always keep (1) target code, (2) callee one-liners, (3) caller one-liners, (4) docstring/signature; drop or truncate the lowest priority first.
- **Oversized function:** split body at top-level statement boundaries (AST-aware), summarize chunks as partials, then combine partials with the signature.
- **Oversized roll-ups:** batch children, merge intermediate summaries.
- **Hard assertion:** if prompt tokens + `num_predict` > `NUM_CTX`, raise; never send.
- **Per-tool output caps** in chat mode (Section 11.4), enforced here.

---

## 9. Symbol pass (Stage 3)

**Order:** reverse topological order over the SCC condensation (callees before callers). Within a layer: PageRank descending, then file-grouped (helps cache reuse). If stopped early, the most important code is already done.

**Skip trivial symbols** (`status='skipped_trivial'`): getters/setters, one-line wrappers, `__repr__`/`__str__`, bodies that are just `pass` or `raise NotImplementedError`. Generate a templated one-liner from the signature; no LLM call.

**Per-symbol call:**
- Fresh isolated context every time (no running history; state lives in Postgres).
- Prompt layout for prefix reuse: stable prefix first (system message + schema + fixed instructions), then variable content (code + callee one-liners), then one short reminder line. Verify gain via `prompt_eval_duration`.
- Structured output via Ollama `format` JSON schema; validate with Pydantic; one repair retry; then quarantine (`status='failed'`, raw output stored in `error`).
- If the model has a reasoning/"thinking" mode, disable it for bulk summaries (check the option name in your Ollama version).
- `temperature=0.1`, fixed `seed`, `num_predict` capped.

**Classes:** after their methods, summarize from docstring, attributes assigned in `__init__`, and method one-liners.

**Retry:** exponential backoff on 5xx/OOM, max 3 attempts, then quarantine; never block the run.

**Progress:** rich bar with done/total, tokens/s, rolling ETA; `cfl status` shows pending/done/failed/skipped counts.

**Throughput sanity check:** at roughly 1,500 prompt + 150 output tokens per symbol, a 7B Q4 model on a mid-range GPU takes on the order of 3-6 s per symbol, so 1-3 h is on the order of 2,000-3,500 symbols. `cfl build --estimate` computes your real number from symbol count, token totals and the measured tokens/s, and offers `--skip-tests`, `--min-lines N`, `--priority entrypoints-first` if the estimate exceeds your time budget.

**Per-call trace log (JSONL):** symbol id, estimated vs actual prompt tokens, output tokens, latency, validation result.

---

## 10. Module tree and features (Stages 4-5)

### 10.1 Module tree (replaces plain directory roll-up)
1. **Top-down clustering (deterministic):** Louvain over CALLS and IMPORTS edges, with the directory tree as a prior (boost edge weight inside a package). Recurse into clusters exceeding a size/token threshold.
2. **Leaf modules:** one LLM call from member `summary_short`s + signatures + a few top-PageRank bodies (never all code). Output: purpose, collaboration, public surface, data flow.
3. **Parent modules:** synthesized from child pages, batched under budget.
4. **Names:** LLM proposes titles; keep names stable across rebuilds by matching new clusters to old ones via `member_hash` similarity.
5. **Diagrams:** generated by code (containment/dependency), validated before saving.
6. **Config/build/CI files** (Dockerfile, CI YAML, `pyproject.toml`) are members of a "Build, Deployment and Configuration" module.
7. **Incremental:** regenerate only modules containing changed/added/removed symbols, plus ancestors.

### 10.2 Features
- Seeds = entrypoints (CLI commands, route handlers/decorators, `main`, public API).
- For each seed: depth-limited, confidence-filtered reachable subgraph. Merge seeds with heavily overlapping subgraphs; Louvain communities catch features with no clear entrypoint.
- One LLM call per feature from member one-liners; store in `features` / `feature_members`; embed feature cards for retrieval.

---

## 11. Retrieval and query runtime

### 11.1 Router (no LLM first; regex + FTS/trigram)
| Query pattern | Path |
|---|---|
| mentions a known symbol / "what does X do" | explain |
| "who calls", "what depends on", "what breaks if", "path from A to B", "most important" | graph queries (no LLM; optional `--explain` narration) |
| "flow", "trace", "what happens when" + symbol | flow |
| "how does X work", "where is X handled" | feature + retrieval |
| otherwise | retrieval |

### 11.2 Units and views
Units are addressed by `path:start-end`: **L0** file skeleton (signatures only), **L1** class summary, **L2** callable.
Views are built independently and failure-isolated (tracked in `view_status`): lexical (`tsvector` with identifier splitting + `pg_trgm`), dense (pgvector over L2 and L0; embedded text = signature + summary_short + docstring + code head), structural (edges table).

### 11.3 Default retrieval plan
Lexical + dense fused with RRF; take top ~8-10 candidates; assemble within the token budget (expect only about 3-6 full blocks at 8k, so budget by tokens, not count); every block tagged with `path:start-end`.

**Off by default, enabled only if your eval shows a gain:**
- *Graph expansion* (add neighbors of top seeds): published results were inconclusive; if tested, tune the graph weight on held-out questions.
- *LLM reranking:* costs seconds per query for a few points of recall.

**Vector index:** exact scan first; add pgvector HNSW only beyond roughly 100k vectors (measure).
**Embedding model:** benchmark a code-capable embedder (e.g., a Qwen3-Embedding 0.6B-class model) against `nomic-embed-text`; verify Ollama tag and dimension; set `EMBED_DIM` accordingly before running migrations.

### 11.4 Answering
**Default: fixed pipeline**
```
classify -> structural? (SQL, no LLM)
         -> retrieve -> assemble (budgeted) -> answer (stream) -> validate
```
**Modes:** `--brief` (from `summary_short` / feature card, often zero LLM calls, or at most ~120 output tokens) and `--detailed` (uses `summary_json` + code + neighbor one-liners; generates and caches `summary_long` on demand).

**Chat (`cfl chat`):** DB, vector access and Ollama connection stay warm; streaming. Session memory is **deterministic compaction**: mask old tool observations, keep last N turns plus a short "direction seed" (symbols/files already discussed). No LLM summarization of history.

**Optional bounded loop (`cfl chat --deep`):** at most 3 tool calls, JSON-schema-constrained, read-only tools only: `search(query)`, `get_symbol(id, detail)`, `callers(id)`, `callees(id)`, `read_range(path, start, end)`. Results are handles plus one-liners first (progressive disclosure); oversized results keep head and tail plus a pointer; per-tool token caps enforced.

### 11.5 Verification hooks (run after every generated answer)
1. Every `[path:start-end]` citation must exist and overlap a retrieved chunk; strip or flag the rest.
2. Backticked identifiers that look like symbols must resolve in the index; else warn.
3. Mermaid passes a syntax check.
4. If nothing from context is cited: respond "insufficient context" and show top hits.

---

## 12. Flow engine

1. Resolve entrypoint (fuzzy match; ask to disambiguate if several).
2. **DFS over call sites in source-line order** (not topological order), depth-limited (`--depth`, default 4), cycle-safe via a visiting set; recursion marked `↻`.
3. Build a **trace tree**: node = symbol; children = ordered callees, each annotated with line, control context (`if`/loop/`try`), edge source and confidence. External calls are leaves.
4. Render Mermaid (`flowchart TD` or `sequenceDiagram`) **in code**; collapse beyond depth or ~40 nodes into "...N more"; validate.
5. Narrative: one small LLM call per step group using stored one-liners and call-site snippets (not full bodies), then a stitching call. Every prompt stays within budget regardless of trace size.
6. Output `flow.md`: narrative, diagram, and a table (`file:line`, symbol, condition, edge source/confidence).

Topological order is used only for bottom-up summarization (Section 9), never for runtime ordering.

---

## 13. Prompts and schemas

Prompts live in `prompts/prompts.py`; `PROMPT_VERSION` constant is stored in `meta` and included in `ctx_hash`.

**System message (all ingest calls):** "You are a precise code documentation engine. Describe only what the provided code and context show. Never invent symbols, files or behavior not present."

**Symbol summary** (JSON schema via Ollama `format`):
```json
{
  "one_liner": "string, <= 25 words",
  "purpose": "string, 1-2 sentences",
  "inputs": "string",
  "returns": "string",
  "side_effects": ["string"],
  "raises": ["string"],
  "notable_logic": "string, <= 60 words"
}
```
Variable section of the prompt: `[FILE] path`, `[SIGNATURE]`, `[CALLEES: id: one_liner ...]`, `[CODE]`, then a one-line reminder ("Return only the JSON object").

**Other prompts (same conventions, concise):**
- `PROMPT_EXPLAIN_FUNCTION`: target code + caller/callee one-liners; output: purpose (1-2 sentences), chronological logic, assumptions/side effects/returns. Brief and detailed variants.
- `PROMPT_FLOW_STEP` / `PROMPT_FLOW_STITCH`: step-group narrative from one-liners and call-site snippets.
- `PROMPT_MODULE_LEAF`, `PROMPT_MODULE_PARENT`: purpose, how members collaborate, public surface, data flow; < 300 words.
- `PROMPT_FEATURE_CARD`: feature name + end-to-end description from member one-liners.
- `PROMPT_ANSWER`: "Answer using ONLY the context blocks. Cite as [path:start-end]. If the context is insufficient, say what is missing." Context blocks are tagged with ranges.

---

## 14. CLI

| Command | Purpose |
|---|---|
| `cfl doctor` | Ollama + models + GPU residency + DB/extensions; prints tokens/s and VRAM at max-length prompt |
| `cfl build [REPO] [--estimate] [--resume] [--skip-tests] [--min-lines N] [--priority entrypoints-first]` | Stages 1-6 |
| `cfl status [--failed]` | Progress, ETA, stale counts, failures |
| `cfl explain <name\|path::name> [--brief\|--detailed]` | Function/class explanation (fuzzy lookup) |
| `cfl flow <entrypoint> [--depth N] [--out flow.md]` | Trace + Mermaid + narrative |
| `cfl feature <query\|name>` | Feature card + retrieval |
| `cfl ask "<question>" [--brief\|--detailed]` | Router + pipeline |
| `cfl chat [--deep]` | Warm REPL |
| `cfl callers / callees / path / impact / hubs / dead / where` | Graph and lookup queries (no LLM) |
| `cfl trace -- <cmd>` | Optional dynamic edge capture |
| `cfl docs [--out ./docs]` | Export `SUMMARY.md`, `ARCHITECTURE.md`, module pages, feature pages from cache (no LLM) |
| `cfl eval [--config ...]` | Run eval set; store metrics in `eval_runs` |

---

## 15. Evaluation

`eval/questions.yaml`: 40-60 items, hand-verified, across six types:

| Type | Example | Expected evidence |
|---|---|---|
| symbol hint | "What does `validate_token` do?" | that symbol's range |
| file/module hint | "What is `auth/` responsible for?" | module page + key files |
| behavioral | "Where are retries handled?" | spans in the right files |
| traversal | "Trace from `run_pipeline` to the DB write" | ordered spans along a path |
| structural | "What calls `parse_config`?" | exact caller set (SQL-checkable) |
| reasoning | "Why could `save()` fail when X is None?" | relevant spans + correct conclusion |

Metrics: **AnswerRecall@5** (expected spans overlapped by the first five distinct cited spans), **citation validity**, **structural exactness** (set equality), **tokens per answer**, and a human 1-5 rubric on ~10 docs/answers per release. If you add an LLM judge, treat it as a trend signal only.

Use the eval to decide: graph expansion on/off, reranking on/off, embedding model, `NUM_CTX`, resolver thresholds, model choice.

**Model/embedding bake-off (one hour, your code):** ~40 symbols + ~30 eval questions; candidates: the baseline 7B coder, a ~9B-class newer model, an ~8B general model, optionally an MoE coder with expert offload (experimental; do not plan around it). Verify tags/sizes/licenses before pulling. Measure rubric quality, JSON-validity rate, tokens/s, peak VRAM (`/api/ps`), and projected ingest hours (`cfl build --estimate`). Choose the best quality that meets your hour budget and stays under 10 GB at your `NUM_CTX`.

---

## 16. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| 0 | Workspace, config, `docker-compose` Postgres, `cfl doctor` | GPU residency, tokens/s, VRAM and DB extensions verified |
| 1 | DB layer, migrations, hashing, `FakeOllama` | No-op rebuild does zero work (tested); migrations apply cleanly |
| 2 | Python adapter + layered resolver (with confidence) | Fixture repo (methods, imports, aliases, recursion): expected edges and SCCs; ambiguous edges stored |
| 3 | Graph commands (`callers/callees/path/impact/hubs/dead/where`) | All run from SQL; exactness tests pass; zero LLM calls |
| 4 | Budget module + Ollama client | Over-budget prompt raises; `prompt_eval_count` logged; retry/backoff tested with fake 503s |
| 5 | Eval set v0 + metrics harness | Structural and retrieval metrics runnable before LLM work |
| 6 | Model/embedding bake-off | Decision recorded with numbers |
| 7 | Symbol pass (bottom-up, structured output, quarantine, resume) | Kill + `--resume` works; one-function edit re-runs only affected symbols |
| 8 | Hybrid retrieval + citation validator + `cfl ask` | AnswerRecall@5 and citation validity reported |
| 9 | `cfl explain` and `cfl flow` | Source-order trace; Mermaid validated; works on cycles |
| 10 | Module tree + `cfl docs` | Incremental regeneration touches only affected modules |
| 11 | `cfl chat` (+ optional `--deep`) | Per-turn tokens within budget; eval no worse than pipeline |
| 12 | Optional overlays (SCIP, dynamic edges) and eval-gated options | Each shown to help on eval, or left off |
| 13 | Full-scale run on your real repo | Time, VRAM (< 10 GB) and RAM (< 16 GB) budgets met, or warned beforehand |

**Minimum viable cut:** milestones 0-5, 7, 8. Milestone 3 is the cheapest, highest-value early win.

---

## 17. Invariants for the implementing agent

1. Exactly one in-flight LLM generation at any time; a lock wraps every generation call.
2. `NUM_CTX` is a single constant used by every generation request; never vary it per call.
3. No prompt is sent without passing the budget assertion; `num_predict` is always set.
4. No LLM call extracts edges, line numbers, citations or Mermaid; those are produced and validated in code.
5. All SQL lives in `core/db.py`; engines call repository functions.
6. Every pipeline stage is idempotent and resumable; one transaction per symbol summary.
7. File/symbol hashes are checked before any LLM call; unchanged work is skipped.
8. Optional components (SCIP, dynamic tracing, reranker, graph expansion) must degrade gracefully and default to off unless eval-approved.
9. Model/prompt/version changes invalidate through `meta` and `ctx_hash`, never silently reuse stale summaries.
10. Never execute user code except via the explicit `cfl trace` command.
11. Every generated answer passes the verification hooks (Section 11.5) before it is shown as final.
12. Tests use `FakeOllama`; no test requires a GPU.

---

## 18. Defaults and risks at a glance

| Setting / risk | Default or mitigation |
|---|---|
| `NUM_CTX` | 8192 (try 16384 after measuring) |
| `TEMPERATURE` | 0.1, fixed seed |
| Edge confidence threshold | 0.6 |
| Retrieval top-k | ~8-10 candidates, 3-6 injected blocks by token budget |
| Graph expansion / reranker / SCIP / dynamic tracing | off until eval-approved |
| VRAM spill to CPU | `cfl doctor` aborts |
| Silent truncation | budget assertions + `prompt_eval_count` check |
| Ingest overruns time budget | `--estimate`, trivial-skip, PageRank ordering, resumable runs |
| Wrong flows from bad edges | confidence + source shown in output; eval fixtures |
| Summary hallucination | structured output, callee grounding, verification hooks |
| Stale summaries | `ctx_hash` ripple invalidation; `cfl status` flags stale |
| Evidence transfer | Published gains come mostly from preprints and larger models; validate on your own repo with the eval set |
