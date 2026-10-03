# CodeFlowLens: Detailed Execution Order

Companion to `CodeFlowLens_Spec_v3_Consolidated.md` (the spec). The spec says *what* and *why*; this file says *in which order, which file, which function, which command*. If the two conflict, the spec's Section 17 invariants win, then the spec, then this file. Where this file adds something the spec does not say, it is marked **[D#]** and listed in Section 1.

---

## 0. How an agent uses this document

1. Execute steps strictly in order (M0.1, M0.2, ...). A step is finished only when its **Verify** command passes.
2. Before each step, re-read the spec sections cited in that step.
3. After each step: `git add -A && git commit -m "<step id>: <title>"`.
4. Never write SQL outside `cfl/core/db.py`. Never send an LLM prompt without the budget assertion. Never vary `NUM_CTX`. (Invariants 1-12 are mapped to tests in Section 6.)
5. Every test uses `FakeOllama`. Tests that need Postgres carry `@pytest.mark.db`. Fast loop: `pytest -q -m "not db"`; full loop: `pytest -q`.
6. If a step is blocked (tool missing, tag does not exist), do not improvise silently: write the blocker into `docs/BLOCKERS.md` and continue with the next step that does not depend on it.
7. Minimum viable cut (spec Section 16): M0-M5, M7, M8. Do those first; M6 (bake-off) is a decision step that can be done with the MVP in place.

---

## 1. Spec gaps found and the decisions taken

The spec's schema and layout are slightly incomplete for the behavior it requires. These additions are deliberate and minimal.

| ID | Gap in spec | Decision |
|---|---|---|
| D1 | `edges` PK `(caller_id, callee_expr, line, source)` cannot store several `ambiguous` candidates for one call site, and `callee_id` is nullable | Surrogate `id BIGSERIAL` PK plus `UNIQUE INDEX (caller_id, callee_expr, line, source, COALESCE(callee_id,''))` |
| D2 | Stage 1 produces raw call sites, imports, class bases, `parse_failed`; nowhere to keep them for Stage 2 | Add `files.parse_status/parse_error/imports/exports/embed_hash`; `symbols.call_sites/extra/is_async/is_entrypoint/entry_kind/scc_id/layer` (all JSONB/simple columns) |
| D3 | Module-level code (`if __name__ == "__main__":`, `app = create_app()`) has no symbol, so entrypoints are missed and `dead` gives false positives | Allow `symbols.kind = 'module'` pseudo-symbol `path::<module>`, only when module-level code contains calls; never sent to the LLM (`skipped_trivial` with templated one-liner) |
| D4 | Classes must be summarized after their methods, but ctor edges point *to* classes | Ordering graph = call edges + structural `class -> its methods` edges. Call graph (PageRank, Louvain, queries) stays calls-only |
| D5 | Missing meta keys | `meta`: `repo_root`, `embed_dim`, `index_epoch`, `resolve_fingerprint`, `last_rate`, `tokenizer_file` |
| D6 | `name-unique` on `.get/.append/.items` would link stdlib calls to your project | `name-unique` skips names in `COMMON_METHOD_BLOCKLIST` (dict/list/str/set/file methods) and dunders |
| D7 | `confidence` is NOT NULL but external edges are "n/a" | External edges store `callee_id NULL`, `confidence 0.0` |
| D8 | "PageRank descending, then file-grouped" is ambiguous | Within a layer: files ordered by max PageRank of their symbols (desc); symbols inside a file by PageRank desc |
| D9 | `EMBED_DIM` must be set before migrations but the embedder is chosen in M6 | `embeddings` table lives in its own migration `0002`, with a dim guard in `meta.embed_dim` and `cfl db reset-embeddings --dim N` |
| D10 | Layout lacks files needed by the spec's features | Added: `cfl/core/mermaid.py`, `cfl/core/trace_log.py`, `cfl/engines/verify.py`, `cfl/pipeline/build.py`, `cfl/eval/{metrics,runner}.py`, `scripts/*`, `docs/decisions/*` |
| D11 | With prefix reuse Ollama may report a *lower* `prompt_eval_count` than the full prompt | Calibrate chars/token only from samples with `actual >= 0.6 * estimate`; log the rest as `cache_hit_suspected` |
| D12 | Schema has no repo id | One database per repo (`CFL_DSN` per project) |
| D13 | Silent truncation also exists on `/api/embed` | Send `truncate:false` and cap embed text by `EMBED_MAX_TOKENS`; send `think:false` for generation when `GEN_DISABLE_THINKING=true` (name may differ by Ollama version: check) |
| D14 | Config/build files (Dockerfile, CI YAML) are module members but not symbols | New table `module_files(module_id, file_path)` |

---

## 2. Dependency manifest

**System (install once, M0.1):** Python 3.11+, Docker + Docker Compose v2, Ollama, NVIDIA driver (`nvidia-smi` works), git. Optional later: Node.js 18+ (SCIP overlay, M12).

**Python runtime (`pyproject.toml` `dependencies`):**

| Package | Min version | Used for |
|---|---|---|
| `typer` | 0.12 | CLI (pulls `rich`) |
| `rich` | 13.7 | progress, tables, streaming output |
| `httpx` | 0.27 | Ollama client (sync, streaming) |
| `psycopg[binary]` | 3.1 | Postgres driver (sync) |
| `pgvector` | 0.2.5 | `register_vector` for psycopg 3 |
| `networkx` | 3.2 | SCC, condensation, PageRank, Louvain |
| `numpy`, `scipy` | 1.26, 1.11 | required by `networkx.pagerank` |
| `pathspec` | 0.12 | `.gitignore` handling |
| `tokenizers` | 0.19 | exact token counts when a `tokenizer.json` is available |
| `pydantic` | 2.6 | config + LLM output validation |
| `pyyaml` | 6 | `eval/questions.yaml` |

**Extras:** `dev` = `pytest>=8`, `pytest-mock>=3.12`, `ruff`; `sys` = `psutil` (RAM check in doctor, M13 monitor); `scip` = `protobuf>=4` (M12). Tree-sitter is out of MVP scope.

**Docker image:** `pgvector/pgvector:pg16`. **Ollama models (provisional, verify tags exist before pulling):** generator `qwen2.5-coder:7b`, embedder `nomic-embed-text`; M6 decides final tags.

---

## 3. Global conventions

- **Paths:** project root = repo of CodeFlowLens. State dir `.cfl/` (gitignored): `.cfl/logs/calls-YYYYMMDD.jsonl` (per-call trace), `.cfl/bench.json` (doctor benchmark), `.cfl/calibration.json`, `.cfl/tokenizers/`.
- **Config precedence:** env `CFL_*` > `cfl.toml` (parsed with stdlib `tomllib`) > defaults in `config.py`.
- **Errors:** define in `cfl/core/errors.py`: `CflError`, `BudgetExceeded`, `PreflightError`, `LLMTransportError`, `LLMValidationError`, `AmbiguousSymbol`. CLI catches `CflError` and prints one line + exit code 1.
- **Typing/style:** type hints everywhere, `from __future__ import annotations`, `ruff check` clean, no globals except `get_settings()` cache and the generation lock.
- **Logging:** stdlib `logging`, level from `CFL_LOG`; rich handler on CLI.
- **Time/DB:** every connection UTF-8; query paths set `statement_timeout`; autocommit per unit of work; Stage 3 = one transaction per symbol summary.
- **Determinism:** `temperature=0.1`, `seed=42`, sort everything that is hashed.

---

## 4. File inventory and creation order

| Step | Files created (all paths relative to project root) |
|---|---|
| M0.2 | `pyproject.toml`, `.gitignore`, `README.md`, `cfl/__init__.py`, `cfl/core/__init__.py`, `cfl/parser/__init__.py`, `cfl/parser/overlays/__init__.py`, `cfl/pipeline/__init__.py`, `cfl/engines/__init__.py`, `cfl/prompts/__init__.py`, `tests/__init__.py` |
| M0.3 | `docker-compose.yml`, `.env.example`, `scripts/ollama_env.sh`, `scripts/pull_models.sh` |
| M0.4 | `cfl/config.py`, `cfl/core/errors.py` |
| M0.5 | `cfl/core/client.py` (health subset), `cfl/core/db.py` (connect subset), `cfl/core/preflight.py`, `cfl/cli.py` (skeleton + `doctor`) |
| M1.1-1.6 | `cfl/core/hashing.py`, `migrations/0001_core.sql`, `migrations/0002_embeddings.sql`, `cfl/core/db.py` (full), `tests/fakes.py`, `tests/conftest.py`, `cfl/pipeline/scan.py` (discovery + file registry), `tests/test_hashing.py`, `tests/test_db.py`, `tests/test_scan.py` |
| M2.1-2.7 | `cfl/parser/base.py`, `cfl/parser/python_adapter.py`, `cfl/parser/resolver.py`, `cfl/parser/graph.py`, `tests/fixtures/resolution_repo/**`, `tests/test_python_adapter.py`, `tests/test_resolver.py`, `tests/test_graph.py`, `cfl/pipeline/scan.py` (Stage 1+2 runners) |
| M3.1-3.3 | `cfl/engines/graph_queries.py`, CLI commands `callers/callees/path/impact/hubs/dead/where`, `tests/test_graph_queries.py` |
| M4.1-4.3 | `cfl/core/budget.py`, `cfl/core/client.py` (full), `cfl/core/trace_log.py`, `tests/test_budget.py`, `tests/test_client.py` |
| M5.1-5.4 | `eval/sample_repo/**`, `eval/questions.yaml`, `eval/rubric.md`, `cfl/eval/__init__.py`, `cfl/eval/metrics.py`, `cfl/eval/runner.py`, `cfl/pipeline/indexer.py` (lexical half), `tests/test_eval_metrics.py` |
| M6.1-6.3 | `scripts/bakeoff.py`, `docs/decisions/0001-model-choice.md` |
| M7.1-7.6 | `cfl/prompts/schemas.py`, `cfl/prompts/prompts.py`, `cfl/pipeline/symbol_pass.py`, `cfl/pipeline/build.py`, CLI `build/status`, `tests/test_symbol_pass.py` |
| M8.1-8.5 | `cfl/pipeline/indexer.py` (dense half), `cfl/engines/ask.py`, `cfl/engines/verify.py`, CLI `ask`, `tests/test_ask.py`, `tests/test_verify.py` |
| M9.1-9.4 | `cfl/core/mermaid.py`, `cfl/engines/explain.py`, `cfl/engines/flow.py`, CLI `explain/flow`, `tests/test_explain.py`, `tests/test_flow.py`, `tests/test_mermaid.py` |
| M10.1-10.5 | `cfl/pipeline/modules.py`, `cfl/engines/docs_export.py`, CLI `feature/docs`, `tests/test_modules.py`, `tests/test_docs_export.py` |
| M11.1-11.3 | `cfl/engines/chat.py`, CLI `chat`, `tests/test_chat.py` |
| M12.1-12.4 | `cfl/parser/overlays/scip.py`, `cfl/parser/overlays/dynamic.py`, `cfl/parser/overlays/_tracer_shim/sitecustomize.py`, CLI `trace`, eval configs `eval/configs/*.toml` |
| M13.1-13.2 | `scripts/monitor.sh`, `docs/fullscale_report.md` |

---

## 5. Execution steps

### MILESTONE 0: Workspace, config, Postgres, `cfl doctor`

**M0.1 System prerequisites** (spec 0, 2)
```bash
python3.11 --version                    # must be >= 3.11
docker --version && docker compose version
nvidia-smi                              # confirm GPU + VRAM (>= 10 GB)
git --version
# Ollama (Linux). macOS/Windows: use the installer from ollama.com
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```
Verify: all commands succeed. If `nvidia-smi` fails, stop: the whole design assumes GPU residency.

**M0.2 Repo scaffold, venv, pyproject** (spec 3, 4)
```bash
mkdir codeflowlens && cd codeflowlens && git init
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
mkdir -p cfl/{core,parser/overlays,pipeline,engines,prompts} migrations eval/sample_repo tests/fixtures scripts docs/decisions .cfl
touch cfl/__init__.py cfl/core/__init__.py cfl/parser/__init__.py cfl/parser/overlays/__init__.py cfl/pipeline/__init__.py cfl/engines/__init__.py cfl/prompts/__init__.py tests/__init__.py
```
Create `pyproject.toml` (setuptools backend): `requires-python = ">=3.11"`; `dependencies` from Section 2; `[project.optional-dependencies]` `dev`, `sys`, `scip`; `[project.scripts] cfl = "cfl.cli:app"`; `[tool.pytest.ini_options]` with `markers = ["db: needs Postgres"]`, `testpaths = ["tests"]`; `[tool.ruff]` line-length 100. `.gitignore`: `.venv/ .cfl/ __pycache__/ *.egg-info/ .env docs/_out/`. `README.md`: 10-line stub (filled in M13).
```bash
pip install -U pip
pip install -e ".[dev,sys]"
cfl --help    # fails until M0.5, that is fine
```
Verify: `python -c "import typer, rich, httpx, psycopg, pgvector, networkx, scipy, pathspec, tokenizers, pydantic, yaml"` exits 0.

**M0.3 Postgres via Docker; Ollama server environment** (spec 2.2, 3)
`docker-compose.yml`: service `db`, image `pgvector/pgvector:pg16`, env `POSTGRES_USER/PASSWORD/DB=cfl`, port `5432:5432`, volume `cfl_pgdata`, `shm_size: 256mb`, `mem_limit: 2g`, healthcheck `pg_isready -U cfl`, command
`postgres -c shared_buffers=768MB -c work_mem=8MB -c maintenance_work_mem=128MB -c effective_cache_size=1GB -c max_connections=20`.
`.env.example`: `CFL_DSN=postgresql://cfl:cfl@localhost:5432/cfl`, `CFL_OLLAMA_URL=http://localhost:11434`, `CFL_GEN_MODEL=qwen2.5-coder:7b`, `CFL_EMBED_MODEL=nomic-embed-text`, `CFL_EMBED_DIM=768`, `CFL_NUM_CTX=8192`.
`scripts/ollama_env.sh`: prints (and with `--apply` writes a systemd drop-in for) the five variables of spec 2.2.
```bash
docker compose up -d && docker compose ps
# Ollama server env (Linux/systemd). Other OS: set the same variables, restart Ollama.
sudo systemctl edit ollama.service
#   [Service]
#   Environment="OLLAMA_NUM_PARALLEL=1"
#   Environment="OLLAMA_MAX_LOADED_MODELS=2"
#   Environment="OLLAMA_KEEP_ALIVE=-1"
#   Environment="OLLAMA_FLASH_ATTENTION=1"
#   Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama pull qwen2.5-coder:7b && ollama pull nomic-embed-text    # scripts/pull_models.sh wraps this
```
Verify: `docker compose ps` shows `healthy`; `curl -s localhost:11434/api/tags` lists both models.

**M0.4 `cfl/config.py`, `cfl/core/errors.py`** (spec 2.3, 18)
`Settings(pydantic.BaseModel)` fields and defaults: `dsn`, `ollama_url`, `gen_model`, `embed_model`, `embed_dim=768`, `num_ctx=8192`, `temperature=0.1`, `seed=42`, `num_predict_symbol=350`, `num_predict_brief=120`, `num_predict_detailed=1200`, `template_overhead=64`, `gen_disable_thinking=True`, `edge_conf_threshold=0.6`, `max_file_bytes=500_000`, `embed_max_tokens=1024`, `retrieval_top_k=10`, `retrieval_inject_max=6`, `rrf_k=60`, `flow_depth=4`, `flow_max_nodes=40`, `feature_depth=4`, `statement_timeout_ms=5000`, `chars_per_token=3.2`, `token_margin=0.15`, `tokenizer_file=None`, `state_dir=".cfl"`, `migrations_dir="migrations"`, feature flags `graph_expansion=False`, `llm_rerank=False`, `scip_overlay=False`, `dynamic_overlay=False`, `vram_limit_gb=10`. `get_settings()` (cached, env + toml loader). Function `ensure_state_dir()`.
Verify: `python -c "from cfl.config import get_settings as g; print(g().num_ctx)"` prints `8192`.

**M0.5 Preflight, minimal client/db, `cfl doctor`** (spec 2.3, 6 Stage 0, 14)
- `core/client.py` (health subset now; extended in M4): `class OllamaClient` with `version()`, `tags()`, `ps()`, `show(model)`, constructor accepts `transport=` (for FakeOllama) and uses one `httpx.Client` with `timeout=httpx.Timeout(600, connect=10)`.
- `core/db.py` (subset): `connect(dsn, *, autocommit=True, statement_timeout_ms=None)` (applies `client_encoding=UTF8`, calls `pgvector.psycopg.register_vector` only if the `vector` extension exists).
- `core/preflight.py`:
  - `check_ollama(client)`; `check_models_pulled(client, settings)`.
  - `warm_and_benchmark(client, settings) -> Bench`: send `/api/generate` with `num_ctx=NUM_CTX`, `keep_alive=-1`, a padded prompt of about `NUM_CTX - 512` tokens, `num_predict=64`, `stream=false`; compute `prefill_tps = prompt_eval_count / prompt_eval_duration`, `gen_tps = eval_count / eval_duration`; flag truncation if `prompt_eval_count` is far below what was sent; also call `/api/embed` once to load the embedder and record the vector length. Save `.cfl/bench.json`.
  - `check_residency(client)`: read `/api/ps`; for each model compare `size_vram` vs `size`; **raise `PreflightError` if the generator has `size_vram < size`**; warn for the embedder; warn if total VRAM > `vram_limit_gb`; warn if reported context length differs from `NUM_CTX` (when the field exists).
  - `check_db(settings)`: connect, ensure extensions `vector` and `pg_trgm` are available/created, report migration version if `meta` exists.
  - `check_ram()` (via `psutil` if installed): warn when available RAM < 4 GB.
  - `run_doctor(settings) -> int` prints a rich table, returns exit code.
- `cli.py`: `app = typer.Typer(no_args_is_help=True)`, shared error handler, command `doctor`. Register stubs for every command in spec Section 14 that print "not implemented yet" (so `--help` shows the full surface).
Verify: `cfl doctor` prints tokens/s, VRAM, extension status and exits 0; stop the Ollama process or unload to CPU to confirm exit 1.

**Milestone 0 done when:** spec Section 16 row 0 holds (GPU residency, tokens/s, VRAM, DB extensions verified).

---

### MILESTONE 1: DB layer, migrations, hashing, `FakeOllama`

**M1.1 `cfl/core/hashing.py`** (spec 5 invalidation)
- `sha256_hex(data: str | bytes) -> str`; `file_sha256(path) -> str` (chunked).
- `normalize_code(s)`: `\r\n`->`\n`, rstrip each line, strip outer blank lines.
- `code_hash(raw_code)` = sha256 of `normalize_code`.
- `join_hash(*parts)`: parts joined with `\x1f` (unambiguous), then sha256.
- `ctx_hash(code_hash, callee_pairs, prompt_version, gen_model_tag)`: `callee_pairs` = list of `(callee_id, summary_short)`; hash each `summary_short`, sort by `callee_id`, then `join_hash`.
- `embed_key(embed_text, embed_model_tag)`; `member_hash(symbol_ids)` (sorted); `aggregate_hash(parts)` (order-preserving).
- `symbol_id(path, qualname, start_line, ambiguous: bool)` = `f"{path}::{qualname}"` plus `f"@{start_line}"` when ambiguous. Ambiguity = duplicate qualname within the same file (all duplicates get the suffix).
Test `tests/test_hashing.py`: whitespace-only change keeps `code_hash`; callee `summary_short` change changes `ctx_hash`; unchanged callee text does not; order of callee pairs irrelevant.
Verify: `pytest tests/test_hashing.py -q`.

**M1.2 Migrations** (spec 5)
- `migrations/0001_core.sql`: everything in spec Section 5 except `embeddings`, with D1, D2, D3, D14 applied: `edges.id BIGSERIAL PRIMARY KEY` + the COALESCE unique index; extra columns on `files` and `symbols`; `kind` CHECK includes `'module'`; `module_tree.embed_hash`, `features.embed_hash`; `module_files` table. Extensions `vector`, `pg_trgm` created at the top. Keep `search` as the generated `tsvector` column exactly as in the spec.
- `migrations/0002_embeddings.sql`: `CREATE TABLE embeddings (hash TEXT PRIMARY KEY, vec vector(:EMBED_DIM) NOT NULL);` (no ANN index yet; exact scan first, spec 11.3).
Verify (after M1.3): migrations apply on an empty DB and a second run is a no-op.

**M1.3 `cfl/core/db.py` (full)** (spec 4, 5, 17.5)
Implement the connection helpers plus the complete repository layer. All SQL for the whole project lives here; later steps add functions to this file, never SQL elsewhere.
- Migrations: `run_migrations(conn, embed_dim, migrations_dir)`: applies files in numeric order inside one transaction each, replaces `:EMBED_DIM` via regex `(?<!:):EMBED_DIM\b`, records `meta.schema_version`; guard: if `meta.embed_dim` exists and differs from config, raise with the message to run `cfl db reset-embeddings --dim N` (D9).
- Meta: `get_meta`, `set_meta`, `bump_epoch`, `index_version()`.
- Files: `upsert_file(...) -> bool` (True only if sha changed or new), `delete_missing_files(present_paths) -> int`, `mark_parsed(path, status, error, imports, exports)`, `files_needing_parse()`, `list_files()`.
- Symbols: `sync_symbols(conn, file_path, parsed_rows) -> SyncResult` implementing the invalidation rules: same id + same `code_hash` keeps the row (updates lines/structure only); same id + new code resets `status='pending'`, keeps the old summary until replaced; unmatched new symbols look for a removed symbol with the same `code_hash` (rename/move) and inherit its summary fields; missing symbols are deleted (cascade).
- Graph: `replace_edges(conn, source, rows)` (delete-then-insert for that `source` in one transaction), `fetch_edges(min_conf)`, `set_graph_metrics(rows)` (pagerank, scc_id, layer).
- Traversal (SQL recursive CTEs, `UNION` not `UNION ALL`, depth-capped, so complexity stays polynomial): `callers_rows(id, depth, min_conf)`, `callees_rows(id, depth, min_conf)`, `reach_rows(seed, direction, depth, min_conf)` returning `(id, depth, parent_id)`, `top_pagerank(n)`, `dead_candidates(min_conf)`, `lookup_symbols(query, limit)` (exact id, exact qualname, exact name, suffix, then `pg_trgm` similarity), `symbols_in_file(path)`, `get_symbol(id)`, `get_symbols(ids)`.
- Summaries: `save_symbol_summary(...)` (single transaction; sets `summary_long = NULL`, `status='done'`, `error=NULL`), `save_trivial_summary(...)`, `quarantine_symbol(id, raw_output)`, `set_summary_long(id, text)`, `status_counts()`, `ctx_inputs()` (for stale detection).
- Search/vectors: `set_search_text(rows)`, `lexical_search(terms, limit)` (`ts_rank_cd` on `search`), `trigram_search(q, limit)`, `existing_embedding_hashes(hashes)`, `upsert_embeddings(rows)`, `dense_search(vec, limit, source)`, `set_view_status(view, status, config)`, `get_view_status()`.
- Modules/features: `load_module_tree()`, `replace_module_subtree(...)`, `save_module_summary(...)`, `save_features(...)`, `feature_members(...)`.
- Cache/eval: `get_cached_answer(hash, index_version)`, `put_cached_answer(...)`, `save_eval_metrics(...)`.
Every function takes `conn` first, returns plain dataclasses or dicts, and uses parameterized queries only.
Verify: `pytest tests/test_db.py -q -m db` (migrations idempotent, `sync_symbols` cases above, recursive CTE cycle safety on a 3-node cycle).

**M1.4 `tests/fakes.py` (`FakeOllama`) and `tests/conftest.py`** (spec 17.12)
- `FakeOllama` builds an `httpx.MockTransport`. Endpoints: `/api/version`, `/api/tags`, `/api/ps` (configurable `size`/`size_vram`), `/api/generate`, `/api/chat` (both streaming NDJSON and non-streaming), `/api/embed`.
- Behaviors: `responder(request_json) -> str` callback (default returns schema-valid symbol-summary JSON whose `one_liner` embeds a hash of the code, so editing a body changes it; flag `stable_summary=True` returns identical text regardless of code); `fail_next(n, status=503)`; returns realistic `prompt_eval_count`/`eval_count`/durations; records `call_log`; tracks `max_in_flight` (used to prove the lock); deterministic embeddings as hashed bag-of-words of dimension `EMBED_DIM` (similar text gives similar vectors).
- `conftest.py`: fixture `pg_conn` creates a throwaway database (`CREATE DATABASE cfl_test_<uuid>` using autocommit on the admin DSN), runs migrations, drops it afterwards; fixtures `fake_ollama`, `client`, `settings` (small `num_ctx` overrides allowed in tests only), `fixture_repo_path`.
Verify: `pytest -q` collects; `tests/test_db.py` passes.

**M1.5 `cfl/pipeline/scan.py` (discovery + file registry)** (spec 7.1)
- `discover_files(root, settings) -> Iterator[FileInfo]`: walk with `os.scandir`; honor root and nested `.gitignore` via `pathspec.PathSpec.from_lines("gitwildmatch", ...)`; default excludes `.git node_modules venv .venv dist build __pycache__ .tox .mypy_cache`; skip binaries (NUL byte in first 8 KB), minified/generated files (line length > 1000 or header markers `generated`, `DO NOT EDIT`), lockfiles (`*.lock package-lock.json poetry.lock`), files over `max_file_bytes`, secret-like files (`.env*`, `*.pem`, `*.key`, `id_rsa*`). Language by extension: `.py` -> `python`; config set (`Dockerfile`, `docker-compose*.yml`, `.github/workflows/*.yml`, `pyproject.toml`, `setup.cfg`, `requirements*.txt`, `Makefile`) -> `config`. Paths stored POSIX-style relative to the repo root.
- `register_files(conn, files) -> RegisterResult(changed, unchanged, removed)` using `upsert_file` and `delete_missing_files`; stores `meta.repo_root`.
Test: second run over an unchanged tree reports zero changed files and zero DB writes (this is the "no-op rebuild" check from milestone 1).
Verify: `pytest tests/test_scan.py -q -m db`.

**Milestone 1 done when:** migrations apply cleanly, repo-level no-op rebuild does zero work, `FakeOllama` ready.

---

### MILESTONE 2: Python adapter and layered resolver

**M2.1 `cfl/parser/base.py`** (spec 4, 7.1)
Dataclasses: `ImportEntry(module, name, alias, level, line, is_from)`; `CallSite(expr, line, kind, control_ctx)` (kind: `call`|`decorator`, the resolver upgrades `call` to `constructor`); `ParsedSymbol(kind, qualname, name, parent_qualname, signature, decorators, docstring, start_line, end_line, raw_code, is_async, bases, init_attrs, param_types, local_types, call_sites)`; `ParsedFile(path, language, imports, exports, symbols, parse_error)`.
`class LanguageAdapter(Protocol)`: `language: str`, `extensions: tuple[str, ...]`, `parse(path, source) -> ParsedFile`, `statement_spans(raw_code) -> list[tuple[int,int]]` (top-level statement line spans of a function body, for AST-aware splitting in M4), `skeleton(parsed) -> str` (signatures only; used for L0). `get_adapter(language)` registry.

**M2.2 `cfl/parser/python_adapter.py`** (spec 7.1)
- Read bytes, detect encoding with `tokenize.detect_encoding`, fallback UTF-8 with `errors="replace"`.
- `ast.parse`; on `SyntaxError | ValueError | RecursionError | MemoryError` return `ParsedFile(parse_error=str(e), symbols=[])`, never raise.
- Walk modules/classes/functions. Kinds: top-level functions `function`; functions in classes `method`; classes `class`; functions inside functions `nested`. `qualname` is dotted without `<locals>`. `start_line = min(node.lineno, first decorator lineno)`, `end_line = node.end_lineno`, `raw_code` = exact source lines. `signature` from `ast.unparse` of args and returns, prefixed `async` when relevant. Docstring via `ast.get_docstring(clean=True)`. `bases` = unparsed base expressions. `init_attrs` = `self.x` targets assigned in `__init__`, with a type guess when assigned from `Foo(...)` or an annotated param. `param_types` from annotations (unwrap `Optional[X]`, `X | None`). `local_types` from `x = Foo(...)`.
- Module pseudo-symbol (D3): collect module-level statements that are not `def`/`class`/imports; emit `path::<module>` when they contain at least one call; `raw_code` = those statements.
- Call-site extraction: a visitor over each symbol body that does **not** descend into nested `def`/`class` (they own their calls) but does descend into lambdas and comprehensions. Maintain a control stack with `if`, `for`, `while`, `try`, `with`, `comp`; `control_ctx = ">".join(stack)`. `expr = ast.unparse(node.func)` capped at 200 chars. Decorator expressions become `kind="decorator"` call sites on the decorated symbol. Class-body calls belong to the class symbol.
- `exports` from `__all__` and, for `__init__.py`, imported public names.
- `imports`: absolute and relative (`level`), aliases.
Test `tests/test_python_adapter.py` with inline snippets: async def, decorators, nested functions, class with `__init__`, comprehension control ctx, syntax-error file returns `parse_error`, `if __name__ == "__main__":` produces a module symbol.

**M2.3 Fixture repo `tests/fixtures/resolution_repo/`** (spec 16 M2)
Small package with: module-level function calls; `from a import b as c`; `import pkg.mod as m`; relative imports; class with `self.method()` and inheritance (`self` call resolved through a base class); constructor call `Foo()`; typed call (`x = Foo(); x.bar()`, annotated param, `self.repo = Repo()` then `self.repo.save()`); two classes both defining `save` (ambiguous); a uniquely named function called via an unknown object (`name-unique`); self-recursion; mutual recursion (`is_even`/`is_odd`); an external call (`os.path.join`, `len`); a `common-method` collision (`d.get(...)`); an `if __name__` block. Include `expected_edges.json` (caller, callee expr, callee id or null, resolution) and `expected_sccs.json`.

**M2.4 `cfl/parser/resolver.py`** (spec 7.2, D6, D7)
- Build indexes once: `by_id`, `by_file_qualname`, `by_name` (simple name to ids, excluding blocklisted/dunder for name-unique), class index with bases (bases resolved through the file's import map), `module_index` (dotted module name to file, including `__init__`, tolerating a `src/` prefix), per-file import maps (alias to `(module_file, name)`), per-class `init_attr_types`.
- `resolve_call(caller, call_site) -> list[EdgeRow]` applies layers in this exact order and stops at the first hit (except `ambiguous`, which keeps all candidates):
  1. `import` 0.95: head resolves through the file's import map (module alias, `from` import, class imported, relative imports resolved from the file's package).
  2. `local` 0.90: nested symbol in the caller's chain, then module-level name in the same file; a class name becomes a constructor edge.
  3. `self` 0.90: `self.x()` / `cls.x()` to the enclosing class, walking bases through the index.
  4. `ctor` 0.90: `Foo()` creates two edges: to the class symbol and to `Foo.__init__` if present; `kind='constructor'`.
  5. `typed` 0.80: head is a local var, annotated param, or `self.attr` with a known type.
  6. `name-unique` 0.75: last attribute segment matches exactly one project symbol and is not blocklisted.
  7. several candidates at any layer: store all as `ambiguous` 0.40.
  8. nothing: `external`, `callee_id NULL`, confidence 0.0.
- `COMMON_METHOD_BLOCKLIST` constant (get, set, items, keys, values, append, extend, pop, update, add, remove, join, split, strip, format, read, write, close, open, copy, sort, count, index...).
- Entrypoint detection `detect_entrypoints(symbols)`: `main`, module pseudo-symbols with `__main__` guard, decorators matching route/CLI patterns (`*.route`, `*.get|post|put|delete|patch`, `*.command`, `*.task`, `*.callback`), `__main__.py` contents, names in `__all__`/`__init__` exports. Writes `is_entrypoint`, `entry_kind`.
- `run_resolution(conn, settings)`: loads symbols, resolves all call sites, calls `replace_edges(conn, 'ast', rows)` in one transaction, writes entrypoint flags; skip entirely when `resolve_fingerprint` (hash of all file sha256 + resolver version + threshold) is unchanged.
Test `tests/test_resolver.py`: compare against `expected_edges.json`; assert ambiguous candidates are all present; external edges stored with NULL callee; blocklisted name not linked.

**M2.5 `cfl/parser/graph.py`** (spec 7.3, D4, D8)
- `build_call_graph(conn, threshold) -> nx.DiGraph` (edges with `confidence >= threshold`, weight = max confidence, external edges excluded).
- `build_order_graph(call_graph, symbols)`: call graph plus structural edges `class -> each method` and `module-symbol` nodes.
- `condense(order_graph)`: `nx.strongly_connected_components` then `nx.condensation` (never `simple_cycles`); layer 0 = SCCs with no outgoing dependencies, `layer(n) = 1 + max(layer(dep))`; returns `scc_id` and `layer` per symbol and `scc_members`.
- `compute_pagerank(call_graph)` with `alpha=0.85` (edges caller to callee, so widely called symbols rank high); `compute_communities(call_graph)` with `nx.community.louvain_communities(undirected, weight="weight", seed=42)`.
- `processing_order(conn, *, priority=None) -> list[WorkItem]`: layers ascending; within a layer files sorted by max PageRank (D8); `priority="entrypoints-first"` places everything reachable from entrypoints (with its dependencies) before the rest. Each `WorkItem` is one SCC (singleton or multi-member).
- `run_graph_stage(conn, settings)`: persist `pagerank`, `scc_id`, `layer`; `set_view_status('graph', 'fresh', ...)`.
Test `tests/test_graph.py`: fixture SCCs equal `expected_sccs.json`; mutual recursion forms one SCC; layers respect callee-before-caller; a class sorts after its methods.

**M2.6 Stage 1 and 2 runners in `cfl/pipeline/scan.py`**
- `run_stage1(conn, settings)`: for each file whose sha changed run the adapter, `mark_parsed`, `sync_symbols` (computes `code_hash`, `token_est`, `symbol_id` with ambiguity rule, stores `call_sites` and `extra`); unchanged files are skipped without parsing; parse failures are recorded and counted.
- `run_stage2(conn, settings)`: `run_resolution` then `run_graph_stage`.
- Add CLI hidden command `cfl scan [REPO]` that runs Stages 1-2 (the first part of `cfl build`).
Test: running Stage 1+2 twice on the fixture repo performs zero parses and zero edge writes the second time; touching one file reparses only that file.

**M2.7 Verify milestone 2:** `pytest tests/test_python_adapter.py tests/test_resolver.py tests/test_graph.py -q -m "db or not db"` all pass; `cfl scan tests/fixtures/resolution_repo` then `psql` shows ambiguous rows.

---

### MILESTONE 3: Graph commands (no LLM)

**M3.1 `cfl/engines/graph_queries.py`** (spec 11.1, 16 M3; engines never write SQL)
- `resolve_symbol(conn, query) -> Symbol` using `db.lookup_symbols`; ranking: exact id, `path::name`, exact qualname, exact name, qualname suffix, trigram top 5; if several remain raise `AmbiguousSymbol(candidates)` (CLI shows a numbered choice on a TTY, otherwise prints candidates and exits 2).
- `callers(sym, depth=1, min_conf)`, `callees(sym, depth=1, min_conf)`: rows carry `file:line`, symbol, edge `resolution`, `confidence`, `source`.
- `path(a, b, max_depth=8, min_conf)`: call `db.reach_rows` from `a`, pick the shallowest occurrence of `b`, backtrack through parent ids in Python; returns ordered hops with call-site lines.
- `impact(sym, depth, min_conf)`: reverse transitive closure, grouped by depth and file, with counts.
- `hubs(n=20)`: PageRank top-N with fan-in/fan-out.
- `dead(min_conf, include_tests=False)`: symbols with no inbound edge above threshold, excluding entrypoints, dunder methods, symbols in exports, methods that override a base-class method, test files, and `__init__` whose class has inbound ctor edges; output labelled "candidates".
- `where(query)`: fuzzy locations as `path:start-end`.
Rule: every function returns data; rendering happens in the CLI.

**M3.2 CLI** (spec 14)
Commands `callers`, `callees`, `path`, `impact`, `hubs`, `dead`, `where`, each with `--depth`, `--min-conf`, `--json`; rich tables show confidence and edge source; ambiguous/low-confidence edges are visually marked. Add `cfl db reset-embeddings --dim N` (D9) and `cfl db migrate`.

**M3.3 Tests `tests/test_graph_queries.py`**
Exactness against the fixture: the caller set for a known function equals the expected set from `expected_edges.json`; `path` returns the expected hops on an acyclic and a cyclic graph; `impact` terminates on cycles; zero LLM calls (assert `FakeOllama.call_log` is empty).
Verify: `pytest tests/test_graph_queries.py -q`; `cfl callers is_even`.
**Milestone 3 done when:** all commands run from SQL, exactness tests pass, zero LLM calls.

---

### MILESTONE 4: Budget module and Ollama client

**M4.1 `cfl/core/budget.py`** (spec 8, 17.3)
- `TokenCounter`: if `settings.tokenizer_file` exists load it with `tokenizers.Tokenizer.from_file`; otherwise `ceil(len(text) / ratio * (1 + margin))` with ratio from `.cfl/calibration.json` (default 3.2) and margin 0.15. `calibrate(est_chars, actual_tokens)` updates an EMA ratio only when `actual >= 0.6 * estimate` (D11); logs drift above 15%.
- `PromptPart(name, text, priority, truncatable, min_tokens)`. `assemble(fixed_prefix, parts, *, num_ctx, num_predict, overhead) -> Assembled`: budget = `num_ctx - overhead - num_predict - tokens(fixed_prefix)`; keep priorities in the spec order (1 target code, 2 callee one-liners, 3 caller one-liners, 4 docstring/signature); drop/truncate the lowest priority first; code truncation keeps head and tail with a `...[N lines omitted]` marker.
- `assert_fits(prompt_tokens, num_predict, num_ctx)` raises `BudgetExceeded`. Called inside the client before every send; there is no bypass flag.
- `split_oversized(symbol, adapter, budget) -> list[Chunk]` using `adapter.statement_spans`; `combine_prompt` for the partials-plus-signature step.
- `batch_by_budget(items, token_fn, budget)` for roll-ups; `cap_tool_output(text, cap_tokens)` keeps head and tail plus a pointer line; `output_reserve(task)`.
Test `tests/test_budget.py`: lowest priority dropped first; oversize function splits at statement boundaries; over-budget raises; ratio calibration ignores suspected cache hits.

**M4.2 `cfl/core/client.py` (full)** (spec 2.3, 9, 17.1-17.3)
- Module-level `GENERATION_LOCK = threading.Lock()` wrapped around every `generate`/`chat` call (stream consumption included).
- `generate(prompt, system, *, fmt=None, num_predict, task, symbol_id=None, stream=False) -> GenResult | Iterator[str]` and `chat(messages, ...)`; native endpoints only; options always include `num_ctx=settings.num_ctx` (the single constant), `num_predict`, `temperature`, `seed`; `keep_alive=-1`; `format` accepts a JSON schema; `think=false` when configured (D13).
- `embed(texts) -> list[list[float]]`: batches of 32, `truncate=false`, raises if returned dim differs from `settings.embed_dim`.
- Retry: exponential backoff with jitter (base 2 s) on connect errors, 5xx and out-of-memory payloads, max 3 attempts, then `LLMTransportError`; 4xx raise immediately.
- After each call: compute estimated vs `prompt_eval_count`; log through `trace_log`; raise `BudgetExceeded` if `prompt_eval_count + num_predict > num_ctx`; call `counter.calibrate`.
- `GenResult(text, prompt_tokens, output_tokens, latency_s, prompt_eval_duration, eval_duration, load_duration, done_reason)`.

**M4.3 `cfl/core/trace_log.py`** and tokenizer download
- JSONL append-only writer to `.cfl/logs/calls-YYYYMMDD.jsonl`: `ts, task, symbol_id, est_prompt_tokens, actual_prompt_tokens, output_tokens, latency_s, validation`.
- Optional exact tokenizer (verify the repo name for your chosen model):
```bash
mkdir -p .cfl/tokenizers/gen
curl -L -o .cfl/tokenizers/gen/tokenizer.json \
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/tokenizer.json
# then set CFL_TOKENIZER_FILE=.cfl/tokenizers/gen/tokenizer.json
```
Tests `tests/test_client.py`: over-budget prompt raises before any HTTP call (assert `FakeOllama.call_log` empty); `prompt_eval_count` logged; `fail_next(2, 503)` succeeds on the third attempt, `fail_next(3, 503)` raises; two threads calling `generate` never exceed `max_in_flight == 1`; `num_ctx` equal in every recorded request.
**Milestone 4 done when:** spec row 4 holds.

---

### MILESTONE 5: Eval set v0 and metrics harness

**M5.1 `eval/sample_repo/`** (spec 15)
A realistic ~20-file Python project ("taskboard"): `auth/{tokens,passwords}.py` (`validate_token`, `issue_token`, `TokenError`), `db/{connection,repository}.py` (`Repository.save/get`, a second unrelated `save`), `services/{tasks,notifications}.py`, `utils/{retry,config}.py` (`with_retries` decorator), `api/routes.py` (decorated handlers), `pipeline.py` (`run_pipeline -> transform -> write_to_db`), `cli.py` (`main`), `trees.py` (recursion), mutual recursion pair, inheritance, import aliases, two `process` functions, a `tests/` folder. Commit as static fixture data.

**M5.2 `eval/questions.yaml`, `eval/rubric.md`** (40-60 items)
Schema per item: `id`, `type` (`symbol|module|behavioral|traversal|structural|reasoning`), `question`, `expected` with `symbols` (ids, spans derived from DB at run time so line shifts do not break items), `ordered_symbols` (traversal), `callers` (structural exact set), `keywords` (reasoning). Distribution across the six types as in spec Section 15. A human must verify every expected value; the agent drafts, a person signs off (`verified: true`). `rubric.md`: 1-5 scale and a CSV template `eval/human_scores.csv`.

**M5.3 `cfl/pipeline/indexer.py` (lexical half) and retrieval function**
- `build_search_text(symbol) -> str`: qualname, name, signature, decorators, docstring, `summary_short` if present, identifier-split tokens (snake_case, camelCase, digits) appended; lowercased.
- `build_lexical(conn)`: `set_search_text` in batches; `set_view_status('lexical','fresh')`.
- `retrieve_lexical(conn, query, limit)` = FTS on identifier-split query terms fused with trigram on qualname by simple rank merge. Works before any summaries exist, which is what lets retrieval metrics run before LLM work.

**M5.4 `cfl/eval/metrics.py`, `cfl/eval/runner.py`, CLI `eval`** (spec 15, 18)
- Metrics: `answer_recall_at_5(expected_spans, returned_spans)` (range overlap on same path, first five distinct spans); `citation_validity(citations, retrieved)`; `structural_exactness(expected_set, got_set)` (set equality); `tokens_per_answer` (from the trace log).
- `run_eval(conn, config, modes)`: structural questions through `graph_queries`; retrieval questions through retrieval only (no LLM); full answers only when the LLM pipeline exists (later milestones add modes). Persist to `eval_runs(run_id, question_id, metrics, config)`; print per-type summary; `cfl eval --config eval/configs/x.toml`, `--compare RUN_A RUN_B`.
Test `tests/test_eval_metrics.py` on hand-made spans.
Verify: `cfl scan eval/sample_repo && cfl eval --retrieval-only` prints structural exactness and AnswerRecall@5 for lexical retrieval.
**Milestone 5 done when:** structural and retrieval metrics run without any LLM.

---

### MILESTONE 6: Model and embedding bake-off (decision step)

**M6.1 `scripts/bakeoff.py`** (spec 15)
Inputs: candidate list (`--gen qwen2.5-coder:7b --gen <newer ~9B> --gen <general ~8B>`; optional MoE with expert offload is experimental, do not plan around it) and (`--embed nomic-embed-text --embed qwen3-embedding:0.6b`). Verify every tag, size and licence on the Ollama library page before pulling.
```bash
ollama pull <candidate-tag>          # one at a time; unload previous with `ollama stop <tag>`
cfl doctor                           # per candidate: tokens/s, /api/ps VRAM at NUM_CTX
```
For each generator: run ~40 sampled symbols through the symbol-summary prompt (needs M7.1 prompts; if not ready, use a temporary copy of the prompt in the script), record JSON-validity rate, tokens/s, peak VRAM, rubric score (human, 1-5), and projected ingest hours via `cfl build --estimate` (after M7). For each embedder: run the ~30 retrieval questions with dense-only retrieval and record AnswerRecall@5; record the embedding dimension from `/api/embed`.

**M6.2 Apply the decision**
Write `docs/decisions/0001-model-choice.md` with the numbers and the chosen `gen_model`, `embed_model`, `NUM_CTX`. Update `.env`/`cfl.toml`. If the embedding dim changed: `cfl db reset-embeddings --dim <N>` (D9). Set `meta.gen_model_tag/embed_model_tag/num_ctx`.

**M6.3 Check:** `cfl doctor` passes with the final models; total VRAM below 10 GB at `NUM_CTX`.
**Milestone 6 done when:** decision recorded with numbers.

---

### MILESTONE 7: Symbol pass (Stage 3)

**M7.1 `cfl/prompts/schemas.py`, `cfl/prompts/prompts.py`** (spec 13)
- `schemas.py`: Pydantic `SymbolSummary(one_liner, purpose, inputs, returns, side_effects: list[str], raises: list[str], notable_logic)`; word limits are enforced by deterministic trimming in code (one_liner <= 25 words, notable_logic <= 60), not by failing. `SYMBOL_SUMMARY_JSON_SCHEMA = SymbolSummary.model_json_schema()` passed as Ollama `format`. Also schemas for module/feature outputs (added in M10).
- `prompts.py`: `PROMPT_VERSION = "1"`, `SYSTEM_INGEST` (the spec's exact system message), `build_symbol_prompt(symbol, callees) -> (system, user_parts)` with layout: stable prefix (system + schema + fixed instructions) first, then `[FILE]`, `[SIGNATURE]`, `[CALLEES: id: one_liner ...]`, `[CODE]`, then the one-line reminder "Return only the JSON object". Other prompt builders (`PROMPT_EXPLAIN_FUNCTION`, `PROMPT_FLOW_STEP`, `PROMPT_FLOW_STITCH`, `PROMPT_MODULE_LEAF`, `PROMPT_MODULE_PARENT`, `PROMPT_FEATURE_CARD`, `PROMPT_ANSWER`) are added as stubs now and filled in their milestones.

**M7.2 Trivial detection and templated summaries** (spec 9)
In `symbol_pass.py`: `is_trivial(symbol, min_lines, is_test, skip_tests)`: getters/setters (single `return self.x` / single assignment to `self.x`), one-line wrappers (single `return f(...)` with args forwarded), `__repr__`/`__str__`, bodies of only `pass` or `raise NotImplementedError`, module pseudo-symbols, plus `--skip-tests` and `--min-lines` rules. `templated_one_liner(symbol)` builds a sentence from the signature (e.g. "Returns `x` of the instance"). Persist via `save_trivial_summary` with `status='skipped_trivial'`.

**M7.3 `cfl/pipeline/symbol_pass.py` worker loop** (spec 9, 17.6-17.7)
`run_symbol_pass(conn, client, settings, *, resume, priority, skip_tests, min_lines, retry_failed, progress)`:
1. Compare `meta.prompt_version/gen_model_tag/num_ctx` to current values; update meta (invalidation happens through `ctx_hash`, never by silent reuse).
2. `items = graph.processing_order(conn, priority=priority)`.
3. For each work item (one SCC): compute `ctx_hash` from current `code_hash` and callee `summary_short` values (callees are already summarized because of bottom-up order); if the stored `ctx_hash` equals it and status is `done`/`skipped_trivial`, **skip with zero LLM calls**. Otherwise:
   - trivial: templated summary, no LLM.
   - singleton: assemble prompt with `budget.assemble`, send once.
   - multi-member SCC: one combined prompt with all members' code if it fits; otherwise one call per member with the others' signatures only. Output one summary per member.
   - oversized: `split_oversized` into chunks, summarize chunks as partials, then one combining call with the signature.
   - class: summarize after its methods from docstring, `init_attrs` and method one-liners.
4. Call `client.generate(..., fmt=SYMBOL_SUMMARY_JSON_SCHEMA, num_predict=settings.num_predict_symbol)`; validate with Pydantic; on invalid output do **one repair retry** (previous output plus error plus "Return only valid JSON"); still invalid then `quarantine_symbol` (`status='failed'`, raw output in `error`). Transport failure after 3 attempts: quarantine and continue; never block the run.
5. `save_symbol_summary` in **one transaction per symbol**; `attempts` incremented on every LLM request.
6. Progress: rich bar (done/total, tokens/s from `GenResult`, rolling ETA); store `meta.last_rate`.
7. `KeyboardInterrupt` finishes the current save then exits cleanly; `bump_epoch` in `finally`.

**M7.4 `cfl/pipeline/build.py` and `cfl build`** (spec 6, 9, 14)
- `run_build(repo, *, estimate, resume, skip_tests, min_lines, priority, budget_hours=3)`: Stage 0 `preflight` (abort unless fully on GPU) -> Stage 1 -> Stage 2 -> lexical index (`build_lexical`) -> `--estimate` exits here -> Stage 3 -> Stage 4/5 (added in M10) -> Stage 6 dense (M8).
- `estimate(conn, bench)`: LLM-bound symbol count x (`avg_prompt_tokens / prefill_tps + avg_output_tokens / gen_tps`), with `avg_prompt_tokens` = min(`token_est` + about 40 per callee + prefix tokens, budget) and `avg_output_tokens` about 150; print hours; when above `budget_hours` print the three suggested flags (`--skip-tests`, `--min-lines N`, `--priority entrypoints-first`).
- `--resume` is the default behavior (idempotent); the flag only affects messaging.
`cfl status [--failed]`: counts of `pending/done/failed/skipped_trivial`, stale count (recompute expected `ctx_hash` using `db.ctx_inputs()`), ETA from `meta.last_rate`, failed list with truncated errors, view statuses.

**M7.5 Tests `tests/test_symbol_pass.py`** (all with `FakeOllama`)
- Callees are processed before callers (assert on `call_log` order).
- Kill and resume: make `FakeOllama` raise `KeyboardInterrupt` after N calls; rerun; total calls equal total non-trivial symbols (no repeats).
- One-function edit: change one body, rescan, rerun; only that symbol and callers whose callee `summary_short` actually changed are re-summarized; with `stable_summary=True` only the edited symbol is re-summarized (ripple stops).
- Bumping `PROMPT_VERSION` re-summarizes everything.
- Invalid JSON twice quarantines the symbol and the run continues; 503s are retried.
- Trivial symbols never call the LLM.
- Mutual recursion produces one combined call.

**M7.6 Verify:** `cfl build eval/sample_repo --estimate`, then real `cfl build eval/sample_repo` against Ollama; `cfl status`; Ctrl-C mid-run then `cfl build eval/sample_repo` resumes.
**Milestone 7 done when:** kill + resume works and a one-function edit re-runs only affected symbols.

---

### MILESTONE 8: Hybrid retrieval, citation validator, `cfl ask`

**M8.1 `cfl/pipeline/indexer.py` (dense half)** (spec 11.2, 11.3, D9)
- `build_dense(conn, client, settings)`: embedded text per callable = signature + `summary_short` + docstring + code head (capped at `embed_max_tokens`); per file (L0) = skeleton from the adapter; per feature/module = their summary. Compute `embed_key(text, embed_model_tag)`; skip keys present in `embeddings` (`existing_embedding_hashes`); batch the rest through `/api/embed`; `upsert_embeddings`; store keys in `symbols.embed_hash`, `files.embed_hash`, `features.embed_hash`, `module_tree.embed_hash`; `set_view_status('dense', ...)`. Each view fails independently: an exception sets that view to `failed`, never aborts the build.
- Exact scan only; no HNSW (add `CREATE INDEX ... USING hnsw` only if vectors exceed about 100k and measurements justify it).

**M8.2 `cfl/engines/ask.py` router and retrieval** (spec 11.1, 11.3)
- `classify(question, conn) -> Route` with no LLM: regex tables for structural (`who calls`, `callers of`, `what depends on`, `what breaks if`, `impact of`, `path from .* to`, `most important`, `hubs`, `dead code`, `unused`), flow (`flow`, `trace`, `what happens when` plus a symbol), explain (a known symbol mentioned by backticks, dotted name, snake/camel token verified via `lookup_symbols`, with "what does ... do"), feature+retrieval (`how does .* work`, `where is .* handled`, `where are`), else retrieval. Unit-test as a table of (question, expected route).
- `retrieve(conn, client, question, settings) -> list[Candidate]`: lexical + dense, fuse with RRF (`score = sum(1/(rrf_k + rank))`), take top `retrieval_top_k`. `graph_expansion` and `llm_rerank` are no-ops unless their flags are true (default off, invariant 8).
- `render_blocks(candidates, budget) -> list[ContextBlock]`: callable -> code plus one-liner; class -> L1 summary; file -> L0 skeleton; feature/module -> summary plus tagged member spans; each block tagged `[path:start-end]`; add blocks in rank order until the token budget is reached (expect 3-6 at 8k).

**M8.3 `cfl/engines/verify.py`** (spec 11.5, 17.4, 17.11)
`verify_answer(answer, blocks, conn) -> VerifiedAnswer`: (1) every `[path:start-end]` citation must exist in `files`/`symbols` ranges and overlap a retrieved block, else strip and flag; (2) backticked identifiers that look like symbols must resolve through `lookup_symbols`, else warn; (3) any ```mermaid block passes `core/mermaid.validate_mermaid` (the validator lands in M9.1; until then this hook is a no-op stub guarded by a TODO test that M9 turns on); (4) if nothing from context is cited return the text "insufficient context" plus the top hits.

**M8.4 Answer pipeline and `cfl ask`** (spec 11.4, 14)
- `answer(question, mode)`: classify; structural questions answered from `graph_queries` with no LLM (`--explain` optionally narrates with the LLM); otherwise retrieve, assemble under budget (`PROMPT_ANSWER`: "Answer using ONLY the context blocks. Cite as [path:start-end]. If the context is insufficient, say what is missing."), stream the answer from `client.chat(stream=True)` with `num_predict` per mode, run `verify_answer`, print flagged items, store in `answer_cache` keyed by `hash(question, mode)` plus `index_version`.
- `--brief`: from `summary_short` or a feature card; zero LLM calls when the top hit is a symbol, otherwise one call with `num_predict=120`. `--detailed`: uses `summary_json`, code and neighbor one-liners with `num_predict=1200`.
- Features and modules do not exist until M10: the feature path degrades to plain retrieval when the tables are empty.
Update `eval/runner.py`: add the full-answer mode; report AnswerRecall@5 from retrieved/cited spans and citation validity.

**M8.5 Tests and verify**
`tests/test_ask.py` (router table, RRF fusion order, budget never exceeded, cache hit skips the LLM) and `tests/test_verify.py` (invalid citation stripped, unknown identifier warned, zero-citation answer replaced by the insufficient-context message).
Verify: `cfl build eval/sample_repo`, `cfl ask "Where are retries handled?"`, `cfl eval`.
**Milestone 8 done when:** AnswerRecall@5 and citation validity are reported.

---

### MILESTONE 9: `cfl explain` and `cfl flow`

**M9.1 `cfl/core/mermaid.py`** (spec 12, 17.4)
`safe_id(symbol_id)` (stable `n1, n2...` ids plus label map), `escape_label`, `render_flowchart(trace_tree)`, `render_sequence(trace_tree)`, `collapse(tree, max_nodes, depth)` (replaces overflow with a `...N more` node), `validate_mermaid(text) -> list[str]` (checks a known header, unique node ids, balanced brackets/quotes, valid arrows, no empty labels). Test `tests/test_mermaid.py`. Turn on the Mermaid hook in `verify.py` (M8.3).

**M9.2 `cfl/engines/explain.py`** (spec 0, 11.4)
`explain(query, detail)`: `resolve_symbol`; brief returns `summary_short` plus `purpose` (no LLM); detailed returns cached `summary_long` or generates it with `PROMPT_EXPLAIN_FUNCTION` (target code plus caller/callee one-liners; output: purpose in 1-2 sentences, chronological logic, assumptions/side effects/returns), stores it with `set_summary_long`; classes explained from docstring, `init_attrs` and method one-liners. Output lists `file:line` and callers/callees. Pass through `verify_answer`.

**M9.3 `cfl/engines/flow.py`** (spec 12, 17.4)
1. `resolve_symbol(entrypoint)`; ask to disambiguate if several.
2. `build_trace(conn, root, depth, min_conf)`: DFS following each symbol's edges **ordered by call-site line** (`ORDER BY line`), depth-limited (default 4, `--depth`), cycle-safe with a visiting set; recursion nodes are leaves marked `↻`; external calls are leaves; each child carries `line`, `control_ctx`, `source`, `confidence`.
3. Render Mermaid in code via `mermaid.py`, collapse beyond depth or about 40 nodes, validate (on failure fall back to a plain indented tree and flag it).
4. Narrative: split the trace into step groups that fit the budget; one `PROMPT_FLOW_STEP` call per group using stored one-liners and call-site snippets (never full bodies); one `PROMPT_FLOW_STITCH` call over the group narratives.
5. `render_flow_md(...)`: narrative, diagram, table (`file:line`, symbol, condition, edge source/confidence). Write to `--out`.

**M9.4 Tests and verify**
`tests/test_flow.py`: children appear in source-line order (not topological); mutual recursion terminates with `↻`; node cap collapses; Mermaid output validates; every flow prompt passes the budget assertion for a large synthetic trace. `tests/test_explain.py`: second detailed call served from cache; editing the symbol and re-summarizing clears `summary_long`.
Verify: `cfl explain with_retries --detailed`, `cfl flow run_pipeline --out flow.md`.
**Milestone 9 done when:** source-order trace, validated Mermaid, works on cycles.

---

### MILESTONE 10: Module tree, features, `cfl docs`

**M10.1 `cfl/pipeline/modules.py`: clustering** (spec 10.1)
- `build_cluster_graph(conn)`: nodes = files; weight = sum of call-edge confidences between files plus 0.5 per import between files; multiply by (1 + `DIR_BOOST`, default 1.0) when both files share a package directory.
- `cluster(graph)`: Louvain (`seed=42`); recurse into clusters above `MAX_MODULE_SYMBOLS=60` or token total above 6x the budget, up to depth 4; indivisible single-file clusters split by top-level class; if Louvain returns one cluster, fall back to directory split.
- Config/build files (`language == 'config'`) go into the module "Build, Deployment and Configuration" through `module_files` (D14).
- `member_hash` for every cluster.

**M10.2 Names, summaries, incremental regeneration** (spec 10.1 items 2-7)
- Name stability: greedy one-to-one matching of new clusters to existing modules by Jaccard similarity of member symbol sets (threshold 0.5); matched modules keep their id and name.
- Leaf module LLM call (`PROMPT_MODULE_LEAF`): member `summary_short`s, signatures of public members, 2-3 top-PageRank bodies truncated, never all code; JSON output `{title, purpose, collaboration, public_surface, data_flow}` under 300 words.
- Parent modules (`PROMPT_MODULE_PARENT`) from child summaries, batched with `batch_by_budget`, merging intermediate summaries.
- `aggregate_hash` = hash of member `summary_short` hashes (leaf) or child aggregate hashes (parent); regenerate only modules whose hash changed plus their ancestors.
- Each module's embed text and `embed_hash` set so Stage 6 picks it up.

**M10.3 Features (Stage 5)** (spec 10.2)
- Seeds = `is_entrypoint` symbols. For each seed: BFS over edges above threshold to `feature_depth`, treating the top 2% PageRank symbols as hub leaves (included but not expanded) so features do not all merge through utilities.
- Merge seeds whose subgraphs overlap with Jaccard >= 0.6; Louvain communities with at least 5 symbols not covered by any feature become seedless features.
- `PROMPT_FEATURE_CARD` per feature from member one-liners; JSON `{name, description, key_steps}`; `feature_members.role` is `entry`, `core` (above median PageRank) or `helper`; same Jaccard matching to keep feature names stable; regenerate on `aggregate_hash` change.
- Wire into `build.py` as Stages 4 and 5, then re-run `build_dense` so module/feature cards are embedded. Update the router: the feature path in `ask.py` now retrieves feature cards.
CLI `cfl feature <query|name>`: prints the card, members with roles, and retrieval hits.

**M10.4 `cfl/engines/docs_export.py`** (spec 0.5, 14; no LLM)
`export_docs(conn, out_dir)`: writes `SUMMARY.md` (index), `ARCHITECTURE.md` (top modules, code-generated Mermaid containment and sibling-dependency diagrams validated by `mermaid.py`, entrypoints, hubs), `modules/<slug>.md` (purpose, collaboration, public surface, data flow, member table with `path:start-end` and one-liners, diagrams), `features/<slug>.md`. Stable slugs from module/feature names; pages written only when their content hash changed (manifest `.cfl-docs.json`), which makes "incremental regeneration touches only affected modules" testable. Missing summaries are flagged in the page rather than invented.

**M10.5 Tests and verify**
`tests/test_modules.py`: stable names across rebuilds; editing one function regenerates only its module and ancestors (count LLM calls); configs land in the build module. `tests/test_docs_export.py`: second export rewrites zero files; all Mermaid blocks validate.
Verify: `cfl build eval/sample_repo && cfl docs --out ./docs/_out`.
**Milestone 10 done when:** incremental regeneration touches only affected modules.

---

### MILESTONE 11: `cfl chat` (+ optional `--deep`)

**M11.1 `cfl/engines/chat.py` session** (spec 11.4)
- `ChatSession`: `turns` list, `direction_seed` (ordered set of symbols/files discussed, capped at 15), `max_turns=4`. Deterministic compaction: older turns' assistant text and tool observations replaced with `[masked: N chars]`, last N turns kept verbatim, plus one seed line. No LLM summarization of history.
- `ChatEngine.turn(question)`: reuses the same `classify`/retrieve/answer path; the seed adds hint terms to retrieval when the question lacks a symbol; history is added at lower priority than context blocks in `budget.assemble`; keeps DB, vectors and the Ollama client warm; streams output; records spans cited into the seed.
- REPL in `cli.py` using `rich.prompt` plus `readline`: `/help /quit /clear /seed /mode brief|detailed /deep on|off`.
- Per-turn tokens stay within budget by construction (assert in `assemble`).

**M11.2 `--deep` bounded loop**
- JSON-schema-constrained step `{"action": "search|get_symbol|callers|callees|read_range|answer", "args": {...}}`; maximum 3 tool calls, then a forced answer step; read-only tools only.
- Tool results are handles plus one-liners first (progressive disclosure); per-tool caps via `cap_tool_output`: search 600, `get_symbol` brief 300 / full 1500, callers/callees 500, `read_range` 1200 tokens. `read_range(path, start, end)` only for paths present in `files`, resolved under `meta.repo_root`, at most 200 lines, path traversal rejected.

**M11.3 Tests and verify**
`tests/test_chat.py`: never more than 3 tool calls; tool output caps enforced; compaction keeps last N turns and masks older observations; every turn's prompt tokens <= `NUM_CTX - num_predict`; `read_range` outside the repo rejected. Add a `chat-deep` mode to `cfl eval` and compare to the pipeline mode.
Verify: `cfl chat` then `cfl chat --deep`.
**Milestone 11 done when:** per-turn tokens within budget and eval no worse than the pipeline.

---

### MILESTONE 12: Optional overlays and eval-gated options

All of this defaults to off and must degrade silently (invariant 8).

**M12.1 SCIP overlay `cfl/parser/overlays/scip.py`** (spec 7.4.1)
```bash
pip install -e ".[scip]"                 # protobuf
npm install -g @sourcegraph/scip-python  # needs Node.js 18+
# generate Python bindings from scip.proto (pin the commit you used in docs/decisions)
#   protoc --python_out=cfl/parser/overlays/ scip.proto   # then commit scip_pb2.py
```
`run_scip_overlay(conn, repo_root, settings)`: check `shutil.which("scip-python")`; run `scip-python index . --project-name <name> --output <tmp>/index.scip` with `subprocess.run(timeout=...)` and a per-file exclusion list; decode occurrences; definition occurrences (role bit 1) map `symbol string -> (file, range)`; reference occurrences inside a caller's line range whose symbol has a definition in the project produce an edge. Upgrade existing `ast` edges to 1.0 (matching caller, callee, line) and insert missing ones with `source='scip'`. Any exception: log, set view `scip` to `failed`, continue. Verify the proto field numbers against the installed `scip-python` output before trusting the decoder.

**M12.2 Dynamic overlay `cfl/parser/overlays/dynamic.py`, CLI `cfl trace -- <cmd>`** (spec 7.4.2, 17.10)
- Requires explicit confirmation (`--yes` to skip) because it executes user code; recommend a throwaway environment.
- Runs the command in a subprocess with `PYTHONPATH` prefixed by `cfl/parser/overlays/_tracer_shim/`; `sitecustomize.py` installs `sys.monitoring` (Python 3.12+, tool id + `PY_START` and `CALL` events) or `sys.setprofile` (earlier), records `(caller file, caller function first line, callee file, callee first line)` pairs restricted to files under `repo_root`, and writes JSONL via `atexit`.
- `ingest_dynamic(conn, jsonl)`: map `(file, co_firstlineno)` to symbols by `start_line`; insert edges `source='dynamic'`, confidence 1.0.

**M12.3 Eval-gated options** (spec 11.3, 15)
Create `eval/configs/{baseline,graph_expansion,rerank,scip,dynamic,embed_alt,ctx16k}.toml` and implement the flag behavior behind `settings.graph_expansion` (add neighbors of top seeds with a tunable graph weight tested on held-out questions), `settings.llm_rerank` (one reranking call per query), SCIP/dynamic edges feeding the resolver threshold. Run `cfl eval --config <each>` and `cfl eval --compare`. Record in `docs/decisions/0002-optional-features.md`; an option stays off unless eval shows a gain.

**M12.4 Tests:** SCIP missing binary degrades to a skip with no exception; dynamic ingest maps pairs to the right symbols using a tiny fixture script; defaults still produce `source='ast'` only.
**Milestone 12 done when:** each option is shown to help on eval, or left off.

---

### MILESTONE 13: Full-scale run on your real repo

**M13.1 Run and measure** (spec 0 limits, 9 throughput)
```bash
cfl doctor
cfl build /path/to/your/repo --estimate      # decide flags from the printed hours
cfl build /path/to/your/repo [--skip-tests --min-lines 5 --priority entrypoints-first]
# in another terminal, throughout the run:
bash scripts/monitor.sh                        # nvidia-smi memory.used every 5 s, `ollama ps`, docker stats for db, process RSS
cfl status
cfl eval
cfl docs --out ./docs/_out
```
`scripts/monitor.sh` appends timestamped GPU memory, Ollama `ps` residency, Postgres container memory and CFL process RSS to `.cfl/monitor.log`.
Use one database per repo (D12): `createdb`/new `CFL_DSN` database name before building a second repo.

**M13.2 Report and README**
`docs/fullscale_report.md`: symbol counts by status, wall time, tokens/s, peak VRAM (must be under 10 GB), peak RAM (must be under 16 GB including Postgres), failed symbols and causes, eval metrics. Finish `README.md`: install, `docker compose up`, Ollama env, `cfl doctor`, command table from spec Section 14, state dir layout, troubleshooting (CPU spill, truncation warnings, migration dim mismatch).
**Milestone 13 done when:** time, VRAM and RAM budgets are met, or the estimate warned beforehand.

---

## 6. Invariant-to-test map (spec Section 17)

| Invariant | Test |
|---|---|
| 1 one in-flight generation | `test_client.py::test_lock_serializes_generation` (`max_in_flight == 1`, includes chat and streaming) |
| 2 single `NUM_CTX` | `test_client.py::test_num_ctx_constant` over all recorded requests from all engines |
| 3 budget assertion, `num_predict` always set | `test_client.py::test_over_budget_never_sent`; grep test that every `generate`/`chat` call site passes `num_predict` |
| 4 no LLM for edges, lines, citations, Mermaid | `test_flow.py`, `test_graph_queries.py` (empty `call_log`), `test_verify.py` |
| 5 SQL only in `core/db.py` | `test_architecture.py`: scans `cfl/` (except `db.py` and `migrations/`) for `SELECT|INSERT|UPDATE|DELETE|CREATE TABLE` string literals and fails if found |
| 6 idempotent, resumable, one transaction per symbol | `test_symbol_pass.py::test_kill_and_resume`, `test_scan.py::test_noop` |
| 7 hashes checked before any LLM call | `test_symbol_pass.py::test_unchanged_zero_calls` |
| 8 optional components degrade, default off | `test_overlays.py`, `test_config.py::test_defaults_off` |
| 9 version bumps invalidate | `test_symbol_pass.py::test_prompt_version_bump` |
| 10 no user code executed except `cfl trace` | `test_architecture.py`: no `exec`, `eval`, `importlib` of user paths, `subprocess` outside `overlays/` |
| 11 verification hooks on every answer | `test_ask.py`, `test_explain.py`, `test_flow.py` assert `verify_answer` was invoked |
| 12 no GPU in tests | whole suite runs on `FakeOllama`; CI runs `pytest -q` on a machine without a GPU |

---

## 7. Command cheat sheet (in order of first use)

```bash
docker compose up -d                       # M0.3
pip install -e ".[dev,sys]"                # M0.2
cfl doctor                                 # M0.5, again after M6
pytest -q -m "not db"                      # fast loop
pytest -q                                  # full loop (needs docker db)
cfl db migrate                             # M3.2
cfl scan eval/sample_repo                  # M2.6
cfl callers|callees|path|impact|hubs|dead|where <name>     # M3
cfl eval --retrieval-only                  # M5
cfl build <repo> --estimate                # M7
cfl build <repo> [--skip-tests --min-lines N --priority entrypoints-first]
cfl status [--failed]
cfl ask "<question>" [--brief|--detailed]  # M8
cfl explain <name> [--brief|--detailed]    # M9
cfl flow <entrypoint> [--depth N] --out flow.md
cfl feature <query|name>                   # M10
cfl docs --out ./docs/_out
cfl chat [--deep]                          # M11
cfl trace -- <command>                     # M12 (executes your code)
cfl eval [--config eval/configs/x.toml]
```
