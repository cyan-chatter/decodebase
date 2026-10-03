# Milestone 1 Implementation Review

## Summary

Milestone 1 implements the foundational layer for CodeFlowLens: database migrations, hashing utilities, and core infrastructure. The migration files correctly define all required tables per the spec (Section 5 plus D1/D2/D14 additions). The hashing module implements all ten required functions with correct order-independence semantics for `ctx_hash`. However, **critical infrastructure is missing**: `db.py` contains only a connection function with no repository methods, the `FakeOllama` stub is absent, and the file scanner has not been implemented. A no-op rebuild cannot actually run because the pipeline layer is empty.

**Verdict**: NEEDS_CHANGES

---

## High-level view

The migrations layer is solid. The `edges` table correctly uses a surrogate BIGSERIAL PK with the UNIQUE constraint per D1, and `symbols.kind` includes 'module' per D2. The hashing functions pass their tests and implement the specified order-independence for callee pairs in `ctx_hash`. The gap is everything above the data layer: `db.py` has no repository methods to insert or query any data, so a pipeline cannot actually store its results. The fakes and conftest files that would enable testing against a mock Ollama are missing entirely. The scan stage that discovers files and respects `.gitignore` is a stub with no implementation.

---

## Details

### Migration completeness

**confirmed** — The migrations include all tables from spec Section 5: `meta`, `files`, `symbols`, `edges`, `embeddings`, `module_tree`, `module_members`, `features`, `feature_members`, `view_status`, `answer_cache`, `eval_runs`. The D1 addition (surrogate PK + unique constraint on edges) is present. The D2 addition (`'module'` in symbols.kind CHECK) is present. The D14 addition (module_files table) is present. The embeddings table uses the `:EMBED_DIM` placeholder correctly for runtime substitution.

### Hashing implementation

**confirmed** — All ten required functions are present:
- `sha256_hex` handles both str and bytes, returns hex string
- `file_sha256` reads in 8KB chunks
- `normalize_code` replaces CRLF, rstrip each line, strips outer blank lines
- `code_hash` normalizes then hashes
- `join_hash` joins with unit separator (0x1f)
- `ctx_hash` sorts callee_pairs by callee_id before hashing, achieving order-independence on callee_pairs per the spec
- `embed_key` concatenates text and model tag with unit separator
- `member_hash` sorts symbol_ids before hashing
- `aggregate_hash` preserves order with unit separator
- `symbol_id` includes `@line` suffix when ambiguous

The test `test_callee_pair_order_independent` verifies the order-independence property.

### Database layer gap

**confirmed** — `db.py` contains only `connect()`. The following required functions are missing:
- `run_migrations(embed_dim)` — applies migrations with dimension substitution, guards against embed_dim mismatch
- `get_file_by_path(path)` / `upsert_file(...)` — file CRUD
- `get_symbol(id)` / `upsert_symbol(...)` / `list_symbols(...)` — symbol CRUD
- `get_edge(...)` / `upsert_edge(...)` — edge CRUD
- `list_files()`, `list_pending_symbols()`, `update_symbol_status()`, `get_symbols_for_file()`

The spec requires "ALL SQL lives in core/db.py; engines call repository functions." Currently no engine could function because there is no data access layer. The `connect()` function does use parameterized queries (no f-string SQL injection), but without repository functions, there is nothing to parameterize.

### scan.py absence

**confirmed** — `cfl/pipeline/scan.py` does not exist. The pipeline directory contains only `__init__.py`. The spec requires scanning to respect `.gitignore` (pathspec) plus defaults (`.git`, `node_modules`, `venv`, `dist`, `build`, `__pycache__`), skip binaries, minified/generated files, lockfiles, files over a size cap, and secret-like files (`.env*`, key files). This functionality cannot be verified because the file does not exist.

### FakeOllama and test fixtures

**confirmed** — `tests/fakes.py` does not exist, so there is no `FakeOllama` to implement `fail_next`, `call_log`, `max_in_flight` tracking, and deterministic embeddings. `tests/conftest.py` does not exist, so there are no pytest fixtures for test isolation. The test directory contains only `test_hashing.py`, which tests the hashing module in isolation but cannot test integration with a mock Ollama client because no fake exists.

### Code quality

**confirmed** — `hashing.py` starts with `from __future__ import annotations`. `db.py` also has this import. No other Python files exist in the milestone scope to check.

---

## Issues

1. **Missing db.py repository functions** — The database module only has `connect()`. All data access (files, symbols, edges) must be implemented before the pipeline can store results. Implement all specified repository functions following the spec's connection rules (one transaction per symbol summary, autocommit per unit of work).
2. **Missing FakeOllama** — The fakes module must provide a mock Ollama client with `fail_next`, `call_log`, `max_in_flight`, and deterministic embeddings for testing the pipeline without a real LLM.
3. **Missing scan.py** — The pipeline scan stage must be implemented to discover files, respect `.gitignore`, and filter binaries/secrets/lockfiles/generated files.
4. **Missing conftest.py** — Test fixtures should be added to support pytest-based testing of the pipeline.

</details>