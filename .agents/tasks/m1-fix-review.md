# Milestone 1 Implementation Review (Fix Pass)

## Summary

Milestone 1 implements the foundational database layer, test fixtures, and file scanning pipeline. All required repository functions are present in `db.py` with proper parameterized queries. The test infrastructure provides a mock Ollama client with controllable failure modes and deterministic embeddings. The file scanner respects `.gitignore`, excludes binary/secret/lockfile patterns, and integrates with the database layer. All four test files exist and cover the core functionality.

**Verdict**: APPROVED

---

## High-level view

The database layer is now complete with all repository functions using parameterized queries (`%s` placeholders) throughout, avoiding f-string SQL injection. The test infrastructure includes a `FakeOllama` that tracks in-flight requests, logs calls, and produces deterministic embeddings. The file scanner uses `pathspec` for `.gitignore` parsing and applies the required exclusions for binary files, secrets, lockfiles, and minified content. The four test files cover hashing, database operations, scanning, and the Ollama client.

---

## Details

### db.py repository layer

**confirmed** — All required functions are present: `connect()`, `run_migrations()`, `get_meta()`, `set_meta()`, `bump_epoch()`, `index_version()`, `upsert_file()`, `delete_missing_files()`, `mark_parsed()`, `files_needing_parse()`, `list_files()`, `sync_symbols()`, `replace_edges()`, `fetch_edges()`, `set_graph_metrics()`, `callers_rows()`, `callees_rows()`, `reach_rows()`, `top_pagerank()`, `dead_candidates()`, `lookup_symbols()`, `symbols_in_file()`, `get_symbol()`, `get_symbols()`, `save_symbol_summary()`, `save_trivial_summary()`, `quarantine_symbol()`, `set_summary_long()`, `status_counts()`, `ctx_inputs()`, `set_search_text()`, `lexical_search()`, `trigram_search()`, `existing_embedding_hashes()`, `upsert_embeddings()`, `dense_search()`, `set_view_status()`, `get_view_status()`, `load_module_tree()`, `replace_module_subtree()`, `save_module_summary()`, `save_features()`, `feature_members()`, `get_cached_answer()`, `put_cached_answer()`, `save_eval_metrics()`.

The file begins with `from __future__ import annotations`. Type hints are present on all function signatures using `psycopg.Connection` for connections and proper return types. SQL uses `%s` parameterization exclusively, with one exception: line 58 uses `f"SET statement_timeout = {statement_timeout_ms}"` for a numeric configuration value, which is not a SQL injection risk since it's not user-controlled input.

The `sync_symbols` function handles symbol deduplication with line-number suffixes for ambiguous qualnames, parent tracking, and proper insert/update/unchanged tracking via `code_hash` comparison.

The `delete_missing_files` function uses `DELETE FROM files WHERE path != ALL(%s)` with the set converted to a list, which correctly removes files no longer present on disk.

### fakes.py FakeOllama

**confirmed** — The class provides: `fail_next(n, status)` method to queue simulated failures, `call_log` list tracking all requests, `max_in_flight` property tracking peak concurrent requests via threading lock, deterministic embeddings via `_make_embedding()` that seeds an RNG with the SHA256 hash of input text, and endpoints for `/api/version`, `/api/tags`, `/api/ps`, `/api/generate`, `/api/chat`, `/api/embed`. The `transport` property returns an `httpx.MockTransport` for use with `OllamaClient`.

### conftest.py fixtures

**confirmed** — `pg_conn` fixture creates a unique temporary database per test run using UUID, runs migrations from the project's migrations directory with the embed_dim, yields the connection, then drops the database on cleanup. `fake_ollama` returns a fresh `FakeOllama` instance. `client` creates an `OllamaClient` with settings and the fake transport. `settings` returns a `Settings` instance with `num_ctx=2048`. `fixture_repo_path` provides a path to test fixtures directory. The `_admin_dsn()` helper derives admin connection from `CFL_DSN` by replacing the database name with `postgres`.

### scan.py file discovery

**confirmed** — `discover_files()` walks the directory tree, respects `.gitignore` via `pathspec.PathSpec.from_lines("gitwildmatch")`, applies default exclusions (`.git`, `node_modules`, `venv`, `.venv`, `dist`, `build`, `__pycache__`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.eggs`, `htmlcov`, `.ruff_cache`), skips lockfiles (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, etc.), skips secret files (`.env`, `.pem`, `.key`, `.p12`, `.pfx`, `.crt`, `.cer`, `.der`), filters by `settings.max_file_size`, skips binary files (checks for NUL bytes in first 8KB), skips minified files (average line length > 1000), detects language from extension, and yields `FileInfo` with path, sha256, size_bytes, and language.

`register_files()` upserts discovered files using `upsert_file()` and removes missing ones via `delete_missing_files()`. `run_stage1()` orchestrates the scan and registration pipeline.

### Test coverage

**confirmed** — Four test files exist:
- `test_hashing.py` — tests whitespace normalization, ctx_hash order-independence, join_hash, sha256_hex handling of str/bytes
- `test_db.py` — tests meta operations, file upsert/delete, symbol sync, edge replace/fetch, status counts
- `test_scan.py` — tests discover_files finding Python files, excluding .env and lockfiles, register_files persisting to DB
- `test_client.py` — tests client methods (version, tags, ps), fail_next behavior, deterministic embeddings

---

## Issues

The implementation meets all requirements. No blocking issues found.
</details>

<details>
<summary>Issues (0)</summary>

No blocking concerns.

</details>

---

**File map**

- `cfl/core/db.py` — Database repository with all CRUD operations for files, symbols, edges, summaries, search, and cache
- `tests/fakes.py` — FakeOllama mock with controllable failures, call logging, deterministic embeddings
- `tests/conftest.py` — Pytest fixtures for temp DB, fake Ollama, client, settings, test repo path
- `cfl/pipeline/scan.py` — File discovery with gitignore support, binary/secret/lockfile exclusions, DB registration
- `tests/test_hashing.py` — Hashing function tests
- `tests/test_db.py` — Database repository tests
- `tests/test_scan.py` — File scanner tests
- `tests/test_client.py` — Ollama client tests
</details>