# CodeFlowLens

Local code intelligence powered by local LLMs and PostgreSQL.

## Quick start

1. `docker compose up -d`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -e ".[dev,sys]"`
4. `cfl doctor`

## Requirements

Python 3.11+, Docker, Ollama (with qwen2.5-coder:7b and nomic-embed-text pulled).

## Scan and graph queries

These commands use PostgreSQL and do not call Ollama:

```sh
cfl db migrate
cfl scan tests/fixtures/resolution_repo
cfl callers is_even
cfl callees create_task --depth 2
cfl path is_even is_odd --json
cfl impact Repository.save --depth 4
cfl hubs --top 10
cfl dead
cfl where save
```

Use `--min-conf 0.3` to include ambiguous edges (confidence 0.4); the default is
`edge_conf_threshold` from configuration (0.6). Traversal commands accept `--depth`,
all graph queries accept `--json`, and `dead --include-tests` includes test files.
Tables mark ambiguous and low-confidence edges with `?`. JSON is written to stdout.
`path` returns a list of call-site hops, an empty list for a symbol queried against
itself, and `null` when no path exists within the requested limits.

Use an exact symbol ID or `path::name` when a name is ambiguous. Interactive terminals
show a numbered choice; noninteractive queries exit 2 with candidates. `where` lists
all matching definitions as `path:start-end`. Dead-code results are candidates, with
entrypoints, exports, dunder methods, and inherited overrides excluded.

`--repo DIR` loads that directory's `cfl.toml`; `CFL_*` environment variables take
precedence. Graph queries apply the configured database statement timeout.

To change the embedding dimension, run `cfl db reset-embeddings --dim N`, then update
`CFL_EMBED_DIM` or `embed_dim` in `cfl.toml` to match. This explicitly clears the dense
view and embedding hashes while preserving parsed symbols, summaries, and graph data.
Migrations reject a dimension mismatch until the reset is performed.
