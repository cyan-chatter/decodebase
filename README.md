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

## Token budgets and Ollama calls

Milestone 4 provides the native [Ollama client](https://docs.ollama.com/api/generate)
for subsequent summary and answer stages. Every generation reserves `num_predict`
and template overhead inside the configured `num_ctx`, checks the prompt before
sending, and checks Ollama's reported prompt count afterward. A process-wide lock
serializes generation, including stream consumption. Streaming callers must consume
or close the iterator to release it. Connection failures, HTTP 5xx, and OOM payloads
retry at most three attempts; HTTP 4xx fails immediately.

Prompt assembly removes or truncates lower-priority context first. Oversized functions
can be split at complete AST statements; a single statement that cannot fit raises
`BudgetExceeded`. Tool output caps retain the head, tail, an omission marker, and a
pointer. Embeddings use batches of 32 with server truncation disabled and dimension
validation.

Without an exact tokenizer, counts use the configured character ratio plus a 15%
default margin. Calibration persists under `state_dir/calibration.json` and ignores
suspected prompt-cache hits. Call metadata appends to
`state_dir/logs/calls-YYYYMMDD.jsonl`; prompts and source text are not stored.
`GenResult` reports native duration fields in nanoseconds and wall latency in seconds.

For optional exact counting with the default generator, download its
[Qwen tokenizer](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/tokenizer.json):

```sh
mkdir -p .cfl/tokenizers/gen
curl --fail --location \
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/tokenizer.json \
  --output .cfl/tokenizers/gen/tokenizer.json
export CFL_TOKENIZER_FILE="$PWD/.cfl/tokenizers/gen/tokenizer.json"
```

Choose a matching tokenizer when changing `gen_model`. Milestone 4 tests use a small
local tokenizer and `FakeOllama`, and require no GPU or running Ollama server:

```sh
pytest tests/test_budget.py tests/test_client.py tests/test_trace_log.py
pytest  # Full regression suite also requires the PostgreSQL container.
```
