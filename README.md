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

## Lexical evaluation (milestone 5)

Use a dedicated database for the sample fixture: scanning replaces the database's
indexed repository. With that database migrated, run:

```sh
cfl scan eval/sample_repo
cfl eval --retrieval-only
cfl eval --config eval/configs/lexical.toml --json
cfl eval --structural-only
cfl eval --compare RUN_A RUN_B
```

Scanning now builds the lexical view from file/module paths, names, signatures, decorators,
docstrings, and available short summaries. Retrieval merges PostgreSQL full-text
ranks for split identifiers with trigram symbol-name ranks. No summaries, embeddings,
GPU, or Ollama server are required for these evaluations.

`cfl.engines.retrieval.retrieve_evidence` augments lexical hits for `Trace/Path/Flow
SOURCE to TARGET` queries with every source block on the selected call path, in
order, including call-site locations and edge confidence. It resolves named or
module-qualified endpoints; prose endpoints use lexical action terms and graph
reachability. Ambiguous, unresolved, and unreachable requests have explicit
statuses. Paths longer than the result limit stay complete and set
`requires_batching`; a future generation caller must budget or split these blocks.
Evaluation reports pure lexical `AnswerRecall@5` separately from enriched
`EvidenceRecall@5`. All eight traversal questions now have complete source evidence;
see `docs/validation/path-evidence.md`.

The static taskboard fixture has 20 Python files. Its 48 questions cover symbol,
module, behavioral, traversal, structural, and reasoning evidence. Expected IDs
resolve to current database spans at run time. Runs persist per-question metrics,
evidence spans, suite snapshots, and configuration in `eval_runs`; comparisons show
changes for shared questions and flag differing suites or question selections.

All 48 expectations have been independently reviewed against fixture source by
Codex at the user's request. They carry `verified: true` with agent provenance,
source rationale, and a fixture fingerprint. The CLI and stored runs identify the
review method as `agent`. `--verified-only` selects reviewed expectations; newly
unreviewed questions remain provisional. See `eval/rubric.md` and
`docs/validation/milestone-5-agent-review.md` for the review and remaining output gaps.
The supplied CSV is optional for future human answer/document scoring.
Milestone 5 scores lexical evidence recall and graph exactness; reasoning answer
quality, generated citations, and answer token cost await the later answer pipeline.

Reusable synchronous operations can use the bounded retry decorator:

```python
from cfl.core.retry import with_retries

@with_retries(max_retries=3, retry_on=(TimeoutError, ConnectionError))
def fetch_data():
    return fetch_from_service()
```

`max_retries` counts retries after the initial call, so `3` permits four total
attempts and `0` runs once. The default retry exception is `RuntimeError`; other
exceptions propagate immediately. Exhaustion re-raises the last exception.
