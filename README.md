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

For exact counting with the milestone 6 selected generator, download its
[Qwen tokenizer](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/tokenizer.json):

```sh
mkdir -p .cfl/tokenizers/qwen3.5-9b
curl --fail --location \
  https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/tokenizer.json \
  --output .cfl/tokenizers/qwen3.5-9b/tokenizer.json
export CFL_TOKENIZER_FILE="$PWD/.cfl/tokenizers/qwen3.5-9b/tokenizer.json"
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

## Cache, memory, and optional KV quantization

Generation metrics now retain cached prompt tokens and native durations.
`cfl cache-benchmark --repetitions 3` measures fresh prompts, repetitions, shared
prefixes, and changed prefixes separately. Doctor reports uncached prefill throughput;
when cache metrics are unavailable or almost no tokens were evaluated, it does not
claim a fresh-input rate. Cached tokens still count toward the context limit.

Code fingerprints preserve Python indentation and string whitespace. Summary keys
include callee identities, resolved model digests, prompt/schema versions, and
generation settings. Index revisions include repository identity and the scan epoch;
content edits advance the epoch, while unchanged scans keep it stable. The parser
version bump reparses old fingerprints on the next scan.

Run `cfl db migrate` for the additive memory migration. `KnowledgeMemory` in
`cfl/core/memory.py` provides validated structured summary reuse and cached embeddings,
including duplicate-text batching. `SessionMemory` persists recent turns, user
constraints, and bounded source references; it rehydrates current source, clears old
answers when the index changes, and budgets source before discussion history.
These APIs support the upcoming ingestion and chat stages; the build/ask/chat CLI
pipelines remain scheduled for their milestones. Answer-cache entries require
explicit verification and unchanged source evidence.

KV quantization defaults to **off**, using `f16`. Enable it explicitly through
`kv_quantization = true` in `cfl.toml` or `CFL_KV_QUANTIZATION=true`. The optional
`kv_quantization_type` / `CFL_KV_QUANTIZATION_TYPE` accepts `q8_0` (default choice
when enabled) or `q4_0`. These settings generate **Ollama server environment variables**,
not per-request options; the server must be restarted to apply a change.

For a manually launched server:

```sh
eval "$(cfl runtime-env --kv-quantization)"
ollama serve
# Switch off before the next server launch:
eval "$(cfl runtime-env --no-kv-quantization)"
```

For the existing systemd service, write the drop-in with
`sudo scripts/ollama_env.sh --apply --kv-quantization`, then run
`sudo systemctl daemon-reload` and `sudo systemctl restart ollama`. Use
`--no-kv-quantization` in the same sequence to disable it. `cfl runtime-env --format
systemd` prints a reviewable drop-in without applying it.

The live comparison harness is `scripts/validate_kv_cache.py`. It needs an existing
model directory and GPU access; it temporarily unloads resident models, launches
an isolated server, and restores the original model names and context sizes.
See `docs/validation/cache-and-memory.md` for measured memory and performance,
quality review, and repeat commands. `q8_0` was live-tested; `q4_0` is selectable
but was not included in this comparison.

## Milestone 6 model decision

The measured repository configuration selects `qwen3.5:9b` and
`nomic-embed-text` (768 dimensions), with an 8,192-token generator context and KV
quantization off. See [the decision and measurements](docs/decisions/0001-model-choice.md)
for the three-generator comparison, dense-only embedding recall, source-review
rationales, and final GPU validation. `scripts/bakeoff.py --help` exposes candidate
lists, exact tokenizers, and archived prompt-contract replay. The script uses the
fixture without changing the production index and restores previous residency.

All 40 selected-model summaries pass the corrected JSON contract. Their content
still requires the later evidence verification pipeline; contract validity alone
is not semantic correctness. M7's build-time estimator remains pending, so M6
records a measured per-symbol projection rather than a completed build estimate.
