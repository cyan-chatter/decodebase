# Cache, memory, and KV quantization validation

Verified on 2026-10-04. KV quantization remains off by default (`f16`). The opt-in
`q8_0` path was tested against the live RTX 4070 and Ollama 0.35.1.

## Implemented changes

- Native cached-token counts and durations retained in generation results and JSONL logs.
- Fresh-input throughput uses uncached tokens; unknown or nearly complete cache hits do not produce misleading throughput estimates. Full prompt tokens still determine context fit.
- `cfl cache-benchmark` separates fresh, repeated, shared-prefix, and changed-prefix requests.
- Code fingerprints preserve indentation/string whitespace. Callee IDs, model digests, schema/prompt versions, settings, repository identity, parser/resolver versions, and content epochs participate in compatibility keys. Metadata is read atomically, and epoch increments are atomic across concurrent connections.
- Parser version `python_v3` reparses old fingerprints. Source edits advance the epoch; unchanged scans avoid churn.
- `KnowledgeMemory` supplies validated summary and embedding reuse, with deterministic stable prompt prefixes and duplicate embedding batching. Corrupt summaries regenerate; source changes during generation prevent saving stale output.
- Verified-answer storage includes source fingerprints and ranges; entries without verification or current evidence are not served.
- `SessionMemory` persists bounded recent turns, constraints, and source references, revalidates sources, clears stale discussion after index changes, and budgets history below current source.
- Additive `0003_memory.sql` migration adds verified-answer evidence and session storage. Applied to the configured DB without resetting existing data.
- `runtime-env` and `scripts/ollama_env.sh` generate explicit `f16`/`q8_0` settings. `q4_0` is also selectable but not live-tested here.

The reusable memory/prompt APIs are implemented and exercised now. Complete build,
ask, and chat orchestration remains assigned to its milestones; this change does
not mark M6 model selection or the subsequent ingestion/chat milestones complete.

## Live GPU comparison

The harness launched isolated servers with Flash Attention enabled and one parallel
sequence. It unloaded the original models, waited for their GPU allocations to
release, and restored their names/context sizes afterward. Exact Qwen token counting
was used. A separate warm-up preceded the measured fresh-input benchmark; three
rounds of each short cache workload were recorded at each context.

| KV type | Context | KV allocation from runner logs | Combined model VRAM | Peak total GPU | Warm uncached prefill | Generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f16 | 8,192 | 448 MiB | 5.457 GB | 6,295 MiB | 4,645.0 tokens/s | 78.8 tokens/s |
| f16 | 16,384 | 896 MiB | 5.944 GB | 6,757 MiB | 4,153.5 tokens/s | 78.3 tokens/s |
| q8_0 | 8,192 | 238 MiB | 5.237 GB | 6,088 MiB | 4,622.0 tokens/s | 80.1 tokens/s |
| q8_0 | 16,384 | 476 MiB | 5.503 GB | 6,340 MiB | 4,090.0 tokens/s | 71.5 tokens/s |

Both generator (29/29 layers) and embedder (13/13 layers) were fully on GPU.
All final peak allocations were below the configured 10 GB budget. q8_0 reduced
KV storage by 210 MiB at 8K and 420 MiB at 16K. The 8K generation measurement was
similar; the 16K q8_0 measurement was about 9% slower. These are small diagnostics,
not a statistically representative model-selection benchmark. Quantization is a
memory option, not a guaranteed speed improvement.

| KV type / context | Fresh short prompt median | Exact repeat median | Shared prefix median |
| --- | ---: | ---: | ---: |
| f16 / 8,192 | 204.2 ms | 67.4 ms | 75.4 ms |
| f16 / 16,384 | 207.0 ms | 67.5 ms | 72.8 ms |
| q8_0 / 8,192 | 195.2 ms | 66.1 ms | 74.6 ms |
| q8_0 / 16,384 | 199.5 ms | 65.0 ms | 75.4 ms |

Runner logs explicitly confirm `K (f16), V (f16)` in off mode and `K (q8_0),
V (q8_0)` in on mode. The main systemd service configuration was left unchanged;
original generator 8K and embedder 2K contexts were restored. Newly initialized
temporary user identity files were removed. The final report confirms restoration.

An earlier warm run observed a 10,112 MiB transient allocation, consistent with
overlapping model allocations during test startup. The harness now waits for release before
starting the comparison. That earlier report is retained; the final measurements
above come from `.cfl/kv-validation-final/report.json`, with matching runner logs.

## Output quality review

At 8K, both modes generated structured summaries for the repository, retry, and
pipeline fixture files. All six outputs passed the strict Pydantic schema. Source
review confirmed the repository save/get and audit behavior, RuntimeError-only
bounded retry and final re-raise, and pipeline normalization followed by repository
persistence. Summaries remained broadly consistent across modes. Some fields are
incomplete: both pipeline summaries omit indirect exceptions, and the q8_0 retry
summary omits RuntimeError from its raises list while describing re-raise in the
logic field. This small sample does not establish equivalence of answer quality.
The full M6 evaluation is still required before selecting models/settings.

The fixture retry uses total `attempts`; the production helper introduced earlier
uses `max_retries` after the initial call. This comparison did not alter the fixture.

## Reproduction and user controls

```sh
cfl runtime-env --no-kv-quantization
cfl runtime-env --kv-quantization
cfl runtime-env --kv-quantization --kv-type q8_0 --format systemd
cfl cache-benchmark --repetitions 3
.venv/bin/python scripts/validate_kv_cache.py \
  --models /usr/share/ollama/.ollama/models \
  --tokenizer .cfl/tokenizers/gen/tokenizer.json \
  --out .cfl/kv-validation-final
```

Environment generation alone does not change an already running server. For a
manual server, evaluate the shell output before `ollama serve`. For systemd, use
`sudo scripts/ollama_env.sh --apply --kv-quantization`, then daemon-reload/restart;
use `--no-kv-quantization` in the same sequence to switch off. The configuration
`kv_quantization = false` / `CFL_KV_QUANTIZATION=false` is the default.

## Regression and durable memory validation

- Full suite: **252 passed**.
- Final exact-tokenizer `cfl doctor` passed against the restored main service: approximately 4,734 uncached prompt tokens/s and 81 generated tokens/s; GPU residency, database, embedding dimension, and RAM checks passed.
- Ruff production/maintained-test/fixture checks, changed-file formatting, shell syntax, and whitespace checks passed; existing resolver fixtures and `test_db.py` lint findings are excluded.
- Tests cover cached-token validation and full-context budgeting, unknown cache metrics, distinct workloads, stable prefixes, semantic indentation changes, digest/callee invalidation, no-op scan stability, verified source evidence, bounded session restoration, history budgeting, atomic concurrent epoch increments, and rejected invalid or stale summaries.
- Live summary/embedding/session evidence is saved under `.cfl/memory-validation/`; one generation and one embedding request sufficed, with subsequent calls served from PostgreSQL. The saved session restored and assembled a 187-token prompt. Its database was disposable and removed after validation.

The prior assessment is `docs/analysis/cache-and-memory.md`; the issues recorded
there are historical findings corrected by this implementation.
