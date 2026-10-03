# Milestone 4 live GPU validation

Validated on 2026-10-04 (Asia/Kolkata), using real HTTP requests to the local
Ollama server. The regression suite uses fakes separately.

| Setting | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070, 12,282 MiB |
| Driver | 595.91.07 |
| Ollama | 0.35.1 |
| Generator | qwen2.5-coder:7b, Q4_K_M |
| Generator context | 8,192, confirmed by request options and `/api/ps` |
| Embedder | nomic-embed-text:latest, F16 |
| Embedding dimension | 768 |
| Combined model VRAM | 5,457,093,589 bytes (5.46 GB) |
| Residency | Both models fully on GPU (`size_vram == size`) |

## Checks

- Structured generation returned a JSON object with a nonempty string summary.
- Both generate and chat succeeded in streaming and non-streaming modes.
- Stream output was consumed and final token metrics were written to JSONL.
- Thirty-three embeddings arrived in batches of 32 and 1; every vector had 768
  finite components. Requests set `truncate=false`.
- Oversized prompts raised `BudgetExceeded` before any live HTTP request, in both
  streaming and non-streaming modes.
- Two clients invoked concurrently had a maximum of one HTTP generation request
  in flight, measured through a transport wrapper that tracks response consumption.
- Every generation request used `num_ctx=8192`, explicit `num_predict`, configured
  temperature and seed, `keep_alive=-1`, and `think=false`.
- The exact-tokenizer benchmark reserved an estimated 7,680 prompt tokens including
  template overhead. Ollama reported 7,645 prompt tokens plus 64 output tokens,
  within the 8,192-token context. No truncation warning occurred.
- `cfl doctor` exited zero with Ollama, models, throughput, embedding dimension,
  residency, database extensions/schema metadata, and RAM all passing.
- Final regression suite: **175 passed**. Ruff checks passed for production code
  and maintained tests, excluding existing fixture/test_db lint findings.

## Measurements and interpretation

The first exact-tokenizer benchmark observed approximately 4,601 prompt tokens/s
and 79 generated tokens/s. Subsequent repeated prompts reused the prefix cache and
reported much higher effective prefill throughput; those figures do not measure
fresh-input processing speed. The first cold generation took 43.34 seconds,
including 22.76 seconds reported as load duration. A warm structured request took
approximately 0.46 seconds.

The heuristic benchmark estimated 7,680 tokens but Ollama reported 3,584 for its
repetitive English padding. Its warning disappeared with the matching tokenizer:
this was an estimate mismatch, rather than evidence of model truncation. Doctor's
warning wording now describes this uncertainty instead of asserting truncation.

Live doctor also uncovered an incorrect `schema_version` column lookup. Metadata
is stored as key/value rows; doctor now uses database diagnostics and `get_meta`.
Regression tests cover migrated metadata, an unmigrated database, and connection
failure.

## Local evidence and configuration

Raw reports and their JSONL traces are retained under the ignored state directory:

- `.cfl/live-validation/20261003T190435Z/report.json`: heuristic run.
- `.cfl/live-validation/20261003T190624Z/report.json`: first exact-tokenizer run.
- `.cfl/live-validation/20261003T190904Z/report.json`: final exact run, including
  concurrent generate and non-streaming chat.

The matching tokenizer was downloaded to `.cfl/tokenizers/gen/tokenizer.json`.
Exact counting was enabled through an environment override for validation; the
repository's default configuration remains unchanged. To repeat doctor with it:

```sh
CFL_TOKENIZER_FILE="$PWD/.cfl/tokenizers/gen/tokenizer.json" cfl doctor
```

The report's GPU memory is model residency from Ollama. `nvidia-smi` observed
approximately 6,399 MiB total GPU allocation, including other GPU users and runtime
allocations. Both measurements were below the configured 10 GB budget.
