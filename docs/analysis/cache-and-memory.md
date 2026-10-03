# Cache and persistent memory analysis

Subsequently implemented and tested: see `../validation/cache-and-memory.md`.
The findings below describe the pre-improvement state.

Analyzed on 2026-10-04 before milestone 6. This is an assessment and proposal;
application code and Ollama service settings were not changed.

Ollama already reuses prompt KV state. PostgreSQL is the durable knowledge store.
Conversation memory, generated summary reuse, and generated answer reuse are
planned but are not yet wired into the application. KV reuse speeds computation;
it does not make omitted source evidence available to the model.

## Existing behavior

| Layer | Current implementation | Persistence and limitations |
| --- | --- | --- |
| Inference KV / prefix cache | Ollama manages it; live requests confirm prefix reuse | Runtime state; not an application-managed durable memory or knowledge index. Reuse depends on matching token prefixes and available runner cache. |
| Model residency | Generation and embedding requests send `keep_alive=-1` | Keeps models loaded; does not automatically include previous questions or answers in new prompts. |
| Repository knowledge | PostgreSQL stores files, symbols, source, edges, graph metadata, lexical search fields, and evaluation runs | Active and durable; retrieval supplies selected evidence to each request. |
| Summary and embedding reuse | Summary/embedding fields, hash helpers, and DB persistence helpers exist | Summary generation arrives in M7; dense indexing in M8. Helpers alone do not mean the pipelines are using these caches. |
| Answer reuse | `answer_cache` plus read/write helpers exist | Planned for M8; `cfl ask` remains a stub, so no answer-cache lookup currently runs. |
| Conversation working memory | Client accepts a caller-supplied messages list | No session store or automatic retained history. M11 plans recent turns plus a bounded symbol/file direction seed and deterministic compaction. |
| Operational records | Calibration JSON, benchmark JSON, JSONL call metadata | Durable measurements, not model-visible repository or conversation memory. |

Source references: `cfl/core/client.py`, `cfl/core/db.py`, `cfl/core/hashing.py`,
`cfl/core/trace_log.py`, `cfl/cli.py`, `migrations/0001_core.sql`, and
`Execution_Order.md` M6–M11.

Ollama documents model residency separately from context/cache configuration in its
[FAQ](https://docs.ollama.com/faq). Its
[generation API](https://docs.ollama.com/api/generate) reports total prompt tokens,
cached prompt tokens, and time spent evaluating uncached prompt tokens. The
[installed v0.35.1 API source](https://raw.githubusercontent.com/ollama/ollama/v0.35.1/api/types.go)
also subtracts cached tokens when reporting prompt-evaluation throughput.

## Live measurements

Ollama 0.35.1 currently has Qwen2.5-Coder 7B Q4_K_M and nomic-embed-text loaded
fully on GPU. The generator uses an 8,192-token context. Combined reported model
VRAM is 5,457,093,589 bytes, below the configured 10 GB limit. This figure is
Ollama model residency, not a measurement of total GPU allocation or peak usage.

Four sequential synthetic requests used the already resident generator, identical
system instructions, temperature 0, seed 42, and an eight-token output limit.
No models were unloaded and no configuration was changed.

| Request | Prompt tokens | Cached tokens | Prompt evaluation | Wall time |
| --- | ---: | ---: | ---: | ---: |
| First unique prompt | 780 | 3 | 166.4 ms | 447.3 ms |
| Exact repetition | 780 | 779 | 13.0 ms | 34.1 ms |
| Changed ending | 780 | 770 | 18.9 ms | 43.1 ms |
| Changed early prefix | 779 | 27 | 149.3 ms | 197.8 ms |

These are four diagnostic observations, not a statistically representative
throughput or answer-quality benchmark. The first request is a prompt-cache miss,
not a cold model load. Saved metrics are in `.cfl/cache-analysis/report.json`.

## Improvements before and during milestone 6

1. Preserve `prompt_eval_cached_count` in `GenResult` and JSONL traces, alongside
   total prompt tokens and the native durations. Treat absent cache metrics as
   unknown for runtimes that do not report them. Continue budgeting the complete
   prompt: cached tokens still occupy context space.
2. Correct `warm_and_benchmark`: it currently divides total prompt tokens by
   uncached evaluation time. On cache hits this inflates prefill speed, distorting
   projected ingest hours and model comparisons. Use uncached token count for
   prefill throughput, and report cold loading, warm uncached prompts, exact
   repetition, and shared-prefix/different-symbol requests separately. Do not
   estimate fresh-input throughput from nearly fully cached prompts.
3. Implement M7's stable prompt prefix: fixed system instructions and task/schema
   instructions first; variable file, signature, dependencies, and code afterward.
   Keep formatting and dependency ordering deterministic. Avoid timestamps,
   random IDs, or changing question text before shared context in production
   prompts. Preserve dependency-first processing; locality optimization must not
   reorder callers ahead of their dependencies.
4. In M6, compare candidates under the same prompt/schema, cache workload,
   context/output limits, and quality tests. Retain both uncached and realistic
   shared-prefix timings. Measure peak total GPU usage as well as model residency.
   Cache reuse accelerates evaluation; it does not improve factual grounding by
   itself. Grounding still requires the complete source evidence retrieval added
   after M5.

## Durable cache correctness gaps

Three small, direct checks reproduced the following issues; results are saved in
`.cfl/cache-analysis/hash-checks.json`.

- `index_version()` includes model tags and prompt version but omits the existing
  repository epoch. Changing epoch from 1 to 2 produced the same index version.
  An M8 answer cache could therefore reuse an answer after source/graph changes.
  Include repository identity, a reliably advanced content/index revision, and
  retrieval/router/verification versions in the key. Include query, mode, relevant
  settings, and session state when answers depend on conversation history.
- `normalize_code()` strips indentation from every line. Moving `finish()` inside
  an `if` changed Python's AST but preserved `code_hash`. Preserve semantic
  indentation or use a validated language-aware fingerprint before summary reuse
  relies on this key.
- `ctx_hash()` sorts by callee IDs but includes only summary hashes in its final
  digest. Replacing a callee ID while retaining its summary text preserved the
  key. Include the ID-summary pairs that actually appear in the prompt, or hash
  the final deterministic prompt inputs.

Model tags can also be repointed. Future summary/embedding/answer keys should
include the resolved model digest, relevant generation parameters, and prompt
schema version. Generated summaries should retain source IDs/spans and input
fingerprints. An answer cache should serve only verified answers whose referenced
evidence is still valid for the current index revision.

The build should keep each symbol's generation context isolated, as already
specified in the plan. Persist structured summaries and retrieve them for future
requests rather than accumulating all prior symbol prompts in one conversation.

## KV memory and session memory

The installed Qwen architecture reports 28 layers, four KV heads, and 128 values
per head. A conventional f16 K/V allocation is approximately
`2 * 28 * 4 * 128 * 2 = 57,344 bytes/token`: 448 MiB at 8K or 896 MiB at 16K per
sequence, excluding padding, buffers, and other runtime overhead. These are
architecture-based estimates, not measured cache allocations. Other M6 candidates
can have different KV costs even at similar parameter counts.

Ollama's FAQ lists f16 as the default cache type, with q8_0 and q4_0 alternatives;
quantized caches require Flash Attention. No explicit cache-related overrides
were returned by the service-environment query; the effective backend cache type
and Flash Attention configuration were not independently confirmed. Start the
bake-off with a consistent baseline. Test q8_0 if longer context or a larger model
becomes memory-bound, and retain it only if measured quality and latency justify
the change. Benchmark JSON validity and source accuracy as well as memory.

For M11, use the planned bounded recent turns and direction seed as working
memory. If cross-process resumption is desired, persist a session ID, recent turns,
source references, user constraints, and index revision in PostgreSQL. Rehydrate
source evidence from the current index and revalidate references after edits.
Keep verified facts distinct from tentative model conclusions. The model sees
this memory only when selected material is explicitly included in its prompt.
KV persistence on disk is not needed for this design; durable text and structured
records survive restarts and can be used with a different selected model.

Recommended order: correct metrics and invalidation first; run the M6 comparison
with separate cache workloads; implement summary and embedding reuse in M7/M8;
add bounded session memory in M11. KV quantization is a measured hardware option,
not a prerequisite for durable memory.
