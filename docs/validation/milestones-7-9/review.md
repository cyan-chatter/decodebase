# Milestones 7–9 validation

Validated on 2026-10-04 against `Execution_Order.md`, using the selected Qwen3.5 9B generator, Nomic 768-dimensional embeddings, 8,192-token context, and KV quantization off. No human signoff was required: all 48 answers were independently reviewed against the fixture source by Codex.

## Implementation coverage

| Plan | Implemented and checked |
| --- | --- |
| M7.1–M7.2 | Native structured summary contract, deterministic word trimming, trivial templates and skip/minimum-line rules. |
| M7.3 | Bottom-up SCC worker, fitting combined SCC requests, signature-only fallback, AST partial reduction, method-first class rollups, context/hash reuse, one repair, quarantine, physical-attempt accounting, per-symbol transactions and interrupt-safe saves. |
| M7.4 | GPU preflight, parse/resolve/lexical stages, measured estimator, default resume, progress/rate/ETA, stale/failed status and independent dense stage. |
| M7.5–M7.6 | Fake-client order, edit ripple, stable-description stop, prompt invalidation, malformed JSON, transport retries, oversized symbols, class ordering and kill/resume tests; actual GPU estimate, Ctrl-C and resume. |
| M8.1 | Source-head-capped callable embeddings, file skeletons, future rollup keys, persistent digest-aware cache, exact cosine scans and independent dense failure. |
| M8.2–M8.4 | Deterministic routing, RRF, optional graph expansion/reranking, scoped module retrieval, configured context cap, detailed neighbor one-liners, source verification, structural zero-generation answers, answer caching and full-answer metrics. |
| M8.5 | Router/RRF/budget/cache/verification/failure tests plus live ask and all 48 evaluation questions. |
| M9.1–M9.2 | Escaped, bounded, validated Mermaid; brief zero-generation explanations; detailed source/neighbor/class-rollup explanations with source-aware cache invalidation. |
| M9.3–M9.4 | Source-line-ordered DFS, conditional ternary calls, external leaves, cycle markers, diagram collapse/fallback, bounded step/stitch prompts, Markdown output and live cycle/flow validation. |

Explicit trace questions retain every node of the requested indexed path, even when deeper than the generic flow command's default depth. Diagram collapse does not discard that requested narrative evidence. Graph questions now recognize both “Who calls” and “What calls”; module-qualified names resolve without fuzzy endpoint substitution. The main database received the additive migration while its repository identity and index epoch stayed unchanged (`main-schema.json`). Milestone 10 rollups and milestone 11 chat remain outside this work.

## Reproducible evidence

- `tests.log`: **297 tests passed**, including all existing regressions. The 16 warnings are existing pathspec deprecations. Ruff and `git diff --check` passed.
- `estimate.log`: 36 model-bound symbols plus four trivial templates; approximately 0.022 hours for summaries. Estimates explicitly exclude repairs, reductions, embeddings and I/O.
- `build-interrupt.log` and `status-interrupt.json`: Ctrl-C left **23 durable summaries**, with 17 pending.
- `build-resume.log`: resumed **17**, reused all **23** completed summaries, and generated no repeated completed requests. Its initial dense failure exposed the native pgvector `Vector` decoding issue; the subsequent complete run fixes that decoder.
- `final-build.log` and `final-status.json`: **40 cached summaries**, zero symbol-generation requests, **60 vectors**, all views fresh, zero stale/failed/pending symbols. The build still runs its explicit preflight benchmark; zero repeated summary/embedding work does not mean zero preflight requests.
- `explain-first-generation.json` and `explain-first-cache-hit.json`: the first validated detailed explanation is generated; its repeat is cached. `explain.json` and `explain-cached.json` confirm reuse in the final run.
- `flow.md`, `flow.json`, and `cycle.json`: source-order evidence, generated diagrams and recursion validation.
- `execution.json` and `calls.jsonl`: local GPU measurements and native request/token metadata. Peak device usage was **8.075 GB** (7701 MiB), below the 10 GB limit. Every recorded final command exited successfully.

The isolated fixture validation database was removed after the artifacts were saved. The main source index was preserved.

## Full-answer measurements

| Category | Retrieved Recall@5 | Cited Recall@5 | Citation validity | Tokens/answer |
| --- | --- | --- | --- | --- |
| behavioral | 0.938 | 0.812 | 0.875 | 1552.9 |
| module | 1.000 | 0.938 | 1.000 | 1000.5 |
| reasoning | 0.938 | 0.875 | 1.000 | 1622.0 |
| structural | 1.000 | 1.000 | 1.000 | 0.0 |
| symbol | 1.000 | 1.000 | 1.000 | 655.9 |
| traversal | 1.000 | 0.746 | 1.000 | 1004.1 |

Across all 48 questions: retrieved Recall@5 **0.979**, cited Recall@5 **0.895**, mean per-question citation validity **0.979**, and mean native input plus output tokens **972.6**. Structural and traversal graph exactness are **1.000**. All eight traversal queries retrieve the complete expected path; lower cited recall reflects narrative citation selection rather than dropped path evidence. These are mixed-cache measurements (4 answers cached), not cold latency benchmarks.

**47/48** answers contain accepted source citations. The one uncited answer (`behavioral-08`) receives a bounded repair and then the required insufficient-context fallback. Every emitted citation accepted in the final run is within its supplied source evidence; the mean validity metric gives the uncited question a zero.

## Independent content review and limits

`source-review.json` records each question, source rationale and assessment: **36 adequate**, **8 partial**, **3 incorrect**, and **1 unanswered**. Citation validity does not mean every claim is true.

Material remaining model errors are actual-email wording for the fixture's `send_email` stub (`behavioral-02`), an inverted short-circuit explanation around token signing (`traversal-05`), and unsupported SQL wording for the in-memory connection (`reasoning-08`). Partial answers include a decorated handler called a factory, a loop described as conditional, overly broad parity-termination/exception claims, and an ordinary path endpoint called a recursion leaf. The code-generated path, source snippets, conditions and diagrams retain the correct evidence. These limitations are recorded rather than treated as a semantic pass or silently rewritten into model output.

The specified M7–M9 mechanics and acceptance checks pass. The selected model's narrative quality still needs the later quality/hardening work; this report does not claim production-level semantic accuracy.
