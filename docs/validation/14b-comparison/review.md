# 14B comparison results

Measured on 2026-10-04 using the RTX 4070 12 GB, Ollama 0.35.1, Q4_K_M weights, Nomic embeddings, 8,192-token context, temperature 0.1, seed 42 and thinking disabled. See [method.md](method.md) for controls and [summary.json](summary.json) for exact measurements.

## Result

**Keep Qwen3.5 9B as the default.** Neither 14B candidate improved overall independently reviewed source correctness. Qwen3 increased cited-answer coverage to 48/48, but produced more materially incorrect answers and exceeded the resident-model budget even with Q8 KV. Qwen2.5-Coder corrected the authenticated-handler summaries and reduced materially incorrect answers to one, but introduced other omissions and another unanswered query, with slower answers and much higher memory use.

| Model / KV | Adequate | Partial | Incorrect | Unanswered | Mean generated answer, s | Device peak, GB | App residency / 10 GB gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-f16 | 38 | 7 | 2 | 1 | 3.20 | 8.32 | Pass |
| baseline-q8_0 | 36 | 9 | 2 | 1 | 2.78 | 8.25 | Pass |
| coder14-f16 | — | — | — | — | — | 11.38 | GPU residency failed |
| coder14-q8_0 | 36 | 10 | 1 | 1 | 5.34 | 11.34 | Pass |
| qwen14-f16 | — | — | — | — | — | 11.40 | GPU residency failed |
| qwen14-q8_0 | 36 | 8 | 4 | 0 | 3.88 | 11.17 | Fail |

All semantic counts include the eight identical deterministic structural answers. Generated-answer latency excludes those eight zero-generation answers and includes retrieval, source verification, flow generation and bounded citation repair. A cited answer can be incorrect; “adequate” is the independent source assessment, not the application's sufficient-answer flag.

## Memory and cache findings

Both 14B FP16-KV cases stopped after preflight: each generator was fully on GPU, but Nomic partially spilled to CPU (about 0.103 GB GPU out of 0.397 GB reported total). Those cases have no answer-quality results. No CPU-offloaded answers are included in the comparison.

With Q8 KV, Qwen2.5-Coder's resident models use **9.848 GB**, passing the app's resident-model gate with little headroom. Qwen3 uses **10.016 GB**, failing that gate. Both have total device peaks above 10 GB, including desktop and runtime allocations. Consequently, neither meets the earlier milestone's total-device peak target. The diagnostic Qwen3 run records the policy failure; no application limit was relaxed.

The baseline uses **6.052 GB** of reported model memory with FP16 KV and **5.927 GB** with Q8 KV. The Q8 baseline has fewer adequate answers in this run, although a single seeded comparison cannot establish a statistically reliable quantization penalty. Quantization is a capacity tradeoff, not evidence of improved correctness.

## Source review

All 192 answers from the four completed cases were independently checked against implementation; each case has `source-review.json` with per-question notes. All 160 summary one-liners were also inspected, with material issues identified in those files. Detailed source records and generated summary fields remain in `summaries.json`.

- Baseline FP16: handler descriptions confuse authenticated handlers with the route decorator factory. Notification delivery and short-circuit signing claims remain incorrect. The unanswered query is `behavioral-08`.
- Coder Q8: authenticated handler summaries and their module answer improve. It still invents actual email delivery in `behavioral-02`, overclaims exception/unique-salt guarantees, conflates audit behavior with the base repository, and omits some configuration/import details. Its unanswered query is `traversal-08`.
- Qwen3 Q8: answers every query with accepted citations, but invents an even-value recursion base case in both the `is_even` one-liner and `traversal-08`. It also claims actual email sending in three answers. These errors explain why increased coverage does not constitute increased accuracy.

Both 14B models fix some baseline errors while introducing new ones. For the requested tracing question, their signing descriptions improve on the baseline's inverted call condition but still omit the separator-dependent short-circuit detail. Per-question differences matter more than parameter count alone.

## Verification and limitations

Every completed case regenerated 40 summaries (four trivial templates, zero failures), built 60 dense vectors, and evaluated all 48 questions with **zero application answer-cache hits**. Structural and traversal graph exactness are 1.0 in all cases; retrieved and cited recall are recorded separately. Native prefix reuse remains enabled as in normal application use. All temporary databases were dropped, and the original selected generator/embedder residency and contexts were restored. `cfl.toml`, the default-off KV setting and the main source index were preserved.

This is one run per feasible model/cache setting on a small fixture, not a broad coding benchmark or a repeated statistical study. Rebuilt summaries also change retrieval, so this tests the complete application rather than isolating the final answer model. The earlier milestone report used mixed application caches; the fresh baseline here is the appropriate latency and quality comparator. GPU peaks include background desktop activity and can vary.

The reproducible runner is `scripts/compare_14b.py`; report aggregation is `scripts/report_14b.py`. Ruff, syntax compilation and artifact consistency checks passed. No application implementation or default configuration was changed for this experiment.
