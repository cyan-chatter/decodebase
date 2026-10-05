# CPU placement comparison — 2026-10-04

## Scope

Live RTX 4070 12 GB / Ryzen 5 5500 experiments requested to test two uses of a 24 GB CPU-memory budget: CPU-only embeddings with fully GPU-resident 14B FP16-KV generators, and a larger Qwen3.5 27B Q4_K_M generator split across CPU and GPU. The 27B experiment tests FP16 and Q8 KV. See [method.md](method.md) for controls and reproduction commands.

The host already has unlimited Ollama cgroup memory allocation. These experiments therefore use existing physical RAM; increasing a software allocation alone does not create capacity. The 24 decimal GB limit is sampled proportional memory for the isolated server family, not an OS-enforced cgroup cap. GPU peaks include desktop allocations.

Each model rebuilt 40 summaries and 60 embeddings before answering 48 questions in a fresh database. All answers and summary one-liners were independently inspected against fixture code. Eight structural answers are deterministic graph queries; generated-answer latency excludes them. Citation acceptance is separate from correctness.

## Completed observations

- Coder14 FP16 with CPU embeddings: 33 adequate, 11 partial, 2 incorrect, 2 unanswered; mean generated-answer latency 4.81 seconds. Peak GPU 11.44 GB, CPU PSS 9.93 GB, no process swap observed.
- Qwen14 FP16 with CPU embeddings: 37 adequate, 6 partial, 4 incorrect, 1 unanswered; mean generated-answer latency 3.83 seconds. Peak GPU 11.34 GB, CPU PSS 9.84 GB, no process swap observed.
- Qwen27 FP16 hybrid: 40 adequate, 6 partial, 2 incorrect, no unanswered; mean generated-answer latency 45.46 seconds. Peak GPU 11.77 GB, CPU PSS 18.03 GB, process swap peaked at 0.119 GB. It incorrectly describes real email delivery through a return-only stub and misinterprets short-circuit signing conditions. Its fresh summary one-liners avoid the material handler/recursion errors seen in smaller candidates.

Earlier comparable runs: default Qwen3.5 9B FP16 scored 38 adequate, 7 partial, 2 incorrect, 1 unanswered at 3.20 seconds mean; all-GPU Coder14 Q8 scored 36 adequate at 5.34 seconds, and Qwen14 Q8 scored 36 adequate at 3.88 seconds. Results are one seeded run on a small fixture, not a statistical model ranking. Rebuilding model-specific summaries makes this an end-to-end pipeline comparison rather than a pure final-answer-model experiment.

The CPU-embedding approach fits both 14B FP16 generators fully on GPU, but does not show a consistent quality improvement. The 27B FP16 hybrid gains two adequate answers over the default at approximately 14 times its mean latency and retains both important reasoning errors.

## Final comparison

| Configuration | Adequate / 48 | Partial | Incorrect | Unanswered | Mean generated answer | Peak CPU PSS | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default 9B, FP16 KV, all GPU (prior run) | 38 | 7 | 2 | 1 | 3.20 s | Not sampled | 8.32 GB |
| Coder14, FP16 KV, CPU embeddings | 33 | 11 | 2 | 2 | 4.81 s | 9.93 GB | 11.44 GB |
| Qwen14, FP16 KV, CPU embeddings | 37 | 6 | 4 | 1 | 3.83 s | 9.84 GB | 11.34 GB |
| Qwen27, FP16 KV, hybrid | 40 | 6 | 2 | 0 | 45.46 s | 18.03 GB | 11.77 GB |
| Qwen27, Q8 KV, hybrid | 38 | 9 | 1 | 0 | 47.73 s | 17.71 GB | 11.86 GB |

Qwen27 Q8 completed all questions and saved all summaries without failures. It avoids FP16's explicitly inverted signing condition, but omits the actual separator guard and evaluation order. It repeats the false delivery claim, introduces an input-mutation claim for audited persistence, and mislabels nonrecursive path endpoints. See each case's `source-review.json` for individual assessments and `summary.json` for raw aggregates.

Native Q8 KV allocation is 272 MiB versus FP16's 512 MiB: CPU cache 119 versus 224 MiB, GPU cache 153 versus 288 MiB. Both runs still place 36 of 66 layers on GPU, so the cache reduction does not offload additional model layers. Observed generation throughput is 4.10 tokens/s for Q8 versus 4.18 for FP16. Q8 shows no process swap; FP16's process swap peak is 0.119 GB. Peak GPU measurements include desktop variation and do not establish that Q8 uses more model memory. The native cache allocation is the clearer cache-memory evidence.

Summary generation took 25.46 minutes for FP16 and 25.76 minutes for Q8. Mean path-answer latency was 89.61 and 98.39 seconds respectively; deterministic structural queries remained fast. CPU embedding construction took roughly 3.2–3.4 seconds across all four cases.

## Decision and verification

Retain Qwen3.5 9B and KV quantization off as application defaults. The CPU-embedding approach is feasible and may be useful when GPU capacity is the constraint, but these runs do not show a consistent accuracy benefit. The 27B hybrid is feasible inside the monitored 24 GB budget and provides complete accepted answers on this fixture, but its modest quality gain in FP16 costs roughly 14 times the latency and still leaves important reasoning mistakes. Q8 saves cache memory without improving speed or adequate-answer count here. A 27B mode could be an explicitly selected slow experiment; these measurements do not justify adopting it as the default.

The normal application preflight still rejects CPU placements and the 14B FP16 generators exceed its 10 GB model-VRAM gate. These are diagnostic capabilities, not a production CLI offload mode. No service allocation, model default, or production index was changed.

All four cases have zero application answer-cache hits, 40 saved summaries with zero failures, and 60 CPU-generated embeddings. Intended placement was verified before and after each question set. Structural and traversal exactness are 1.0; deterministic structural answer text matches the independently verified baseline. Fixture/question/configuration SHA256 hashes remain unchanged. Temporary databases were dropped, and original 9B/Nomic model residency and contexts were restored. Five placement-guard tests pass; Ruff and `git diff --check` pass. Verification evidence is saved in `post-verification.json`.

