# Retain Qwen3.5 9B after the 14B comparison

Date: 2026-10-04. Status: measured; retain current default.

Compared Qwen2.5-Coder 14B Q4_K_M and Qwen3 14B Q4_K_M with the selected Qwen3.5 9B using the complete current fixture pipeline, regenerated summaries and all 48 source-reviewed questions. FP16 and Q8 KV were probed at 8,192 tokens. See the [validation report](../validation/14b-comparison/review.md), [method](../validation/14b-comparison/method.md) and [machine-readable results](../validation/14b-comparison/summary.json).

The fresh baseline has 38 adequate, seven partial, two incorrect and one unanswered answer. Coder 14B with Q8 KV has 36 adequate, ten partial, one incorrect and one unanswered; Qwen3 14B with Q8 has 36 adequate, eight partial, four incorrect and none unanswered. Coverage improves only for Qwen3, while correctness does not improve overall. Both larger models correct some baseline handler-summary errors but introduce other errors. This is a single seeded run on a small fixture; it does not establish a universal model ranking.

Both 14B models with FP16 KV cause partial CPU placement of the embedding model and are rejected before quality evaluation. With Q8, Coder uses 9.848 GB of reported resident-model memory; Qwen3 uses 10.016 GB and fails the current 10 GB model gate. Total device peaks are 11.34 GB and 11.17 GB respectively, exceeding the earlier total-device peak target. Generated-answer mean latency is 5.34 seconds for Coder and 3.88 for Qwen3, versus 3.20 for the FP16 baseline.

Keep `gen_model = "qwen3.5:9b"` and `kv_quantization = false`. Improving source grounding and summary correctness is the next useful direction; parameter count alone does not solve these observed failures. Any future promotion should repeat the evaluation on broader repositories and satisfy the chosen GPU budget. No main index or application configuration was modified by this experiment.
