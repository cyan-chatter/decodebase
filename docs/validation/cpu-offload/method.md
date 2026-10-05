# CPU placement experiments

Two approaches requested on 2026-10-04:

1. Qwen2.5-Coder 14B and Qwen3 14B with FP16 KV, keeping each generator completely on GPU while explicitly setting `num_gpu=0` on Nomic embedding requests.
2. Qwen3.5 27B Q4_K_M split across CPU and GPU with CPU-only embeddings, testing both FP16 and Q8 KV at 8,192-token context.

The normal application continues to require full GPU residency and keeps Qwen3.5 9B with KV quantization off. The diagnostic runner validates the intended placement instead: a CPU-embedding experiment is invalid if its generator spills, and a hybrid experiment is invalid unless its generator occupies both CPU and GPU. Preflight rejection by the normal application is captured rather than hidden.

Each case runs a separate Ollama server and temporary fixture database, regenerates all 40 summaries, builds its own 60 embeddings and answers all 48 source-reviewed questions. Temperature 0.1, seed 42, thinking off and existing output/context limits remain fixed. Matching tokenizers have pinned official provenance. Application caches start empty; native prefix caching operates normally. The baseline and earlier Q8 GPU runs are retained in `../14b-comparison/` for comparison.

Host RAM is sampled every half-second for the isolated Ollama process family. Reported PSS apportions shared pages; RSS may double-count shared memory and include file-backed model mappings. Swap and system-available RAM are also captured. A sampled family-PSS budget of 24 decimal GB terminates the test server on excess. This is a monitored diagnostic budget, not a new OS cgroup allocation or a cap on the entire desktop. GPU peaks include desktop allocations. CPU model placement does not add physical RAM or VRAM.

All outputs and one-liners are independently reviewed by Codex against fixture implementation. Citation sufficiency and source correctness are scored separately. Each feasible case is one seeded run, not a repeated statistical benchmark. Temporary databases are removed and originally loaded models/contexts restored after each batch.

```sh
.venv/bin/python -m scripts.compare_14b \
  --cases coder14-f16 qwen14-f16 --placement cpu-embed \
  --host-ram-limit-gb 24 --out docs/validation/cpu-offload

.venv/bin/python -m scripts.compare_14b \
  --cases qwen27-f16 qwen27-q8_0 --placement hybrid \
  --host-ram-limit-gb 24 --out docs/validation/cpu-offload
```

The diagnostic script requires `psutil` (the project `sys` extra), native GPU/process access, existing local PostgreSQL credentials and the downloaded model/tokenizer files.
