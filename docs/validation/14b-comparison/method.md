# 14B model comparison

Requested comparison: Qwen2.5-Coder 14B Q4_K_M and Qwen3 14B Q4_K_M against the selected Qwen3.5 9B Q4_K_M. Each model is tested with FP16 and Q8_0 KV caches at 8,192 tokens. KV quantization remains disabled in the application configuration.

## Controls

- Same current parser, resolver, dependency-aware summary worker, hybrid retrieval, answer/flow engines, prompts and 48-question source-reviewed fixture.
- Temperature 0.1, seed 42, thinking disabled, symbol output ceiling 350 and detailed output ceiling 1,200 tokens. Model-specific tokenizers downloaded from pinned official Hugging Face revisions; checksums saved in each report.
- One isolated Ollama server and one isolated database per case. No reuse of application summary, embedding or answer caches between cases. Summaries regenerated using each model; source and graph evidence unchanged. Native prompt-prefix caching remains enabled as in normal application use.
- An uncached near-context-limit preflight request measures generation and prefill throughput. Both the generator and Nomic embedder must be fully GPU-resident with the requested context. CPU-offloaded cases stop before quality evaluation.
- The application retains its 10 GB resident-model budget. Diagnostic quality measurements may proceed when the only preflight failure is that budget; such a case is explicitly ineligible for the current configuration. Total device peak, including desktop and runtime allocations, is also reported separately.
- Each case saves all 40 summary records, every answer, citation/retrieval metrics, per-question wall latency, native request logs, server logs and sampled GPU usage. Eight structural answers are deterministic and require no answer generation; aggregate latency includes them and will also be broken down by route.
- Semantic quality is independently reviewed by Codex against fixture source. Citation validity and recall are separate measurements, not correctness grades. Docstrings are checked against implementation, especially the email-return stub and dictionary-backed connection.
- Temporary fixture databases are dropped and original loaded models restored. The main application configuration, selected model and source index are preserved.

## Reproduction

Pull `qwen2.5-coder:14b` and `qwen3:14b-q4_K_M`, and provide matching tokenizer files under `.cfl/tokenizers/qwen25-coder-14b/` and `.cfl/tokenizers/qwen3-14b/`. The baseline uses `.cfl/tokenizers/qwen3.5-9b/`.

```sh
.venv/bin/python -m scripts.compare_14b \
  --cases baseline-f16 baseline-q8_0 coder14-f16 coder14-q8_0 qwen14-f16 qwen14-q8_0
```

Requires the existing local PostgreSQL credentials to create temporary databases, the shared Ollama model directory, an unused port 11436 and native GPU access. Downloads are not included in run latency. Each result is a single seeded run, not a repeated statistical benchmark.
