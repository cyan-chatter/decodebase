# 0001: Milestone 6 model choice

Accepted 2026-10-04. Use **`qwen3.5:9b`**, **`nomic-embed-text`**, **768 embedding dimensions**, and **`NUM_CTX=8192`**. Keep KV quantization **off** (`f16` server cache). Disable generator thinking, use temperature 0.1, seed 42, and a 350-token summary output budget.

The newer generator had the highest independent source-review score in the controlled comparison. Its advantage was modest, rather than a general benchmark claim. Nomic tied the larger embedder on this fixture and used substantially less GPU memory. Retaining 768 dimensions requires no embedding reset.

## Verified candidates

Tags, download sizes and licenses were checked before downloading the new candidates, which were pulled sequentially. Existing baseline models were already installed. Download size is not GPU residency. The full source record is [candidates.json](../validation/milestone-6/candidates.json); actual resolved digests are stored in the results and database metadata.

| Candidate | Library download size | Weight format | License/source |
| --- | --- | --- | --- |
| [Qwen2.5-Coder 7B](https://ollama.com/library/qwen2.5-coder:7b) | 4.7 GB | Q4_K_M | Apache-2.0, library page |
| [Qwen3.5 9B](https://ollama.com/library/qwen3.5:9b) | 6.6 GB | Q4_K_M | Apache-2.0, library page |
| [Qwen3 8B](https://ollama.com/library/qwen3:8b) | 5.2 GB | Q4_K_M | Apache-2.0, library page |
| [Nomic](https://ollama.com/library/nomic-embed-text:latest) | 274 MB | F16 | Apache-2.0, library page |
| [Qwen3 embedding 0.6B](https://ollama.com/library/qwen3-embedding:0.6b) | 639 MB | Q8_0 | Apache-2.0, [publisher model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |

The Qwen embedding library page does not expose a license blob, so its license was verified with the publisher. Its Q8_0 **weights** are distinct from the optional KV cache quantization setting.

## Controlled generator comparison

Ollama 0.35.1 on an RTX 4070 with 12 GB VRAM. Every generator received the same prompt/schema v1 and all 40 parsed fixture symbols, including classes, nested decorators, recursion, tests, and stubs. Callees were left empty consistently: unseen implementations must not be invented. No generation failures were silently repaired or discarded. Each candidate was unloaded before loading the next; original model residency was restored after each benchmark invocation.

| Generator | JSON syntax | Summary contract | Source rubric /5 | Native generation t/s | Uncached prefill t/s | Mean seconds/symbol | Projected ingest hours, 334 symbols | Doctor resident models, GB | Peak total GPU, GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Coder 7B | 40/40 | 39/40 | 3.850 | 93.0 | 3311 | 1.393 | 0.129 | 5.457 | 6.642 |
| Qwen3.5 9B | 40/40 | 37/40 | **4.025** | 73.3 | 2651 | 2.154 | 0.200 | **6.052** | 8.284 |
| Qwen3 8B | 40/40 | 40/40 | 3.300 | 82.9 | 3111 | 1.635 | 0.152 | 6.619 | 7.785 |

Every candidate's doctor returned zero. Resident totals include Nomic during doctor; peak totals also include desktop use and transient loading allocations sampled every 0.2 seconds. GB is decimal. Generation t/s uses total output tokens divided by native evaluation duration. Prefill uses **uncached** prompt tokens, excluding samples with fewer than 32 uncached tokens; it does not count cached prefix tokens as newly evaluated work.

Rubric review was performed by **Codex against the source**, replacing human review as explicitly requested by the user. Every output has a source hash, score, and rationale in [review.json](../validation/milestone-6/review.json). This is agent review, not human sign-off or a second model judge. Pure summaries have no citation field, so the answer rubric is adapted to correctness, completeness, and grounding in supplied source.

Examples that affected scores: the baseline invented SQL behavior and actual delivery in the no-op email sink; Qwen3 invented `AuthenticationError`, copied-dictionary mutation, and nonexistent `json.JSONEncodeError`. Qwen3.5 correctly described the no-op email sink and several factory/closure details, but still contradicted itself on `AuditedRepository.save` and inferred database operations not established by the supplied code.

## Dense-only embedding comparison

32 previously agent-verified questions: all symbol, module, behavioral, and reasoning items. Structural and traversal questions were excluded because their required evidence belongs to graph retrieval. Both models indexed exactly the same 40 documents: file path, signature, docstring, and fixture source, without generated summaries or expected-answer text. Rankings are exact cosine top-five over normalized vectors, with stable ties; no lexical fusion, graph expansion, or reranking.

Nomic uses the publisher's [document/query prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). Qwen uses the publisher's [query instruction format](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B); documents have no instruction. Dimension was measured through `/api/embed`, with truncation disabled and the normal client dimension/finite-number validation applied.

| Embedder | Measured dimension | Mean AnswerRecall@5 | Full-recall questions | Embed workload seconds | Embedder-only resident GB |
| --- | --- | --- | --- | --- | --- |
| Nomic | 768 | **0.984375** | 31/32 | 23.99 | **0.323** |
| Qwen3 embedding 0.6B | 1024 | **0.984375** | 31/32 | 19.31 | 2.371 |

Recall uses the existing metric: expected source spans overlapping the first five distinct retrieved spans. It is evidence retrieval recall, not generated-answer correctness. Nomic missed half of `reasoning-08`'s expected evidence; Qwen missed half of `module-01`'s. All per-question rankings and scores are in [results.json](../validation/milestone-6/results.json).

## Contract correction and selected-model validation

The shared v1 JSON schema permitted empty strings, while our application validator rejected them. This affected empty classes/methods rather than JSON syntax. Added `minLength: 1` to the schema, explicit nonempty not-applicable instructions, and clearer factory/copy/stub guidance. Prompt and summary schema versions both advanced to 2, invalidating old summary cache keys.

The selected Qwen3.5 model was then run against **all 40 symbols again**, rather than only repairing failures: **40/40 valid JSON and 40/40 accepted summaries**, 73.3 generation t/s, 2.162 seconds/symbol. The independently reviewed score was **3.925/5**; the contract correction did not eliminate semantic errors. This second run is separately recorded in [selected-validation.json](../validation/milestone-6/selected-validation.json) and is not mixed into the controlled v1 comparison. Remaining copy/mutation contradictions, speculative SQL behavior, and closure/input confusion must still be caught by the later evidence verification pipeline.

## Applied configuration and final gate

Updated `cfl.toml`, `.env.example`, the local `.env` if present, and `scripts/pull_models.sh`. Database metadata now contains the resolved generator/embedder tags and digests, `num_ctx`, generation options, and prompt/schema versions. The embedding dimension remains 768; no embeddings were cleared. The final models remain resident.

With the [matching official tokenizer](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/tokenizer.json), final `cfl doctor` **passed**:

- Generator context: **8192**, both models fully on GPU.
- Combined `/api/ps` model VRAM: **6.052 GB**.
- Peak total GPU usage: **7996 MiB / 8.384 GB**, below 10 GB.
- Prefill: **3666.7 t/s** on **7628 uncached tokens**; generation **72.7 t/s**.
- No prompt truncation warning; embedding dimension **768**; PostgreSQL extensions/schema healthy.

The full snapshot and tokenizer checksum are in [final-doctor.json](../validation/milestone-6/final-doctor.json). Doctor now fails for an absent selected model, CPU spill by either model, a generator context mismatch, or total model VRAM reaching the configured limit. This also removes the previous loose-tag check that could accept a different model size.

## Projection limits and reproduction

M7's `cfl build --estimate` is still a stub. M6's reported estimates therefore use measured sample mean wall time × parsed production symbol count, and do **not** claim an end-to-end build measurement. The starting repository had 334 production symbols; the selected-model regression run counted 336 after preflight changes, giving **0.202 hours** of projected summary generation. These short fixture symbols underrepresent large functions, dependency context, retries, embeddings, and database costs. A standardized 10,000-symbol projection is about 6 hours for the selected model, so a 3-hour target for arbitrarily large repositories is not promised. Recalculate with M7's full estimator when available.

```sh
# Verify/update candidates.json from official sources before pulling changed tags.
ollama pull qwen2.5-coder:7b
ollama pull qwen3.5:9b
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama pull qwen3-embedding:0.6b

# Replay the controlled comparison's archived contract.
.venv/bin/python scripts/bakeoff.py \
  --contract docs/validation/milestone-6/prompt-v1.json \
  --gen qwen2.5-coder:7b --gen qwen3.5:9b --gen qwen3:8b \
  --embed nomic-embed-text --embed qwen3-embedding:0.6b \
  --out .cfl/bakeoff/replayed.json

# Current selected-model contract, with exact counting.
.venv/bin/python scripts/bakeoff.py --gen qwen3.5:9b \
  --tokenizer qwen3.5:9b=.cfl/tokenizers/qwen3.5-9b/tokenizer.json \
  --out .cfl/bakeoff/current.json
.venv/bin/cfl doctor
.venv/bin/pytest -q
```

The script checkpoints after each summary and restores previous model residency even on failure. It does not mutate the production index or reset dimensions. Model tags are mutable: compare resolved digests when reproducing results. KV quantization remains an opt-in server setting; M6's measurements use the existing f16 runtime, and the earlier Qwen2.5 cache comparison does not imply that q8_0 has been quality-validated for the newly selected generator.

## Repository verification

Final regression suite: **268 passed**, with 16 existing pathspec deprecation warnings. Ruff passed for production code, scripts, and maintained tests (excluding the existing parser fixtures and legacy `test_db.py` lint findings); `git diff --check` and pull-script shell syntax passed. Artifact checks verified three 40-symbol generator records, two 32-question embedding records, matching review/source fingerprints, and the selected model's 100% contract validity.
