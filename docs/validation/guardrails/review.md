# 9B generation and independent validation

Implemented generation/RAG with `qwen3.5:9b` and a separate source-only review pass with `qwen2.5-coder:7b`. The generator is unloaded during review and restored afterwards. KV quantization remains off by default.

The build generates symbol, file and feature drafts, then publishes supported claims only. Routed handlers are mandatory feature candidates. Symbol reviews include two callee levels and containing classes; their source hashes become certificate dependencies. Validation resumes from persisted drafts and certificates, reapplies current deterministic checks, and rebuilds retrieval views. Legacy, rejected and stale prose cannot become answer context.

Model answers are buffered until review. Partial answers explicitly warn that they may be incomplete; failures explicitly abstain. Only complete semantic answers can be cached. AST diagrams and call-site tables remain available when narrative validation fails.

## Verification

- Full regression suite: **324 passed**. The final small exclusions also passed the 22 guardrail tests. Existing pathspec deprecation warnings remain.
- Ruff and `git diff --check`: passed.
- Live GPU validation: all **48** evaluation questions, with affected cases rerun after corrections. All final answer text and published knowledge were inspected against the fixture source.
- Six adversarial cases—delivery stub, inverted short circuit, input mutation, no-exceptions assertion, concurrency guarantee and security guarantee—each produced an explicit abstention.
- Final state: 9B at 8,192 context and Nomic at 2,048, both fully on GPU. KV quantization off; no temporary validation databases remain.

| Final answers | Count |
| --- | ---: |
| Partial | 28 |
| Abstained | 20 |
| Complete | 0 |
| Source review: adequate for the asked scope, despite conservative partial status | 19 |
| Source review: useful but materially incomplete | 9 |

The eight deterministic caller queries matched the indexed source call sites. Flow diagrams/tables retained bounded AST path evidence, including intermediate calls, even when prose was reduced or withheld.

The knowledge build produced **63 drafts**: 40 symbols, 20 files and three features. **Four complete and 42 partial cards** are published; **17 drafts** are rejected. The features cover the CLI and the two routed task handlers. A partial feature card does not establish complete feature coverage.

## Corrections from live inspection

The second model accepted some incorrect or ambiguous draft claims. Added checks for documentation-only proof, dictionary copies versus input mutation, class/method scope, dictionary keys versus attributes, named exception conditions, short-circuit call guards, unsupported language/security/concurrency/performance claims, completed email delivery, and historical issuance assertions. Removed empty exception/effect guarantees, placeholders, citation-only fragments, model self-assurance, and unnamed function claims whose scope was lost during filtering. Reviewer diagnostics remain separate from factual output.

No remaining incorrect factual claims were observed in the final reviewed answers. **This is an agent source review of one fixture, not a general correctness guarantee.** The filter is deliberately conservative: 20 questions still have no certified narrative answer, including questions the source itself could answer. Accuracy filtering has reduced answer coverage; a partial status is not proof that the explanation is sufficient.

[Per-question source review](source-review.csv), [answers](answers.json), [published knowledge](published.json), [summary](summary.json), [adversarial cases](adversarial.json), and [model/source verification](verification.json) contain the evidence. `knowledge.json` and `checkpoint.json` also contain untrusted drafts and review diagnostics; they are not published answer content. Build timing in `build.json` covers the final checkpoint refresh/index rebuild, not the entire ingest and review run.
