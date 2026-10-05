# Decision: 9B generation with independent source validation

Date: 2026-10-04. User selected a different checkpoint for the validation pass.

Keep `qwen3.5:9b` for ingest generation and RAG answers, with KV quantization off by default. Use the installed `qwen2.5-coder:7b` as a separate, configurable verifier. Both use the configured 8,192-token context. The verifier uses fresh source-only prompts, not a conversation with the draft author. Generation is sequential: unload the generator, run the verifier, unload the verifier, restore the generator. The verifier must have a distinct resolved model digest and full GPU residency.

## Build and publication

1. Parse source and construct AST call graphs as before. The model never creates authoritative graph edges, source spans or diagrams.
2. Generate symbol explanations, file explanations, and proposed major-feature entry points with 9B. Detected application entry points and routed handlers are mandatory feature candidates; the model can add more. Feature membership comes from bounded AST traversals; feature descriptions discuss the member implementations. Symbol review includes two callee levels and containing classes, with dependency hashes in the certificate. Large files are divided into contiguous, explicitly scoped source parts rather than silently truncated.
3. Keep all generated material in `knowledge_drafts`. Existing symbol summaries are drafts, not validated facts.
4. Unload 9B and validate with Coder7 using raw implementations and necessary callee/class context. Review small claim batches with exact implementation quotes. Check all claim indices, quote identity, source hashes and model identity. Quotes consisting only of documentation cannot establish behavior. Missing/invalid/truncated reviews cannot approve their claims.
5. Publish supported claims only. Remove rejected claims and their original citation tags; render citations from the reviewer's exact source excerpts. Mark partial artifacts visibly. Reject artifacts with no accepted claims. Keep drafts and review diagnostics for inspection/resume.
6. Build lexical and dense retrieval using current published knowledge and raw source. Pending, rejected, stale, incompatible-verifier and legacy unvalidated prose is excluded. Feature cards and file parts remain explicitly scoped partial knowledge.

`cfl build <repo>` runs both passes. `cfl knowledge --json` reports publication status. `cfl knowledge --validate` resumes validation and rebuilds retrieval views; `--retry-rejected` rechecks semantic rejections. Unchanged feature selection and semantic reviews are reused, with current deterministic checks and physical source checks reapplied to cached reviews. Transport/schema failures can be retried on later validation runs. One database per repository remains required.

## Answer filtering

All model-authored answers, including brief explanations, get source-only review before display. Buffer streamed drafts; callbacks receive only filtered output. Valid citation locations alone do not make a claim correct. A valid tag mixed with an unsupported tag causes abstention instead of leaving an unsupported sentence behind.

Outputs carry `status`:

- `complete`: all displayed claims passed the current checks and the reviewer detected no gap in the supplied scope. This is not a mathematical correctness guarantee.
- `partial`: supported material is available, but claims were removed, the evidence is incomplete, an output limit was reached, or a static traversal is bounded. Text explicitly says the answer may be incomplete.
- `abstained`: no reliable cited explanation passed, source changed, necessary evidence exceeded context, or validation was unavailable. Text explicitly says a reliable answer cannot be provided.

Only complete semantic answers are reused from the answer cache. Deterministic scoped graph results may also be cached. Certificates depend on source identity/ranges, model digests, options, guard version and index revision. Source is checked against disk before trusting a cached result. Changing the verifier checkpoint invalidates its knowledge certificates and answer caches.

Deterministic checks supplement model review for observed failure modes: invented identifiers or exception types, empty exception/effect guarantees, return-only delivery stubs, dictionary-copy mutation claims, inverted short-circuit call conditions, class/method scope confusion, model-authored diagrams, and unsupported language, concurrency, performance or security assertions. Named exception conditions require explicit raises or re-raises in supplied source. Unverified library failure conditions are omitted. Function scope is retained when filtering removes headings. Empty placeholders, self-assurance phrases and disclaimers do not count as factual answers. Reviewer reasons are diagnostic model output; they are not published as new factual explanation.

## Limits

Two model checkpoints can still share errors. Exact quotes establish provenance, not semantic entailment. Both acceptance and rejection must be assessed through adversarial tests and source-based live evaluation. The filter deliberately favors explicit partial output or abstention over unsupported confidence. Large evidence sets that cannot fit the verifier are rejected rather than truncated; complete knowledge coverage is not promised. Online review adds model-switching latency even after the offline knowledge build.

This extends the original execution plan's verification hooks and invalidation rules. The old citation-only acceptance and brief-without-review behavior are deliberately replaced by the user's stricter accuracy requirement.
