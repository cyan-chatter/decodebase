# Milestone 5 verification

Implementation verified on 2026-10-04 against `Execution_Order.md` M5.1–M5.4
and `plan.md` Section 15. The subsequent user instruction replaced the human-review
step with agent validation. All 48 expectations are now source-reviewed with agent
provenance; see `milestone-5-agent-review.md` for the independent review and output gaps.

| Requirement | Implementation and evidence |
| --- | --- |
| M5.1 static taskboard fixture | `eval/sample_repo/`: 20 Python files, 40 parsed symbols; authentication, repository, routes, retry decorator, aliases, duplicate save/process names, inheritance, recursion, tests, and the requested pipeline chain. |
| M5.2 question suite | `eval/questions.yaml`: 48 questions, eight of each of six types; expected symbol IDs, caller sets, ordered paths, and reasoning keywords. All 48 have `verified: true` with `review.kind: agent` and source rationale. |
| M5.2 review rubric | `eval/rubric.md`, `eval/human_scores.csv`; 1–5 scoring and review instructions. |
| M5.3 lexical view | `cfl/pipeline/indexer.py`: split identifiers, source/summary fields, bounded batches, fresh view marker, FTS/trigram rank merge and current source spans. Scan builds this view after parsing and resolution. |
| M5.4 metrics | `cfl/eval/metrics.py`: first-five-distinct overlap recall, contained citation validity, caller-set equality, actual trace token cost. |
| M5.4 runner | `cfl/eval/runner.py`: graph callers and paths, lexical retrieval, atomic persistence in `eval_runs`, suite/config snapshots, evidence spans, review filtering, comparisons. Full-answer modes explicitly remain unavailable. |
| M5.4 CLI | `cfl eval --retrieval-only`, `--config`, `--structural-only`, `--verified-only`, `--compare RUN_A RUN_B`, and `--json`. |

## External CLI verification

Ran migrations and the following commands against a disposable PostgreSQL database,
preserving the existing repository index:

```sh
cfl scan eval/sample_repo
cfl eval --retrieval-only
cfl eval --config eval/configs/lexical.toml --json
cfl eval --structural-only --json
cfl eval --compare RUN_A RUN_B --json
```

No Ollama server, models, embeddings, summaries, or GPU calls were used. Integration
tests prohibit constructing `OllamaClient` while running this workflow.

| Type | Questions | Lexical AnswerRecall@5 | Enriched EvidenceRecall@5 | Graph exactness |
| --- | ---: | ---: | ---: | ---: |
| Symbol | 8 | 1.0000 | 1.0000 | — |
| Module | 8 | 1.0000 | 1.0000 | — |
| Behavioral | 8 | 0.9375 | 0.9375 | — |
| Traversal | 8 | 0.8146 | 1.0000 | 1.0000 ordered paths |
| Structural | 8 | 1.0000 | 1.0000 | 1.0000 caller sets |
| Reasoning evidence | 8 | 0.8750 | 0.8750 | — |

These are retrieval/graph measurements against agent-reviewed expectations.
Generated answer-quality scores await the answer pipeline. The fixture does not yet have synthesized
module pages, and lexical retrieval is the baseline for subsequent milestones.

Local CLI output, a complete persisted-run snapshot, and comparison output are
saved under `.cfl/m5-validation/`. The disposable database was removed after
verification; those smoke-test run IDs are available in the saved JSON, rather
than the user's main database.

## Regression verification

- Full suite after agent review: **200 passed**.
- Standalone taskboard test: **1 passed**.
- Ruff checks pass for production code, the new fixture, and maintained tests;
  existing resolver fixture and `test_db.py` lint findings were excluded.
- Formatting checks pass for new/changed modules other than the existing mixed-format
  database module. `git diff --check` passes.
- Tests cover malformed citations, duplicate span handling, trace costs, safe lexical
  queries, deterministic fusion, no summaries/LLM dependency, atomic rollback,
  config paths, draft validation, verified-only filtering, comparison deltas,
  missing expected symbols, printed CLI metrics, and scan stage ordering.

The taskboard caller-set check exposed `super().save()` falling back to ambiguous
name lookup. Resolver now resolves zero-argument `super()` methods against indexed
base classes; its version was bumped to `resolver_v3` to invalidate older graph
caches. The fixture's expected caller set was retained.

## Expectation review

At the user's request, Codex reviewed every expectation independently against
source, ran 25 direct fixture behavior checks, and checked all expected ranges
with the standard-library AST. Review provenance is explicitly agent, and all
48 expectations are selectable through `--verified-only`. Six questions still
have partial or missing pure lexical evidence. The subsequent path-evidence fix
backfills the four affected traversal questions with complete source blocks,
leaving two non-traversal retrieval gaps. All graph checks pass. Detailed
per-question findings are in `milestone-5-agent-review.md`.

## Traversal source evidence follow-up

The retrieval engine now hydrates every symbol on requested call paths from the
index. It uses question text without expected IDs or traversal oracle hints.
The runner persists enriched spans and path-completeness statuses alongside the
unchanged lexical baseline. See `path-evidence.md` for the verified paths and
regression results.

After this follow-up, the full regression suite passes **213 tests**; Ruff, changed-file
formatting, and whitespace checks pass.
