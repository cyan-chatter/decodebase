# Traversal source evidence verification

Verified on 2026-10-04 after the user requested that intermediate source evidence be retained.

The lexical ranker can miss a function whose name does not appear in the question. The new `retrieve_evidence` engine resolves endpoints from question text, finds a bounded call path, and fetches each source block directly from the symbol index. It returns blocks in call-path order with incoming call-site and confidence evidence. Evaluation expectations are used only for scoring, never endpoint selection.

Named endpoints use exact IDs, qualified names, or module-qualified references. Prose endpoints expand generic action terms and rank reachable candidates by name/docstring support, then path depth; tied candidates are reported as ambiguous. These are deterministic inferred endpoints, not a general natural-language understanding guarantee.

| Question | Pure lexical recall | Enriched source recall | Endpoint status | Source path |
| --- | ---: | ---: | --- | --- |
| traversal-01 | 0.500 | 1.000 | explicit_path | `pipeline.py::run_pipeline` → `pipeline.py::transform` → `pipeline.py::write_to_db` → `db/repository.py::Repository.save` |
| traversal-02 | 0.600 | 1.000 | inferred_path | `cli.py::main` → `pipeline.py::run_pipeline` → `pipeline.py::transform` → `pipeline.py::write_to_db` → `db/repository.py::Repository.save` |
| traversal-03 | 0.750 | 1.000 | inferred_path | `api/routes.py::create_handler` → `services/tasks.py::create_task` → `services/notifications.py::notify` → `services/notifications.py::send_email` |
| traversal-04 | 0.667 | 1.000 | inferred_path | `api/routes.py::complete_handler` → `services/tasks.py::complete_task` → `db/repository.py::Repository.get` |
| traversal-05 | 1.000 | 1.000 | inferred_path | `auth/tokens.py::validate_token` → `auth/tokens.py::issue_token` |
| traversal-06 | 1.000 | 1.000 | inferred_path | `services/tasks.py::process` → `services/tasks.py::create_task` → `db/repository.py::Repository.save` |
| traversal-07 | 1.000 | 1.000 | explicit_path | `trees.py::is_even` → `trees.py::is_odd` |
| traversal-08 | 1.000 | 1.000 | explicit_path | `trees.py::is_odd` → `trees.py::is_even` |

All eight requests return complete indexed source spans. The raw lexical scores remain unchanged. Each path block includes raw code, signature, docstring, file, and inclusive line range, plus the incoming edge for intermediate and endpoint symbols.

Required paths are never cut to satisfy the result-count limit. Longer paths set `requires_batching`, so the future generation pipeline must budget/split evidence before building prompts. Unresolved or ambiguous endpoints, depth/confidence-excluded paths, and missing source hydration never claim complete evidence. One shortest resolved path is returned; this does not enumerate every possible runtime branch.

The CLI scan/eval workflow passed against a disposable PostgreSQL database. Saved output is in `.cfl/m5-validation/`; no Ollama calls were made. Regression tests compare every returned path block with actual fixture source and check call sites, module-qualified references, no lexical hits, ambiguity, unavailable source, depth/confidence limits, cycles, self paths, long paths, and unchanged lexical ordering.

Final verification: **213 tests passed**. Production code, maintained tests, and the evaluation fixture pass Ruff checks; changed-file formatting and `git diff --check` pass.

The remaining non-traversal gaps are `behavioral-02` (missing email sink) and `reasoning-06` (missing retry wrapper). This change fixes path evidence; those general lexical ranking gaps remain visible. Generated explanations are not implemented at milestone 5.
