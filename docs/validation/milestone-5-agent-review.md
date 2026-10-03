# Agent validation of milestone 5 outputs

The user explicitly requested agent validation in place of the original human-review gate.
Codex reviewed all 48 expectations against the fixture source, retained all expected
IDs/caller sets/paths/keywords, and recorded `verified: true` with `review.kind: agent`.
This is agent review; no human sign-off is claimed.

The review includes 25 direct fixture behavior checks and independent standard-library
AST range checks for all 48 expected evidence entries. Runtime checks confirm token
and password behavior, repository writes, inheritance, authenticated handlers, pipeline
normalization, closed-connection errors, configuration rejection, retry counts and
exception types, recursion, and JSON exports. None of these behavior checks depend
on the application resolver or retrieval scorer.

Module-path indexing and removal of grammatical query filler fixed the module
evidence misses. Every module question now retrieves its expected evidence.

**Final retrieval:** 42 complete, 5 partial, 1 missing. All eight caller sets and
all eight ordered graph paths are correct. Remaining retrieval misses are retained
in the report; verification flags certify the reviewed expectations, not perfect
application output. Milestone 5 returns retrieval and graph results; generated
answer quality is evaluated when the answer pipeline exists.

## Per-question review

| ID | Recall before | Recall after | Output | Independent source review |
| --- | ---: | ---: | --- | --- |
| symbol-01 | 1.000 | 1.000 | complete | validate_token partitions the token, compares it to a newly issued HMAC token, raises TokenError on mismatch, and returns user_id. |
| symbol-02 | 1.000 | 1.000 | complete | issue_token computes HMAC-SHA256 with the secret and returns user_id plus the hexadecimal digest. |
| symbol-03 | 1.000 | 1.000 | complete | hash_password invokes PBKDF2-HMAC-SHA256 with password, salt, and 100000 iterations, returning a hex digest. |
| symbol-04 | 1.000 | 1.000 | complete | verify_password derives a digest with hash_password and compares it to expected using hmac.compare_digest. |
| symbol-05 | 1.000 | 1.000 | complete | Repository.save validates task/id, executes a connection write, and returns the task. |
| symbol-06 | 1.000 | 1.000 | complete | Repository.get calls connection.execute with the identifier and returns a record or None. |
| symbol-07 | 1.000 | 1.000 | complete | Transform converts id to str, strips title whitespace, and forwards the normalized record to write_to_db. |
| symbol-08 | 1.000 | 1.000 | complete | walk_tree collects the root name, recursively walks each child, and extends the preorder names list. |
| module-01 | 0.000 | 1.000 | complete | auth contains signed token verification and password digest verification; their dependencies provide issuance and hashing. |
| module-02 | 1.000 | 1.000 | complete | Connection.execute implements read/write and closed-state protection; Connection.close marks it closed. |
| module-03 | 1.000 | 1.000 | complete | Repository.save and get provide persistence and lookup; AuditedRepository adds a write marker through inheritance. |
| module-04 | 0.500 | 1.000 | complete | create_task saves and notifies; complete_task loads, checks existence, marks done, and saves. |
| module-05 | 0.000 | 1.000 | complete | notify joins notification formatting and email-sink delivery; format_notification constructs the title message. |
| module-06 | 1.000 | 1.000 | complete | with_retries validates the attempt bound and builds a wrapper that catches only RuntimeError. |
| module-07 | 1.000 | 1.000 | complete | load_config reads the file and delegates to parse_config for JSON-object and secret checks. |
| module-08 | 0.000 | 1.000 | complete | The two decorated handlers authenticate through an import alias, then create or complete a task. |
| behavioral-01 | 1.000 | 1.000 | complete | The nested wrapped function contains the bounded for loop, RuntimeError catch, and final re-raise. |
| behavioral-02 | 0.500 | 0.500 | partial | format_notification creates the message; send_email is the fixture sink and returns that message without real email delivery. |
| behavioral-03 | 1.000 | 1.000 | complete | exports.save opens a path in write mode and calls json.dump; this is distinct from Repository.save. |
| behavioral-04 | 1.000 | 1.000 | complete | exports.process calls json.dumps with sort_keys=True; this is distinct from services.tasks.process. |
| behavioral-05 | 1.000 | 1.000 | complete | services.tasks.process uses a list comprehension to invoke create_task for each title. |
| behavioral-06 | 1.000 | 1.000 | complete | Connection.close is the explicit operation setting self.closed=True. |
| behavioral-07 | 1.000 | 1.000 | complete | cli.main calls parse_config, constructs Connection and Repository, and delegates records to run_pipeline. |
| behavioral-08 | 1.000 | 1.000 | complete | is_even calls is_odd and is_odd calls is_even, with complementary zero base cases. |
| traversal-01 | 0.500 | 0.500 | partial; graph correct | Source calls establish run_pipeline -> transform -> write_to_db -> Repository.save. |
| traversal-02 | 0.400 | 0.600 | partial; graph correct | main delegates to run_pipeline, followed by transform, write_to_db, and Repository.save. |
| traversal-03 | 0.750 | 0.750 | partial; graph correct | create_handler delegates to create_task, which calls notify; notify calls send_email. |
| traversal-04 | 0.667 | 0.667 | partial; graph correct | complete_handler calls complete_task, which retrieves through Repository.get. |
| traversal-05 | 1.000 | 1.000 | complete; graph correct | validate_token directly invokes issue_token when its separator exists. |
| traversal-06 | 0.667 | 1.000 | complete; graph correct | services.tasks.process invokes create_task, whose repo parameter is annotated Repository and whose body calls repo.save. |
| traversal-07 | 1.000 | 1.000 | complete; graph correct | is_even directly invokes is_odd for nonzero input. |
| traversal-08 | 1.000 | 1.000 | complete; graph correct | is_odd directly invokes is_even for nonzero input. |
| structural-01 | 1.000 | 1.000 | complete; graph correct | The only indexed direct callers are create_handler and complete_handler through the authenticate alias. |
| structural-02 | 1.000 | 1.000 | complete; graph correct | The only indexed direct caller is validate_token; no fixture test calls issue_token. |
| structural-03 | 1.000 | 1.000 | complete; graph correct | The only direct callers are cli.main and utils.config.load_config. |
| structural-04 | 1.000 | 1.000 | complete; graph correct | Direct calls come from create_task, complete_task, write_to_db, and AuditedRepository.save through super(); exports.save is unrelated. |
| structural-05 | 1.000 | 1.000 | complete; graph correct | The only direct caller is complete_task; its repo annotation identifies Repository. |
| structural-06 | 1.000 | 1.000 | complete; graph correct | The only direct caller is create_task; tests invoke create_task but not notify directly. |
| structural-07 | 1.000 | 1.000 | complete; graph correct | The only direct caller is notify, where its result becomes the send_email argument. |
| structural-08 | 1.000 | 1.000 | complete; graph correct | The only direct caller is run_pipeline in its list comprehension. |
| reasoning-01 | 1.000 | 1.000 | complete | The task-is-None guard short-circuits before task.get and raises ValueError; an absent id triggers the same error. |
| reasoning-02 | 1.000 | 1.000 | complete | close sets the closed flag; execute checks it first and raises RuntimeError before reading or writing. |
| reasoning-03 | 1.000 | 1.000 | complete | validate_token recomputes the HMAC token and rejects a mismatch with TokenError. |
| reasoning-04 | 1.000 | 1.000 | complete | parse_config rejects non-object or false/missing secret configuration with ValueError. |
| reasoning-05 | 1.000 | 1.000 | complete | with_retries checks attempts<1 and raises ValueError before creating the decorator. |
| reasoning-06 | 0.000 | 0.000 | missing | The wrapper catches RuntimeError only, so a ValueError propagates immediately without another attempt. |
| reasoning-07 | 1.000 | 1.000 | complete | complete_task checks the None result of Repository.get and raises KeyError before mutation or saving. |
| reasoning-08 | 0.500 | 1.000 | complete | exports.save uses file I/O and JSON serialization, whereas Repository.save writes through its connection; shared names do not imply calls. |

## Pure lexical retrieval gaps

- `behavioral-02`: email-sink evidence is missing from the first five results.
- `traversal-01` through `traversal-04`: lexical results omit some intermediate
  path evidence. The subsequent retrieval fix now hydrates all intermediate source
  blocks for these requests; enriched evidence recall is 1.0 for all eight traversal
  questions. See `path-evidence.md`.
- `reasoning-06`: lexical search misses the wrapper that catches RuntimeError
  but lets ValueError propagate. Source review and direct execution confirm
  that ValueError is raised after one attempt.

The expected evidence was not weakened to raise retrieval scores. Module names
were added generically; no question-specific answers or retrieval rules were added.

## Evidence

- `eval/questions.yaml` contains the review rationale and fixture fingerprint for every question.
- `.cfl/agent-review/questions.json` contains reviewed outputs and missing evidence IDs.
- `.cfl/agent-review/runtime-checks.json` contains the 25 passing behavior checks.
- `.cfl/agent-review/baseline.json` preserves the pre-review app output.
- `.cfl/m5-validation/run.json` contains the latest app output from a disposable database, including enriched evidence scores.
- `cfl eval --verified-only` now selects all 48 agent-reviewed expectations.
  CLI and persisted configurations explicitly identify the review method as agent.

## Regression result

Full suite after the review and namespace fix: **200 passed**. Ruff checks and
changed-file formatting checks pass; `git diff --check` passes.
