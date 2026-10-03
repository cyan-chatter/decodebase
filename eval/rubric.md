# Expectation review and answer rubric

The suite has 48 questions, eight of each type in plan Section 15. At the user's
explicit request, **agent source validation replaces the original human-review
gate** for this milestone. All expectations are reviewed by Codex and carry
`verified: true` plus `review.kind: agent`, a reviewer, source rationale, timestamp,
and fixture fingerprint. This records agent review and does not claim human sign-off.

Verification applies to the **expected evidence**, not a guarantee that application
retrieval or answers are perfect. The independent review and remaining app-output
misses are documented in `docs/validation/milestone-5-agent-review.md`.

For future expectation changes, inspect every symbol ID, caller set, ordered path,
and reasoning keyword against fixture source, including omissions. Record whether
the reviewer is an agent or human using `review.kind`. Update the rationale and
fingerprint after relevant fixture changes. Symbol IDs use relative `path::qualname`;
runtime spans come from the scanned database. Caller sets use the configured
confidence threshold (0.6), including inherited `super()` calls. Traversal
expectations specify call-graph paths rather than runtime execution traces.

Milestone 5 measures graph structure and lexical evidence retrieval. Generated
answer/document quality will use the following rubric when answer modes exist:

| Score | Meaning |
| --- | --- |
| 1 | Incorrect or unsupported; fails the question. |
| 2 | Major omissions or factual errors; evidence is weak. |
| 3 | Mostly correct; important details or citations are missing. |
| 4 | Correct, clear, grounded; minor omissions only. |
| 5 | Complete, precise, grounded in valid citations; explains relevant limits. |

`human_scores.csv` remains an optional template for future human answer/document
reviews. It is not a gate for the agent-reviewed milestone 5 baseline. Agent
reviews must identify themselves as agents instead of being entered as human
sign-off. `--verified-only` selects reviewed expectations; questions without review
remain provisional. The CLI and stored configurations identify the review method.
