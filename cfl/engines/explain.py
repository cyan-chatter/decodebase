from __future__ import annotations

import json

from cfl.core import db
from cfl.core.budget import ChatCounter, PromptPart, assemble, batch_by_budget, split_oversized
from cfl.core.errors import CflError
from cfl.core.hashing import fingerprint
from cfl.core.memory import KnowledgeMemory
from cfl.engines.ask import block_for
from cfl.engines.graph_queries import callees, callers, resolve_symbol
from cfl.engines.guardrails import GUARD_VERSION, Guarded, guard_answer, source_blocks
from cfl.engines.verify import verify_answer
from cfl.parser.python_adapter import PythonAdapter
from cfl.pipeline.knowledge import trusted_rows
from cfl.prompts.prompts import PROMPT_EXPLAIN_FUNCTION, PROMPT_VERSION


def explain(conn, client, settings, query, detail="brief", *, on_token=None):
    if detail not in {"brief", "detailed"}:
        raise ValueError("detail must be brief or detailed")
    counter = ChatCounter(client.counter)
    resolved = resolve_symbol(conn, query)
    symbol = trusted_rows(conn, [db.get_symbol(conn, resolved.id)])[0]
    incoming = callers(conn, resolved, 1, settings.edge_conf_threshold)
    outgoing = callees(conn, resolved, 1, settings.edge_conf_threshold)
    ids = list(dict.fromkeys([symbol["id"]] + [e["id"] for e in incoming + outgoing if e["id"]]))
    neighbors = trusted_rows(conn, db.get_symbols(conn, ids))
    blocks = []
    methods = (
        trusted_rows(
            conn,
            [
                s
                for s in db.get_all_symbols(conn)
                if s["parent_id"] == symbol["id"] and s["kind"] == "method"
            ],
        )
        if symbol["kind"] == "class"
        else []
    )
    # Classes use their structural rollup; ordinary targets retain source code.
    for s in neighbors:
        row = {**s, "kind": "function"} if s["id"] == symbol["id"] else s
        blocks.extend(block_for(row))
    target = next(b for b in blocks if b["id"] == symbol["id"])
    tag = f"[{symbol['file_path']}:{symbol['start_line']}-{symbol['end_line']}]"
    cached = False
    tags = {m["name"]: m for m in client.tags()}
    verifier_tag = (
        settings.verifier_model
        if ":" in settings.verifier_model
        else settings.verifier_model + ":latest"
    )
    verifier_digest = tags.get(verifier_tag, {}).get("digest", "unavailable")
    db.set_meta(conn, "verifier_model_digest", verifier_digest)
    if detail == "brief":
        text = (
            (symbol.get("summary_short") or symbol.get("docstring") or symbol["qualname"])
            + "\n\n"
            + ((symbol.get("summary_json") or {}).get("purpose") or "")
            + " "
            + tag
        )
    else:
        memory = KnowledgeMemory(conn, client, settings)
        key = fingerprint(
            {
                "version": 3,
                "guard_version": GUARD_VERSION,
                "verifier": settings.verifier_model,
                "verifier_digest": verifier_digest,
                "num_predict_review": settings.num_predict_review,
                "code_hash": symbol["code_hash"],
                "ctx_hash": symbol["ctx_hash"],
                "methods": [(m["id"], m["code_hash"], m["summary_short"]) for m in methods],
                "neighbors": [
                    (s["id"], s["code_hash"], s["start_line"], s["end_line"], s["summary_short"])
                    for s in sorted(neighbors, key=lambda s: s["id"])
                ],
                "model": memory.digests[settings.gen_model],
                "options": memory.options,
                "reserve": settings.num_predict_detailed,
                "prompt_version": PROMPT_VERSION,
            }
        )
        if symbol.get("summary_long") and symbol.get("summary_long_hash") == key:
            text = symbol["summary_long"]
            cached = True
        else:
            parts = [
                PromptPart(
                    "target",
                    "Explain the TARGET symbol "
                    + symbol["qualname"]
                    + " ("
                    + symbol["id"]
                    + "). Use its implementation below. Caller/callee summaries are supporting context, not the target.\n[TARGET]\n"
                    + target["text"],
                    1,
                    False,
                ),
                PromptPart("summary", json.dumps(symbol.get("summary_json")), 2, True),
            ]
            parts += [
                PromptPart(
                    s["id"],
                    f"[NEIGHBOR] [{s['file_path']}:{s['start_line']}-{s['end_line']}] {s['qualname']}: {s['summary_short'] or 'No summary'}",
                    3,
                    True,
                )
                for s in neighbors
                if s["id"] != symbol["id"]
            ]
            try:
                assembled = assemble(
                    PROMPT_EXPLAIN_FUNCTION,
                    parts
                    + [
                        PromptPart(
                            "instruction",
                            "Now explain only "
                            + symbol["qualname"]
                            + " using the TARGET source above. Include its exact source citation "
                            + tag
                            + ".",
                            1,
                            False,
                        )
                    ],
                    num_ctx=settings.num_ctx,
                    num_predict=settings.num_predict_detailed,
                    overhead=settings.template_overhead,
                    counter=counter,
                )
                supplied = {p.name for p in assembled.parts}
                blocks = [b for b in blocks if b["id"] == symbol["id"] or b["id"] in supplied]
                texts = [assembled.prompt]
            except Exception as exc:
                from cfl.core.errors import BudgetExceeded

                if not isinstance(exc, BudgetExceeded):
                    raise
                budget = (
                    settings.num_ctx
                    - settings.template_overhead
                    - 350
                    - counter.count(PROMPT_EXPLAIN_FUNCTION + tag)
                    - 32
                )
                blocks = [target]
                chunks = split_oversized(symbol, PythonAdapter(), budget, counter=counter)
                texts = [
                    PROMPT_EXPLAIN_FUNCTION + "\n" + tag + "\n" + chunk.text for chunk in chunks
                ]
            outputs = []
            for prompt in texts:
                pieces = []
                for chunk in client.chat(
                    [{"role": "user", "content": prompt}],
                    num_predict=settings.num_predict_detailed if len(texts) == 1 else 350,
                    task="explain",
                    stream=True,
                ):
                    pieces.append(chunk)  # noqa: PERF402 - buffered until validation
                outputs.append("".join(pieces))
            if len(outputs) > 1:
                budget = (
                    settings.num_ctx
                    - settings.template_overhead
                    - settings.num_predict_detailed
                    - counter.count(PROMPT_EXPLAIN_FUNCTION + tag)
                    - 32
                )
                while len(outputs) > 1:
                    batches = batch_by_budget(outputs, lambda s: counter.count(s) + 8, budget)
                    if all(len(b) == 1 for b in batches):
                        raise BudgetExceeded("Explanation partials cannot be merged")
                    outputs = [
                        client.chat(
                            [
                                {
                                    "role": "user",
                                    "content": PROMPT_EXPLAIN_FUNCTION
                                    + "\nCombine all partial explanations in source order.\n"
                                    + tag
                                    + "\n"
                                    + "\n".join(b),
                                }
                            ],
                            num_predict=settings.num_predict_detailed,
                            task="explain_combine",
                        ).text
                        for b in batches
                    ]
            text = outputs[0]
    if cached:
        try:
            source_blocks(conn, blocks)
            citation_check = verify_answer(text, blocks, conn)
            checked = Guarded(
                citation_check.text,
                "complete" if citation_check.sufficient else "abstained",
                citation_check.citations,
            )
        except (CflError, ValueError, OSError):
            cached = False
    if not cached:
        checked = guard_answer(
            conn,
            client,
            settings,
            text,
            blocks,
            "Explain " + symbol["qualname"],
            incomplete=["Supporting knowledge has not been fully validated."]
            if symbol.get("knowledge_status") != "complete"
            else [],
        )
        if detail == "detailed" and checked.status == "complete":
            with conn.transaction():
                hashes = db.lock_symbol_hashes(conn, [symbol["id"]])
                if hashes.get(symbol["id"]) == symbol["code_hash"]:
                    db.set_summary_long(conn, symbol["id"], checked.text, key)
    if on_token:
        on_token(checked.text)
    return {
        "raw_text": checked.text,
        "raw_citations": checked.citations,
        "text": checked.text,
        "status": checked.status,
        "route": "explain",
        "cached": cached,
        "blocks": blocks,
        "citations": checked.citations,
        "warnings": checked.reasons,
        "sufficient": checked.sufficient,
        "location": f"{symbol['file_path']}:{symbol['start_line']}",
        "callers": incoming,
        "callees": outgoing,
    }
