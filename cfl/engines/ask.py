from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx
import psycopg

from cfl.core import db
from cfl.core.budget import ChatCounter, PromptPart, assemble
from cfl.core.errors import CflError
from cfl.core.memory import KnowledgeMemory, answer_key, evidence_reference
from cfl.engines import graph_queries
from cfl.engines.guardrails import ABSTENTION, GUARD_VERSION, guard_answer, source_blocks
from cfl.engines.retrieval import retrieve_evidence
from cfl.engines.verify import CITATION_RE, VERIFICATION_VERSION, verify_answer
from cfl.pipeline.indexer import embedding_text, retrieve_lexical
from cfl.pipeline.knowledge import trusted_rows
from cfl.prompts.prompts import PROMPT_ANSWER

ROUTER_VERSION = "5"
RETRIEVAL_VERSION = "7"


@dataclass(frozen=True)
class Route:
    kind: str
    target: str | None = None
    operation: str | None = None
    source: str | None = None


@dataclass
class Candidate:
    row: dict
    score: float


def _known(conn, question):
    tokens = re.findall(r"[\w/.-]+\.py::[\w.@]+|[A-Za-z_]\w*(?:\.\w+)*", question)
    for token in tokens:
        rows = db.lookup_symbols(conn, token, 100)
        if any(
            r["id"] == token
            or r["name"] == token
            or r["qualname"] == token
            or r["qualname"].endswith("." + token)
            or r["file_path"].removesuffix(".py").removesuffix("/__init__").replace("/", ".")
            + "."
            + r["qualname"]
            == token
            for r in rows
        ):
            return token
    return None


def classify(question, conn):
    text = question.strip()
    lower = text.lower()
    path = re.search(
        r'path\s+from\s+[`\'"]?(.+?)[`\'"]?\s+to\s+[`\'"]?(.+?)[`\'"]?[?.]*$', text, re.IGNORECASE
    )
    if path:
        return Route("structural", path[2].strip("`'\" ?."), "path", path[1].strip("`'\" "))
    for pattern, op in [
        (r"(?:who|what) calls|which .* call|callers of", "callers"),
        (r"what depends on|what breaks if|impact of", "impact"),
        (r"most important|\bhubs\b", "hubs"),
        (r"dead code|\bunused\b", "dead"),
    ]:
        if re.search(pattern, lower):
            return Route("structural", _known(conn, text), op)
    known = _known(conn, text)
    if known and re.search(r"\bflow\b|\btrace\b|what happens when", lower):
        return Route("flow", known)
    if known and (re.search(r"what does .* do|\bexplain\b", lower) or "`" in text):
        return Route("explain", known)
    if re.search(r"how does .* work|where is .* handled|where are", lower):
        return Route("feature", known)
    return Route("retrieval", known)


def retrieve(conn, client, question, settings):
    lexical = retrieve_lexical(conn, question, settings.retrieval_top_k)
    dense = []
    view = db.get_view_status(conn).get("dense", {})
    if view.get("status") == "fresh":
        try:
            memory = KnowledgeMemory(conn, client, settings)
            config = view.get("config") or {}
            if config.get("digest") == memory.digests[settings.embed_model]:
                vector = memory.embed([embedding_text(question, settings.embed_model, query=True)])[
                    0
                ]
                dense = db.dense_units(conn, vector, settings.retrieval_top_k)
        except (CflError, ValueError, httpx.HTTPError, psycopg.Error):
            pass  # Lexical evidence survives an unavailable dense view.
    scores = {}
    rows = {}
    for ranked in (lexical, dense):
        for rank, row in enumerate(ranked, 1):
            id = row["id"]
            rows[id] = row
            scores[id] = scores.get(id, 0) + 1 / (settings.rrf_k + rank)
    order = sorted(scores, key=lambda id: (-scores[id], id))[: settings.retrieval_top_k]
    # An explicitly named indexed module scopes its capabilities to its own members.
    files = {f["path"] for f in db.list_files(conn)}
    scope = {
        token.replace(".", "/") + ".py" for token in re.findall(r"[A-Za-z_]\w*(?:\.\w+)+", question)
    } & files
    if scope:
        owned = db.get_symbols(
            conn, [s["id"] for file in sorted(scope) for s in db.symbols_in_file(conn, file)]
        )
        rows.update({s["id"]: s for s in owned})
        order = [
            s["id"] for s in sorted(owned, key=lambda s: (-scores.get(s["id"], 0), s["start_line"]))
        ][: settings.retrieval_top_k]
    # Explicit graph paths preserve intermediate source evidence; no evaluation oracle.
    evidence = retrieve_evidence(conn, question, settings.retrieval_top_k, lexical_rows=lexical)
    path_ids = evidence.path_ids if evidence.complete else []
    if path_ids:
        rows.update({b["id"]: b for b in evidence.blocks})
        order = list(dict.fromkeys(path_ids + order))
    if settings.graph_expansion:
        for id in list(order[:3]):
            if id.startswith(("file:", "feature:", "module:")):
                continue
            for edge in graph_queries.callees(
                conn, id, 1, settings.edge_conf_threshold
            ) + graph_queries.callers(conn, id, 1, settings.edge_conf_threshold):
                if edge["id"] and edge["id"] not in rows:
                    row = db.get_symbol(conn, edge["id"])
                    if row:
                        rows[row["id"]] = row
                        order.append(row["id"])
    if settings.llm_rerank and order:
        prompt = (
            "Rank these candidate IDs for the question. Return ONLY a JSON array of the IDs.\nQuestion: "
            + question
            + "\n"
            + "\n".join(
                id + ": " + (rows[id].get("summary_short") or rows[id].get("signature") or "")
                for id in order
            )
        )
        try:
            assembled = assemble(
                "Rank using supplied evidence only.",
                [PromptPart("question and candidates", prompt, 1, False)],
                num_ctx=settings.num_ctx,
                num_predict=350,
                overhead=settings.template_overhead,
                counter=client.counter,
            )
            result = client.generate(
                assembled.prompt,
                "",
                fmt={"type": "array", "items": {"type": "string"}},
                num_predict=350,
                task="rerank",
            )
            ranked = json.loads(result.text)
            if isinstance(ranked, list) and len(ranked) == len(order) and set(ranked) == set(order):
                order = ranked
        except (CflError, ValueError, TypeError):
            pass
    return [
        Candidate(row, scores.get(row["id"], 0))
        for row in trusted_rows(conn, [rows[id] for id in order])
    ]


def block_for(row):
    if row.get("kind") in {"feature", "module", "knowledge"}:
        return [
            block_for(
                {
                    **s,
                    "summary_short": row["summary_short"],
                    "summary_json": None,
                    "knowledge_status": row.get("knowledge_status", "unvalidated"),
                }
            )[0]
            for s in row.get("members", [])
        ]
    body = row.get("raw_code") or ""
    row = {
        **row,
        "summary_long": None,
        "summary_json": row.get("summary_json")
        if row.get("knowledge_status") in {"complete", "partial"}
        else None,
        "summary_short": row.get("summary_short")
        if row.get("knowledge_status") in {"complete", "partial"}
        else None,
    }
    if (
        row.get("summary_json")
        and row.get("knowledge_status") in {"complete", "partial"}
        and row.get("kind") not in {"class", "file"}
    ):
        body = json.dumps(row["summary_json"], ensure_ascii=False) + "\n" + body
    if row.get("kind") == "file":
        body = row.get("skeleton") or row.get("raw_code") or ""
        row = {**row, "code_hash": row["sha256"]}
    tag = f"[{row['file_path']}:{row['start_line']}-{row['end_line']}]"
    summary = (
        row.get("summary_short") if row.get("knowledge_status") in {"complete", "partial"} else ""
    )
    return [{**row, "text": tag + "\n" + (summary or "") + "\n" + body}]


def render_blocks(candidates, budget, *, counter, max_blocks=None):
    blocks = []
    seen = set()
    for candidate in candidates:
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
        for block in block_for(candidate.row):
            if block["id"] in seen:
                continue
            cost = counter.count(block["text"]) + 8
            if cost > budget:
                continue
            budget -= cost
            blocks.append(block)
            seen.add(block["id"])
    return blocks


def _structural(conn, route, settings):
    target = graph_queries.resolve_symbol(conn, route.target) if route.target else None
    op = route.operation
    if op in {"callers", "impact"} and target is None:
        raise ValueError("Specify an indexed symbol for this graph question")
    if op == "callers":
        rows = graph_queries.callers(conn, target, 1, settings.edge_conf_threshold)
    elif op == "impact":
        rows = [
            s
            for g in graph_queries.impact(
                conn, target, settings.feature_depth, settings.edge_conf_threshold
            )["groups"]
            for s in g["symbols"]
        ]
    elif op == "hubs":
        rows = graph_queries.hubs(conn, settings.retrieval_top_k, settings.edge_conf_threshold)
    elif op == "dead":
        rows = graph_queries.dead(conn, settings.edge_conf_threshold)
    else:
        source = graph_queries.resolve_symbol(conn, route.source)
        if target is None:
            raise ValueError("Specify the path target")
        hops = graph_queries.path(conn, source, target, 8, settings.edge_conf_threshold)
        rows = [] if hops is None else [db.get_symbol(conn, source.id)] + hops
    ids = list(
        dict.fromkeys([s["id"] for s in rows if s.get("id")] + ([target.id] if target else []))
    )
    source_rows = {s["id"]: s for s in db.get_symbols(conn, ids)}
    blocks = [block_for(source_rows[id])[0] for id in ids]
    lines = [
        "This answer may be incomplete: results cover the indexed static graph within the configured depth and confidence threshold.",
        f"Graph query: {op}. Confidence >= {settings.edge_conf_threshold}.",
    ]
    if target:
        s = source_rows[target.id]
        lines.append(
            f"Target: {s['qualname']} [{s['file_path']}:{s['start_line']}-{s['end_line']}]"
        )
    for row in rows:
        id = row.get("id")
        if id and id in source_rows:
            s = source_rows[id]
            lines.append(
                f"- {s['qualname']} [{s['file_path']}:{s['start_line']}-{s['end_line']}]"
                + (
                    f"; call-site {row.get('callsite_file')}:{row['line']} ({row.get('source')}, confidence {row.get('confidence')})"
                    if "line" in row
                    else ""
                )
            )
    if not rows:
        lines.append("No matching graph results within the configured limits.")
        if target:
            s = source_rows[target.id]
            lines.append(f"Target [{s['file_path']}:{s['start_line']}-{s['end_line']}]")
    return "\n".join(lines), blocks


def answer(
    conn, client, settings, question, mode="detailed", *, explain_graph=False, on_token=None
):
    if mode not in {"brief", "detailed"}:
        raise ValueError("mode must be brief or detailed")
    counter = ChatCounter(client.counter)
    route = classify(question, conn)
    db.set_meta(conn, "router_version", ROUTER_VERSION)
    db.set_meta(conn, "retrieval_version", RETRIEVAL_VERSION)
    db.set_meta(conn, "verification_version", VERIFICATION_VERSION)
    db.set_meta(conn, "guard_version", GUARD_VERSION)
    tags = {m["name"]: m for m in client.tags()} if route.kind != "structural" or explain_graph else {}
    verifier_tag = (
        settings.verifier_model
        if ":" in settings.verifier_model
        else settings.verifier_model + ":latest"
    )
    verifier_digest = tags.get(verifier_tag, {}).get("digest", db.get_meta(conn, "verifier_model_digest") or "unavailable")
    if route.kind != "structural" or explain_graph:
        db.set_meta(conn, "verifier_model_digest", verifier_digest)
    if route.kind != "structural":
        KnowledgeMemory(conn, client, settings)
    revision = db.index_version(conn)
    key = answer_key(
        question,
        mode,
        {
            "gen_model": settings.gen_model,
            "num_ctx": settings.num_ctx,
            "output_reserve": settings.num_predict_brief
            if mode == "brief"
            else settings.num_predict_detailed,
            "kv_quantization": settings.kv_quantization,
            "kv_quantization_type": settings.kv_quantization_type,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "graph_expansion": settings.graph_expansion,
            "llm_rerank": settings.llm_rerank,
            "top_k": settings.retrieval_top_k,
            "inject_max": settings.retrieval_inject_max,
            "rrf_k": settings.rrf_k,
            "min_conf": settings.edge_conf_threshold,
            "feature_depth": settings.feature_depth,
            "explain_graph": explain_graph,
            "verifier_digest": verifier_digest,
            "guard_version": GUARD_VERSION,
            "num_predict_review": settings.num_predict_review,
        },
    )
    cached = db.get_cached_answer(conn, key, revision)
    if cached is not None:
        blocks = [
            block_for(s)[0]
            for s in db.evidence_units(conn, db.cached_answer_evidence(conn, key, revision))
        ]
        try:
            source_blocks(conn, blocks)
        except (CflError, ValueError, OSError) as exc:
            return {
                "text": ABSTENTION,
                "status": "abstained",
                "sufficient": False,
                "cached": False,
                "route": route.kind,
                "blocks": [],
                "citations": [],
                "warnings": [str(exc)],
            }
        verified = verify_answer(cached, blocks, conn)
        return {
            "text": verified.text,
            "route": route.kind,
            "cached": True,
            "blocks": blocks,
            "citations": verified.citations,
            "warnings": verified.warnings,
            "sufficient": verified.sufficient,
            "status": "partial"
            if verified.text.startswith("This answer may be incomplete")
            else "complete"
            if verified.sufficient
            else "abstained",
        }
    if route.kind == "structural" and not explain_graph:
        text, blocks = _structural(conn, route, settings)
    elif route.kind == "flow":
        from cfl.engines.flow import flow

        path_evidence = retrieve_evidence(
            conn, question, settings.retrieval_top_k, min_conf=settings.edge_conf_threshold
        )
        return flow(
            conn,
            client,
            settings,
            route.target,
            depth=settings.flow_depth,
            on_token=on_token,
            path_evidence=path_evidence,
        )
    elif route.kind == "explain":
        from cfl.engines.explain import explain

        return explain(conn, client, settings, route.target, detail=mode, on_token=on_token)
    else:
        if route.kind == "structural":
            graph_text, graph_blocks = _structural(conn, route, settings)
            candidates = [Candidate(b, 0) for b in graph_blocks]
            question = question + "\nVerified graph results:\n" + graph_text
        else:
            candidates = retrieve(conn, client, question, settings)
        if (
            mode == "brief"
            and route.kind != "structural"
            and candidates
            and not candidates[0].row["id"].startswith(("file:", "module:", "feature:"))
            and candidates[0].row.get("summary_short")
        ):
            blocks = block_for(candidates[0].row)
            text = (
                blocks[0]["summary_short"]
                + f" [{blocks[0]['file_path']}:{blocks[0]['start_line']}-{blocks[0]['end_line']}]"
            )
        else:
            reserve = (
                settings.num_predict_brief if mode == "brief" else settings.num_predict_detailed
            )
            budget = (
                settings.num_ctx
                - settings.template_overhead
                - reserve
                - counter.count(PROMPT_ANSWER + "\n\n" + question)
                - 128
            )
            blocks = render_blocks(
                candidates,
                budget,
                counter=client.counter,
                max_blocks=None if route.kind == "structural" else settings.retrieval_inject_max,
            )
            if not blocks:
                return {
                    "text": ABSTENTION,
                    "status": "abstained",
                    "route": route.kind,
                    "cached": False,
                    "blocks": [],
                    "citations": [],
                    "warnings": [],
                    "sufficient": False,
                }
            neighbors = []
            if mode == "detailed":
                seen = {b["id"] for b in blocks}
                for block in blocks[:3]:
                    if block.get("kind") == "file":
                        continue
                    edges = graph_queries.callees(
                        conn, block["id"], 1, settings.edge_conf_threshold
                    ) + graph_queries.callers(conn, block["id"], 1, settings.edge_conf_threshold)
                    for row in trusted_rows(
                        conn,
                        db.get_symbols(
                            conn,
                            list(
                                dict.fromkeys(
                                    e["id"] for e in edges if e["id"] and e["id"] not in seen
                                )
                            ),
                        ),
                    ):
                        if scope := re.findall(r"[A-Za-z_]\w*(?:\.\w+)+", question):
                            scoped_files = {token.replace(".", "/") + ".py" for token in scope} & {
                                f["path"] for f in db.list_files(conn)
                            }
                            if scoped_files and row["file_path"] not in scoped_files:
                                continue
                        tag = f"[{row['file_path']}:{row['start_line']}-{row['end_line']}]"
                        neighbors.append(
                            {
                                **row,
                                "text": "Supporting caller/callee one-liner: "
                                + tag
                                + " "
                                + row["qualname"]
                                + ": "
                                + (row["summary_short"] or "No stored summary"),
                                "context_role": "neighbor",
                            }
                        )
                        seen.add(row["id"])
            assembled = assemble(
                PROMPT_ANSWER,
                [PromptPart("question", question, 1, False)]
                + [PromptPart(b["id"], b["text"], 2, False) for b in blocks]
                + [PromptPart(b["id"], b["text"], 3, False) for b in neighbors],
                num_ctx=settings.num_ctx,
                num_predict=reserve,
                overhead=settings.template_overhead,
                counter=counter,
            )
            supplied = {p.name for p in assembled.parts}
            blocks = [b for b in blocks + neighbors if b["id"] in supplied]
            chunks = []
            for chunk in client.chat(
                [{"role": "user", "content": assembled.prompt}],
                num_predict=reserve,
                task="answer",
                stream=True,
            ):
                chunks.append(chunk)  # noqa: PERF402 - buffered until validation
            text = "".join(chunks)
            if not CITATION_RE.search(text):
                # One bounded citation repair; an uncited retry still fails verification.
                retry = assemble(
                    PROMPT_ANSWER,
                    [
                        PromptPart("question", question, 1, False),
                        PromptPart(
                            "citation reminder",
                            "Your previous response supplied no source citations. Answer again using the blocks below and include their exact [path:start-end] tags. Do not introduce new facts.",
                            1,
                            False,
                        ),
                    ]
                    + [PromptPart(b["id"], b["text"], 2, False) for b in blocks],
                    num_ctx=settings.num_ctx,
                    num_predict=reserve,
                    overhead=settings.template_overhead,
                    counter=counter,
                )
                supplied = {p.name for p in retry.parts}
                blocks = [b for b in blocks if b["id"] in supplied]
                result = client.chat(
                    [{"role": "user", "content": retry.prompt}],
                    num_predict=reserve,
                    task="answer_citation_repair",
                )
                text = result.text
    verified = verify_answer(text, blocks, conn)
    if route.kind != "structural" or explain_graph:
        incomplete = (
            ["Some supporting knowledge is partial or unvalidated."]
            if any(b.get("knowledge_status") in {"partial", "unvalidated"} for b in blocks)
            else []
        )
        verified = guard_answer(
            conn, client, settings, text, blocks, question, incomplete=incomplete
        )
    if verified.sufficient and (getattr(verified, "status", "complete") == "complete" or route.kind == "structural" and not explain_graph):
        references = [evidence_reference(b) for b in blocks]
        if references and db.index_version(conn) == revision:
            db.put_cached_answer(
                conn, key, mode, verified.text, revision, verified=True, evidence=references
            )
    if on_token:
        on_token(verified.text)
    return {
        "raw_text": verified.text,
        "raw_citations": verified.citations,
        "text": verified.text,
        "status": getattr(verified, "status", "partial"),
        "route": route.kind,
        "cached": False,
        "blocks": blocks,
        "citations": verified.citations,
        "warnings": getattr(verified, "warnings", getattr(verified, "reasons", [])),
        "sufficient": verified.sufficient,
    }
