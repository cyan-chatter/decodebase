from __future__ import annotations

from cfl.core import db, mermaid
from cfl.core.budget import ChatCounter, PromptPart, assemble, batch_by_budget
from cfl.core.errors import BudgetExceeded
from cfl.engines.graph_queries import resolve_symbol
from cfl.engines.guardrails import guard_answer
from cfl.engines.verify import verify_answer
from cfl.prompts.prompts import PROMPT_FLOW_STEP, PROMPT_FLOW_STITCH


def build_trace(conn, root, depth=4, min_conf=0.6, *, max_nodes=4000):
    if depth < 0 or depth > 64 or max_nodes < 2:
        raise ValueError("Require depth 0..64 and max_nodes >= 2")
    root = root.id if hasattr(root, "id") else root
    symbols = {s["id"]: s for s in db.get_all_symbols(conn)}
    if root not in symbols:
        raise ValueError("Unknown trace root")
    outgoing = {}
    for e in db.fetch_edges(conn, 0):
        if e["callee_id"] is None or e["confidence"] >= min_conf:
            outgoing.setdefault(e["caller_id"], []).append(e)
    for edges in outgoing.values():
        edges.sort(
            key=lambda e: (
                e["line"],
                e["callee_expr"],
                e["callee_id"] or "",
                -e["confidence"],
                e["source"],
            )
        )
    count = [0]

    def visit(id, level, visiting, edge=None):
        count[0] += 1
        symbol = symbols.get(id)
        node = {
            "id": id,
            "label": symbol["qualname"] if symbol else (edge or {}).get("callee_expr", id),
            "kind": "symbol" if symbol else "external",
            "depth": level,
            "children": [],
            **(edge or {}),
        }
        if symbol:
            node["symbol"] = symbol
        if edge:
            node["callsite_file"] = symbols[edge["caller_id"]]["file_path"]
        if id in visiting:
            node["recursion"] = True
            return node
        edges = outgoing.get(id, [])
        if level >= depth:
            if edges:
                node["omitted_count"] = len(edges)
            return node
        seen = set()
        for i, e in enumerate(edges):
            identity = (e["line"], e["callee_id"], e["callee_expr"])
            if identity in seen:
                continue
            seen.add(identity)
            if count[0] >= max_nodes - 1:
                node["children"].append(
                    {
                        "id": id + "::cap",
                        "label": f"...{len(edges) - i} more calls",
                        "kind": "collapsed",
                        "children": [],
                        "count": len(edges) - i,
                    }
                )
                break
            target = e["callee_id"] or f"external:{id}:{e['line']}:{e['callee_expr']}"
            node["children"].append(visit(target, level + 1, visiting | {id}, e))
        return node

    return visit(root, 0, set())


def flatten(tree):
    rows = []

    def visit(node):
        rows.append(node)
        for child in node.get("children", []):
            visit(child)

    visit(tree)
    return rows


def _step_blocks(conn, tree):
    symbols = {s["id"]: s for s in db.get_all_symbols(conn)}
    blocks = []
    steps = []
    for index, node in enumerate(flatten(tree)):
        if node.get("kind") == "collapsed":
            steps.append(node["label"])
            continue
        if index == 0:
            s = node["symbol"]
            start = s["start_line"]
            snippet = s["raw_code"]
        else:
            s = symbols[node["caller_id"]]
            start = node["line"]
            lines = s["raw_code"].splitlines()
            offset = start - s["start_line"]
            snippet = lines[offset] if 0 <= offset < len(lines) else node["callee_expr"]
        block = {
            **s,
            "symbol_start": s["start_line"],
            "symbol_end": s["end_line"],
            "start_line": start,
            "end_line": s["end_line"] if index == 0 else start,
            "text": f"[{s['file_path']}:{start}-{s['end_line'] if index == 0 else start}]\n{snippet}",
        }
        blocks.append(block)
        callee_text = (
            s["raw_code"] if index == 0 else "No indexed implementation; external/unresolved leaf."
        )
        if index and node.get("symbol"):
            callee = node["symbol"]
            callee_tag = f"[{callee['file_path']}:{callee['start_line']}-{callee['end_line']}]"
            callee_text = callee_tag + " " + callee["qualname"] + ": " + callee["raw_code"]
            blocks.append(
                {
                    **callee,
                    "symbol_start": callee["start_line"],
                    "symbol_end": callee["end_line"],
                    "text": callee_text,
                }
            )
        steps.append(
            (
                f"Entry definition: {node['label']} (not a call or self-recursion step)"
                if index == 0
                else f"Step {index}: {s['qualname']} -> {node['label']} at call-site {s['file_path']}:{start}"
            )
            + (" ↻ recursion leaf" if node.get("recursion") else "")
            + f"; condition={node.get('control_ctx') or 'unconditional'}; source={node.get('source', 'root')}; confidence={node.get('confidence', 1)}\n"
            + block["text"]
            + "\nCallee definition summary: "
            + callee_text
        )
    return steps, blocks


def render_flow_md(narrative, diagram, tree, *, diagram_valid=True):
    rows = ["| Call site | Symbol | Enclosing syntax / traversal annotation | Source / confidence |", "| --- | --- | --- | --- |"]
    for node in flatten(tree):
        symbol = node.get("symbol", {})
        path = node.get("callsite_file") or symbol.get("file_path", "")
        line = node.get("line") or symbol.get("start_line", "")
        label = node["label"] + (" ↻" if node.get("recursion") else "")
        values = [
            f"{path}:{line}",
            label,
            node.get("control_ctx") or "",
            f"{node.get('source', 'root')} / {node.get('confidence', '')}",
        ]
        rows.append(
            "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |"
        )
    return (
        narrative
        + "\n\n```"
        + ("mermaid" if diagram_valid else "text")
        + "\n"
        + diagram
        + "\n```\n\n"
        + "\n".join(rows)
    )


def flow(
    conn,
    client,
    settings,
    query,
    *,
    depth=4,
    min_conf=None,
    on_token=None,
    diagram_type="flowchart",
    path_evidence=None,
):
    counter = ChatCounter(client.counter)
    root = resolve_symbol(conn, query)
    tree = build_trace(
        conn, root, depth, settings.edge_conf_threshold if min_conf is None else min_conf
    )
    if path_evidence is not None and path_evidence.complete:
        rows = {b["id"]: b for b in path_evidence.blocks}
        chain = []
        for index, id in enumerate(path_evidence.path_ids):
            symbol = rows[id]
            incoming = path_evidence.path_hops[index - 1] if index else {}
            node = {
                **incoming,
                "id": id,
                "label": symbol["qualname"],
                "kind": "symbol",
                "symbol": symbol,
                "depth": index,
                "children": [],
            }
            if index:
                node["callsite_file"] = rows[incoming["caller_id"]]["file_path"]
                chain[-1]["children"].append(node)
            chain.append(node)
        tree = chain[0]
        depth = max(depth, len(chain) - 1)
    full_tree = tree
    tree = mermaid.collapse(tree, settings.flow_max_nodes, depth)
    narrative_tree = full_tree if path_evidence is not None and path_evidence.complete else tree
    diagram, labels = (
        mermaid.render_sequence(tree)
        if diagram_type == "sequence"
        else mermaid.render_flowchart(tree)
    )
    warnings = mermaid.validate_mermaid(diagram)
    diagram_valid = not warnings
    if warnings:
        diagram = "\n".join(
            "  " * min(n.get("depth", 0), depth) + n["label"] for n in flatten(tree)
        )
    steps, blocks = _step_blocks(conn, narrative_tree)
    reserve = min(350, settings.num_predict_detailed)
    budget = (
        settings.num_ctx
        - settings.template_overhead
        - reserve
        - counter.count(PROMPT_FLOW_STEP)
        - 128
    )
    batches = batch_by_budget(steps, lambda step: counter.count(step) + 8, budget)
    narratives = []
    step_outputs = []
    for batch in batches:
        assembled = assemble(
            PROMPT_FLOW_STEP,
            [PromptPart("steps", "\n".join(batch), 1, False)],
            num_ctx=settings.num_ctx,
            num_predict=reserve,
            overhead=settings.template_overhead,
            counter=counter,
        )
        result = client.chat(
            [{"role": "user", "content": assembled.prompt}], num_predict=reserve, task="flow_step"
        )
        group_blocks = [b for b in blocks if b["text"] in "\n".join(batch)]
        step_outputs.append(result.text)
        checked = verify_answer(result.text, group_blocks, conn)
        warnings.extend(checked.warnings)
        narratives.append(
            checked.text
            if checked.sufficient
            else "Source evidence for this step group (model supplied no supported citations):\n"
            + "\n".join(batch)
        )
    budget = (
        settings.num_ctx
        - settings.template_overhead
        - settings.num_predict_detailed
        - counter.count(PROMPT_FLOW_STITCH)
        - 128
    )
    while True:
        groups = batch_by_budget(narratives, lambda text: counter.count(text) + 8, budget)
        if len(groups) > 1 and all(len(g) == 1 for g in groups):
            raise BudgetExceeded("Flow narratives cannot be stitched within context")
        merged = []
        for group in groups:
            assembled = assemble(
                PROMPT_FLOW_STITCH,
                [PromptPart("narratives", "\n".join(group), 1, False)],
                num_ctx=settings.num_ctx,
                num_predict=settings.num_predict_detailed,
                overhead=settings.template_overhead,
                counter=counter,
            )
            pieces = []
            for piece in client.chat(
                [{"role": "user", "content": assembled.prompt}],
                num_predict=settings.num_predict_detailed,
                task="flow_stitch",
                stream=True,
            ):
                pieces.append(piece)  # noqa: PERF402 - buffered until validation
            merged.append("".join(pieces))
        narratives = merged
        if len(groups) == 1:
            break
    checked = guard_answer(
        conn,
        client,
        settings,
        narratives[0],
        blocks,
        "Explain the static call flow for " + root.qualname,
        incomplete=[
            "This is a bounded static call trace; runtime paths and external behavior may be missing."
        ],
    )
    warnings.extend(checked.reasons)
    text = render_flow_md(checked.text, diagram, narrative_tree, diagram_valid=diagram_valid)
    if on_token:
        on_token(text)
    # Public graph JSON carries source structure, never unvalidated draft prose.
    for node in flatten(tree):
        if "symbol" in node:
            node["symbol"] = {
                **node["symbol"],
                "summary_short": None,
                "summary_long": None,
                "summary_json": None,
            }
    return {
        "raw_text": checked.text,
        "raw_citations": checked.citations,
        "text": text,
        "status": checked.status,
        "route": "flow",
        "cached": False,
        "blocks": list(
            {
                b["id"]: {
                    **b,
                    "summary_short": None,
                    "summary_json": None,
                    "summary_long": None,
                    "start_line": b["symbol_start"],
                    "end_line": b["symbol_end"],
                }
                for b in blocks
            }.values()
        ),
        "citations": checked.citations,
        "warnings": warnings,
        "sufficient": checked.sufficient,
        "tree": tree,
        "diagram": diagram,
        "labels": labels,
    }
