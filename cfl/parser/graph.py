from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from cfl.core.db import fetch_edges, get_all_symbols, set_graph_metrics, set_view_status

if TYPE_CHECKING:
    from psycopg import Connection

    from cfl.config import Settings


@dataclass
class WorkItem:
    scc_id: str
    symbol_ids: list[str]
    layer: int


def build_call_graph(conn: Connection, threshold: float) -> nx.DiGraph:
    """Build caller-to-callee edges, retaining the strongest evidence per pair."""
    graph = nx.DiGraph()
    for edge in fetch_edges(conn, min_conf=threshold):
        caller, callee = edge["caller_id"], edge["callee_id"]
        graph.add_node(caller)
        if callee is None:
            continue
        weight = edge["confidence"]
        if graph.has_edge(caller, callee):
            weight = max(weight, graph[caller][callee]["weight"])
        graph.add_edge(caller, callee, weight=weight)
    return graph


def build_order_graph(call_graph: nx.DiGraph, symbols: list[dict]) -> nx.DiGraph:
    """Include isolated symbols and structural dependencies for summarisation."""
    graph = call_graph.copy()
    graph.add_nodes_from(symbol["id"] for symbol in symbols)
    classes = {s["id"] for s in symbols if s["kind"] == "class"}
    by_file: dict[str, list[str]] = {}
    for symbol in symbols:
        if symbol["kind"] in {"class", "function"}:
            by_file.setdefault(symbol["file_path"], []).append(symbol["id"])
        if symbol["kind"] == "method" and symbol.get("parent_id") in classes:
            graph.add_edge(symbol["parent_id"], symbol["id"])
    for symbol in symbols:
        if symbol["kind"] == "module":
            graph.add_edges_from(
                (symbol["id"], target) for target in by_file.get(symbol["file_path"], [])
            )
    return graph


def condense(order_graph: nx.DiGraph) -> dict:
    """Collapse recursion groups and assign dependency layers without cycle enumeration."""
    # Canonical member order keeps SCC identifiers stable across insertion orders.
    sccs = sorted(nx.strongly_connected_components(order_graph), key=lambda s: sorted(s))
    dag = nx.condensation(order_graph, scc=sccs)
    layers: dict[int, int] = {}
    for node in reversed(list(nx.topological_sort(dag))):
        layers[node] = max((layers[dep] + 1 for dep in dag.successors(node)), default=0)
    result: dict = {"__sccs__": {}}
    for node, attrs in dag.nodes(data=True):
        scc_id = f"scc-{node}"
        members = sorted(attrs["members"])
        result["__sccs__"][scc_id] = members
        for member in members:
            result[member] = {"scc_id": scc_id, "layer": layers[node]}
    return result


def compute_pagerank(call_graph: nx.DiGraph) -> dict[str, float]:
    return nx.pagerank(call_graph, alpha=0.85, weight="weight")


def compute_communities(call_graph: nx.DiGraph) -> dict[str, int]:
    if not call_graph:
        return {}
    graph = call_graph.to_undirected()
    # Louvain divides by total edge weight; edgeless/zero-confidence graphs need singletons.
    if graph.size(weight="weight") == 0:
        groups = [{node} for node in sorted(graph)]
    else:
        groups = nx.community.louvain_communities(graph, weight="weight", seed=42)
    return {node: index for index, group in enumerate(groups) for node in group}


def processing_order(
    conn: Connection, *, priority: str | None = None, min_conf: float = 0.6
) -> list[WorkItem]:
    symbols = get_all_symbols(conn)
    call_graph = build_call_graph(conn, threshold=min_conf)
    order_graph = build_order_graph(call_graph, symbols)
    condensed = condense(order_graph)
    ranks = compute_pagerank(call_graph)
    file_by_symbol = {symbol["id"]: symbol["file_path"] for symbol in symbols}
    file_ranks: dict[str, float] = {}
    for sid, file_path in file_by_symbol.items():
        file_ranks[file_path] = max(file_ranks.get(file_path, 0.0), ranks.get(sid, 0.0))
    reachable: set[str] = set()
    if priority == "entrypoints-first":
        for symbol in symbols:
            if symbol.get("is_entrypoint"):
                reachable.add(symbol["id"])
                reachable.update(nx.descendants(order_graph, symbol["id"]))
    items = [
        WorkItem(scc_id, members, condensed[members[0]]["layer"])
        for scc_id, members in condensed["__sccs__"].items()
    ]
    items.sort(
        key=lambda item: (
            item.layer,
            -int(bool(reachable.intersection(item.symbol_ids))),
            -max((file_ranks[file_by_symbol[sid]] for sid in item.symbol_ids), default=0.0),
            min(file_by_symbol[sid] for sid in item.symbol_ids),
            item.scc_id,
        )
    )
    return items


def run_graph_stage(conn: Connection, settings: Settings) -> None:
    symbols = get_all_symbols(conn)
    call_graph = build_call_graph(conn, threshold=settings.edge_conf_threshold)
    condensed = condense(build_order_graph(call_graph, symbols))
    ranks = compute_pagerank(call_graph)
    compute_communities(call_graph)
    rows = [
        {"id": s["id"], "pagerank": ranks.get(s["id"], 0.0), **condensed[s["id"]]} for s in symbols
    ]
    with conn.transaction():
        set_graph_metrics(conn, rows)
        set_view_status(conn, "graph", "fresh", {"threshold": settings.edge_conf_threshold})
