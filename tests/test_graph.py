from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import Mock

import networkx as nx
import pytest

from cfl.config import Settings
from cfl.parser import graph


def test_mutual_recursion_is_one_scc():
    result = graph.condense(
        nx.DiGraph([("is_even", "is_odd"), ("is_odd", "is_even"), ("factorial", "factorial")])
    )
    assert result["is_even"]["scc_id"] == result["is_odd"]["scc_id"]
    assert result["factorial"]["scc_id"] != result["is_even"]["scc_id"]


def test_layers_respect_callee_before_caller():
    result = graph.condense(nx.DiGraph([("A", "B"), ("B", "C")]))
    assert result["C"]["layer"] < result["B"]["layer"] < result["A"]["layer"]


def test_class_sorts_after_methods():
    symbols = [
        {"id": "Foo", "kind": "class", "file_path": "f.py"},
        {"id": "Foo.bar", "kind": "method", "parent_id": "Foo", "file_path": "f.py"},
        {"id": "Foo.baz", "kind": "method", "parent_id": "Foo", "file_path": "f.py"},
        {"id": "module", "kind": "module", "file_path": "f.py"},
        {"id": "alone", "kind": "function", "file_path": "other.py"},
    ]
    result = graph.condense(graph.build_order_graph(nx.DiGraph(), symbols))
    assert result["Foo"]["layer"] > result["Foo.bar"]["layer"]
    assert result["module"]["layer"] > result["Foo"]["layer"]
    assert result["alone"]["layer"] == 0


def test_expected_sccs_match_fixture(resolution_data, fixture_repo_path):
    symbols, _ = resolution_data
    by_name = {s["qualname"]: s["id"] for s in symbols}
    calls = nx.DiGraph()
    calls.add_nodes_from(by_name.values())
    for edge in json.loads((fixture_repo_path / "expected_edges.json").read_text()):
        if edge["callee_qualname"] is not None:
            module, name = edge["caller"].split(".", 1)
            calls.add_edge(f"{module}.py::{name}", by_name[edge["callee_qualname"]])
    result = graph.condense(calls)
    qualnames = {s["id"]: s["qualname"] for s in symbols}
    actual = {
        frozenset(qualnames[sid] for sid in members) for members in result["__sccs__"].values()
    }
    expected = {
        frozenset(group)
        for group in json.loads((fixture_repo_path / "expected_sccs.json").read_text())
    }
    assert actual == expected


def test_pagerank_higher_for_more_called():
    calls = nx.DiGraph((f"caller{i}", "hub") for i in range(5))
    calls.add_node("leaf")
    ranks = graph.compute_pagerank(calls)
    assert ranks["hub"] > ranks["leaf"]
    assert sum(ranks.values()) == pytest.approx(1.0)


def test_build_call_graph_max_confidence_and_external(monkeypatch):
    def fetch(conn, min_conf):
        assert min_conf == 0.6
        return [
            {"caller_id": "A", "callee_id": "B", "confidence": 0.7},
            {"caller_id": "A", "callee_id": "B", "confidence": 0.9},
            {"caller_id": "C", "callee_id": None, "confidence": 0.8},
        ]

    monkeypatch.setattr(graph, "fetch_edges", fetch)
    calls = graph.build_call_graph(None, 0.6)
    assert calls["A"]["B"]["weight"] == 0.9
    assert set(calls) == {"A", "B", "C"}
    assert list(calls.edges) == [("A", "B")]


def test_processing_order_layers_ascending(monkeypatch):
    symbols = [
        {"id": node, "kind": "function", "file_path": "f.py", "is_entrypoint": node == "A"}
        for node in ["A", "B", "C", "X", "Y"]
    ]
    calls = nx.DiGraph([("A", "B"), ("B", "C"), ("X", "Y")])
    monkeypatch.setattr(graph, "get_all_symbols", lambda conn: symbols)
    monkeypatch.setattr(graph, "build_call_graph", lambda conn, threshold: calls)
    items = graph.processing_order(None, priority="entrypoints-first")
    assert [i.layer for i in items] == sorted(i.layer for i in items)
    assert items[0].symbol_ids == ["C"]
    assert {sid for item in items for sid in item.symbol_ids} == set(calls)
    assert all(isinstance(item, graph.WorkItem) for item in items)


def test_scc_identifiers_stable_and_deep_chain():
    edges = [(str(i), str(i + 1)) for i in range(1500)]
    first = graph.condense(nx.DiGraph(edges))
    assert first == graph.condense(nx.DiGraph(reversed(edges)))
    assert first["0"]["layer"] == 1500


def test_empty_and_edgeless_metrics():
    assert graph.condense(nx.DiGraph()) == {"__sccs__": {}}
    assert graph.compute_pagerank(nx.DiGraph()) == {}
    assert graph.compute_communities(nx.DiGraph()) == {}
    calls = nx.DiGraph()
    calls.add_nodes_from(["a", "b"])
    assert len(set(graph.compute_communities(calls).values())) == 2
    calls.add_edge("a", "b", weight=0.0)
    assert set(graph.compute_communities(calls)) == {"a", "b"}


def test_run_graph_stage_persists_isolated_symbols(monkeypatch):
    conn = Mock()
    conn.transaction.side_effect = nullcontext
    monkeypatch.setattr(
        graph,
        "get_all_symbols",
        lambda conn: [{"id": "alone", "kind": "function", "file_path": "f.py"}],
    )
    monkeypatch.setattr(graph, "build_call_graph", lambda conn, threshold: nx.DiGraph())
    metrics, status = Mock(), Mock()
    monkeypatch.setattr(graph, "set_graph_metrics", metrics)
    monkeypatch.setattr(graph, "set_view_status", status)
    graph.run_graph_stage(conn, Settings())
    assert metrics.call_args.args[1] == [
        {"id": "alone", "pagerank": 0.0, "scc_id": "scc-0", "layer": 0}
    ]
    status.assert_called_once_with(conn, "graph", "fresh", {"threshold": 0.6})
