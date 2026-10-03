from __future__ import annotations

import json

import pytest

from cfl.parser.resolver import Resolver, detect_entrypoints


@pytest.fixture
def resolver(resolution_data):
    return Resolver(*resolution_data)


def edges(resolver, caller, expr):
    return [
        e for e in resolver.resolve_all() if e["caller_id"] == caller and e["callee_expr"] == expr
    ]


def test_import_alias_resolution(resolver):
    edge = edges(resolver, "aliases.py::run", "proc")[0]
    assert (edge["resolution"], edge["callee_id"]) == ("import", "utils.py::process")


@pytest.mark.parametrize("expr", ["repo.save", "r.save"])
def test_typed_calls(resolver, expr):
    edge = edges(resolver, "services.py::create_task", expr)[0]
    assert (edge["resolution"], edge["callee_id"]) == ("typed", "models.py::Repository.save")


@pytest.mark.parametrize("expr", ["Base().save", "Repository(None).save"])
def test_ambiguous_candidates_all_present(resolver, expr):
    result = edges(resolver, "models.py::save_record", expr)
    assert {e["callee_id"] for e in result} == {
        "models.py::Base.save",
        "models.py::Repository.save",
    }
    assert all(e["resolution"] == "ambiguous" and e["confidence"] == 0.4 for e in result)


@pytest.mark.parametrize("expr", ["os.path.join", "len", "d.get"])
def test_external_edge_has_null_callee(resolver, expr):
    result = edges(resolver, "external.py::do_work", expr)
    assert len(result) == 1
    assert result[0]["callee_id"] is None


def test_blocklist_prevents_linking(resolution_data):
    symbols, files = resolution_data
    symbols = [
        *symbols,
        {
            "id": "utils.py::get",
            "name": "get",
            "qualname": "get",
            "file_path": "utils.py",
            "kind": "function",
        },
    ]
    resolver = Resolver(symbols, files)
    assert edges(resolver, "external.py::do_work", "d.get")[0]["callee_id"] is None


def test_self_recursion_resolves_local(resolver):
    edge = edges(resolver, "recursion.py::factorial", "factorial")[0]
    assert edge["callee_id"] == edge["caller_id"]
    assert edge["resolution"] == "local"


def test_mutual_recursion_edges(resolver):
    for caller, callee in [("is_even", "is_odd"), ("is_odd", "is_even")]:
        assert (
            edges(resolver, f"recursion.py::{caller}", callee)[0]["callee_id"]
            == f"recursion.py::{callee}"
        )


def test_all_expected_edges(resolver, fixture_repo_path):
    for expected in json.loads((fixture_repo_path / "expected_edges.json").read_text()):
        module, qualname = expected["caller"].split(".", 1)
        result = edges(resolver, f"{module}.py::{qualname}", expected["callee_expr"])
        assert any(
            e["resolution"] == expected["resolution"]
            and e["confidence"] >= expected["min_confidence"]
            and (resolver.by_id[e["callee_id"]]["qualname"] if e["callee_id"] else None)
            == expected["callee_qualname"]
            for e in result
        ), expected


def test_entrypoints(resolver):
    result = detect_entrypoints(resolver.symbols)
    assert result["entrypoint.py::entrypoint::<module>"] == "main_guard"
    symbols = [
        {"id": "guard", "kind": "module", "raw_code": 'if __name__ == "__main__":\n run()'},
        {"id": "cli", "decorators": ["app.command()"]},
        {"id": "route", "decorators": ['app.get("/")']},
        {"id": "task", "decorators": ["queue.task"]},
        {"id": "export", "qualname": "f", "exports": ["f"]},
        {"id": "false", "decorators": ["app.getter()"]},
    ]
    assert detect_entrypoints(symbols) == {
        "guard": "main_guard",
        "cli": "cli",
        "route": "route",
        "task": "task",
        "export": "exported",
    }


def test_nested_import_and_self_typed_resolution():
    from dataclasses import asdict
    from pathlib import Path

    from cfl.parser.python_adapter import PythonAdapter

    source = (
        "from utils import Helper\nclass Base:\n def save(self): pass\n"
        "class Child(Base):\n def __init__(self, dep: Helper): self.dep = dep\n"
        " def run(self):\n  self.save()\n  self.dep.work()\n"
        "def outer():\n def inner(): pass\n inner()"
    )
    parsed = PythonAdapter().parse(Path("src/models.py"), source)
    symbols = []
    for symbol in parsed.symbols:
        data = asdict(symbol)
        symbols.append(
            {
                **data,
                "id": f"m::{symbol.qualname}",
                "file_path": "src/models.py",
                "parent_id": f"m::{symbol.parent_qualname}" if symbol.parent_qualname else None,
            }
        )
    symbols.extend(
        [
            {
                "id": "helper",
                "name": "Helper",
                "qualname": "Helper",
                "kind": "class",
                "file_path": "src/utils.py",
            },
            {
                "id": "work",
                "name": "work",
                "qualname": "Helper.work",
                "kind": "method",
                "file_path": "src/utils.py",
            },
        ]
    )
    resolver = Resolver(
        symbols,
        [
            {"path": "src/models.py", "imports": [asdict(i) for i in parsed.imports]},
            {"path": "src/utils.py"},
        ],
    )
    assert edges(resolver, "m::outer", "inner")[0]["callee_id"] == "m::outer.inner"
    assert edges(resolver, "m::Child.run", "self.save")[0]["callee_id"] == "m::Base.save"
    edge = edges(resolver, "m::Child.run", "self.dep.work")[0]
    assert (edge["callee_id"], edge["resolution"]) == ("work", "typed")
