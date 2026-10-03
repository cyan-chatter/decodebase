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
        result = edges(resolver, expected["caller_id"], expected["callee_expr"])
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


def resolver_for_sources(sources):
    from collections import Counter
    from dataclasses import asdict
    from pathlib import Path

    from cfl.core.hashing import symbol_id
    from cfl.parser.python_adapter import PythonAdapter

    files, symbols = [], []
    for path, source in sources.items():
        parsed = PythonAdapter().parse(Path(path), source)
        assert parsed.parse_error is None
        counts = Counter(symbol.qualname for symbol in parsed.symbols)
        ids = {
            symbol.start_line: symbol_id(
                path, symbol.qualname, symbol.start_line, counts[symbol.qualname] > 1
            )
            for symbol in parsed.symbols
        }
        files.append(
            {
                "path": path,
                "imports": [asdict(entry) for entry in parsed.imports],
                "exports": parsed.exports,
            }
        )
        for symbol in parsed.symbols:
            parents = [
                parent
                for parent in parsed.symbols
                if parent.qualname == symbol.parent_qualname
                and parent.start_line <= symbol.start_line <= parent.end_line
            ]
            symbols.append(
                {
                    **asdict(symbol),
                    "id": ids[symbol.start_line],
                    "file_path": path,
                    "parent_id": ids[parents[-1].start_line] if parents else None,
                }
            )
    return Resolver(symbols, files)


@pytest.mark.parametrize(
    "source,caller,expr",
    [
        ("def f(): pass\ndef f(): pass\ndef caller(): f()", "caller", "f"),
        (
            "class C:\n def f(self): pass\n def f(self): pass\n def caller(self): self.f()",
            "C.caller",
            "self.f",
        ),
        (
            "class C:\n def f(self): pass\n def f(self): pass\ndef caller(x: C): x.f()",
            "caller",
            "x.f",
        ),
    ],
)
def test_ambiguity_is_retained_at_local_self_and_typed_layers(source, caller, expr):
    resolver = resolver_for_sources({"a.py": source})
    result = edges(resolver, f"a.py::{caller}", expr)
    assert len(result) == 2
    assert len({e["callee_id"] for e in result}) == 2
    assert all(e["resolution"] == "ambiguous" and e["confidence"] == 0.4 for e in result)


def test_ambiguity_is_retained_at_import_layer():
    resolver = resolver_for_sources(
        {"a.py": "from b import f\ndef caller(): f()", "b.py": "def f(): pass\ndef f(): pass"}
    )
    result = edges(resolver, "a.py::caller", "f")
    assert {edge["callee_id"] for edge in result} == {"b.py::f@1", "b.py::f@2"}
    assert all(edge["resolution"] == "ambiguous" for edge in result)


def test_ambiguous_constructor_and_type_classes():
    resolver = resolver_for_sources(
        {
            "types.py": "class C:\n def f(self): pass\nclass C:\n def f(self): pass",
            "a.py": "def caller(x: C):\n C()\n x.f()",
        }
    )
    constructor = edges(resolver, "a.py::caller", "C")
    assert len(constructor) == 2
    assert all(e["resolution"] == "ambiguous" and e["kind"] == "constructor" for e in constructor)
    typed = edges(resolver, "a.py::caller", "x.f")
    assert len(typed) == 2
    assert all(e["resolution"] == "ambiguous" for e in typed)


def test_missing_self_method_is_external():
    resolver = resolver_for_sources({"a.py": "class C:\n def caller(self): self.missing()"})
    edge = edges(resolver, "a.py::C.caller", "self.missing")[0]
    assert (edge["callee_id"], edge["resolution"], edge["confidence"]) == (None, "external", 0.0)


def test_src_packages_module_aliases_and_relative_imports():
    resolver = resolver_for_sources(
        {
            "src/pkg/__init__.py": "from .mod import C",
            "src/pkg/mod.py": "class C:\n def f(self): pass",
            "src/pkg/client.py": "from . import mod as m\ndef relative(x: m.C): x.f()",
            "src/client.py": "from pkg import C\nimport pkg.mod as m\n"
            "def caller(x: m.C):\n x.f()\n C()",
        }
    )
    assert resolver.module_index["pkg"] == "src/pkg/__init__.py"
    assert resolver.module_index["pkg.mod"] == "src/pkg/mod.py"
    assert (
        edges(resolver, "src/pkg/client.py::relative", "x.f")[0]["callee_id"]
        == "src/pkg/mod.py::C.f"
    )
    assert edges(resolver, "src/client.py::caller", "x.f")[0]["resolution"] == "typed"
    assert edges(resolver, "src/client.py::caller", "C")[0]["resolution"] == "import"


def test_constructor_links_class_and_initializer(resolver):
    result = edges(resolver, "services.py::create_task", "Repository")
    assert {e["callee_id"] for e in result} == {
        "models.py::Repository",
        "models.py::Repository.__init__",
    }
    assert all(e["kind"] == "constructor" for e in result)


def test_name_unique_and_relative_import_fixture(resolver):
    edge = edges(resolver, "import_cases.py::call_unknown", "obj.unique_action")[0]
    assert (edge["resolution"], edge["callee_id"], edge["confidence"]) == (
        "name-unique",
        "utils.py::unique_action",
        0.75,
    )
    assert (
        edges(resolver, "pkg/relative.py::run_relative", "sibling.leaf")[0]["resolution"]
        == "import"
    )
