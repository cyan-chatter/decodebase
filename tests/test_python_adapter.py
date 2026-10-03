from __future__ import annotations

from pathlib import Path

from cfl.parser.base import get_adapter
from cfl.parser.python_adapter import PythonAdapter


def parse(source):
    return PythonAdapter().parse(Path("sample.py"), source)


def test_async_function():
    symbol = parse("async def f(): pass").symbols[0]
    assert symbol.kind == "function"
    assert symbol.is_async
    assert get_adapter("python").language == "python"


def test_decorator_extraction():
    symbol = parse("class C:\n @property\n def x(self): pass").symbols[1]
    assert symbol.decorators == ["property"]
    assert symbol.start_line == 2
    assert [(c.expr, c.kind) for c in symbol.call_sites] == [("property", "decorator")]


def test_nested_function_kind():
    outer, inner = parse("def outer():\n def inner(): pass").symbols
    assert outer.kind == "function"
    assert (inner.kind, inner.qualname, inner.parent_qualname) == ("nested", "outer.inner", "outer")


def test_class_with_init():
    symbol = parse("class C:\n def __init__(self, x: int):\n  self.val = x").symbols[1]
    assert symbol.init_attrs == {"val": "int"}
    assert symbol.param_types == {"x": "int"}


def test_comprehension_control_ctx():
    symbol = parse("def f():\n [g(x) for x in xs]").symbols[0]
    assert symbol.call_sites[0].control_ctx == "comp"


def test_syntax_error():
    parsed = parse("def f(:")
    assert parsed.parse_error
    assert parsed.symbols == []


def test_main_guard_produces_module_symbol():
    symbol = parse("if __name__ == '__main__':\n run()").symbols[0]
    assert symbol.kind == "module"
    assert symbol.qualname == "sample::<module>"
    assert symbol.call_sites[0].expr == "run"


def test_call_sites_do_not_descend_into_nested():
    outer, inner = parse("def outer():\n def inner():\n  g()\n h()").symbols
    assert [c.expr for c in outer.call_sites] == ["h"]
    assert [c.expr for c in inner.call_sites] == ["g"]


def test_method_kind():
    assert parse("class C:\n def f(self): pass").symbols[1].kind == "method"


def test_imports_collected():
    imports = parse("import os\nfrom sys import path as p").imports
    assert len(imports) == 2
    assert (imports[1].is_from, imports[1].alias) == (True, "p")


def test_skeleton_and_statement_spans():
    adapter = PythonAdapter()
    source = "async def f(x: int) -> int:\n if x:\n  return g(x)\n return 0"
    assert adapter.skeleton(parse(source)) == "async def f(x: int) -> int:"
    assert adapter.statement_spans(source) == [(2, 3), (4, 4)]
    assert adapter.statement_spans("  def f():\n    g()") == [(2, 2)]


def test_class_body_and_conditional_definitions():
    parsed = parse("class C:\n value = factory()\n def f(self): g()\nif True:\n def h(): pass")
    assert [c.expr for c in parsed.symbols[0].call_sites] == ["factory"]
    assert parsed.symbols[2].qualname == "h"


def test_optional_types_and_exports():
    parsed = parse("__all__ = ['f']\ndef f(x: Optional[C], y: C | None): pass")
    assert parsed.exports == ["f"]
    assert parsed.symbols[0].param_types == {"x": "C", "y": "C"}


def test_async_signature_has_one_prefix():
    parsed = parse("async def f(x: int) -> int:\n return x")
    assert parsed.symbols[0].signature == "async x: int -> int"
    assert PythonAdapter().skeleton(parsed) == "async def f(x: int) -> int:"


def test_source_encoding_fallback():
    from cfl.parser.python_adapter import decode_source

    assert "café" in decode_source(b'# coding: latin-1\nvalue = "caf\xe9"\n')
    assert "caf�" in decode_source(b'value = "caf\xff"\n')
    assert "ok" in decode_source(b'# coding: nonexistent\nvalue = "ok"\n')


def test_nested_types_do_not_leak_across_scopes():
    parsed = parse(
        "class C:\n def __init__(self):\n  if True:\n   self.repo = Repo()\n"
        "  def nested(): self.repo = Other()\n"
        "def f():\n if True:\n  x = Repo()\n def nested(): y = Other()"
    )
    by_name = {symbol.qualname: symbol for symbol in parsed.symbols}
    assert by_name["C.__init__"].init_attrs == {"repo": "Repo"}
    assert by_name["f"].local_types == {"x": "Repo"}


def test_try_and_match_definitions_and_async_control_contexts():
    parsed = parse(
        "try:\n g()\nexcept Exception:\n def f(): pass\n"
        "match value:\n case 0:\n  def h(): pass\n"
        "async def a():\n async for x in items:\n  async with x:\n   g()"
    )
    by_name = {symbol.qualname: symbol for symbol in parsed.symbols}
    assert {"f", "h", "a"} <= set(by_name)
    assert by_name["a"].call_sites[0].control_ctx == "for>with"


def test_conditional_imports_and_forward_optional_annotations():
    parsed = parse(
        'if TYPE_CHECKING:\n from package import Repo\ndef f(x: "typing.Optional[Repo]"): pass'
    )
    assert parsed.imports[0].name == "Repo"
    assert parsed.symbols[0].param_types == {"x": "Repo"}
    parsed = PythonAdapter().parse(Path("__init__.py"), "import package.mod as public_module")
    assert parsed.exports == ["public_module"]
