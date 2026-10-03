# Milestone 2 Implementation Plan

## Ground truth from exploration

**Stack:** Python 3.14, psycopg3, networkx 3.7, pytest 8, ruff 0.16, venv at `.venv/`.  
**Conventions observed:**
- Every file opens with `from __future__ import annotations`
- No SQL outside `cfl/core/db.py` (already enforced there)
- `replace_edges(conn, source, rows)` takes a `list[dict]` with keys: `caller_id`, `callee_id`, `callee_expr`, `line`, `kind`, `resolution`, `source`, `confidence`, `control_ctx` — callee_id is `None` for external edges (the column is nullable in the schema)
- `set_graph_metrics(conn, rows)` takes `list[dict]` with keys: `id`, `pagerank`, `scc_id`, `layer`
- `symbol_id(path, qualname, start_line, ambiguous)` from `cfl/core/hashing.py` — returns `{path}::{qualname}` or `{path}::{qualname}@{start_line}` when ambiguous
- `code_hash(raw_code)` normalizes before hashing
- `ParsedRow` dataclass in `db.py` has: `kind, qualname, name, parent_qualname, signature, decorators, docstring, start_line, end_line, raw_code, code_hash, token_est, call_sites, extra, is_async`
- `sync_symbols(conn, file_path, parsed_rows)` derives `symbol_id` itself from `file_path::qualname`
- Tests: `pg_conn` fixture creates isolated ephemeral DB; `@pytest.mark.db` gates DB tests; `fixture_repo_path` is already declared in `conftest.py` pointing at `tests/fixtures/resolution_repo/`
- `symbols` table `kind` CHECK: `'function' | 'method' | 'class' | 'nested' | 'module'`
- `cfl/parser/__init__.py` exists (empty); `cfl/pipeline/scan.py` already has `run_stage1`
- `set_view_status(conn, view, status, config)` — already implemented in `db.py`
- `Settings.edge_conf_threshold` = 0.6 default; `Settings.dsn` available
- Baseline: 14 non-db tests pass

---

## Decisions

1. **`ParsedFile.imports` stores `ImportEntry` as dicts for JSON serialisation** (same pattern as `call_sites` in `ParsedRow` — stored as `list` in JSONB column via `json.dumps`). The dataclass stays typed; the DB layer serialises. No change to db.py needed.

2. **`Resolver` is pure in-memory** — takes pre-loaded symbol lists/dicts; never queries the DB itself. `run_resolution` is the thin DB-facing wrapper that loads data, calls `Resolver`, then writes edges. This keeps SQL isolated in `db.py` and makes `Resolver` fully unit-testable without a DB.

3. **`resolve_fingerprint` is stored in `meta` via `set_meta/get_meta`** (already in db.py). Key: `'resolve_fingerprint'`. Skip resolution when fingerprint unchanged.

4. **`condense()` returns a flat dict `{symbol_id: {"scc_id": str, "layer": int}}` plus a `"__sccs__"` key mapping `scc_id → [symbol_ids]`**. This avoids a separate return type and lets callers destructure easily. SCC ids are stable strings like `"scc-{condensation_node_id}"`.

5. **`processing_order` returns `list[WorkItem]`** where `WorkItem` is a dataclass in `graph.py` with `scc_id: str, symbol_ids: list[str], layer: int`. Exported from `cfl/parser/graph.py`.

6. **`run_stage2` loads `Settings` from the argument** (passed in by the caller) — no `get_settings()` call inside pipeline functions, matching the existing `run_stage1` pattern which also receives `settings` as parameter.

7. **`scan` CLI command** added to `cfl/cli.py` as a new `@app.command()` that calls `run_stage1` then `run_stage2`. It follows the same pattern as the existing `build` command (takes REPO argument). A separate simpler `scan` command is added rather than filling in `build`, since `build` is reserved for the full pipeline (parse + embed + summarise).

8. **`test_resolver.py` and `test_graph.py` are NOT marked `@pytest.mark.db`** — they use the fixture repo files + an in-memory Resolver, so they run without Postgres. Only the integration path inside `run_resolution`/`run_graph_stage` (which calls real DB functions) would need `@pytest.mark.db`; those are not tested in M2.

9. **`skeleton()` in `PythonAdapter`** emits one line per symbol: `[async] def qualname(signature):` or `class qualname(bases):`, newline-separated, with no bodies.

10. **`detect_entrypoints` writes to DB** via a direct `UPDATE symbols SET is_entrypoint=TRUE, entry_kind=%s WHERE id=%s` — but since no SQL may live outside `db.py`, this gets a new helper `set_entrypoints(conn, rows: list[dict])` added to `db.py`. The dict keys are `id` and `entry_kind`.

---

## Items

- [ ] 1. Create `cfl/parser/base.py` — all dataclasses, Protocol, and registry.

  Define:
  - `@dataclass ImportEntry(module, name, alias, level, line, is_from)` — all fields typed
  - `@dataclass CallSite(expr, line, kind, control_ctx)` — `kind` is `'call'|'decorator'`
  - `@dataclass ParsedSymbol` — all fields from spec (kind, qualname, name, parent_qualname, signature, decorators, bases, docstring, start_line, end_line, raw_code, is_async, init_attrs, param_types, local_types, call_sites)
  - `@dataclass ParsedFile(path, language, imports, exports, symbols, parse_error)` — `parse_error: str | None`
  - `class LanguageAdapter(Protocol)` with `language: str`, `extensions: tuple[str, ...]`, `parse(path, source) -> ParsedFile`, `statement_spans(raw_code) -> list[tuple[int,int]]`, `skeleton(parsed) -> str`
  - Module-level `_adapter_registry: dict[str, LanguageAdapter] = {}`
  - `register_adapter(adapter) -> None`
  - `get_adapter(language) -> LanguageAdapter` — raises `KeyError` with a descriptive message

  **Files:** `cfl/parser/base.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/python -c "from cfl.parser.base import ParsedFile, LanguageAdapter, get_adapter; print('ok')"` — prints `ok`.

- [ ] 2. Add `set_entrypoints` helper to `cfl/core/db.py`.

  Append a new function `set_entrypoints(conn, rows: list[dict]) -> None` that iterates `rows` (each with `id` and `entry_kind`) and runs `UPDATE symbols SET is_entrypoint = TRUE, entry_kind = %s WHERE id = %s`. This is the only DB write needed by `detect_entrypoints` in the resolver, and it must live in `db.py` per the no-SQL-elsewhere constraint.

  **Files:** `cfl/core/db.py`  
  **Verify:** Existing test suite still passes — `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest -q -m 'not db'` — 14 passed, 0 failures.

- [ ] 3. Create `tests/fixtures/resolution_repo/` with all Python source files and JSON expectation files.

  Create these files:
  - `tests/fixtures/resolution_repo/__init__.py` — empty
  - `tests/fixtures/resolution_repo/utils.py` — `def helper(): pass` and `def process(): pass`
  - `tests/fixtures/resolution_repo/models.py` — class `Base` with `save(self)`; class `Repository(Base)` with `__init__(self, conn)` setting `self.conn = conn` and method `save(self)` calling `self.conn.execute()`; standalone function `save_record()` calling both `Base().save()` and `Repository(None).save()` (two candidates → ambiguous)
  - `tests/fixtures/resolution_repo/services.py` — `from models import Repository`; `def create_task(repo: Repository): repo.save(); r = Repository(None); r.save()`
  - `tests/fixtures/resolution_repo/aliases.py` — `from utils import process as proc`; `def run(): proc()`
  - `tests/fixtures/resolution_repo/recursion.py` — `def is_even(n): return is_odd(n-1) if n else True`; `def is_odd(n): return is_even(n-1) if n else False`; `def factorial(n): return n * factorial(n-1) if n > 1 else 1`
  - `tests/fixtures/resolution_repo/entrypoint.py` — `from aliases import run`; `if __name__ == '__main__': run()` (module pseudo-symbol)
  - `tests/fixtures/resolution_repo/external.py` — `import os`; `def do_work(): os.path.join('a', 'b'); len([1]); d = {}; d.get('key')`
  - `tests/fixtures/resolution_repo/expected_edges.json` — list of edge objects: `{caller, callee_expr, callee_qualname, resolution, min_confidence}`. Must include:
    - `aliases.run` → `process` via import alias (resolution=`import`, callee_qualname=`utils.process`)
    - `services.create_task` → `Repository.save` via typed param `repo` (resolution=`typed`)
    - `services.create_task` → `Repository.save` via typed local var `r` (resolution=`typed`)
    - `models.Repository.save` → `self.conn.execute` (resolution=`self`, callee_qualname=`null` — external, `execute` is blocklisted)
    - `models.save_record` entries — ambiguous (two `save` candidates), resolution=`ambiguous`
    - `recursion.is_even` → `recursion.is_odd`, `recursion.is_odd` → `recursion.is_even` (resolution=`local` or `name-unique`)
    - `recursion.factorial` → `recursion.factorial` (self-recursion, resolution=`local`)
    - `external.do_work` → `os.path.join` (external, callee_qualname=`null`)
    - `external.do_work` → `len` (external, callee_qualname=`null`)
    - `external.do_work` → `d.get` — blocklisted, should NOT appear or should have callee_qualname=`null`
  - `tests/fixtures/resolution_repo/expected_sccs.json` — list of SCCs (each a sorted list of qualnames). Must contain: `["is_even", "is_odd"]` as one SCC; `["factorial"]` alone; all other symbols as single-member SCCs.

  **Files:** All 10 files above in `tests/fixtures/resolution_repo/`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/python -c "import json, pathlib; p = pathlib.Path('tests/fixtures/resolution_repo'); print(sorted(f.name for f in p.iterdir()))"` — lists all expected filenames.

- [ ] 4. Create `cfl/parser/python_adapter.py` — `PythonAdapter` implementing `LanguageAdapter`.

  **Encoding detection:** `tokenize.detect_encoding` on `source.encode('latin-1')` (round-trip safe); fall back to UTF-8 with `errors='replace'` on exception.

  **`parse(path, source) -> ParsedFile`:**
  - `ast.parse(source)` wrapped in try/except `(SyntaxError, ValueError, RecursionError, MemoryError)` → return `ParsedFile(path=str(path), language='python', imports=[], exports=[], symbols=[], parse_error=str(e))`
  - Source split into a `lines: list[str]` once; `raw_code` for any symbol = `'\n'.join(lines[start_line-1:end_line])`
  - **Walker:** `ast.NodeVisitor` subclass `_SymbolVisitor`. Tracks a `_scope: list[ast.AST]` stack.
    - `visit_FunctionDef` / `visit_AsyncFunctionDef`: determine `kind` from depth (top-level → `'function'`; direct child of a `ClassDef` on the scope stack → `'method'`; otherwise → `'nested'`). Build `qualname` as dot-join of all def/class names on the stack (no `<locals>`). `start_line = min(node.lineno, node.decorator_list[0].lineno if node.decorator_list else node.lineno)`. `end_line = node.end_lineno`. Extract `signature` via `ast.unparse` of `node.args` + returns annotation. Extract `docstring` via `ast.get_docstring(node, clean=True)`. Extract `decorators` as `[ast.unparse(d) for d in node.decorator_list]`. For `__init__` methods: scan body for `self.attr = value` assignments → `init_attrs`; param annotations → `param_types` (unwrap `Optional[X]`/`X | None`). Extract `param_types` for all methods from annotations. Push node to `_scope`, visit children (stops descending into nested def/class for call-site purposes — see call visitor below), pop.
    - `visit_ClassDef`: kind=`'class'`. `bases` = `[ast.unparse(b) for b in node.bases]`. Push, visit, pop.
    - For each symbol, run a second targeted `_CallVisitor` that starts at the symbol's AST node: collects `CallSite` objects. Does NOT descend into nested `FunctionDef`/`AsyncFunctionDef`/`ClassDef` bodies (they own their calls). DOES descend into `Lambda`, `ListComp`, `SetComp`, `DictComp`, `GeneratorExp`. Tracks control context stack: `ast.If`→`'if'`; `ast.For`→`'for'`; `ast.While`→`'while'`; `ast.Try`→`'try'`; `ast.With`→`'with'`; comprehension iter/condition→`'comp'`. `control_ctx = '>'.join(stack)` (empty string when no context). `expr = ast.unparse(node.func)[:200]`.
    - Decorator call sites: for each decorator that is a `Call` node (or an `Attribute`/`Name` used as decorator), emit a `CallSite` with `kind='decorator'` on the decorated symbol.
    - Class-body calls: calls in the class body (not in any method) belong to the class symbol's call_sites.
  - **Module pseudo-symbol (D3):** After collecting all def/class symbols, scan top-level statements: skip `Import`, `ImportFrom`, `FunctionDef`, `AsyncFunctionDef`, `ClassDef`. Collect remaining statements. If any remaining statement contains a `ast.Call` node anywhere in its subtree, create one `ParsedSymbol` with `qualname = path.stem + '::<module>'`, `kind = 'module'`, `name = '<module>'`, `parent_qualname = None`, `start_line = min(stmt.lineno for stmt in remaining)`, `end_line = max(stmt.end_lineno for stmt in remaining)`, `raw_code = '\n'.join(lines)` for those statement ranges concatenated, `call_sites = []` (calls extracted separately using `_CallVisitor` on those statements), `is_async = False`.
  - **`exports`:** check for top-level `__all__` assignment; `ast.literal_eval` its value. For `__init__.py`: also collect imported names that do not start with `_`.
  - **`imports`:** all `ast.Import` and `ast.ImportFrom` at module level → `ImportEntry` list.
  - **`local_types`:** in function bodies, scan for `Name = Call(func=Name(...))` assignments; map local var name to `ast.unparse(node.func)`.

  **`statement_spans(raw_code) -> list[tuple[int,int]]`:**
  - `ast.parse(raw_code)` to get the body of the first `FunctionDef` (or the module body). Return `[(stmt.lineno, stmt.end_lineno) for stmt in body]`.

  **`skeleton(parsed) -> str`:**
  - For each symbol in `parsed.symbols`, emit: `"async def {qualname}({sig}):"` for async functions/methods; `"def {qualname}({sig}):"` for sync; `"class {qualname}({', '.join(bases)}):"` for classes. One symbol per line, no body.

  **Registration:** module-level `register_adapter(PythonAdapter())`.

  **Files:** `cfl/parser/python_adapter.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_python_adapter.py -q` — all tests pass.

- [ ] 5. Create `tests/test_python_adapter.py` — unit tests for `PythonAdapter`.

  Write tests covering (no DB required, no `@pytest.mark.db`):
  1. `test_async_function`: parse a snippet with `async def f(): pass` → symbol has `kind='function'`, `is_async=True`.
  2. `test_decorator_extraction`: parse `@property\ndef x(self): pass` inside a class → `symbol.decorators == ['property']`; a `CallSite` with `kind='decorator'` appears on the method.
  3. `test_nested_function_kind`: parse `def outer():\n  def inner(): pass` → outer has `kind='function'`, inner has `kind='nested'`.
  4. `test_class_with_init`: parse a class with `__init__(self, x: int)` and `self.val = x` → `init_attrs == {'val': ...}`, `param_types` has `'x': 'int'`.
  5. `test_comprehension_control_ctx`: parse `def f():\n  [g(x) for x in xs]` → call `g` has `control_ctx='comp'`.
  6. `test_syntax_error`: parse `'def f(:'` → `ParsedFile.parse_error` is not None, `symbols == []`.
  7. `test_main_guard_produces_module_symbol`: parse `if __name__ == '__main__':\n    run()` → a symbol of `kind='module'` is present.
  8. `test_call_sites_do_not_descend_into_nested`: parse `def outer():\n  def inner():\n    g()\n  h()` → `outer.call_sites` contains `h` but NOT `g`; inner is a separate symbol with `g` in its call_sites.
  9. `test_method_kind`: parse a class with a method → symbol has `kind='method'`.
  10. `test_imports_collected`: parse `import os\nfrom sys import path as p` → `ParsedFile.imports` has two entries; the second has `is_from=True`, `alias='p'`.

  **Files:** `tests/test_python_adapter.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_python_adapter.py -q` — all tests pass.

- [ ] 6. Create `cfl/parser/resolver.py` — `Resolver` class and `run_resolution`.

  **Module-level constant:**
  ```python
  COMMON_METHOD_BLOCKLIST: frozenset[str] = frozenset({
      "get", "set", "items", "keys", "values", "append", "extend", "pop",
      "update", "add", "remove", "join", "split", "strip", "format",
      "read", "write", "close", "open", "copy", "sort", "count", "index",
      "lower", "upper", "encode", "decode", "replace", "find",
      "startswith", "endswith",
      "__init__", "__repr__", "__str__", "__len__", "__iter__", "__next__",
      "__enter__", "__exit__", "__eq__", "__hash__",
  })
  ```

  **`Resolver.__init__(symbols: list[dict], files: list[dict])`** — receives raw DB dicts (from `get_symbols` / `list_files`):
  - `by_id: dict[str, dict]` — symbol_id → symbol dict
  - `by_file_qualname: dict[tuple[str,str], str]` — (file_path, qualname) → symbol_id
  - `by_name: dict[str, list[str]]` — simple name → [symbol_ids], excluding blocklisted names and dunder methods
  - `class_bases: dict[str, list[str]]` — class symbol_id → list of resolved base class symbol_ids (resolved through import maps when the base name appears in the file's imports)
  - `import_maps: dict[str, dict[str, tuple[str, str]]]` — file_path → {alias: (resolved_file_path, exported_name)}. Build from each symbol's file's imports JSONB. Handle `from module import name as alias` and `import module as alias`. Use `module_index` for file resolution.
  - `module_index: dict[str, str]` — dotted module path → file_path. Built from `files` list: `path.stem` and `path` without extension, plus `__init__` handling. Also strip leading `src/` when looking up.
  - `per_class_init_attr_types: dict[str, dict[str, str]]` — class symbol_id → {attr: type_hint} from the class's `__init__` symbol's `extra` or by re-examining `init_attrs` stored in the symbol's `extra` JSONB.

  **`resolve_call(caller_id: str, call_site: dict) -> list[dict]`** — `call_site` is a dict from JSONB (has `expr`, `line`, `kind`, `control_ctx`). Returns a list of edge-row dicts (keys matching `replace_edges` schema). Apply layers in order; stop at first non-empty result except layer 7 (ambiguous) which collects all:

  1. **import** (confidence 0.95): Split `expr` on `.` → head. If head is in the file's import_map, resolve to the target file and exported name. If the resolved name is a class, also add an edge to its `__init__` with kind=`'constructor'`.
  2. **local** (confidence 0.90): Head matches a symbol in the same file (by `by_file_qualname[(file_path, head)]`). Includes nested symbols within the caller's qualname prefix. If head is a class name → kind=`'constructor'`.
  3. **self** (confidence 0.90): `expr` starts with `self.` or `cls.` → strip prefix, look up method name in the enclosing class (find class by walking caller's qualname up). Walk `class_bases` recursively to find inherited methods.
  4. **ctor** (confidence 0.90): `expr` matches a class name in `by_name` (unique) and ends with `(` in context — i.e., the call site expr is just a name. Emit edge to class + edge to `ClassName.__init__` if it exists.
  5. **typed** (confidence 0.80): Head is a local variable with a known type (from `local_types` in the caller symbol's extra JSONB) or an annotated parameter (from `param_types` in the caller's extra JSONB) or a `self.attr` with known type from `per_class_init_attr_types`. Resolve the method on that type's class symbol.
  6. **name-unique** (confidence 0.75): The last attribute segment of `expr` (i.e., `expr.rsplit('.', 1)[-1]`) matches exactly one symbol in `by_name` and is not in `COMMON_METHOD_BLOCKLIST`.
  7. **ambiguous** (confidence 0.40): Last attribute segment matches multiple symbols in `by_name` and not blocklisted — emit one edge row per candidate, all with confidence 0.40.
  8. **external** (confidence 0.0): `callee_id=None`; emit one edge row with the expression.

  Each returned edge-row dict has: `caller_id`, `callee_id` (str or None), `callee_expr`, `line`, `kind` (from call_site kind, upgraded to `'constructor'` when appropriate), `resolution` (layer name string), `source='ast'`, `confidence`, `control_ctx`.

  **`detect_entrypoints(symbols: list[dict]) -> dict[str, str]`** — returns `{symbol_id: entry_kind}`:
  - Symbol with `name == 'main'` → `'main'`
  - Module pseudo-symbols (kind=`'module'`) whose `raw_code` contains `__name__ == '__main__'` → `'main_guard'`
  - Symbols whose decorators (stored as JSONB list) match any of: `*.route`, `*.get`, `*.post`, `*.put`, `*.delete`, `*.patch`, `*.command`, `*.task`, `*.callback` → `'route'` or `'cli'` or `'task'` as appropriate
  - Symbols in files named `__main__.py` → `'main_module'`
  - Symbols whose `qualname` appears in the file's `exports` JSONB list → `'exported'`

  **`run_resolution(conn, settings) -> None`:**
  - Compute `resolve_fingerprint`: `join_hash` of all file sha256 values (sorted by path) + `"resolver_v1"` + `str(settings.edge_conf_threshold)`. Check against `get_meta(conn, 'resolve_fingerprint')`; if equal, log and return early.
  - Load all symbols via `get_symbols(conn, ids)` where ids come from `conn.execute("SELECT id FROM symbols").fetchall()`. Load all files via `list_files(conn)`.
  - Build `Resolver(symbols, files)`.
  - For each symbol: parse `call_sites` from symbol's JSONB; call `resolve_call` for each call site; collect all edge rows.
  - Call `replace_edges(conn, 'ast', all_edge_rows)` in one pass.
  - Call `detect_entrypoints(symbols)` → call `set_entrypoints(conn, rows)`.
  - Store new fingerprint: `set_meta(conn, 'resolve_fingerprint', fingerprint)`.

  **Files:** `cfl/parser/resolver.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_resolver.py -q` — all tests pass.

- [ ] 7. Create `tests/test_resolver.py` — unit tests for `Resolver` (no DB required).

  Tests load the `resolution_repo` fixture directly via `pathlib.Path`, parse all `.py` files with `PythonAdapter`, build symbol dicts manually (or via a helper), construct a `Resolver`, run `resolve_call` for each call site, then compare results to `expected_edges.json`.

  Specific test cases:
  1. `test_import_alias_resolution`: `aliases.run` → calls `proc()` which resolves via import map to `utils.process`. Assert an edge row with `resolution='import'` and `callee_id` pointing to `utils.process`.
  2. `test_typed_call_via_param`: `services.create_task(repo: Repository)` → `repo.save()` resolves via typed param to `Repository.save`. Assert `resolution='typed'`.
  3. `test_typed_call_via_local_var`: `r = Repository(None); r.save()` in `create_task` → resolves to `Repository.save` via local var typing. Assert `resolution='typed'`.
  4. `test_ambiguous_candidates_all_present`: `save_record` calls produce ambiguous edges (two `save` methods). Assert two edge rows both with `resolution='ambiguous'` and `confidence=0.40`.
  5. `test_external_edge_has_null_callee`: `external.do_work` → `os.path.join` → edge with `callee_id=None`.
  6. `test_blocklist_prevents_linking`: `d.get('key')` in `external.do_work` → `get` is in `COMMON_METHOD_BLOCKLIST` → no edge linking to any project symbol named `get`. The edge, if any, has `callee_id=None` (external) or does not appear.
  7. `test_self_recursion_resolves_local`: `recursion.factorial` → `factorial(n-1)` → resolves with `resolution='local'` to itself.
  8. `test_mutual_recursion_edges`: both `is_even→is_odd` and `is_odd→is_even` edges exist.
  9. `test_all_expected_edges`: iterate `expected_edges.json`; for each entry with a non-null `callee_qualname`, assert that at least one resolved edge row with `confidence >= min_confidence` exists mapping caller → callee.

  **Files:** `tests/test_resolver.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_resolver.py -q` — all tests pass.

- [ ] 8. Create `cfl/parser/graph.py` — graph construction, condensation, metrics, and processing order.

  **Imports:** `from __future__ import annotations`, `networkx as nx`, `dataclasses`, `typing`, `cfl.core.db` (for `fetch_edges`, `get_symbols`, `set_graph_metrics`, `set_view_status`).

  **`build_call_graph(conn, threshold: float) -> nx.DiGraph`:**
  - Fetch edges with `fetch_edges(conn, min_conf=threshold)`.
  - Add node for each unique `caller_id` and `callee_id` (non-null only).
  - Add edge `(caller_id, callee_id)` with `weight = max confidence` when the same pair appears multiple times (take max).
  - Return the DiGraph.

  **`build_order_graph(call_graph: nx.DiGraph, symbols: list[dict]) -> nx.DiGraph`:**
  - Start from a copy of `call_graph`.
  - For each symbol with `kind='class'`, add directed edges from the class symbol_id to each method symbol_id (method's `parent_id == class symbol_id`). Also add an edge from each module-pseudo-symbol node to the class/function nodes defined in the same file.
  - Return the augmented DiGraph.

  **`condense(order_graph: nx.DiGraph) -> dict`:**
  - `sccs = list(nx.strongly_connected_components(order_graph))` — each is a frozenset.
  - Use `nx.condensation(order_graph)` to get the DAG of SCCs.
  - Compute `layer` per SCC node in condensation: layer 0 = sink nodes (no outgoing edges in condensation = no dependencies the SCC calls); layer n = 1 + max(layer of successors). (Callee-before-caller: callees are lower layers.)
  - Build result dict: `{symbol_id: {"scc_id": f"scc-{condensation_node}", "layer": layer}}` for every symbol_id, plus `"__sccs__": {scc_id: sorted(list(member_ids))}`.
  - **NEVER call `nx.simple_cycles`.**

  **`compute_pagerank(call_graph: nx.DiGraph) -> dict[str, float]`:**
  - `nx.pagerank(call_graph, alpha=0.85, weight='weight')`.
  - Return the dict.

  **`compute_communities(call_graph: nx.DiGraph) -> dict[str, int]`:**
  - Convert to undirected: `call_graph.to_undirected()`.
  - `nx.community.louvain_communities(undirected, weight='weight', seed=42)`.
  - Map each node to its community index.

  **`@dataclass WorkItem(scc_id: str, symbol_ids: list[str], layer: int)`.**

  **`processing_order(conn, *, priority: str | None = None) -> list[WorkItem]`:**
  - Load symbols, call `build_call_graph(conn, threshold=0.0)` (include all edges for ordering).
  - `build_order_graph(call_graph, symbols)`.
  - `condense(order_graph)` → condensation dict.
  - `compute_pagerank(call_graph)` → pagerank dict.
  - Group symbol_ids by scc_id, compute max pagerank per scc.
  - Sort SCCs: primary key = layer (ascending), secondary key = max pagerank of SCC members (descending).
  - If `priority == 'entrypoints-first'`: symbols reachable from entrypoints (symbols where `is_entrypoint=True`) get a bonus — place their SCCs first within each layer by setting a flag.
  - Return `list[WorkItem]` ordered accordingly.

  **`run_graph_stage(conn, settings) -> None`:**
  - Load all symbols from DB.
  - `build_call_graph(conn, threshold=settings.edge_conf_threshold)`.
  - `build_order_graph(call_graph, symbols)`.
  - `condense_result = condense(order_graph)`.
  - `pagerank = compute_pagerank(call_graph)`.
  - `communities = compute_communities(call_graph)`.
  - Build `set_graph_metrics` rows: one per symbol with `id`, `pagerank` (from pagerank dict, default 0.0), `scc_id`, `layer`.
  - Call `set_graph_metrics(conn, rows)`.
  - Call `set_view_status(conn, 'graph', 'fresh', {'threshold': settings.edge_conf_threshold})`.

  **Files:** `cfl/parser/graph.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_graph.py -q` — all tests pass.

- [ ] 9. Create `tests/test_graph.py` — unit tests for graph and condensation (no DB required).

  Build an in-memory `nx.DiGraph` from the resolution_repo expected edges rather than a real DB connection, then test the graph functions directly.

  Tests:
  1. `test_mutual_recursion_is_one_scc`: build a DiGraph with edges `is_even→is_odd` and `is_odd→is_even` plus `factorial→factorial`. Call `condense()` on the result. Assert `is_even` and `is_odd` share the same `scc_id`. Assert `factorial` is in its own SCC.
  2. `test_layers_respect_callee_before_caller`: build graph where `A→B→C` (A calls B, B calls C). Condense. Assert `layer[C] < layer[B] < layer[A]` (callees are lower layers, i.e., processed first).
  3. `test_class_sorts_after_methods`: build an order_graph with a class node `Foo` and method nodes `Foo.bar`, `Foo.baz` and structural edges `Foo→Foo.bar`, `Foo→Foo.baz`. Condense. Assert `layer[Foo] > layer[Foo.bar]` (class is processed after its methods).
  4. `test_expected_sccs_match_fixture`: load `expected_sccs.json`; parse the resolution_repo with `PythonAdapter`; build a synthetic call graph from `expected_edges.json` edges where `callee_qualname` is non-null; call `condense()`; extract SCC groups; assert they contain the same member sets as `expected_sccs.json` (compare as sets of frozensets).
  5. `test_pagerank_higher_for_more_called`: build a graph where `hub` is called by 5 nodes and `leaf` is called by 0. Assert `pagerank[hub] > pagerank[leaf]`.
  6. `test_processing_order_layers_ascending`: call `processing_order` on the synthetic graph. Assert that `WorkItem.layer` values are non-decreasing.

  **Files:** `tests/test_graph.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_graph.py -q` — all tests pass.

- [ ] 10. Extend `cfl/pipeline/scan.py` — wire Stage 2.

  Read the existing file carefully before editing (it already has `run_stage1`). Add:

  ```python
  def run_stage2(conn: Connection, settings: Settings, repo_root: str) -> None:
      """Stage 2: resolve call-graph edges and compute graph metrics."""
      from cfl.parser.resolver import run_resolution
      from cfl.parser.graph import run_graph_stage

      run_resolution(conn, settings)
      run_graph_stage(conn, settings)
  ```

  Also add a `scan` CLI command to `cfl/cli.py`. Read the existing `cli.py` first. Append:

  ```python
  @app.command()
  def scan(
      repo: str = typer.Argument(".", help="Path to the repository to scan"),
  ) -> None:
      """Scan a repository: discover files (Stage 1) and resolve call graph (Stage 2)."""
      from cfl.core.db import connect
      from cfl.pipeline.scan import run_stage1, run_stage2

      settings = get_settings()
      conn = connect(settings.dsn)
      try:
          run_stage1(conn, settings, repo)
          run_stage2(conn, settings, repo)
          console.print("[green]Scan complete.[/green]")
      finally:
          conn.close()
  ```

  **Files:** `cfl/pipeline/scan.py`, `cfl/cli.py`  
  **Verify:** `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/python -c "from cfl.pipeline.scan import run_stage2; print('ok')"` — prints `ok`. Then `/home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest -q -m 'not db'` — all existing tests still pass.

- [ ] 11. Final verification — full fast test suite.

  Run all four verification commands in order:
  ```
  /home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_python_adapter.py -q
  /home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_resolver.py -q
  /home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest tests/test_graph.py -q
  /home/cyan/shelf/cs/CodeFlowLens/.venv/bin/pytest -q -m 'not db'
  ```
  All must pass. Then run ruff on the new files:
  ```
  /home/cyan/shelf/cs/CodeFlowLens/.venv/bin/ruff check cfl/parser/ cfl/pipeline/scan.py cfl/cli.py tests/test_python_adapter.py tests/test_resolver.py tests/test_graph.py --line-length 100
  ```
  Zero errors.

  **Files:** none (verification only)  
  **Verify:** All four pytest commands exit 0; ruff exits 0.

---

## Dependency order summary

Items must be done in order: 1 → 2 → 3 → 4+5 (parallel: adapter and its tests together) → 6+7 (parallel: resolver and its tests together) → 8+9 (parallel: graph and its tests together) → 10 → 11.

Item 4 depends on item 1 (imports `ParsedFile` etc. from `base.py`).  
Item 5 depends on items 3 and 4 (tests need fixture files and the adapter).  
Item 6 depends on items 1, 2, and 4 (`ParsedSymbol`, `set_entrypoints`, `PythonAdapter` for the call_sites format).  
Item 7 depends on items 3, 4, and 6.  
Item 8 depends on item 2 (`db.py` functions already exist; just needs the new `set_entrypoints`).  
Item 9 depends on items 3, 4, and 8.  
Item 10 depends on items 6 and 8 (`run_resolution`, `run_graph_stage`).  
Item 11 depends on all prior items.
