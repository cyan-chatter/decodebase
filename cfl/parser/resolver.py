from __future__ import annotations

import ast
import json
import logging
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg

    from cfl.config import Settings

logger = logging.getLogger(__name__)

COMMON_METHOD_BLOCKLIST: frozenset[str] = frozenset(
    {
        "get",
        "set",
        "items",
        "keys",
        "values",
        "append",
        "extend",
        "pop",
        "update",
        "add",
        "remove",
        "join",
        "split",
        "strip",
        "format",
        "read",
        "write",
        "close",
        "open",
        "copy",
        "sort",
        "count",
        "index",
        "lower",
        "upper",
        "encode",
        "decode",
        "replace",
        "find",
        "startswith",
        "endswith",
        "__init__",
        "__repr__",
        "__str__",
        "__len__",
        "__iter__",
        "__next__",
        "__enter__",
        "__exit__",
        "__eq__",
        "__hash__",
    }
)


def _method_name(expr: str) -> str:
    """Return the last attribute/name segment of expr."""
    return expr.rsplit(".", 1)[-1]


def _head(expr: str) -> str:
    """Return the first dotted segment of expr."""
    return expr.split(".")[0]


def _resolve_module_path(
    module: str,
    level: int,
    file_path: str,
    module_index: dict[str, str],
) -> str | None:
    """Resolve a possibly-relative import to an absolute file path."""
    if level == 0:
        # Absolute import
        candidates = [module, module.replace(".", "/")]
        for key in (module, module.replace(".", "/")):
            if key in module_index:
                return module_index[key]
        # Try stripping src/ prefix
        for key in candidates:
            stripped = key.removeprefix("src/")
            if stripped in module_index:
                return module_index[stripped]
        return None

    # Relative import: level=1 → same package, level=2 → parent package
    parts = file_path.replace("\\", "/").split("/")
    # Remove filename to get directory parts
    dir_parts = parts[:-level] if level <= len(parts) else []
    if module:
        dir_parts = dir_parts + module.split(".")
    target_key = "/".join(dir_parts)
    if target_key in module_index:
        return module_index[target_key]
    # Also try with just the stem
    stem_key = target_key
    if stem_key in module_index:
        return module_index[stem_key]
    return None


def _build_module_index(files: list[dict]) -> dict[str, str]:
    """Build dotted-module → file_path index from file records."""
    index: dict[str, str] = {}
    for f in files:
        path = f["path"]
        p = pathlib.PurePosixPath(path)
        # stem key: foo/bar.py → foo/bar
        stem_key = str(p.with_suffix(""))
        index[stem_key] = path
        # also dotted: foo/bar → foo.bar
        dotted = stem_key.replace("/", ".")
        index[dotted] = path
        if stem_key.startswith("src/"):
            short = stem_key.removeprefix("src/")
            index[short] = path
            index[short.replace("/", ".")] = path
        # __init__.py → package key
        if p.name == "__init__.py":
            pkg_key = str(p.parent)
            if pkg_key != ".":
                index[pkg_key] = path
                index[pkg_key.replace("/", ".")] = path
    return index


def _build_import_maps(
    files: list[dict],
    module_index: dict[str, str],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Build per-file alias → (resolved_file_path, exported_name) map."""
    maps: dict[str, dict[str, tuple[str, str]]] = {}
    for f in files:
        file_path = f["path"]
        raw_imports = f.get("imports") or []
        if isinstance(raw_imports, str):
            try:
                raw_imports = json.loads(raw_imports)
            except (ValueError, TypeError):
                raw_imports = []
        alias_map: dict[str, tuple[str, str]] = {}
        for imp in raw_imports:
            if not isinstance(imp, dict):
                continue
            is_from = imp.get("is_from", False)
            module = imp.get("module", "")
            name = imp.get("name")
            alias = imp.get("alias")
            level = imp.get("level", 0)

            resolved_path = _resolve_module_path(module, level, file_path, module_index)
            if resolved_path is None:
                continue

            if is_from and name:
                # from module import name [as alias]
                key = alias if alias else name
                alias_map[key] = (resolved_path, name)
            elif not is_from:
                # import module [as alias]
                key = alias if alias else module.split(".")[0]
                alias_map[key] = (resolved_path, module)
        maps[file_path] = alias_map
    return maps


class Resolver:
    """Pure in-memory call-site resolver.

    Accepts symbol dicts and file dicts (as returned from db.py helpers),
    builds internal lookup tables, and resolves call sites to edges.
    """

    def __init__(self, symbols: list[dict], files: list[dict]) -> None:
        self.symbols = symbols
        self.files = files

        self.module_index = _build_module_index(files)
        self.import_maps = _build_import_maps(files, self.module_index)

        # by_id: symbol_id → symbol dict
        self.by_id: dict[str, dict] = {s["id"]: s for s in symbols}

        # by_file_qualname: (file_path, qualname) → symbol_id
        self.by_file_qualname: dict[tuple[str, str], str] = {}
        for s in symbols:
            key = (s["file_path"], s["qualname"])
            self.by_file_qualname[key] = s["id"]

        # by_name: simple name → [symbol_ids]
        self.by_name: dict[str, list[str]] = {}
        for s in symbols:
            name = s["name"]
            if name in COMMON_METHOD_BLOCKLIST or name.startswith("__"):
                continue
            self.by_name.setdefault(name, []).append(s["id"])

        # class_bases: class_symbol_id → [resolved base class symbol_ids]
        self.class_bases: dict[str, list[str]] = {}
        for s in symbols:
            if s["kind"] != "class":
                continue
            raw_extra = s.get("extra") or {}
            if isinstance(raw_extra, str):
                try:
                    raw_extra = json.loads(raw_extra)
                except (ValueError, TypeError):
                    raw_extra = {}
            bases_raw = raw_extra.get("bases", [])
            if not bases_raw:
                # Try from ParsedSymbol stored in extra or from direct field
                # For dict symbols built directly from ParsedSymbol, check 'bases' key at top level
                bases_raw = s.get("bases") or []
            resolved_bases: list[str] = []
            file_path = s["file_path"]
            import_map = self.import_maps.get(file_path, {})
            for base_str in bases_raw:
                base_head = base_str.split(".")[0]
                # Try import map first
                if base_head in import_map:
                    resolved_file, exported_name = import_map[base_head]
                    candidate_key = (resolved_file, exported_name)
                    if candidate_key in self.by_file_qualname:
                        resolved_bases.append(self.by_file_qualname[candidate_key])
                        continue
                # Try same file
                candidate_key = (file_path, base_str)
                if candidate_key in self.by_file_qualname:
                    resolved_bases.append(self.by_file_qualname[candidate_key])
                    continue
                # Try by_name
                if base_head in self.by_name:
                    candidates = self.by_name[base_head]
                    if len(candidates) == 1:
                        resolved_bases.append(candidates[0])
            self.class_bases[s["id"]] = resolved_bases

        # per_class_init_attr_types: class_sym_id → {attr: type}
        # Gathered from __init__ method of each class
        self.per_class_init_attr_types: dict[str, dict[str, str]] = {}
        for s in symbols:
            if s["kind"] == "class":
                self.per_class_init_attr_types[s["id"]] = {}

        for s in symbols:
            if s["name"] == "__init__" and s.get("parent_id"):
                parent_id = s["parent_id"]
                if parent_id in self.per_class_init_attr_types:
                    # get init_attrs from extra or direct field
                    raw_extra = s.get("extra") or {}
                    if isinstance(raw_extra, str):
                        try:
                            raw_extra = json.loads(raw_extra)
                        except (ValueError, TypeError):
                            raw_extra = {}
                    init_attrs = raw_extra.get("init_attrs", s.get("init_attrs", {}))
                    self.per_class_init_attr_types[parent_id] = init_attrs

    def _get_extra(self, sym: dict) -> dict:
        """Return extra JSONB dict for a symbol."""
        raw = sym.get("extra") or {}
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (ValueError, TypeError):
                return {}
        return raw

    def _get_call_sites(self, sym: dict) -> list[dict]:
        """Return call_sites list for a symbol (from JSONB or direct)."""
        raw = sym.get("call_sites") or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                return []
        return raw

    def _enclosing_class_id(self, caller_id: str) -> str | None:
        """Find the class symbol_id that contains the given symbol."""
        sym = self.by_id.get(caller_id)
        if sym is None:
            return None
        parent_id = sym.get("parent_id")
        if parent_id is None:
            return None
        parent = self.by_id.get(parent_id)
        if parent and parent.get("kind") == "class":
            return parent_id
        # Walk up further
        return self._enclosing_class_id(parent_id)

    def _mro_lookup(self, class_id: str, method_name: str) -> str | None:
        """Look up method_name in class_id's MRO (BFS), return symbol_id or None."""
        visited: set[str] = set()
        queue = [class_id]
        while queue:
            cid = queue.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            cls = self.by_id.get(cid)
            if cls is None:
                continue
            file_path = cls["file_path"]
            qualname = cls["qualname"]
            method_qn = f"{qualname}.{method_name}"
            candidate = self.by_file_qualname.get((file_path, method_qn))
            if candidate:
                return candidate
            # Walk bases
            queue.extend(self.class_bases.get(cid, []))
        return None

    def _make_edge(
        self,
        caller_id: str,
        callee_id: str | None,
        call_site: dict,
        resolution: str,
        confidence: float,
        kind: str | None = None,
    ) -> dict:
        cs_kind = kind or call_site.get("kind", "call")
        return {
            "caller_id": caller_id,
            "callee_id": callee_id,
            "callee_expr": call_site.get("expr", ""),
            "line": call_site.get("line", 0),
            "kind": cs_kind,
            "resolution": resolution,
            "source": "ast",
            "confidence": confidence,
            "control_ctx": call_site.get("control_ctx", ""),
        }

    def resolve_call(self, caller_id: str, call_site: dict) -> list[dict]:
        """Resolve a single call site to a list of edge-row dicts."""
        sym = self.by_id.get(caller_id)
        if sym is None:
            return [self._make_edge(caller_id, None, call_site, "external", 0.0)]

        expr: str = call_site.get("expr", "")
        file_path: str = sym.get("file_path", "")
        import_map = self.import_maps.get(file_path, {})
        extra = self._get_extra(sym)
        param_types: dict[str, str] = extra.get("param_types", sym.get("param_types") or {})
        local_types: dict[str, str] = extra.get("local_types", sym.get("local_types") or {})

        head = _head(expr)
        method = _method_name(expr)
        is_attribute_call = "." in expr

        # ------------------------------------------------------------------
        # Layer 1: import resolution
        # ------------------------------------------------------------------
        if head in import_map:
            resolved_file, exported_name = import_map[head]
            suffix = expr[len(head) :]
            # Module imports map their prefix to a file; from imports map to a symbol.
            imported_symbol = self.by_file_qualname.get((resolved_file, exported_name))
            target = exported_name + suffix if imported_symbol else suffix.lstrip(".")
            callee_id = self.by_file_qualname.get((resolved_file, target))
            if callee_id:
                return self._target_edges(caller_id, callee_id, call_site, "import", 0.95)

        # Layer 2: search lexical scopes from innermost to module scope.
        if not is_attribute_call:
            scopes = sym["qualname"].split(".")
            for depth in range(len(scopes), -1, -1):
                qualname = ".".join([*scopes[:depth], head])
                callee_id = self.by_file_qualname.get((file_path, qualname))
                if callee_id:
                    return self._target_edges(caller_id, callee_id, call_site, "local", 0.90)

        # ------------------------------------------------------------------
        # Layer 3: self/cls method resolution
        # ------------------------------------------------------------------
        if expr.startswith(("self.", "cls.")):
            prefix_len = len("self.") if expr.startswith("self.") else len("cls.")
            rest = expr[prefix_len:]
            meth_name = rest.split(".")[0]  # first segment after self.
            # But if there are further dots (self.conn.execute) → can't resolve
            if "." not in rest:
                class_id = self._enclosing_class_id(caller_id)
                if class_id:
                    callee_id = self._mro_lookup(class_id, meth_name)
                    if callee_id:
                        return [self._make_edge(caller_id, callee_id, call_site, "self", 0.90)]
            # self.conn.execute and similar: method on unknown typed attr → external
            if "." not in rest:
                return [self._make_edge(caller_id, None, call_site, "self", 0.0)]

        # ------------------------------------------------------------------
        # Layer 4: constructor (head is a class name, no dots)
        # ------------------------------------------------------------------
        if not is_attribute_call:
            candidates = self.by_name.get(head, [])
            class_candidates = [c for c in candidates if self.by_id[c].get("kind") == "class"]
            if len(class_candidates) == 1:
                class_id = class_candidates[0]
                cls_sym = self.by_id[class_id]
                edges: list[dict] = [
                    self._make_edge(caller_id, class_id, call_site, "ctor", 0.90, "constructor")
                ]
                init_key = (cls_sym["file_path"], f"{cls_sym['qualname']}.__init__")
                init_id = self.by_file_qualname.get(init_key)
                if init_id:
                    edges.append(
                        self._make_edge(caller_id, init_id, call_site, "ctor", 0.90, "constructor")
                    )
                return edges

        # ------------------------------------------------------------------
        # Layer 5: typed resolution (param type or local var type)
        # ------------------------------------------------------------------
        if is_attribute_call:
            # head might be a typed variable
            type_name: str | None = None
            if head in param_types:
                type_name = param_types[head]
            elif head in local_types:
                type_name = local_types[head]
            elif head in {"self", "cls"} and expr.count(".") == 2:
                class_id = self._enclosing_class_id(caller_id)
                attr = expr.split(".")[1]
                type_name = self.per_class_init_attr_types.get(class_id, {}).get(attr)
                head = f"{head}.{attr}"

            if type_name:
                # Resolve type_name to a class symbol
                class_id = self._resolve_type_to_class(type_name, file_path)
                if class_id:
                    meth_name = expr[len(head) + 1 :].split(".")[0]
                    callee_id = self._mro_lookup(class_id, meth_name)
                    if callee_id:
                        return [self._make_edge(caller_id, callee_id, call_site, "typed", 0.80)]

        # ------------------------------------------------------------------
        # Layer 6: name-unique
        # ------------------------------------------------------------------
        last_seg = method
        if last_seg not in COMMON_METHOD_BLOCKLIST:
            candidates = self.by_name.get(last_seg, [])
            if len(candidates) == 1:
                return [self._make_edge(caller_id, candidates[0], call_site, "name-unique", 0.75)]

        # ------------------------------------------------------------------
        # Layer 7: ambiguous
        # ------------------------------------------------------------------
        if last_seg not in COMMON_METHOD_BLOCKLIST:
            candidates = self.by_name.get(last_seg, [])
            if len(candidates) > 1:
                return [
                    self._make_edge(caller_id, cid, call_site, "ambiguous", 0.40)
                    for cid in candidates
                ]

        # ------------------------------------------------------------------
        # Layer 8: external
        # ------------------------------------------------------------------
        return [self._make_edge(caller_id, None, call_site, "external", 0.0)]

    def _target_edges(self, caller_id, callee_id, call_site, resolution, confidence):
        symbol = self.by_id[callee_id]
        is_constructor = symbol["kind"] == "class" and call_site.get("kind") != "decorator"
        edges = [
            self._make_edge(
                caller_id,
                callee_id,
                call_site,
                resolution,
                confidence,
                "constructor" if is_constructor else None,
            )
        ]
        if is_constructor:
            init_id = self._mro_lookup(callee_id, "__init__")
            if init_id:
                edges.append(
                    self._make_edge(
                        caller_id, init_id, call_site, resolution, confidence, "constructor"
                    )
                )
        return edges

    def _resolve_type_to_class(self, type_name: str, file_path: str) -> str | None:
        """Resolve a type name string to a class symbol_id."""
        # Try same file first
        candidate = self.by_file_qualname.get((file_path, type_name))
        if candidate and self.by_id[candidate].get("kind") == "class":
            return candidate
        # Try import map
        import_map = self.import_maps.get(file_path, {})
        head = type_name.split(".")[0]
        if head in import_map:
            resolved_file, exported_name = import_map[head]
            key = (resolved_file, exported_name)
            candidate = self.by_file_qualname.get(key)
            if candidate and self.by_id[candidate].get("kind") == "class":
                return candidate
        # Try by_name (unique)
        candidates = self.by_name.get(type_name, [])
        class_candidates = [c for c in candidates if self.by_id[c].get("kind") == "class"]
        if len(class_candidates) == 1:
            return class_candidates[0]
        return None

    def resolve_all(self) -> list[dict]:
        """Resolve all call sites for all symbols, return list of edge dicts."""
        all_edges: list[dict] = []
        for sym in self.symbols:
            caller_id = sym["id"]
            call_sites = self._get_call_sites(sym)
            for cs in call_sites:
                if isinstance(cs, dict):
                    cs_dict = cs
                else:
                    # Handle CallSite dataclass (when used directly from parser)
                    cs_dict = {
                        "expr": cs.expr,
                        "line": cs.line,
                        "kind": cs.kind,
                        "control_ctx": cs.control_ctx,
                    }
                edges = self.resolve_call(caller_id, cs_dict)
                all_edges.extend(edges)
        return all_edges


def detect_entrypoints(symbols: list[dict]) -> dict[str, str]:
    """Detect entrypoint symbols and return {symbol_id: entry_kind}."""
    result: dict[str, str] = {}
    route_patterns = (
        ".route",
        ".get",
        ".post",
        ".put",
        ".delete",
        ".patch",
        ".command",
        ".task",
        ".callback",
    )

    for sym in symbols:
        sid = sym["id"]
        name = sym.get("name", "")
        kind = sym.get("kind", "")
        file_path = sym.get("file_path", "")
        raw_code = sym.get("raw_code", "") or ""

        # main() function
        if name == "main" and kind == "function":
            result[sid] = "main"
            continue

        # AST matching accepts either quote style and arbitrary whitespace.
        if kind == "module":
            try:
                tree = ast.parse(raw_code)
            except SyntaxError:
                tree = ast.Module(body=[], type_ignores=[])
            if any(
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "__name__"
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "__main__"
                for node in ast.walk(tree)
            ):
                result[sid] = "main_guard"
                continue

        # __main__.py
        if pathlib.PurePosixPath(file_path).name == "__main__.py":
            result[sid] = "main_module"
            continue

        # Decorator-based routes/CLI
        raw_decorators = sym.get("decorators") or []
        if isinstance(raw_decorators, str):
            try:
                raw_decorators = json.loads(raw_decorators)
            except (ValueError, TypeError):
                raw_decorators = []
        for dec in raw_decorators:
            dec_str = str(dec)
            for pat in route_patterns:
                if dec_str.split("(", 1)[0].endswith(pat):
                    if ".command" in dec_str:
                        result[sid] = "cli"
                    elif ".task" in dec_str or ".callback" in dec_str:
                        result[sid] = "task"
                    else:
                        result[sid] = "route"
                    break
            if sid in result:
                break

        if sid in result:
            continue

        exports = sym.get("exports") or []
        if isinstance(exports, str):
            exports = json.loads(exports)
        if sym.get("qualname") in exports:
            result[sid] = "exported"

    return result


def run_resolution(conn: psycopg.Connection, settings: Settings) -> None:
    """Stage 2A: resolve call-graph edges from parsed AST data."""
    from cfl.core.db import (
        get_all_symbols,
        get_meta,
        list_files,
        replace_edges,
        set_entrypoints,
        set_meta,
    )
    from cfl.core.hashing import join_hash

    files = list_files(conn)
    # Compute fingerprint
    sha_parts = [f"{f['path']}:{f['sha256']}" for f in sorted(files, key=lambda f: f["path"])]
    fingerprint = join_hash(*sha_parts, "resolver_v1", str(settings.edge_conf_threshold))

    stored = get_meta(conn, "resolve_fingerprint")
    if stored == fingerprint:
        logger.info("resolve_fingerprint unchanged, skipping resolution")
        return

    symbols = get_all_symbols(conn)
    resolver = Resolver(symbols, files)
    all_edges = resolver.resolve_all()

    exports = {f["path"]: f.get("exports") or [] for f in files}
    entrypoints = detect_entrypoints(
        [{**s, "exports": exports.get(s["file_path"], [])} for s in symbols]
    )
    with conn.transaction():
        replace_edges(conn, "ast", all_edges)
        set_entrypoints(conn, [{"id": sid, "entry_kind": ek} for sid, ek in entrypoints.items()])
        set_meta(conn, "resolve_fingerprint", fingerprint)
    logger.info("Resolution complete: %d edges, %d entrypoints", len(all_edges), len(entrypoints))
