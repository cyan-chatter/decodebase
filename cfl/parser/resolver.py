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


def _json_value(raw, default):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return default
    return raw or default


def _build_module_index(files: list[dict]) -> dict[str, str]:
    index: dict[str, str] = {}
    for file in files:
        path = file["path"]
        p = pathlib.PurePosixPath(path)
        keys = [str(p.with_suffix(""))]
        if p.name == "__init__.py" and str(p.parent) != ".":
            keys.append(str(p.parent))
        for key in list(keys):
            if key.startswith("src/"):
                keys.append(key.removeprefix("src/"))
        for key in keys:
            index[key] = path
            index[key.replace("/", ".")] = path
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
            if is_from and name:
                submodule = f"{module}.{name}" if module else name
                submodule_path = _resolve_module_path(submodule, level, file_path, module_index)
                if submodule_path:
                    alias_map[alias or name] = (submodule_path, "")
                elif resolved_path:
                    alias_map[alias or name] = (resolved_path, name)
            elif not is_from and resolved_path:
                key = alias if alias else module.split(".")[0]
                alias_map[key] = (resolved_path, "")
        maps[file_path] = alias_map
    return maps


class Resolver:
    """Resolve calls in memory, retaining every candidate at the first matching layer."""

    def __init__(self, symbols: list[dict], files: list[dict]) -> None:
        self.symbols = symbols
        self.files = files
        self.by_id = {symbol["id"]: symbol for symbol in symbols}
        self.module_index = _build_module_index(files)
        self.import_maps = _build_import_maps(files, self.module_index)
        self.by_file_candidates: dict[tuple[str, str], list[str]] = {}
        self.by_file_qualname: dict[tuple[str, str], str] = {}
        self.by_name: dict[str, list[str]] = {}
        for symbol in symbols:
            key = (symbol["file_path"], symbol["qualname"])
            self.by_file_candidates.setdefault(key, []).append(symbol["id"])
            # Retain the existing unique-ID index API; resolution uses the full candidate index.
            self.by_file_qualname.setdefault(key, symbol["id"])
            name = symbol["name"]
            if name not in COMMON_METHOD_BLOCKLIST and not name.startswith("__"):
                self.by_name.setdefault(name, []).append(symbol["id"])
        self.module_alias_suffixes: dict[tuple[str, str], str] = {}
        for file in files:
            for entry in _json_value(file.get("imports"), []):
                if not entry.get("is_from") and not entry.get("alias"):
                    parts = entry["module"].split(".")
                    self.module_alias_suffixes[file["path"], parts[0]] = ".".join(parts[1:])
        self.class_bases: dict[str, list[str]] = {}
        for symbol in symbols:
            if symbol["kind"] == "class":
                bases = self._get_extra(symbol).get("bases", symbol.get("bases", []))
                self.class_bases[symbol["id"]] = list(
                    dict.fromkeys(
                        candidate
                        for base in bases
                        for candidate in self._class_candidates(base, symbol["file_path"])
                    )
                )
        self.per_class_init_attr_types: dict[str, dict[str, list[str]]] = {}
        for symbol in symbols:
            if symbol["name"] == "__init__" and symbol.get("parent_id"):
                attributes = self.per_class_init_attr_types.setdefault(symbol["parent_id"], {})
                for attr, type_name in (
                    self._get_extra(symbol).get("init_attrs", symbol.get("init_attrs", {})).items()
                ):
                    if type_name:
                        types = attributes.setdefault(attr, [])
                        if type_name not in types:
                            types.append(type_name)

    def _get_extra(self, symbol: dict) -> dict:
        return _json_value(symbol.get("extra"), {})

    def _get_call_sites(self, symbol: dict) -> list[dict]:
        return _json_value(symbol.get("call_sites"), [])

    def _enclosing_class_id(self, caller_id: str) -> str | None:
        visited = set()
        while caller_id in self.by_id and caller_id not in visited:
            visited.add(caller_id)
            symbol = self.by_id[caller_id]
            if symbol["kind"] == "class":
                return caller_id
            caller_id = symbol.get("parent_id")
        return None

    def _import_target(self, file_path: str, expr: str) -> tuple[str, str] | None:
        head, _, suffix = expr.partition(".")
        imported = self.import_maps.get(file_path, {}).get(head)
        if imported is None:
            return None
        target_file, exported = imported
        if exported:
            return target_file, exported + ("." + suffix if suffix else "")
        prefix = self.module_alias_suffixes.get((file_path, head), "")
        if prefix:
            if not suffix.startswith(prefix + "."):
                return None
            suffix = suffix[len(prefix) + 1 :]
        return target_file, suffix

    def _file_candidates(
        self, file_path: str, qualname: str, visited: set[tuple[str, str]] | None = None
    ) -> list[str]:
        key = (file_path, qualname)
        if visited is None:
            visited = set()
        if key in visited:
            return []
        visited.add(key)
        candidates = self.by_file_candidates.get(key, [])
        if candidates:
            return candidates
        target = self._import_target(file_path, qualname)
        return self._file_candidates(*target, visited) if target else []

    def _class_candidates(self, type_name: str, file_path: str) -> list[str]:
        candidates = self._file_candidates(file_path, type_name)
        if not candidates:
            candidates = self.by_name.get(type_name, [])
        return [candidate for candidate in candidates if self.by_id[candidate]["kind"] == "class"]

    def _method_candidates(
        self, class_id: str, method: str, visited: set[str] | None = None
    ) -> list[str]:
        if visited is None:
            visited = set()
        if class_id in visited:
            return []
        visited.add(class_id)
        cls = self.by_id[class_id]
        candidates = self.by_file_candidates.get(
            (cls["file_path"], f"{cls['qualname']}.{method}"), []
        )
        candidates = [
            sid for sid in candidates if self.by_id[sid].get("parent_id") in {None, class_id}
        ]
        if candidates:
            return candidates
        for base in self.class_bases.get(class_id, []):
            candidates = self._method_candidates(base, method, visited)
            if candidates:
                return candidates
        return []

    def _attribute_types(self, class_id: str, attr: str) -> list[str]:
        pending, visited = [class_id], set()
        while pending:
            candidate = pending.pop(0)
            if candidate in visited:
                continue
            visited.add(candidate)
            types = self.per_class_init_attr_types.get(candidate, {}).get(attr, [])
            if types:
                return types
            pending.extend(self.class_bases.get(candidate, []))
        return []

    def _make_edge(
        self,
        caller_id: str,
        callee_id: str | None,
        call_site: dict,
        resolution: str,
        confidence: float,
        kind: str | None = None,
    ) -> dict:
        return {
            "caller_id": caller_id,
            "callee_id": callee_id,
            "callee_expr": call_site.get("expr", ""),
            "line": call_site.get("line", 0),
            "kind": kind or call_site.get("kind", "call"),
            "resolution": resolution,
            "source": "ast",
            "confidence": confidence,
            "control_ctx": call_site.get("control_ctx", ""),
        }

    def _target_edges(
        self,
        caller_id: str,
        targets: list[str],
        call_site: dict,
        resolution: str,
        confidence: float,
    ) -> list[dict]:
        targets = sorted(set(targets))
        if len(targets) > 1:
            resolution, confidence = "ambiguous", 0.4
        edges = []
        for target in targets:
            constructor = (
                self.by_id[target]["kind"] == "class" and call_site.get("kind") != "decorator"
            )
            edges.append(
                self._make_edge(
                    caller_id,
                    target,
                    call_site,
                    resolution,
                    confidence,
                    "constructor" if constructor else None,
                )
            )
            if constructor:
                initializers = self._method_candidates(target, "__init__")
                init_resolution = "ambiguous" if len(initializers) > 1 else resolution
                init_confidence = 0.4 if len(initializers) > 1 else confidence
                edges.extend(
                    self._make_edge(
                        caller_id, init, call_site, init_resolution, init_confidence, "constructor"
                    )
                    for init in initializers
                )
        # Multiple class candidates may share the same inherited initializer.
        return list({(e["callee_id"], e["kind"]): e for e in edges}.values())

    def resolve_call(self, caller_id: str, call_site: dict) -> list[dict]:
        symbol = self.by_id.get(caller_id)
        external = [self._make_edge(caller_id, None, call_site, "external", 0.0)]
        if symbol is None:
            return external
        expr = call_site.get("expr", "")
        file_path = symbol["file_path"]
        head = _head(expr)
        extra = self._get_extra(symbol)

        imported = self._import_target(file_path, expr)
        if imported:
            candidates = self._file_candidates(*imported)
            if candidates:
                return self._target_edges(caller_id, candidates, call_site, "import", 0.95)
        if "." not in expr:
            scopes = symbol["qualname"].split(".")
            for depth in range(len(scopes), -1, -1):
                qualname = ".".join([*scopes[:depth], head])
                candidates = self.by_file_candidates.get((file_path, qualname), [])
                if candidates:
                    return self._target_edges(caller_id, candidates, call_site, "local", 0.9)
        class_id = self._enclosing_class_id(caller_id)
        if expr.startswith("super().") and expr.count(".") == 1:
            candidates = [
                method
                for base in self.class_bases.get(class_id, [])
                for method in self._method_candidates(base, _method_name(expr))
            ]
            if candidates:
                return self._target_edges(caller_id, candidates, call_site, "super", 0.9)
            return external
        if head in {"self", "cls"} and expr.count(".") == 1:
            candidates = self._method_candidates(class_id, _method_name(expr)) if class_id else []
            if candidates:
                return self._target_edges(caller_id, candidates, call_site, "self", 0.9)
            return external
        if "." not in expr:
            candidates = [
                sid for sid in self.by_name.get(head, []) if self.by_id[sid]["kind"] == "class"
            ]
            if candidates:
                return self._target_edges(caller_id, candidates, call_site, "ctor", 0.9)
        types = []
        if expr.count(".") == 1:
            type_name = extra.get("local_types", symbol.get("local_types", {})).get(head)
            type_name = type_name or extra.get("param_types", symbol.get("param_types", {})).get(
                head
            )
            if type_name:
                types = [type_name]
        elif head in {"self", "cls"} and expr.count(".") == 2 and class_id:
            types = self._attribute_types(class_id, expr.split(".")[1])
        if types:
            candidates = [
                method
                for type_name in types
                for cls in self._class_candidates(type_name, file_path)
                for method in self._method_candidates(cls, _method_name(expr))
            ]
            if candidates:
                return self._target_edges(caller_id, candidates, call_site, "typed", 0.8)
        candidates = self.by_name.get(_method_name(expr), [])
        if candidates:
            return self._target_edges(caller_id, candidates, call_site, "name-unique", 0.75)
        return external

    def resolve_all(self) -> list[dict]:
        from dataclasses import asdict, is_dataclass

        return [
            edge
            for symbol in self.symbols
            for site in self._get_call_sites(symbol)
            for edge in self.resolve_call(
                symbol["id"], asdict(site) if is_dataclass(site) else site
            )
        ]


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
    fingerprint = join_hash(
        *sha_parts,
        "resolver_v3",
        get_meta(conn, "parser_version") or "",
        str(settings.edge_conf_threshold),
    )

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
    # Public imports in package __init__ files designate their defining symbols too.
    for file in files:
        for exported in _json_value(file.get("exports"), []):
            for sid in resolver._file_candidates(file["path"], exported):
                entrypoints.setdefault(sid, "exported")
    with conn.transaction():
        replace_edges(conn, "ast", all_edges)
        set_entrypoints(conn, [{"id": sid, "entry_kind": ek} for sid, ek in entrypoints.items()])
        set_meta(conn, "resolve_fingerprint", fingerprint)
    logger.info("Resolution complete: %d edges, %d entrypoints", len(all_edges), len(entrypoints))
