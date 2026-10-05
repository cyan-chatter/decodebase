from __future__ import annotations

import ast
import io
import pathlib
import textwrap
import tokenize
from typing import Any

from cfl.parser.base import (
    CallSite,
    ImportEntry,
    ParsedFile,
    ParsedSymbol,
    register_adapter,
)


def decode_source(source_bytes: bytes) -> str:
    """Decode Python encoding cookies, falling back to UTF-8 replacement on failure."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)
        return source_bytes.decode(encoding)
    except (SyntaxError, UnicodeError, LookupError):
        return source_bytes.decode("utf-8", errors="replace")


def _unwrap_optional(annotation_str: str) -> str:
    """Unwrap Optional[X] or X | None to just X."""
    s = annotation_str.strip()
    # Optional[X]
    optional = s.split("[", 1)[0]
    if optional in {"Optional", "typing.Optional"} and s.endswith("]"):
        return s[len(optional) + 1 : -1]
    # X | None
    if " | None" in s:
        return s.replace(" | None", "").strip()
    if "None | " in s:
        return s.replace("None | ", "").strip()
    return s


def _annotation_str(node: ast.expr | None) -> str | None:
    """Convert an annotation node to string, or None."""
    if node is None:
        return None
    try:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ast.unparse(node)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


class _CallVisitor(ast.NodeVisitor):
    """Collect call sites within a symbol body.

    Does NOT descend into nested FunctionDef/AsyncFunctionDef/ClassDef.
    DOES descend into Lambda and comprehensions.
    Tracks control context stack.
    """

    def __init__(self) -> None:
        self.call_sites: list[CallSite] = []
        self._ctrl_stack: list[str] = []

    def _ctx(self) -> str:
        return ">".join(self._ctrl_stack)

    def visit_Call(self, node: ast.Call) -> None:
        try:
            expr = ast.unparse(node.func)[:200]
        except (SyntaxError, ValueError, TypeError, RecursionError):
            expr = "<unknown>"
        self.call_sites.append(
            CallSite(expr=expr, line=node.lineno, kind="call", control_ctx=self._ctx())
        )
        self.generic_visit(node)

    # Stop at nested defs/classes — they own their calls
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # do not descend

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass  # do not descend

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass  # do not descend

    # Control context tracking
    def visit_If(self, node: ast.If) -> None:
        self._ctrl_stack.append("if")
        self.generic_visit(node)
        self._ctrl_stack.pop()

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # The test runs first; only one branch executes at runtime.
        self.visit(node.test)
        condition = ast.unparse(node.test)[:200]
        for expression, label in (
            (node.body, "if " + condition),
            (node.orelse, "if not (" + condition + ")"),
        ):
            self._ctrl_stack.append(label)
            self.visit(expression)
            self._ctrl_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        self._ctrl_stack.append("for")
        self.generic_visit(node)
        self._ctrl_stack.pop()

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self._ctrl_stack.append("while")
        self.generic_visit(node)
        self._ctrl_stack.pop()

    def visit_Try(self, node: ast.Try) -> None:
        self._ctrl_stack.append("try")
        self.generic_visit(node)
        self._ctrl_stack.pop()

    def visit_TryStar(self, node: Any) -> None:
        self._ctrl_stack.append("try")
        self.generic_visit(node)
        self._ctrl_stack.pop()

    def visit_With(self, node: ast.With) -> None:
        self._ctrl_stack.append("with")
        self.generic_visit(node)
        self._ctrl_stack.pop()

    visit_AsyncWith = visit_With

    # Comprehensions — push 'comp' context
    def _visit_comp(self, node: ast.AST) -> None:
        self._ctrl_stack.append("comp")
        self.generic_visit(node)
        self._ctrl_stack.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comp(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comp(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comp(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comp(node)


def _extract_call_sites(body_nodes: list[ast.stmt]) -> list[CallSite]:
    """Extract call sites from a list of statement nodes."""
    visitor = _CallVisitor()
    for stmt in body_nodes:
        visitor.visit(stmt)
    return visitor.call_sites


def _extract_decorator_call_sites(decorator_list: list[ast.expr]) -> list[CallSite]:
    """Extract decorator call sites (kind='decorator')."""
    sites: list[CallSite] = []
    for dec in decorator_list:
        try:
            expr = ast.unparse(dec.func if isinstance(dec, ast.Call) else dec)[:200]
        except (SyntaxError, ValueError, TypeError, RecursionError):
            expr = "<unknown>"
        line = dec.lineno
        sites.append(CallSite(expr=expr, line=line, kind="decorator", control_ctx=""))
    return sites


def _scope_nodes(nodes):
    """Visit control blocks in one lexical scope, excluding nested definitions."""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        yield from _scope_nodes(ast.iter_child_nodes(node))


def _extract_init_attrs(init_node: ast.FunctionDef) -> dict[str, str]:
    """Extract self.x = ... assignments from __init__, return {attr: type}."""
    attrs: dict[str, str] = {}
    param_types = _extract_param_types(init_node.args)
    for stmt in _scope_nodes(init_node.body):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attr = target.attr
                    # Try to guess type from RHS
                    type_str = ""
                    if isinstance(stmt.value, ast.Call):
                        try:
                            type_str = ast.unparse(stmt.value.func)
                        except (SyntaxError, ValueError, TypeError, RecursionError):
                            type_str = ""
                    elif isinstance(stmt.value, ast.Name):
                        type_str = param_types.get(stmt.value.id, "")
                    attrs[attr] = type_str
        elif isinstance(stmt, ast.AnnAssign) and (
            isinstance(stmt.target, ast.Attribute)
            and isinstance(stmt.target.value, ast.Name)
            and stmt.target.value.id == "self"
        ):
            attr = stmt.target.attr
            ann = _annotation_str(stmt.annotation) or ""
            attrs[attr] = _unwrap_optional(ann)
    return attrs


def _extract_param_types(args: ast.arguments) -> dict[str, str]:
    """Extract annotated parameter types (excluding self/cls), unwrap Optional."""
    types: dict[str, str] = {}
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        all_args.append(args.vararg)
    if args.kwarg:
        all_args.append(args.kwarg)
    for arg in all_args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is not None:
            raw = _annotation_str(arg.annotation) or ""
            types[arg.arg] = _unwrap_optional(raw)
    return types


def _extract_local_types(body: list[ast.stmt]) -> dict[str, str]:
    """Extract x = Foo(...) assignments (local var → type)."""
    types: dict[str, str] = {}
    for stmt in _scope_nodes(body):
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Call):
                type_str = ""
                try:
                    type_str = ast.unparse(stmt.value.func)
                except (SyntaxError, ValueError, TypeError, RecursionError):
                    type_str = ""
                if type_str:
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            types[target.id] = type_str
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.target, ast.Name)
        ):
            type_str = ""
            try:
                type_str = ast.unparse(stmt.value.func)
            except (SyntaxError, ValueError, TypeError, RecursionError):
                type_str = ""
            if type_str:
                types[stmt.target.id] = type_str
    return types


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a readable signature string from a function node."""
    if node.returns is not None:
        ret = _annotation_str(node.returns)
    else:
        ret = None

    try:
        args_str = ast.unparse(node.args)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        args_str = "..."

    sig = args_str
    if ret:
        sig = f"{sig} -> {ret}"
    if isinstance(node, ast.AsyncFunctionDef):
        sig = "async " + sig
    return sig


def _has_call(node: ast.AST) -> bool:
    """Return True if any ast.Call exists anywhere in the node's subtree."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            return True
    return False


class PythonAdapter:
    """Language adapter for Python source files."""

    language = "python"
    extensions = (".py",)

    def parse(self, path: pathlib.Path, source: str) -> ParsedFile:
        """Parse Python source and return a ParsedFile."""
        try:
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            return ParsedFile(
                path=str(path),
                language="python",
                imports=[],
                exports=[],
                symbols=[],
                parse_error=str(exc),
            )

        lines = source.splitlines()

        # Collect imports
        imports: list[ImportEntry] = []
        for node in _scope_nodes(tree.body):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        ImportEntry(
                            module=alias.name,
                            name=None,
                            alias=alias.asname,
                            level=0,
                            line=node.lineno,
                            is_from=False,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0
                for alias in node.names:
                    imports.append(
                        ImportEntry(
                            module=module,
                            name=alias.name,
                            alias=alias.asname,
                            level=level,
                            line=node.lineno,
                            is_from=True,
                        )
                    )

        # Collect exports from __all__
        exports: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        try:
                            val = ast.literal_eval(node.value)
                            if isinstance(val, (list, tuple)):
                                exports = [str(x) for x in val]
                        except (SyntaxError, ValueError, TypeError, RecursionError):
                            pass

        # For __init__.py: also collect imported public names
        if path.name == "__init__.py":
            for imp in imports:
                if imp.is_from:
                    exported_name = imp.alias if imp.alias else imp.name
                else:
                    exported_name = imp.alias or imp.module.split(".")[0]
                if (
                    exported_name
                    and not exported_name.startswith("_")
                    and exported_name not in exports
                ):
                    exports.append(exported_name)

        # Walk AST to collect symbols
        symbols: list[ParsedSymbol] = []
        self._walk_scope(tree.body, lines, path, symbols, scope_stack=[])

        # Module pseudo-symbol: top-level statements that are not def/class/import
        module_stmts: list[ast.stmt] = []
        skip_types = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Import,
            ast.ImportFrom,
        )
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.stmt) and not isinstance(node, skip_types):
                module_stmts.append(node)

        if module_stmts and any(_has_call(s) for s in module_stmts):
            mod_start = min(s.lineno for s in module_stmts)
            mod_end = max(s.end_lineno for s in module_stmts)  # type: ignore[attr-defined]
            # Collect raw code from the statement ranges
            raw_parts: list[str] = []
            for s in module_stmts:
                raw_parts.append("\n".join(lines[s.lineno - 1 : s.end_lineno]))  # type: ignore[attr-defined]
            raw_code = "\n".join(raw_parts)

            call_sites = _extract_call_sites(module_stmts)
            module_sym = ParsedSymbol(
                kind="module",
                qualname=f"{path.stem}::<module>",
                name="<module>",
                parent_qualname=None,
                signature="",
                decorators=[],
                docstring=None,
                start_line=mod_start,
                end_line=mod_end,
                raw_code=raw_code,
                is_async=False,
                bases=[],
                init_attrs={},
                param_types={},
                local_types={},
                call_sites=call_sites,
            )
            symbols.append(module_sym)

        return ParsedFile(
            path=str(path),
            language="python",
            imports=imports,
            exports=exports,
            symbols=symbols,
            parse_error=None,
        )

    def _walk_scope(
        self,
        stmts: list[ast.stmt],
        lines: list[str],
        path: pathlib.Path,
        symbols: list[ParsedSymbol],
        scope_stack: list[ast.AST],
    ) -> None:
        """Recursively walk statements, collecting symbols into `symbols`."""
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._handle_function(node, lines, path, symbols, scope_stack)
            elif isinstance(node, ast.ClassDef):
                self._handle_class(node, lines, path, symbols, scope_stack)
            else:
                # Control statements do not introduce a new lexical scope.
                children = []
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.stmt):
                        children.append(child)
                    elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                        children.extend(child.body)
                self._walk_scope(children, lines, path, symbols, scope_stack)

    def _qualname_from_stack(self, stack: list[ast.AST], current_name: str) -> str:
        """Build a dotted qualname from the scope stack plus current name."""
        parts: list[str] = []
        for node in stack:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parts.append(node.name)
        parts.append(current_name)
        return ".".join(parts)

    def _parent_qualname_from_stack(self, stack: list[ast.AST]) -> str | None:
        """Return the qualname of the direct parent scope, or None."""
        for node in reversed(stack):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return self._qualname_from_stack(stack[: stack.index(node)], node.name)
        return None

    def _kind_from_stack(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, stack: list[ast.AST]
    ) -> str:
        """Determine function kind: 'function', 'method', or 'nested'."""
        if not stack:
            return "function"
        # Direct parent
        parent = stack[-1]
        if isinstance(parent, ast.ClassDef):
            return "method"
        return "nested"

    def _handle_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        path: pathlib.Path,
        symbols: list[ParsedSymbol],
        scope_stack: list[ast.AST],
    ) -> None:
        qualname = self._qualname_from_stack(scope_stack, node.name)
        parent_qualname = self._parent_qualname_from_stack(scope_stack)
        kind = self._kind_from_stack(node, scope_stack)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        dec_start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        start_line = min(node.lineno, dec_start)
        end_line = node.end_lineno  # type: ignore[attr-defined]

        raw_code = "\n".join(lines[start_line - 1 : end_line])
        signature = _build_signature(node)
        docstring = ast.get_docstring(node, clean=True)
        decorators = [ast.unparse(d) for d in node.decorator_list]

        # param_types from annotations
        param_types = _extract_param_types(node.args)

        # init_attrs and local_types
        init_attrs: dict[str, str] = {}
        if node.name == "__init__":
            init_attrs = _extract_init_attrs(node)

        local_types = _extract_local_types(node.body)

        # Call sites: only from this function's body (not nested defs/classes)
        call_sites = _extract_call_sites(node.body)

        # Decorator call sites
        dec_call_sites = _extract_decorator_call_sites(node.decorator_list)
        all_call_sites = dec_call_sites + call_sites

        sym = ParsedSymbol(
            kind=kind,
            qualname=qualname,
            name=node.name,
            parent_qualname=parent_qualname,
            signature=signature,
            decorators=decorators,
            docstring=docstring,
            start_line=start_line,
            end_line=end_line,
            raw_code=raw_code,
            is_async=is_async,
            bases=[],
            init_attrs=init_attrs,
            param_types=param_types,
            local_types=local_types,
            call_sites=all_call_sites,
        )
        symbols.append(sym)

        # Recurse into nested defs/classes
        new_stack = scope_stack + [node]
        self._walk_scope(node.body, lines, path, symbols, new_stack)

    def _handle_class(
        self,
        node: ast.ClassDef,
        lines: list[str],
        path: pathlib.Path,
        symbols: list[ParsedSymbol],
        scope_stack: list[ast.AST],
    ) -> None:
        qualname = self._qualname_from_stack(scope_stack, node.name)
        parent_qualname = self._parent_qualname_from_stack(scope_stack)

        dec_start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        start_line = min(node.lineno, dec_start)
        end_line = node.end_lineno  # type: ignore[attr-defined]

        raw_code = "\n".join(lines[start_line - 1 : end_line])
        docstring = ast.get_docstring(node, clean=True)
        decorators = [ast.unparse(d) for d in node.decorator_list]
        bases = [ast.unparse(b) for b in node.bases]

        # Class-body calls (not in methods)
        class_body_stmts = [
            s
            for s in node.body
            if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        call_sites = _extract_call_sites(class_body_stmts)
        dec_call_sites = _extract_decorator_call_sites(node.decorator_list)
        all_call_sites = dec_call_sites + call_sites

        # init_attrs: gather from __init__ method in this class
        init_attrs: dict[str, str] = {}
        for stmt in node.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name == "__init__"
            ):
                init_attrs = _extract_init_attrs(stmt)
                break

        sym = ParsedSymbol(
            kind="class",
            qualname=qualname,
            name=node.name,
            parent_qualname=parent_qualname,
            signature="",
            decorators=decorators,
            docstring=docstring,
            start_line=start_line,
            end_line=end_line,
            raw_code=raw_code,
            is_async=False,
            bases=bases,
            init_attrs=init_attrs,
            param_types={},
            local_types={},
            call_sites=all_call_sites,
        )
        symbols.append(sym)

        # Recurse
        new_stack = scope_stack + [node]
        self._walk_scope(node.body, lines, path, symbols, new_stack)

    def statement_spans(self, raw_code: str) -> list[tuple[int, int]]:
        """Return (start_line, end_line) of each top-level statement in raw_code."""
        try:
            tree = ast.parse(textwrap.dedent(raw_code))
        except (SyntaxError, ValueError, TypeError, RecursionError):
            return []
        body = tree.body
        if body and isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = body[0].body
        return [(stmt.lineno, stmt.end_lineno) for stmt in body]

    def skeleton(self, parsed: ParsedFile) -> str:
        """Return one line per symbol: kind + qualname + signature."""
        lines: list[str] = []
        for sym in parsed.symbols:
            if sym.kind == "class":
                bases_str = ", ".join(sym.bases)
                lines.append(f"class {sym.qualname}({bases_str}):")
            elif sym.kind == "module":
                lines.append(f"# module {sym.qualname}")
            else:
                prefix = "async def" if sym.is_async else "def"
                signature = sym.signature.removeprefix("async ") if sym.is_async else sym.signature
                args, sep, returns = signature.partition(" -> ")
                suffix = f" -> {returns}" if sep else ""
                lines.append(f"{prefix} {sym.qualname}({args}){suffix}:")
        return "\n".join(lines)


# Register the adapter at module load time
register_adapter(PythonAdapter())
