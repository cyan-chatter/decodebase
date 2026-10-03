from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cfl.core import db
from cfl.core.errors import AmbiguousSymbol, SymbolNotFound
from cfl.parser.resolver import Resolver

if TYPE_CHECKING:
    from psycopg import Connection


@dataclass(frozen=True)
class Symbol:
    id: str
    qualname: str
    name: str
    file_path: str
    kind: str
    start_line: int
    end_line: int

    @classmethod
    def from_row(cls, row: dict) -> Symbol:
        return cls(**{key: row[key] for key in cls.__dataclass_fields__})


def resolve_symbol(conn: Connection, query: str) -> Symbol:
    """Resolve by exact ID, scoped name, qualname, name, suffix, then fuzzy lookup."""
    exact = db.get_symbol(conn, query)
    if exact:
        return Symbol.from_row(exact)
    rows = db.lookup_symbols(conn, query, limit=100)
    _, sep, scoped_name = query.partition("::")
    name = scoped_name if sep else query
    ranks = [
        lambda row: row["qualname"] == name,
        lambda row: row["name"] == name,
        lambda row: row["qualname"].endswith("." + name),
    ]
    for matches in ranks:
        candidates = [Symbol.from_row(row) for row in rows if matches(row)]
        if candidates:
            if len(candidates) == 1:
                return candidates[0]
            raise AmbiguousSymbol(candidates)
    # Substring matches do not silently beat the fuzzy fallback.
    if sep:
        candidates = [Symbol.from_row(row) for row in rows]
    else:
        candidates = [Symbol.from_row(row) for row in db.trigram_search(conn, query, limit=5)]
    if not candidates:
        raise SymbolNotFound(query)
    if len(candidates) > 1:
        raise AmbiguousSymbol(candidates)
    return candidates[0]


def _symbol_id(symbol: Symbol | str) -> str:
    return symbol.id if isinstance(symbol, Symbol) else symbol


def _evidence(row: dict) -> dict:
    return {
        **row,
        "symbol": row["qualname"] or row["callee_expr"],
        "location": f"{row['callsite_file']}:{row['line']}",
    }


def callers(
    conn: Connection, symbol: Symbol | str, depth: int = 1, min_conf: float = 0.6
) -> list[dict]:
    return [_evidence(row) for row in db.callers_rows(conn, _symbol_id(symbol), depth, min_conf)]


def callees(
    conn: Connection, symbol: Symbol | str, depth: int = 1, min_conf: float = 0.6
) -> list[dict]:
    return [_evidence(row) for row in db.callees_rows(conn, _symbol_id(symbol), depth, min_conf)]


def path(
    conn: Connection,
    source: Symbol | str,
    target: Symbol | str,
    max_depth: int = 8,
    min_conf: float = 0.6,
) -> list[dict] | None:
    """Return call-site hops for one shortest path; None means unreachable."""
    source_id, target_id = _symbol_id(source), _symbol_id(target)
    if source_id == target_id:
        return []
    rows = db.reach_rows(conn, source_id, "callees", max_depth, min_conf)
    candidates = [row for row in rows if row["id"] == target_id]
    if not candidates:
        return None
    current = min(candidates, key=lambda row: (row["depth"], tuple(row["path_ids"]), row["line"]))
    # Match the exact path prefix, so merging branches cannot mix unrelated hops.
    by_prefix = {tuple(row["path_ids"]): row for row in reversed(rows)}
    hops = []
    while current["parent_id"] is not None:
        hops.append(_evidence(current))
        prefix = tuple(current["path_ids"][:-1])
        if len(prefix) == 1:
            break
        current = by_prefix[prefix]
    return list(reversed(hops))


def impact(conn: Connection, symbol: Symbol | str, depth: int = 4, min_conf: float = 0.6) -> dict:
    """Group distinct reverse-reachable symbols by their shortest depth and file."""
    seed = _symbol_id(symbol)
    reached: dict[str, dict] = {}
    for row in db.reach_rows(conn, seed, "callers", depth, min_conf):
        if row["id"] == seed:
            continue
        if row["id"] not in reached or row["depth"] < reached[row["id"]]["depth"]:
            reached[row["id"]] = _evidence(row)
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in reached.values():
        groups[row["depth"], row["file_path"]].append(row)
    return {
        "symbol_id": seed,
        "total": len(reached),
        "groups": [
            {
                "depth": d,
                "file_path": f,
                "count": len(members),
                "symbols": sorted(members, key=lambda row: row["id"]),
            }
            for (d, f), members in sorted(groups.items())
        ],
    }


def hubs(conn: Connection, n: int = 20, min_conf: float = 0.6) -> list[dict]:
    return db.top_pagerank(conn, n, min_conf)


def _overrides_base(symbol: dict, resolver: Resolver) -> bool:
    pending = list(resolver.class_bases.get(symbol.get("parent_id"), []))
    visited = set()
    while pending:
        class_id = pending.pop()
        if class_id in visited:
            continue
        visited.add(class_id)
        base = resolver.by_id[class_id]
        if (base["file_path"], f"{base['qualname']}.{symbol['name']}") in resolver.by_file_qualname:
            return True
        pending.extend(resolver.class_bases.get(class_id, []))
    return False


def dead(conn: Connection, min_conf: float = 0.6, include_tests: bool = False) -> list[dict]:
    """Uncalled candidates with exports, protocol methods and overrides excluded."""
    files = db.list_files(conn)
    exports = {file["path"]: file.get("exports") or [] for file in files}
    resolver = Resolver(db.get_all_symbols(conn), files)
    result = []
    for symbol in db.dead_candidates(conn, min_conf, include_tests):
        if symbol["name"].startswith("__") and symbol["name"].endswith("__"):
            continue
        if symbol["qualname"] in exports.get(symbol["file_path"], []):
            continue
        if symbol["kind"] == "method" and _overrides_base(symbol, resolver):
            continue
        result.append(
            {
                **symbol,
                "label": "candidate",
                "location": f"{symbol['file_path']}:{symbol['start_line']}-{symbol['end_line']}",
            }
        )
    return result


def where(conn: Connection, query: str) -> list[dict]:
    """Return all matching locations; ambiguous names are useful in this report."""
    exact = db.get_symbol(conn, query)
    rows = [exact] if exact else db.lookup_symbols(conn, query, limit=100)
    if not rows and "::" not in query:
        rows = db.trigram_search(conn, query, limit=5)
    return [
        {**row, "location": f"{row['file_path']}:{row['start_line']}-{row['end_line']}"}
        for row in rows
    ]
