from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cfl.core import db
from cfl.core.errors import AmbiguousSymbol, SymbolNotFound
from cfl.engines import graph_queries
from cfl.pipeline.indexer import retrieve_lexical, split_identifiers

if TYPE_CHECKING:
    from psycopg import Connection

RETRIEVAL_VERSION = "1"
_TRACE = re.compile(
    r"^\s*(?:trace|path|flow)\s+(?:from\s+)?(.+?)\s+to\s+(.+?)\s*[.!?]*$", re.IGNORECASE
)
_ACTIONS = (
    frozenset({"write", "save", "persist", "persistence", "store"}),
    frozenset({"lookup", "get", "find", "read", "retrieve", "retrieval"}),
    frozenset({"send", "deliver", "delivery", "email"}),
    frozenset({"sign", "signing", "issue"}),
)


@dataclass
class EvidenceResult:
    blocks: list[dict]
    lexical_blocks: list[dict]
    path_ids: list[str] = field(default_factory=list)
    path_hops: list[dict] = field(default_factory=list)
    status: str = "lexical"
    complete: bool = False
    requires_batching: bool = False
    candidates: list[str] = field(default_factory=list)


def _resolve_reference(conn: Connection, reference: str):
    reference = reference.strip().strip("`").rstrip(".?!")
    # Module-qualified names refer to file-relative IDs, not globally unique names.
    pieces = reference.split(".")
    for boundary in range(len(pieces) - 1, 0, -1):
        id = "/".join(pieces[:boundary]) + ".py::" + ".".join(pieces[boundary:])
        if row := db.get_symbol(conn, id):
            return graph_queries.Symbol.from_row(row)
    if row := db.get_symbol(conn, reference):
        return graph_queries.Symbol.from_row(row)
    rows = db.lookup_symbols(conn, reference, limit=100)
    name = reference.partition("::")[2] if "::" in reference else reference
    for predicate in (
        lambda row: row["qualname"] == name,
        lambda row: row["name"] == name,
        lambda row: row["qualname"].endswith("." + name),
    ):
        candidates = [graph_queries.Symbol.from_row(row) for row in rows if predicate(row)]
        if len(candidates) > 1:
            raise AmbiguousSymbol(candidates)
        if candidates:
            return candidates[0]
    # A prose endpoint must go through inference, not fuzzy symbol selection.
    raise SymbolNotFound(reference)


def _infer_target(conn: Connection, source: str, phrase: str, max_depth: int, min_conf: float):
    terms = set(split_identifiers(phrase))
    actions = set().union(*(group for group in _ACTIONS if terms & group))
    if not actions:
        return None, [], []
    rows = retrieve_lexical(conn, phrase + " " + " ".join(sorted(actions)), 20)
    matches = []
    for row in rows:
        name = set(split_identifiers(row["name"]))
        doc = set(split_identifiers(row.get("docstring") or ""))
        score = 2 * len(name & actions) if name & actions else len(doc & actions)
        if not score or row["id"] == source:
            continue
        hops = graph_queries.path(conn, source, row["id"], max_depth, min_conf)
        if hops is not None:
            matches.append((score, len(hops), row["id"], hops))
    if not matches:
        return None, [], []
    matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best = matches[0]
    tied = [item[2] for item in matches if item[:2] == best[:2]]
    if len(tied) > 1:
        return None, [], tied
    return best[2], best[3], []


def retrieve_evidence(
    conn: Connection,
    query: str,
    limit: int = 5,
    *,
    max_depth: int = 8,
    min_conf: float = 0.6,
    lexical_rows: list[dict] | None = None,
) -> EvidenceResult:
    """Retrieve source blocks, preserving every node of an explicitly requested path.

    Endpoint inference uses query text and indexed source, never eval expectations.
    Required path blocks can exceed limit: requires_batching makes that explicit.
    Generation callers must budget/split those blocks before sending a prompt.
    """
    if limit < 1 or max_depth < 1 or not 0 <= min_conf <= 1:
        raise ValueError("Require positive limits and confidence in [0,1]")
    lexical = retrieve_lexical(conn, query, limit) if lexical_rows is None else list(lexical_rows)
    result = EvidenceResult(list(lexical), lexical)
    match = _TRACE.fullmatch(query)
    if not match:
        return result
    try:
        source = _resolve_reference(conn, match[1])
    except AmbiguousSymbol as exc:
        result.status = "ambiguous_source"
        result.candidates = [candidate.id for candidate in exc.candidates]
        return result
    except SymbolNotFound:
        result.status = "unresolved_source"
        return result
    inferred = False
    try:
        target = _resolve_reference(conn, match[2])
    except AmbiguousSymbol as exc:
        result.status = "ambiguous_target"
        result.candidates = [candidate.id for candidate in exc.candidates]
        return result
    except SymbolNotFound:
        target_id, hops, candidates = _infer_target(conn, source.id, match[2], max_depth, min_conf)
        if target_id is None:
            result.status = "ambiguous_target" if candidates else "unresolved_target"
            result.candidates = candidates
            return result
        inferred = True
    else:
        target_id = target.id
        hops = graph_queries.path(conn, source.id, target_id, max_depth, min_conf)
    if hops is None:
        result.status = "no_path"
        return result
    ids = list(dict.fromkeys([source.id, *(hop["id"] for hop in hops)]))
    hydrated = {row["id"]: row for row in db.get_symbols(conn, ids)}
    if any(id not in hydrated for id in ids):
        result.status = "missing_source"
        return result
    blocks = []
    for index, id in enumerate(ids):
        incoming = hops[index - 1] if index else None
        blocks.append({**hydrated[id], "retrieval_source": "graph_path", "incoming_call": incoming})
    # Path blocks take precedence over high-ranking distractors; never cut the path.
    remaining = max(0, limit - len(blocks))
    blocks.extend(
        {**row, "retrieval_source": "lexical"} for row in lexical if row["id"] not in hydrated
    )
    blocks = blocks[: len(ids) + remaining]
    result.blocks = blocks
    result.path_ids = ids
    result.path_hops = hops
    result.complete = True
    result.requires_batching = len(ids) > limit
    result.status = "inferred_path" if inferred else "explicit_path"
    return result
