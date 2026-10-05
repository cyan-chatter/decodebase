from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cfl.core import db

if TYPE_CHECKING:
    from psycopg import Connection

LEXICAL_VERSION = "3"
_WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+|[^\W\d_]+", re.UNICODE)

_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "does",
        "do",
        "how",
        "why",
        "where",
        "who",
        "which",
        "when",
        "can",
        "could",
        "would",
        "should",
        "to",
        "of",
        "for",
        "from",
        "with",
        "and",
        "or",
        "in",
        "on",
        "by",
        "it",
        "this",
        "that",
        "these",
        "those",
        "as",
    }
)


def split_identifiers(text: str) -> list[str]:
    """Split snake_case, CamelCase, acronym transitions and digits into safe lexemes."""
    return list(dict.fromkeys(word.lower() for word in _WORDS.findall(text)))


def build_search_text(symbol: dict) -> str:
    fields = [
        symbol.get(key) or ""
        for key in ("file_path", "qualname", "name", "signature", "docstring", "summary_short")
    ]
    fields.extend(symbol.get("decorators") or [])
    text = "\n".join(field for field in fields if field)
    return (text + "\n" + " ".join(split_identifiers(text))).lower()


def build_lexical(conn: Connection, *, batch_size: int = 256) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = 0
    with conn.transaction():
        for batch in db.lexical_symbol_batches(conn, batch_size):
            from cfl.pipeline.knowledge import trusted_rows

            batch = trusted_rows(conn, batch)
            db.set_search_text(
                conn, [{"id": row["id"], "search_text": build_search_text(row)} for row in batch]
            )
            count += len(batch)
        db.set_view_status(conn, "lexical", "fresh", {"version": LEXICAL_VERSION})
    return count


def retrieve_lexical(conn: Connection, query: str, limit: int = 5) -> list[dict]:
    """Fuse FTS and qualname/name trigram ranks, with stable ID tie-breaking."""
    if limit < 1:
        raise ValueError("limit must be positive")
    terms = [term for term in split_identifiers(query) if term not in _QUERY_STOP_WORDS]
    if not terms:
        return []
    candidates = max(20, limit * 4)
    ranked = [
        db.lexical_search(conn, terms, candidates),
        db.trigram_search(conn, query, candidates),
    ]
    scores: dict[str, float] = {}
    for result in ranked:
        for rank, row in enumerate(result, 1):
            scores[row["id"]] = scores.get(row["id"], 0) + 1 / rank
    ids = sorted(scores, key=lambda id: (-scores[id], id))[:limit]
    # Hydrate spans from the current index; never use static line numbers in the suite.
    hydrated = {row["id"]: row for row in db.get_symbols(conn, ids)}
    return [{**hydrated[id], "rank": scores[id]} for id in ids if id in hydrated]


def embedding_text(text, model, *, query=False):
    """Publisher-prescribed asymmetric retrieval instructions."""
    if model.split(":")[0] == "nomic-embed-text":
        return ("search_query: " if query else "search_document: ") + text
    if model.split(":")[0] == "qwen3-embedding" and query:
        return (
            "Instruct: Given a code question, retrieve code that answers the question.\nQuery:"
            + text
        )
    return text


def embedding_head(text, cap, counter):
    """Keep the document prefix and source head within the embedding token cap."""
    if counter.count(text) <= cap:
        return text
    suffix = "\n[remaining code omitted from embedding]"
    low, high, best = 0, len(text), ""
    while low <= high:
        length = (low + high) // 2
        candidate = text[:length] + suffix
        if counter.count(candidate) <= cap:
            best, low = candidate, length + 1
        else:
            high = length - 1
    if not best:
        raise ValueError("Embedding cap cannot fit the truncation marker")
    return best


def build_dense(conn, client, settings):
    from pathlib import Path

    from cfl.core.hashing import embed_key
    from cfl.core.memory import KnowledgeMemory
    from cfl.parser.python_adapter import PythonAdapter

    try:
        memory = KnowledgeMemory(conn, client, settings)
        old_keys = {}
        units = []
        from cfl.pipeline.knowledge import trusted_rows

        for s in trusted_rows(conn, db.get_all_symbols(conn)):
            if s["status"] not in {"done", "skipped_trivial"}:
                continue
            text = "\n".join(
                [
                    s["file_path"],
                    s["signature"] or "",
                    s["summary_short"] or "",
                    s["docstring"] or "",
                    s["raw_code"],
                ]
            )
            old_keys[("symbol", s["id"])] = s["embed_hash"]
            units.append(("symbol", s["id"], text))
        root = Path(db.get_meta(conn, "repo_root") or ".")
        for f in db.list_files(conn):
            path = root / f["path"]
            if (
                f["parse_status"] not in {"done", "parsed"} and f["language"] != "config"
            ) or not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8")
            if f["language"] == "python":
                parsed = PythonAdapter().parse(Path(f["path"]), raw)
                skeleton = PythonAdapter().skeleton(parsed)
            else:
                skeleton = raw
            with conn.transaction():
                db.set_file_skeleton(conn, f["path"], skeleton, max(1, len(raw.splitlines())))
            old_keys[("file", f["path"])] = f["embed_hash"]
            units.append(("file", f["path"], f["path"] + "\n" + skeleton))
        for unit in db.summary_units(conn):
            old_keys[(unit["kind"], unit["unit_id"])] = unit["embed_hash"]
            units.append((unit["kind"], unit["unit_id"], unit["summary_short"]))
        capped = []
        for kind, id, text in units:
            # Conservative cap; server-side truncation remains disabled and failures explicit.
            text = embedding_text(text, settings.embed_model)
            text = embedding_head(text, settings.embed_max_tokens, client.counter)
            capped.append(
                (
                    kind,
                    id,
                    text,
                    embed_key(text, settings.embed_model, memory.digests[settings.embed_model]),
                )
            )
        vectors = memory.embed([u[2] for u in capped])
        with conn.transaction():
            for kind in ("symbol", "file", "feature", "module", "knowledge"):
                db.set_embedding_keys(
                    conn, kind, [{"id": u[1], "hash": u[3]} for u in capped if u[0] == kind]
                )
            db.set_view_status(
                conn,
                "dense",
                "fresh",
                {
                    "version": "1",
                    "model": settings.embed_model,
                    "digest": memory.digests[settings.embed_model],
                    "count": len(capped),
                },
            )
            if any(old_keys.get((u[0], u[1])) != u[3] for u in capped):
                db.bump_epoch(conn)
        return {"count": len(vectors), "failed": False}
    except Exception as exc:  # noqa: BLE001 - independent dense-view failure
        with conn.transaction():
            db.set_view_status(conn, "dense", "failed", {"error": str(exc)})
        return {"count": 0, "failed": True, "error": str(exc)}
