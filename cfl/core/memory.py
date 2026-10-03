from __future__ import annotations

import json
from dataclasses import dataclass, field

from cfl.core import db
from cfl.core.budget import PromptPart, TokenCounter, assemble
from cfl.core.errors import LLMValidationError, SymbolNotFound
from cfl.core.hashing import ctx_hash, embed_key, fingerprint
from cfl.prompts.prompts import PROMPT_VERSION, build_symbol_prompt
from cfl.prompts.schemas import SCHEMA_VERSION, SYMBOL_SUMMARY_JSON_SCHEMA, SymbolSummary


def evidence_reference(symbol: dict) -> dict:
    return {key: symbol[key] for key in ("id", "code_hash", "file_path", "start_line", "end_line")}


def answer_key(question: str, mode: str, settings: dict, session_state: dict | None = None) -> str:
    return fingerprint(
        {
            "version": 2,
            "question": question,
            "mode": mode,
            "settings": settings,
            "session_state": session_state,
        }
    )


class KnowledgeMemory:
    """Durable source-backed summaries and embeddings; no accumulated ingest history."""

    def __init__(self, conn, client, settings):
        self.conn, self.client, self.settings = conn, client, settings
        tags = {row["name"]: row for row in client.tags()}
        self.digests = {}
        for model in (settings.gen_model, settings.embed_model):
            tag = model if ":" in model else model + ":latest"
            row = tags.get(tag)
            if row is None or not row.get("digest"):
                raise LLMValidationError(f"Missing resolved model digest: {tag}")
            self.digests[model] = row["digest"]
        self.options = {
            key: getattr(settings, key)
            for key in (
                "num_ctx",
                "temperature",
                "seed",
                "gen_disable_thinking",
                "num_predict_symbol",
                "kv_quantization",
                "kv_quantization_type",
            )
        }
        with conn.transaction():
            for key, value in {
                "gen_model_tag": settings.gen_model,
                "embed_model_tag": settings.embed_model,
                "gen_model_digest": self.digests[settings.gen_model],
                "embed_model_digest": self.digests[settings.embed_model],
                "prompt_version": PROMPT_VERSION,
                "generation_options": json.dumps(self.options, sort_keys=True),
            }.items():
                db.set_meta(conn, key, value)

    def summarize_symbol(self, id: str, callees: list[tuple[str, str]]) -> SymbolSummary:
        symbol = db.get_symbol(self.conn, id)
        if symbol is None:
            raise SymbolNotFound(id)
        key = ctx_hash(
            symbol["code_hash"],
            callees,
            PROMPT_VERSION,
            self.settings.gen_model,
            gen_model_digest=self.digests[self.settings.gen_model],
            generation_options=self.options,
            schema_version=SCHEMA_VERSION,
        )
        if symbol["ctx_hash"] == key and symbol["status"] == "done" and symbol["summary_json"]:
            try:
                return SymbolSummary.model_validate(symbol["summary_json"])
            except ValueError:
                pass  # Corrupt cached output is regenerated and validated before saving.
        system, prompt = build_symbol_prompt(symbol, callees)
        result = self.client.generate(
            prompt,
            system,
            fmt=SYMBOL_SUMMARY_JSON_SCHEMA,
            num_predict=self.settings.num_predict_symbol,
            task="symbol",
            symbol_id=id,
        )
        try:
            summary = SymbolSummary.model_validate_json(result.text)
        except ValueError as exc:
            raise LLMValidationError("Invalid structured symbol summary") from exc
        with self.conn.transaction():
            current = db.get_symbol(self.conn, id)
            if current is None or evidence_reference(current) != evidence_reference(symbol):
                raise LLMValidationError("Source changed while summarizing; rescan and retry")
            db.save_symbol_summary(
                self.conn, id, summary.model_dump(), summary.one_liner, None, key
            )
            db.bump_epoch(self.conn)
            db.set_view_status(self.conn, "lexical", "stale", {})
        return summary

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [
            embed_key(text, self.settings.embed_model, self.digests[self.settings.embed_model])
            for text in texts
        ]
        cached = db.cached_embeddings(self.conn, keys)
        missing = dict(zip(keys, texts))
        missing = {key: text for key, text in missing.items() if key not in cached}
        if missing:
            vectors = self.client.embed(list(missing.values()))
            if len(vectors) != len(missing):
                raise LLMValidationError("Embedding count mismatch")
            new = dict(zip(missing, vectors))
            with self.conn.transaction():
                db.upsert_embeddings(
                    self.conn, [{"hash": key, "vec": vector} for key, vector in new.items()]
                )
            cached.update(db.cached_embeddings(self.conn, list(new)))
        return [cached[key] for key in keys]


@dataclass
class SessionMemory:
    """Bounded working memory. Only the returned context becomes model-visible."""

    session_id: str
    turns: list[dict] = field(default_factory=list)
    direction_seed: list[dict] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    max_turns: int = 4
    max_seed: int = 15
    index_revision: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_turns <= 16 or not 1 <= self.max_seed <= 64:
            raise ValueError("Session limits must be positive and bounded")
        self.turns = self.turns[-self.max_turns :]
        self.direction_seed = self.direction_seed[-self.max_seed :]

    def add_turn(self, question: str, answer: str, symbols: list[dict]) -> None:
        self.turns.append({"question": question, "answer": answer})
        self.turns = self.turns[-self.max_turns :]
        references = {row["id"]: row for row in self.direction_seed}
        for symbol in symbols:
            references.pop(symbol["id"], None)
            references[symbol["id"]] = evidence_reference(symbol)
        self.direction_seed = list(references.values())[-self.max_seed :]

    def save(self, conn) -> None:
        self.context(conn)
        state = {
            "turns": self.turns[-self.max_turns :],
            "direction_seed": self.direction_seed[-self.max_seed :],
            "constraints": self.constraints[:15],
            "max_turns": self.max_turns,
            "max_seed": self.max_seed,
        }
        db.save_chat_session(
            conn,
            self.session_id,
            db.get_meta(conn, "repo_root") or "",
            db.index_version(conn),
            state,
        )

    @classmethod
    def load(cls, conn, session_id: str) -> SessionMemory:
        row = db.load_chat_session(conn, session_id)
        if row is None:
            return cls(session_id)
        if row["repo_root"] != (db.get_meta(conn, "repo_root") or ""):
            raise ValueError("Session belongs to a different repository")
        result = cls(session_id, **row["state"], index_revision=row["index_version"])
        result.direction_seed = [
            ref for ref in result.direction_seed if db.evidence_is_current(conn, [ref])
        ]
        if row["index_version"] != db.index_version(conn):
            # Old answers may describe deleted behavior; keep constraints and valid hints only.
            result.turns = []
        return result

    def context(self, conn) -> dict:
        current_revision = db.index_version(conn)
        if self.index_revision is not None and self.index_revision != current_revision:
            self.turns = []
        self.index_revision = current_revision
        self.direction_seed = [
            ref for ref in self.direction_seed if db.evidence_is_current(conn, [ref])
        ]
        return {
            "recent_turns": self.turns[-self.max_turns :],
            "user_constraints": self.constraints[:15],
            "sources": db.get_symbols(conn, [ref["id"] for ref in self.direction_seed]),
        }

    def assemble_prompt(self, conn, question: str, settings, *, num_predict: int, counter=None):
        """Budget recalled source before history; never treat remembered answers as facts."""
        context = self.context(conn)
        parts = [PromptPart("question", question, priority=1, truncatable=False)]
        if context["user_constraints"]:
            parts.append(
                PromptPart(
                    "constraints",
                    json.dumps(context["user_constraints"]),
                    priority=1,
                    truncatable=False,
                )
            )
        for symbol in context["sources"]:
            location = f"[{symbol['file_path']}:{symbol['start_line']}-{symbol['end_line']}]"
            parts.append(
                PromptPart(
                    symbol["id"],
                    location + "\n" + symbol["raw_code"],
                    priority=2,
                    truncatable=False,
                )
            )
        parts.append(
            PromptPart("history", json.dumps(context["recent_turns"]), priority=4, truncatable=True)
        )
        return assemble(
            "Answer using only supplied current source evidence. Cite [path:start-end]. "
            "Previous answers are discussion history, not verified facts. "
            "Say what is missing when evidence is insufficient.",
            parts,
            num_ctx=settings.num_ctx,
            num_predict=num_predict,
            overhead=settings.template_overhead,
            counter=counter or TokenCounter(settings),
        )
