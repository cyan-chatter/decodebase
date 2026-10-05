"""Live isolated 9B-generator / Coder7 verifier validation; no production DB changes."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import psycopg
from psycopg import sql

from cfl.config import load_settings
from cfl.core import db
from cfl.core.client import OllamaClient
from cfl.engines.ask import answer, block_for
from cfl.engines.guardrails import GUARD_VERSION, guard_answer
from cfl.eval.runner import load_questions
from cfl.pipeline.build import run_build
from cfl.pipeline.indexer import build_dense, build_lexical
from cfl.pipeline.knowledge import _draft_hash, generate_knowledge, validate_knowledge
from cfl.pipeline.scan import run_stage1, run_stage2

ROOT = Path("docs/validation/guardrails")


def save(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = ROOT / (name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(ROOT / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-checkpoint", action="store_true")
    parser.add_argument("--refresh-drafts", action="store_true")
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--resume-answers", action="store_true")
    parser.add_argument("--reuse-adversarial", action="store_true")
    parser.add_argument("--knowledge-only", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    s = load_settings().model_copy(update={"state_dir": ".cfl/guardrails-validation"})
    base = s.dsn.rsplit("/", 1)[0]
    name = "cfl_guard_" + uuid.uuid4().hex[:12]
    config = s.model_copy(update={"dsn": base + "/" + name})
    with psycopg.connect(base + "/postgres", autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        with OllamaClient(config) as client:
            start = time.monotonic()

            def checkpoint(conn):
                save(
                    "checkpoint.json",
                    {
                        "drafts": db.knowledge_drafts(conn),
                        "symbols": db.get_all_symbols(conn),
                        "files": db.list_files(conn),
                        "feature_selections": conn.execute(
                            "SELECT key,value FROM meta WHERE key LIKE 'feature_selection:%%'"
                        ).fetchall(),
                    },
                )

            if args.resume_checkpoint:
                snapshot = json.loads((ROOT / "checkpoint.json").read_text())
                with db.connect(config.dsn) as conn:
                    db.run_migrations(conn, config.embed_dim, config.migrations_dir)
                    run_stage1(conn, config, "eval/sample_repo")
                    run_stage2(conn, config, "eval/sample_repo")
                    digests = {m["digest"] for m in client.tags() if m["name"] == config.gen_model}
                    with conn.transaction():
                        for key, value in snapshot.get("feature_selections", []):
                            db.set_meta(conn, key, value)
                        for s in snapshot["symbols"]:
                            current = db.get_symbol(conn, s["id"])
                            if not current or current["code_hash"] != s["code_hash"]:
                                raise ValueError("Checkpoint source changed")
                            if s["status"] == "skipped_trivial":
                                db.save_trivial_summary(
                                    conn,
                                    s["id"],
                                    s["summary_short"],
                                    s["ctx_hash"],
                                    s["summary_json"],
                                )
                            elif s["status"] == "done":
                                db.save_symbol_summary(
                                    conn,
                                    s["id"],
                                    s["summary_json"],
                                    s["summary_short"],
                                    None,
                                    s["ctx_hash"],
                                    increment_attempts=False,
                                )
                        current_files = {f["path"]: f for f in db.list_files(conn)}
                        for f in snapshot["files"]:
                            if current_files[f["path"]]["sha256"] != f["sha256"]:
                                raise ValueError("Checkpoint file changed")
                            db.set_file_skeleton(conn, f["path"], f["skeleton"], f["line_count"])
                        for d in snapshot["drafts"]:
                            if d["generator_digest"] not in digests:
                                raise ValueError("Checkpoint generator changed")
                            d["draft_hash"] = _draft_hash(
                                d["draft"], d["evidence"], d["generator_digest"], config
                            )
                            db.put_knowledge_draft(conn, d)
                            if (
                                d["guard_version"] == GUARD_VERSION
                                and d["review"]
                                and d["verifier_digest"]
                            ):
                                db.save_knowledge_review(
                                    conn,
                                    d["id"],
                                    d["draft_hash"],
                                    d["status"],
                                    d["published"],
                                    d["review"],
                                    d["verifier_digest"],
                                    GUARD_VERSION,
                                )
                    print("CHECKPOINT", len(snapshot["drafts"]), "drafts restored", flush=True)
                    generated = (
                        generate_knowledge(conn, client, config) if args.refresh_drafts else None
                    )
                    checkpoint(conn)
                    reviewed = validate_knowledge(conn, client, config)
                    build_lexical(conn)
                    build = {
                        "resumed_checkpoint": True,
                        "knowledge_generation": generated,
                        "knowledge_validation": reviewed,
                        "dense": build_dense(conn, client, config),
                    }
            else:
                build = run_build(
                    "eval/sample_repo",
                    settings=config,
                    client=client,
                    on_drafts=checkpoint,
                    progress=lambda r, total: print(
                        "summaries", r.saved + r.cached + r.failed, "/", total, flush=True
                    ),
                )
            save("build.json", {**build, "wall_s": time.monotonic() - start})
            print("BUILD", build.get("knowledge_validation"), flush=True)
            with db.connect(config.dsn) as conn:
                save("knowledge.json", db.knowledge_drafts(conn))
                tests = [
                    (
                        "stub",
                        "services/notifications.py::send_email",
                        "send_email delivers an email [services/notifications.py:9-11].",
                    ),
                    (
                        "short-circuit",
                        "auth/tokens.py::validate_token",
                        "validate_token invokes issue_token if the separator is missing or comparison fails [auth/tokens.py:17-22].",
                    ),
                    (
                        "mutation",
                        "db/repository.py::AuditedRepository.save",
                        "AuditedRepository.save modifies the input dictionary to add audited=True [db/repository.py:27-30].",
                    ),
                    (
                        "exceptions",
                        "auth/passwords.py::verify_password",
                        "verify_password has no raised exceptions [auth/passwords.py:12-14].",
                    ),
                    (
                        "concurrency-guarantee",
                        "db/connection.py::Connection.execute",
                        "Connection.execute is not thread-safe [db/connection.py:11-17].",
                    ),
                    (
                        "security-guarantee",
                        "auth/passwords.py::hash_password",
                        "hash_password securely stores passwords and prevents timing attacks [auth/passwords.py:7-9].",
                    ),
                ]
                adversarial = []
                if args.reuse_adversarial:
                    adversarial = json.loads((ROOT / "adversarial.json").read_text())
                    tests = []
                for label, id, text in tests:
                    row = db.get_symbol(conn, id)
                    result = guard_answer(
                        conn, client, config, text, block_for(row), "Explain " + row["qualname"]
                    )
                    adversarial.append(
                        {
                            "case": label,
                            "status": result.status,
                            "text": result.text,
                            "reasons": result.reasons,
                            "review": result.review,
                        }
                    )
                    print("ADVERSARIAL", label, result.status, flush=True)
                save("adversarial.json", adversarial)
                rows = (
                    json.loads((ROOT / "answers.json").read_text()) if args.resume_answers else []
                )
                questions = load_questions("eval/questions.yaml")
                if isinstance(questions, dict):
                    questions = questions["questions"]
                for q in questions:
                    if args.knowledge_only:
                        continue
                    if args.question_id and q["id"] not in args.question_id:
                        continue
                    start = time.monotonic()
                    r = answer(conn, client, config, q["question"])
                    elapsed = time.monotonic() - start
                    rows = [row for row in rows if row["id"] != q["id"]]
                    rows.append(
                        {
                            "id": q["id"],
                            "question": q["question"],
                            "status": r.get("status"),
                            "text": r["text"],
                            "citations": r["citations"],
                            "warnings": r["warnings"],
                            "cached": r["cached"],
                            "wall_s": elapsed,
                        }
                    )
                    order = {q["id"]: i for i, q in enumerate(questions)}
                    rows.sort(key=lambda row: order[row["id"]])
                    save("answers.json", rows)
                    print(
                        "QUESTION",
                        len(rows),
                        q["id"],
                        r.get("status"),
                        round(elapsed, 2),
                        flush=True,
                    )
                save("published.json", db.published_knowledge(conn))
    finally:
        with psycopg.connect(base + "/postgres", autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )
        print("Temporary DB removed", flush=True)


if __name__ == "__main__":
    main()
