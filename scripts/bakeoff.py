"""Reproducible M6 comparison. Pull verified candidates separately, one at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from cfl.config import load_settings
from cfl.core.client import OllamaClient
from cfl.core.preflight import run_doctor
from cfl.eval.metrics import answer_recall_at_5
from cfl.eval.runner import load_questions
from cfl.parser.python_adapter import PythonAdapter
from cfl.prompts.prompts import PROMPT_VERSION, SUMMARY_PREFIX, SYSTEM_INGEST, build_symbol_prompt
from cfl.prompts.schemas import SCHEMA_VERSION, SYMBOL_SUMMARY_JSON_SCHEMA, SymbolSummary


def symbols_from(root: Path) -> list[dict]:
    adapter = PythonAdapter()
    rows = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        parsed = adapter.parse(Path(relative), path.read_text())
        if parsed.parse_error:
            raise ValueError(f"{relative}: {parsed.parse_error}")
        for symbol in parsed.symbols:
            row = asdict(symbol)
            row.update(id=f"{relative}::{symbol.qualname}", file_path=relative)
            row["source_sha256"] = hashlib.sha256(symbol.raw_code.encode()).hexdigest()
            rows.append(row)
    if not rows:
        raise ValueError("No symbols to benchmark")
    return rows


def summary_result(client: OllamaClient, symbol: dict, contract: dict | None = None) -> dict:
    system, prompt = build_symbol_prompt(symbol, [])
    schema = SYMBOL_SUMMARY_JSON_SCHEMA
    if contract is not None:
        system = contract["system"]
        prompt = contract["summary_prefix"] + prompt[len(SUMMARY_PREFIX) :]
        schema = contract["schema"]
    start = time.monotonic()
    row = {"id": symbol["id"], "source_sha256": symbol["source_sha256"]}
    try:
        result = client.generate(
            prompt,
            system,
            fmt=schema,
            num_predict=client.settings.num_predict_symbol,
            task="bakeoff",
            symbol_id=symbol["id"],
        )
        row.update(asdict(result))
        try:
            json.loads(result.text)
            row["json_valid"] = True
            row["summary"] = SymbolSummary.model_validate_json(result.text).model_dump()
            row["schema_valid"] = True
        except ValueError as exc:
            row.update(schema_valid=False, error=str(exc))
            row.setdefault("json_valid", False)
    except Exception as exc:  # noqa: BLE001 - preserve failed candidate evidence
        row.update(json_valid=False, schema_valid=False, error=f"{type(exc).__name__}: {exc}")
    row["wall_s"] = time.monotonic() - start
    return row


def generation_metrics(rows: list[dict], project_symbols: int) -> dict:
    if not rows or project_symbols < 1:
        raise ValueError("Positive sample and projection required")
    output = sum(row.get("output_tokens", 0) for row in rows)
    duration = sum(row.get("eval_duration", 0) for row in rows) / 1e9
    seconds = sum(row["wall_s"] for row in rows) / len(rows)
    return {
        "samples": len(rows),
        "json_validity": sum(r["json_valid"] for r in rows) / len(rows),
        "schema_validity": sum(r["schema_valid"] for r in rows) / len(rows),
        "prefill_tps": (
            sum(
                r["prompt_tokens"] - r["cached_prompt_tokens"]
                for r in rows
                if r.get("cached_prompt_tokens") is not None
                and r["prompt_tokens"] - r["cached_prompt_tokens"] >= 32
            )
            / (
                sum(
                    r["prompt_eval_duration"]
                    for r in rows
                    if r.get("cached_prompt_tokens") is not None
                    and r["prompt_tokens"] - r["cached_prompt_tokens"] >= 32
                )
                / 1e9
            )
            if any(
                r.get("cached_prompt_tokens") is not None
                and r["prompt_tokens"] - r["cached_prompt_tokens"] >= 32
                and r.get("prompt_eval_duration")
                for r in rows
            )
            else None
        ),
        "gen_tps": output / duration if duration else None,
        "mean_wall_s": seconds,
        "project_symbols": project_symbols,
        "projected_ingest_hours": seconds * project_symbols / 3600,
        "projected_10000_symbol_hours": seconds * 10000 / 3600,
    }


def embedding_text(text: str, model: str, *, query: bool) -> str:
    if model.split(":")[0] == "nomic-embed-text":
        return ("search_query: " if query else "search_document: ") + text
    if model.split(":")[0] == "qwen3-embedding" and query:
        return (
            "Instruct: Given a code question, retrieve code that answers the question.\nQuery:"
            + text
        )
    return text


def dense_rank(documents: list[list[float]], queries: list[list[float]]) -> list[list[int]]:
    docs, query = np.asarray(documents, dtype=float), np.asarray(queries, dtype=float)
    if (
        docs.ndim != 2
        or query.ndim != 2
        or docs.shape[1] != query.shape[1]
        or not np.isfinite(docs).all()
        or not np.isfinite(query).all()
    ):
        raise ValueError("Invalid or mismatched embedding matrices")
    dnorm, qnorm = np.linalg.norm(docs, axis=1), np.linalg.norm(query, axis=1)
    if np.any(dnorm == 0) or np.any(qnorm == 0):
        raise ValueError("Zero embedding cannot be cosine ranked")
    scores = (query / qnorm[:, None]) @ (docs / dnorm[:, None]).T
    return np.argsort(-scores, axis=1, kind="stable")[:, :5].tolist()


class GPU:
    def __enter__(self):
        self.values = []
        self.stop = threading.Event()

        def sample():
            while not self.stop.is_set():
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                )
                self.values.append(sum(int(v) for v in raw.splitlines()))
                self.stop.wait(0.2)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak_mib = max(self.values) if self.values else None


def save(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def unload(api: httpx.Client) -> None:
    for row in api.get("/api/ps").json()["models"]:
        api.post("/api/generate", json={"model": row["name"], "keep_alive": 0}).raise_for_status()
    deadline = time.monotonic() + 30
    while api.get("/api/ps").json()["models"]:
        if time.monotonic() > deadline:
            raise TimeoutError("Models did not unload")
        time.sleep(0.2)
    time.sleep(1)  # allow the released runner allocations to settle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen", action="append", default=[])
    parser.add_argument("--embed", action="append", default=[])
    parser.add_argument("--contract", type=Path, help="Replay an archived prompt/schema contract")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/validation/milestone-6/candidates.json"),
        help="Prior tag/size/license verification record",
    )
    parser.add_argument("--repo", type=Path, default=Path("eval/sample_repo"))
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.yaml"))
    parser.add_argument("--project-repo", type=Path, default=Path("cfl"))
    parser.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        metavar="TAG=PATH",
        help="Matching exact tokenizer per generation candidate",
    )
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=Path(".cfl/bakeoff/report.json"))
    args = parser.parse_args()
    if not args.gen and not args.embed:
        parser.error("Supply at least one --gen or --embed (already pulled and verified)")
    tokenizers = {}
    for pair in args.tokenizer:
        tag, separator, path = pair.partition("=")
        if not separator or not Path(path).is_file():
            parser.error("--tokenizer must be TAG=existing-file")
        tokenizers[tag] = path
    contract = (
        json.loads(args.contract.read_text())
        if args.contract
        else {
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "system": SYSTEM_INGEST,
            "summary_prefix": SUMMARY_PREFIX,
            "schema": SYMBOL_SUMMARY_JSON_SCHEMA,
        }
    )
    catalog = json.loads(args.catalog.read_text())
    for tag in args.gen + args.embed:
        key = tag if ":" in tag else tag + ":latest"
        record = catalog["models"].get(key, {})
        if not all(
            record.get(field)
            for field in ("library_url", "download_size", "license", "license_url")
        ):
            parser.error(f"Verify tag/size/license and add {key} to --catalog first")
    settings = load_settings().model_copy(update={"num_ctx": args.num_ctx, "tokenizer_file": None})
    symbols = symbols_from(args.repo)
    questions = [
        q
        for q in load_questions(args.questions, verified_only=True)
        if q["type"] not in {"structural", "traversal"}
    ]
    by_id = {row["id"]: row for row in symbols}
    for question in questions:
        if set(question["expected"]["symbols"]) - by_id.keys():
            raise ValueError("Question evidence is absent from this corpus")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "num_ctx": args.num_ctx,
        "candidate_verification": catalog,
        "kv_quantization": settings.kv_quantization,
        "prompt_version": contract["prompt_version"],
        "schema_version": contract["schema_version"],
        "prompt_contract": contract,
        "generation_options": {
            "temperature": settings.temperature,
            "seed": settings.seed,
            "num_predict": settings.num_predict_symbol,
            "disable_thinking": settings.gen_disable_thinking,
        },
        "symbols": symbols,
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "generators": [],
        "embedders": [],
        "projection_method": "sample mean wall time * parsed project symbol count; M7 build --estimate deferred",
        "document_method": "file path + signature + docstring + full fixture source; no generated summaries or expected answers",
    }
    project_count = len(symbols_from(args.project_repo))
    with httpx.Client(base_url=settings.ollama_url, timeout=600) as api:
        report["ollama_version"] = api.get("/api/version").json()["version"]
        available = {m["name"]: m for m in api.get("/api/tags").json()["models"]}
        for tag in args.gen + args.embed:
            if (tag if ":" in tag else tag + ":latest") not in available:
                raise ValueError(f"Candidate not pulled: {tag}")
        original = api.get("/api/ps").json()["models"]
        try:
            for tag in args.gen:
                print(f"Generator {tag}", flush=True)
                unload(api)
                config = settings.model_copy(
                    update={
                        "gen_model": tag,
                        "tokenizer_file": tokenizers.get(tag),
                        "state_dir": str(args.out.parent / tag.replace(":", "-")),
                    }
                )
                with GPU() as gpu, OllamaClient(config) as client:
                    client.generate("Return READY.", "", num_predict=4, task="bakeoff_warmup")
                    rows = []
                    entry = {"model": tag, "digest": available[tag]["digest"], "rows": rows}
                    report["generators"].append(entry)
                    for symbol in symbols:
                        rows.append(summary_result(client, symbol, contract))
                        save(args.out, report)
                        print(
                            f" {len(rows)}/{len(symbols)} {symbol['id']} valid={rows[-1]['json_valid']}",
                            flush=True,
                        )
                    entry["ps"] = client.ps()
                    entry["metrics"] = generation_metrics(rows, project_count)
                    entry["doctor_exit"] = run_doctor(config)
                    entry["doctor_ps"] = client.ps()
                entry["peak_gpu_mib"] = gpu.peak_mib
                save(args.out, report)
            for tag in args.embed:
                print(f"Embedder {tag}", flush=True)
                unload(api)
                with GPU() as gpu:
                    response = api.post(
                        "/api/embed",
                        json={
                            "model": tag,
                            "input": ["dimension probe"],
                            "truncate": False,
                            "keep_alive": -1,
                        },
                    )
                    response.raise_for_status()
                    dimension = len(response.json()["embeddings"][0])
                    config = settings.model_copy(
                        update={
                            "embed_model": tag,
                            "embed_dim": dimension,
                            "state_dir": str(args.out.parent / tag.replace(":", "-")),
                        }
                    )
                    with OllamaClient(config) as client:
                        documents = [
                            embedding_text(
                                "\n".join(
                                    [
                                        s["file_path"],
                                        s["signature"],
                                        s.get("docstring") or "",
                                        s["raw_code"],
                                    ]
                                ),
                                tag,
                                query=False,
                            )
                            for s in symbols
                        ]
                        queries = [
                            embedding_text(q["question"], tag, query=True) for q in questions
                        ]
                        start = time.monotonic()
                        docs = client.embed(documents)
                        vectors = client.embed(queries)
                        elapsed = time.monotonic() - start
                        ranking = dense_rank(docs, vectors)
                        rows = []
                        for q, indices in zip(questions, ranking, strict=True):
                            retrieved = [symbols[i] for i in indices]
                            recall = answer_recall_at_5(
                                [by_id[id] for id in q["expected"]["symbols"]], retrieved
                            )
                            rows.append(
                                {
                                    "id": q["id"],
                                    "type": q["type"],
                                    "question": q["question"],
                                    "retrieved": [s["id"] for s in retrieved],
                                    "answer_recall_at_5": recall,
                                }
                            )
                        entry = {
                            "model": tag,
                            "digest": available[tag if ":" in tag else tag + ":latest"]["digest"],
                            "dimension": dimension,
                            "rows": rows,
                            "answer_recall_at_5": statistics.mean(
                                r["answer_recall_at_5"] for r in rows
                            ),
                            "embedding_wall_s": elapsed,
                            "ps": client.ps(),
                            "doctor_exit": run_doctor(config),
                            "doctor_ps": client.ps(),
                        }
                entry["peak_gpu_mib"] = gpu.peak_mib
                report["embedders"].append(entry)
                save(args.out, report)
        finally:
            unload(api)
            for row in original:
                info = api.post("/api/show", json={"model": row["name"]}).json()
                embedding_only = "embedding" in info.get(
                    "capabilities", []
                ) and "completion" not in info.get("capabilities", [])
                endpoint = "/api/embed" if embedding_only else "/api/generate"
                body = {
                    "model": row["name"],
                    "keep_alive": -1,
                    "options": {"num_ctx": row.get("context_length", args.num_ctx)},
                }
                body.update({"input": ["restore"]} if embedding_only else {"prompt": ""})
                api.post(endpoint, json=body).raise_for_status()
            report["restored"] = api.get("/api/ps").json()["models"]
            save(args.out, report)


if __name__ == "__main__":
    main()
