"""Isolated, cold application-cache model comparison; never changes app configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx
import psutil
import psycopg
import yaml
from psycopg import sql

from cfl.config import load_settings
from cfl.core import db
from cfl.core.client import OllamaClient
from cfl.core.preflight import check_residency, warm_and_benchmark
from cfl.core.runtime import ollama_environment
from cfl.eval.runner import load_config, load_questions, run_eval, summarize
from cfl.pipeline.indexer import build_dense, build_lexical
from cfl.pipeline.scan import run_stage1, run_stage2
from cfl.pipeline.symbol_pass import run_symbol_pass
from scripts.validate_kv_cache import gpu_mib, wait_ready

CANDIDATES = {
    "baseline": ("qwen3.5:9b", "qwen3.5-9b"),
    "coder14": ("qwen2.5-coder:14b", "qwen25-coder-14b"),
    "qwen14": ("qwen3:14b-q4_K_M", "qwen3-14b"),
    "qwen27": ("qwen3.5:27b", "qwen3.5-27b"),
}


class PlacementClient(OllamaClient):
    """Diagnostic-only CPU embedding placement; generation options stay unchanged."""

    def __init__(self, settings, *, cpu_embed=False, transport=None):
        super().__init__(settings, transport=transport)
        self.cpu_embed = cpu_embed

    def _post(self, endpoint, body):
        if self.cpu_embed and endpoint == "/api/embed":
            body = {**body, "options": {**body.get("options", {}), "num_gpu": 0}}
        return super()._post(endpoint, body)


def validate_placement(resident, settings, placement):
    def canonical(tag):
        return tag if ":" in tag else tag + ":latest"

    by_name = {canonical(m["name"]): m for m in resident}
    generator = by_name.get(canonical(settings.gen_model))
    embedder = by_name.get(canonical(settings.embed_model))
    if generator is None or embedder is None:
        raise RuntimeError("Both selected models must remain resident")
    if generator.get("context_length") != settings.num_ctx:
        raise RuntimeError("Generator context differs from requested context")
    if placement == "gpu":
        if any(m["size_vram"] < m["size"] for m in [generator, embedder]):
            raise RuntimeError("Both models must remain fully GPU-resident")
    else:
        if embedder["size_vram"] != 0:
            raise RuntimeError("Embedding model was not placed entirely on CPU")
        if placement == "cpu-embed" and generator["size_vram"] != generator["size"]:
            raise RuntimeError("Generator spilled to CPU during CPU-embedding test")
        if placement == "hybrid" and not 0 < generator["size_vram"] < generator["size"]:
            raise RuntimeError("Generator was not split across CPU and GPU")


def host_memory(process):
    family = [process, *process.children(recursive=True)]
    pss = rss = swap = 0
    for child in family:
        try:
            info = child.memory_full_info()
        except psutil.NoSuchProcess:
            continue
        pss += info.pss
        rss += info.rss
        swap += info.swap
    return {
        "ollama_pss_bytes": pss,
        "ollama_rss_bytes": rss,
        "ollama_swap_bytes": swap,
        "system_available_bytes": psutil.virtual_memory().available,
        "system_swap_used_bytes": psutil.swap_memory().used,
    }


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def unload(api):
    for model in api.get("/api/ps").json()["models"]:
        api.post("/api/generate", json={"model": model["name"], "keep_alive": 0}).raise_for_status()
    deadline = time.monotonic() + 30
    while api.get("/api/ps").json()["models"]:
        if time.monotonic() > deadline:
            raise TimeoutError("Models did not unload")
        time.sleep(0.2)
    time.sleep(1)


def run_case(key, kv, settings, out, model_dir, port, *, placement="gpu", host_ram_limit_gb=None):
    model, tokenizer = CANDIDATES[key]
    label = f"{key}-{kv}" + (f"-{placement}" if placement != "gpu" else "")
    folder = out / label
    folder.mkdir(parents=True, exist_ok=True)
    state = Path(".cfl/compare-14b") / (label + "-" + uuid.uuid4().hex[:8])
    state.mkdir(parents=True, exist_ok=True)
    config = settings.model_copy(
        update={
            "gen_model": model,
            "ollama_url": f"http://127.0.0.1:{port}",
            "state_dir": str(state.resolve()),
            "kv_quantization": kv == "q8_0",
            "tokenizer_file": str(Path(".cfl/tokenizers", tokenizer, "tokenizer.json").resolve()),
        }
    )
    if not Path(config.tokenizer_file).is_file():
        raise FileNotFoundError(config.tokenizer_file)
    report = {
        "model": model,
        "kv_type": kv,
        "num_ctx": config.num_ctx,
        "temperature": config.temperature,
        "seed": config.seed,
        "thinking": not config.gen_disable_thinking,
        "application_cache": "cold",
        "policy_limit_gb": config.vram_limit_gb,
        "rows": [],
        "placement": placement,
        "host_ram_limit_gb": host_ram_limit_gb,
        "host_ram_limit_method": "sampled Ollama-family PSS; terminate test server on excess",
    }
    report["tokenizer_sha256"] = hashlib.sha256(
        Path(config.tokenizer_file).read_bytes()
    ).hexdigest()
    provenance = Path(config.tokenizer_file).with_name("provenance.json")
    if provenance.exists():
        report["tokenizer_provenance"] = json.loads(provenance.read_text())
    samples, stop = [], threading.Event()
    process = None
    memory_exceeded = threading.Event()

    def monitor():
        while not stop.is_set():
            sample = {"time": time.time(), "gpu_mib": gpu_mib()}
            if process is not None and process.poll() is None:
                try:
                    sample.update(host_memory(psutil.Process(process.pid)))
                except psutil.NoSuchProcess:
                    pass
                if (
                    host_ram_limit_gb
                    and sample.get("ollama_pss_bytes", 0) > host_ram_limit_gb * 1e9
                ):
                    memory_exceeded.set()
                    os.killpg(process.pid, signal.SIGTERM)
            samples.append(sample)
            stop.wait(0.5)

    environment = {
        **os.environ,
        **ollama_environment(config),
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_MODELS": str(model_dir),
        "OLLAMA_NO_CLOUD": "1",
    }
    name = "cfl_cmp_" + uuid.uuid4().hex[:12]
    base = settings.dsn.rsplit("/", 1)[0]
    created = False
    process = None
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with (folder / "server.log").open("w") as log:
            process = subprocess.Popen(
                ["ollama", "serve"], env=environment, stdout=log, stderr=log, start_new_session=True
            )
            wait_ready(config.ollama_url, process)
            with PlacementClient(config, cpu_embed=placement != "gpu") as client:
                report["ollama_version"] = client.version()
                report["model_identity"] = next(m for m in client.tags() if m["name"] == model)
                report["model_details"] = client.show(model)
                started = time.monotonic()
                report["benchmark"] = asdict(warm_and_benchmark(client, config))
                report["benchmark_wall_s"] = time.monotonic() - started
                resident = client.ps()
                report["resident_before"] = resident
                validate_placement(resident, config, placement)
                report["model_vram_bytes"] = sum(m["size_vram"] for m in resident)
                try:
                    check_residency(client, config)
                    report["app_preflight_pass"] = True
                except Exception as exc:
                    report["app_preflight_pass"] = False
                    report["app_preflight_error"] = str(exc)
                    if placement == "gpu" and "exceeds limit" not in str(exc):
                        raise
                if (
                    report["benchmark"]["truncation_warning"]
                    or not report["benchmark"]["prefill_tps"]
                ):
                    raise RuntimeError("Benchmark did not verify context/throughput")
                with psycopg.connect(base + "/postgres", autocommit=True) as admin:
                    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
                created = True
                config = config.model_copy(update={"dsn": base + "/" + name})
                with db.connect(config.dsn) as conn:
                    db.run_migrations(conn, config.embed_dim, config.migrations_dir)
                    run_stage1(conn, config, str(Path("eval/sample_repo").resolve()))
                    run_stage2(conn, config, str(Path("eval/sample_repo").resolve()))
                    build_lexical(conn)
                    started = time.monotonic()
                    result = run_symbol_pass(
                        conn,
                        client,
                        config,
                        resume=False,
                        progress=lambda r, total: print(
                            label, "summaries", r.saved, "/", total, flush=True
                        ),
                    )
                    report["summary_pass"] = asdict(result)
                    report["summary_wall_s"] = time.monotonic() - started
                    save(folder / "summaries.json", db.ctx_inputs(conn))
                    build_lexical(conn)
                    started = time.monotonic()
                    report["dense"] = build_dense(conn, client, config)
                    report["dense_wall_s"] = time.monotonic() - started
                    eval_config = load_config("eval/configs/lexical.toml")
                    questions = load_questions(eval_config["questions"])
                    single = state / "question.yaml"
                    for i, question in enumerate(questions, 1):
                        single.write_text(yaml.safe_dump({"version": 1, "questions": [question]}))
                        started = time.monotonic()
                        result = run_eval(
                            conn,
                            {**eval_config, "questions": str(single.resolve())},
                            {"full", "retrieval", "structural"},
                            client=client,
                            settings=config,
                        )
                        row = result["rows"][0]
                        row["wall_s"] = time.monotonic() - started
                        report["rows"].append(row)
                        save(folder / "report.json", report)
                        print(
                            label,
                            f"{i}/48",
                            question["id"],
                            "sufficient=",
                            row["metrics"]["answer_sufficient"],
                            round(row["wall_s"], 2),
                            flush=True,
                        )
                    report["summary"] = summarize(report["rows"])
                    report["answer_latency_mean_s"] = statistics.mean(
                        r["wall_s"] for r in report["rows"]
                    )
                    report["answer_latency_median_s"] = statistics.median(
                        r["wall_s"] for r in report["rows"]
                    )
                    report["sufficient_answers"] = sum(
                        r["metrics"]["answer_sufficient"] for r in report["rows"]
                    )
                    report["cached_answers"] = sum(
                        r["metrics"]["answer_cached"] for r in report["rows"]
                    )
                    report["resident_after"] = client.ps()
                    validate_placement(report["resident_after"], config, placement)
                    if memory_exceeded.is_set():
                        raise RuntimeError("Sampled CPU memory budget exceeded")
                    report["completed"] = True
    except Exception as exc:  # noqa: BLE001 - retain failed candidate evidence
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(label, report["error"], flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stop.set()
        thread.join()
        if created:
            with psycopg.connect(base + "/postgres", autocommit=True) as admin:
                admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        report["isolated_database_removed"] = created
        report["cpu_memory_budget_exceeded"] = memory_exceeded.is_set()
        report["peak_ollama_pss_gb"] = (
            max((s.get("ollama_pss_bytes", 0) for s in samples), default=0) / 1e9
        )
        report["peak_ollama_rss_gb"] = (
            max((s.get("ollama_rss_bytes", 0) for s in samples), default=0) / 1e9
        )
        report["peak_ollama_swap_gb"] = (
            max((s.get("ollama_swap_bytes", 0) for s in samples), default=0) / 1e9
        )
        report["peak_gpu_mib"] = max((s["gpu_mib"] for s in samples), default=0)
        report["peak_gpu_gb"] = report["peak_gpu_mib"] * 1024**2 / 1e9
        report["device_within_10gb"] = report["peak_gpu_gb"] < config.vram_limit_gb
        save(folder / "gpu.json", samples)
        save(folder / "report.json", report)
        logs = list(state.glob("logs/calls-*.jsonl"))
        (folder / "calls.jsonl").write_text("".join(p.read_text() for p in logs))
        print(
            label,
            "COMPLETE",
            report.get("completed", False),
            "peak GB",
            report["peak_gpu_gb"],
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        choices=[f"{key}-{kv}" for key in CANDIDATES for kv in ["f16", "q8_0"]],
    )
    parser.add_argument("--models", type=Path, default=Path("/usr/share/ollama/.ollama/models"))
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--out", type=Path, default=Path("docs/validation/14b-comparison"))
    parser.add_argument("--placement", choices=["gpu", "cpu-embed", "hybrid"], default="gpu")
    parser.add_argument("--host-ram-limit-gb", type=float)
    args = parser.parse_args()
    if args.host_ram_limit_gb is not None and args.host_ram_limit_gb <= 0:
        parser.error("CPU memory budget must be positive")
    try:
        httpx.get(f"http://127.0.0.1:{args.port}/api/version", timeout=1).raise_for_status()
    except httpx.HTTPError:
        pass
    else:
        parser.error("Isolated test port is already in use")
    settings = load_settings()
    args.out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=settings.ollama_url, timeout=600) as api:
        original = api.get("/api/ps").json()["models"]
        save(args.out / "original-resident.json", original)
        try:
            unload(api)
            for case in args.cases:
                key, kv = case.split("-", 1)
                run_case(
                    key,
                    kv,
                    settings,
                    args.out,
                    args.models.resolve(),
                    args.port,
                    placement=args.placement,
                    host_ram_limit_gb=args.host_ram_limit_gb,
                )
        finally:
            for m in original:
                body = {
                    "model": m["name"],
                    "keep_alive": -1,
                    "stream": False,
                    "options": {"num_ctx": m.get("context_length", settings.num_ctx)},
                }
                if "bert" in m.get("details", {}).get("family", ""):
                    body["input"] = "restore"
                    response = api.post("/api/embed", json=body)
                else:
                    response = api.post("/api/generate", json=body)
                response.raise_for_status()
            restored = api.get("/api/ps").json()["models"]
            save(args.out / "restored-resident.json", restored)
            if {m["name"]: m.get("context_length") for m in original} != {
                m["name"]: m.get("context_length") for m in restored
            }:
                raise RuntimeError("Original residency was not restored")
            print("Original model residency restored", flush=True)


if __name__ == "__main__":
    main()
