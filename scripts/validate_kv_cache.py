from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

import httpx

from cfl.config import load_settings
from cfl.core.benchmark import cache_workloads
from cfl.core.client import OllamaClient
from cfl.core.preflight import warm_and_benchmark
from cfl.core.runtime import ollama_environment
from cfl.prompts.prompts import build_symbol_prompt
from cfl.prompts.schemas import SYMBOL_SUMMARY_JSON_SCHEMA, SymbolSummary


def gpu_mib() -> int:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return sum(int(line) for line in result.splitlines())


def wait_ready(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Isolated Ollama server failed; inspect its saved log")
        try:
            if httpx.get(url + "/api/version", timeout=1).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError("Isolated Ollama did not start")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare f16 and q8_0; restore existing model residency"
    )
    parser.add_argument("--models", required=True, help="Existing Ollama model directory")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--out", type=Path, default=Path(".cfl/kv-validation"))
    args = parser.parse_args()
    settings = load_settings()
    args.out.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    try:
        httpx.get(url + "/api/version", timeout=1)
    except httpx.HTTPError:
        pass
    else:
        raise RuntimeError("Test port already in use")
    report = {"cases": [], "restored": False}
    original = []
    identity_dir = Path.home() / ".ollama"
    identity_existed = identity_dir.exists()
    with httpx.Client(base_url=settings.ollama_url, timeout=180) as main_client:
        response = main_client.get("/api/ps")
        response.raise_for_status()
        original = response.json()["models"]
        idle_ceiling = (
            gpu_mib() - sum(model["size_vram"] for model in original) // (1024 * 1024) + 256
        )
        try:
            for model in original:
                response = main_client.post(
                    "/api/generate", json={"model": model["name"], "keep_alive": 0}
                )
                response.raise_for_status()
            deadline = time.monotonic() + 30
            while main_client.get("/api/ps").json()["models"] or gpu_mib() > idle_ceiling:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Previous GPU allocations did not release; avoid overlapping runs"
                    )
                time.sleep(0.2)
            for quantized in [False, True]:
                config = settings.model_copy(update={"kv_quantization": quantized})
                environment = {
                    **os.environ,
                    **ollama_environment(config),
                    "OLLAMA_HOST": f"127.0.0.1:{args.port}",
                    "OLLAMA_MODELS": str(Path(args.models).resolve()),
                    "OLLAMA_NO_CLOUD": "1",
                }
                log_path = args.out / ("q8_0.log" if quantized else "f16.log")
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        ["ollama", "serve"],
                        env=environment,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    try:
                        wait_ready(url, process)
                        for context in args.contexts:
                            config = config.model_copy(
                                update={
                                    "ollama_url": url,
                                    "num_ctx": context,
                                    "state_dir": str(args.out / f"{quantized}-{context}"),
                                    "tokenizer_file": str(args.tokenizer.resolve())
                                    if args.tokenizer
                                    else settings.tokenizer_file,
                                }
                            )
                            readings = [gpu_mib()]
                            stop = threading.Event()

                            def monitor(stop=stop, readings=readings):
                                while not stop.wait(0.2):
                                    readings.append(gpu_mib())

                            thread = threading.Thread(target=monitor)
                            thread.start()
                            try:
                                with OllamaClient(config) as client:
                                    cold = warm_and_benchmark(client, config)
                                    bench = warm_and_benchmark(client, config)
                                    if not bench.prefill_tps:
                                        raise RuntimeError("Live uncached benchmark failed")
                                    workloads = cache_workloads(
                                        client, repetitions=args.repetitions
                                    )
                                    quality = []
                                    if context == args.contexts[0]:
                                        for relative in [
                                            "db/repository.py",
                                            "utils/retry.py",
                                            "pipeline.py",
                                        ]:
                                            raw = (Path("eval/sample_repo") / relative).read_text()
                                            symbol = {
                                                "file_path": relative,
                                                "qualname": "module",
                                                "raw_code": raw,
                                            }
                                            system, prompt = build_symbol_prompt(symbol, [])
                                            result = client.generate(
                                                prompt,
                                                system,
                                                fmt=SYMBOL_SUMMARY_JSON_SCHEMA,
                                                num_predict=350,
                                                task="kv_quality",
                                            )
                                            summary = SymbolSummary.model_validate_json(result.text)
                                            quality.append(
                                                {
                                                    "source": relative,
                                                    "summary": summary.model_dump(),
                                                    "output_tokens": result.output_tokens,
                                                    "gen_tps": result.output_tokens
                                                    / (result.eval_duration / 1e9),
                                                }
                                            )
                                    resident = client.ps()
                                    if any(
                                        model["size_vram"] < model["size"] for model in resident
                                    ):
                                        raise RuntimeError(
                                            "Partial CPU offload during KV validation"
                                        )
                                    if (
                                        sum(model["size_vram"] for model in resident)
                                        > config.vram_limit_gb * 1e9
                                    ):
                                        raise RuntimeError("Model VRAM budget exceeded")
                                readings.append(gpu_mib())
                            finally:
                                stop.set()
                                thread.join()
                            case = {
                                "kv_type": environment["OLLAMA_KV_CACHE_TYPE"],
                                "num_ctx": context,
                                "peak_gpu_mib": max(readings),
                                "peak_within_budget": max(readings) * 1024**2
                                <= config.vram_limit_gb * 1e9,
                                "resident": resident,
                                "model_vram_bytes": sum(model["size_vram"] for model in resident),
                                "prefill_tps": bench.prefill_tps,
                                "gen_tps": bench.gen_tps,
                                "benchmark_metrics": bench.extra,
                                "cold_load_duration": cold.extra.get("load_duration"),
                                "workloads": workloads,
                                "quality": quality,
                                "median_latency_s": {
                                    name: statistics.median(
                                        row["latency_s"] for row in workloads if row["case"] == name
                                    )
                                    for name in [
                                        "uncached",
                                        "repeat",
                                        "shared_prefix",
                                        "changed_prefix",
                                    ]
                                },
                            }
                            report["cases"].append(case)
                            (args.out / "report.json").write_text(
                                json.dumps(report, indent=2) + "\n"
                            )
                            print(
                                json.dumps(
                                    {
                                        key: value
                                        for key, value in case.items()
                                        if key not in {"workloads", "quality", "resident"}
                                    }
                                ),
                                flush=True,
                            )
                    finally:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
        finally:
            # Rehydrate only models that were resident before the test, at their original contexts.
            for model in original:
                body = {
                    "model": model["name"],
                    "keep_alive": -1,
                    "stream": False,
                    "options": {"num_ctx": model.get("context_length", settings.num_ctx)},
                }
                if "bert" in model.get("details", {}).get("family", ""):
                    body["input"] = "restore"
                    response = main_client.post("/api/embed", json=body)
                else:
                    response = main_client.post("/api/generate", json=body)
                response.raise_for_status()
            restored = main_client.get("/api/ps")
            restored.raise_for_status()
            report["restored"] = {
                row["name"]: row.get("context_length") for row in restored.json()["models"]
            } == {row["name"]: row.get("context_length") for row in original}
            report["original_resident"] = original
            report["restored_resident"] = restored.json()["models"]
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if not identity_existed:
                for name in ["id_ed25519", "id_ed25519.pub"]:
                    (identity_dir / name).unlink(missing_ok=True)
                try:
                    identity_dir.rmdir()
                except OSError:
                    pass
    if not report["restored"]:
        raise RuntimeError("Original residency was not restored")


if __name__ == "__main__":
    main()
