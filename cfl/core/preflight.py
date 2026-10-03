from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from cfl.config import Settings
    from cfl.core.client import OllamaClient

from cfl.core.errors import PreflightError

logger = logging.getLogger(__name__)


@dataclass
class Bench:
    """Results of the warm-and-benchmark run."""

    prefill_tps: float
    gen_tps: float
    embed_dim: int
    truncation_warning: bool
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_ollama(client: OllamaClient) -> None:
    """Raise PreflightError if Ollama is not reachable."""
    try:
        client.version()
    except Exception as exc:
        raise PreflightError(f"Cannot reach Ollama: {exc}") from exc


def check_models_pulled(client: OllamaClient, settings: Settings) -> None:
    """Raise PreflightError listing any models that have not been pulled."""
    try:
        available = {m["name"] for m in client.tags()}
    except Exception as exc:
        raise PreflightError(f"Cannot list Ollama models: {exc}") from exc

    def canonical(tag: str) -> str:
        return tag if ":" in tag else tag + ":latest"

    available = {canonical(tag) for tag in available}
    missing = [
        model
        for model in (settings.gen_model, settings.embed_model)
        if canonical(model) not in available
    ]

    if missing:
        raise PreflightError(f"Models not pulled: {missing}. Run scripts/pull_models.sh")


def warm_and_benchmark(client: OllamaClient, settings: Settings) -> Bench:
    """Load both models and measure throughput. Never raises; returns zeros on error."""
    try:
        from cfl.core.budget import cap_tool_output

        target_tokens = settings.num_ctx - settings.template_overhead - 512
        unit = "The quick brown fox jumps over the lazy dog. "
        padded_prompt = cap_tool_output(
            "Uncached benchmark " + uuid.uuid4().hex + "\n" + unit * settings.num_ctx,
            target_tokens,
            counter=client.counter,
            pointer="Benchmark padding omitted.",
        )
        result = client.generate(padded_prompt, "", num_predict=64, task="benchmark")
        prompt_eval_count = result.prompt_tokens
        uncached = result.uncached_prompt_tokens
        prompt_eval_duration = result.prompt_eval_duration
        eval_count = result.output_tokens
        eval_duration = result.eval_duration
        prefill_tps = (
            uncached / (prompt_eval_duration / 1e9)
            if prompt_eval_duration and uncached is not None and uncached >= 32
            else 0.0
        )
        gen_tps = eval_count / (eval_duration / 1e9) if eval_duration else 0.0
        truncation_warning = prompt_eval_count < 0.8 * client.counter.count(padded_prompt)
        embed_dim = len(client.embed(["hello world"])[0])

        bench = Bench(
            prefill_tps=round(prefill_tps, 1),
            gen_tps=round(gen_tps, 1),
            embed_dim=embed_dim,
            truncation_warning=truncation_warning,
            extra={
                "prompt_tokens": prompt_eval_count,
                "cached_prompt_tokens": result.cached_prompt_tokens,
                "uncached_prompt_tokens": uncached,
                "load_duration": result.load_duration,
                "cache_metrics_available": result.cached_prompt_tokens is not None,
            },
        )

        # Persist to .cfl/bench.json
        state_path = Path(settings.state_dir)
        state_path.mkdir(parents=True, exist_ok=True)
        (state_path / "bench.json").write_text(
            json.dumps(
                {
                    "prefill_tps": bench.prefill_tps,
                    "gen_tps": bench.gen_tps,
                    "embed_dim": bench.embed_dim,
                    "truncation_warning": bench.truncation_warning,
                    **bench.extra,
                }
            )
        )
        return bench

    except Exception as exc:  # noqa: BLE001
        logger.warning("warm_and_benchmark failed (non-fatal): %s", exc)
        return Bench(prefill_tps=0.0, gen_tps=0.0, embed_dim=0, truncation_warning=True)


def check_residency(client: OllamaClient, settings: Settings) -> list[str]:
    """Check that models are loaded on GPU; return a list of warning strings."""
    warnings: list[str] = []
    try:
        loaded = client.ps()
    except Exception as exc:  # noqa: BLE001
        return [f"Cannot check model residency: {exc}"]

    def canonical(tag: str) -> str:
        return tag if ":" in tag else tag + ":latest"

    by_name = {canonical(m["name"]): m for m in loaded}
    for role, tag in (("Generator", settings.gen_model), ("Embedder", settings.embed_model)):
        model = by_name.get(canonical(tag))
        if model is None:
            raise PreflightError(f"{role} model not resident: {tag}")
        if model.get("size_vram", 0) < model.get("size", 1):
            raise PreflightError(
                f"{role} model not fully on GPU (size_vram={model.get('size_vram')}, "
                f"size={model.get('size')}). CPU spill is ~10x slower — abort."
            )
        if role == "Generator" and model.get("context_length") != settings.num_ctx:
            raise PreflightError(
                f"Generator context {model.get('context_length')} differs from NUM_CTX={settings.num_ctx}"
            )

    total_vram = sum(m.get("size_vram", 0) for m in loaded)
    if total_vram >= settings.vram_limit_gb * 1e9:
        raise PreflightError(
            f"Total VRAM used {total_vram / 1e9:.2f} GB exceeds limit {settings.vram_limit_gb} GB"
        )

    return warnings


def check_db(settings: Settings) -> str:
    """Check DB connectivity and extension availability; return a status string."""
    from cfl.core import db  # local import to avoid circular dep at module level

    try:
        conn = db.connect(settings.dsn)
    except Exception as exc:  # noqa: BLE001
        return f"Cannot connect to DB: {exc}"

    try:
        exts, schema_version = db.database_diagnostics(conn)
        missing = [e for e in ("vector", "pg_trgm") if e not in exts]

        parts = ["Connected"]
        if missing:
            parts.append(f"missing extensions: {missing}")
        else:
            parts.append("extensions OK (vector, pg_trgm)")
        if schema_version:
            parts.append(f"schema_version={schema_version}")

        return "; ".join(parts)
    except Exception as exc:  # noqa: BLE001
        return f"DB query error: {exc}"
    finally:
        conn.close()


def check_ram() -> str | None:
    """Return a warning string if available RAM is below 4 GB, else None."""
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 1e9
        if available_gb < 4.0:
            return f"Low RAM: only {available_gb:.1f} GB available (recommended ≥ 4 GB)"
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("RAM check failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_doctor(settings: Settings) -> int:
    """Run all preflight checks and print a rich report. Returns 0 or 1."""
    from cfl.core.client import OllamaClient

    console = Console()
    exit_code = 0

    table = Table(title="CFL Doctor", show_header=True, header_style="bold cyan")
    table.add_column("Check", style="bold", min_width=24)
    table.add_column("Status")

    def ok(label: str, msg: str) -> None:
        table.add_row(label, f"[green]✓[/green] {msg}")

    def warn(label: str, msg: str) -> None:
        table.add_row(label, f"[yellow]⚠[/yellow] {msg}")

    def fail(label: str, msg: str) -> None:
        nonlocal exit_code
        exit_code = 1
        table.add_row(label, f"[red]✗[/red] {msg}")

    with OllamaClient(settings) as client:
        # 1. Ollama reachable
        try:
            ver = client.version()
            ok("Ollama", f"version {ver}")
        except PreflightError as exc:
            fail("Ollama", str(exc))
            console.print(table)
            return 1
        except Exception as exc:  # noqa: BLE001
            fail("Ollama", f"unreachable: {exc}")
            console.print(table)
            return 1

        # 2. Models pulled
        try:
            check_models_pulled(client, settings)
            ok("Models pulled", f"{settings.gen_model}, {settings.embed_model}")
        except PreflightError as exc:
            fail("Models pulled", str(exc))

        # 3. Benchmark (non-blocking)
        bench = warm_and_benchmark(client, settings)
        if bench.prefill_tps > 0:
            ok(
                "Throughput",
                f"prefill {bench.prefill_tps:.0f} t/s · gen {bench.gen_tps:.0f} t/s",
            )
            if bench.truncation_warning:
                warn(
                    "Token estimate",
                    "reported prompt count is below estimate; verify tokenizer and num_ctx",
                )
            ok("Embed dim", str(bench.embed_dim))
        else:
            warn("Throughput", "benchmark skipped (models not loaded or error)")

        # 4. VRAM residency
        try:
            vram_warnings = check_residency(client, settings)
            if vram_warnings:
                for w in vram_warnings:
                    warn("VRAM", w)
            else:
                ok("VRAM", "models resident")
        except PreflightError as exc:
            fail("VRAM", str(exc))

    # 5. Database
    db_status = check_db(settings)
    if "Cannot connect" in db_status or "error" in db_status.lower():
        warn("Database", db_status)
    else:
        ok("Database", db_status)

    # 6. RAM
    ram_warn = check_ram()
    if ram_warn:
        warn("RAM", ram_warn)
    else:
        ok("RAM", "sufficient")

    console.print(table)
    return exit_code
