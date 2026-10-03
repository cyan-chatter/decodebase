from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
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

    missing = []
    for model in (settings.gen_model, settings.embed_model):
        # Ollama tags may include a digest suffix; match on prefix.
        if not any(t == model or t.startswith(model + ":") for t in available):
            # Also accept the base name without a tag.
            base = model.split(":")[0]
            if not any(t == base or t.startswith(base + ":") for t in available):
                missing.append(model)

    if missing:
        raise PreflightError(f"Models not pulled: {missing}. Run scripts/pull_models.sh")


def warm_and_benchmark(client: OllamaClient, settings: Settings) -> Bench:
    """Load both models and measure throughput. Never raises; returns zeros on error."""
    try:
        estimated_tokens = settings.num_ctx - 512
        target_chars = int(estimated_tokens * settings.chars_per_token)
        unit = "The quick brown fox jumps over the lazy dog. "
        repeats = max(1, target_chars // len(unit) + 1)
        padded_prompt = (unit * repeats)[:target_chars]

        resp = client._client.post(
            "/api/generate",
            json={
                "model": settings.gen_model,
                "prompt": padded_prompt,
                "options": {
                    "num_ctx": settings.num_ctx,
                    "num_predict": 64,
                },
                "stream": False,
                "keep_alive": -1,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        prompt_eval_count: int = data.get("prompt_eval_count", 0)
        prompt_eval_duration: float = data.get("prompt_eval_duration", 1)
        eval_count: int = data.get("eval_count", 0)
        eval_duration: float = data.get("eval_duration", 1)

        prefill_tps = prompt_eval_count / (prompt_eval_duration / 1e9) if prompt_eval_duration else 0.0
        gen_tps = eval_count / (eval_duration / 1e9) if eval_duration else 0.0

        estimated_tokens_sent = len(padded_prompt) / settings.chars_per_token
        truncation_warning = prompt_eval_count < 0.8 * estimated_tokens_sent

        # Embedder warm-up
        embed_resp = client._client.post(
            "/api/embed",
            json={"model": settings.embed_model, "input": ["hello world"]},
        )
        embed_resp.raise_for_status()
        embed_data = embed_resp.json()
        embed_dim = len(embed_data["embeddings"][0])

        bench = Bench(
            prefill_tps=round(prefill_tps, 1),
            gen_tps=round(gen_tps, 1),
            embed_dim=embed_dim,
            truncation_warning=truncation_warning,
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
    except Exception as exc:
        return [f"Cannot check model residency: {exc}"]

    if not loaded:
        return ["Models not loaded yet — run cfl doctor after warming models"]

    by_name: dict[str, dict] = {m["name"]: m for m in loaded}

    # Generator check
    gen = by_name.get(settings.gen_model) or by_name.get(settings.gen_model.split(":")[0])
    if gen:
        if gen.get("size_vram", 0) < gen.get("size", 1):
            raise PreflightError(
                f"Generator model not fully on GPU (size_vram={gen.get('size_vram')}, "
                f"size={gen.get('size')}). CPU spill is ~10x slower — abort."
            )

    # Embedder check (warn only)
    emb = by_name.get(settings.embed_model) or by_name.get(settings.embed_model.split(":")[0])
    if emb and emb.get("size_vram", 0) < emb.get("size", 1):
        warnings.append(
            f"Embedder model not fully on GPU "
            f"(size_vram={emb.get('size_vram')}, size={emb.get('size')})"
        )

    # Total VRAM check
    total_vram = sum(m.get("size_vram", 0) for m in loaded)
    limit_bytes = settings.vram_limit_gb * 1e9
    if total_vram > limit_bytes:
        warnings.append(
            f"Total VRAM used {total_vram / 1e9:.1f} GB exceeds limit {settings.vram_limit_gb} GB"
        )

    return warnings


def check_db(settings: Settings) -> str:
    """Check DB connectivity and extension availability; return a status string."""
    from cfl.core import db  # local import to avoid circular dep at module level

    try:
        conn = db.connect(settings.dsn)
    except Exception as exc:
        return f"Cannot connect to DB: {exc}"

    try:
        exts = {
            row[0]
            for row in conn.execute(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')"
            ).fetchall()
        }
        missing = [e for e in ("vector", "pg_trgm") if e not in exts]

        schema_version: str | None = None
        meta_exists = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'meta' LIMIT 1"
        ).fetchone()
        if meta_exists:
            row = conn.execute("SELECT schema_version FROM meta LIMIT 1").fetchone()
            if row:
                schema_version = str(row[0])

        parts = ["Connected"]
        if missing:
            parts.append(f"missing extensions: {missing}")
        else:
            parts.append("extensions OK (vector, pg_trgm)")
        if schema_version:
            parts.append(f"schema_version={schema_version}")

        return "; ".join(parts)
    except Exception as exc:
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
        except Exception as exc:
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
                warn("Truncation", "prompt was truncated — check num_ctx setting")
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
