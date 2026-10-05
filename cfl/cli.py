from __future__ import annotations

import json
import sys
from dataclasses import asdict

import click
import psycopg
import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from cfl.config import get_settings, load_settings
from cfl.core.errors import CflError

app = typer.Typer(no_args_is_help=True, help="CodeFlowLens – local code intelligence")
console = Console(stderr=True)


@app.callback()
def _callback() -> None:
    """CodeFlowLens top-level callback — catches CflError and exits cleanly."""


@app.command("runtime-env")
def runtime_env(
    kv_quantization: bool | None = typer.Option(
        None, "--kv-quantization/--no-kv-quantization", help="Opt into KV cache quantization"
    ),
    kv_type: str | None = typer.Option(None, "--kv-type", help="q8_0 or q4_0"),
    output_format: str = typer.Option("shell", "--format", help="shell or systemd"),
) -> None:
    """Print Ollama server settings. Restart Ollama to apply them."""
    from cfl.config import Settings
    from cfl.core.runtime import ollama_environment

    values = get_settings().model_dump()
    if kv_quantization is not None:
        values["kv_quantization"] = kv_quantization
    if kv_type is not None:
        values["kv_quantization_type"] = kv_type
    try:
        settings = Settings(**values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if output_format not in {"shell", "systemd"}:
        raise typer.BadParameter("format must be shell or systemd")
    if output_format == "systemd":
        typer.echo("[Service]")
    for name, value in ollama_environment(settings).items():
        typer.echo(
            f'Environment="{name}={value}"'
            if output_format == "systemd"
            else f'export {name}="{value}"'
        )


@app.command("cache-benchmark")
def cache_benchmark(
    repetitions: int = typer.Option(3, min=1, max=20),
    out: str = typer.Option(".cfl/cache-benchmark.json", "--out"),
) -> None:
    """Measure fresh, repeated and shared-prefix prompts separately."""
    from pathlib import Path

    import httpx

    from cfl.core.benchmark import cache_workloads
    from cfl.core.client import OllamaClient

    settings = get_settings()
    try:
        with OllamaClient(settings) as client:
            tags = client.tags()
            rows = cache_workloads(client, repetitions=repetitions)
        report = {
            "gen_model": settings.gen_model,
            "num_ctx": settings.num_ctx,
            "tags": tags,
            "rows": rows,
        }
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        typer.echo(json.dumps(report))
    except (CflError, httpx.HTTPError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Run preflight checks: Ollama, models, VRAM, DB, RAM."""
    from cfl.core.preflight import run_doctor

    try:
        code = run_doctor(get_settings())
        raise typer.Exit(code)
    except CflError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# Ingestion / index
# ---------------------------------------------------------------------------


@app.command()
def build(
    repo: str = typer.Argument(".", help="Repository to index"),
    estimate: bool = typer.Option(False, "--estimate"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_tests: bool = typer.Option(False, "--skip-tests"),
    min_lines: int = typer.Option(0, "--min-lines", min=0),
    priority: str = typer.Option("default", "--priority"),
    budget_hours: float = typer.Option(3, "--budget-hours", min=0.001),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    force: bool = typer.Option(False, "--force", "-f"),
    workers: int = typer.Option(1, "--workers", "-w"),
) -> None:
    """Generate drafts, independently validate knowledge, then publish RAG views."""
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from cfl.pipeline.build import run_build

    if priority not in {"default", "entrypoints-first"} or workers != 1:
        raise typer.BadParameter(
            "priority must be default/entrypoints-first; one GPU worker is supported"
        )
    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=estimate,
        ) as bar:
            task = bar.add_task("Summarizing", total=None)

            def progress(result, count):
                tps = result.tokens / (result.eval_ns / 1e9) if result.eval_ns else 0
                bar.update(
                    task,
                    completed=result.saved + result.cached + result.failed,
                    total=count,
                    description=f"Saved {result.saved}; cached {result.cached}; failed {result.failed}; {tps:.1f} t/s",
                )

            result = run_build(
                repo,
                estimate_only=estimate,
                resume=resume,
                skip_tests=skip_tests,
                min_lines=min_lines,
                priority=priority,
                budget_hours=budget_hours,
                retry_failed=retry_failed,
                force=force,
                settings=get_settings(),
                progress=progress,
            )
        console.print(
            f"Estimated summary generation: {result['estimate']['hours']:.3f} hours; "
            f"{result['estimate']['llm_symbols']} LLM symbols."
        )
        if result["over_budget"]:
            console.print(
                "Above budget. Consider --skip-tests, --min-lines N, --priority entrypoints-first."
            )
        if not estimate:
            console.print(json.dumps(result, default=str))
            if result["symbols"]["interrupted"]:
                raise typer.Exit(130)
            if result["symbols"]["failed"] or result.get("dense", {}).get("failed"):
                raise typer.Exit(1)
    except (CflError, psycopg.Error, OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc


@app.command(hidden=True)
def scan(
    repo: str = typer.Argument(".", help="Path to the repository to scan"),
) -> None:
    """Scan a repository: parse files (Stage 1) and resolve call graph (Stage 2)."""
    from cfl.core.db import connect
    from cfl.pipeline.scan import run_stage1, run_stage2

    settings = get_settings()
    conn = connect(settings.dsn)
    try:
        result = run_stage1(conn, settings, repo)
        if result is not None:
            console.print(
                f"Parsed {result.parsed} files; {result.failed} parse failures; "
                f"{result.unsupported} unsupported files."
            )
        run_stage2(conn, settings, repo)
        from cfl.pipeline.indexer import build_lexical

        indexed = build_lexical(conn)
        console.print(f"Lexical index built for {indexed} symbols.")
        console.print("[green]Scan complete.[/green]")
    finally:
        conn.close()


@app.command()
def status(
    repo: str = typer.Argument("."),
    failed: bool = typer.Option(False, "--failed"),
    min_lines: int = typer.Option(0, "--min-lines", min=0),
    skip_tests: bool = typer.Option(False, "--skip-tests"),
) -> None:
    """Show summary freshness, failures, ETA and independent view statuses."""
    from cfl.core import db
    from cfl.core.client import OllamaClient
    from cfl.pipeline.symbol_pass import symbol_status

    settings = get_settings()
    try:
        with db.connect(settings.dsn) as conn, OllamaClient(settings) as client:
            result = symbol_status(
                conn, client, settings, min_lines=min_lines, skip_tests=skip_tests
            )
            if not failed:
                result.pop("failed")
            typer.echo(json.dumps(result, default=str, ensure_ascii=False))
    except (CflError, psycopg.Error, OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# Symbol / code queries
# ---------------------------------------------------------------------------


def _run_answer(operation, *, repo=".", out=None, json_output=False):
    from pathlib import Path

    from cfl.core import db
    from cfl.core.client import OllamaClient
    from cfl.core.errors import AmbiguousSymbol

    settings = get_settings() if repo == "." else load_settings(repo)
    try:
        with (
            db.connect(settings.dsn, statement_timeout_ms=settings.statement_timeout_ms) as conn,
            OllamaClient(settings) as client,
        ):
            result = operation(conn, client, settings)
        text = (
            json.dumps(result, default=str, ensure_ascii=False) if json_output else result["text"]
        )
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(text + "\n", encoding="utf-8")
        else:
            typer.echo(text)
        if not json_output:
            for warning in result.get("warnings", []):
                console.print("[yellow]" + escape(warning) + "[/yellow]")
            if result.get("location"):
                console.print("Source: " + escape(result["location"]))
            for direction in ("callers", "callees"):
                if direction in result:
                    console.print(
                        direction
                        + ": "
                        + ", ".join(
                            escape(e.get("qualname") or e["callee_expr"]) for e in result[direction]
                        )
                    )
    except AmbiguousSymbol as exc:
        console.print("Ambiguous symbol; use path::name:")
        for candidate in exc.candidates:
            console.print(escape(candidate.id))
        raise typer.Exit(2) from exc
    except (CflError, psycopg.Error, OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc


@app.command()
def explain(
    symbol: str = typer.Argument(...),
    detailed: bool = typer.Option(False, "--detailed", "-d"),
    repo: str = typer.Option(".", "--repo"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Explain a symbol with source review and explicit incomplete/abstention status."""
    from cfl.engines.explain import explain as explain_symbol

    def operation(conn, client, settings):
        selected = _resolve_query_symbol(conn, symbol, json_output)
        return explain_symbol(
            conn, client, settings, selected.id, "detailed" if detailed else "brief"
        )

    _run_answer(operation, repo=repo, json_output=json_output)


@app.command()
def flow(
    entrypoint: str = typer.Argument(...),
    depth: int = typer.Option(4, "--depth", "-d", min=0, max=64),
    repo: str = typer.Option(".", "--repo"),
    out: str | None = typer.Option(None, "--out"),
    min_conf: float = typer.Option(0.6, "--min-conf", min=0, max=1),
    sequence: bool = typer.Option(False, "--sequence"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Render a source-ordered static call trace and verified narrative."""
    from cfl.engines.flow import flow as trace_flow

    def operation(conn, client, settings):
        selected = _resolve_query_symbol(conn, entrypoint, json_output)
        return trace_flow(
            conn,
            client,
            settings,
            selected.id,
            depth=depth,
            min_conf=min_conf,
            diagram_type="sequence" if sequence else "flowchart",
        )

    _run_answer(operation, repo=repo, out=out, json_output=json_output)


@app.command()
def feature(
    name: str = typer.Argument(..., help="Feature or topic to trace"),
    depth: int = typer.Option(4, "--depth", "-d", help="Expansion depth"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Explain how a feature is implemented across files."""
    console.print("[yellow]not implemented yet[/yellow]")


# ---------------------------------------------------------------------------
# Free-form Q&A
# ---------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(...),
    repo: str = typer.Option(".", "--repo"),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1),
    brief: bool = typer.Option(False, "--brief"),
    detailed: bool = typer.Option(False, "--detailed"),
    explain: bool = typer.Option(
        False, "--explain", help="Narrate structural evidence with the model"
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Answer with hybrid retrieval, source citations and a verified answer cache."""
    from cfl.engines.ask import answer

    if brief and detailed:
        raise typer.BadParameter("Choose --brief or --detailed")
    _run_answer(
        lambda conn, client, settings: answer(
            conn,
            client,
            settings.model_copy(update={"retrieval_top_k": top_k}),
            question,
            "brief" if brief else "detailed",
            explain_graph=explain,
        ),
        repo=repo,
        json_output=json_output,
    )


@app.command()
def chat(
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Start an interactive chat session about the codebase."""
    console.print("[yellow]not implemented yet[/yellow]")


# ---------------------------------------------------------------------------
# Graph queries: data comes from the engine; CLI owns selection and rendering.
# ---------------------------------------------------------------------------


def _resolve_query_symbol(conn, query, json_output):
    from cfl.core.errors import AmbiguousSymbol
    from cfl.engines.graph_queries import resolve_symbol

    try:
        return resolve_symbol(conn, query)
    except AmbiguousSymbol as exc:
        if json_output or not (sys.stdin.isatty() and console.is_terminal):
            raise
        table = Table(title=f"Choose a symbol for {query!r}")
        for heading in ("#", "Symbol", "Location"):
            table.add_column(heading)
        for index, candidate in enumerate(exc.candidates, 1):
            table.add_row(
                str(index),
                escape(candidate.qualname),
                escape(f"{candidate.file_path}:{candidate.start_line}"),
            )
        console.print(table)
        choice = typer.prompt("Symbol number", type=click.IntRange(1, len(exc.candidates)))
        return exc.candidates[choice - 1]


def _render_graph_result(command, result):
    if result is None:
        console.print("No call path found within the requested depth and confidence threshold.")
        return
    if command == "impact":
        console.print(f"Affected symbols: {result['total']}")
        if result["groups"]:
            groups = Table(title="Impact by depth and file")
            for heading in ("Depth", "File", "Count"):
                groups.add_column(heading)
            for group in result["groups"]:
                groups.add_row(str(group["depth"]), escape(group["file_path"]), str(group["count"]))
            console.print(groups)
        rows = [symbol for group in result["groups"] for symbol in group["symbols"]]
    else:
        rows = result
    if not rows:
        console.print("No candidates found." if command == "dead" else "No results.")
        return
    table = Table(title="Dead-code candidates" if command == "dead" else command.capitalize())
    if command == "hubs":
        columns = [
            ("Symbol", "qualname"),
            ("File", "file_path"),
            ("PageRank", "pagerank"),
            ("Fan-in", "fan_in"),
            ("Fan-out", "fan_out"),
        ]
    elif command in {"where", "dead"}:
        columns = [("Symbol", "qualname"), ("Location", "location"), ("Kind", "kind")]
    else:
        columns = [
            ("Symbol", "symbol"),
            ("Call site", "location"),
            ("Depth", "depth"),
            ("Resolution", "resolution"),
            ("Confidence", "confidence"),
            ("Source", "source"),
        ]
    for title, _ in columns:
        table.add_column(title)
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if key in {"pagerank", "confidence"}:
                value = f"{value:.3f}"
            if key == "resolution" and (
                row["resolution"] == "ambiguous" or row["confidence"] < 0.6
            ):
                value = f"? {value}"
            values.append(escape(str(value)))
        style = (
            "yellow"
            if row.get("resolution") == "ambiguous" or row.get("confidence", 1) < 0.6
            else ""
        )
        table.add_row(*values, style=style)
    console.print(table)


def _run_query(command, operation, repo, json_output):
    from cfl.config import load_settings
    from cfl.core.db import connect
    from cfl.core.errors import AmbiguousSymbol

    conn = None
    try:
        settings = get_settings() if repo == "." else load_settings(repo)
        conn = connect(settings.dsn, statement_timeout_ms=settings.statement_timeout_ms)
        result = operation(conn, settings)
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False))
        else:
            _render_graph_result(command, result)
    except AmbiguousSymbol as exc:
        if json_output:
            typer.echo(
                json.dumps(
                    {"error": "Ambiguous symbol", "candidates": [asdict(c) for c in exc.candidates]}
                )
            )
        else:
            console.print("[yellow]Ambiguous symbol; use an exact ID or path::name:[/yellow]")
            for candidate in exc.candidates:
                console.print(escape(candidate.id))
        raise typer.Exit(2) from exc
    except (CflError, psycopg.Error, OSError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}))
        else:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    finally:
        if conn is not None:
            conn.close()


@app.command()
def callers(
    symbol: str = typer.Argument(..., help="Symbol or path::name"),
    depth: int = typer.Option(1, "--depth", "-d", min=1, help="Maximum reverse-call depth"),
    min_conf: float | None = typer.Option(None, "--min-conf", min=0.0, max=1.0),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """List callers with call-site evidence, confidence and source."""
    from cfl.engines import graph_queries as queries

    _run_query(
        "callers",
        lambda conn, settings: queries.callers(
            conn,
            _resolve_query_symbol(conn, symbol, json_output),
            depth,
            settings.edge_conf_threshold if min_conf is None else min_conf,
        ),
        repo,
        json_output,
    )


@app.command()
def callees(
    symbol: str = typer.Argument(..., help="Symbol or path::name"),
    depth: int = typer.Option(1, "--depth", "-d", min=1, help="Maximum forward-call depth"),
    min_conf: float | None = typer.Option(None, "--min-conf", min=0.0, max=1.0),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """List callees with call-site evidence, confidence and source."""
    from cfl.engines import graph_queries as queries

    _run_query(
        "callees",
        lambda conn, settings: queries.callees(
            conn,
            _resolve_query_symbol(conn, symbol, json_output),
            depth,
            settings.edge_conf_threshold if min_conf is None else min_conf,
        ),
        repo,
        json_output,
    )


@app.command()
def path(
    source: str = typer.Argument(..., help="Source symbol"),
    target: str = typer.Argument(..., help="Target symbol"),
    depth: int = typer.Option(8, "--depth", "-d", min=1, help="Maximum path length"),
    min_conf: float | None = typer.Option(None, "--min-conf", min=0.0, max=1.0),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """Find a shortest call path and show each call site."""
    from cfl.engines import graph_queries as queries

    _run_query(
        "path",
        lambda conn, settings: queries.path(
            conn,
            _resolve_query_symbol(conn, source, json_output),
            _resolve_query_symbol(conn, target, json_output),
            depth,
            settings.edge_conf_threshold if min_conf is None else min_conf,
        ),
        repo,
        json_output,
    )


@app.command()
def impact(
    symbol: str = typer.Argument(..., help="Symbol or path::name"),
    depth: int = typer.Option(4, "--depth", "-d", min=1, help="Maximum reverse-call depth"),
    min_conf: float | None = typer.Option(None, "--min-conf", min=0.0, max=1.0),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """Show distinct affected callers grouped by depth and file."""
    from cfl.engines import graph_queries as queries

    _run_query(
        "impact",
        lambda conn, settings: queries.impact(
            conn,
            _resolve_query_symbol(conn, symbol, json_output),
            depth,
            settings.edge_conf_threshold if min_conf is None else min_conf,
        ),
        repo,
        json_output,
    )


@app.command()
def hubs(
    top_n: int = typer.Option(20, "--top", "-n", min=1, help="Number of hubs"),
    min_conf: float | None = typer.Option(None, "--min-conf", min=0.0, max=1.0),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """Show PageRank hubs with distinct fan-in and fan-out counts."""
    from cfl.engines import graph_queries as queries

    _run_query(
        "hubs",
        lambda conn, settings: queries.hubs(
            conn, top_n, settings.edge_conf_threshold if min_conf is None else min_conf
        ),
        repo,
        json_output,
    )


@app.command()
def dead(
    min_conf: float | None = typer.Option(None, "--min-conf", min=0.0, max=1.0),
    include_tests: bool = typer.Option(False, "--include-tests", help="Include test symbols"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """Find uncalled candidates, excluding exports, entrypoints and overrides."""
    from cfl.engines import graph_queries as queries

    _run_query(
        "dead",
        lambda conn, settings: queries.dead(
            conn, settings.edge_conf_threshold if min_conf is None else min_conf, include_tests
        ),
        repo,
        json_output,
    )


@app.command()
def where(
    symbol: str = typer.Argument(..., help="Symbol or fuzzy name"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """Show matching definitions as path:start-end locations."""
    from cfl.engines import graph_queries as queries

    _run_query("where", lambda conn, settings: queries.where(conn, symbol), repo, json_output)


# ---------------------------------------------------------------------------
# Database maintenance
# ---------------------------------------------------------------------------

db_app = typer.Typer(no_args_is_help=True, help="Migrate the index or reset its dense view")
app.add_typer(db_app, name="db")


def _run_database(operation, repo):
    from cfl.config import load_settings
    from cfl.core.db import connect

    conn = None
    try:
        settings = get_settings() if repo == "." else load_settings(repo)
        conn = connect(settings.dsn, statement_timeout_ms=settings.statement_timeout_ms)
        message = operation(conn, settings)
        console.print(f"[green]{escape(message)}[/green]")
    except (CflError, psycopg.Error, OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    finally:
        if conn is not None:
            conn.close()


@db_app.command("migrate")
def db_migrate(
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """Apply database migrations with an embedding dimension guard."""
    from pathlib import Path

    from cfl.core.db import run_migrations

    def migrate(conn, settings):
        directory = Path(settings.migrations_dir)
        if not directory.is_absolute():
            directory = Path(repo) / directory
        run_migrations(conn, settings.embed_dim, str(directory))
        return "Database migrations complete."

    _run_database(migrate, repo)


@db_app.command("reset-embeddings")
def db_reset_embeddings(
    dim: int = typer.Option(..., "--dim", min=1, help="New vector dimension; clears embeddings"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
) -> None:
    """Clear embeddings and their hashes, preserving symbols, summaries and graph."""
    from cfl.core.db import reset_embeddings

    def reset(conn, settings):
        reset_embeddings(conn, dim)
        return f"Embeddings reset to dimension {dim}; update CFL_EMBED_DIM or cfl.toml to match."

    _run_database(reset, repo)


@app.command()
def trace(
    symbol: str = typer.Argument(..., help="Symbol to trace dynamically"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Trace dynamic execution (opt-in; requires test harness)."""
    console.print("[yellow]not implemented yet[/yellow]")


# ---------------------------------------------------------------------------
# Documentation export
# ---------------------------------------------------------------------------


@app.command()
def docs(
    output: str = typer.Option("wiki/", "--output", "-o", help="Output directory"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Export wiki documentation (per-module pages, ARCHITECTURE.md, feature pages)."""
    console.print("[yellow]not implemented yet[/yellow]")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@app.command()
def eval(
    full: bool = typer.Option(
        False, "--full", help="Evaluate verified generated answers and hybrid retrieval"
    ),
    suite: str = typer.Option("eval/", "--suite", "-s", help="Path to evaluation suite"),
    repo: str = typer.Option(".", "--repo", help="Repository configuration directory"),
    config: str | None = typer.Option(None, "--config", help="TOML evaluation config"),
    compare: tuple[str, str] | None = typer.Option(
        None, "--compare", help="Compare two stored run IDs"
    ),
    retrieval_only: bool = typer.Option(
        False, "--retrieval-only", help="Graph checks and lexical retrieval without LLM answers"
    ),
    structural_only: bool = typer.Option(
        False, "--structural-only", help="Only caller and traversal graph checks"
    ),
    verified_only: bool = typer.Option(
        False, "--verified-only", help="Only questions with reviewed expectations"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable results"),
) -> None:
    """Evaluate lexical/graph evidence or full verified answers."""
    from pathlib import Path

    from cfl.config import load_settings
    from cfl.core import db
    from cfl.eval.runner import compare_runs, load_config, run_eval

    conn = None
    try:
        if sum((retrieval_only, structural_only, full)) > 1:
            raise ValueError("Choose --retrieval-only or --structural-only")
        settings = get_settings() if repo == "." else load_settings(repo)
        conn = db.connect(settings.dsn, statement_timeout_ms=settings.statement_timeout_ms)
        if compare is not None:
            result = compare_runs(conn, *compare)
        else:
            options = (
                load_config(config)
                if config
                else {"questions": str(Path(suite) / "questions.yaml")}
            )
            options.setdefault("min_conf", settings.edge_conf_threshold)
            options["verified_only"] = verified_only or options.get("verified_only", False)
            if full:
                from cfl.core.client import OllamaClient

                with OllamaClient(settings) as client:
                    result = run_eval(
                        conn, options, ("full", "structural"), client=client, settings=settings
                    )
            else:
                result = run_eval(
                    conn,
                    options,
                    ("structural",) if structural_only else ("retrieval", "structural"),
                )
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False))
        elif compare is not None:
            table = Table(title=f"Eval comparison: {compare[0]} -> {compare[1]}")
            for heading in ("Question", "Metric", "Run A", "Run B", "Delta"):
                table.add_column(heading)
            for row in result["differences"]:
                table.add_row(
                    row["question_id"],
                    row["metric"],
                    f"{row['run_a']:.3f}",
                    f"{row['run_b']:.3f}",
                    f"{row['delta']:+.3f}",
                )
            console.print(table)
            if result["suite_changed"] or result["only_a"] or result["only_b"]:
                console.print(
                    "[yellow]Suite or selected questions changed; compare shared items with care.[/yellow]"
                )
        else:
            console.print(f"Eval run: {result['run_id']}")
            if result["provisional"]:
                console.print(
                    "[yellow]Provisional: expected values still require verification.[/yellow]"
                )
            if result.get("review_methods"):
                console.print("Expectation review: " + ", ".join(result["review_methods"]))
            console.print("AnswerRecall@5: lexical; EvidenceRecall@5: source with path evidence.")
            table = Table(
                title="Full-answer evaluation"
                if full
                else "Retrieval and graph evaluation (no LLM)"
            )
            headings = (
                (
                    "Type",
                    "Count",
                    "Retrieved @5",
                    "Cited @5",
                    "Citation validity",
                    "Tokens / answer",
                )
                if full
                else ("Type", "Count", "Lexical @5", "Evidence @5", "Structural", "Traversal")
            )
            for heading in headings:
                table.add_column(heading, no_wrap=True)
            for kind, row in result["summary"].items():

                def metric(name, row=row):
                    return f"{row[name]:.3f}" if name in row else "—"

                names = (
                    (
                        "retrieved_recall_at_5",
                        "cited_recall_at_5",
                        "citation_validity",
                        "tokens_per_answer",
                    )
                    if full
                    else (
                        "answer_recall_at_5",
                        "evidence_recall_at_5",
                        "structural_exactness",
                        "traversal_exactness",
                    )
                )
                table.add_row(kind, str(row["count"]), *(metric(name) for name in names))
            console.print(table)
    except (CflError, psycopg.Error, OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}))
        else:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    finally:
        if conn is not None:
            conn.close()


@app.command()
def knowledge(
    retry_rejected: bool = typer.Option(
        False, "--retry-rejected", help="Recheck semantically rejected drafts"
    ),
    validate: bool = typer.Option(
        False, "--validate", help="Run the source-only second pass on existing drafts"
    ),
    json_output: bool = typer.Option(False, "--json"),
    repo: str = typer.Option(".", "--repo"),
) -> None:
    """Inspect draft publication status or resume independent source validation."""
    from cfl.core import db
    from cfl.core.client import OllamaClient
    from cfl.pipeline.indexer import build_dense, build_lexical
    from cfl.pipeline.knowledge import validate_knowledge

    settings = get_settings() if repo == "." else load_settings(repo)
    try:
        with db.connect(settings.dsn) as conn:
            db.run_migrations(conn, settings.embed_dim, settings.migrations_dir)
            if validate:
                with OllamaClient(settings) as client:
                    result = validate_knowledge(
                        conn, client, settings, retry_rejected=retry_rejected
                    )
                    result["lexical"] = build_lexical(conn)
                    result["dense"] = build_dense(conn, client, settings)
            else:
                drafts = db.knowledge_drafts(conn)
                active = {r["id"] for r in db.published_knowledge(conn)}
                result = {
                    "counts": {
                        status: sum(r["status"] == status for r in drafts)
                        for status in ["pending", "complete", "partial", "rejected"]
                    },
                    "currently_publishable": len(active),
                    "drafts": [
                        {
                            "id": r["id"],
                            "status": r["status"],
                            "current": r["id"] in active,
                            "review": r["review"],
                        }
                        for r in drafts
                    ],
                }
            typer.echo(json.dumps(result, default=str, ensure_ascii=False))
    except (CflError, psycopg.Error, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}) if json_output else "Error: " + str(exc))
        raise typer.Exit(1) from exc
