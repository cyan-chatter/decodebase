from __future__ import annotations

import json
import sys
from dataclasses import asdict

import click
import psycopg
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from cfl.config import get_settings
from cfl.core.errors import CflError

app = typer.Typer(no_args_is_help=True, help="CodeFlowLens – local code intelligence")
console = Console(stderr=True)


@app.callback()
def _callback() -> None:
    """CodeFlowLens top-level callback — catches CflError and exits cleanly."""


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
    repo: str = typer.Argument(".", help="Path to the repository to index"),
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel worker count"),
) -> None:
    """Index a repository (parse, embed, summarise)."""
    console.print("[yellow]not implemented yet[/yellow]")


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
        console.print("[green]Scan complete.[/green]")
    finally:
        conn.close()


@app.command()
def status(
    repo: str = typer.Argument(".", help="Repository path"),
) -> None:
    """Show index status and staleness for a repository."""
    console.print("[yellow]not implemented yet[/yellow]")


# ---------------------------------------------------------------------------
# Symbol / code queries
# ---------------------------------------------------------------------------


@app.command()
def explain(
    symbol: str = typer.Argument(..., help="Fully-qualified symbol (e.g. mymod.MyClass.method)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Emit detailed explanation"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Explain a symbol (brief or detailed)."""
    console.print("[yellow]not implemented yet[/yellow]")


@app.command()
def flow(
    entrypoint: str = typer.Argument(..., help="Entry-point symbol for flow analysis"),
    depth: int = typer.Option(4, "--depth", "-d", help="Maximum call-graph depth"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Show control flow from an entrypoint."""
    console.print("[yellow]not implemented yet[/yellow]")


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
    question: str = typer.Argument(..., help="Question about the codebase"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Retrieval top-k"),
) -> None:
    """Answer a one-shot question about the codebase."""
    console.print("[yellow]not implemented yet[/yellow]")


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
    suite: str = typer.Option("eval/", "--suite", "-s", help="Path to evaluation suite"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Run the evaluation suite and report metrics."""
    console.print("[yellow]not implemented yet[/yellow]")
