from __future__ import annotations

import typer
from rich.console import Console

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


@app.command()
def scan(
    repo: str = typer.Argument(".", help="Path to the repository to scan"),
) -> None:
    """Scan a repository: parse files (Stage 1) and resolve call graph (Stage 2)."""
    from cfl.core.db import connect
    from cfl.pipeline.scan import run_stage1, run_stage2

    settings = get_settings()
    conn = connect(settings.dsn)
    try:
        run_stage1(conn, settings, repo)
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
# Graph queries
# ---------------------------------------------------------------------------


@app.command()
def callers(
    symbol: str = typer.Argument(..., help="Symbol to find callers of"),
    depth: int = typer.Option(2, "--depth", "-d", help="Search depth"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """List functions that call the given symbol."""
    console.print("[yellow]not implemented yet[/yellow]")


@app.command()
def callees(
    symbol: str = typer.Argument(..., help="Symbol to find callees of"),
    depth: int = typer.Option(2, "--depth", "-d", help="Search depth"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """List functions called by the given symbol."""
    console.print("[yellow]not implemented yet[/yellow]")


@app.command()
def path(
    source: str = typer.Argument(..., help="Source symbol"),
    target: str = typer.Argument(..., help="Target symbol"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Find call path between two symbols."""
    console.print("[yellow]not implemented yet[/yellow]")


@app.command()
def impact(
    symbol: str = typer.Argument(..., help="Symbol to analyse for change impact"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Show what would be affected if this symbol changed."""
    console.print("[yellow]not implemented yet[/yellow]")


@app.command()
def hubs(
    top_n: int = typer.Option(20, "--top", "-n", help="Number of hub symbols to show"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Show the most-connected hub symbols."""
    console.print("[yellow]not implemented yet[/yellow]")


@app.command()
def dead(
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Find likely dead code (unreachable symbols)."""
    console.print("[yellow]not implemented yet[/yellow]")


# ---------------------------------------------------------------------------
# Source / tracing
# ---------------------------------------------------------------------------


@app.command()
def where(
    symbol: str = typer.Argument(..., help="Symbol to locate in source"),
    repo: str = typer.Option(".", "--repo", help="Repository path"),
) -> None:
    """Show source location of a symbol."""
    console.print("[yellow]not implemented yet[/yellow]")


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
