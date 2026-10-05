from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from cfl.config import load_settings
from cfl.core import db
from cfl.core.client import OllamaClient
from cfl.core.errors import PreflightError
from cfl.core.preflight import check_models_pulled, check_residency, warm_and_benchmark
from cfl.pipeline.indexer import build_dense, build_lexical
from cfl.pipeline.scan import run_stage1, run_stage2
from cfl.pipeline.symbol_pass import is_test, is_trivial, run_symbol_pass
from cfl.prompts.prompts import SUMMARY_PREFIX, SYSTEM_INGEST


def estimate(conn, bench, settings, *, counter, skip_tests=False, min_lines=0):
    if bench.prefill_tps <= 0 or bench.gen_tps <= 0:
        raise PreflightError("A measured throughput benchmark is required")
    symbols = db.ctx_inputs(conn)
    candidates = [s for s in symbols if not is_trivial(s, min_lines, is_test(s), skip_tests)]
    edges = db.fetch_edges(conn, settings.edge_conf_threshold)
    prefix = counter.count(SYSTEM_INGEST + "\n" + SUMMARY_PREFIX)
    ceiling = settings.num_ctx - settings.template_overhead - settings.num_predict_symbol
    total_prompt = sum(
        min(
            ceiling,
            counter.count(s["raw_code"])
            + 40 * sum(e["caller_id"] == s["id"] for e in edges)
            + prefix,
        )
        for s in candidates
    )
    seconds = total_prompt / bench.prefill_tps + len(candidates) * 150 / bench.gen_tps
    return {
        "llm_symbols": len(candidates),
        "trivial_symbols": len(symbols) - len(candidates),
        "prompt_tokens": total_prompt,
        "output_tokens": len(candidates) * 150,
        "hours": seconds / 3600,
        "prefill_tps": bench.prefill_tps,
        "gen_tps": bench.gen_tps,
        "limits": "Summary-generation estimate excludes file/feature drafts, independent validation, model switching, schema repairs, chunk reductions, embedding and database I/O.",
    }


def run_build(
    repo,
    *,
    estimate_only=False,
    resume=True,
    skip_tests=False,
    min_lines=0,
    priority=None,
    budget_hours=3,
    retry_failed=False,
    settings=None,
    client=None,
    progress=None,
    force=False,
    on_drafts=None,
):
    settings = settings or load_settings()
    root = Path(repo).resolve()
    own_client = client is None
    client = client or OllamaClient(settings)
    try:
        check_models_pulled(client, settings)
        tags = {row["name"]: row for row in client.tags()}
        verifier = (
            settings.verifier_model
            if ":" in settings.verifier_model
            else settings.verifier_model + ":latest"
        )
        generator = (
            settings.gen_model if ":" in settings.gen_model else settings.gen_model + ":latest"
        )
        if verifier not in tags or tags[verifier]["digest"] == tags[generator]["digest"]:
            raise PreflightError("Build requires a distinct installed verifier_model checkpoint")
        bench = warm_and_benchmark(client, settings)
        check_residency(client, settings)
        if bench.truncation_warning or bench.embed_dim != settings.embed_dim:
            raise PreflightError("Token context or embedding dimension could not be verified")
        with db.connect(settings.dsn, statement_timeout_ms=settings.statement_timeout_ms) as conn:
            migrations = Path(settings.migrations_dir)
            if not migrations.is_dir():
                migrations = Path(__file__).resolve().parents[2] / "migrations"
            db.run_migrations(conn, settings.embed_dim, str(migrations))
            if force:
                db.invalidate_parsed_files(conn, "python")
                db.invalidate_summaries(conn)
            scan = run_stage1(conn, settings, str(root))
            run_stage2(conn, settings, str(root))
            build_lexical(conn)
            projection = estimate(
                conn,
                bench,
                settings,
                counter=client.counter,
                skip_tests=skip_tests,
                min_lines=min_lines,
            )
            result = {
                "scan": asdict(scan) if scan else None,
                "estimate": projection,
                "over_budget": projection["hours"] > budget_hours,
            }
            if estimate_only:
                return result
            summaries = run_symbol_pass(
                conn,
                client,
                settings,
                resume=resume,
                priority=priority,
                skip_tests=skip_tests,
                min_lines=min_lines,
                retry_failed=retry_failed,
                progress=progress,
            )
            result["symbols"] = asdict(summaries)
            if not summaries.interrupted:
                from cfl.pipeline.knowledge import generate_knowledge, validate_knowledge

                result["knowledge_generation"] = generate_knowledge(conn, client, settings)
                if on_drafts:
                    on_drafts(conn)
                result["knowledge_validation"] = validate_knowledge(
                    conn, client, settings, retry_rejected=retry_failed
                )
                build_lexical(conn)
                result["dense"] = build_dense(conn, client, settings)
            return result
    finally:
        if own_client:
            client.close()
