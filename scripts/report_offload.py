"""Aggregate CPU-placement results after independent source review."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from cfl.config import load_settings
from scripts.compare_14b import validate_placement

ROOT = Path("docs/validation/cpu-offload")
CASES = ["coder14-f16-cpu-embed", "qwen14-f16-cpu-embed", "qwen27-f16-hybrid", "qwen27-q8_0-hybrid"]


def metrics(folder, case):
    report = json.loads((folder / "report.json").read_text())
    result = {
        "case": case,
        **{
            key: report.get(key)
            for key in [
                "model",
                "kv_type",
                "placement",
                "completed",
                "error",
                "peak_gpu_gb",
                "peak_ollama_pss_gb",
                "peak_ollama_rss_gb",
                "peak_ollama_swap_gb",
                "model_vram_bytes",
                "app_preflight_pass",
                "app_preflight_error",
                "cpu_memory_budget_exceeded",
                "sufficient_answers",
                "cached_answers",
                "summary_pass",
                "dense_wall_s",
                "summary_wall_s",
                "benchmark",
            ]
        },
    }
    if not report.get("completed"):
        return result
    assert not report.get("error")
    assert len(report["rows"]) == 48 and report["cached_answers"] == 0
    assert report["summary_pass"]["saved"] == 40 and report["summary_pass"]["failed"] == 0
    assert report["dense"]["count"] == 60 and not report["dense"]["failed"]
    assert not report.get("cpu_memory_budget_exceeded")
    validate_placement(
        report["resident_before"],
        load_settings().model_copy(update={"gen_model": report["model"]}),
        report["placement"],
    )
    validate_placement(
        report["resident_after"],
        load_settings().model_copy(update={"gen_model": report["model"]}),
        report["placement"],
    )
    review = json.loads((folder / "source-review.json").read_text())
    assert len(review["questions"]) == 48
    assert {r["question_id"] for r in review["questions"]} == {
        r["question_id"] for r in report["rows"]
    }
    result["source_review_counts"] = review["counts"]
    generated = [r for r in report["rows"] if r["metrics"]["type"] != "structural"]
    assert len(generated) == 40
    result["generated_answer_mean_s"] = statistics.mean(r["wall_s"] for r in generated)
    result["generated_answer_median_s"] = statistics.median(r["wall_s"] for r in generated)
    result["latency_by_question_type"] = {
        kind: {
            "count": len(rows),
            "mean_s": statistics.mean(r["wall_s"] for r in rows),
            "median_s": statistics.median(r["wall_s"] for r in rows),
        }
        for kind in sorted({r["metrics"]["type"] for r in report["rows"]})
        if (rows := [r for r in report["rows"] if r["metrics"]["type"] == kind])
    }
    result["means"] = {
        key: statistics.mean(r["metrics"][key] for r in report["rows"])
        for key in [
            "retrieved_recall_at_5",
            "cited_recall_at_5",
            "citation_validity",
            "tokens_per_answer",
        ]
    }
    calls = [json.loads(line) for line in (folder / "calls.jsonl").read_text().splitlines()]
    ok = [c for c in calls if c["validation"] == "ok" and c["task"] != "benchmark"]
    seconds = sum(c.get("eval_duration") or 0 for c in ok) / 1e9
    result["native_generation_tps"] = sum(c["output_tokens"] or 0 for c in ok) / seconds
    assert all(
        r["metrics"].get("structural_exactness", 1) == 1
        and r["metrics"].get("traversal_exactness", 1) == 1
        for r in report["rows"]
    )
    return result


def main():
    cases = [metrics(ROOT / case, case) for case in CASES]
    baseline = json.loads(Path("docs/validation/14b-comparison/summary.json").read_text())
    baseline = [row for row in baseline["cases"] if row["completed"]]
    verified = json.loads(
        Path("docs/validation/14b-comparison/baseline-f16/report.json").read_text()
    )
    expected_structural = {
        row["question_id"]: row["metrics"]["answer_text"]
        for row in verified["rows"]
        if row["metrics"]["type"] == "structural"
    }
    for case in CASES:
        report = json.loads((ROOT / case / "report.json").read_text())
        assert {
            row["question_id"]: row["metrics"]["answer_text"]
            for row in report["rows"]
            if row["metrics"]["type"] == "structural"
        } == expected_structural
    original = json.loads((ROOT / "original-resident.json").read_text())
    restored = json.loads((ROOT / "restored-resident.json").read_text())
    assert {m["name"]: m["context_length"] for m in original} == {
        m["name"]: m["context_length"] for m in restored
    }
    summary = {
        "cases": cases,
        "previous_comparison": baseline,
        "original_residency_restored": True,
        "cpu_budget_gb": 24,
        "cpu_budget_method": "sampled isolated Ollama-family PSS",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in cases:
        c = row.get("source_review_counts", {})
        print(
            row["case"],
            c,
            "latency",
            row.get("generated_answer_mean_s"),
            "CPU PSS GB",
            row["peak_ollama_pss_gb"],
            "GPU GB",
            row["peak_gpu_gb"],
        )


if __name__ == "__main__":
    main()
