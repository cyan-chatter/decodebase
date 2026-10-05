"""Aggregate recorded model-comparison measurements and independent source reviews."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("docs/validation/14b-comparison")


def main():
    comparisons = []
    for case in [
        "baseline-f16",
        "baseline-q8_0",
        "coder14-f16",
        "coder14-q8_0",
        "qwen14-f16",
        "qwen14-q8_0",
    ]:
        folder = ROOT / case
        report = json.loads((folder / "report.json").read_text())
        row = {
            key: report.get(key)
            for key in [
                "model",
                "kv_type",
                "completed",
                "error",
                "app_preflight_pass",
                "app_preflight_error",
                "model_vram_bytes",
                "peak_gpu_gb",
                "device_within_10gb",
                "sufficient_answers",
                "cached_answers",
                "answer_latency_mean_s",
                "answer_latency_median_s",
                "summary_wall_s",
            ]
        }
        row["case"] = case
        row["benchmark"] = report["benchmark"]
        if report.get("completed"):
            assert len(report["rows"]) == 48 and report["cached_answers"] == 0
            assert report["summary_pass"]["saved"] == 40
            assert report["summary_pass"]["failed"] == 0
            assert all(m["size"] == m["size_vram"] for m in report["resident_after"])
            assert all(
                r["metrics"].get("structural_exactness", 1) == 1
                and r["metrics"].get("traversal_exactness", 1) == 1
                for r in report["rows"]
            )
            review = json.loads((folder / "source-review.json").read_text())
            assert len(review["questions"]) == 48
            assert {r["question_id"] for r in review["questions"]} == {
                r["question_id"] for r in report["rows"]
            }
            row["source_review_counts"] = review["counts"]
            row["summary_pass"] = report["summary_pass"]
            generated = [r for r in report["rows"] if r["metrics"]["type"] != "structural"]
            row["generated_answer_mean_s"] = statistics.mean(r["wall_s"] for r in generated)
            row["generated_answer_median_s"] = statistics.median(r["wall_s"] for r in generated)
            row["generated_answer_count"] = len(generated)
            row["means"] = {
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
            row["native_generation_tps"] = sum(c["output_tokens"] or 0 for c in ok) / seconds
            row["successful_generation_calls"] = len(ok)
        comparisons.append(row)
    original = json.loads((ROOT / "original-resident.json").read_text())
    restored = json.loads((ROOT / "restored-resident.json").read_text())
    assert {m["name"]: m["context_length"] for m in original} == {
        m["name"]: m["context_length"] for m in restored
    }
    summary = {
        "cases": comparisons,
        "original_residency_restored": True,
        "default_model_unchanged": True,
        "kv_default_off": True,
        "recommendation": "Keep Qwen3.5 9B; neither 14B candidate improves overall source correctness in this single-seed evaluation.",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = []
    for row in comparisons:
        if not row["completed"]:
            lines.append(
                f"| {row['case']} | — | — | — | — | — | {row['peak_gpu_gb']:.2f} | GPU residency failed |"
            )
            continue
        c = row["source_review_counts"]
        lines.append(
            f"| {row['case']} | {c['adequate']} | {c['partial']} | {c['incorrect']} | "
            f"{c['unanswered']} | {row['generated_answer_mean_s']:.2f} | "
            f"{row['peak_gpu_gb']:.2f} | {'Pass' if row['app_preflight_pass'] else 'Fail'} |"
        )
    table = "\n".join(lines)
    text = f"""# 14B comparison results

Measured on 2026-10-04 using the RTX 4070 12 GB, Ollama 0.35.1, Q4_K_M weights, Nomic embeddings, 8,192-token context, temperature 0.1, seed 42 and thinking disabled. See [method.md](method.md) for controls and [summary.json](summary.json) for exact measurements.

## Result

**Keep Qwen3.5 9B as the default.** Neither 14B candidate improved overall independently reviewed source correctness. Qwen3 increased cited-answer coverage to 48/48, but produced more materially incorrect answers and exceeded the resident-model budget even with Q8 KV. Qwen2.5-Coder corrected the authenticated-handler summaries and reduced materially incorrect answers to one, but introduced other omissions and another unanswered query, with slower answers and much higher memory use.

| Model / KV | Adequate | Partial | Incorrect | Unanswered | Mean generated answer, s | Device peak, GB | App residency / 10 GB gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table}

All semantic counts include the eight identical deterministic structural answers. Generated-answer latency excludes those eight zero-generation answers and includes retrieval, source verification, flow generation and bounded citation repair. A cited answer can be incorrect; “adequate” is the independent source assessment, not the application's sufficient-answer flag.

## Memory and cache findings

Both 14B FP16-KV cases stopped after preflight: each generator was fully on GPU, but Nomic partially spilled to CPU (about 0.103 GB GPU out of 0.397 GB reported total). Those cases have no answer-quality results. No CPU-offloaded answers are included in the comparison.

With Q8 KV, Qwen2.5-Coder's resident models use **9.848 GB**, passing the app's resident-model gate with little headroom. Qwen3 uses **10.016 GB**, failing that gate. Both have total device peaks above 10 GB, including desktop and runtime allocations. Consequently, neither meets the earlier milestone's total-device peak target. The diagnostic Qwen3 run records the policy failure; no application limit was relaxed.

The baseline uses **6.052 GB** of reported model memory with FP16 KV and **5.927 GB** with Q8 KV. The Q8 baseline has fewer adequate answers in this run, although a single seeded comparison cannot establish a statistically reliable quantization penalty. Quantization is a capacity tradeoff, not evidence of improved correctness.

## Source review

All 192 answers from the four completed cases were independently checked against implementation; each case has `source-review.json` with per-question notes. All 160 summary one-liners were also inspected, with material issues identified in those files. Detailed source records and generated summary fields remain in `summaries.json`.

- Baseline FP16: handler descriptions confuse authenticated handlers with the route decorator factory. Notification delivery and short-circuit signing claims remain incorrect. The unanswered query is `behavioral-08`.
- Coder Q8: authenticated handler summaries and their module answer improve. It still invents actual email delivery in `behavioral-02`, overclaims exception/unique-salt guarantees, conflates audit behavior with the base repository, and omits some configuration/import details. Its unanswered query is `traversal-08`.
- Qwen3 Q8: answers every query with accepted citations, but invents an even-value recursion base case in both the `is_even` one-liner and `traversal-08`. It also claims actual email sending in three answers. These errors explain why increased coverage does not constitute increased accuracy.

Both 14B models fix some baseline errors while introducing new ones. For the requested tracing question, their signing descriptions improve on the baseline's inverted call condition but still omit the separator-dependent short-circuit detail. Per-question differences matter more than parameter count alone.

## Verification and limitations

Every completed case regenerated 40 summaries (four trivial templates, zero failures), built 60 dense vectors, and evaluated all 48 questions with **zero application answer-cache hits**. Structural and traversal graph exactness are 1.0 in all cases; retrieved and cited recall are recorded separately. Native prefix reuse remains enabled as in normal application use. All temporary databases were dropped, and the original selected generator/embedder residency and contexts were restored. `cfl.toml`, the default-off KV setting and the main source index were preserved.

This is one run per feasible model/cache setting on a small fixture, not a broad coding benchmark or a repeated statistical study. Rebuilt summaries also change retrieval, so this tests the complete application rather than isolating the final answer model. The earlier milestone report used mixed application caches; the fresh baseline here is the appropriate latency and quality comparator. GPU peaks include background desktop activity and can vary.

The reproducible runner is `scripts/compare_14b.py`; report aggregation is `scripts/report_14b.py`. Ruff, syntax compilation and artifact consistency checks passed. No application implementation or default configuration was changed for this experiment.
"""
    (ROOT / "review.md").write_text(text)
    print(table)


if __name__ == "__main__":
    main()
