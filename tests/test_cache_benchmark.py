from __future__ import annotations

import json

from typer.testing import CliRunner

from cfl.cli import app
from cfl.core import client as client_module
from cfl.core.benchmark import cache_workloads


def test_workloads_separate_fresh_prompts_and_reuse(client, fake_ollama):
    rows = cache_workloads(client, repetitions=2)
    assert [row["case"] for row in rows] == [
        "uncached",
        "repeat",
        "shared_prefix",
        "changed_prefix",
    ] * 2
    bodies = [call["body"] for call in fake_ollama.call_log]
    assert bodies[0]["prompt"] == bodies[1]["prompt"]
    assert bodies[0]["prompt"].split("Request")[0] == bodies[2]["prompt"].split("Request")[0]
    assert bodies[0]["prompt"] != bodies[3]["prompt"] != bodies[4]["prompt"]
    assert all(row["uncached_prompt_tokens"] == 100 for row in rows)


def test_cache_benchmark_cli_saves_metrics(fake_ollama, monkeypatch, tmp_path):
    original = client_module.OllamaClient.__init__

    def initialize(self, settings):
        original(
            self,
            settings.model_copy(update={"state_dir": str(tmp_path)}),
            transport=fake_ollama.transport,
        )

    monkeypatch.setattr(client_module.OllamaClient, "__init__", initialize)
    path = tmp_path / "benchmark.json"
    result = CliRunner().invoke(app, ["cache-benchmark", "--repetitions", "1", "--out", str(path)])
    assert result.exit_code == 0, result.output
    report = json.loads(path.read_text())
    assert len(report["rows"]) == 4 and report["tags"][0]["digest"]
