from __future__ import annotations

from typer.testing import CliRunner

from cfl.cli import app
from cfl.config import Settings
from cfl.core.runtime import ollama_environment


def test_kv_quantization_is_explicit_and_default_off():
    assert ollama_environment(Settings())["OLLAMA_KV_CACHE_TYPE"] == "f16"
    assert ollama_environment(Settings(kv_quantization=True))["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    result = CliRunner().invoke(app, ["runtime-env", "--no-kv-quantization"])
    assert result.exit_code == 0 and 'OLLAMA_KV_CACHE_TYPE="f16"' in result.output
    result = CliRunner().invoke(app, ["runtime-env", "--kv-quantization"])
    assert result.exit_code == 0 and 'OLLAMA_KV_CACHE_TYPE="q8_0"' in result.output
    result = CliRunner().invoke(
        app, ["runtime-env", "--kv-quantization", "--kv-type", "q4_0", "--format", "systemd"]
    )
    assert result.exit_code == 0 and 'Environment="OLLAMA_KV_CACHE_TYPE=q4_0"' in result.output
    assert CliRunner().invoke(app, ["runtime-env", "--kv-type", "invalid"]).exit_code != 0
