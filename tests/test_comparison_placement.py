"""Guard diagnostic claims against accidentally measuring a different placement."""

import pytest

from cfl.config import Settings
from scripts.compare_14b import PlacementClient, validate_placement


def resident(gen_vram=100, embed_vram=0):
    return [
        {"name": "qwen3.5:9b", "size": 100, "size_vram": gen_vram, "context_length": 8192},
        {"name": "nomic-embed-text:latest", "size": 10, "size_vram": embed_vram},
    ]


def test_cpu_embed_options_do_not_affect_generator(fake_ollama, tmp_path):
    settings = Settings(gen_model="qwen3.5:9b", state_dir=str(tmp_path))
    with PlacementClient(settings, cpu_embed=True, transport=fake_ollama.transport) as client:
        client.embed(["hello"])
        assert fake_ollama.call_log[-1]["body"]["options"]["num_gpu"] == 0
        client.generate("small prompt", "", num_predict=64, task="placement-test")
        assert "num_gpu" not in fake_ollama.call_log[-1]["body"]["options"]


def test_cpu_embedding_test_rejects_generator_spill():
    settings = Settings(gen_model="qwen3.5:9b")
    validate_placement(resident(), settings, "cpu-embed")
    with pytest.raises(RuntimeError, match="Generator spilled"):
        validate_placement(resident(gen_vram=80), settings, "cpu-embed")
    with pytest.raises(RuntimeError, match="entirely on CPU"):
        validate_placement(resident(embed_vram=2), settings, "cpu-embed")


def test_hybrid_test_requires_both_cpu_and_gpu_layers():
    settings = Settings(gen_model="qwen3.5:9b")
    validate_placement(resident(gen_vram=80), settings, "hybrid")
    for vram in [0, 100]:
        with pytest.raises(RuntimeError, match="split across"):
            validate_placement(resident(gen_vram=vram), settings, "hybrid")


def test_missing_embedder_and_context_mismatch_are_rejected():
    settings = Settings(gen_model="qwen3.5:9b")
    with pytest.raises(RuntimeError, match="Both selected"):
        validate_placement(resident()[:1], settings, "cpu-embed")
    with pytest.raises(RuntimeError, match="context"):
        validate_placement(resident(), settings.model_copy(update={"num_ctx": 4096}), "cpu-embed")


def test_default_comparison_still_requires_full_gpu():
    settings = Settings(gen_model="qwen3.5:9b")
    validate_placement(resident(embed_vram=10), settings, "gpu")
    with pytest.raises(RuntimeError, match="fully GPU"):
        validate_placement(resident(), settings, "gpu")
