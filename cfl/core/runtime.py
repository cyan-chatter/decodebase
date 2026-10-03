from __future__ import annotations

from cfl.config import Settings


def ollama_environment(settings: Settings) -> dict[str, str]:
    """Server settings: unquantized f16 by default, quantized only on opt-in."""
    return {
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "2",
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": (
            settings.kv_quantization_type if settings.kv_quantization else "f16"
        ),
    }
