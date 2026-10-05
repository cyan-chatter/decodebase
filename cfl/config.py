from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel


class Settings(BaseModel):
    """CodeFlowLens runtime configuration.

    Priority (highest first):
    1. Environment variables with CFL_ prefix
    2. cfl.toml in the current working directory
    3. Defaults below
    """

    # Database
    dsn: str = "postgresql://cfl:cfl@localhost:5432/cfl"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    gen_model: str = "qwen3.5:9b"
    verifier_model: str = "qwen2.5-coder:7b"
    verifier_tokenizer_file: str | None = None
    num_predict_review: int = 1800
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    # Ollama server environment, not per-request options. Restart to apply.
    kv_quantization: bool = False
    kv_quantization_type: Literal["q8_0", "q4_0"] = "q8_0"

    # Generation
    num_ctx: int = 8192
    temperature: float = 0.1
    seed: int = 42
    num_predict_symbol: int = 350
    num_predict_brief: int = 120
    num_predict_detailed: int = 1200
    template_overhead: int = 64
    gen_disable_thinking: bool = True

    # Graph / retrieval
    edge_conf_threshold: float = 0.6
    retrieval_top_k: int = 10
    retrieval_inject_max: int = 6
    rrf_k: int = 60
    flow_depth: int = 4
    flow_max_nodes: int = 40
    feature_depth: int = 4

    # I/O limits
    max_file_bytes: int = 500_000
    embed_max_tokens: int = 1024

    # Postgres
    statement_timeout_ms: int = 5000

    # Token budget helpers
    chars_per_token: float = 3.2
    token_margin: float = 0.15
    tokenizer_file: str | None = None

    # State
    state_dir: str = ".cfl"
    migrations_dir: str = "migrations"

    # Feature flags
    graph_expansion: bool = False
    llm_rerank: bool = False
    scip_overlay: bool = False
    dynamic_overlay: bool = False

    # Hardware
    vram_limit_gb: float = 10.0


def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning an empty dict if it doesn't exist."""
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _env_overrides() -> dict:
    """Collect CFL_* environment variables, strip prefix, lowercase."""
    prefix = "CFL_"
    result: dict = {}
    for key, val in os.environ.items():
        if key.startswith(prefix):
            field = key[len(prefix) :].lower()
            result[field] = val
    return result


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) Settings instance.

    Merge order: env vars > cfl.toml > defaults.
    """
    return load_settings()


def load_settings(repo_root: str | Path = ".") -> Settings:
    """Load repo configuration with environment overrides, without changing cwd."""
    toml_data = _load_toml(Path(repo_root) / "cfl.toml")
    env_data = _env_overrides()

    merged: dict = {**toml_data, **env_data}
    return Settings(**merged)


def ensure_state_dir(settings: Settings) -> None:
    """Create the state directory (default: .cfl/) if it does not exist."""
    Path(settings.state_dir).mkdir(parents=True, exist_ok=True)
