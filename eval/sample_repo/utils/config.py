from __future__ import annotations

import json


def parse_config(text: str) -> dict:
    """Parse a JSON configuration object and require an authentication secret."""
    config = json.loads(text)
    if not isinstance(config, dict) or not config.get("secret"):
        raise ValueError("Configuration requires a secret")
    return config


def load_config(path: str) -> dict:
    """Read configuration from a file and validate it."""
    with open(path, encoding="utf-8") as file:
        return parse_config(file.read())
