from __future__ import annotations

import json

from cfl.prompts.schemas import SYMBOL_SUMMARY_JSON_SCHEMA

PROMPT_VERSION = "2"
SYSTEM_INGEST = (
    "You are a precise code documentation engine. Describe only what the provided code "
    "and context show. Never invent symbols, files or behavior not present."
)
SUMMARY_PREFIX = (
    "Summarize the supplied code as a JSON object matching this schema. "
    "Do not infer missing implementations. Use a nonempty string such as 'None shown' "
    "when a string field does not apply. Distinguish parameters from captured variables, "
    "decorator factories from returned wrappers, copies from mutation, and stubs from "
    "real external effects.\n[SCHEMA]\n"
    + json.dumps(SYMBOL_SUMMARY_JSON_SCHEMA, sort_keys=True, separators=(",", ":"))
)


def build_symbol_prompt(symbol: dict, callees: list[tuple[str, str]]) -> tuple[str, str]:
    """Place shared instructions/schema before deterministic variable evidence."""
    dependencies = "\n".join(f"{id}: {summary}" for id, summary in sorted(callees))
    prompt = "\n".join(
        [
            SUMMARY_PREFIX,
            "[FILE] " + symbol["file_path"],
            "[SIGNATURE] " + (symbol.get("signature") or symbol["qualname"]),
            "[CALLEES]\n" + dependencies,
            "[CODE]\n" + symbol["raw_code"],
            "Return only the JSON object",
        ]
    )
    return SYSTEM_INGEST, prompt
