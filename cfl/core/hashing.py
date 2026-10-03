from __future__ import annotations

import hashlib
import json
import pathlib


def sha256_hex(data: str | bytes) -> str:
    """Return the SHA256 hash of the given data as a hex string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    """Compute SHA256 of a file, reading in 8KB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def normalize_code(s: str) -> str:
    """Normalize line endings and outer blank lines, preserving semantic whitespace."""
    # Replace CRLF with LF
    s = s.replace("\r\n", "\n")
    lines = s.split("\n")
    # Strip outer blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def code_hash(raw_code: str) -> str:
    """Return SHA256 hex of the normalized code."""
    return sha256_hex(normalize_code(raw_code))


def join_hash(*parts: str) -> str:
    """Join parts with unit separator and return SHA256 hex."""
    return sha256_hex("\x1f".join(parts))


def ctx_hash(
    code_hash: str,
    callee_pairs: list[tuple[str, str]],
    prompt_version: str,
    gen_model_tag: str,
    *,
    gen_model_digest: str | None = None,
    generation_options: dict | None = None,
    schema_version: str = "1",
) -> str:
    """Compute context hash from code hash and callee summaries.

    Args:
        code_hash: The hash of the current code
        callee_pairs: List of (callee_id, summary_short) tuples
        prompt_version: Version identifier for the prompt
        gen_model_tag: Model tag used for generation

    Returns:
        SHA256 hex of the combined context
    """
    return fingerprint(
        {
            "version": 2,
            "code_hash": code_hash,
            "callees": sorted(callee_pairs),
            "prompt_version": prompt_version,
            "schema_version": schema_version,
            "model": gen_model_tag,
            "model_digest": gen_model_digest,
            "generation_options": generation_options or {},
        }
    )


def embed_key(embed_text: str, embed_model_tag: str, model_digest: str | None = None) -> str:
    """Generate embedding key from text and model tag."""
    return fingerprint(
        {"version": 2, "text": embed_text, "model": embed_model_tag, "digest": model_digest}
    )


def fingerprint(value: object) -> str:
    """Hash structured cache inputs with canonical, unambiguous serialization."""
    return sha256_hex(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def member_hash(symbol_ids: list[str]) -> str:
    """Compute hash from sorted symbol IDs."""
    return sha256_hex("\x1f".join(sorted(symbol_ids)))


def aggregate_hash(parts: list[str]) -> str:
    """Compute hash from parts, preserving order with unit separator."""
    return sha256_hex("\x1f".join(parts))


def symbol_id(path: str, qualname: str, start_line: int, ambiguous: bool) -> str:
    """Generate a unique symbol ID.

    Args:
        path: File path
        qualname: Qualified name
        start_line: Starting line number
        ambiguous: Whether the qualname is ambiguous in the file

    Returns:
        Unique symbol identifier
    """
    base = f"{path}::{qualname}"
    if ambiguous:
        return f"{base}@{start_line}"
    return base
