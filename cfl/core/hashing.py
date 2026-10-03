from __future__ import annotations

import hashlib
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
    """Normalize code string: replace CRLF, rstrip each line, strip outer blank lines."""
    # Replace CRLF with LF
    s = s.replace("\r\n", "\n")
    # Rstrip each line
    lines = [line.rstrip() for line in s.splitlines()]
    # Strip outer blank lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
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
    # Hash each summary_short
    hashes = [sha256_hex(summary) for _, summary in callee_pairs]
    # Sort by callee_id
    sorted_pairs = sorted(
        [(cid, h) for (cid, _), h in zip(callee_pairs, hashes)],
        key=lambda x: x[0],
    )
    sorted_hashes = [h for _, h in sorted_pairs]
    # Join and hash with code_hash, prompt_version, gen_model_tag
    return join_hash(code_hash, *sorted_hashes, prompt_version, gen_model_tag)


def embed_key(embed_text: str, embed_model_tag: str) -> str:
    """Generate embedding key from text and model tag."""
    return sha256_hex(embed_text + "\x1f" + embed_model_tag)


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