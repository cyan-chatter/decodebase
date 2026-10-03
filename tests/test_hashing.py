from __future__ import annotations

from cfl.core.hashing import (
    code_hash,
    ctx_hash,
    join_hash,
    normalize_code,
    sha256_hex,
)


def test_whitespace_only_change_keeps_code_hash():
    """Whitespace-only change keeps code_hash identical."""
    code1 = """def foo():
    return 1
"""
    code2 = """def foo():
        return 1
"""
    assert code_hash(code1) == code_hash(code2)


def test_callee_summary_short_change_changes_ctx_hash():
    """A callee summary_short change changes ctx_hash."""
    code_h = "abc123"
    pairs1 = [("callee1", "does thing A"), ("callee2", "does thing B")]
    pairs2 = [("callee1", "does thing A"), ("callee2", "does thing DIFFERENT")]

    h1 = ctx_hash(code_h, pairs1, "v1", "qwen2.5-coder:7b")
    h2 = ctx_hash(code_h, pairs2, "v1", "qwen2.5-coder:7b")

    assert h1 != h2


def test_callee_pair_order_independent():
    """The order of callee pairs does NOT matter (ctx_hash is order-independent)."""
    code_h = "abc123"
    pairs1 = [("callee1", "does A"), ("callee2", "does B")]
    pairs2 = [("callee2", "does B"), ("callee1", "does A")]

    h1 = ctx_hash(code_h, pairs1, "v1", "qwen2.5-coder:7b")
    h2 = ctx_hash(code_h, pairs2, "v1", "qwen2.5-coder:7b")

    assert h1 == h2


def test_normalize_code_basic():
    """Basic normalization works."""
    result = normalize_code("  hello  \r\n  world  \n")
    assert result == "hello\nworld"


def test_join_hash():
    """join_hash works with multiple parts."""
    h = join_hash("a", "b", "c")
    assert len(h) == 64  # SHA256 hex is 64 chars


def test_sha256_hex():
    """sha256_hex handles both str and bytes."""
    h_str = sha256_hex("hello")
    h_bytes = sha256_hex(b"hello")
    assert h_str == h_bytes