from __future__ import annotations

import hashlib
import hmac


def hash_password(password: str, salt: str) -> str:
    """Derive a password digest using PBKDF2 and a supplied random salt."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()


def verify_password(password: str, salt: str, expected: str) -> bool:
    """Compare a derived password digest in constant time."""
    return hmac.compare_digest(hash_password(password, salt), expected)
