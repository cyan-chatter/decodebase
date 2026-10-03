from __future__ import annotations

import hashlib
import hmac


class TokenError(ValueError):
    """An invalid or unauthenticated access token."""


def issue_token(user_id: str, secret: str) -> str:
    """Sign a user identifier with an HMAC digest."""
    digest = hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()
    return user_id + ":" + digest


def validate_token(token: str, secret: str) -> str:
    """Verify a signed access token and return its user identifier."""
    user_id, separator, _ = token.partition(":")
    if not separator or not hmac.compare_digest(token, issue_token(user_id, secret)):
        raise TokenError("Invalid token")
    return user_id
