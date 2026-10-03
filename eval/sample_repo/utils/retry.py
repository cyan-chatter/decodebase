from __future__ import annotations

from functools import wraps


def with_retries(attempts: int = 3):
    """Decorate an operation to retry RuntimeError up to a bounded attempt count."""
    if attempts < 1:
        raise ValueError("Attempts must be positive")

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            for index in range(attempts):
                try:
                    return function(*args, **kwargs)
                except RuntimeError:
                    if index == attempts - 1:
                        raise

        return wrapped

    return decorate
