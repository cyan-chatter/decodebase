from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def with_retries(
    max_retries: int = 3,
    *,
    retry_on: tuple[type[Exception], ...] = (RuntimeError,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry a synchronous function after matching exceptions, at most max_retries times.

    The initial call does not count as a retry. Zero disables retries; exhaustion
    re-raises the final exception. Exceptions outside retry_on propagate immediately.
    """
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise TypeError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    if (
        not isinstance(retry_on, tuple)
        or not retry_on
        or any(
            not isinstance(exception, type) or not issubclass(exception, Exception)
            for exception in retry_on
        )
    ):
        raise TypeError("retry_on must be a nonempty tuple of Exception subclasses")

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            retries = 0
            while True:
                try:
                    return function(*args, **kwargs)
                except retry_on:
                    if retries >= max_retries:
                        raise
                    retries += 1

        return wrapped

    return decorate
