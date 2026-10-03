from __future__ import annotations


def is_even(n: int) -> bool:
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n: int) -> bool:
    if n == 0:
        return False
    return is_even(n - 1)


def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return factorial(n - 1)
