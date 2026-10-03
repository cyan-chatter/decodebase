from __future__ import annotations

from .mod import leaf
from . import mod as sibling


def run_relative() -> None:
    leaf()
    sibling.leaf()
