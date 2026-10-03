from __future__ import annotations

import pkg.mod as m
import pkg.mod
from pkg import public_leaf
from pkg.relative import run_relative


def exercise_imports() -> None:
    m.leaf()
    pkg.mod.leaf()
    public_leaf()
    run_relative()


def call_unknown(obj) -> None:
    obj.unique_action()
