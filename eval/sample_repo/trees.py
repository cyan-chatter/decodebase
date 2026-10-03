from __future__ import annotations


def walk_tree(node: dict) -> list[str]:
    """Recursively collect names in a task hierarchy in preorder."""
    names = [node["name"]]
    for child in node.get("children", []):
        names.extend(walk_tree(child))
    return names


def is_even(value: int) -> bool:
    """Determine parity through mutual recursion with is_odd."""
    return True if value == 0 else is_odd(value - 1)


def is_odd(value: int) -> bool:
    """Determine parity through mutual recursion with is_even."""
    return False if value == 0 else is_even(value - 1)
