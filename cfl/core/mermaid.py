from __future__ import annotations

import hashlib
import re
from copy import deepcopy


def safe_id(symbol_id):
    return "n" + str(int(hashlib.sha256(symbol_id.encode()).hexdigest()[:12], 16))


def escape_label(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("|", "&#124;")
        .replace("\n", " ")
        .replace("\\", "&#92;")
    )


def count_nodes(tree):
    stack = [tree]
    n = 0
    while stack:
        node = stack.pop()
        n += 1
        stack.extend(node.get("children", []))
    return n


def collapse(tree, max_nodes=40, depth=4):
    if max_nodes < 2 or depth < 0:
        raise ValueError("Require max_nodes >= 2 and depth >= 0")
    root = deepcopy(tree)
    remaining = [max_nodes - 1]

    def visit(node, level):
        children = node.get("children", [])
        if not children:
            return
        kept = []
        for i, child in enumerate(children):
            if level >= depth or remaining[0] <= 1:
                if remaining[0] > 0:
                    more = sum(count_nodes(c) for c in children[i:])
                    kept.append(
                        {
                            "id": node["id"] + "::collapsed",
                            "label": f"...{more} more",
                            "kind": "collapsed",
                            "count": more,
                            "children": [],
                        }
                    )
                    remaining[0] -= 1
                else:
                    node["omitted_count"] = sum(count_nodes(c) for c in children[i:])
                break
            remaining[0] -= 1
            kept.append(child)
            visit(child, level + 1)
        node["children"] = kept

    visit(root, 0)
    return root


def render_flowchart(trace_tree):
    lines = ["flowchart TD"]
    labels = {}
    index = [0]

    def render(node, parent=None):
        id = f"n{index[0]}"
        index[0] += 1
        label = node.get("label", node["id"]) + (" ↻" if node.get("recursion") else "")
        if node.get("omitted_count"):
            label += f" (...{node['omitted_count']} more)"
        labels[id] = label
        lines.append(f'  {id}["{escape_label(label)}"]')
        if parent is not None:
            condition = escape_label(node.get("control_ctx") or "")
            lines.append(f"  {parent} -->" + (f"|{condition}|" if condition else "") + f" {id}")
        for child in node.get("children", []):
            render(child, id)

    render(trace_tree)
    return "\n".join(lines), labels


def render_sequence(trace_tree):
    lines = ["sequenceDiagram"]
    participants = {}
    events = []

    def visit(node, parent=None):
        key = node["id"]
        id = participants.setdefault(key, f"n{len(participants)}")
        if parent is not None:
            events.append((parent, id, node.get("control_ctx") or node.get("label", key)))
        for child in node.get("children", []):
            visit(child, id)

    visit(trace_tree)
    for key, id in participants.items():
        lines.append(f"  participant {id} as {escape_label(key)}")
    for a, b, label in events:
        lines.append(f"  {a}->>{b}: {escape_label(label)}")
    return "\n".join(lines), {id: key for key, id in participants.items()}


def validate_mermaid(text):
    """Validate the safe subset emitted by CFL; reject unsupported directives."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not re.fullmatch(
        r"(?:flowchart|graph) (?:TD|TB|LR|RL|BT)|sequenceDiagram", lines[0]
    ):
        return ["Unknown Mermaid header"]
    errors = []
    declared = set()
    edges = []
    sequence = lines[0] == "sequenceDiagram"
    for line in lines[1:]:
        if sequence:
            node = re.fullmatch(r"participant (n\w+) as (.+)", line)
            edge = re.fullmatch(r"(n\w+)(?:->>|-->>)(n\w+): (.+)", line)
        else:
            node = re.fullmatch(r'(n\w+)\["([^"\[\]]+)"\]', line)
            edge = re.fullmatch(r"(n\w+)\s*(?:-->|-\.->|==>)(?:\|[^|]+\|)?\s*(n\w+)", line)
        if node:
            if node[1] in declared:
                errors.append("Duplicate node ID: " + node[1])
            declared.add(node[1])
        elif edge:
            edges.append((edge[1], edge[2]))
        else:
            errors.append("Invalid syntax, arrows, brackets, quotes or empty label: " + line[:100])
    for a, b in edges:
        if a not in declared or b not in declared:
            errors.append("Undefined arrow endpoint")
    return errors
