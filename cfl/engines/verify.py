from __future__ import annotations

import builtins
import re
from dataclasses import dataclass, field

from cfl.core import db
from cfl.core.mermaid import validate_mermaid
from cfl.eval.metrics import Span

CITATION_RE = re.compile(r"\[([^\]\n]+?):(\d+)-(\d+)\]")
VERIFICATION_VERSION = "1"


@dataclass
class VerifiedAnswer:
    text: str
    citations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sufficient: bool = False


def verify_answer(answer, blocks, conn):
    warnings = []
    citations = []
    files = {f["path"]: f for f in db.list_files(conn)}

    def verify(match):
        path, start, end = match[1], int(match[2]), int(match[3])
        try:
            span = Span(path, start, end)
        except ValueError:
            warnings.append("Invalid citation: " + match[0])
            return ""
        file = files.get(path)
        contained = [
            b
            for b in blocks
            if b["file_path"] == path and b["start_line"] <= start <= end <= b["end_line"]
        ]
        in_source = file is not None and (
            file.get("line_count") is None or end <= file["line_count"]
        )
        if not contained or not in_source:
            warnings.append("Unsupported citation stripped: " + match[0])
            return ""
        current = []
        for block in contained:
            if block.get("kind") == "file":
                good = block.get("code_hash") == file["sha256"]
            else:
                symbol = db.get_symbol(conn, block["id"])
                good = symbol is not None and symbol["code_hash"] == block.get("code_hash")
            if good:
                current.append(block)
        if not current:
            warnings.append("Stale citation stripped: " + match[0])
            return ""
        citation = {"path": span.path, "start_line": start, "end_line": end}
        if citation not in citations:
            citations.append(citation)
        return match[0]

    text = CITATION_RE.sub(verify, answer)
    for name in re.findall(r"(?<!`)`([^`\n]+)`(?!`)", text):
        if (
            re.fullmatch(r"[A-Za-z_]\w*(?:\.\w+)*", name)
            and not hasattr(builtins, name)
            and name not in files
        ):
            rows = db.lookup_symbols(conn, name, 100)
            if not any(
                r["qualname"] == name
                or r["name"] == name
                or r["qualname"].endswith("." + name)
                or r["file_path"].removesuffix(".py").removesuffix("/__init__").replace("/", ".")
                + "."
                + r["qualname"]
                == name
                for r in rows
            ):
                warnings.append("Unknown identifier: " + name)

    def diagram(match):
        errors = validate_mermaid(match[1])
        if errors:
            warnings.extend("Mermaid: " + error for error in errors)
            return "[Invalid Mermaid diagram omitted]"
        return match[0]

    text = re.sub(r"```mermaid\s*\n(.*?)```", diagram, text, flags=re.DOTALL)
    if not citations:
        hits = "\n".join(
            f"- {b['id']} [{b['file_path']}:{b['start_line']}-{b['end_line']}]" for b in blocks[:5]
        )
        return VerifiedAnswer("insufficient context\n\nTop hits:\n" + hits, [], warnings, False)
    return VerifiedAnswer(text, citations, warnings, True)
