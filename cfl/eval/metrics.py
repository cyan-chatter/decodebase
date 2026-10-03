from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Span:
    path: str
    start_line: int
    end_line: int

    def __post_init__(self):
        if (
            not isinstance(self.path, str)
            or not self.path
            or any(
                not isinstance(line, int) or isinstance(line, bool)
                for line in (self.start_line, self.end_line)
            )
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ValueError("Invalid source span")
        object.__setattr__(self, "path", str(PurePosixPath(self.path.replace("\\", "/"))))

    @classmethod
    def parse(cls, value: Span | dict | tuple | str) -> Span:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                value.get("file_path", value.get("path", "")),
                value["start_line"],
                value["end_line"],
            )
        if isinstance(value, str):
            if value.startswith("["):
                if not value.endswith("]"):
                    raise ValueError(f"Invalid citation: {value}")
                value = value[1:-1]
            elif value.endswith("]"):
                raise ValueError(f"Invalid citation: {value}")
            match = re.fullmatch(r"(.+?):(\d+)-(\d+)", value)
            if not match:
                raise ValueError(f"Invalid citation: {value}")
            return cls(match[1], int(match[2]), int(match[3]))
        return cls(*value)


def answer_recall_at_5(expected_spans: Iterable, returned_spans: Iterable) -> float:
    expected = set(map(Span.parse, expected_spans))
    returned = list(dict.fromkeys(map(Span.parse, returned_spans)))[:5]
    if not expected:
        return 1.0
    matches = sum(
        any(
            want.path == got.path
            and want.start_line <= got.end_line
            and got.start_line <= want.end_line
            for got in returned
        )
        for want in expected
    )
    return matches / len(expected)


def citation_validity(citations: Iterable, retrieved: Iterable) -> float:
    available = list(map(Span.parse, retrieved))
    values = list(citations)
    valid = 0
    for value in values:
        try:
            span = Span.parse(value)
        except (ValueError, KeyError, TypeError):
            continue
        valid += any(
            span.path == block.path
            and block.start_line <= span.start_line
            and span.end_line <= block.end_line
            for block in available
        )
    return valid / len(values) if values else 0.0


def structural_exactness(expected_set: Iterable[str], got_set: Iterable[str]) -> float:
    return float(set(expected_set) == set(got_set))


def tokens_per_answer(
    records: Iterable[dict] | str | Path,
    *,
    answer_count: int | None = None,
    task: str | None = None,
    symbol_id: str | None = None,
) -> float | None:
    """Average actual input + output tokens; answer_count supports multi-call answers.

    Without answer_count, each successful matching call is treated as an answer.
    No qualifying calls returns None, rather than claiming measured zero cost.
    """
    if isinstance(records, str | Path):
        path = Path(records)
        files = sorted(path.glob("calls-*.jsonl")) if path.is_dir() else [path]
        records = [
            json.loads(line)
            for file in files
            for line in file.read_text().splitlines()
            if line.strip()
        ]
    counts = []
    for record in records:
        if record.get("validation") != "ok" or (task is not None and record.get("task") != task):
            continue
        if symbol_id is not None and record.get("symbol_id") != symbol_id:
            continue
        prompt, output = record.get("actual_prompt_tokens"), record.get("output_tokens")
        if isinstance(prompt, int) and isinstance(output, int) and prompt >= 0 and output >= 0:
            counts.append(prompt + output)
    if answer_count is not None and answer_count < 1:
        raise ValueError("answer_count must be positive")
    return sum(counts) / (answer_count or len(counts)) if counts else None
