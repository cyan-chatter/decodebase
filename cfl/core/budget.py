from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from cfl.config import Settings
from cfl.core.errors import BudgetExceeded

logger = logging.getLogger(__name__)
CALIBRATION_LOCK = threading.Lock()
T = TypeVar("T")


class TokenCounter:
    """Exact counting when configured, otherwise a calibrated estimate with margin."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or Settings()
        self.ratio = settings.chars_per_token
        self.margin = settings.token_margin
        if (
            self.ratio <= 0
            or not math.isfinite(self.ratio)
            or self.margin < 0
            or not math.isfinite(self.margin)
        ):
            raise ValueError("Token ratio must be positive and margin nonnegative")
        self.calibration_path = Path(settings.state_dir) / "calibration.json"
        self.ratio = self._stored_ratio(self.ratio)
        self.tokenizer = None
        if settings.tokenizer_file and Path(settings.tokenizer_file).is_file():
            from tokenizers import Tokenizer

            try:
                self.tokenizer = Tokenizer.from_file(settings.tokenizer_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tokenizer unavailable, using calibrated estimates: %s", exc)

    def _stored_ratio(self, default: float) -> float:
        try:
            ratio = float(json.loads(self.calibration_path.read_text())["ratio"])
            return ratio if ratio > 0 and math.isfinite(ratio) else default
        except (OSError, ValueError, TypeError, KeyError):
            return default

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        return math.ceil(len(text) / self.ratio * (1 + self.margin))

    def calibrate(self, est_chars: int, actual_tokens: int) -> None:
        if est_chars <= 0 or actual_tokens <= 0:
            return
        with CALIBRATION_LOCK:
            old_ratio = self._stored_ratio(self.ratio)
            estimated = math.ceil(est_chars / old_ratio * (1 + self.margin))
            if actual_tokens < 0.6 * estimated:
                logger.debug("Ignoring suspected prompt cache hit during calibration")
                return
            observed_ratio = est_chars / actual_tokens
            if abs(actual_tokens - estimated) / max(estimated, 1) > 0.15:
                logger.warning(
                    "Token estimate drift: estimated %d, actual %d", estimated, actual_tokens
                )
            self.ratio = 0.9 * old_ratio + 0.1 * observed_ratio
            try:
                self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.calibration_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"ratio": self.ratio, "actual_tokens": actual_tokens})
                )
                temporary.replace(self.calibration_path)
            except OSError as exc:
                logger.warning("Could not persist token calibration: %s", exc)


@dataclass(frozen=True)
class PromptPart:
    name: str
    text: str
    priority: int
    truncatable: bool = True
    min_tokens: int = 0


@dataclass
class Assembled:
    prompt: str
    prompt_tokens: int  # Includes the template overhead reserved during assembly.
    parts: list[PromptPart]
    dropped: list[str]
    truncated: list[str]


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start_line: int  # Original source range of this chunk's body; header is repeated.
    end_line: int
    token_est: int
    signature: str


def assert_fits(prompt_tokens: int, num_predict: int, num_ctx: int) -> None:
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (prompt_tokens, num_predict, num_ctx)
    ):
        raise ValueError("Token budgets must be integers")
    if prompt_tokens < 0 or num_predict < 0 or num_ctx <= 0:
        raise ValueError("Token counts must be nonnegative and context positive")
    if prompt_tokens + num_predict > num_ctx:
        raise BudgetExceeded(
            f"Prompt ({prompt_tokens}) + output ({num_predict}) exceeds context ({num_ctx})"
        )


def _truncate(text: str, cap: int, counter: TokenCounter, pointer: str = "") -> str:
    if counter.count(text) <= cap:
        return text
    lines = text.splitlines(keepends=True)
    suffix = "\n" + pointer if pointer else ""
    # Prefer complete lines, keeping both the beginning and the end.
    for kept in range(len(lines) - 1, 1, -1):
        head, tail = (kept + 1) // 2, kept // 2
        candidate = (
            "".join(lines[:head]).rstrip()
            + f"\n...[{len(lines) - kept} lines omitted]\n"
            + "".join(lines[-tail:]).lstrip()
            + suffix
        )
        if counter.count(candidate) <= cap:
            return candidate
    # A single long line needs character-level head/tail truncation.
    marker = f"\n...[{max(1, len(lines) - 2)} lines omitted]\n"
    low, high, best = 2, max(2, len(text) - 1), ""
    while low <= high:
        kept = (low + high) // 2
        candidate = text[: (kept + 1) // 2] + marker + text[-(kept // 2) :] + suffix
        if counter.count(candidate) <= cap:
            best, low = candidate, kept + 1
        else:
            high = kept - 1
    return best


def assemble(
    fixed_prefix: str,
    parts: list[PromptPart],
    *,
    num_ctx: int,
    num_predict: int,
    overhead: int,
    counter: TokenCounter | None = None,
) -> Assembled:
    counter = counter or TokenCounter()
    if overhead < 0 or any(part.priority < 1 or part.min_tokens < 0 for part in parts):
        raise ValueError("Overhead and minimum tokens must be nonnegative; priorities start at 1")
    assert_fits(counter.count(fixed_prefix) + overhead, num_predict, num_ctx)
    retained = list(parts)
    dropped, truncated = [], []

    def render():
        return "\n\n".join(
            text for text in [fixed_prefix, *(part.text for part in retained)] if text
        )

    for part in sorted(parts, key=lambda part: part.priority, reverse=True):
        if counter.count(render()) + overhead + num_predict <= num_ctx:
            break
        index = retained.index(part)
        if part.truncatable:
            retained[index] = replace(part, text="")
            available = num_ctx - overhead - num_predict - counter.count(render())
            minimum = max(1 if part.priority == 1 else 0, part.min_tokens)
            text = _truncate(part.text, max(0, available), counter)
            while text:
                retained[index] = replace(part, text=text)
                if counter.count(render()) <= num_ctx - overhead - num_predict:
                    break
                available -= 1
                text = _truncate(part.text, max(0, available), counter)
            if text and counter.count(text) >= minimum:
                truncated.append(part.name)
                continue
            retained[index] = part
        if part.priority == 1:
            raise BudgetExceeded(
                f"Required target part {part.name!r} needs statement-aware splitting"
            )
        retained.pop(index)
        dropped.append(part.name)
    prompt = render()
    tokens = counter.count(prompt) + overhead
    assert_fits(tokens, num_predict, num_ctx)
    return Assembled(prompt, tokens, retained, dropped, truncated)


def split_oversized(
    symbol: Any, adapter: Any, budget: int, *, counter: TokenCounter | None = None
) -> list[Chunk]:
    counter = counter or TokenCounter()
    raw = symbol["raw_code"] if isinstance(symbol, dict) else symbol.raw_code
    signature = symbol.get("signature", "") if isinstance(symbol, dict) else symbol.signature
    start = symbol.get("start_line", 1) if isinstance(symbol, dict) else symbol.start_line
    if budget < 1:
        raise ValueError("Chunk budget must be positive")
    if counter.count(raw) <= budget:
        return [
            Chunk(0, raw, start, start + len(raw.splitlines()) - 1, counter.count(raw), signature)
        ]
    spans = adapter.statement_spans(raw)
    if not spans:
        raise BudgetExceeded("No statement boundaries available for splitting")
    lines = raw.splitlines(keepends=True)
    header = "".join(lines[: spans[0][0] - 1])
    chunks = []
    pending, first_line, previous = "", spans[0][0], spans[0][0] - 1
    for index, (_, end) in enumerate(spans):
        if index == len(spans) - 1:
            end = len(lines)
        statement = "".join(lines[previous:end])
        if counter.count(header + statement) > budget:
            raise BudgetExceeded(f"Statement at line {start + previous} exceeds chunk budget")
        if pending and counter.count(header + pending + statement) > budget:
            text = header + pending
            chunks.append(
                Chunk(
                    len(chunks),
                    text,
                    start + first_line - 1,
                    start + previous - 1,
                    counter.count(text),
                    signature,
                )
            )
            pending, first_line = "", previous + 1
        pending += statement
        previous = end
    text = header + pending
    chunks.append(
        Chunk(
            len(chunks),
            text,
            start + first_line - 1,
            start + previous - 1,
            counter.count(text),
            signature,
        )
    )
    return chunks


def combine_prompt(
    signature: str,
    partials: Iterable[str],
    *,
    budget: int | None = None,
    counter: TokenCounter | None = None,
) -> str:
    prompt = (
        "Combine these partial summaries into one grounded summary.\n"
        f"Signature: {signature}\n\n"
        + "\n\n".join(f"Partial {index}: {text}" for index, text in enumerate(partials, 1))
    )
    if budget is not None:
        assert_fits((counter or TokenCounter()).count(prompt), 0, budget)
    return prompt


def batch_by_budget(items: Iterable[T], token_fn: Callable[[T], int], budget: int) -> list[list[T]]:
    if budget < 1:
        raise ValueError("Batch budget must be positive")
    batches, batch, used = [], [], 0
    for item in items:
        tokens = token_fn(item)
        if tokens < 0:
            raise ValueError("Item token count cannot be negative")
        if tokens > budget:
            raise BudgetExceeded("One item exceeds the batch budget")
        if batch and used + tokens > budget:
            batches.append(batch)
            batch, used = [], 0
        batch.append(item)
        used += tokens
    if batch:
        batches.append(batch)
    return batches


def cap_tool_output(
    text: str,
    cap_tokens: int,
    *,
    pointer: str = "See original tool output for omitted text.",
    counter: TokenCounter | None = None,
) -> str:
    counter = counter or TokenCounter()
    if cap_tokens < 1:
        raise ValueError("Tool output cap must be positive")
    result = _truncate(text, cap_tokens, counter, pointer)
    if not result:
        raise BudgetExceeded("Tool output cap cannot fit a head, tail and pointer")
    return result


def output_reserve(task: str, settings: Settings | None = None) -> int:
    settings = settings or Settings()
    return {
        "brief": settings.num_predict_brief,
        "detailed": settings.num_predict_detailed,
        "answer": settings.num_predict_detailed,
        "chat": settings.num_predict_detailed,
        "module": 300,
        "one_liner": 300,
    }.get(task, settings.num_predict_symbol)


class ChatCounter:
    """Count the exact single-user-message envelope used by answer calls."""

    def __init__(self, counter):
        self.counter = counter

    def count(self, text):
        return self.counter.count(
            json.dumps([{"role": "user", "content": text}], ensure_ascii=False)
        )
