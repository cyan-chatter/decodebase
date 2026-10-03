from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

TRACE_LOCK = threading.Lock()


def trace_log(
    state_dir: str | Path,
    *,
    task: str,
    symbol_id: str | None,
    est_prompt_tokens: int,
    actual_prompt_tokens: int | None,
    output_tokens: int | None,
    latency_s: float,
    validation: str,
    attempt: int = 1,
    cached_prompt_tokens: int | None = None,
    prompt_eval_duration: int | None = None,
    eval_duration: int | None = None,
    load_duration: int | None = None,
) -> None:
    """Append call metadata without retaining source code, prompts or credentials."""
    now = datetime.now().astimezone()
    record = {
        "ts": now.isoformat(),
        "task": task,
        "symbol_id": symbol_id,
        "est_prompt_tokens": est_prompt_tokens,
        "actual_prompt_tokens": actual_prompt_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency_s,
        "validation": validation,
        "attempt": attempt,
        "cached_prompt_tokens": cached_prompt_tokens,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_duration": eval_duration,
        "load_duration": load_duration,
    }
    directory = Path(state_dir) / "logs"
    with TRACE_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"calls-{now:%Y%m%d}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
