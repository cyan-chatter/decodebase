from __future__ import annotations

import uuid


def cache_workloads(client, *, repetitions: int = 3, num_predict: int = 32) -> list[dict]:
    """Separate fresh prompts, exact reuse, shared prefixes, and changed prefixes."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rows = []
    for round in range(repetitions):
        marker = uuid.uuid4().hex
        prefix = (
            "Cache benchmark "
            + marker
            + "\n"
            + "\n".join(
                f"Record {i}: stable reference material for prompt prefix reuse." for i in range(45)
            )
        )
        cases = [
            ("uncached", prefix + "\nRequest A: return READY."),
            ("repeat", prefix + "\nRequest A: return READY."),
            ("shared_prefix", prefix + "\nRequest B: return READY."),
            (
                "changed_prefix",
                prefix.replace(marker, uuid.uuid4().hex) + "\nRequest A: return READY.",
            ),
        ]
        for case, prompt in cases:
            result = client.generate(
                prompt,
                "This is a cache measurement. Respond with READY only.",
                num_predict=num_predict,
                task="cache_benchmark",
            )
            uncached = result.uncached_prompt_tokens
            rows.append(
                {
                    "round": round,
                    "case": case,
                    "prompt_tokens": result.prompt_tokens,
                    "cached_prompt_tokens": result.cached_prompt_tokens,
                    "uncached_prompt_tokens": uncached,
                    "prefill_tps": (
                        uncached / (result.prompt_eval_duration / 1e9)
                        if uncached is not None and uncached >= 32 and result.prompt_eval_duration
                        else None
                    ),
                    "gen_tps": (
                        result.output_tokens / (result.eval_duration / 1e9)
                        if result.eval_duration
                        else None
                    ),
                    "prompt_eval_duration": result.prompt_eval_duration,
                    "eval_duration": result.eval_duration,
                    "load_duration": result.load_duration,
                    "latency_s": result.latency_s,
                    "output_tokens": result.output_tokens,
                }
            )
    return rows
