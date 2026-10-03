from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from cfl.core.trace_log import trace_log


def test_append_only_concurrent_records(tmp_path):
    def write(index):
        trace_log(
            tmp_path,
            task="symbol",
            symbol_id=f"symbol-{index}",
            est_prompt_tokens=150,
            actual_prompt_tokens=140,
            output_tokens=20,
            latency_s=0.1,
            validation="ok",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(50)))
    files = list((tmp_path / "logs").glob("calls-????????.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(records) == 50
    assert {record["symbol_id"] for record in records} == {f"symbol-{i}" for i in range(50)}
    assert all(
        {
            "ts",
            "task",
            "symbol_id",
            "est_prompt_tokens",
            "actual_prompt_tokens",
            "output_tokens",
            "latency_s",
            "validation",
        }
        <= record.keys()
        for record in records
    )
    assert all(record["validation"] == "ok" for record in records)
    write(50)
    assert len(files[0].read_text().splitlines()) == 51
