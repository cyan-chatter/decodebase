from __future__ import annotations

import ast
import json
import math
from itertools import pairwise

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from cfl.config import Settings
from cfl.core.budget import (
    PromptPart,
    TokenCounter,
    assemble,
    assert_fits,
    batch_by_budget,
    cap_tool_output,
    combine_prompt,
    output_reserve,
    split_oversized,
)
from cfl.core.errors import BudgetExceeded
from cfl.parser.python_adapter import PythonAdapter


@pytest.fixture
def counter(tmp_path):
    return TokenCounter(Settings(state_dir=str(tmp_path)))


def test_estimate_and_calibration(counter, caplog):
    assert counter.count("x" * 100) == math.ceil(100 / 3.2 * 1.15)
    counter.calibrate(320, 10)  # Cached prefix: ignore low count.
    assert counter.ratio == 3.2
    assert not counter.calibration_path.exists()
    counter.calibrate(320, 200)
    assert counter.ratio == pytest.approx(3.04)
    assert "drift" in caplog.text.lower()
    restored = TokenCounter(Settings(state_dir=str(counter.calibration_path.parent)))
    assert restored.ratio == counter.ratio
    assert json.loads(counter.calibration_path.read_text())["ratio"] == counter.ratio


def test_exact_tokenizer(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    file = tmp_path / "tokenizer.json"
    tokenizer.save(str(file))
    counter = TokenCounter(Settings(state_dir=str(tmp_path), tokenizer_file=str(file)))
    assert counter.count("hello world unknown") == 3
    assert counter.count("") == 0


def test_corrupt_calibration_and_tokenizer_fall_back(tmp_path):
    (tmp_path / "calibration.json").write_text("{broken")
    file = tmp_path / "tokenizer.json"
    file.write_text("{}")
    counter = TokenCounter(Settings(state_dir=str(tmp_path), tokenizer_file=str(file)))
    assert counter.count("x" * 32) == 12


def test_priority_and_fixed_reservations(counter):
    parts = [
        PromptPart("code", "target code", 1, False),
        PromptPart("callees", "c" * 50, 2, False),
        PromptPart("callers", "r" * 50, 3, False),
        PromptPart("signature", "s" * 50, 4, False),
    ]
    result = assemble("fixed", parts, num_ctx=55, num_predict=10, overhead=5, counter=counter)
    assert result.dropped == ["signature", "callers"]
    assert [part.name for part in result.parts] == ["code", "callees"]
    assert result.prompt_tokens == counter.count(result.prompt) + 5
    assert result.prompt_tokens + 10 <= 55


def test_code_truncation_keeps_head_tail(counter):
    text = "\n".join(f"line {i}: " + "x" * 30 for i in range(100))
    result = assemble(
        "fixed",
        [PromptPart("code", text, 1, True, 20)],
        num_ctx=120,
        num_predict=20,
        overhead=5,
        counter=counter,
    )
    assert result.truncated == ["code"]
    assert "line 0:" in result.prompt and "line 99:" in result.prompt
    assert "lines omitted]" in result.prompt
    assert result.prompt_tokens <= 100


@pytest.mark.parametrize("tokens,output,context", [(100, 1, 100), (1, 100, 100)])
def test_hard_budget(tokens, output, context):
    with pytest.raises(BudgetExceeded):
        assert_fits(tokens, output, context)
    assert_fits(80, 20, 100)


def test_required_parts_never_silently_dropped(counter):
    with pytest.raises(BudgetExceeded):
        assemble("fixed" * 20, [], num_ctx=10, num_predict=2, overhead=0, counter=counter)
    with pytest.raises(BudgetExceeded):
        assemble(
            "",
            [PromptPart("code", "x" * 1000, 1, False)],
            num_ctx=100,
            num_predict=20,
            overhead=0,
            counter=counter,
        )
    with pytest.raises(BudgetExceeded):
        assemble(
            "",
            [PromptPart("code", "x" * 1000, 1, True, 100)],
            num_ctx=100,
            num_predict=20,
            overhead=0,
            counter=counter,
        )


def test_split_at_complete_ast_statements(counter):
    raw = (
        "@decorator\ndef run(value):\n"
        + "".join(f"    value = transform_{i}(\n        value, {i}\n    )\n" for i in range(8))
        + "    return value\n"
    )
    chunks = split_oversized(
        {"raw_code": raw, "signature": "run(value)", "start_line": 20},
        PythonAdapter(),
        80,
        counter=counter,
    )
    assert len(chunks) > 1
    counts = []
    for chunk in chunks:
        assert chunk.token_est <= 80
        tree = ast.parse(chunk.text)
        counts.extend(ast.unparse(statement) for statement in tree.body[0].body)
        assert chunk.text.startswith("@decorator\ndef run(value):")
        assert chunk.signature == "run(value)"
    assert len(counts) == 9 and len(set(counts)) == 9
    assert chunks[0].start_line == 22
    assert chunks[-1].end_line == 46
    assert all(a.end_line + 1 == b.start_line for a, b in pairwise(chunks))
    prompt = combine_prompt("run(value)", ["first", "second"], budget=100, counter=counter)
    assert "Partial 2: second" in prompt and "run(value)" in prompt
    with pytest.raises(BudgetExceeded):
        combine_prompt("run(value)", ["x" * 1000], budget=20, counter=counter)
    with pytest.raises(BudgetExceeded):
        split_oversized(
            {"raw_code": 'def run():\n    return "' + "x" * 1000 + '"'},
            PythonAdapter(),
            80,
            counter=counter,
        )


def test_batching_and_output_cap(counter):
    assert batch_by_budget([2, 3, 4, 1], lambda n: n, 5) == [[2, 3], [4, 1]]
    assert batch_by_budget([], lambda n: n, 5) == []
    with pytest.raises(BudgetExceeded):
        batch_by_budget([6], lambda n: n, 5)
    text = "\n".join(f"item {i}: " + "x" * 30 for i in range(50))
    result = cap_tool_output(text, 100, pointer="Read source: path:1-50", counter=counter)
    assert "item 0:" in result and "item 49:" in result
    assert result.endswith("Read source: path:1-50") and counter.count(result) <= 100
    assert cap_tool_output("small", 100, counter=counter) == "small"
    with pytest.raises(BudgetExceeded):
        cap_tool_output(text, 1, counter=counter)
    assert output_reserve("brief") == 120
    assert output_reserve("detailed") == 1200
    assert output_reserve("symbol", Settings(num_predict_symbol=400)) == 400


@pytest.mark.parametrize(
    "values", [(1.5, 1, 100), (1, float("nan"), 100), (True, 1, 100), (-1, 1, 100), (0, 0, 0)]
)
def test_invalid_budgets_rejected(values):
    with pytest.raises(ValueError):
        assert_fits(*values)
