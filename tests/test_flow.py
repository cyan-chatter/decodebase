import pytest

from cfl.config import Settings
from cfl.core import db, mermaid
from cfl.core.client import OllamaClient
from cfl.engines.flow import build_trace, flatten, flow
from cfl.pipeline.scan import run_stage1, run_stage2
from cfl.pipeline.symbol_pass import run_symbol_pass
from tests.fakes import FakeOllama


@pytest.mark.db
def test_flow_source_order_cycles_conditions_and_budget(ready):
    conn, client, settings, fake = ready
    root = next(s for s in db.get_all_symbols(conn) if s["name"] == "run_pipeline")
    tree = build_trace(conn, root["id"], 4)
    lines = [n["line"] for n in tree["children"]]
    assert lines == sorted(lines)
    assert all("source" in n and "confidence" in n and "control_ctx" in n for n in tree["children"])
    even = next(s for s in db.get_all_symbols(conn) if s["name"] == "is_even")
    assert any(n.get("recursion") for n in flatten(build_trace(conn, even["id"], 10)))
    result = flow(conn, client, settings, "run_pipeline")
    assert result["sufficient"] and not mermaid.validate_mermaid(result["diagram"])
    assert "| Call site |" in result["text"] and "```mermaid" in result["text"]
    assert all(
        node["symbol"][field] is None
        for node in flatten(result["tree"])
        if "symbol" in node
        for field in ("summary_short", "summary_long", "summary_json")
    )
    for call in fake.call_log:
        if call["path"] == "/api/chat":
            import json

            count = client.counter.count(json.dumps(call["body"]["messages"], ensure_ascii=False))
            assert (
                count + call["body"]["options"]["num_predict"] + settings.template_overhead
                <= settings.num_ctx
            )
            assert "[CODE]" not in call["body"]["messages"][0]["content"]


@pytest.mark.db
def test_large_flow_groups_fit_exact_chat_envelope(pg_conn, tmp_path):
    import json

    code = "".join(f"def f{i}(x):\n    y=x+{i}\n    return y\n\n" for i in range(50))
    code += "def entry(x):\n" + "".join(f"    f{i}(x)\n" for i in range(50)) + "    return x\n"
    (tmp_path / "a.py").write_text(code)
    settings = Settings(
        num_ctx=2048, num_predict_detailed=350, flow_max_nodes=40, state_dir=str(tmp_path / ".cfl")
    )
    fake = FakeOllama()
    fake.source_answers = True
    with OllamaClient(settings, transport=fake.transport) as client:
        run_stage1(pg_conn, settings, str(tmp_path))
        run_stage2(pg_conn, settings, str(tmp_path))
        run_symbol_pass(pg_conn, client, settings)
        result = flow(pg_conn, client, settings, "entry", depth=4)
        assert result["status"] == "abstained" and mermaid.count_nodes(result["tree"]) <= 40
        assert "cannot provide a reliable answer" in result["text"]
        calls = [c["body"] for c in fake.call_log if c["path"] == "/api/chat"]
        assert len(calls) > 2
        for call in calls:
            count = client.counter.count(json.dumps(call["messages"], ensure_ascii=False))
            assert (
                count + call["options"]["num_predict"] + settings.template_overhead
                <= settings.num_ctx
            )


@pytest.mark.db
def test_explicit_path_keeps_intermediate_evidence_beyond_default_depth(ready):
    from cfl.engines.ask import answer

    conn, client, settings, _ = ready
    result = answer(conn, client, settings, "Trace main to the database write.")
    ids = {b["id"] for b in result["blocks"]}
    assert {
        "cli.py::main",
        "pipeline.py::run_pipeline",
        "pipeline.py::transform",
        "pipeline.py::write_to_db",
        "db/repository.py::Repository.save",
    } <= ids
    assert result["sufficient"]


@pytest.mark.db
def test_conditional_recursion_is_not_an_unconditional_call(ready):
    conn, _, _, _ = ready
    root = next(s for s in db.get_all_symbols(conn) if s["name"] == "is_even")
    trace = build_trace(conn, root["id"], 6)
    assert trace["children"][0]["control_ctx"] == "if not (value == 0)"
    assert trace["children"][0]["children"][0]["recursion"]
