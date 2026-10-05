import pytest

from cfl.core import db
from cfl.engines.ask import block_for
from cfl.engines.verify import verify_answer


@pytest.mark.db
def test_verifier_source_ranges_stale_and_mermaid(ready):
    conn, _, _, _ = ready
    symbol = db.get_all_symbols(conn)[0]
    blocks = block_for(symbol)
    tag = f"[{symbol['file_path']}:{symbol['start_line']}-{symbol['end_line']}]"
    checked = verify_answer(
        "Source "
        + tag
        + " [missing.py:2-1] `ghost`\n```mermaid\nflowchart TD\nn0 --> missing\n```",
        blocks,
        conn,
    )
    assert checked.sufficient and len(checked.citations) == 1
    assert "Invalid Mermaid diagram omitted" in checked.text
    assert "Unknown identifier: ghost" in checked.warnings
    blocks[0]["code_hash"] = "stale"
    assert not verify_answer(tag, blocks, conn).sufficient


@pytest.mark.db
def test_known_file_and_module_qualified_symbol_are_not_unknown(ready):
    conn, _, _, _ = ready
    symbol = db.get_symbol(conn, "utils/retry.py::with_retries")
    tag = f"[{symbol['file_path']}:{symbol['start_line']}-{symbol['end_line']}]"
    checked = verify_answer(
        "`utils.retry.with_retries` in `utils/retry.py` " + tag, block_for(symbol), conn
    )
    assert checked.sufficient and not checked.warnings
    even = db.get_symbol(conn, "trees.py::is_even")
    tag = f"[{even['file_path']}:{even['start_line']}-{even['end_line']}]"
    assert not verify_answer("`trees.py` " + tag, block_for(even), conn).warnings
