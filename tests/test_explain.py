import pytest

from cfl.core import db
from cfl.engines.explain import explain
from tests.test_ask import generations


@pytest.mark.db
def test_detailed_explanation_cache_invalidates_on_resummarize(ready):
    conn, client, settings, fake = ready
    first = explain(conn, client, settings, "with_retries", "detailed")
    assert first["sufficient"] and not first["cached"]
    before = generations(fake)
    assert explain(conn, client, settings, "with_retries", "detailed")["cached"]
    assert generations(fake) == before
    symbol = db.get_symbol(conn, "utils/retries.py::with_retries")
    if symbol is None:
        symbol = next(s for s in db.get_all_symbols(conn) if s["name"] == "with_retries")
    db.save_symbol_summary(
        conn, symbol["id"], symbol["summary_json"], symbol["summary_short"], None, "changed"
    )
    assert not explain(conn, client, settings, "with_retries", "detailed")["cached"]
