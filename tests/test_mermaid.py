from cfl.core import mermaid


def test_mermaid_escaping_caps_sequence_and_invalid_syntax():
    tree = {
        "id": "a",
        "label": 'a["x"]|<b>',
        "children": [{"id": str(i), "label": str(i), "children": []} for i in range(100)],
    }
    capped = mermaid.collapse(tree, 40, 4)
    assert mermaid.count_nodes(capped) <= 40
    assert any(n["kind"] == "collapsed" for n in capped["children"] if "kind" in n)
    for render in (mermaid.render_flowchart, mermaid.render_sequence):
        diagram, _ = render(capped)
        assert not mermaid.validate_mermaid(diagram)
    for diagram in (
        "unknown",
        'flowchart TD\nn0[""]',
        'flowchart TD\nn0["a"]\nn0["b"]',
        "flowchart TD\nn0 --> n1",
    ):
        assert mermaid.validate_mermaid(diagram)
