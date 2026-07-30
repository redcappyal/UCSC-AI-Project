from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "index.html"
).read_text(encoding="utf-8")


def test_analysis_session_list_uses_larger_scoped_typography():
    assert "#p-matches .viewhead h2{font-size:26px}" in INDEX_HTML
    assert "#p-matches .viewhead .dim{font-size:15px}" in INDEX_HTML
    assert "#p-matches .cliptop .name{font-size:18px}" in INDEX_HTML
    assert "#p-matches .cliptop .metaline{font-size:15px}" in INDEX_HTML
    assert "#p-matches .statechip{font-size:12px; padding:7px 13px}" in INDEX_HTML
    assert "#p-matches .emptycard{font-size:16px}" in INDEX_HTML
