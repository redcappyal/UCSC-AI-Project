from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "index.html"
).read_text(encoding="utf-8")


def test_analysis_rows_include_an_accessible_trash_button():
    assert 'class="deleteRunBtn"' in INDEX_HTML
    assert 'data-delete-run="${escapeHtml(id)}"' in INDEX_HTML
    assert 'aria-label="Delete session ${escapeHtml(fmtRunDate(r.created))}"' in INDEX_HTML
    assert 'title="Delete session"' in INDEX_HTML


def test_delete_requires_confirmation_and_calls_the_delete_endpoint():
    assert "async function deleteRunSession(runId, button)" in INDEX_HTML
    assert "if(!window.confirm(" in INDEX_HTML
    assert "permanently removes its analysis and annotations" in INDEX_HTML
    assert "method:'DELETE'" in INDEX_HTML
    assert "LIVE.runs = LIVE.runs.filter" in INDEX_HTML
