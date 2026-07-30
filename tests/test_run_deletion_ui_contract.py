from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "index.html"
).read_text(encoding="utf-8")


def test_analysis_rows_include_an_accessible_trash_button():
    assert 'class="deleteRunBtn"' in INDEX_HTML
    assert 'data-delete-run="${escapeHtml(id)}"' in INDEX_HTML
    assert 'aria-label="Delete session ${escapeHtml(fmtRunDate(r.created))}"' in INDEX_HTML
    assert 'title="Delete session"' in INDEX_HTML


def test_delete_arms_then_confirms_and_calls_the_delete_endpoint():
    """The iOS shell's WKWebView has no dialog UI: window.confirm() silently
    returns false and window.alert() never shows. Deletion must therefore use
    the arm pattern (first tap flips the button to "Confirm", disarms after a
    beat) and report failures through the in-page error banner."""
    assert "async function deleteRunSession(runId, button)" in INDEX_HTML
    assert "window.confirm" not in INDEX_HTML
    assert "window.alert" not in INDEX_HTML
    assert "button.classList.add('arm')" in INDEX_HTML
    assert "button.textContent = 'Confirm'" in INDEX_HTML
    assert ", 2600)" in INDEX_HTML
    assert ".deleteRunBtn.arm" in INDEX_HTML
    assert "showError(error.message" in INDEX_HTML
    assert "method:'DELETE'" in INDEX_HTML
    assert "LIVE.runs = LIVE.runs.filter" in INDEX_HTML
