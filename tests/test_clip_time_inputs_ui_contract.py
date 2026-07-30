from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "index.html"
).read_text(encoding="utf-8")


def test_clip_start_and_end_times_are_editable_text_fields():
    assert 'id="clipStartOut" type="text" inputmode="decimal"' in INDEX_HTML
    assert 'id="clipEndOut" type="text" inputmode="decimal"' in INDEX_HTML
    assert 'aria-label="Clip start time in seconds"' in INDEX_HTML
    assert 'aria-label="Clip end time in seconds"' in INDEX_HTML


def test_edited_clip_times_update_the_selection_safely():
    assert "function parseClipTime(value)" in INDEX_HTML
    assert "function setClipEdge(which, requestedTime)" in INDEX_HTML
    assert "function commitClipTime(which)" in INDEX_HTML
    assert "input.addEventListener('blur', () => commitClipTime(which))" in INDEX_HTML
    assert "S.clip.end - minClip()" in INDEX_HTML
    assert "S.clip.start + minClip()" in INDEX_HTML
