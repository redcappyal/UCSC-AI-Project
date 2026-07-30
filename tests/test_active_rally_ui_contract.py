from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "index.html"
).read_text(encoding="utf-8")


def test_current_playback_rally_has_a_strong_highlight():
    assert "#rallyRibbon .rallySegment.active" in INDEX_HTML
    assert "#rallyRibbon.hasActive .rallySegment:not(.active){opacity:.52}" in INDEX_HTML
    assert "box-shadow:inset 0 0 0 3px var(--text)" in INDEX_HTML


def test_active_rally_uses_true_segmentation_bounds():
    assert "segment.dataset.start = Number(rally.start_frame)" in INDEX_HTML
    assert "segment.dataset.end = Number(rally.end_frame)" in INDEX_HTML
    assert "const active = frame >= start && frame <= end" in INDEX_HTML
    assert "el.setAttribute('aria-current', 'true')" in INDEX_HTML
    assert "$('rallyRibbon').classList.toggle('hasActive', hasActive)" in INDEX_HTML


def test_playback_updates_active_rally_continuously():
    assert "updateTrackReadout(frame);" in INDEX_HTML
    assert "syncActiveRally(frame);" in INDEX_HTML
