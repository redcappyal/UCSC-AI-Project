from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_auto_calibration_drag_handles_have_stable_numbers():
    assert "number:i + 1" in INDEX_HTML
    assert "number:landmark.id === 'short_line_left' ? 5 : 6" in INDEX_HTML
    assert "ctx.fillText(String(anchor.number)" in INDEX_HTML


def test_confirm_screen_spends_its_height_on_the_frame():
    """The confirm screen exists to show whether the overlay sits on the real
    paint. Its controls used to leave the frame 185x104 CSS px at 390x844 --
    the numbered anchor guide alone was 300 px restating the pucks."""
    assert "Drag each numbered circle to the matching court point." not in INDEX_HTML
    assert 'id="confirmAnchorList"' not in INDEX_HTML
    assert 'id="confirmAnchorHelp"' not in INDEX_HTML
    assert 'id="confirmCount"' not in INDEX_HTML


def test_confirm_drift_warning_cannot_reflow_the_frame():
    """#confirmDrift is the only confirm-screen copy a drag rewrites. In flow it
    re-centered the canvas under the finger, so it is pinned inside #stage where
    it has no layout height at all."""
    stage = INDEX_HTML.split('<div id="stage"', 1)[1].split("<main>", 1)[0]
    assert 'id="confirmDrift"' in stage
    section = INDEX_HTML.split('id="p-confirm"', 1)[1].split("</section>", 1)[0]
    assert 'id="confirmDrift"' not in section
    assert "#confirmDrift{position:absolute" in INDEX_HTML


def test_six_point_wall_drags_refit_every_front_wall_line():
    assert "function refitFrontWallLinesFromCorners()" in INDEX_HTML
    assert "refitFrontWallLinesFromCorners();" in INDEX_HTML
    assert "front_wall_line_heights_ft" in INDEX_HTML
    assert "x_span_source: f.wallHomography ? 'wall_homography'" in INDEX_HTML


def test_judging_page_reserves_an_enlarged_video_stage():
    assert "const TRACK_MIN_STAGE_PX = 500;" in INDEX_HTML
    assert "window.innerHeight - chrome - TRACK_MIN_STAGE_PX" in INDEX_HTML
