from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_auto_calibration_drag_handles_have_stable_numbers_and_guide():
    assert "Drag each numbered circle to the matching court point." in INDEX_HTML
    assert "number:i + 1" in INDEX_HTML
    assert "number:landmark.id === 'short_line_left' ? 5 : 6" in INDEX_HTML
    assert "ctx.fillText(String(anchor.number)" in INDEX_HTML
    assert "${anchor.number}.</b> ${anchor.guide}" in INDEX_HTML


def test_six_point_wall_drags_refit_every_front_wall_line():
    assert "function refitFrontWallLinesFromCorners()" in INDEX_HTML
    assert "refitFrontWallLinesFromCorners();" in INDEX_HTML
    assert "front_wall_line_heights_ft" in INDEX_HTML
    assert "x_span_source: f.wallHomography ? 'wall_homography'" in INDEX_HTML


def test_judging_page_reserves_an_enlarged_video_stage():
    assert "const TRACK_MIN_STAGE_PX = 500;" in INDEX_HTML
    assert "window.innerHeight - chrome - TRACK_MIN_STAGE_PX" in INDEX_HTML
