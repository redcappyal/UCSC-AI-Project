from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_auto_calibration_drag_handles_have_stable_numbers_and_guide():
    assert "Drag each numbered circle to the matching court point." in INDEX_HTML
    assert "number:i + 1" in INDEX_HTML
    assert "number:landmark.id === 'short_line_left' ? 5 : 6" in INDEX_HTML
    assert "ctx.fillText(String(anchor.number)" in INDEX_HTML
    assert "${anchor.number}.</b> ${anchor.guide}" in INDEX_HTML
