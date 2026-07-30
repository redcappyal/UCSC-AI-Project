"""The Clip step's select-entire-clip shortcut.

Trimming defaults to a ~4 s window around the calibration frame, so analyzing a
whole recording meant dragging a handle across the strip or riding the +1 s
steppers to both ends. One full-width secondary button under the nudge rows now
sets the selection to the entire clip in a single tap.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_select_entire_clip_button_sits_between_nudge_rows_and_frame_summary():
    """The button lives under the Start/End nudge rows, above the frame summary."""
    end_nudge_row = INDEX_HTML.index('id="nEndP1s"')  # last control of the End row
    button = INDEX_HTML.index('id="clipSelectAllBtn"')
    frame_summary = INDEX_HTML.index('class="clipFrameSummary"')
    assert end_nudge_row < button < frame_summary
    # Default <button> styling is the DESIGN.md §8.1 Secondary variant
    # (full-width surface pill); a variant class here would change that.
    assert '<button type="button" id="clipSelectAllBtn">Select entire clip</button>' in INDEX_HTML


def test_select_entire_clip_selects_time_zero_through_duration():
    assert "function selectEntireClip()" in INDEX_HTML
    assert "S.clip.start = 0;" in INDEX_HTML
    assert "S.clip.end = clipDuration();" in INDEX_HTML
    assert "$('clipSelectAllBtn').onclick = selectEntireClip;" in INDEX_HTML
