"""Audio-driven recall: unmatched audio peaks rescue or synthesize hits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect_wall_hits import detect_hits_from_rows
from job_runner import judge_hits

FPS = 30.0


def rows_for(path_x, path_y, frames):
    return [
        {
            "source_frame": f,
            "timestamp_seconds": f"{f / FPS:.6f}",
            "detected": "True",
            "x_center": f"{path_x(f / FPS):.3f}",
            "y_center": f"{path_y(f / FPS):.3f}",
        }
        for f in frames
    ]


def weak_bounce_rows(frames=range(0, 120)):
    """Bounce at t=2.0s (frame 60) too gentle for is_significant():
    vx 400 -> 320 px/s is |dv| 80 (< 200) and a ~2 degree turn (< 25)."""
    def x(t):
        return 100 + 400 * t if t <= 2.0 else 100 + 400 * 2.0 + 320 * (t - 2.0)

    return rows_for(x, lambda t: 300 - 60 * t, frames)


def peak_at(t):
    return {"frame": int(round(t * FPS)), "time_seconds": t, "score": 12.0, "rms": 0.1}


def test_subthreshold_bounce_rescued_by_audio_peak():
    rows = weak_bounce_rows()
    assert detect_hits_from_rows(rows) == []

    hits = detect_hits_from_rows(rows, audio_candidates=[peak_at(2.0)])
    assert len(hits) == 1
    hit = hits[0]
    assert hit["audio_assisted"] is True
    assert hit["source"] == "audio_rescued"
    assert abs(hit["hit_frame"] - 60) <= 4
    # Promoted candidates carry full trajectory fields for judging.
    assert "dv_magnitude" in hit and "speed_before" in hit


def test_untracked_bounce_synthesizes_audio_only_hit():
    # Ball never detected around the impact: no candidate within tolerance.
    rows = weak_bounce_rows(frames=[f for f in range(0, 120) if not 48 <= f <= 72])
    hits = detect_hits_from_rows(rows, audio_candidates=[peak_at(2.0)])
    audio_only = [h for h in hits if h.get("source") == "audio"]
    assert len(audio_only) == 1
    assert audio_only[0]["hit_frame"] == 60
    assert audio_only[0]["timestamp_seconds"] == 2.0


def test_matched_peak_adds_no_duplicate_hit():
    # Strong bounce the trajectory detector already finds.
    def x(t):
        return 100 + 400 * t if t <= 2.0 else 100 + 400 * 2.0 - 400 * (t - 2.0)

    rows = rows_for(x, lambda t: 300 - 60 * t, range(0, 120))
    without_audio = detect_hits_from_rows(rows)
    with_audio = detect_hits_from_rows(rows, audio_candidates=[peak_at(2.0)])
    assert len(with_audio) == len(without_audio)
    assert not any(h.get("audio_assisted") for h in with_audio)


def test_peak_near_picked_hit_promotes_no_duplicate():
    # Strong bounce at frame 60; trajectory ends at frame 69. A peak at
    # frame 70 clears the anchor checks, but every trajectory candidate in
    # its window sits within min_gap of the picked hit — promoting one would
    # duplicate it, so the peak must be treated as already covered.
    def x(t):
        return 100 + 400 * t if t <= 2.0 else 100 + 400 * 2.0 - 400 * (t - 2.0)

    rows = rows_for(x, lambda t: 300 - 60 * t, range(0, 70))
    without_audio = detect_hits_from_rows(rows)
    assert any(abs(h["hit_frame"] - 60) <= 2 for h in without_audio)
    with_audio = detect_hits_from_rows(rows, audio_candidates=[peak_at(70 / FPS)])
    assert len(with_audio) == len(without_audio)


def test_judge_keeps_audio_only_hit_without_ball_detection(tmp_path):
    results = {
        10: {
            "source_frame": 10,
            "timestamp_seconds": f"{10 / FPS:.6f}",
            "detected": "True",
            "x_center": "500.000",
            "y_center": "200.000",
        }
    }
    hit = {
        "hit_frame": 60,
        "timestamp_seconds": 2.0,
        "score": None,
        "source": "audio",
        "audio_assisted": True,
    }
    entries = judge_hits(tmp_path, results, [hit])
    assert len(entries) == 1
    entry = entries[0]
    assert entry["frame"] == 60
    assert entry["timestamp_seconds"] == 2.0
    assert entry["audio_assisted"] is True
    assert entry["source"] == "audio"
    assert entry["call"] == "UNKNOWN"
    assert entry["dv_magnitude"] is None
