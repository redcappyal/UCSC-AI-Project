"""Pure-function tests for wall-vs-racket event classification."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classify_events import ball_size_px, classify_events, clip_median_ball_size

FPS = 30.0


def make_row(frame, size=None):
    if size is None:
        return {"source_frame": frame, "detected": False, "width": "", "height": ""}
    return {
        "source_frame": frame,
        "detected": True,
        "width": f"{size:.3f}",
        "height": f"{size:.3f}",
    }


def make_results(sizes_by_frame, total_frames=120, default_size=20.0):
    return {
        frame: make_row(frame, sizes_by_frame.get(frame, default_size))
        for frame in range(total_frames)
    }


def make_hit(frame, impact_time=None):
    hit = {
        "hit_frame": frame,
        "timestamp_seconds": frame / FPS,
        "dv_magnitude": 500.0,
        "after_gap": False,
    }
    if impact_time is not None:
        hit["impact_time"] = impact_time
    return hit


def audio_peak(time_seconds, score):
    return {"time_seconds": time_seconds, "score": score, "rms": 0.3, "frame": int(time_seconds * FPS)}


def test_all_signals_agree_wall_and_racket():
    # Racket hit at frame 30 (big ball, quiet, long gap before it), wall hit
    # at frame 45 (small ball, loud, short gap after the racket hit).
    results = make_results({29: 32.0, 30: 32.0, 31: 32.0, 44: 8.0, 45: 8.0, 46: 8.0})
    hits = [make_hit(30), make_hit(45)]
    audio = [audio_peak(30 / FPS, 6.0), audio_peak(45 / FPS, 26.0)]

    classified = classify_events(hits, results, audio, FPS)

    racket, wall = classified[0], classified[1]
    assert wall["event_type"] == "wall"
    assert wall["wall_score"] > 0
    assert wall["signals"]["audio_score"] == 26.0
    assert wall["signals"]["size_ratio"] < 1.0
    assert wall["signals"]["gap_prev_s"] < 0.9

    assert racket["event_type"] == "racket"
    assert racket["wall_score"] < 0
    assert racket["signals"]["size_ratio"] > 1.0
    # First event has no previous event, so no timing vote.
    assert racket["signals"]["gap_prev_s"] is None
    assert racket["signals"]["gap_next_s"] == wall["signals"]["gap_prev_s"]


def test_no_audio_track_classifies_on_size_and_timing():
    results = make_results({44: 8.0, 45: 8.0, 46: 8.0})
    hits = [make_hit(30), make_hit(45)]

    classified = classify_events(hits, results, None, FPS)

    wall = classified[1]
    assert wall["signals"]["audio_score"] is None
    assert wall["event_type"] == "wall"


def test_audio_present_but_unmatched_votes_racket():
    results = make_results({})
    hits = [make_hit(30)]

    # Audio track exists, but its only peak is far from the event.
    classified = classify_events(hits, results, [audio_peak(3.5, 25.0)], FPS)

    entry = classified[0]
    assert entry["signals"]["audio_score"] is None
    # size_ratio ~1.0 votes 0; the unmatched-audio -0.5 vote dominates.
    assert entry["wall_score"] < 0


def test_single_event_has_no_timing_signal():
    classified = classify_events([make_hit(30)], make_results({}), None, FPS)
    signals = classified[0]["signals"]
    assert signals["gap_prev_s"] is None
    assert signals["gap_next_s"] is None


def test_no_detections_near_event_has_no_size_signal():
    results = make_results({}, total_frames=120)
    for frame in range(27, 34):
        results[frame] = make_row(frame)  # not detected

    classified = classify_events([make_hit(30)], results, None, FPS)
    assert classified[0]["signals"]["ball_size_px"] is None
    assert classified[0]["signals"]["size_ratio"] is None


def test_no_signals_at_all_is_unknown():
    results = {frame: make_row(frame) for frame in range(120)}

    classified = classify_events([make_hit(30)], results, None, FPS)

    assert classified[0]["event_type"] == "unknown"
    assert classified[0]["wall_score"] is None


def test_input_hits_are_not_mutated():
    hits = [make_hit(30), make_hit(45)]
    snapshot = copy.deepcopy(hits)

    classify_events(hits, make_results({}), [audio_peak(1.0, 20.0)], FPS)

    assert hits == snapshot


def test_impact_time_preferred_for_audio_matching():
    results = make_results({})
    # Event's nominal frame time is 1.0 s but the fitted impact is at 1.3 s;
    # a peak at 1.3 s must match, one at 1.0 s must not.
    hits = [make_hit(30, impact_time=1.3)]

    classified = classify_events(hits, results, [audio_peak(1.3, 25.0)], FPS)
    assert classified[0]["signals"]["audio_score"] == 25.0

    classified = classify_events(hits, results, [audio_peak(1.0, 25.0)], FPS)
    assert classified[0]["signals"]["audio_score"] is None


def test_size_helpers():
    results = make_results({29: 10.0, 30: 12.0, 31: 14.0})
    assert ball_size_px(results, 30, 1) == 12.0
    assert ball_size_px({}, 30, 2) is None
    assert clip_median_ball_size(results) == 20.0
