"""Fusion engine tests: synthetic rallies, sequence grammar, audio repetition."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_events import repeating_impact_windows
from event_engine import (
    _emission_scores,
    _emission_scores_3d,
    decode_sequence,
    detect_events_fused,
    make_wall_region,
    merge_fusion_config,
)
from job_runner import judge_hits

FPS = 30.0

CALIBRATION = {
    "lines": [
        {"name": "out_line_lower_edge", "endpoints": [[0, 100], [2000, 100]]},
        {"name": "tin_top_edge", "endpoints": [[0, 700], [2000, 700]]},
    ]
}


def rows_from_segments(start, segments, end_time, size_at=None):
    """Piecewise-constant-velocity path: segments = [(t_break, vx, vy), ...]
    where each velocity applies from its break time until the next.
    size_at(t) supplies the apparent ball size in px (default: constant 9)."""
    rows = []
    frame = 0
    while frame / FPS <= end_time:
        t = frame / FPS
        x, y = start
        for index, (break_time, vx, vy) in enumerate(segments):
            segment_end = segments[index + 1][0] if index + 1 < len(segments) else end_time
            lo, hi = break_time, min(t, segment_end)
            if hi > lo:
                x += vx * (hi - lo)
                y += vy * (hi - lo)
        size = 9.0 if size_at is None else float(size_at(t))
        rows.append(
            {
                "source_frame": frame,
                "timestamp_seconds": f"{t:.6f}",
                "detected": "True",
                "x_center": f"{x:.3f}",
                "y_center": f"{y:.3f}",
                "width": f"{size:.1f}",
                "height": f"{size:.1f}",
            }
        )
        frame += 1
    return rows


def size_profile(racket_times, wall_times, racket_size=22.0, wall_size=6.0, base=9.0):
    """Ball looks big near the striking player, small at the far wall."""
    def size_at(t):
        if any(abs(t - rt) <= 0.15 for rt in racket_times):
            return racket_size
        if any(abs(t - wt) <= 0.15 for wt in wall_times):
            return wall_size
        return base
    return size_at


def window_at(t, cluster_id=0, cluster_size=3, score=9.0):
    frame = int(round(t * FPS))
    return {
        "frame": frame,
        "time_seconds": t,
        "window_start_frame": frame - 2,
        "window_end_frame": frame + 2,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "score": score,
        "rms": 0.08,
    }


def labels_by_time(hits):
    return [hit["event_type"] for hit in sorted(hits, key=lambda h: h["hit_frame"])]


def test_full_rally_racket_wall_floor_racket_wall():
    # racket(0.6) -> wall(1.2, near out line) -> floor(1.8, vy flips up)
    # -> racket(2.4, speed gain) -> wall(3.0). Audio windows on walls only.
    rows = rows_from_segments(
        (100.0, 300.0),
        [
            (0.0, 80.0, 120.0),
            (0.6, 600.0, -330.0),
            (1.2, -450.0, 780.0),
            (1.8, -350.0, -420.0),
            (2.4, 650.0, -350.0),
            (3.0, -400.0, 300.0),
        ],
        end_time=3.4,
        size_at=size_profile(racket_times=[0.6, 2.4], wall_times=[1.2, 3.0]),
    )
    hits = detect_events_fused(
        rows,
        audio_windows=[window_at(1.2), window_at(3.0)],
        calibration=CALIBRATION,
    )
    assert labels_by_time(hits) == ["racket", "wall", "floor", "racket", "wall"]
    expected_frames = [18, 36, 54, 72, 90]
    for hit, expected in zip(sorted(hits, key=lambda h: h["hit_frame"]), expected_frames):
        assert abs(hit["hit_frame"] - expected) <= 3
    walls = [hit for hit in hits if hit["event_type"] == "wall"]
    assert all("derivative" in hit["methods"] for hit in walls)
    assert all(hit["signals"]["audio_cluster"] == 0 for hit in walls)


def test_volley_rally_skips_floor():
    # racket -> wall -> racket (volley) -> wall: no floor bounce anywhere.
    rows = rows_from_segments(
        (100.0, 300.0),
        [
            (0.0, 80.0, 120.0),
            (0.6, 600.0, -330.0),
            (1.2, -450.0, 300.0),
            (1.8, 700.0, -280.0),
            (2.4, -500.0, 250.0),
        ],
        end_time=2.8,
        size_at=size_profile(racket_times=[0.6, 1.8], wall_times=[1.2, 2.4]),
    )
    hits = detect_events_fused(
        rows,
        audio_windows=[window_at(1.2), window_at(2.4)],
        calibration=CALIBRATION,
    )
    assert labels_by_time(hits) == ["racket", "wall", "racket", "wall"]


def test_grammar_blocks_racket_then_floor():
    # Middle event slightly prefers floor, but racket -> floor is illegal
    # squash (a floor bounce only follows a wall hit), so wall must win.
    cfg = merge_fusion_config(None)
    emissions = [
        {"racket": 1.0, "wall": 0.0, "floor": 0.0, "side": -9.0},
        {"racket": 0.0, "wall": 0.5, "floor": 0.8, "side": -9.0},
        {"racket": 1.0, "wall": 0.0, "floor": 0.0, "side": -9.0},
    ]
    assert decode_sequence(emissions, cfg) == ["racket", "wall", "racket"]


def test_skip_state_absorbs_weak_phantom_event():
    # A weak event between two strong ones must become noise, not force a
    # grammar phase shift (the Bay Club phantom audio-window failure).
    cfg = merge_fusion_config(None)
    emissions = [
        {"racket": 1.5, "wall": 0.0, "floor": 0.0, "side": -9.0},
        {"racket": 0.25, "wall": 0.25, "floor": -0.25, "side": -9.0},
        {"racket": 0.0, "wall": 1.5, "floor": 0.0, "side": -9.0},
        {"racket": 0.0, "wall": 0.0, "floor": 1.5, "side": -9.0},
    ]
    assert decode_sequence(emissions, cfg) == ["racket", "none", "wall", "floor"]


def test_lone_unclaimed_audio_window_is_dropped_as_noise():
    # No trajectory and no grammar pressure: a lone audio window is more
    # likely a squeak or shout than an event (Bay Club GT: frames 461-490).
    hits = detect_events_fused([], audio_windows=[window_at(2.0)], calibration=CALIBRATION)
    assert hits == []


def test_grammar_gap_pulls_audio_window_in_as_wall():
    # racket ... floor is illegal without a wall between; the unclaimed
    # audio window in the gap must be promoted to the missing wall hit.
    rows = rows_from_segments(
        (150.0, 400.0),
        [
            (0.0, 100.0, -100.0),
            (0.6, 550.0, 300.0),   # racket: sharp turn, big speed gain
            (1.8, 450.0, -250.0),  # floor: vertical flip, no energy gain
        ],
        end_time=2.4,
        size_at=size_profile(racket_times=[0.6], wall_times=[]),
    )
    hits = detect_events_fused(
        rows, audio_windows=[window_at(1.2)], calibration=CALIBRATION
    )
    assert labels_by_time(hits) == ["racket", "wall", "floor"]
    wall = next(h for h in hits if h["event_type"] == "wall")
    assert wall["source"] == "audio"
    assert wall["hit_frame"] == 36


def base_event(**overrides):
    event = {
        "x": 500.0,
        "y": 400.0,
        "v_in": np.array([300.0, 400.0]),
        "v_out": np.array([280.0, -380.0]),
        "speed_before": 500.0,
        "speed_after": 472.0,
        "audio_window": None,
        "size_ratio": 1.0,
    }
    event.update(overrides)
    return event


def test_size_ratio_separates_low_racket_from_floor():
    # Identical falling->rising kinematics; only the apparent ball size
    # differs. Large ball = near the striking player = racket; small ball
    # = far away = floor bounce.
    cfg = merge_fusion_config(None)
    racket_strike = base_event(
        size_ratio=2.4,
        v_out=np.array([500.0, -600.0]),
        speed_after=781.0,  # racket adds energy
    )
    floor_bounce = base_event(size_ratio=0.9)
    racket_scores = _emission_scores(racket_strike, False, None, cfg)
    floor_scores = _emission_scores(floor_bounce, False, None, cfg)
    assert max(racket_scores, key=racket_scores.get) == "racket"
    assert max(floor_scores, key=floor_scores.get) == "floor"


def test_wall_region_vetoes_wall_below_tin():
    # An event well below the tin line cannot be a wall hit, even when an
    # audio window matched it (the frame-248/362 mislabel case).
    cfg = merge_fusion_config(None)
    region = make_wall_region(CALIBRATION, cfg)
    assert region.wall(500.0, 400.0)          # between the lines
    assert region.wall(500.0, 745.0)          # tin-face hit: below, within pad
    assert not region.wall(500.0, 820.0)      # floor territory
    assert region.in_x_span(500.0)
    assert not region.in_x_span(2200.0)       # side-wall territory
    below_tin = base_event(y=820.0, audio_window=window_at(2.0))
    scores = _emission_scores(below_tin, True, region, cfg)
    assert scores["wall"] < scores["floor"]
    assert max(scores, key=scores.get) != "wall"


def test_repeating_waveform_windows_ignore_one_off_sounds():
    # Three identical damped-sine impacts at very different volumes must
    # cluster together; a one-off broadband burst must not become a window.
    rng = np.random.default_rng(7)
    sample_rate = 16000
    samples = rng.normal(0, 0.002, sample_rate * 6).astype(np.float64)
    burst_t = np.arange(int(0.030 * sample_rate)) / sample_rate
    impact = np.sin(2 * np.pi * 1800 * burst_t) * np.exp(-burst_t / 0.006)
    for time, amplitude in [(1.0, 1.0), (2.0, 0.35), (3.0, 0.7)]:
        start = int(time * sample_rate)
        samples[start : start + len(impact)] += amplitude * impact
    noise_burst = rng.normal(0, 1.0, len(impact))
    samples[int(4.5 * sample_rate) : int(4.5 * sample_rate) + len(impact)] += noise_burst

    windows = repeating_impact_windows(sample_rate, samples, 0, int(6 * FPS), FPS)
    times = sorted(round(w["time_seconds"], 1) for w in windows)
    assert times == [1.0, 2.0, 3.0]
    assert all(w["cluster_size"] == 3 for w in windows)


def test_camera_none_is_default_behavior():
    # Passing camera=None must equal not passing it at all.
    rows = rows_from_segments(
        (100.0, 300.0),
        [
            (0.0, 80.0, 120.0),
            (0.6, 600.0, -330.0),
            (1.2, -450.0, 780.0),
        ],
        end_time=1.6,
        size_at=size_profile(racket_times=[0.6], wall_times=[1.2]),
    )
    baseline = detect_events_fused(rows, calibration=CALIBRATION)
    explicit = detect_events_fused(rows, calibration=CALIBRATION, camera=None)
    assert baseline == explicit


def test_ballistic_source_used_with_camera():
    from synthetic3d import make_camera
    from tests_ballistic_helpers import make_bounce_rows

    camera = make_camera()
    rows, expected_frame = make_bounce_rows(camera)
    hits = detect_events_fused(rows, camera=camera)
    assert any("ballistic" in hit["methods"] for hit in hits)
    matched = [h for h in hits if abs(h["hit_frame"] - expected_frame) <= 3]
    assert matched and "contact_3d" in matched[0]


def test_judge_labels_floor_bounce(tmp_path):
    results = {
        54: {
            "source_frame": 54,
            "timestamp_seconds": "1.800000",
            "detected": "True",
            "x_center": "238.000",
            "y_center": "642.000",
        }
    }
    hit = {
        "hit_frame": 54,
        "timestamp_seconds": 1.8,
        "event_type": "floor",
        "score": 1.5,
        "dv_magnitude": 1200.0,
        "after_gap": False,
    }
    entries = judge_hits(tmp_path, results, [hit])
    # Verdicts apply to front-wall hits only: no call, just the classification.
    assert entries[0]["call"] is None
    assert entries[0]["reason"] == "classified_as_floor"


def _contact_event(point_ft, v_in, v_out):
    return {
        "x": 900.0, "y": 500.0, "time": 1.0, "frame": 60, "index": 10,
        "v_in": np.array([0.0, 0.0]), "v_out": np.array([0.0, 0.0]),
        "speed_before": 1.0, "speed_after": 1.0,
        "methods": {"ballistic"}, "audio_window": None, "size_ratio": None,
        "contact_3d": {
            "time": 1.0, "point_ft": list(point_ft),
            "v_in_ft_s": list(v_in), "v_out_ft_s": list(v_out),
            "arc_rms_px": [0.5, 0.5],
        },
    }


def test_3d_emissions_floor_bounce():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((10.0, 15.0, 0.2), (5.0, -20.0, -18.0), (4.0, -16.0, 12.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["floor"] == max(scores.values())


def test_3d_emissions_front_wall_bounce():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((10.0, 0.4, 6.0), (2.0, -60.0, 4.0), (1.5, 40.0, 1.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["wall"] == max(scores.values())


def test_3d_emissions_racket_interior_energy_gain():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((10.0, 25.0, 4.0), (3.0, 30.0, -4.0), (2.0, -70.0, 10.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["racket"] == max(scores.values())


def test_3d_emissions_side_wall():
    from synthetic3d import make_camera
    camera = make_camera()
    cfg = merge_fusion_config(None)
    event = _contact_event((0.3, 12.0, 5.0), (-30.0, -30.0, 2.0), (22.0, -26.0, 1.0))
    scores = _emission_scores_3d(event, False, camera, cfg)
    assert scores["side"] == max(scores.values())
