"""Synthetic-trajectory tests for the swappable bounce detectors."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect_wall_hits import (
    detect_bounce,
    detect_bounce_legacy,
    detect_bounce_two_stage,
    detect_hits_from_rows,
)

FPS = 30.0
CALIBRATION = {
    "lines": [
        {"name": "out_line_lower_edge", "endpoints": [[0, 100], [2000, 100]]},
        {"name": "tin_top_edge", "endpoints": [[0, 700], [2000, 700]]},
    ]
}


def trajectory_from_points(points):
    """(x, y) samples at consecutive frames -> (frames, timestamps, positions)."""
    frames = np.arange(len(points), dtype=np.int64)
    return frames, frames / FPS, np.array(points, dtype=np.float64)


def rows_from_trajectory(trajectory):
    frames, timestamps, positions = trajectory
    return [
        {
            "source_frame": int(frame),
            "timestamp_seconds": f"{t:.6f}",
            "detected": "True",
            "x_center": f"{x:.3f}",
            "y_center": f"{y:.3f}",
        }
        for frame, t, (x, y) in zip(frames, timestamps, positions)
    ]


def lofted_shot(*, vertex_t=1.0, n=60, noise=None):
    """V-shaped fall toward the tin line, bouncing back up at vertex_t."""
    points = []
    for f in range(n):
        t = f / FPS
        points.append((200 + 300 * t, 685 - 450 * abs(t - vertex_t)))
    frames, timestamps, positions = trajectory_from_points(points)
    if noise is not None:
        positions = positions + noise
    return (frames, timestamps, positions), (200 + 300 * vertex_t, 685.0, vertex_t)


def flat_drive(*, vertex_t=0.8, n=48):
    """Flat, hard drive: image-space velocity change too small for the legacy
    detector, but the distance to the tin line shrinks monotonically to impact."""
    points = []
    for f in range(n):
        t = f / FPS
        points.append((536 - 45 * abs(t - vertex_t), 684 - 30 * abs(t - vertex_t)))
    return trajectory_from_points(points)


def test_lofted_shot_both_methods_agree():
    trajectory, (_, _, impact_t) = lofted_shot()
    frames = trajectory[0]

    legacy = detect_bounce_legacy(trajectory, CALIBRATION)
    two_stage = detect_bounce_two_stage(trajectory, CALIBRATION)

    assert legacy is not None and two_stage is not None
    assert abs(int(frames[legacy.impact_index]) - int(frames[two_stage.impact_index])) <= 1
    assert abs(two_stage.impact_t - impact_t) < 1.0 / FPS
    assert legacy.method == "legacy_sign_flip"
    assert two_stage.method == "two_stage"


def test_flat_drive_found_by_two_stage_only():
    trajectory = flat_drive()

    assert detect_bounce_legacy(trajectory, CALIBRATION) is None

    result = detect_bounce_two_stage(trajectory, CALIBRATION)
    assert result is not None
    assert int(trajectory[0][result.impact_index]) == 24
    assert abs(result.impact_t - 0.8) < 1.0 / FPS
    assert result.diagnostics["nearest_line"] == "tin_top_edge"


def test_noise_robustness_two_stage_beats_legacy():
    rng = np.random.default_rng(7)
    legacy_errors, two_stage_errors = [], []

    for _ in range(20):
        noise = rng.normal(0.0, 2.0, size=(60, 2))
        trajectory, (impact_x, impact_y, _) = lofted_shot(noise=noise)

        two_stage = detect_bounce_two_stage(trajectory, CALIBRATION)
        assert two_stage is not None
        legacy = detect_bounce_legacy(trajectory, CALIBRATION)
        if legacy is None:
            continue

        legacy_errors.append(
            float(np.hypot(legacy.impact_xy[0] - impact_x, legacy.impact_xy[1] - impact_y))
        )
        two_stage_errors.append(
            float(np.hypot(two_stage.impact_xy[0] - impact_x, two_stage.impact_xy[1] - impact_y))
        )

    assert len(legacy_errors) >= 15
    assert np.mean(two_stage_errors) < np.mean(legacy_errors)


def test_straight_line_returns_none():
    points = [(200 + 400 * (f / FPS), 50 + 500 * (f / FPS)) for f in range(60)]
    trajectory = trajectory_from_points(points)

    diagnostics = {}
    assert detect_bounce_two_stage(trajectory, CALIBRATION, diagnostics_out=diagnostics) is None
    assert "rejected" in diagnostics

    rng = np.random.default_rng(3)
    frames, timestamps, positions = trajectory
    noisy = (frames, timestamps, positions + rng.normal(0.0, 2.0, size=positions.shape))
    assert detect_bounce_two_stage(noisy, CALIBRATION) is None


def test_short_trajectory_returns_none():
    trajectory = trajectory_from_points([(200 + 10 * f, 600 + 10 * f) for f in range(5)])
    diagnostics = {}
    assert detect_bounce_two_stage(trajectory, CALIBRATION, diagnostics_out=diagnostics) is None
    assert "rejected" in diagnostics


def test_subframe_impact_lands_between_frames():
    vertex_t = 24.5 / FPS  # exactly between frames 24 and 25
    trajectory = flat_drive(vertex_t=vertex_t)
    timestamps = trajectory[1]

    result = detect_bounce_two_stage(trajectory, CALIBRATION)
    assert result is not None
    assert timestamps[24] < result.impact_t < timestamps[25]
    assert abs(result.impact_t - vertex_t) < 0.5 / FPS


# The legacy pipeline output on this exact clip, captured before the
# refactor; the "legacy_sign_flip" switch must reproduce it byte for byte.
def legacy_reference_rows():
    """Copy of test_impact.bounce_rows(): x reverses at t=2.0, y keeps falling."""
    rows = []
    vx_in, vy = 400.0, -60.0
    for f in range(0, 120):
        t = f / FPS
        x = 100 + vx_in * t if t <= 2.0 else 100 + vx_in * 2.0 - vx_in * (t - 2.0)
        y = 300.0 + vy * t
        rows.append({
            "source_frame": f,
            "timestamp_seconds": f"{t:.6f}",
            "detected": "True",
            "x_center": f"{x:.3f}",
            "y_center": f"{y:.3f}",
        })
    return rows


PRE_REFACTOR_OUTPUT = (
    '[{"after_gap": false, "candidate_x": 891.1113333333333, "candidate_y": 180.0, '
    '"dv_magnitude": 266.6826668266676, "hit_frame": 60, '
    '"impact_frame": 59.99999495798363, "impact_mismatch_px": 5.399272619250843e-10, '
    '"impact_time": 2.0000000000000004, "impact_x": 900.0000000027, '
    '"impact_y": 179.99999999999994, "score": 2.648139959547268, '
    '"speed_after": 146.21895635493377, "speed_before": 146.21895635493243, '
    '"timestamp_seconds": 2.0, "turn_degrees": 131.54665339974753}]'
)


def test_legacy_switch_matches_pre_refactor_output():
    hits = detect_hits_from_rows(legacy_reference_rows(), bounce_detector="legacy_sign_flip")
    assert json.dumps(hits, sort_keys=True) == PRE_REFACTOR_OUTPUT

    # The legacy switch restores the old behavior even with a calibration.
    hits = detect_hits_from_rows(
        legacy_reference_rows(),
        bounce_detector="legacy_sign_flip",
        calibration=CALIBRATION,
    )
    assert json.dumps(hits, sort_keys=True) == PRE_REFACTOR_OUTPUT

    # Without a calibration the two_stage default falls back to the same path.
    hits = detect_hits_from_rows(legacy_reference_rows())
    assert json.dumps(hits, sort_keys=True) == PRE_REFACTOR_OUTPUT


def test_two_stage_is_default_in_pipeline():
    trajectory, (impact_x, impact_y, _) = lofted_shot()
    hits = detect_hits_from_rows(rows_from_trajectory(trajectory), calibration=CALIBRATION)
    hit = max(hits, key=lambda h: h["dv_magnitude"])

    assert hit["method"] == "two_stage"
    assert abs(hit["impact_x"] - impact_x) < 3.0
    assert abs(hit["impact_y"] - impact_y) < 3.0
    assert hit["impact_frame"] == 30.0


def test_detect_bounce_dispatches_on_config():
    trajectory, _ = lofted_shot()
    assert detect_bounce(trajectory, CALIBRATION).method == "two_stage"
    assert (
        detect_bounce(trajectory, CALIBRATION, {"bounce_detector": "legacy_sign_flip"}).method
        == "legacy_sign_flip"
    )
