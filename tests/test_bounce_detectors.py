"""Synthetic-trajectory tests for the swappable bounce detectors."""

import json
import pytest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect_wall_hits import (
    detect_bounce,
    detect_bounce_legacy,
    detect_bounce_two_stage,
    detect_hits_from_rows,
)
from train_bounce_classifier import filter_stationary_ball_rows
from train_bounce_classifier import (
    app_filtered_eval_predictions,
    geometry_features,
    load_geometry,
)
from tracking_common import select_motion_consistent_ball_predictions
from bounce_gb_model_detector import (
    calibrated_wall_gate,
    collapse_front_wall_chunks,
    collapse_wall_area_duplicates,
    inside_front_wall_chunk_gate,
    inside_lenient_sidewall_gate,
    is_stationary_false_track,
    rows_by_frame,
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


def detector_row(frame, x, y):
    return {
        "source_frame": frame,
        "timestamp_seconds": f"{frame / FPS:.6f}",
        "detected": "True",
        "confidence": "0.900",
        "x_center": f"{x:.3f}",
        "y_center": f"{y:.3f}",
        "width": "12.000",
        "height": "12.000",
    }


def pred(x, y, confidence, class_name="ball"):
    return {
        "x": x,
        "y": y,
        "width": 10,
        "height": 10,
        "confidence": confidence,
        "class": class_name,
    }


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


def test_flat_drive_found_by_line_distance_candidates():
    trajectory = flat_drive()
    hits = detect_hits_from_rows(rows_from_trajectory(trajectory), calibration=CALIBRATION)

    assert hits
    hit = min(hits, key=lambda h: abs(h.get("impact_frame", -999) - 24))
    assert hit["method"] == "two_stage"
    assert "line_distance" in hit["diagnostics"]["candidate_source"]
    assert hit["diagnostics"]["nearest_line"] == "tin_top_edge"
    assert abs(hit["impact_frame"] - 24) <= 1


def test_lenient_sidewall_gate_rejects_obvious_sidewall_points():
    wall_gate = calibrated_wall_gate(CALIBRATION)

    assert inside_lenient_sidewall_gate(1000, 500, wall_gate, (0, 2000))
    assert inside_lenient_sidewall_gate(-300, 500, wall_gate, (0, 2000))
    assert inside_lenient_sidewall_gate(2300, 500, wall_gate, (0, 2000))
    assert not inside_lenient_sidewall_gate(-700, 500, wall_gate, (0, 2000))
    assert not inside_lenient_sidewall_gate(2700, 500, wall_gate, (0, 2000))


def test_sidewall_gate_prefers_tilted_wall_corners():
    calibration = {
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[100, 100], [900, 100]]},
            {"name": "tin_top_edge", "endpoints": [[100, 700], [900, 700]]},
        ],
        "planes": {
            "wall": {
                "corners": [
                    {"id": "top_left", "tap_px": [180, 90]},
                    {"id": "top_right", "tap_px": [860, 120]},
                    {"id": "bottom_right", "tap_px": [940, 720]},
                    {"id": "bottom_left", "tap_px": [120, 690]},
                ]
            }
        },
    }
    wall_gate = calibrated_wall_gate(calibration)

    assert wall_gate[0] == "wall_corners"
    assert inside_lenient_sidewall_gate(150, 650, wall_gate, (100, 900))
    assert inside_lenient_sidewall_gate(910, 650, wall_gate, (100, 900))
    assert not inside_lenient_sidewall_gate(-250, 650, wall_gate, (100, 900))
    assert not inside_lenient_sidewall_gate(1300, 650, wall_gate, (100, 900))


def test_wall_area_duplicate_collapse_keeps_highest_confidence():
    candidates = [
        {"hit_frame": 100, "score": 0.42},
        {"hit_frame": 112, "score": 0.91},
        {"hit_frame": 125, "score": 0.61},
        {"hit_frame": 170, "score": 0.55},
    ]

    picked = collapse_wall_area_duplicates(candidates, max_gap=24)

    assert [hit["hit_frame"] for hit in picked] == [112, 170]
    assert picked[0]["wall_visit_candidate_count"] == 3
    assert picked[0]["wall_visit_frames"] == [100, 112, 125]


def test_front_wall_chunks_keep_best_confidence_across_long_wall_visit():
    calibration = {
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[100, 100], [900, 100]]},
            {"name": "tin_top_edge", "endpoints": [[100, 700], [900, 700]]},
        ],
        "planes": {
            "wall": {
                "corners": [
                    {"id": "top_left", "tap_px": [100, 100]},
                    {"id": "top_right", "tap_px": [900, 100]},
                    {"id": "bottom_right", "tap_px": [900, 700]},
                    {"id": "bottom_left", "tap_px": [100, 700]},
                ]
            }
        },
    }
    parsed_rows = rows_by_frame([
        detector_row(frame, 300 + frame, 400)
        for frame in range(100, 171)
    ])
    candidates = [
        {"hit_frame": 100, "score": 0.50},
        {"hit_frame": 140, "score": 0.92},
        {"hit_frame": 170, "score": 0.70},
    ]

    picked = collapse_front_wall_chunks(
        candidates,
        parsed_rows,
        calibrated_wall_gate(calibration),
        (100, 900),
    )

    assert [hit["hit_frame"] for hit in picked] == [140]
    assert picked[0]["wall_visit_candidate_count"] == 3
    assert picked[0]["front_wall_chunk_start_frame"] == 100
    assert picked[0]["front_wall_chunk_end_frame"] == 170


def test_front_wall_chunks_split_when_ball_leaves_wall_bounds():
    calibration = {
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[100, 100], [900, 100]]},
            {"name": "tin_top_edge", "endpoints": [[100, 700], [900, 700]]},
        ],
        "planes": {
            "wall": {
                "corners": [
                    {"id": "top_left", "tap_px": [100, 100]},
                    {"id": "top_right", "tap_px": [900, 100]},
                    {"id": "bottom_right", "tap_px": [900, 700]},
                    {"id": "bottom_left", "tap_px": [100, 700]},
                ]
            }
        },
    }
    parsed_rows = rows_by_frame([
        detector_row(100, 450, 400),
        detector_row(101, 460, 400),
        detector_row(102, 1300, 400),
        detector_row(103, 500, 420),
        detector_row(104, 510, 420),
    ])
    candidates = [
        {"hit_frame": 100, "score": 0.50},
        {"hit_frame": 101, "score": 0.80},
        {"hit_frame": 103, "score": 0.60},
        {"hit_frame": 104, "score": 0.70},
    ]

    picked = collapse_front_wall_chunks(
        candidates,
        parsed_rows,
        calibrated_wall_gate(calibration),
        (100, 900),
    )

    assert [hit["hit_frame"] for hit in picked] == [101, 104]
    assert picked[0]["front_wall_chunk_end_frame"] == 101
    assert picked[1]["front_wall_chunk_start_frame"] == 103


def test_front_wall_chunks_do_not_split_on_missing_detection_hiccup():
    calibration = {
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[100, 100], [900, 100]]},
            {"name": "tin_top_edge", "endpoints": [[100, 700], [900, 700]]},
        ],
        "planes": {
            "wall": {
                "corners": [
                    {"id": "top_left", "tap_px": [100, 100]},
                    {"id": "top_right", "tap_px": [900, 100]},
                    {"id": "bottom_right", "tap_px": [900, 700]},
                    {"id": "bottom_left", "tap_px": [100, 700]},
                ]
            }
        },
    }
    parsed_rows = rows_by_frame([
        detector_row(100, 450, 400),
        detector_row(101, 460, 400),
        {
            "source_frame": 102,
            "timestamp_seconds": f"{102 / FPS:.6f}",
            "detected": "False",
            "confidence": "0.000",
            "x_center": "",
            "y_center": "",
            "width": "0.000",
            "height": "0.000",
        },
        detector_row(103, 500, 420),
        detector_row(104, 510, 420),
    ])
    candidates = [
        {"hit_frame": 101, "score": 0.80},
        {"hit_frame": 103, "score": 0.70},
    ]

    picked = collapse_front_wall_chunks(
        candidates,
        parsed_rows,
        calibrated_wall_gate(calibration),
        (100, 900),
    )

    assert [hit["hit_frame"] for hit in picked] == [101]
    assert picked[0]["wall_visit_frames"] == [101, 103]


def test_front_wall_chunk_gate_is_horizontal_and_tin_bounded():
    calibration = {
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[100, 100], [900, 100]]},
            {"name": "tin_top_edge", "endpoints": [[100, 700], [900, 700]]},
        ],
        "planes": {
            "wall": {
                "corners": [
                    {"id": "top_left", "tap_px": [100, 100]},
                    {"id": "top_right", "tap_px": [900, 100]},
                    {"id": "bottom_right", "tap_px": [900, 700]},
                    {"id": "bottom_left", "tap_px": [100, 700]},
                ]
            }
        },
    }
    wall_gate = calibrated_wall_gate(calibration)

    assert inside_front_wall_chunk_gate(500, 50, wall_gate, (100, 900))
    assert not inside_front_wall_chunk_gate(950, 400, wall_gate, (100, 900))
    assert not inside_front_wall_chunk_gate(500, 760, wall_gate, (100, 900))


def test_stationary_false_track_rejects_dust_like_detection():
    rows = rows_by_frame([
        detector_row(frame, 500 + (frame % 2) * 0.7, 300 + (frame % 3) * 0.5)
        for frame in range(20)
    ])

    stationary, stats = is_stationary_false_track(rows, 10)

    assert stationary
    assert stats["span_px"] < 3


def test_stationary_false_track_keeps_moving_ball():
    rows = rows_by_frame([
        detector_row(frame, 300 + frame * 9, 260 + frame * 4)
        for frame in range(20)
    ])

    stationary, stats = is_stationary_false_track(rows, 10)

    assert not stationary
    assert stats["span_px"] > 100


def test_training_stationary_filter_marks_static_rows_missing():
    rows = {
        frame: {
            "frame": frame,
            "timestamp": frame / FPS,
            "detected": True,
            "confidence": 0.9,
            "x": 640.0 + (frame % 2) * 0.4,
            "y": 360.0 + (frame % 3) * 0.4,
            "width": 11.0,
            "height": 11.0,
        }
        for frame in range(20)
    }

    filtered, rejected = filter_stationary_ball_rows(rows)

    assert rejected
    assert not filtered[10]["detected"]
    assert filtered[10]["confidence"] == 0.0


def test_training_runtime_eval_filters_group_app_style_predictions(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "lines": [
                    {"name": "out_line_lower_edge", "endpoints": [[100, 100], [900, 100]]},
                    {"name": "tin_top_edge", "endpoints": [[100, 700], [900, 700]]},
                ],
                "planes": {
                    "wall": {
                        "corners": [
                            {"id": "top_left", "tap_px": [100, 100]},
                            {"id": "top_right", "tap_px": [900, 100]},
                            {"id": "bottom_right", "tap_px": [900, 700]},
                            {"id": "bottom_left", "tap_px": [100, 700]},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    geometry = load_geometry(calibration_path)
    features = pd.DataFrame(
        [
            {"frame": 100, "t+0_detected": 1.0, "t+0_x": 500.0, "t+0_y": 500.0},
            {"frame": 108, "t+0_detected": 1.0, "t+0_x": 520.0, "t+0_y": 500.0},
            {"frame": 116, "t+0_detected": 1.0, "t+0_x": 530.0, "t+0_y": 500.0},
            {"frame": 170, "t+0_detected": 1.0, "t+0_x": 1300.0, "t+0_y": 500.0},
        ]
    )

    predictions, stats = app_filtered_eval_predictions(
        features,
        np.array([0.40, 0.91, 0.61, 0.88]),
        0.25,
        geometry=geometry,
        spatial_filter=True,
        spatial_filter_mode="sidewall",
        wall_visit_gap=24,
        min_gap=0,
    )

    assert predictions.tolist() == [0, 1, 0, 0]
    assert stats["threshold_candidates"] == 4
    assert stats["kept_candidates"] == 1


def test_geometry_features_include_video_level_calibration_context(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "lines": [
                    {"name": "out_line_lower_edge", "endpoints": [[100, 120], [900, 140]]},
                    {"name": "tin_top_edge", "endpoints": [[80, 720], [920, 760]]},
                ]
            }
        ),
        encoding="utf-8",
    )
    geometry = load_geometry(calibration_path)
    row = {
        "detected": True,
        "confidence": 0.8,
        "x": 500.0,
        "y": 500.0,
        "width": 12.0,
        "height": 12.0,
    }

    features = geometry_features(row, geometry)

    assert features["calibration_wall_height_px"] > 0.0
    assert features["calibration_wall_width_px"] > 0.0
    assert features["calibration_roll_degrees"] > 0.0
    assert features["calibration_perspective_shear"] != 0.0


def test_motion_consistent_selector_prefers_moving_ball_over_static_dust():
    predictions_by_frame = {
        frame: [
            pred(500, 300, 0.60),
            pred(200 + frame * 9, 430 + frame * 2, 0.50),
        ]
        for frame in range(12)
    }

    selected = select_motion_consistent_ball_predictions(predictions_by_frame)

    assert selected[6]["confidence"] == 0.50
    assert selected[6]["x"] == 254


def test_motion_consistent_selector_rejects_only_stationary_dust():
    predictions_by_frame = {
        frame: [pred(500 + (frame % 2), 300, 0.60)]
        for frame in range(12)
    }

    selected = select_motion_consistent_ball_predictions(predictions_by_frame)

    assert selected[6] is None


def test_motion_consistent_selector_falls_back_to_confidence_without_context():
    selected = select_motion_consistent_ball_predictions({
        100: [pred(10, 20, 0.40), pred(30, 40, 0.75)]
    })

    assert selected[100]["confidence"] == 0.75


def test_motion_consistent_selector_does_not_teleport_link_fragmented_dust():
    predictions_by_frame = {}
    for frame in range(9):
        predictions_by_frame[frame] = [pred(180 + frame * 11, 420 + frame * 3, 0.50)]

    for frame in [0, 1, 2, 6, 7, 8]:
        predictions_by_frame[frame].append(pred(520 + (frame % 2), 300, 0.60))

    selected = select_motion_consistent_ball_predictions(predictions_by_frame)

    assert selected[2]["confidence"] == 0.50
    assert selected[6]["confidence"] == 0.50


def test_motion_consistent_selector_uses_multiple_non_ball_class_fallbacks():
    predictions_by_frame = {
        frame: [
            pred(500, 300, 0.60, class_name="object"),
            pred(200 + frame * 9, 430 + frame * 2, 0.50, class_name="object"),
        ]
        for frame in range(12)
    }

    selected = select_motion_consistent_ball_predictions(predictions_by_frame)

    assert selected[6]["confidence"] == 0.50
    assert selected[6]["x"] == 254


def test_straight_line_crossing_calibrated_line_is_not_a_hit():
    points = [(200 + 250 * (f / FPS), 620 + 100 * (f / FPS)) for f in range(60)]
    hits = detect_hits_from_rows(
        rows_from_trajectory(trajectory_from_points(points)),
        calibration=CALIBRATION,
    )

    assert hits == []


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
# refactor; the "legacy_sign_flip" switch must reproduce it (to float
# tolerance — see assert_matches_pre_refactor).
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


def assert_matches_pre_refactor(hits):
    # Structural comparison, not byte-for-byte: the reference floats were
    # captured on macOS, and Linux libm differs in the last ulp (CI showed
    # turn_degrees ...74724 vs ...74753). Near-zero residuals get an absolute
    # tolerance; everything else must agree to 1e-9 relative.
    expected = json.loads(PRE_REFACTOR_OUTPUT)
    assert len(hits) == len(expected)
    for actual_hit, expected_hit in zip(hits, expected):
        assert sorted(actual_hit) == sorted(expected_hit)
        for key, expected_value in expected_hit.items():
            actual_value = actual_hit[key]
            if isinstance(expected_value, float):
                assert actual_value == pytest.approx(
                    expected_value, rel=1e-9, abs=1e-6
                ), f"{key}: {actual_value!r} != {expected_value!r}"
            else:
                assert actual_value == expected_value, f"{key}"


def test_legacy_switch_matches_pre_refactor_output():
    hits = detect_hits_from_rows(legacy_reference_rows(), bounce_detector="legacy_sign_flip")
    assert_matches_pre_refactor(hits)

    # The legacy switch restores the old behavior even with a calibration.
    hits = detect_hits_from_rows(
        legacy_reference_rows(),
        bounce_detector="legacy_sign_flip",
        calibration=CALIBRATION,
    )
    assert_matches_pre_refactor(hits)

    # Without a calibration the two_stage default falls back to the same path.
    hits = detect_hits_from_rows(legacy_reference_rows())
    assert_matches_pre_refactor(hits)


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
