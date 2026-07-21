"""Synthetic-trajectory tests for impact-point estimation and judging."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect_wall_hits import detect_hits_from_rows
from judge_call import (
    Line,
    Point,
    WallCorners,
    judge_ball,
    judge_margin_px,
    load_wall_corners,
    wall_diagram_coordinates,
)
from job_runner import judge_hits
from tracking_common import ball_csv_row

FPS = 30.0
WALL_Y = 300.0


def bounce_rows(*, drop_frames=(), stride=1, vx_out=None):
    """Ball travels toward the wall along +x, bounces at t=2.0s (frame 60).

    The trajectory also descends in y so the impact y-coordinate is
    distinguishable from every sampled position.
    """
    rows = []
    vx_in, vy = 400.0, -60.0
    vx_out = -vx_in if vx_out is None else vx_out
    for f in range(0, 120, stride):
        if f in drop_frames:
            continue
        t = f / FPS
        if t <= 2.0:
            x = 100 + vx_in * t
        else:
            x = 100 + vx_in * 2.0 + vx_out * (t - 2.0)
        y = WALL_Y + vy * t if t <= 2.0 else WALL_Y + vy * 2.0 + vy * (t - 2.0)
        rows.append({
            "source_frame": f,
            "timestamp_seconds": f"{t:.6f}",
            "detected": "True",
            "x_center": f"{x:.3f}",
            "y_center": f"{y:.3f}",
        })
    return rows


TRUE_IMPACT = (100 + 400 * 2.0, WALL_Y - 60.0 * 2.0)  # (900, 180) at frame 60


def strongest(hits):
    return max(hits, key=lambda h: h["dv_magnitude"])


def test_clean_bounce_impact_between_samples():
    hits = detect_hits_from_rows(bounce_rows(stride=1))
    hit = strongest(hits)
    assert hit["score"] > 0
    assert "impact_x" in hit
    assert abs(hit["impact_x"] - TRUE_IMPACT[0]) < 3.0
    assert abs(hit["impact_y"] - TRUE_IMPACT[1]) < 3.0
    assert abs(hit["impact_frame"] - 60) < 2.0


def test_horizontal_wall_gate_rejects_hits_outside_tin_span():
    rows = bounce_rows()
    assert detect_hits_from_rows(rows, wall_x_range=(0, 800)) == []


def test_horizontal_wall_gate_keeps_hits_inside_tin_span():
    rows = bounce_rows()
    hits = detect_hits_from_rows(rows, wall_x_range=(0, 1000))
    assert strongest(hits)["impact_x"] == pytest.approx(TRUE_IMPACT[0], abs=3.0)


def test_impact_recovered_inside_detection_gap():
    # Detector misses 4 frames straddling the impact; the gap splits the
    # track, so the true impact lies between the tracks.
    hits = detect_hits_from_rows(bounce_rows(drop_frames={57, 58, 59, 60, 61, 62}))
    with_impact = [h for h in hits if "impact_x" in h and abs(h["impact_frame"] - 60) < 3]
    assert with_impact, [
        (h["hit_frame"], h.get("impact_frame"), h["after_gap"]) for h in hits
    ]
    hit = min(with_impact, key=lambda h: abs(h["impact_frame"] - 60))
    assert abs(hit["impact_x"] - TRUE_IMPACT[0]) < 5.0
    assert abs(hit["impact_y"] - TRUE_IMPACT[1]) < 5.0


def test_collinear_return_is_stable():
    # Ball returns along the same image-space line (vy unchanged path shape,
    # x reverses exactly) - the case where a spatial line intersection
    # degenerates. The temporal fit must still land on the impact.
    hits = detect_hits_from_rows(bounce_rows(vx_out=-400.0))
    hit = strongest(hits)
    assert "impact_x" in hit
    assert abs(hit["impact_x"] - TRUE_IMPACT[0]) < 3.0


def test_no_velocity_change_produces_no_impact():
    rows = []
    for f in range(0, 90):
        t = f / FPS
        rows.append({
            "source_frame": f,
            "timestamp_seconds": f"{t:.6f}",
            "detected": "True",
            "x_center": f"{100 + 400 * t:.3f}",
            "y_center": f"{300 - 30 * t:.3f}",
        })
    for hit in detect_hits_from_rows(rows):
        # Any numerical-noise candidates must not carry confident impacts far
        # from their own hit frame.
        if "impact_frame" in hit:
            assert abs(hit["impact_frame"] - hit["hit_frame"]) < 5


@pytest.fixture()
def calibrated_run(tmp_path):
    import json

    (tmp_path / "calibration.json").write_text(json.dumps({
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[0, 100], [2000, 100]]},
            {"name": "tin_top_edge", "endpoints": [[0, 700], [2000, 700]]},
        ]
    }))
    return tmp_path


def results_map(rows):
    return {int(r["source_frame"]): r for r in rows}


def test_judge_prefers_impact_estimate(calibrated_run):
    rows = bounce_rows()
    hits = detect_hits_from_rows(rows)
    judged = judge_hits(calibrated_run, results_map(rows), [strongest(hits)])
    entry = judged[0]
    assert entry["judge_source"] == "impact_estimate"
    assert entry["call"] == "IN"
    assert entry["margin_px"] == pytest.approx(min(entry["impact"]["y"] - 100, 700 - entry["impact"]["y"]))
    assert entry["velocity"]["scale_source"] == "tin_top_edge_21ft"
    assert entry["velocity"]["pixels_per_foot"] == pytest.approx(2000 / 21)
    assert entry["velocity"]["speed_before"]["mph"] > 0
    assert entry["velocity"]["speed_after"]["mph"] > 0


def test_judge_displays_nearest_detected_impact_frame(calibrated_run):
    rows = bounce_rows()
    hit = dict(strongest(detect_hits_from_rows(rows)))
    hit["hit_frame"] = 64
    hit["impact_frame"] = 60.2
    judged = judge_hits(calibrated_run, results_map(rows), [hit])
    assert judged[0]["frame"] == 60
    assert judged[0]["candidate_frame"] == 64
    assert judged[0]["judge_source"] == "impact_estimate"


def test_judge_falls_back_to_detected_center(calibrated_run):
    rows = bounce_rows()
    hits = [dict(strongest(detect_hits_from_rows(rows)))]
    for key in list(hits[0]):
        if key.startswith("impact_"):
            del hits[0][key]
    judged = judge_hits(calibrated_run, results_map(rows), hits)
    assert judged[0]["judge_source"] == "detected_center"
    assert judged[0]["call"] == "IN"


def test_judge_filters_hits_without_display_frame_detection(calibrated_run):
    rows = bounce_rows()
    hit = dict(strongest(detect_hits_from_rows(rows)))
    missing_frame = int(hit["hit_frame"])
    hit["impact_frame"] = float(missing_frame)
    results = results_map(rows)
    results[missing_frame] = {
        **results[missing_frame],
        "detected": "False",
        "x_center": "",
        "y_center": "",
    }

    judged = judge_hits(calibrated_run, results, [hit])
    assert judged
    assert judged[0]["frame"] != missing_frame
    assert results[judged[0]["frame"]]["detected"] == "True"


def test_judge_filters_hits_when_display_frame_is_missing(calibrated_run):
    rows = bounce_rows()
    hits = [dict(strongest(detect_hits_from_rows(rows)))]
    frame = hits[0]["hit_frame"]
    for key in list(hits[0]):
        if key.startswith("impact_"):
            del hits[0][key]
    results = results_map(rows)
    del results[frame]
    assert judge_hits(calibrated_run, results, hits) == []


def test_out_call_with_negative_margin(calibrated_run):
    # Shift the whole trajectory above the out line: y ends around 180 -> use
    # lines that exclude it.
    import json

    (calibrated_run / "calibration.json").write_text(json.dumps({
        "lines": [
            {"name": "out_line_lower_edge", "endpoints": [[0, 250], [2000, 250]]},
            {"name": "tin_top_edge", "endpoints": [[0, 700], [2000, 700]]},
        ]
    }))
    rows = bounce_rows()
    hits = detect_hits_from_rows(rows)
    judged = judge_hits(calibrated_run, results_map(rows), [strongest(hits)])
    assert judged[0]["call"] == "OUT"
    assert judged[0]["margin_px"] < 0


def test_tilted_judge_uses_perpendicular_margin():
    top = Line(Point(0, 100), Point(1000, 200))
    bottom = Line(Point(0, 700), Point(1000, 800))
    ball = Point(500, 450)

    call, reason, top_y, bottom_y = judge_ball(ball, top, bottom)

    assert call == "IN"
    assert reason == "between_lines"
    assert top_y == pytest.approx(150)
    assert bottom_y == pytest.approx(750)
    # Vertical distance would be 300px; tilted-line perpendicular distance is
    # slightly smaller and is the value callers should display as margin.
    assert judge_margin_px(ball, top, bottom) == pytest.approx(300 / (1.01 ** 0.5))


def test_tilted_wall_diagram_coordinates_follow_line_tilt():
    top = Line(Point(100, 100), Point(1100, 200))
    bottom = Line(Point(200, 700), Point(1200, 800))
    top_mid = top.point_at(0.25)
    bottom_mid = bottom.point_at(0.25)
    ball = Point(
        top_mid.x + 0.4 * (bottom_mid.x - top_mid.x),
        top_mid.y + 0.4 * (bottom_mid.y - top_mid.y),
    )

    diagram = wall_diagram_coordinates(ball, top, bottom)

    assert diagram["x"] == pytest.approx(0.25, abs=0.002)
    assert diagram["y"] == pytest.approx(0.4, abs=0.002)


def test_wall_corners_override_line_span_for_judge_bounds():
    top = Line(Point(100, 100), Point(1100, 100))
    bottom = Line(Point(100, 700), Point(1100, 700))
    wall = WallCorners(
        top_left=Point(200, 50),
        top_right=Point(1000, 80),
        bottom_right=Point(940, 760),
        bottom_left=Point(260, 730),
    )

    ball = Point(160, 400)
    call, reason, _, _ = judge_ball(ball, top, bottom, wall)

    assert call == "OUT"
    assert reason == "outside_wall_bounds"
    assert judge_margin_px(ball, top, bottom, wall) < 0


def test_wall_diagram_coordinates_use_corner_width_when_present():
    top = Line(Point(100, 100), Point(1100, 100))
    bottom = Line(Point(100, 700), Point(1100, 700))
    wall = WallCorners(
        top_left=Point(200, 50),
        top_right=Point(1000, 50),
        bottom_right=Point(900, 750),
        bottom_left=Point(300, 750),
    )

    diagram = wall_diagram_coordinates(Point(300, 400), top, bottom, wall_corners=wall)

    assert diagram["x_reference"] == "wall_corners"
    assert diagram["x_span"] == [250.0, 950.0]
    assert diagram["x"] == pytest.approx(50 / 700)
    assert diagram["y"] == pytest.approx(0.5, abs=0.002)


def test_load_wall_corners_from_calibration():
    calibration = {
        "planes": {
            "wall": {
                "corners": [
                    {"id": "top_left", "tap_px": [10, 20]},
                    {"id": "top_right", "tap_px": [110, 25]},
                    {"id": "bottom_right", "tap_px": [100, 220]},
                    {"id": "bottom_left", "tap_px": [20, 215]},
                ]
            }
        }
    }

    wall = load_wall_corners(calibration)

    assert wall is not None
    assert wall.x_bounds_at_y(120)[0] == pytest.approx(15.0, abs=1.0)
