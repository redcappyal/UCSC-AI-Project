"""Synthetic-trajectory tests for impact-point estimation and judging."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect_wall_hits import detect_hits_from_rows
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
    assert "impact_x" in hit
    assert abs(hit["impact_x"] - TRUE_IMPACT[0]) < 3.0
    assert abs(hit["impact_y"] - TRUE_IMPACT[1]) < 3.0
    assert abs(hit["impact_frame"] - 60) < 2.0


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


def test_judge_falls_back_to_detected_center(calibrated_run):
    rows = bounce_rows()
    hits = [dict(strongest(detect_hits_from_rows(rows)))]
    for key in list(hits[0]):
        if key.startswith("impact_"):
            del hits[0][key]
    judged = judge_hits(calibrated_run, results_map(rows), hits)
    assert judged[0]["judge_source"] == "detected_center"
    assert judged[0]["call"] == "IN"


def test_judge_unknown_when_nothing_available(calibrated_run):
    rows = bounce_rows()
    hits = [dict(strongest(detect_hits_from_rows(rows)))]
    frame = hits[0]["hit_frame"]
    for key in list(hits[0]):
        if key.startswith("impact_"):
            del hits[0][key]
    results = results_map(rows)
    del results[frame]
    judged = judge_hits(calibrated_run, results, hits)
    assert judged[0]["call"] == "UNKNOWN"
    assert judged[0]["judge_source"] is None


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
