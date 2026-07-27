# ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
# Not imported by any runtime module, not collected by pytest, not linted.
# Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import court_model
import stereo_engine
from stereo_engine import TrackSample
from court_model import CameraModel

GOLDENS = Path(__file__).resolve().parent / "stereo_goldens.json"


def load():
    data = json.loads(GOLDENS.read_text())
    assert data["schema"] == "stereo-goldens-v3"
    left = CameraModel.from_dict(data["cameras"]["left"])
    right = CameraModel.from_dict(data["cameras"]["right"])
    return data, left, right


def test_goldens_file_matches_ios_fixture():
    ios_copy = GOLDENS.parents[1] / "ios" / "Tests" / "Fixtures" / "stereo_goldens.json"
    assert GOLDENS.read_text() == ios_copy.read_text()


def test_triangulation_goldens():
    data, left, right = load()
    for case in data["triangulation_cases"]:
        point, gap = stereo_engine.triangulate(left, right, case["px_a"], case["px_b"])
        assert np.allclose(point, case["point_ft"], atol=1e-9)
        assert abs(gap - case["gap_ft"]) < 1e-9


def test_snap_and_call_goldens():
    data, left, right = load()
    models = {"left": left, "right": right}
    for case in data["snap_cases"]:
        snap = stereo_engine.snap_to_plane(models[case["camera"]], case["px"], case["surface"])
        assert np.allclose(snap, case["point_ft"], atol=1e-9)
    for case in data["call_cases"]:
        call, margin = stereo_engine.call_for_impact(case["surface"], np.array(case["point_ft"]))
        assert call == case["call"] and abs(margin - case["margin_ft"]) < 1e-9


def test_trajectory_impact_goldens():
    data, left, right = load()
    for case in data["trajectories"]:
        samples_a = [TrackSample(t_s=s["t_s"], px=tuple(s["px"])) for s in case["samples_a"]]
        samples_b = [TrackSample(t_s=s["t_s"], px=tuple(s["px"])) for s in case["samples_b"]]
        track = stereo_engine.build_track3d(left, samples_a, right, samples_b,
                                             timeline_s=case["timeline_s"])
        impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b, track=track)
        expected = case["impacts"]
        assert len(impacts) == len(expected), case["name"]
        for got, want in zip(impacts, expected):
            assert got.surface == want["surface"] and got.call == want["call"]
            assert got.confidence == want["confidence"]
            assert abs(got.t_s - want["t_s"]) < 1e-9
            assert np.allclose(got.point_ft, want["point_ft"], atol=1e-9)


def test_trajectory_cases_cover_confidence_tiers():
    data, _, _ = load()
    tiers = {i["confidence"] for c in data["trajectories"] for i in c["impacts"]}
    assert {"high", "one_view", "no_call"} <= tiers


def test_pair_agreement_goldens():
    data, left, right = load()
    case = data["pair_agreement"]
    obs_a = [(np.array(e["court_ft"]), np.array(e["px_a"])) for e in case["obs_lattice"]]
    obs_b = [(np.array(e["court_ft"]), np.array(e["px_b"])) for e in case["obs_lattice"]]
    good = stereo_engine.pair_agreement(left, obs_a, right, obs_b)
    assert good["ok_pair"] == case["good"]["ok_pair"]
    assert abs(good["median_err_ft"] - case["good"]["median_err_ft"]) < 1e-9
    import dataclasses
    biased_model = dataclasses.replace(
        right, camera_center_ft=right.camera_center_ft + np.array(case["biased"]["bias_ft"]))
    biased = stereo_engine.pair_agreement(left, obs_a, biased_model, obs_b)
    assert biased["ok_pair"] is False and biased["ok_pair"] == case["biased"]["ok_pair"]


def test_scaled_model_goldens():
    """Pins the pixel-space scaling both runtimes have to agree on: the
    source model at 1080x1920, its 2160x3840 re-expression, and rays cast
    through matching pixels in each space."""
    data, _left, _right = load()
    case = data["scaled_model"]
    source = CameraModel.from_dict(case["source"])
    assert (source.frame_width, source.frame_height) == (1080.0, 1920.0)

    scaled = court_model.scale_camera_model(
        source, case["to_width"], case["to_height"])
    assert scaled.to_dict() == case["scaled"]

    for ray_case in case["rays"]:
        origin, direction = source.ray(
            court_model.undistort_point(ray_case["px"], source.distortion))
        assert np.allclose(origin, ray_case["origin_ft"], atol=1e-12)
        assert np.allclose(direction, ray_case["dir"], atol=1e-12)
        # The invariance itself: the same ray through the scaled pixel.
        origin_s, direction_s = scaled.ray(
            court_model.undistort_point(ray_case["scaled_px"], scaled.distortion))
        assert np.array_equal(origin_s, origin)
        assert np.allclose(direction_s, direction, atol=1e-9)
