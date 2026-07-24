import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import stereo_engine
from stereo_engine import TrackSample
from court_model import CameraModel

GOLDENS = Path(__file__).resolve().parent / "stereo_goldens.json"


def load():
    data = json.loads(GOLDENS.read_text())
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
    samples_a = [TrackSample(t_s=s["t_s"], px=tuple(s["px"])) for s in data["trajectory"]["samples_a"]]
    samples_b = [TrackSample(t_s=s["t_s"], px=tuple(s["px"])) for s in data["trajectory"]["samples_b"]]
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)
    expected = data["trajectory"]["impacts"]
    assert len(impacts) == len(expected)
    for got, want in zip(impacts, expected):
        assert got.surface == want["surface"] and got.call == want["call"]
        assert abs(got.t_s - want["t_s"]) < 1e-9
        assert np.allclose(got.point_ft, want["point_ft"], atol=1e-9)
