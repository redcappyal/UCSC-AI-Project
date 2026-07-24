"""Regenerate the stereo golden vectors (deterministic — rng seed 7).

Run from the repo root:  python tests/generate_stereo_goldens.py
Writes tests/stereo_goldens.json and ios/Tests/Fixtures/stereo_goldens.json.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

import stereo_engine
from stereo_engine import TrackSample
from synthetic3d import make_camera
from test_stereo_track import sample_camera, simulate_front_wall_shot


def main():
    rng = np.random.default_rng(7)
    left = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    right = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))

    triangulation_cases = []
    for _ in range(12):
        point = np.array([rng.uniform(2, 19), rng.uniform(2, 28), rng.uniform(0.5, 12)])
        px_a, px_b = left.project(point), right.project(point)
        got, gap = stereo_engine.triangulate(left, right, px_a, px_b)
        triangulation_cases.append({
            "px_a": list(map(float, px_a)), "px_b": list(map(float, px_b)),
            "point_ft": [float(v) for v in got], "gap_ft": float(gap)})

    snap_cases, call_cases = [], []
    wall_points = [("front_wall", np.array([13.0, 0.0, 12.0])),
                   ("front_wall", np.array([6.0, 0.0, 15.8])),
                   ("front_wall", np.array([10.5, 0.0, 1.2])),
                   ("left_wall", np.array([0.0, 10.0, 12.9])),
                   ("right_wall", np.array([21.0, 24.0, 8.0])),
                   ("floor", np.array([8.0, 20.0, 0.0]))]
    for surface, point in wall_points:
        for name, model in (("left", left), ("right", right)):
            snap = stereo_engine.snap_to_plane(model, model.project(point), surface)
            snap_cases.append({"camera": name, "px": list(map(float, model.project(point))),
                               "surface": surface, "point_ft": [float(v) for v in snap]})
        call, margin = stereo_engine.call_for_impact(surface, point)
        call_cases.append({"surface": surface, "point_ft": [float(v) for v in point],
                           "call": call, "margin_ft": float(margin)})

    states, _ = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)

    goldens = {
        "schema": "stereo-goldens-v1",
        "cameras": {"left": left.to_dict(), "right": right.to_dict()},
        "triangulation_cases": triangulation_cases,
        "snap_cases": snap_cases,
        "call_cases": call_cases,
        "trajectory": {
            "samples_a": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_a],
            "samples_b": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_b],
            "impacts": [{"t_s": i.t_s, "surface": i.surface,
                          "point_ft": [float(v) for v in i.point_ft], "call": i.call,
                          "margin_ft": i.margin_ft, "confidence": i.confidence}
                         for i in impacts],
        },
    }
    payload = json.dumps(goldens, indent=2, sort_keys=True)
    (REPO / "tests" / "stereo_goldens.json").write_text(payload)
    fixture_dir = REPO / "ios" / "Tests" / "Fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "stereo_goldens.json").write_text(payload)
    print(f"wrote {len(payload)} bytes to tests/ and ios/Tests/Fixtures/")


if __name__ == "__main__":
    main()
