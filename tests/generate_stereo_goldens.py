"""Regenerate the stereo golden vectors (deterministic — rng seed 7).

Run from the repo root:  python tests/generate_stereo_goldens.py
Writes tests/stereo_goldens.json and ios/Tests/Fixtures/stereo_goldens.json.
"""
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

import court_model
import stereo_engine
from stereo_engine import TrackSample
from synthetic3d import make_camera
from test_stereo_track import sample_camera, simulate_front_wall_shot


def scaled_model_case():
    """A model solved at 1080x1920 plus its 2160x3840 re-expression, with
    rays through matching pixels in each space.

    This is the parity anchor for the frame-space fix: Swift's
    CameraModel.scaled(toWidth:height:) has to land on the same numbers as
    court_model.scale_camera_model, and a ray through pixel p in the source
    space has to equal the ray through 2p in the scaled one. Off-axis
    corners are included deliberately -- a wrong principal point or an
    unscaled distortion normalizer only shows up away from the center.
    """
    source_size, target_size = (1080.0, 1920.0), (2160.0, 3840.0)
    source = make_camera(focal_px=900.0, center=(540.0, 960.0),
                         position=(10.5, 30.5, 7.0), look_at=(10.5, 0.0, 4.5),
                         frame_size=source_size)
    source = dataclasses.replace(
        source,
        distortion={"model": "division_k1", "k1": -0.08,
                    "center_px": [548.0, 952.0], "norm_px": 1100.0},
        fit_rms_px=1.25, point_count=15)
    scaled = court_model.scale_camera_model(source, *target_size)
    factor = target_size[0] / source_size[0]

    rays = []
    for pixel in [(540.0, 960.0), (540.0, 12.0), (17.0, 960.0), (0.0, 0.0),
                  (1079.0, 0.0), (0.0, 1919.0), (1079.0, 1919.0),
                  (231.0, 1487.0), (908.0, 344.0)]:
        origin, direction = source.ray(
            court_model.undistort_point(pixel, source.distortion))
        rays.append({
            "px": list(pixel),
            "scaled_px": [pixel[0] * factor, pixel[1] * factor],
            "origin_ft": [float(v) for v in origin],
            "dir": [float(v) for v in direction],
        })
    return {"source": source.to_dict(), "scaled": scaled.to_dict(),
            "to_width": target_size[0], "to_height": target_size[1],
            "rays": rays}


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

    def trajectory_case(name, samples_a, samples_b):
        t_lo = max(samples_a[0].t_s, samples_b[0].t_s)
        t_hi = min(samples_a[-1].t_s, samples_b[-1].t_s)
        timeline = [float(t) for t in np.arange(t_lo, t_hi, 1.0 / 120.0)]
        impacts = stereo_engine.detect_impacts(
            left, samples_a, right, samples_b,
            track=stereo_engine.build_track3d(left, samples_a, right, samples_b,
                                              timeline_s=timeline))
        return {
            "name": name,
            "samples_a": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_a],
            "samples_b": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_b],
            "timeline_s": timeline,
            "impacts": [{"t_s": i.t_s, "surface": i.surface,
                          "point_ft": [float(v) for v in i.point_ft], "call": i.call,
                          "margin_ft": i.margin_ft, "confidence": i.confidence,
                          "snap_disagreement_ft": i.snap_disagreement_ft}
                         for i in impacts],
        }

    states, (t_true, _p) = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    occluded_b = [s for s in samples_b if not (t_true - 0.3 <= s.t_s <= t_true + 0.1)]
    gap_a = [s for s in samples_a if not (t_true - 0.3 <= s.t_s <= t_true - 0.005)]
    gap_b = [s for s in samples_b if not (t_true - 0.3 <= s.t_s <= t_true - 0.005)]
    trajectories = [
        trajectory_case("clean", samples_a, samples_b),
        trajectory_case("occluded_one_view", samples_a, occluded_b),
        trajectory_case("no_call", gap_a, gap_b),
    ]

    lattice = [np.array([x, y, z]) for x in (5.25, 10.5, 15.75)
               for y in (4.0, 12.0, 20.0, 28.0) for z in (1.0, 8.0)]
    obs_a = [(p, np.asarray(left.project(p))) for p in lattice]
    obs_b = [(p, np.asarray(right.project(p))) for p in lattice]
    bias = np.array([0.0, 0.0, 0.5])
    biased_right = dataclasses.replace(
        right, camera_center_ft=right.camera_center_ft + bias)

    def _coerce_report(report):
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                for k, v in report.items()}

    pair_case = {
        "obs_lattice": [{"court_ft": [float(v) for v in p],
                          "px_a": [float(v) for v in pa], "px_b": [float(v) for v in pb]}
                         for (p, pa), (_p2, pb) in zip(obs_a, obs_b)],
        "good": _coerce_report(stereo_engine.pair_agreement(left, obs_a, right, obs_b)),
        "biased": {"bias_ft": [0.0, 0.0, 0.5],
                    **_coerce_report(stereo_engine.pair_agreement(left, obs_a, biased_right, obs_b))},
    }

    goldens = {
        "schema": "stereo-goldens-v3",
        "cameras": {"left": left.to_dict(), "right": right.to_dict()},
        "triangulation_cases": triangulation_cases,
        "snap_cases": snap_cases,
        "call_cases": call_cases,
        "trajectories": trajectories,
        "pair_agreement": pair_case,
        "scaled_model": scaled_model_case(),
    }
    payload = json.dumps(goldens, indent=2, sort_keys=True)
    (REPO / "tests" / "stereo_goldens.json").write_text(payload)
    fixture_dir = REPO / "ios" / "Tests" / "Fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "stereo_goldens.json").write_text(payload)
    print(f"wrote {len(payload)} bytes to tests/ and ios/Tests/Fixtures/")


if __name__ == "__main__":
    main()
