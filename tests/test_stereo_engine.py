import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import stereo_engine
from synthetic3d import make_camera


def make_fin_pair():
    """Two phones on the back-wall fins, 3 ft apart, 7 ft up, aimed at the
    front wall — the production mounting from the spec."""
    left = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    right = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    return left, right


def test_triangulate_recovers_known_point():
    left, right = make_fin_pair()
    point = np.array([8.0, 12.0, 3.5])
    point_ft, gap_ft = stereo_engine.triangulate(
        left, right, left.project(point), right.project(point))
    assert gap_ft < 1e-9
    assert np.allclose(point_ft, point, atol=1e-9)


def test_triangulate_pixel_noise_bounded_error():
    left, right = make_fin_pair()
    point = np.array([10.5, 16.0, 2.0])   # mid-court
    px_a = np.asarray(left.project(point)) + np.array([2.0, -2.0])
    px_b = np.asarray(right.project(point)) + np.array([-2.0, 2.0])
    point_ft, gap_ft = stereo_engine.triangulate(left, right, px_a, px_b)
    # Narrow-baseline depth error dominates; lateral/height stay tight.
    assert abs(point_ft[0] - point[0]) < 0.2
    assert abs(point_ft[2] - point[2]) < 0.2
    assert abs(point_ft[1] - point[1]) < 1.5
    assert gap_ft < 0.5


def test_triangulate_parallel_rays_returns_none():
    left, _ = make_fin_pair()
    # Same camera twice with the same pixel: identical rays are degenerate.
    point_ft, gap_ft = stereo_engine.triangulate(
        left, left, (960.0, 540.0), (960.0, 540.0))
    assert point_ft is None
    assert gap_ft == np.inf


def test_triangulate_behind_camera_rejected():
    left, right = make_fin_pair()
    # Extreme opposite-side pixels: rays diverge, closest approach is behind both cameras.
    point_ft, gap_ft = stereo_engine.triangulate(
        left, right, (5900.0, 540.0), (-4000.0, 540.0))
    assert point_ft is None
    assert gap_ft == np.inf


def test_surface_planes_and_side_out_slope():
    point, normal = stereo_engine.surface_plane("front_wall")
    assert point[1] == 0.0 and np.allclose(normal, [0.0, 1.0, 0.0])
    point, normal = stereo_engine.surface_plane("floor")
    assert point[2] == 0.0 and np.allclose(normal, [0.0, 0.0, 1.0])
    assert stereo_engine.side_wall_out_height_ft(0.0) == 15.0
    assert stereo_engine.side_wall_out_height_ft(32.0) == 7.0
    assert stereo_engine.side_wall_out_height_ft(16.0) == 11.0


def test_calls_front_wall():
    out_call = stereo_engine.call_for_impact("front_wall", np.array([10.0, 0.0, 15.4]))
    assert out_call == ("out", 0.3999999999999986) or (
        out_call[0] == "out" and abs(out_call[1] - 0.4) < 1e-9)
    call, margin = stereo_engine.call_for_impact("front_wall", np.array([10.0, 0.0, 1.0]))
    assert call == "down" and abs(margin - (19.0 / 12.0 - 1.0)) < 1e-9
    call, margin = stereo_engine.call_for_impact("front_wall", np.array([10.0, 0.0, 8.0]))
    # Nearest deciding line here is the tin (8 - 19/12), not the out line
    # (15 - 8 = 7): 19/12 ~= 1.583, so the tin distance (~6.417) is smaller.
    assert call == "in" and abs(margin - (8.0 - 19.0 / 12.0)) < 1e-9


def test_calls_side_and_back_walls_and_floor():
    call, margin = stereo_engine.call_for_impact("left_wall", np.array([0.0, 16.0, 11.5]))
    assert call == "out" and abs(margin - 0.5) < 1e-9
    call, margin = stereo_engine.call_for_impact("right_wall", np.array([21.0, 16.0, 10.0]))
    assert call == "in" and abs(margin - 1.0) < 1e-9
    call, margin = stereo_engine.call_for_impact("back_wall", np.array([5.0, 32.0, 7.5]))
    assert call == "out" and abs(margin - 0.5) < 1e-9
    assert stereo_engine.call_for_impact("floor", np.array([5.0, 20.0, 0.0])) == ("bounce", 0.0)


def test_snap_to_plane_recovers_wall_point():
    left, right = make_fin_pair()
    impact = np.array([13.0, 0.0, 12.0])   # on the front wall
    snap_a = stereo_engine.snap_to_plane(left, left.project(impact), "front_wall")
    snap_b = stereo_engine.snap_to_plane(right, right.project(impact), "front_wall")
    assert np.allclose(snap_a, impact, atol=1e-9)
    assert np.allclose(snap_b, impact, atol=1e-9)
    fused = stereo_engine.fuse_snaps(snap_a, snap_b)
    assert np.allclose(fused, impact, atol=1e-9)
    assert stereo_engine.fuse_snaps(None, snap_b) is snap_b
    assert stereo_engine.fuse_snaps(None, None) is None


def test_snap_rejects_out_of_bounds_and_parallel():
    left, _ = make_fin_pair()
    # A pixel whose floor intersection lies far outside the court.
    assert stereo_engine.snap_to_plane(left, (100000.0, 540.0), "floor") is None


def _agreement_grid():
    """3x4x2 lattice over the court volume -- a test fixture only; the
    production endpoint builds obs from real calibration correspondences via
    court_model._camera_correspondences, not this synthetic lattice."""
    grid = []
    for x in (5.25, 10.5, 15.75):
        for y in (4.0, 12.0, 20.0, 28.0):
            for z in (1.0, 8.0):
                grid.append(np.array([x, y, z]))
    return grid


def _project_grid_obs(model, grid):
    """Build (court_ft, px) observations by projecting a lattice through a
    single model -- the "honest" analogue of that camera's own calibration
    taps, skipping any point the model can't see."""
    obs = []
    for point in grid:
        try:
            px = model.project(point)
        except ValueError:
            continue
        obs.append((point, px))
    return obs


def test_pair_agreement_synthetic_pair_passes_gate():
    left, right = make_fin_pair()
    grid = _agreement_grid()
    obs_a = _project_grid_obs(left, grid)
    obs_b = _project_grid_obs(right, grid)
    report = stereo_engine.pair_agreement(left, obs_a, right, obs_b)
    assert report["ok_pair"] is True
    assert report["median_err_ft"] < 1e-6
    assert abs(report["baseline_ft"] - 3.0) < 1e-9
    assert report["point_count"] >= 12


def test_pair_agreement_biased_model_fails_gate():
    import dataclasses
    left, right = make_fin_pair()
    grid = _agreement_grid()
    obs_a = _project_grid_obs(left, grid)
    obs_b = _project_grid_obs(right, grid)   # true observations -- unchanged
    # A biased model_b: same observations, but a mis-solved camera center.
    # Rays through the (fixed, true) observations no longer meet at the
    # known point, since the reconstructing model's assumed origin is wrong.
    shifted = dataclasses.replace(
        right, camera_center_ft=right.camera_center_ft + np.array([0.0, 0.0, 0.5]))
    report = stereo_engine.pair_agreement(left, obs_a, shifted, obs_b)
    assert report["ok_pair"] is False
    assert report["median_err_ft"] > stereo_engine.PAIR_GATE_MAX_MEDIAN_FT


def test_pair_agreement_envelope_rejects_implausible_pose():
    import dataclasses
    left, right = make_fin_pair()
    grid = _agreement_grid()
    obs_a = _project_grid_obs(left, grid)
    # A camera solved as sitting near the floor -- physically implausible
    # for the fin mount, but still internally self-consistent (obs are
    # generated by this same low model), so triangulation alone would call
    # it a perfect agreement. The pose envelope must catch it separately.
    low = dataclasses.replace(
        right, camera_center_ft=np.array(
            [right.camera_center_ft[0], right.camera_center_ft[1], 0.5]))
    obs_b = _project_grid_obs(low, grid)
    report = stereo_engine.pair_agreement(left, obs_a, low, obs_b)
    assert report["envelope_ok"] is False
    assert report["ok_pair"] is False
    assert report["median_err_ft"] < 1e-6


def test_surfaces_registry_is_deterministically_ordered():
    assert isinstance(stereo_engine.SURFACES, tuple)
    assert stereo_engine.SURFACES == (
        "floor", "front_wall", "back_wall", "left_wall", "right_wall")
