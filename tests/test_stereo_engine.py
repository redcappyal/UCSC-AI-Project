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
    # A point BEHIND the cameras (y > camera y): project() would raise, so
    # build pixels from a valid point but flip one ray by picking pixels
    # whose closest approach lands behind: use crossing rays aimed away.
    # Construct directly: pixel far left on one camera, far right on the
    # other, so rays diverge and closest approach is at negative s/t.
    # Extreme opposite-side pixels: rays diverge, closest approach is behind both cameras.
    point_ft, gap_ft = stereo_engine.triangulate(
        left, right, (5900.0, 540.0), (-4000.0, 540.0))
    assert point_ft is None
