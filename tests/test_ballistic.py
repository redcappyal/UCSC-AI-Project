import numpy as np

import ballistic
from ballistic import BallisticArc, GRAVITY_VEC
from synthetic3d import make_camera


def _project_trajectory(camera, x0, v0, times, t_ref=None):
    t_ref = times[0] if t_ref is None else t_ref
    pixels = []
    for t in times:
        tau = t - t_ref
        point = np.asarray(x0) + np.asarray(v0) * tau + 0.5 * GRAVITY_VEC * tau * tau
        pixels.append(camera.project(point))
    return np.asarray(pixels)


def test_fit_arc_recovers_state():
    camera = make_camera()
    times = np.arange(12) / 60.0
    x0 = np.array([4.0, 25.0, 3.0])
    v0 = np.array([10.0, -55.0, 12.0])   # driven toward the front wall
    pixels = _project_trajectory(camera, x0, v0, times)
    arc = ballistic.fit_arc(times, pixels, camera)
    assert arc is not None
    assert np.allclose(arc.x0, x0, atol=0.15)
    assert np.allclose(arc.v0, v0, atol=1.5)
    assert arc.rms_px < 0.1


def test_fit_arc_tolerates_pixel_noise():
    rng = np.random.default_rng(3)
    camera = make_camera()
    times = np.arange(15) / 60.0
    pixels = _project_trajectory(
        camera, [16.0, 22.0, 2.0], [-8.0, -60.0, 15.0], times)
    noisy = pixels + rng.normal(0, 1.0, pixels.shape)
    arc = ballistic.fit_arc(times, noisy, camera)
    assert arc is not None
    assert arc.rms_px < 3.0


def test_fit_arc_too_short_returns_none():
    camera = make_camera()
    times = np.arange(2) / 60.0
    pixels = _project_trajectory(camera, [10.0, 16.0, 3.0], [0.0, -40.0, 5.0], times)
    assert ballistic.fit_arc(times, pixels, camera) is None


def test_position_velocity_evaluation():
    arc = BallisticArc(t_ref=1.0, x0=np.zeros(3), v0=np.array([1.0, 2.0, 3.0]),
                       rms_px=0.0, start=0, end=5)
    assert np.allclose(arc.position(1.0), [0.0, 0.0, 0.0])
    expected_z = 3.0 * 0.5 + 0.5 * GRAVITY_VEC[2] * 0.25
    assert np.allclose(arc.position(1.5), [0.5, 1.0, expected_z])
    assert np.allclose(arc.velocity(1.5), [1.0, 2.0, 3.0 + GRAVITY_VEC[2] * 0.5])


def test_fit_arc_degenerate_duplicate_times_returns_none():
    camera = make_camera()
    times = np.zeros(4)
    pixels = _project_trajectory(camera, [10.0, 20.0, 5.0], [3.0, -40.0, 8.0], times)
    assert ballistic.fit_arc(times, pixels, camera) is None
