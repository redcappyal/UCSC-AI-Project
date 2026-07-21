import numpy as np
import pytest

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


def _bounce_trajectory(camera, fps=60.0):
    """Drive at the front wall, bounce off it, 3D ground truth throughout.
    Returns (times, raw_pixels, bounce_time, bounce_point)."""
    x0 = np.array([10.0, 24.0, 4.0])
    v0 = np.array([2.0, -70.0, 6.0])
    t_hit = None
    times, pixels, points = [], [], []
    t, dt = 0.0, 1.0 / fps
    position, velocity = x0.copy(), v0.copy()
    for _ in range(40):
        times.append(t)
        points.append(position.copy())
        pixels.append(camera.project(position))
        # integrate one step; reflect off the front wall plane y=0
        velocity_next = velocity + GRAVITY_VEC * dt
        position_next = position + velocity * dt + 0.5 * GRAVITY_VEC * dt * dt
        if position_next[1] <= 0.0 and t_hit is None:
            fraction = position[1] / (position[1] - position_next[1])
            t_hit = t + fraction * dt
            hit_point = position + (position_next - position) * fraction
            position_next[1] = -position_next[1]
            velocity_next[1] = -0.7 * velocity_next[1]  # restitution
        position, velocity = position_next, velocity_next
        t += dt
    return (np.asarray(times), np.asarray(pixels), t_hit, hit_point)


def test_segment_track_finds_wall_bounce():
    camera = make_camera()
    times, pixels, t_hit, _ = _bounce_trajectory(camera)
    arcs = ballistic.segment_track(times, pixels, camera,
                                   rms_px=3.0, min_points=5)
    fitted = [a for a in arcs if isinstance(a, BallisticArc)]
    assert len(fitted) == 2
    boundary_time = times[fitted[0].end]
    assert boundary_time == pytest.approx(t_hit, abs=3.0 / 60.0)


def test_refine_impact_locates_wall_contact():
    camera = make_camera()
    times, pixels, t_hit, hit_point = _bounce_trajectory(camera)
    arcs = [a for a in ballistic.segment_track(times, pixels, camera, 3.0, 5)
            if isinstance(a, BallisticArc)]
    arc_a, arc_b = arcs
    t_star, point, v_in, v_out = ballistic.refine_impact(
        arc_a, arc_b, times[arc_a.end - 1], times[arc_b.start])
    assert t_star == pytest.approx(t_hit, abs=1.5 / 60.0)
    assert point[1] == pytest.approx(0.0, abs=1.0)       # on the front wall
    assert np.allclose(point, hit_point, atol=1.5)
    assert v_in[1] < 0 < v_out[1]                        # depth reversal


def test_arc_boundary_events_shape():
    camera = make_camera()
    times, pixels, _, _ = _bounce_trajectory(camera)
    frames = np.arange(len(times))
    cfg = {"arc3d_rms_px": 3.0, "arc_min_points": 5}
    events = ballistic.arc_boundary_events(
        frames, times, pixels, [(0, len(times))], camera, cfg)
    assert len(events) == 1
    event = events[0]
    assert event["methods"] == {"ballistic"}
    assert "contact_3d" in event
    assert len(event["contact_3d"]["point_ft"]) == 3
    for key in ("v_in", "v_out", "speed_before", "dv_magnitude", "turn_degrees"):
        assert event[key] is not None
