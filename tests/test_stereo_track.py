import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import stereo_engine
from stereo_engine import TrackSample
from synthetic3d import make_camera

GRAVITY_FT_S2 = -32.174


def make_fin_pair():
    left = make_camera(position=(9.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    right = make_camera(position=(12.0, 31.95, 7.0), look_at=(10.5, 0.0, 5.0))
    return left, right


def simulate_front_wall_shot(dt=1.0 / 480.0):
    """Ballistic flight into the front wall; returns (states, impact) where
    states is [(t, xyz)] and impact is the exact (t, xyz) of the wall hit."""
    pos = np.array([16.0, 26.0, 4.0])
    vel = np.array([-6.0, -55.0, 9.0])
    t = 0.0
    states, impact = [], None
    while t < 0.9:
        states.append((t, pos.copy()))
        nxt = pos + vel * dt + 0.5 * np.array([0, 0, GRAVITY_FT_S2]) * dt * dt
        vel = vel + np.array([0, 0, GRAVITY_FT_S2]) * dt
        if impact is None and nxt[1] <= 0.0:
            frac = pos[1] / (pos[1] - nxt[1])
            impact = (t + frac * dt, pos + (nxt - pos) * frac)
            vel[1] = -vel[1] * 0.7
            nxt = pos + vel * dt
        pos = nxt
        t += dt
    return states, impact


def sample_camera(states, model, fps=60.0, phase_s=0.0):
    samples, next_t = [], phase_s
    for t, xyz in states:
        if t >= next_t:
            try:
                samples.append(TrackSample(t_s=t, px=tuple(model.project(xyz))))
            except ValueError:
                pass
            next_t += 1.0 / fps
    return samples


def test_eval_pixel_track_matches_projection():
    left, _ = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples = sample_camera(states, left, fps=60.0)
    t_query = samples[6].t_s + 0.004        # off-sample time, pre-impact
    px = stereo_engine.eval_pixel_track(samples, t_query)
    true_xyz = min(states, key=lambda s: abs(s[0] - t_query))[1]
    assert np.linalg.norm(px - np.asarray(left.project(true_xyz))) < 3.0


def test_build_track3d_tracks_flight():
    left, right = make_fin_pair()
    states, impact = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    track = stereo_engine.build_track3d(left, samples_a, right, samples_b)
    assert len(track) > 40
    mid = track[len(track) // 4]
    true_xyz = min(states, key=lambda s: abs(s[0] - mid.t_s))[1]
    # Depth (y) is the weak axis at this baseline; lateral/height are tight.
    assert abs(mid.point_ft[0] - true_xyz[0]) < 0.3
    assert abs(mid.point_ft[2] - true_xyz[2]) < 0.3


def test_detect_impacts_front_wall_call_and_position():
    left, right = make_fin_pair()
    states, (t_true, p_true) = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)
    front = [i for i in impacts if i.surface == "front_wall"]
    assert len(front) == 1
    hit = front[0]
    assert abs(hit.t_s - t_true) < 0.012
    assert hit.confidence == "high"
    assert np.linalg.norm(hit.point_ft - p_true) < 0.15
    call, margin = stereo_engine.call_for_impact("front_wall", p_true)
    assert hit.call == call


def test_detect_impacts_one_view_when_occluded():
    left, right = make_fin_pair()
    states, (t_true, _) = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = [s for s in sample_camera(states, right, fps=60.0, phase_s=0.007)
                 if not (t_true - 0.3 <= s.t_s <= t_true + 0.1)]
    impacts = stereo_engine.detect_impacts(left, samples_a, right, samples_b)
    front = [i for i in impacts if i.surface == "front_wall"]
    assert len(front) == 1
    assert front[0].confidence == "one_view"


def test_build_track3d_accepts_explicit_timeline():
    left, right = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    default_track = stereo_engine.build_track3d(left, samples_a, right, samples_b)
    timeline = [p.t_s for p in default_track]
    replayed = stereo_engine.build_track3d(left, samples_a, right, samples_b,
                                           timeline_s=timeline)
    assert len(replayed) == len(default_track)
    for a, b in zip(replayed, default_track):
        assert a.t_s == b.t_s
        assert np.allclose(a.point_ft, b.point_ft, atol=0.0)
