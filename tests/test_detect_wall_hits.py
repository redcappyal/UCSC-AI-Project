"""fps-normalization of the time-domain detector windows.

Every frame-count constant in detect_wall_hits was tuned against 60 fps
footage and expressed in frames, so at 30 fps each window silently covers
twice the wall-clock it was chosen for. That is not a tuning disagreement --
it is the same detector being handed a different question depending on what
the phone happened to record at.

These tests pin the property that makes the change safe to land: at the 60 fps
reference the scaled values are *identical* to the constants, so nothing about
the existing corpus (all ~60 fps) can move. The scaling only has an effect on
footage the old constants were never valid for in the first place.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from detect_wall_hits import (
    MAX_GAP_FRAMES,
    MIN_GAP_FRAMES,
    REFERENCE_FPS,
    SMOOTH_WINDOW,
    detect_hits_from_rows,
    scaled_hit_kwargs,
    smooth_positions,
)
from tracking_common import MOTION_TRACK_WINDOW_FRAMES, scaled_window_frames


def test_scaled_kwargs_identity_at_reference_fps():
    """The whole safety argument for this refactor, in one assertion."""
    assert scaled_hit_kwargs(REFERENCE_FPS) == {
        "max_gap": MAX_GAP_FRAMES,
        "min_gap": MIN_GAP_FRAMES,
        "smooth": SMOOTH_WINDOW,
    }


def test_reference_is_the_fps_the_constants_were_tuned_at():
    assert REFERENCE_FPS == 60.0


def test_scaled_kwargs_halve_at_30fps():
    scaled = scaled_hit_kwargs(30.0)

    assert scaled["min_gap"] == 5
    assert scaled["max_gap"] == 2


def test_windows_never_collapse_to_zero_frames():
    """A window of 0 frames is not a smaller window, it is a disabled one.

    Slow footage would otherwise turn `smooth` and `max_gap` off entirely at
    the exact frame rates where the detector most needs them.
    """
    for fps in (1.0, 5.0, 12.0):
        scaled = scaled_hit_kwargs(fps)
        assert min(scaled.values()) >= 1, fps


def test_scaled_kwargs_grow_on_faster_footage():
    scaled = scaled_hit_kwargs(120.0)

    assert scaled["min_gap"] == MIN_GAP_FRAMES * 2
    assert scaled["max_gap"] == MAX_GAP_FRAMES * 2


def test_stride_floor_survives_scaling():
    """Mirrors the job_runner wiring rule: the coarse stride wins when larger.

    Coarse samples are `frame_stride` apart, so a max_gap below the stride
    splits every track. Scaling max_gap down at low fps makes that *more*
    likely, which is exactly why the floor is applied after scaling.
    """
    assert max(scaled_hit_kwargs(REFERENCE_FPS)["max_gap"], 4) == 4
    assert max(scaled_hit_kwargs(30.0)["max_gap"], 4) == 4


def test_motion_window_identity_at_reference_fps():
    assert scaled_window_frames(REFERENCE_FPS) == MOTION_TRACK_WINDOW_FRAMES


def test_motion_window_keeps_enough_frames_to_have_a_direction():
    """Two points is the minimum that defines a motion direction at all."""
    for fps in (1.0, 6.0, 30.0):
        assert scaled_window_frames(fps) >= 2, fps


@pytest.mark.parametrize("window", [2, 3, 4, 5])
def test_smooth_positions_preserves_sample_count(window):
    """fps scaling hands the smoother even windows (smooth=2 at exactly 30
    fps), which crashed with a broadcast error: a symmetric window//2 pad
    yields n+1 "valid" convolution samples for an n-sample track."""
    positions = np.arange(24, dtype=np.float64).reshape(12, 2)
    smoothed = smooth_positions(positions, window)

    assert smoothed.shape == positions.shape


def test_smooth_positions_keeps_constant_tracks_exact():
    """The mean filter must be a no-op on a stationary track at any parity."""
    positions = np.full((10, 2), 7.5)
    for window in (2, 3, 4):
        assert np.allclose(smooth_positions(positions, window), positions)


def test_thirty_fps_kwargs_run_the_detector_end_to_end():
    """The prod-shaped regression: exactly-30fps footage produced smooth=2,
    and detect_hits_from_rows crashed before reaching any candidate logic."""
    rows = [
        {
            "source_frame": frame,
            "timestamp_seconds": f"{frame / 30.0:.6f}",
            "detected": "True",
            "x_center": f"{200.0 + 5.0 * frame:.3f}",
            "y_center": f"{600.0 - 400.0 * abs(frame / 30.0 - 0.5):.3f}",
        }
        for frame in range(30)
    ]

    hits = detect_hits_from_rows(rows, **scaled_hit_kwargs(30.0))

    assert isinstance(hits, list)


@pytest.mark.parametrize("fps", [0.0, -1.0, None])
def test_a_missing_frame_rate_falls_back_to_the_reference(fps):
    """Containers report 0 fps often enough that this cannot raise.

    Falling back to the reference reproduces today's behaviour exactly, which
    is the conservative choice: the alternative silently rescales every window
    toward zero on the clips whose metadata is worst.
    """
    assert scaled_hit_kwargs(fps) == scaled_hit_kwargs(REFERENCE_FPS)
    assert scaled_window_frames(fps) == MOTION_TRACK_WINDOW_FRAMES
