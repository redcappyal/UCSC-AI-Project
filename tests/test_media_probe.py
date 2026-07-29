"""Probe tests for media_probe.

The probe exists so later stages can gate on what the footage can actually
support instead of assuming our own 4K60 capture. Everything asserted here is
something a capability gate reads: fps and width decide the ball tier,
sharpness separates a usable clip from a motion-blurred one, and has_audio
tells the rally timeline whether the audio channel is worth extracting.

Clips are synthesised with cv2.VideoWriter (mp4v, not avc1: Linux
opencv-python-headless ships no H.264 encoder and a failed VideoWriter drops no
file rather than raising) following tests/test_pipeline.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import media_probe


def _write_clip(path, fps=30.0, size=(64, 48), frames=30, blur=False):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    assert writer.isOpened(), "mp4v codec unavailable?"
    rng = np.random.default_rng(7)
    for _ in range(frames):
        frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        if blur:
            frame = cv2.GaussianBlur(frame, (15, 15), 6.0)
        writer.write(frame)
    writer.release()
    assert path.exists(), "VideoWriter produced no file (codec unavailable?)"


def test_probe_reports_fps_size_and_frame_count(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    probe = media_probe.probe_video(clip)

    assert round(probe["fps"]) == 30
    assert (probe["width"], probe["height"]) == (64, 48)
    assert probe["frame_count"] == 30


def test_probe_reports_no_audio_for_a_video_only_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    assert media_probe.probe_video(clip)["has_audio"] is False


def test_duration_is_frames_over_fps(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, fps=30.0, frames=45)

    probe = media_probe.probe_video(clip)

    assert probe["duration_s"] == pytest.approx(1.5, abs=0.05)


def test_sharpness_orders_sharp_above_blurred(tmp_path):
    """The ball tier's blur gate is only meaningful if this ordering holds."""
    sharp, soft = tmp_path / "sharp.mp4", tmp_path / "soft.mp4"
    _write_clip(sharp)
    _write_clip(soft, blur=True)

    assert (media_probe.probe_video(sharp)["sharpness"]
            > media_probe.probe_video(soft)["sharpness"])


def test_zero_fps_falls_back_to_30(tmp_path, monkeypatch):
    """Containers that report 0 fps must not divide the duration by zero.

    app.video_info already guards this exact case with `or 30.0`; the probe
    feeds capability gates, so an fps of 0 there would silently disable the
    ball tier on footage that qualifies.
    """
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    real_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return real_get(self, prop)

    monkeypatch.setattr(cv2.VideoCapture, "get", fake_get)

    probe = media_probe.probe_video(clip)

    assert probe["fps"] == 30.0
    assert probe["duration_s"] > 0


def test_unreadable_file_raises_rather_than_reporting_zeroes(tmp_path):
    """A probe that cannot open the file must not answer "0 fps, no audio".

    That answer is indistinguishable from a real unqualified clip, and the
    capability card would then state a reason that was never measured.
    """
    missing = tmp_path / "not_a_video.mp4"
    missing.write_bytes(b"not a video")

    with pytest.raises(ValueError):
        media_probe.probe_video(missing)


def test_sharpness_is_none_when_no_frame_decodes(tmp_path, monkeypatch):
    """Distinguishes "measured as blurry" from "could not measure"."""
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    monkeypatch.setattr(cv2.VideoCapture, "read", lambda self: (False, None))

    assert media_probe.probe_video(clip)["sharpness"] is None


def test_sharpness_samples_at_most_16_frames(tmp_path, monkeypatch):
    """A 40-minute match must not be decoded frame by frame to be probed."""
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, frames=200)

    reads = []
    real_read = cv2.VideoCapture.read

    def counting_read(self):
        reads.append(1)
        return real_read(self)

    monkeypatch.setattr(cv2.VideoCapture, "read", counting_read)

    media_probe.probe_video(clip)

    assert len(reads) <= 16
