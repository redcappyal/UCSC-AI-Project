"""Offline ball tracking of one clip: time-base math and detector selection.

The single-camera half of the former `tests/test_ball_track_offline.py`. The
`fuse_clips` / CLI half moved to archive/stereo/tests/ on 2026-07-27.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import ball_track_offline


def _fake_frames(n):
    # Frame "content" is just its own index -- the fake infer callables in
    # these tests only need the index to know which prediction to return, so
    # there is no need for real pixel data (and no cv2 dependency).
    return [(i, i) for i in range(n)]


# --- detections_to_track_samples: time-base math ---------------------------


def test_detections_to_track_samples_time_base_and_offset(monkeypatch):
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        ball_track_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def fake_infer(frame):
        return [{"x": 100.0, "y": 200.0, "width": 10.0, "height": 10.0,
                 "confidence": 0.9, "class": "ball"}]

    samples = ball_track_offline.detections_to_track_samples(
        "video-a.mp4", confidence=0.4, offset_s=1.5, infer=fake_infer)

    assert len(samples) == 10
    for i, sample in enumerate(samples):
        assert sample.t_s == pytest.approx(i / 60.0 + 1.5)
        assert sample.px == (100.0, 200.0)


def test_detections_to_track_samples_stride_halves_count(monkeypatch):
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        ball_track_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def fake_infer(frame):
        return [{"x": 1.0, "y": 1.0, "width": 10.0, "height": 10.0,
                 "confidence": 0.9, "class": "ball"}]

    samples = ball_track_offline.detections_to_track_samples(
        "video-a.mp4", stride=2, infer=fake_infer)
    assert len(samples) == 5
    assert [round(s.t_s * 60.0) for s in samples] == [0, 2, 4, 6, 8]


def test_detections_to_track_samples_drops_below_confidence(monkeypatch):
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        ball_track_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def low_conf_infer(frame):
        return [{"x": 1.0, "y": 1.0, "width": 5.0, "height": 5.0,
                 "confidence": 0.1, "class": "ball"}]

    samples = ball_track_offline.detections_to_track_samples(
        "video-a.mp4", confidence=0.4, infer=low_conf_infer)
    assert samples == []


def test_detections_to_track_samples_skips_no_ball_frames(monkeypatch):
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        ball_track_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def no_ball_infer(frame):
        return []

    samples = ball_track_offline.detections_to_track_samples(
        "video-a.mp4", infer=no_ball_infer)
    assert samples == []


def test_detections_to_track_samples_raises_on_bad_fps(monkeypatch):
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda video_path: 0.0)
    monkeypatch.setattr(
        ball_track_offline, "_iter_frames", lambda video_path: iter(_fake_frames(1)))

    with pytest.raises(ValueError):
        ball_track_offline.detections_to_track_samples("video-a.mp4", infer=lambda frame: [])


@pytest.mark.parametrize("bad_fps", [float("nan"), float("inf"), -30.0])
def test_detections_to_track_samples_raises_on_nonfinite_fps(monkeypatch, bad_fps):
    """NaN is truthy and every NaN comparison is False, so a `not fps or
    fps <= 0` guard alone lets it through and every t_s silently becomes
    NaN -- which only blows up later, far from the cause."""
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda video_path: bad_fps)
    monkeypatch.setattr(
        ball_track_offline, "_iter_frames", lambda video_path: iter(_fake_frames(1)))

    with pytest.raises(ValueError, match="Invalid fps"):
        ball_track_offline.detections_to_track_samples("video-a.mp4", infer=lambda frame: [])


def test_detections_to_track_samples_rejects_stride_below_one(monkeypatch):
    """stride=0 would otherwise raise a raw ZeroDivisionError from the
    `frame_index % stride` guard on the very first frame."""
    monkeypatch.setattr(ball_track_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        ball_track_offline, "_iter_frames", lambda video_path: iter(_fake_frames(1)))

    with pytest.raises(ValueError, match="stride"):
        ball_track_offline.detections_to_track_samples(
            "video-a.mp4", stride=0, infer=lambda frame: [])




def test_selected_detector_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "rfdtr")  # typo of "rfdetr"
    with pytest.raises(ValueError, match="BALL_DETECTOR"):
        ball_track_offline.selected_detector()




def test_build_infer_defaults_to_yolox(monkeypatch):
    # Ambient BALL_DETECTOR must not change this test's outcome -- the
    # module docstring advertises BALL_DETECTOR=rfdetr as a valid override,
    # and a developer with it exported would otherwise see this test wander
    # into the real inference_engine import.
    monkeypatch.delenv("BALL_DETECTOR", raising=False)
    calls = {}

    class _Manifest:
        conf_threshold = 0.1  # below the 0.4 confidence used below

    _manifest = _Manifest()

    class _Runner:
        manifest = _manifest

    runner_instance = _Runner()

    monkeypatch.setattr(ball_track_offline, "_load_ball_detector",
                        lambda: runner_instance)
    monkeypatch.setattr(
        ball_track_offline, "_detect_frame",
        lambda runner, frame, manifest: calls.setdefault(
            "args", (runner, frame, manifest)) or [])

    infer = ball_track_offline._build_infer(None, 0.4)
    infer("FRAME")

    # Proves the runner _load_ball_detector() returned is the same object
    # handed to detect_frame, not just any object with a `.manifest`.
    assert calls["args"][0] is runner_instance
    assert calls["args"][1] == "FRAME"
    assert calls["args"][2] is _manifest


def test_build_infer_honours_rfdetr_override(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "rfdetr")
    seen = {}

    def _fake_import():
        def _get_model():
            return "RFDETR_MODEL"

        def _infer(model, frame, confidence):
            seen["model"] = model
            return []

        return _get_model, _infer

    monkeypatch.setattr(ball_track_offline, "_import_rfdetr", _fake_import)
    infer = ball_track_offline._build_infer(None, 0.4)
    infer("FRAME")

    assert seen["model"] == "RFDETR_MODEL"


def test_build_infer_rejects_model_without_manifest(monkeypatch):
    """The RF-DETR-object-under-yolox-default case: passing a `model` with no
    `.manifest` attribute while BALL_DETECTOR selects yolox (the default)
    must raise TypeError up front, not wander into an opaque AttributeError
    mid-frame. A defensive monkeypatch on _load_ball_detector proves the
    guard fires before ever reaching that seam -- if the guard regressed,
    this would blow up on the loader instead of failing the assertion below,
    which would still fail the test either way, but this pins down why."""
    monkeypatch.delenv("BALL_DETECTOR", raising=False)

    def _unexpected_load():
        raise AssertionError(
            "guard should have raised before loading a detector")

    monkeypatch.setattr(ball_track_offline, "_load_ball_detector", _unexpected_load)

    with pytest.raises(TypeError, match="manifest"):
        ball_track_offline._build_infer(object(), 0.4)


def test_build_infer_rejects_unreachable_confidence(monkeypatch):
    """If the manifest's conf_threshold sits above the caller's requested
    confidence, the caller's threshold would be silently unreachable dead
    code -- the detector already drops everything below its own
    conf_threshold before selection ever sees it. A defensive monkeypatch on
    _detect_frame proves the guard fires before any inference runs."""
    monkeypatch.delenv("BALL_DETECTOR", raising=False)

    class _Manifest:
        conf_threshold = 0.9  # above the 0.4 confidence used below

    class _Runner:
        manifest = _Manifest()

    def _unexpected_detect(runner, frame, manifest):
        raise AssertionError(
            "guard should have raised before running inference")

    monkeypatch.setattr(ball_track_offline, "_detect_frame", _unexpected_detect)

    with pytest.raises(ValueError, match="conf_threshold"):
        ball_track_offline._build_infer(_Runner(), 0.4)
