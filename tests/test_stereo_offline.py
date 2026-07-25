import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import stereo_offline
from test_camera_model import _synthetic_calibration
from test_stereo_track import make_fin_pair, simulate_front_wall_shot


def _fake_frames(n):
    # Frame "content" is just its own index -- the fake infer callables in
    # these tests only need the index to know which prediction to return, so
    # there is no need for real pixel data (and no cv2 dependency).
    return [(i, i) for i in range(n)]


# --- detections_to_track_samples: time-base math ---------------------------


def test_detections_to_track_samples_time_base_and_offset(monkeypatch):
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        stereo_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def fake_infer(frame):
        return [{"x": 100.0, "y": 200.0, "width": 10.0, "height": 10.0,
                 "confidence": 0.9, "class": "ball"}]

    samples = stereo_offline.detections_to_track_samples(
        "video-a.mp4", confidence=0.4, offset_s=1.5, infer=fake_infer)

    assert len(samples) == 10
    for i, sample in enumerate(samples):
        assert sample.t_s == pytest.approx(i / 60.0 + 1.5)
        assert sample.px == (100.0, 200.0)


def test_detections_to_track_samples_stride_halves_count(monkeypatch):
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        stereo_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def fake_infer(frame):
        return [{"x": 1.0, "y": 1.0, "width": 10.0, "height": 10.0,
                 "confidence": 0.9, "class": "ball"}]

    samples = stereo_offline.detections_to_track_samples(
        "video-a.mp4", stride=2, infer=fake_infer)
    assert len(samples) == 5
    assert [round(s.t_s * 60.0) for s in samples] == [0, 2, 4, 6, 8]


def test_detections_to_track_samples_drops_below_confidence(monkeypatch):
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        stereo_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def low_conf_infer(frame):
        return [{"x": 1.0, "y": 1.0, "width": 5.0, "height": 5.0,
                 "confidence": 0.1, "class": "ball"}]

    samples = stereo_offline.detections_to_track_samples(
        "video-a.mp4", confidence=0.4, infer=low_conf_infer)
    assert samples == []


def test_detections_to_track_samples_skips_no_ball_frames(monkeypatch):
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        stereo_offline, "_iter_frames", lambda video_path: iter(_fake_frames(10)))

    def no_ball_infer(frame):
        return []

    samples = stereo_offline.detections_to_track_samples(
        "video-a.mp4", infer=no_ball_infer)
    assert samples == []


def test_detections_to_track_samples_raises_on_bad_fps(monkeypatch):
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 0.0)
    monkeypatch.setattr(
        stereo_offline, "_iter_frames", lambda video_path: iter(_fake_frames(1)))

    with pytest.raises(ValueError):
        stereo_offline.detections_to_track_samples("video-a.mp4", infer=lambda frame: [])


@pytest.mark.parametrize("bad_fps", [float("nan"), float("inf"), -30.0])
def test_detections_to_track_samples_raises_on_nonfinite_fps(monkeypatch, bad_fps):
    """NaN is truthy and every NaN comparison is False, so a `not fps or
    fps <= 0` guard alone lets it through and every t_s silently becomes
    NaN -- which only blows up later, deep inside stereo_engine."""
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: bad_fps)
    monkeypatch.setattr(
        stereo_offline, "_iter_frames", lambda video_path: iter(_fake_frames(1)))

    with pytest.raises(ValueError, match="Invalid fps"):
        stereo_offline.detections_to_track_samples("video-a.mp4", infer=lambda frame: [])


def test_detections_to_track_samples_rejects_stride_below_one(monkeypatch):
    """stride=0 would otherwise raise a raw ZeroDivisionError from the
    `frame_index % stride` guard on the very first frame."""
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(
        stereo_offline, "_iter_frames", lambda video_path: iter(_fake_frames(1)))

    with pytest.raises(ValueError, match="stride"):
        stereo_offline.detections_to_track_samples(
            "video-a.mp4", stride=0, infer=lambda frame: [])


# --- fuse_clips: synthetic end-to-end geometry ------------------------------


def _nearest_state_xyz(states, t_query):
    return min(states, key=lambda s: abs(s[0] - t_query))[1]


def _projecting_infer(camera, fps, offset_s):
    """Fake detector: given the frame index (used as the fake frame content),
    project the ground-truth ballistic position at that sample's timestamp
    through `camera` -- reproducing what a perfect detector would report."""

    def _infer(frame_index):
        t_query = frame_index / fps + offset_s
        xyz = _nearest_state_xyz(_STATES, t_query)
        try:
            u, v = camera.project(xyz)
        except ValueError:
            return []
        return [{"x": float(u), "y": float(v), "width": 8.0, "height": 8.0,
                 "confidence": 0.9, "class": "ball"}]

    return _infer


_STATES, _IMPACT = simulate_front_wall_shot()


def test_fuse_clips_end_to_end_synthetic(monkeypatch):
    cam_a, cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)
    calibration_b = _synthetic_calibration(cam_b)

    fps = 60.0
    offset_s_b = 0.007
    n_frames = 55
    frames = _fake_frames(n_frames)

    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: fps)
    monkeypatch.setattr(stereo_offline, "_iter_frames", lambda video_path: iter(frames))

    result = stereo_offline.fuse_clips(
        "clip-a.mp4", calibration_a, "clip-b.mp4", calibration_b,
        offset_s_b=offset_s_b,
        infer_a=_projecting_infer(cam_a, fps, 0.0),
        infer_b=_projecting_infer(cam_b, fps, offset_s_b),
    )

    assert result["sample_counts"] == {"a": n_frames, "b": n_frames}

    front_wall_impacts = [i for i in result["impacts"] if i["surface"] == "front_wall"]
    assert len(front_wall_impacts) == 1
    impact = front_wall_impacts[0]
    assert impact["call"] == "in"
    assert impact["confidence"] == "high"

    assert result["pair_agreement"]["ok_pair"] is True


def test_fuse_clips_raises_naming_failing_side():
    cam_a, _cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)

    with pytest.raises(ValueError, match="'b'"):
        stereo_offline.fuse_clips("a.mp4", calibration_a, "b.mp4", {})


def test_fuse_clips_raises_naming_failing_side_a():
    cam_a, cam_b = make_fin_pair()
    calibration_b = _synthetic_calibration(cam_b)

    with pytest.raises(ValueError, match="'a'"):
        stereo_offline.fuse_clips("a.mp4", {}, "b.mp4", calibration_b)


# --- CLI smoke ---------------------------------------------------------------


def test_main_cli_smoke(tmp_path, monkeypatch):
    cam_a, cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)
    calibration_b = _synthetic_calibration(cam_b)

    calib_a_path = tmp_path / "calib_a.json"
    calib_b_path = tmp_path / "calib_b.json"
    calib_a_path.write_text(json.dumps(calibration_a), encoding="utf-8")
    calib_b_path.write_text(json.dumps(calibration_b), encoding="utf-8")
    out_path = tmp_path / "impacts.json"

    # Zero decoded frames on both sides: exercises the CLI plumbing without
    # ever needing the real detector (get_tracking_model), keeping this test
    # free of cv2/inference dependencies.
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(stereo_offline, "_iter_frames", lambda video_path: iter([]))
    # Ambient STEREO_DETECTOR must not change this test's outcome -- the
    # module docstring advertises STEREO_DETECTOR=rfdetr as a valid override,
    # and a developer with it exported would otherwise see this fail.
    monkeypatch.delenv("STEREO_DETECTOR", raising=False)
    # Point at a directory guaranteed not to exist, rather than relying on no
    # model being present at ball_model.DEFAULT_MODEL_DIR: a later task
    # exports a real model to exactly that path, and once it does,
    # describe() would start returning a dict instead of None, silently
    # breaking this assertion on any machine where the export ran (while
    # staying green elsewhere, since models/ is gitignored).
    monkeypatch.setenv("BALL_MODEL_DIR", str(tmp_path / "no-model"))

    argv = [
        "stereo_offline.py",
        "--video-a", "a.mp4", "--calib-a", str(calib_a_path),
        "--video-b", "b.mp4", "--calib-b", str(calib_b_path),
        "--out", str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    stereo_offline.main()

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(result.keys()) == {
        "impacts", "pair_agreement", "sample_counts", "detector"}
    assert result["sample_counts"] == {"a": 0, "b": 0}
    assert result["impacts"] == []
    # No model is present at BALL_MODEL_DIR; describe() is best-effort and
    # reports that absence rather than raising.
    assert result["detector"] == {"backend": "yolox", "model": None}


# --- detector provenance ----------------------------------------------------


def test_fuse_clips_reports_injected_detector(monkeypatch):
    # Injected infer means no model is loaded; provenance must say so rather
    # than reading a manifest that is not there. This is what keeps the
    # existing synthetic tests working with no model on disk.
    cam_a, cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)
    calibration_b = _synthetic_calibration(cam_b)
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(stereo_offline, "_iter_frames",
                        lambda video_path: iter(_fake_frames(5)))

    result = stereo_offline.fuse_clips(
        "a.mp4", calibration_a, "b.mp4", calibration_b,
        infer_a=lambda frame: [], infer_b=lambda frame: [])

    assert result["detector"] == {"backend": "injected"}


def test_fuse_clips_reports_rfdetr_detector(monkeypatch):
    monkeypatch.setenv("STEREO_DETECTOR", "rfdetr")
    cam_a, cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)
    calibration_b = _synthetic_calibration(cam_b)
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    # Zero decoded frames: never builds `infer`, so this stays free of the
    # heavy rfdetr/torch import while still exercising the provenance stamp.
    monkeypatch.setattr(stereo_offline, "_iter_frames", lambda video_path: iter([]))

    result = stereo_offline.fuse_clips(
        "a.mp4", calibration_a, "b.mp4", calibration_b)

    assert result["detector"] == {"backend": "rfdetr"}


def test_selected_detector_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("STEREO_DETECTOR", "rfdtr")  # typo of "rfdetr"
    with pytest.raises(ValueError, match="STEREO_DETECTOR"):
        stereo_offline.selected_detector()


def test_fuse_clips_rejects_unknown_detector_on_zero_frame_path(monkeypatch):
    """Regression for the bug finding 2 fixed: fuse_clips used to re-derive
    the backend with its own unconditional `if backend == "rfdetr": ... else
    yolox`, so an unknown STEREO_DETECTOR value was only caught when
    _build_infer ran. On a clip that decodes zero frames, _build_infer is
    never reached (it is constructed lazily inside the frame loop), so the
    run used to succeed and silently stamp "yolox" despite no detector ever
    running. Validation now lives in selected_detector(), so every caller
    -- including this zero-frame path -- raises."""
    monkeypatch.setenv("STEREO_DETECTOR", "rfdtr")
    cam_a, cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)
    calibration_b = _synthetic_calibration(cam_b)
    monkeypatch.setattr(stereo_offline, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(stereo_offline, "_iter_frames", lambda video_path: iter([]))

    with pytest.raises(ValueError, match="STEREO_DETECTOR"):
        stereo_offline.fuse_clips("a.mp4", calibration_a, "b.mp4", calibration_b)


def test_build_infer_defaults_to_yolox_for_stereo(monkeypatch):
    # Ambient STEREO_DETECTOR must not change this test's outcome -- the
    # module docstring advertises STEREO_DETECTOR=rfdetr as a valid override,
    # and a developer with it exported would otherwise see this test wander
    # into the real inference_engine import.
    monkeypatch.delenv("STEREO_DETECTOR", raising=False)
    calls = {}

    class _Manifest:
        conf_threshold = 0.1  # below the 0.4 confidence used below

    _manifest = _Manifest()

    class _Runner:
        manifest = _manifest

    runner_instance = _Runner()

    monkeypatch.setattr(stereo_offline, "_load_ball_detector",
                        lambda: runner_instance)
    monkeypatch.setattr(
        stereo_offline, "_detect_frame",
        lambda runner, frame, manifest: calls.setdefault(
            "args", (runner, frame, manifest)) or [])

    infer = stereo_offline._build_infer(None, 0.4)
    infer("FRAME")

    # Proves the runner _load_ball_detector() returned is the same object
    # handed to detect_frame, not just any object with a `.manifest`.
    assert calls["args"][0] is runner_instance
    assert calls["args"][1] == "FRAME"
    assert calls["args"][2] is _manifest


def test_build_infer_honours_rfdetr_override(monkeypatch):
    monkeypatch.setenv("STEREO_DETECTOR", "rfdetr")
    seen = {}

    def _fake_import():
        def _get_model():
            return "RFDETR_MODEL"

        def _infer(model, frame, confidence):
            seen["model"] = model
            return []

        return _get_model, _infer

    monkeypatch.setattr(stereo_offline, "_import_rfdetr", _fake_import)
    infer = stereo_offline._build_infer(None, 0.4)
    infer("FRAME")

    assert seen["model"] == "RFDETR_MODEL"
