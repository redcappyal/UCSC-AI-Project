"""ARCHIVED 2026-07-27 -- see archive/stereo/README.md. Not collected by pytest.

The two-clip fusion half of the former `tests/test_stereo_offline_fuse.py`: the
`fuse_clips` geometry tests, the CLI smoke test, and the detector-provenance
tests that reach the detector through `fuse_clips`. The single-camera half
stayed as `tests/test_ball_track_offline.py`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest

import stereo_offline_fuse
from test_camera_model import _synthetic_calibration
from test_stereo_track import make_fin_pair, simulate_front_wall_shot


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

    monkeypatch.setattr(stereo_offline_fuse, "_video_fps", lambda video_path: fps)
    monkeypatch.setattr(stereo_offline_fuse, "_iter_frames", lambda video_path: iter(frames))

    result = stereo_offline_fuse.fuse_clips(
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
        stereo_offline_fuse.fuse_clips("a.mp4", calibration_a, "b.mp4", {})


def test_fuse_clips_raises_naming_failing_side_a():
    cam_a, cam_b = make_fin_pair()
    calibration_b = _synthetic_calibration(cam_b)

    with pytest.raises(ValueError, match="'a'"):
        stereo_offline_fuse.fuse_clips("a.mp4", {}, "b.mp4", calibration_b)


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
    monkeypatch.setattr(stereo_offline_fuse, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(stereo_offline_fuse, "_iter_frames", lambda video_path: iter([]))
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
        "stereo_offline_fuse.py",
        "--video-a", "a.mp4", "--calib-a", str(calib_a_path),
        "--video-b", "b.mp4", "--calib-b", str(calib_b_path),
        "--out", str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    stereo_offline_fuse.main()

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
    monkeypatch.setattr(stereo_offline_fuse, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(stereo_offline_fuse, "_iter_frames",
                        lambda video_path: iter(_fake_frames(5)))

    result = stereo_offline_fuse.fuse_clips(
        "a.mp4", calibration_a, "b.mp4", calibration_b,
        infer_a=lambda frame: [], infer_b=lambda frame: [])

    assert result["detector"] == {"backend": "injected"}


def test_fuse_clips_reports_rfdetr_detector(monkeypatch):
    monkeypatch.setenv("STEREO_DETECTOR", "rfdetr")
    cam_a, cam_b = make_fin_pair()
    calibration_a = _synthetic_calibration(cam_a)
    calibration_b = _synthetic_calibration(cam_b)
    monkeypatch.setattr(stereo_offline_fuse, "_video_fps", lambda video_path: 60.0)
    # Zero decoded frames: never builds `infer`, so this stays free of the
    # heavy rfdetr/torch import while still exercising the provenance stamp.
    monkeypatch.setattr(stereo_offline_fuse, "_iter_frames", lambda video_path: iter([]))

    result = stereo_offline_fuse.fuse_clips(
        "a.mp4", calibration_a, "b.mp4", calibration_b)

    assert result["detector"] == {"backend": "rfdetr"}
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
    monkeypatch.setattr(stereo_offline_fuse, "_video_fps", lambda video_path: 60.0)
    monkeypatch.setattr(stereo_offline_fuse, "_iter_frames", lambda video_path: iter([]))

    with pytest.raises(ValueError, match="STEREO_DETECTOR"):
        stereo_offline_fuse.fuse_clips("a.mp4", calibration_a, "b.mp4", calibration_b)
