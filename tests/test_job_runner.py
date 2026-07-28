"""Integration tests for job_runner.run_tracking_job.

Exercises the full pipeline end-to-end over a tiny synthetic clip, with the
ball model and audio analysis stubbed out (this module never imports
rfdetr or a real ball model) so the two things under test stay real: the
person-detector seam (PersonFramePass wired to a stub detector) and the
"person layer degrades, never fails the job" invariant when
PERSON_DETECTOR=none (CLAUDE.md, fix from job_runner's
person_pass = build_person_pass(...) None-safety).
"""

import cv2
import numpy as np

import job_runner
import person_model
from person_model import PersonDetection


def _write_clip(path, frame_count=12, width=64, height=48, fps=30.0):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for i in range(frame_count):
        writer.write(np.full((height, width, 3), i * 20, dtype=np.uint8))
    writer.release()


def _det(x, y, confidence=0.9):
    return PersonDetection(
        x=x, y=y, width=40.0, height=100.0, confidence=confidence, keypoints=()
    )


def _make_job(tmp_path, run_id, video_path):
    """Writes job.json with the fields run_tracking_job reads (start/end
    frame, fps, stride, inference width, video/run paths) -- the same set
    app.py's /api/track route passes to create_job."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    job_runner.create_job(
        run_id,
        run_dir,
        status="queued",
        message="Queued tracking job.",
        video_path=str(video_path),
        start_frame=0,
        end_frame=11,
        fps=30.0,
        frame_stride=3,
        inference_width=64,
        processed_frames=0,
        total_frames=4,
    )
    return run_dir


def _stub_pipeline(monkeypatch):
    """No ball model, no audio -- both are off-topic for this test, and
    keeping them out avoids pulling in real model weights."""
    monkeypatch.setattr(job_runner, "get_tracking_model", lambda: object())
    monkeypatch.setattr(
        job_runner,
        "infer_frame_predictions",
        lambda model, frame, threshold, width: [],
    )
    monkeypatch.setattr(
        job_runner,
        "extract_audio_candidates",
        lambda video_path, start_frame, end_frame, fps: [],
    )


def test_run_tracking_job_with_stub_person_detector(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-stub-person"
    run_dir = _make_job(tmp_path, run_id, video_path)

    class StubDetector:
        backend = "stub"

        def detect(self, frame_bgr):
            return [_det(100, 300), _det(900, 300)]

    # person_model.load_person_detector is the seam build_person_pass calls;
    # patching it here keeps PersonFramePass (the real tracker/cadence logic)
    # in the loop, wired to a fake backend instead of rfdetr.
    monkeypatch.setattr(person_model, "load_person_detector", lambda: StubDetector())
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    players_v1 = job["players_v1"]
    assert players_v1["detector_backend"] == "stub"
    assert players_v1["attribution_backend"] in ("observed", "assumed")
    assert (run_dir / "players" / "track_samples.json").exists()


def test_run_tracking_job_person_detector_none(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-no-person"
    run_dir = _make_job(tmp_path, run_id, video_path)

    monkeypatch.setenv("PERSON_DETECTOR", "none")
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    players_v1 = job["players_v1"]
    assert players_v1["detector_backend"] == "none"
    assert players_v1["attribution_backend"] == "assumed"
    assert not (run_dir / "players").exists()
