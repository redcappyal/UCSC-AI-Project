"""Integration tests for job_runner.run_tracking_job.

Exercises the full pipeline end-to-end over a tiny synthetic clip, with the
ball model and audio analysis stubbed out (this module never imports
rfdetr or a real ball model) so the two things under test stay real: the
person-detector seam (PersonFramePass wired to a stub detector) and the
"person layer degrades, never fails the job" invariant when
PERSON_DETECTOR=none (CLAUDE.md, fix from job_runner's
person_pass = build_person_pass(...) None-safety).
"""

import json

import cv2
import numpy as np

import job_runner
import person_model
from person_model import PersonDetection
from test_court_model import make_v2_calibration

QUALIFIED_PROBE = {
    "fps": 60.0, "width": 1920, "height": 1080, "frame_count": 12,
    "duration_s": 0.2, "sharpness": 120.0, "has_audio": True,
}


def _qualify_for_ball_tier(run_id, run_dir):
    """Give a run footage and a court good enough to enable ball tracking.

    The ball tier is gated on frame rate, size, sharpness *and* a solved
    court. The person-detector seam below rides on the ball decode pass, so
    without this the gate skips the very stages those tests exercise -- they
    would still pass while testing nothing.
    """
    (run_dir / "calibration.json").write_text(
        json.dumps(make_v2_calibration()), encoding="utf-8"
    )
    job_runner.update_job(run_id, probe=dict(QUALIFIED_PROBE))


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
    _qualify_for_ball_tier(run_id, run_dir)

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
    _qualify_for_ball_tier(run_id, run_dir)

    monkeypatch.setenv("PERSON_DETECTOR", "none")
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    players_v1 = job["players_v1"]
    assert players_v1["detector_backend"] == "none"
    assert players_v1["attribution_backend"] == "assumed"
    assert not (run_dir / "players").exists()


# --- capability gating ------------------------------------------------------
# A clip that cannot support ball tracking must not be run through the ball
# stages anyway and reported as an empty success. These fix the skip.


def _unqualified_probe():
    """What media_probe returns for 30 fps 64x48 camera-roll-ish footage."""
    return {
        "fps": 30.0, "width": 64, "height": 48, "frame_count": 12,
        "duration_s": 0.4, "sharpness": 5.0, "has_audio": False,
    }


def test_ball_stages_are_skipped_when_the_tier_is_off(tmp_path, monkeypatch):
    """The model must never be loaded for footage that cannot support it.

    get_tracking_model raises here: loading it is minutes and gigabytes, and
    the whole point of the gate is not to spend them on a clip whose ball is
    smaller than the detector was ever shown.
    """
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-ball-off"
    run_dir = _make_job(tmp_path, run_id, video_path)
    job_runner.update_job(run_id, probe=_unqualified_probe())

    def explode():
        raise AssertionError("ball model loaded for an unqualified clip")

    monkeypatch.setattr(job_runner, "get_tracking_model", explode)

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    assert job["hits"] == []
    assert job["capabilities"]["ball_tracking"]["enabled"] is False
    assert job["capabilities"]["rally_structure"]["enabled"] is True


def test_a_skipped_ball_tier_still_leaves_a_readable_empty_csv(tmp_path, monkeypatch):
    """Downstream readers open ball_coordinates.csv unconditionally.

    A missing file and a headers-only file are very different to them: the
    first is an error path nobody wrote, the second is "we looked and there
    was nothing", which is what actually happened.
    """
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-ball-off-csv"
    run_dir = _make_job(tmp_path, run_id, video_path)
    job_runner.update_job(run_id, probe=_unqualified_probe())
    monkeypatch.setattr(
        job_runner, "get_tracking_model",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    job_runner.run_tracking_job(run_id)

    csv_path = run_dir / "ball_coordinates.csv"
    assert csv_path.exists()
    lines = [line for line in csv_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1, "expected headers and no rows"


def test_the_reason_the_ball_tier_is_off_reaches_the_job(tmp_path, monkeypatch):
    """Principle 3: disabled with a *stated* reason, never silently."""
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-ball-off-reason"
    _make_job(tmp_path, run_id, video_path)
    job_runner.update_job(run_id, probe=_unqualified_probe())
    monkeypatch.setattr(
        job_runner, "get_tracking_model",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    job_runner.run_tracking_job(run_id)

    reason = job_runner.get_job(run_id)["capabilities"]["ball_tracking"]["reason"]
    assert "50 fps" in reason


def test_a_qualified_clip_still_runs_the_ball_stages(tmp_path, monkeypatch):
    """The gate must not become a blanket off switch.

    Without this, "skip when unqualified" and "skip always" pass the same
    tests, and the ball tier quietly stops running on the footage it was
    built for.
    """
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-ball-on"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)

    loaded = []
    monkeypatch.setattr(job_runner, "get_tracking_model",
                        lambda: loaded.append(1) or object())
    monkeypatch.setattr(job_runner, "infer_frame_predictions",
                        lambda model, frame, threshold, width: [])
    monkeypatch.setattr(job_runner, "extract_audio_candidates",
                        lambda video_path, start_frame, end_frame, fps: [])

    job_runner.run_tracking_job(run_id)

    assert loaded, "ball model was not loaded for qualified footage"
    assert job_runner.get_job(run_id)["capabilities"]["ball_tracking"]["enabled"] is True


def test_a_run_with_no_probe_records_capabilities_anyway(tmp_path, monkeypatch):
    """Legacy runs predate probing and must still say what they could do."""
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-no-probe"
    _make_job(tmp_path, run_id, video_path)

    monkeypatch.setattr(
        job_runner, "get_tracking_model",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    assert job["capabilities"]["ball_tracking"]["enabled"] is False


def test_a_run_reports_what_fraction_of_frames_had_a_ball(tmp_path, monkeypatch):
    """Coverage is what separates "found nothing" from "couldn't look".

    A rally statistic computed from 35% of the frames looks exactly like one
    computed from all of them. Recording the fraction is what lets a reader
    tell the difference, and it is the number the standing recall problem is
    measured in.
    """
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-coverage"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)

    # Every other inferred frame carries a ball.
    calls = []

    def every_other(model, frame, threshold, width):
        calls.append(1)
        if len(calls) % 2:
            return [{"x": 50.0, "y": 50.0, "width": 6.0, "height": 6.0,
                     "confidence": 0.9, "class": "ball"}]
        return []

    monkeypatch.setattr(job_runner, "get_tracking_model", lambda: object())
    monkeypatch.setattr(job_runner, "infer_frame_predictions", every_other)
    monkeypatch.setattr(job_runner, "extract_audio_candidates",
                        lambda video_path, start_frame, end_frame, fps: [])

    job_runner.run_tracking_job(run_id)

    coverage = job_runner.get_job(run_id)["detection_coverage"]
    assert 0.0 < coverage <= 1.0


def test_a_run_that_found_no_ball_reports_zero_coverage_not_a_missing_key(
    tmp_path, monkeypatch
):
    """Zero is a measurement. Absence is not, and reads as "not applicable"."""
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-coverage-zero"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    assert job_runner.get_job(run_id)["detection_coverage"] == 0.0


# --- fps-normalized windows -------------------------------------------------


def test_wired_hit_windows_match_the_pre_refactor_values_at_60fps():
    """The refactor's safety claim, asserted where it is wired rather than
    only where it is defined.

    Before fps-normalization the pipeline passed max_gap=max(3, stride) and
    left min_gap/smooth at their defaults. At the 60 fps reference the scaled
    kwargs must reproduce exactly that, or the whole eval corpus moves.
    """
    from detect_wall_hits import (
        MAX_GAP_FRAMES, MIN_GAP_FRAMES, SMOOTH_WINDOW, scaled_hit_kwargs,
    )

    scaled = scaled_hit_kwargs(60.0)
    frame_stride = 4

    assert max(scaled["max_gap"], frame_stride) == max(MAX_GAP_FRAMES, frame_stride)
    assert scaled["min_gap"] == MIN_GAP_FRAMES
    assert scaled["smooth"] == SMOOTH_WINDOW


def test_audio_pads_are_unchanged_at_the_reference_frame_rate():
    audio = [{"window_start_frame": 100, "window_end_frame": 110}]

    at_reference = job_runner.refine_segments_for_audio_candidates(
        audio, 0, 1000, 60.0
    )
    unscaled = job_runner.refine_segments_for_audio_candidates(audio, 0, 1000)

    assert at_reference == unscaled


def test_audio_pads_shrink_on_slower_footage():
    """A pad counted in frames covers twice the wall-clock at half the rate."""
    audio = [{"window_start_frame": 100, "window_end_frame": 110}]

    slow = job_runner.refine_segments_for_audio_candidates(audio, 0, 1000, 30.0)
    fast = job_runner.refine_segments_for_audio_candidates(audio, 0, 1000, 60.0)

    assert slow[0][0] > fast[0][0]
    assert slow[0][1] < fast[0][1]
