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
import pytest

import job_runner
import person_model
from person_model import PersonDetection
from test_court_model import make_v2_calibration

QUALIFIED_PROBE = {
    "fps": 60.0, "width": 1920, "height": 1080, "frame_count": 12,
    "duration_s": 0.2, "sharpness": 120.0, "has_audio": True,
}


def _qualify_for_ball_tier(run_id, run_dir):
    """Give a run footage and court good enough for every analysis tier.

    The ball tier is gated on frame rate, size, and sharpness. The solved
    floor court additionally enables player movement, which the
    person-detector tests below exercise.
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


@pytest.fixture(autouse=True)
def _default_to_rfdetr_backend(monkeypatch):
    """Pin the historical rfdetr backend for this module.

    These integration tests predate the local WASB default and stub the
    rfdetr seams (get_tracking_model / infer_frame_predictions) -- on the
    local default they would load the real committed artifact instead.
    Tests about the local backend override this explicitly (delenv restores
    the real default)."""
    monkeypatch.setenv("BALL_DETECTOR", "rfdetr")


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
    saved_crops = []
    monkeypatch.setattr(
        person_model,
        "save_person_crop",
        lambda video, frame, detection, path: saved_crops.append(path.name) or True,
    )
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    players_v1 = job["players_v1"]
    assert players_v1["detector_backend"] == "stub"
    assert players_v1["attribution_backend"] in ("observed", "assumed")
    assert players_v1["player_crops"] == {
        "A": "players/player_A.jpg",
        "B": "players/player_B.jpg",
    }
    assert saved_crops == ["player_A.jpg", "player_B.jpg"]
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
    assert "1600" in reason


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


def test_a_30_fps_clip_still_runs_the_ball_stages(tmp_path, monkeypatch):
    """Lower frame rate reduces temporal precision; it must not skip tracking."""
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path, fps=30.0)
    run_id = "test-job-runner-ball-on-30fps"
    run_dir = _make_job(tmp_path, run_id, video_path)
    (run_dir / "calibration.json").write_text(
        json.dumps(make_v2_calibration()), encoding="utf-8"
    )
    job_runner.update_job(run_id, probe=dict(QUALIFIED_PROBE, fps=30.0))

    loaded = []
    monkeypatch.setattr(
        job_runner, "get_tracking_model", lambda: loaded.append(1) or object()
    )
    monkeypatch.setattr(
        job_runner,
        "infer_frame_predictions",
        lambda model, frame, threshold, width: [],
    )
    monkeypatch.setattr(
        job_runner, "extract_audio_candidates", lambda *args, **kwargs: []
    )

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert loaded, "ball model was skipped solely because the clip was 30 fps"
    assert job["processed_frames"] > 0
    assert job["capabilities"]["ball_tracking"]["enabled"] is True
    assert job["capabilities"]["line_calls"]["enabled"] is True


def test_front_wall_calibration_without_floor_still_runs_ball_tracking(
    tmp_path, monkeypatch
):
    """Skipping optional floor calibration must not skip the core tracker."""
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-front-wall-only"
    run_dir = _make_job(tmp_path, run_id, video_path)

    calibration = make_v2_calibration()
    calibration["planes"].pop("floor")
    (run_dir / "calibration.json").write_text(
        json.dumps(calibration), encoding="utf-8"
    )
    job_runner.update_job(run_id, probe=dict(QUALIFIED_PROBE))

    loaded = []
    monkeypatch.setattr(
        job_runner, "get_tracking_model", lambda: loaded.append(1) or object()
    )
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

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert loaded, "ball model was skipped because floor calibration was absent"
    assert job["status"] == "complete"
    assert job["processed_frames"] > 0
    assert job["capabilities"]["player_movement"]["enabled"] is False
    assert job["capabilities"]["ball_tracking"]["enabled"] is True
    assert job["capabilities"]["line_calls"]["enabled"] is True


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


# --- rally timeline ---------------------------------------------------------
# Tier 1 must be emitted whether or not the ball tier ran. These are the tests
# that stop rally structure from quietly becoming ball-dependent again.


def test_a_qualified_run_emits_a_rally_timeline(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-timeline-on"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    timeline = job_runner.get_job(run_id)["rally_timeline"]
    assert "rallies" in timeline
    assert "audio_available" in timeline
    assert "gap_s" in timeline


def test_a_ball_tier_skipped_run_still_emits_a_rally_timeline(tmp_path, monkeypatch):
    """The whole point of the ladder.

    Footage that is too small and blurry for ball tracking still needs rally
    structure. That tier needs neither the ball nor a court, so it must survive
    the skip -- otherwise tier 1 is ball-dependent by omission rather than by
    design.
    """
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-timeline-off"
    _make_job(tmp_path, run_id, video_path)
    job_runner.update_job(run_id, probe=_unqualified_probe())
    monkeypatch.setattr(
        job_runner, "get_tracking_model",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    monkeypatch.setattr(job_runner, "extract_audio_candidates",
                        lambda *args, **kwargs: [])

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    assert job["hits"] == []
    assert "rally_timeline" in job, "tier 1 vanished with the ball tier"


def test_unreadable_audio_is_reported_not_silently_treated_as_quiet(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-timeline-no-audio"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)
    monkeypatch.setattr(job_runner, "get_tracking_model", lambda: object())
    monkeypatch.setattr(job_runner, "infer_frame_predictions",
                        lambda model, frame, threshold, width: [])
    monkeypatch.setattr(job_runner, "extract_audio_candidates",
                        lambda *args, **kwargs: None)

    job_runner.run_tracking_job(run_id)

    timeline = job_runner.get_job(run_id)["rally_timeline"]
    assert timeline["audio_available"] is False


def test_the_timeline_is_written_to_the_run_directory(tmp_path, monkeypatch):
    """Reports are assembled from the run dir, not from the in-memory job."""
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-timeline-file"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    written = json.loads((run_dir / "rally_timeline.json").read_text())
    assert written["schema"] == "rally-timeline-v1"
    assert "rallies" in written


# --- movement tier ----------------------------------------------------------


class _StubPersonDetector:
    backend = "stub"

    def detect(self, frame_bgr):
        return [_det(400, 700), _det(1200, 500)]


def test_movement_stats_are_emitted_when_the_tier_is_on(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-movement-on"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)
    monkeypatch.setattr(person_model, "load_person_detector",
                        lambda: _StubPersonDetector())
    _stub_pipeline(monkeypatch)

    job_runner.run_tracking_job(run_id)

    players_v2 = job_runner.get_job(run_id)["players_v2"]
    assert players_v2["backend"] == "stub"
    assert "player_a" in players_v2 and "player_b" in players_v2
    assert "distance_ft" in players_v2["player_a"]


def test_the_movement_tier_survives_the_ball_tier_being_off(tmp_path, monkeypatch):
    """The coupling this task exists to break.

    Capability gating skipped the ball stages, and the person detector rode on
    the ball decode via frame_observer -- so a clip that could not support ball
    tracking silently lost player movement too, which the analysis ladder says
    is a tier that needs only a solved court.
    """
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-movement-ball-off"
    run_dir = _make_job(tmp_path, run_id, video_path)
    # A solved court, but footage too slow and too small for the ball tier.
    (run_dir / "calibration.json").write_text(
        json.dumps(make_v2_calibration()), encoding="utf-8"
    )
    job_runner.update_job(run_id, probe=_unqualified_probe())
    monkeypatch.setattr(person_model, "load_person_detector",
                        lambda: _StubPersonDetector())
    monkeypatch.setattr(
        job_runner, "get_tracking_model",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    monkeypatch.setattr(job_runner, "extract_audio_candidates",
                        lambda *args, **kwargs: [])

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    assert job["capabilities"]["ball_tracking"]["enabled"] is False
    assert job["capabilities"]["player_movement"]["enabled"] is True
    assert "players_v2" in job, "tier 2 died with tier 3 again"
    assert job["players_v2"]["backend"] == "stub"


def test_no_court_means_no_movement_stats_with_a_reason(tmp_path, monkeypatch):
    """Feet need a homography. Without one there is nothing honest to report."""
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-movement-no-court"
    _make_job(tmp_path, run_id, video_path)
    job_runner.update_job(run_id, probe=dict(QUALIFIED_PROBE))
    monkeypatch.setattr(person_model, "load_person_detector",
                        lambda: _StubPersonDetector())
    monkeypatch.setattr(
        job_runner, "get_tracking_model",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    monkeypatch.setattr(job_runner, "extract_audio_candidates",
                        lambda *args, **kwargs: [])

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    movement = job["capabilities"]["player_movement"]
    assert movement["enabled"] is False
    assert "court" in movement["reason"]
    assert job.get("players_v2") is None


def _drain_decode(video_path, segments, temporal):
    import queue as queue_module
    import threading

    frame_queue = queue_module.Queue()
    stop_event = threading.Event()
    errors = []
    job_runner.decode_segments_to_queue(
        video_path, segments, frame_queue, stop_event, errors,
        temporal=temporal,
    )
    assert errors == []
    items = []
    while True:
        item = frame_queue.get_nowait()
        if item is None:
            return items
        items.append(item)


def _frame_value(frame):
    """Recover the frame's identity from its solid pixel value.

    _write_clip paints frame i as solid i*20, but mp4 encoding is lossy
    (value 20 can decode as 17), so snap to the nearest multiple of 20.
    """
    return round(int(frame[0, 0, 0]) / 20) * 20


def test_temporal_decode_stride1_sliding_windows_and_edge_padding(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=5)
    items = _drain_decode(video, [(0, 4, 1)], temporal=True)
    assert [idx for idx, _ in items] == [0, 1, 2, 3, 4]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[0] == [0, 0, 20]        # left edge pads prev with cur
    assert values[2] == [20, 40, 60]      # interior: true neighbours
    assert values[4] == [60, 80, 80]      # right edge pads nxt with cur


def test_temporal_decode_stride4_centers_get_true_neighbours(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=10)
    items = _drain_decode(video, [(0, 9, 4)], temporal=True)
    assert [idx for idx, _ in items] == [0, 4, 8]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[0] == [0, 0, 20]          # first center: padded prev, true nxt
    assert values[4] == [60, 80, 100]       # strided center: TRUE t-1/t+1, not t-4/t+4
    assert values[8] == [140, 160, 180]


def test_temporal_decode_stride2_shared_neighbours(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=5)
    items = _drain_decode(video, [(0, 4, 2)], temporal=True)
    assert [idx for idx, _ in items] == [0, 2, 4]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[2] == [20, 40, 60]
    assert values[4] == [60, 80, 80]


def test_temporal_decode_resets_across_segments(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=12)
    items = _drain_decode(video, [(0, 3, 1), (8, 11, 1)], temporal=True)
    assert [idx for idx, _ in items] == [0, 1, 2, 3, 8, 9, 10, 11]
    values = {idx: [_frame_value(f) for f in frames] for idx, frames in items}
    assert values[3] == [40, 60, 60]        # segment end pads, never crosses
    assert values[8] == [160, 160, 180]     # new segment starts padded


def test_non_temporal_decode_payload_unchanged(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=6)
    items = _drain_decode(video, [(0, 5, 2)], temporal=False)
    assert [idx for idx, _ in items] == [0, 2, 4]
    assert all(isinstance(frame, np.ndarray) for _, frame in items)


class _StubBallRunner:
    """A ball_model-runner stand-in: manifest + recording run_batch."""

    class _Manifest:
        frames_per_input = 3
        conf_threshold = 0.1
        input_size = (416, 416)
        tile_overlap_px = 64
        max_batch_tiles = 32
        nms_iou = 0.45
        class_names = ("ball",)
        name = "stub-wasb"
        version = 1
        artifact_sha256 = "deadbeef"

    def __init__(self):
        self.manifest = self._Manifest()
        self.batches = []
        self.device = "cpu"

    def run_batch(self, stacks):
        self.batches.append([s.shape for s in stacks])
        return [[] for _ in stacks]


def test_track_segments_local_temporal_feeds_stacks_and_observes_centers(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=8)
    runner = _StubBallRunner()
    results = {}
    observed = []
    job_runner.track_segments(
        runner, video, [(0, 7, 4)], 640, 30.0, results,
        on_frame=lambda idx: None,
        frame_observer=lambda idx, frame: observed.append(
            (idx, _frame_value(frame))),
        backend="local",
    )
    # Observer fired once per CENTER frame with the center frame itself.
    assert observed == [(0, 0), (4, 80)]
    # Every stack reaching the runner is one 9-channel tile (64x48 clip
    # is smaller than one 416 tile, zero-padded).
    for batch in runner.batches:
        for shape in batch:
            assert shape == (416, 416, 9)
    # No detections -> empty rows for the two centers.
    assert sorted(results) == [0, 4]
    assert all(results[idx]["detected"] is False for idx in results)


def test_track_segments_local_uses_manifest_confidence_floor(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=4)
    runner = _StubBallRunner()
    floors = []

    def spy_select(predictions_by_frame, confidence_threshold, **kwargs):
        floors.append(confidence_threshold)
        return {frame: None for frame in predictions_by_frame}

    monkeypatch.setattr(
        job_runner, "select_motion_consistent_ball_predictions", spy_select)
    job_runner.track_segments(
        runner, video, [(0, 3, 1)], 640, 30.0, {},
        on_frame=lambda idx: None, backend="local")
    assert floors == [pytest.approx(0.1)]

    monkeypatch.setattr(job_runner, "infer_frame_predictions",
                        lambda model, frame, threshold, width: [])
    job_runner.track_segments(
        object(), video, [(0, 3, 1)], 640, 30.0, {},
        on_frame=lambda idx: None, backend="rfdetr")
    assert floors[-1] == pytest.approx(0.40)


def test_track_segments_local_single_frame_manifest_uses_detect_frame(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    _write_clip(video, frame_count=4)
    runner = _StubBallRunner()
    runner.manifest.frames_per_input = 1
    calls = []
    monkeypatch.setattr(
        job_runner, "detect_frame",
        lambda model, frame, manifest: calls.append(frame.shape) or [])
    job_runner.track_segments(
        runner, video, [(0, 3, 2)], 640, 30.0, {},
        on_frame=lambda idx: None, backend="local")
    assert len(calls) == 2          # frames 0 and 2 (stride 2), full frames


def test_track_segments_rejects_unsupported_frames_per_input(tmp_path):
    runner = _StubBallRunner()
    runner.manifest.frames_per_input = 5
    with pytest.raises(ValueError, match="frames_per_input"):
        job_runner.track_segments(
            runner, "unused.mp4", [(0, 3, 1)], 640, 30.0, {},
            on_frame=lambda idx: None, backend="local")


def test_track_segments_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        job_runner.track_segments(
            object(), "unused.mp4", [(0, 3, 1)], 640, 30.0, {},
            on_frame=lambda idx: None, backend="coreml")


def test_load_ball_backend_local_never_touches_roboflow(monkeypatch):
    monkeypatch.delenv("BALL_DETECTOR", raising=False)   # default is local
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    stub = _StubBallRunner()
    monkeypatch.setattr(job_runner.ball_model, "load_detector", lambda: stub)

    def explode():
        raise AssertionError("rfdetr path must not load on the local backend")

    monkeypatch.setattr(job_runner, "get_tracking_model", explode)
    backend, model = job_runner.load_ball_backend()
    assert backend == "local"
    assert model is stub


def test_load_ball_backend_rfdetr_branch(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "rfdetr")
    sentinel = object()
    monkeypatch.setattr(job_runner, "get_tracking_model", lambda: sentinel)
    backend, model = job_runner.load_ball_backend()
    assert backend == "rfdetr"
    assert model is sentinel


def test_load_ball_backend_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("BALL_DETECTOR", "coreml")
    with pytest.raises(ValueError, match="coreml"):
        job_runner.load_ball_backend()


def test_ball_backend_summary_shapes():
    local = job_runner.ball_backend_summary("local", _StubBallRunner())
    assert local == {
        "backend": "local", "name": "stub-wasb", "version": 1,
        "artifact_sha256": "deadbeef", "device": "cpu",
    }
    hosted = job_runner.ball_backend_summary("rfdetr", object())
    assert hosted["backend"] == "rfdetr"
    assert hosted["model_id"]          # squashai/1 or ROBOFLOW_MODEL_ID


def test_run_tracking_job_local_backend_end_to_end(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_clip(video_path)
    run_id = "test-job-runner-local-ball"
    run_dir = _make_job(tmp_path, run_id, video_path)
    _qualify_for_ball_tier(run_id, run_dir)

    monkeypatch.delenv("BALL_DETECTOR", raising=False)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setenv("PERSON_DETECTOR", "none")
    stub = _StubBallRunner()
    monkeypatch.setattr(job_runner.ball_model, "load_detector", lambda: stub)
    monkeypatch.setattr(
        job_runner, "extract_audio_candidates",
        lambda video_path, start_frame, end_frame, fps: [])

    job_runner.run_tracking_job(run_id)

    job = job_runner.get_job(run_id)
    assert job["status"] == "complete"
    assert job["ball_backend"]["backend"] == "local"
    assert job["ball_backend"]["name"] == "stub-wasb"
    assert stub.batches            # the stub actually ran
