"""player_tracker: two-track assignment, coasting, ambiguity accounting."""

import numpy as np

from person_model import PersonDetection
from player_tracker import (
    AMBIGUITY_MARGIN,
    COAST_MAX_S,
    PersonFramePass,
    TwoPlayerTracker,
)


def det(x, y, confidence=0.9):
    return PersonDetection(x=x, y=y, width=40.0, height=100.0,
                           confidence=confidence, keypoints=())


def test_separated_walkers_keep_identity():
    tracker = TwoPlayerTracker()
    for step in range(50):
        t = step * 0.25
        tracker.update(t, step, [det(100 + step * 2, 300),
                                 det(900 - step * 2, 300)])
    samples = tracker.samples()
    assert len(samples["A"]) == 50 and len(samples["B"]) == 50
    a_x = [s.foot_px[0] for s in samples["A"]]
    b_x = [s.foot_px[0] for s in samples["B"]]
    assert all(later > earlier for earlier, later in zip(a_x, a_x[1:]))
    assert all(later < earlier for earlier, later in zip(b_x, b_x[1:]))
    assert tracker.ambiguity_times() == []


def test_dropout_coasts_then_stops_then_reacquires():
    tracker = TwoPlayerTracker()
    tracker.update(0.0, 0, [det(100, 300), det(900, 300)])
    # B drops out for 2 s at 4 Hz: first COAST_MAX_S emits coasted samples.
    steps = 8
    for step in range(1, steps + 1):
        tracker.update(step * 0.25, step, [det(100 + step, 300)])
    samples = tracker.samples()
    coasted = [s for s in samples["B"] if s.coasted]
    assert coasted, "B should coast after dropout"
    assert all(s.foot_px == samples["B"][0].foot_px for s in coasted)
    assert max(s.t_s for s in samples["B"]) <= COAST_MAX_S + 1e-9
    # Reacquire near the old position.
    tracker.update((steps + 1) * 0.25, steps + 1, [det(101 + steps, 300), det(905, 300)])
    live_b = [s for s in tracker.samples()["B"] if not s.coasted]
    assert live_b[-1].foot_px[0] == 905.0


def test_crossing_records_ambiguity():
    tracker = TwoPlayerTracker()
    # Both players converge on the same point, then swap-like geometry.
    tracker.update(0.0, 0, [det(400, 300), det(600, 300)])
    tracker.update(0.25, 1, [det(499, 300), det(501, 300)])
    assert len(tracker.ambiguity_times()) >= 1
    stats = tracker.stats()
    assert stats["ambiguous_assignments"] >= 1
    assert stats["updates"] == 2


def test_third_detection_ignored_top2_by_confidence():
    tracker = TwoPlayerTracker()
    tracker.update(0.0, 0, [det(100, 300, 0.95), det(900, 300, 0.9),
                            det(500, 100, 0.3)])
    samples = tracker.samples()
    xs = {samples["A"][0].foot_px[0], samples["B"][0].foot_px[0]}
    assert xs == {100.0, 900.0}


def test_person_frame_pass_cadence_and_wiring():
    calls = []

    class StubDetector:
        backend = "stub"

        def detect(self, frame_bgr):
            calls.append(1)
            return [det(100, 300), det(900, 300)]

    # 60 fps at stride 4 -> 15 Hz coarse cadence -> detect every 4th frame
    # for PERSON_DETECT_HZ = 4.0 (round(15/4) = 4).
    person_pass = PersonFramePass(StubDetector(), source_fps=60.0, frame_stride=4)
    assert person_pass.detect_every == 4
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(8):
        person_pass.observe(i * 4, frame)
    assert len(calls) == 2  # observed frames 0..7 -> detected on 0 and 4
    assert len(person_pass.tracker.samples()["A"]) == 2


def test_track_segments_frame_observer_sees_coarse_frames(tmp_path, monkeypatch):
    import cv2
    import job_runner

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(job_runner, "infer_frame_predictions",
                        lambda model, frame, threshold, width: [])
    observed = []
    job_runner.track_segments(
        model=None, video_path=video, segments=[(0, 11, 3)],
        inference_width=64, source_fps=30.0, results={},
        on_frame=lambda idx: None,
        frame_observer=lambda idx, frame: observed.append((idx, frame.shape)),
    )
    assert [idx for idx, _ in observed] == [0, 3, 6, 9]
    assert all(shape == (48, 64, 3) for _, shape in observed)
