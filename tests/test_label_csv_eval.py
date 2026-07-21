"""label_hits.py CSVs -> eval set -> missed-bounce axis.

Offline labeling used to feed only the trainers; these tests pin the path
that also makes it evaluation data, and in particular that a label whose
video was never tracked is excluded rather than scored as a detector miss.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_eval_set import build_eval_set, load_label_frames, write_eval_set
from eval_line_calls import evaluate_cases

LABELED_SHA = "c" * 64
UNTRACKED_SHA = "d" * 64


def write_labels(labels_dir, name, frames, video_sha, fps=60.0):
    labels_dir.mkdir(parents=True, exist_ok=True)
    csv_path = labels_dir / f"{name}.csv"
    csv_path.write_text(
        "hit_frame\n" + "".join(f"{frame}\n" for frame in frames), encoding="utf-8"
    )
    csv_path.with_suffix(".meta.json").write_text(
        json.dumps({
            "schema_version": "label-run-v1",
            "video_path": f"/anywhere/{name}.mp4",
            "video_sha": video_sha,
            "fps": fps,
            "frame_count": 1000,
            "label_count": len(frames),
        }),
        encoding="utf-8",
    )
    return csv_path


def write_tracked_run(runs_dir, run_id, video_sha, hit_frames, frame_stride=1):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    run_dir.joinpath("job.json").write_text(json.dumps({
        "video_path": f"/anywhere/by-hash/{video_sha}.mp4",
        "fps": 60.0, "frame_stride": frame_stride, "inference_width": 960,
    }))
    run_dir.joinpath("detected_hits.json").write_text(json.dumps({
        "hits": [{"frame": frame, "event_type": "wall"} for frame in hit_frames]
    }))
    return run_dir


def test_label_frames_parse_from_csv(tmp_path):
    csv_path = write_labels(tmp_path, "hits", [59, 272, 59], LABELED_SHA)

    assert load_label_frames(csv_path) == [59, 272]


def test_labels_join_the_missed_bounce_axis(tmp_path):
    runs, labels = tmp_path / "runs", tmp_path / "labels"
    # Detector found 100 but missed 200 and 300.
    write_tracked_run(runs, "run-a", LABELED_SHA, [100])
    write_labels(labels, "bayclub_wall_hits", [100, 200, 300], LABELED_SHA)

    cases, manifest = build_eval_set(runs, labels)
    report = evaluate_cases(cases)

    assert manifest["label_csv_cases"] == 3
    assert manifest["label_scorable_cases"] == 3
    assert manifest["label_missed_at_build"] == 2
    assert report["missed"]["checked"] == 3
    assert len(report["missed"]["missed"]) == 2
    assert report["missed"]["missed_by_type"] == {"wall": 2}


def test_untracked_video_is_excluded_not_counted_as_missed(tmp_path):
    runs, labels = tmp_path / "runs", tmp_path / "labels"
    runs.mkdir()
    # No run at all for this video: the detector never had a chance.
    write_labels(labels, "untracked_wall_hits", [10, 20, 30], UNTRACKED_SHA)

    cases, manifest = build_eval_set(runs, labels)
    report = evaluate_cases(cases)

    assert manifest["label_csv_cases"] == 3
    assert manifest["label_csv_cases_without_detector_run"] == 3
    assert manifest["label_scorable_cases"] == 0
    assert manifest["label_missed_at_build"] == 0
    # The whole point: these must not inflate the miss count.
    assert report["missed"]["checked"] == 0
    assert report["missed"]["missed"] == []
    assert report["label_cases_unscorable"] == 3


def test_frame_stride_widens_the_match_tolerance(tmp_path):
    runs, labels = tmp_path / "runs", tmp_path / "labels"
    # Detection landed 3 frames off; a stride-4 run should still call it found.
    write_tracked_run(runs, "run-a", LABELED_SHA, [103], frame_stride=4)
    write_labels(labels, "hits", [100], LABELED_SHA)

    cases, manifest = build_eval_set(runs, labels)

    assert manifest["label_missed_at_build"] == 0
    assert cases[0]["matched_detected"]["distance_frames"] == 3


def test_duplicate_frames_across_sessions_collapse(tmp_path):
    runs, labels = tmp_path / "runs", tmp_path / "labels"
    write_tracked_run(runs, "run-a", LABELED_SHA, [])
    write_labels(labels, "session_a", [10, 20], LABELED_SHA)
    write_labels(labels, "session_b", [10, 20, 30], LABELED_SHA)

    cases, manifest = build_eval_set(runs, labels)

    # Same video, same frames: the fuller session wins, no double counting.
    assert manifest["label_csv_cases"] == 3
    assert manifest["duplicates_dropped"] == 2
    assert {c["label_source"] for c in cases} == {"session_b.csv"}


def test_labels_are_opt_out(tmp_path):
    runs, labels = tmp_path / "runs", tmp_path / "labels"
    runs.mkdir()
    write_labels(labels, "hits", [10], LABELED_SHA)

    cases, manifest = build_eval_set(runs, labels_dir=None)

    assert cases == []
    assert manifest["label_csv_cases"] == 0


def test_merge_preserves_cases_this_machine_cannot_rebuild(tmp_path):
    """The regression that wiped all 19 committed cases: eval_set/ is tracked
    in git but built from gitignored ui_runs/, so a checkout without those
    runs must not delete their cases."""
    runs, labels, out = tmp_path / "runs", tmp_path / "labels", tmp_path / "out"
    runs.mkdir()
    # A committed eval set built on someone else's machine.
    write_eval_set(
        [{"case_id": "corr:their-run:42", "kind": "correction",
          "run_id": "their-run", "frame": 42, "video_sha": None,
          "human": {"type": "wall", "call": "OUT"}}],
        {"case_count": 1}, out,
    )
    write_labels(labels, "mine", [10], LABELED_SHA)

    cases, manifest = build_eval_set(runs, labels, merge_dir=out)

    assert manifest["preserved_cases"] == 1
    assert {c["case_id"] for c in cases} == {"corr:their-run:42", "lbl:mine.csv:10"}
    assert manifest["correction_cases"] == 1


def test_rebuilt_cases_win_over_committed_copies(tmp_path):
    runs, labels, out = tmp_path / "runs", tmp_path / "labels", tmp_path / "out"
    write_tracked_run(runs, "run-a", LABELED_SHA, [10])
    write_labels(labels, "mine", [10], LABELED_SHA)
    # A stale copy of the same case_id, claiming it was never detected.
    write_eval_set(
        [{"case_id": "lbl:mine.csv:10", "kind": "label_csv_event",
          "run_id": None, "frame": 10, "matched_detected": None,
          "no_detector_run": True, "video_sha": LABELED_SHA,
          "label_source": "mine.csv", "human_type": "wall"}],
        {"case_count": 1}, out,
    )

    cases, manifest = build_eval_set(runs, labels, merge_dir=out)

    assert manifest["preserved_cases"] == 0
    assert len(cases) == 1
    assert cases[0]["matched_detected"] is not None


def test_replace_mode_drops_unrebuildable_cases(tmp_path):
    runs, labels, out = tmp_path / "runs", tmp_path / "labels", tmp_path / "out"
    runs.mkdir()
    write_eval_set(
        [{"case_id": "corr:their-run:42", "kind": "correction",
          "run_id": "their-run", "frame": 42, "video_sha": None,
          "human": {"type": "wall", "call": "OUT"}}],
        {"case_count": 1}, out,
    )
    write_labels(labels, "mine", [10], LABELED_SHA)

    cases, manifest = build_eval_set(runs, labels, merge_dir=None)

    assert manifest["preserved_cases"] == 0
    assert {c["case_id"] for c in cases} == {"lbl:mine.csv:10"}


def test_csv_without_sidecar_is_ignored(tmp_path):
    runs, labels = tmp_path / "runs", tmp_path / "labels"
    runs.mkdir()
    labels.mkdir()
    # A bare CSV has no video identity, so it cannot become a case.
    labels.joinpath("orphan.csv").write_text("hit_frame\n5\n", encoding="utf-8")

    cases, manifest = build_eval_set(runs, labels)

    assert manifest["label_csv_cases"] == 0
    assert cases == []
