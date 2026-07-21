"""build_eval_set + eval_line_calls: labels -> eval set -> multi-axis replay."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_eval_set import build_eval_set, video_sha_from_path, write_eval_set
from eval_line_calls import evaluate_cases, load_eval_set

VIDEO_SHA = "a" * 64
GT_VIDEO_SHA = "b" * 64

CALIBRATION = {
    "frame_width": 640,
    "frame_height": 360,
    "lines": [
        {"name": "out_line_lower_edge", "endpoints": [[50, 100], [590, 100]]},
        {"name": "tin_top_edge", "endpoints": [[50, 300], [590, 300]]},
    ],
}


def corr(frame, ctype, call=None, ball=None, frame_is_bounce=True,
         corrected_frame=None, predicted=None, recorded_at=None):
    return {
        "frame": frame,
        "corrected": {
            "type": ctype, "call": call, "ball": ball,
            "frame_is_bounce": None if ctype == "none" else frame_is_bounce,
            "frame": corrected_frame,
        },
        "predicted": predicted or {},
        "recorded_at": recorded_at,
    }


def write_run(runs_dir, run_id, corrections=None, job=None, calibration=None,
              ground_truth=None, detected_hits=None):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    if corrections is not None:
        run_dir.joinpath("corrections.json").write_text(
            json.dumps({"schema_version": "corrections-v2",
                        "corrections": corrections})
        )
    if job is not None:
        run_dir.joinpath("job.json").write_text(json.dumps(job))
    if calibration is not None:
        run_dir.joinpath("calibration.json").write_text(json.dumps(calibration))
    if ground_truth is not None:
        run_dir.joinpath("ground_truth.json").write_text(json.dumps(ground_truth))
    if detected_hits is not None:
        run_dir.joinpath("detected_hits.json").write_text(json.dumps(detected_hits))
    return run_dir


def make_runs_tree(tmp_path):
    runs = tmp_path / "runs"
    job = {
        "video_path": f"/anywhere/by-hash/{VIDEO_SHA}.mp4",
        "fps": 30.0, "frame_stride": 2, "inference_width": 960,
    }
    # Run A: frame 10 will be superseded by run B's fresher label; frame 20
    # is a wall OUT whose geometry replays IN (a judge mismatch) with a
    # timing correction; frame 30 is a detector false positive. The last
    # entry is v1-shaped and must be skipped, not crash the build.
    write_run(runs, "run-a", corrections=[
        corr(10, "wall", call="IN", ball={"x": 300, "y": 200},
             predicted={"type": "wall", "call": "IN",
                        "ball": {"x": 302, "y": 205}},
             recorded_at="2026-07-18T01:00:00Z"),
        corr(20, "wall", call="OUT", ball={"x": 300, "y": 200},
             frame_is_bounce=False, corrected_frame=22,
             predicted={"type": "wall", "call": "IN", "margin_px": 4.0,
                        "source": "detected_center",
                        "ball": {"x": 310, "y": 210}},
             recorded_at="2026-07-18T01:05:00Z"),
        corr(30, "none",
             predicted={"type": "wall", "call": "IN",
                        "ball": {"x": 300, "y": 200}},
             recorded_at="2026-07-18T01:10:00Z"),
        {"frame": 40, "predicted_call": "IN", "corrected_call": "NOT_A_HIT"},
    ], job=job, calibration=CALIBRATION)
    # Run B: same video, same source frame 10, labeled an hour later. The
    # label-time prediction (IN on a ball above the out line) replays OUT,
    # so this case also exercises drift detection.
    write_run(runs, "run-b", corrections=[
        corr(10, "wall", call="OUT", ball={"x": 300, "y": 50},
             predicted={"type": "wall", "call": "IN",
                        "source": "impact_estimate",
                        "ball": {"x": 300, "y": 55}},
             recorded_at="2026-07-18T02:00:00Z"),
    ], job=job, calibration=CALIBRATION)
    # Run C: labels from a run with no job.json and no calibration (the
    # UI-test-fixture shape) - kept, but geometric replay must skip it.
    write_run(runs, "run-c", corrections=[
        corr(5, "wall", call="IN", ball={"x": 10, "y": 10},
             recorded_at="2026-07-18T03:00:00Z"),
    ])
    # Run D: corrupt corrections file - counted, never crashes the build.
    run_d = runs / "run-d"
    run_d.mkdir()
    run_d.joinpath("corrections.json").write_text("{not json")

    # Ground-truth runs: three labeled events. Frame 100 matches a hit two
    # frames away (allowed: tolerance stretches to the stride), frame 200
    # matches a hit of the wrong type, frame 300 was missed entirely.
    gt_job = {
        "video_path": f"/anywhere/by-hash/{GT_VIDEO_SHA}.mp4",
        "fps": 30.0, "frame_stride": 2, "inference_width": 960,
    }
    write_run(runs, "run-gt", job=gt_job,
              ground_truth={"tolerance_frames": 1, "events": [
                  {"frame": 100, "type": "wall"},
                  {"frame": 200, "type": "floor"},
                  {"frame": 300, "type": "racket"},
              ]},
              detected_hits={"hits": [
                  {"frame": 102, "event_type": "wall"},
                  {"frame": 200, "event_type": "wall"},
              ]})
    # A second, sparser labeling session over the same video: its frame-200
    # event loses the dedupe to run-gt's more complete session.
    write_run(runs, "run-gt2", job=gt_job,
              ground_truth={"tolerance_frames": 1, "events": [
                  {"frame": 200, "type": "wall"},
              ]},
              detected_hits={"hits": []})
    return runs


def test_video_sha_from_path():
    assert video_sha_from_path(f"/x/by-hash/{VIDEO_SHA}.mp4") == VIDEO_SHA
    assert video_sha_from_path("/x/uploads/clip.mp4") is None
    assert video_sha_from_path(None) is None


def test_build_dedupes_and_counts(tmp_path):
    cases, manifest = build_eval_set(make_runs_tree(tmp_path))

    assert manifest["case_count"] == len(cases) == 7
    assert manifest["correction_cases"] == 4
    assert manifest["ground_truth_cases"] == 3
    assert manifest["runs_with_corrections"] == 3
    assert manifest["runs_with_ground_truth"] == 2
    assert manifest["unreadable_runs"] == ["run-d"]
    assert manifest["duplicates_dropped"] == 2       # corr frame 10 + gt frame 200
    assert manifest["skipped_legacy_entries"] == 1   # run-a's v1-shaped entry
    assert manifest["cases_without_calibration"] == 1
    assert manifest["cases_without_video_sha"] == 1
    assert manifest["types"] == {"floor": 0, "none": 1, "racket": 0,
                                 "side_wall": 0, "wall": 3}
    assert manifest["calls"] == {"IN": 1, "OUT": 2}
    assert manifest["gt_missed_at_build"] == 1
    assert manifest["latest_correction_at"] == "2026-07-18T03:00:00Z"

    by_id = {c["case_id"]: c for c in cases}
    # Frame 10 kept run B's fresher label, not run A's.
    assert "corr:run-b:10" in by_id and "corr:run-a:10" not in by_id
    assert by_id["corr:run-b:10"]["human"]["call"] == "OUT"
    assert by_id["corr:run-b:10"]["video_sha"] == VIDEO_SHA
    assert by_id["corr:run-b:10"]["fps"] == 30.0
    assert by_id["corr:run-c:5"]["calibration"] is None
    # GT dedupe kept the more complete session's frame-200 event.
    assert "gt:run-gt:200" in by_id and "gt:run-gt2:200" not in by_id
    # Build-time matching: stride-stretched tolerance catches the 2-frame
    # offset; the missed event has no match.
    assert by_id["gt:run-gt:100"]["matched_detected"] == {
        "frame": 102, "event_type": "wall", "distance_frames": 2}
    assert by_id["gt:run-gt:300"]["matched_detected"] is None
    # Sorted by (kind, run_id, frame) for stable diffs.
    assert [c["case_id"] for c in cases] == [
        "corr:run-a:20", "corr:run-a:30", "corr:run-b:10", "corr:run-c:5",
        "gt:run-gt:100", "gt:run-gt:200", "gt:run-gt:300",
    ]


def test_write_is_deterministic_and_loadable(tmp_path):
    runs = make_runs_tree(tmp_path)

    cases, manifest = build_eval_set(runs)
    path_one = write_eval_set(cases, manifest, tmp_path / "out1")
    cases_again, manifest_again = build_eval_set(runs)
    path_two = write_eval_set(cases_again, manifest_again, tmp_path / "out2")

    assert path_one.name == "cases.jsonl"
    assert path_one.read_bytes() == path_two.read_bytes()
    assert (tmp_path / "out1" / "manifest.json").read_bytes() == \
        (tmp_path / "out2" / "manifest.json").read_bytes()
    assert load_eval_set(path_one) == cases


def test_evaluate_in_out_axis(tmp_path):
    cases, _ = build_eval_set(make_runs_tree(tmp_path))
    report = evaluate_cases(cases)

    assert report["total_cases"] == 7
    assert [case_id for case_id, _ in report["unreplayable"]] == ["corr:run-c:5"]

    # corr:run-a:20 - corrected ball between the lines: judge IN, human OUT.
    # corr:run-b:10 - corrected ball above the out line: judge OUT, human agrees.
    assert report["judged"] == 2
    assert report["correct"] == 1
    assert report["accuracy"] == 0.5
    assert report["confusion"] == {("IN", "OUT"): 1, ("OUT", "OUT"): 1}
    assert [m["case_id"] for m in report["mismatches"]] == ["corr:run-a:20"]

    # Drift replays the label-time ball: only run-b's prediction (IN on a
    # ball that judges OUT) flips; run-a's two predictions replay as recorded.
    assert report["drift_checked"] == 3
    assert report["drift"] == [
        {"case_id": "corr:run-b:10", "at_label_time": "IN", "now": "OUT"},
    ]


def test_evaluate_type_axis(tmp_path):
    cases, _ = build_eval_set(make_runs_tree(tmp_path))
    report = evaluate_cases(cases)

    # run-a:20 wall/wall, run-a:30 wall/none, run-b:10 wall/wall.
    assert report["type_checked"] == 3
    assert report["type_correct"] == 2
    assert report["type_confusion"] == {
        ("wall", "wall"): 2, ("wall", "none"): 1}
    assert report["false_positive_hits"] == ["corr:run-a:30"]


def test_evaluate_position_axis(tmp_path):
    cases, _ = build_eval_set(make_runs_tree(tmp_path))
    report = evaluate_cases(cases)

    # run-a:20: (310,210) vs (300,200) = sqrt(200); run-b:10: (300,55) vs
    # (300,50) = 5. The "none" case contributes no pair.
    pos = report["position"]
    assert pos["count"] == 2
    assert abs(pos["max"] - 200 ** 0.5) < 1e-9
    assert pos["max_case"] == "corr:run-a:20"
    assert abs(pos["mean"] - (5 + 200 ** 0.5) / 2) < 1e-9
    assert pos["by_source"]["detected_center"]["count"] == 1
    assert pos["by_source"]["impact_estimate"]["count"] == 1


def test_evaluate_timing_axis(tmp_path):
    cases, _ = build_eval_set(make_runs_tree(tmp_path))
    report = evaluate_cases(cases)

    # Timing is labeled on run-a:20 (wrong frame, +2), run-b:10 and
    # run-c:5 (confirmed); the "none" case carries no timing.
    timing = report["timing"]
    assert timing["checked"] == 3
    assert timing["frame_confirmed"] == 2
    assert timing["offsets"] == [(2, "corr:run-a:20")]


def test_evaluate_missed_bounce_axis(tmp_path):
    cases, _ = build_eval_set(make_runs_tree(tmp_path))
    report = evaluate_cases(cases)

    missed = report["missed"]
    assert missed["checked"] == 3
    assert missed["missed"] == [("gt:run-gt:300", "racket")]
    assert missed["missed_by_type"] == {"racket": 1}
    # Matched pairs: frame 100 wall/wall agrees, frame 200 wall-for-floor doesn't.
    assert missed["matched_type_checked"] == 2
    assert missed["matched_type_correct"] == 1


def test_ground_truth_skipped_without_detected_hits(tmp_path):
    run_dir = tmp_path / "runs" / "labelonly"
    run_dir.mkdir(parents=True)
    (run_dir / "ground_truth.json").write_text(json.dumps({
        "events": [{"frame": 10, "type": "floor"}],
    }))
    # NOTE: no detected_hits.json on purpose
    cases, manifest = build_eval_set(tmp_path / "runs")
    gt_cases = [c for c in cases if c["kind"] == "ground_truth_event"]
    assert gt_cases == []
    assert manifest["ground_truth_runs_skipped_no_detections"] == 1


def test_evaluate_empty_set():
    report = evaluate_cases([])
    assert report["accuracy"] is None
    assert report["judged"] == 0
    assert report["type_accuracy"] is None
    assert report["position"]["count"] == 0
    assert report["missed"]["checked"] == 0
