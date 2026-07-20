"""Distill human bounce labels from run dirs into a versioned eval set.

Two label streams feed the set, both living in gitignored ui_runs/:

- `<run_dir>/corrections.json` (schema corrections-v2): per-hit human
  verdicts — hit type, corrected ball position, bounce timing, and IN/OUT
  for front-wall hits — captured by the track-phase correction panel.
- `<run_dir>/ground_truth.json`: labeling-mode event lists ({frame, type}),
  which also record bounces the detector never saw. Each event is matched
  against the run's detected_hits.json here, at build time, so the eval
  keeps needing no ui_runs, no video, and no model.

Output: `eval_set/cases.jsonl` + `eval_set/manifest.json` — small,
deterministic, git-trackable artifacts that grow as labels accumulate.
`eval_line_calls.py` replays multi-axis evals against them.

Usage: python build_eval_set.py [--runs-dir ui_runs] [--out eval_set]
"""

import argparse
import json
import re
from pathlib import Path

SCHEMA_VERSION = "bounce-eval-v2"
HIT_TYPES = {"wall", "side_wall", "floor", "racket"}
CORRECTION_TYPES = HIT_TYPES | {"none"}
SHA_NAME_RE = re.compile(r"^[0-9a-f]{64}$")


def video_sha_from_path(video_path):
    """The by-hash upload scheme names files <sha256>.<ext>; anything else
    (older runs, external paths) has no stable video identity."""
    if not video_path:
        return None
    stem = Path(video_path).stem
    return stem if SHA_NAME_RE.match(stem) else None


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_context(run_dir):
    """Shared per-run fields every case embeds."""
    job = load_json(run_dir / "job.json") or {}
    video_path = job.get("video_path")
    return {
        "video_sha": video_sha_from_path(video_path),
        "video_path": video_path,
        "fps": job.get("fps"),
        "frame_stride": job.get("frame_stride"),
        "inference_width": job.get("inference_width"),
    }


def collect_correction_cases(run_dir, context):
    """Well-formed v2 corrections in one run dir -> (cases, skipped_legacy)."""
    data = load_json(run_dir / "corrections.json")
    if data is None:
        return None, 0
    corrections = data.get("corrections")
    if not isinstance(corrections, list):
        return None, 0

    calibration = load_json(run_dir / "calibration.json")
    cases, skipped = [], 0
    for corr in corrections:
        corrected = corr.get("corrected") if isinstance(corr, dict) else None
        try:
            frame = int(corr["frame"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if (not isinstance(corrected, dict)
                or corrected.get("type") not in CORRECTION_TYPES):
            skipped += 1          # v1 entries and malformed hand-edits
            continue
        cases.append({
            "kind": "correction",
            "case_id": f"corr:{run_dir.name}:{frame}",
            "run_id": run_dir.name,
            "frame": frame,
            "human": corrected,
            "predicted_at_label_time": corr.get("predicted"),
            "agrees_at_label_time": corr.get("agrees"),
            "calibration": calibration,
            "recorded_at": corr.get("recorded_at"),
            "app_version": corr.get("app_version"),
            **context,
        })
    return cases, skipped


def match_events_to_hits(events, hits, tolerance):
    """Greedy-nearest one-to-one matching of ground-truth events to detected
    hits: sorted events each claim their nearest unclaimed hit within the
    tolerance, so one detected hit can't satisfy two labeled bounces."""
    claimed = set()
    matches = {}
    for event in sorted(events, key=lambda e: e["frame"]):
        best_index, best_dist = None, None
        for index, hit in enumerate(hits):
            if index in claimed:
                continue
            dist = abs(hit["frame"] - event["frame"])
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_index, best_dist = index, dist
        if best_index is not None:
            claimed.add(best_index)
            hit = hits[best_index]
            matches[event["frame"]] = {
                "frame": hit["frame"],
                "event_type": hit.get("event_type"),
                "distance_frames": best_dist,
            }
    return matches


def collect_ground_truth_cases(run_dir, context):
    """Labeling-mode events -> cases with a build-time detector match.

    Returns None when the run has no detected_hits.json at all (a
    label-only run where detection never ran): an unmatched event there
    isn't a detector miss, so the caller must skip and count it rather
    than fold it into the missed-bounce axis as a silent false miss.
    """
    data = load_json(run_dir / "ground_truth.json")
    if data is None:
        return []
    raw_events = data.get("events")
    if not isinstance(raw_events, list):
        return []

    events = []
    for event in raw_events:
        try:
            events.append({"frame": int(event["frame"]),
                           "type": str(event["type"]).lower()})
        except (KeyError, TypeError, ValueError):
            continue
    events = [e for e in events if e["type"] in HIT_TYPES]
    if not events:
        return []

    detected_path = run_dir / "detected_hits.json"
    if not detected_path.exists():
        # Label-only run: detection never ran, so an unmatched event is not
        # a detector miss. Skip the run and count it in the manifest.
        return None
    detected = load_json(detected_path) or {}
    hits = [h for h in detected.get("hits", [])
            if isinstance(h, dict) and isinstance(h.get("frame"), int)]
    tolerance = max(int(data.get("tolerance_frames") or 1),
                    int(context["frame_stride"] or 1))
    matches = match_events_to_hits(events, hits, tolerance)

    return [{
        "kind": "ground_truth_event",
        "case_id": f"gt:{run_dir.name}:{event['frame']}",
        "run_id": run_dir.name,
        "frame": event["frame"],
        "human_type": event["type"],
        "tolerance_frames": tolerance,
        "matched_detected": matches.get(event["frame"]),
        **{k: context[k] for k in
           ("video_sha", "video_path", "fps", "frame_stride")},
    } for event in events]


def dedupe_corrections(cases):
    """One case per physical moment: the same video frame labeled in several
    runs keeps only the freshest label. Runs without a video sha can't alias
    another run's video, so they dedupe within the run only (by frame)."""
    best = {}
    for case in cases:
        key = (case["video_sha"] or f"run:{case['run_id']}", case["frame"])
        prior = best.get(key)
        if prior is None:
            best[key] = case
            continue
        # recorded_at is ISO-8601 UTC, so string order is time order; break
        # exact ties by run_id for determinism.
        new = (case["recorded_at"] or "", case["run_id"])
        old = (prior["recorded_at"] or "", prior["run_id"])
        if new > old:
            best[key] = case
    return list(best.values()), len(cases) - len(best)


def dedupe_ground_truth(cases):
    """ground_truth.json has no timestamps, so when two runs of the same
    video label the same frame, keep the run that labeled more events
    (the more complete session), then the larger run_id (run ids embed
    creation time, so larger = newer)."""
    events_per_run = {}
    for case in cases:
        events_per_run[case["run_id"]] = events_per_run.get(case["run_id"], 0) + 1
    best = {}
    for case in cases:
        key = (case["video_sha"] or f"run:{case['run_id']}", case["frame"])
        prior = best.get(key)
        if prior is None:
            best[key] = case
            continue
        new = (events_per_run[case["run_id"]], case["run_id"])
        old = (events_per_run[prior["run_id"]], prior["run_id"])
        if new > old:
            best[key] = case
    return list(best.values()), len(cases) - len(best)


def build_eval_set(runs_dir):
    runs_dir = Path(runs_dir)
    correction_cases, gt_cases = [], []
    runs_with_corrections = runs_with_ground_truth = 0
    unreadable_runs = []
    skipped_legacy = 0
    ground_truth_runs_skipped_no_detections = 0

    run_dirs = sorted({p.parent for pattern in
                       ("*/corrections.json", "*/ground_truth.json")
                       for p in runs_dir.glob(pattern)})
    for run_dir in run_dirs:
        context = run_context(run_dir)
        if (run_dir / "corrections.json").exists():
            cases, skipped = collect_correction_cases(run_dir, context)
            skipped_legacy += skipped
            if cases is None:
                unreadable_runs.append(run_dir.name)
            else:
                runs_with_corrections += 1
                correction_cases.extend(cases)
        gt = collect_ground_truth_cases(run_dir, context)
        if gt is None:
            ground_truth_runs_skipped_no_detections += 1
        elif gt:
            runs_with_ground_truth += 1
            gt_cases.extend(gt)

    correction_cases, corr_dupes = dedupe_corrections(correction_cases)
    gt_cases, gt_dupes = dedupe_ground_truth(gt_cases)
    cases = sorted(correction_cases + gt_cases,
                   key=lambda c: (c["kind"], c["run_id"], c["frame"]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(cases),
        "correction_cases": len(correction_cases),
        "ground_truth_cases": len(gt_cases),
        "runs_with_corrections": runs_with_corrections,
        "runs_with_ground_truth": runs_with_ground_truth,
        "unreadable_runs": unreadable_runs,
        "duplicates_dropped": corr_dupes + gt_dupes,
        "skipped_legacy_entries": skipped_legacy,
        "ground_truth_runs_skipped_no_detections":
            ground_truth_runs_skipped_no_detections,
        "cases_without_video_sha": sum(
            1 for c in cases if c["video_sha"] is None),
        "cases_without_calibration": sum(
            1 for c in correction_cases if c["calibration"] is None),
        "types": {
            kind: sum(1 for c in correction_cases
                      if c["human"]["type"] == kind)
            for kind in sorted(CORRECTION_TYPES)
        },
        "calls": {
            call: sum(1 for c in correction_cases
                      if c["human"].get("call") == call)
            for call in ("IN", "OUT")
        },
        "gt_missed_at_build": sum(
            1 for c in gt_cases if c["matched_detected"] is None),
        "latest_correction_at": max(
            (c["recorded_at"] for c in correction_cases if c["recorded_at"]),
            default=None),
    }
    return cases, manifest


def write_eval_set(cases, manifest, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines_path = out_dir / "cases.jsonl"
    with lines_path.open("w", encoding="utf-8") as out:
        for case in cases:
            out.write(json.dumps(case, sort_keys=True) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lines_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path,
                        default=Path(__file__).with_name("ui_runs"))
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("eval_set"))
    args = parser.parse_args()

    cases, manifest = build_eval_set(args.runs_dir)
    lines_path = write_eval_set(cases, manifest, args.out)

    print(f"Eval set: {manifest['correction_cases']} correction + "
          f"{manifest['ground_truth_cases']} ground-truth cases "
          f"from {manifest['runs_with_corrections']}/"
          f"{manifest['runs_with_ground_truth']} runs -> {lines_path}")
    print(f"  types: {manifest['types']}  calls: {manifest['calls']}")
    if manifest["gt_missed_at_build"]:
        print(f"  {manifest['gt_missed_at_build']} labeled bounce(s) had no "
              "detected hit at build time (missed-bounce eval axis)")
    if manifest["duplicates_dropped"]:
        print(f"  {manifest['duplicates_dropped']} stale duplicate label(s) dropped")
    if manifest["skipped_legacy_entries"]:
        print(f"  {manifest['skipped_legacy_entries']} legacy/malformed "
              "correction(s) skipped — re-label them in the UI")
    if manifest["ground_truth_runs_skipped_no_detections"]:
        print(f"  {manifest['ground_truth_runs_skipped_no_detections']} "
              "label-only run(s) skipped — no detected_hits.json, so their "
              "events can't count as detector misses")
    if manifest["unreadable_runs"]:
        print(f"  WARNING unreadable corrections in: {manifest['unreadable_runs']}")
    if manifest["cases_without_calibration"]:
        print(f"  {manifest['cases_without_calibration']} case(s) lack calibration "
              "(kept for tier-2 replay; geometric axes will skip them)")


if __name__ == "__main__":
    main()
