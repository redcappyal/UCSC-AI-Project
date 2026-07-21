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
import csv
import json
import re
from pathlib import Path

SCHEMA_VERSION = "bounce-eval-v3"
HIT_TYPES = {"wall", "side_wall", "floor", "racket"}
CORRECTION_TYPES = HIT_TYPES | {"none"}
SHA_NAME_RE = re.compile(r"^[0-9a-f]{64}$")
LABEL_SIDECAR_SCHEMA = "label-run-v1"


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
        # Provenance, absent on runs predating it: lets an eval regression be
        # attributed to a model version rather than guessed at.
        "model_id": job.get("model_id"),
        "tracking_backend": job.get("tracking_backend"),
        "app_version_at_track": job.get("app_version"),
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
    """Labeling-mode events -> cases with a build-time detector match."""
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

    detected = load_json(run_dir / "detected_hits.json") or {}
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
           ("video_sha", "video_path", "fps", "frame_stride",
            "model_id", "tracking_backend")},
    } for event in events]


def index_runs_by_video_sha(runs_dir):
    """video_sha -> (run_id, detected hits, frame_stride) for every run that
    produced detections. Label CSVs carry no run of their own, so this is how
    an offline labeling session finds the detector output to be scored against."""
    index = {}
    for job_path in sorted(Path(runs_dir).glob("*/job.json")):
        run_dir = job_path.parent
        context = run_context(run_dir)
        sha = context["video_sha"]
        if not sha:
            continue
        detected = load_json(run_dir / "detected_hits.json") or {}
        hits = [h for h in detected.get("hits", [])
                if isinstance(h, dict) and isinstance(h.get("frame"), int)]
        prior = index.get(sha)
        # More detections = the more complete run; ties break on run_id, whose
        # embedded creation time makes larger mean newer.
        if prior is None or (len(hits), run_dir.name) > (len(prior["hits"]), prior["run_id"]):
            index[sha] = {"run_id": run_dir.name, "hits": hits,
                          "frame_stride": context["frame_stride"],
                          "model_id": context["model_id"],
                          "tracking_backend": context["tracking_backend"]}
    return index


def collect_label_csv_cases(labels_dir, run_index):
    """`label_hits.py` CSVs (+ their .meta.json sidecars) -> wall-hit cases.

    These are pure frame labels from offline labeling, so they only carry the
    `wall` type. A CSV whose video was never tracked has no detector output to
    compare against; it is still emitted, flagged `no_detector_run`, and the
    missed-bounce axis skips it — otherwise "never ran the model" would score
    identically to "model missed everything".
    """
    cases = []
    stats = {"label_files": 0, "label_files_without_sidecar": 0,
             "label_files_without_run": 0}

    for sidecar in sorted(Path(labels_dir).glob("*.meta.json")):
        meta = load_json(sidecar)
        if not meta or meta.get("schema_version") != LABEL_SIDECAR_SCHEMA:
            continue
        csv_path = sidecar.with_suffix("").with_suffix(".csv")
        if not csv_path.exists():
            stats["label_files_without_sidecar"] += 1
            continue

        frames = load_label_frames(csv_path)
        if not frames:
            continue
        stats["label_files"] += 1

        sha = meta.get("video_sha")
        run = run_index.get(sha) if sha else None
        if run is None:
            stats["label_files_without_run"] += 1

        events = [{"frame": frame, "type": "wall"} for frame in frames]
        tolerance = max(1, int(run["frame_stride"] or 1) if run else 1)
        matches = match_events_to_hits(events, run["hits"], tolerance) if run else {}

        label_source = csv_path.name
        for frame in frames:
            cases.append({
                "kind": "label_csv_event",
                "case_id": f"lbl:{label_source}:{frame}",
                "run_id": run["run_id"] if run else None,
                "label_source": label_source,
                "frame": frame,
                "human_type": "wall",
                "tolerance_frames": tolerance,
                "matched_detected": matches.get(frame),
                "no_detector_run": run is None,
                "video_sha": sha,
                "video_path": meta.get("video_path"),
                "fps": meta.get("fps"),
                "frame_stride": run["frame_stride"] if run else None,
                "model_id": run["model_id"] if run else None,
                "tracking_backend": run["tracking_backend"] if run else None,
            })
    return cases, stats


def load_label_frames(csv_path):
    """Frame numbers from a label_hits.py CSV, tolerating its historical
    header variants."""
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            column = next((name for name in ("hit_frame", "source_frame", "frame")
                           if name in reader.fieldnames), None)
            if column is None:
                return []
            frames = set()
            for row in reader:
                try:
                    frames.add(int(str(row[column]).strip()))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    return sorted(frames)


def dedupe_label_cases(cases):
    """One case per (video, frame); a video labeled twice keeps the session
    that labeled more frames."""
    per_source = {}
    for case in cases:
        per_source[case["label_source"]] = per_source.get(case["label_source"], 0) + 1
    best = {}
    for case in cases:
        key = (case["video_sha"] or f"src:{case['label_source']}", case["frame"])
        prior = best.get(key)
        if prior is None:
            best[key] = case
            continue
        new = (per_source[case["label_source"]], case["label_source"])
        old = (per_source[prior["label_source"]], prior["label_source"])
        if new > old:
            best[key] = case
    return list(best.values()), len(cases) - len(best)


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


def build_eval_set(runs_dir, labels_dir=None, merge_dir=None):
    runs_dir = Path(runs_dir)
    correction_cases, gt_cases = [], []
    runs_with_corrections = runs_with_ground_truth = 0
    unreadable_runs = []
    skipped_legacy = 0

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
        if gt:
            runs_with_ground_truth += 1
            gt_cases.extend(gt)

    label_cases, label_stats = ([], {"label_files": 0,
                                     "label_files_without_sidecar": 0,
                                     "label_files_without_run": 0})
    if labels_dir is not None:
        label_cases, label_stats = collect_label_csv_cases(
            labels_dir, index_runs_by_video_sha(runs_dir)
        )

    correction_cases, corr_dupes = dedupe_corrections(correction_cases)
    gt_cases, gt_dupes = dedupe_ground_truth(gt_cases)
    label_cases, label_dupes = dedupe_label_cases(label_cases)
    cases = sorted(correction_cases + gt_cases + label_cases,
                   key=lambda c: (c["kind"], c["run_id"] or "", c["frame"]))

    preserved_count = 0
    if merge_dir is not None:
        cases, preserved_count = merge_with_existing(cases, merge_dir)
        # Preserved cases still belong in the per-kind tallies below, which
        # are what the manifest reports.
        rebuilt = {id(c) for c in correction_cases + gt_cases + label_cases}
        for case in cases:
            if id(case) in rebuilt:
                continue
            kind = case.get("kind")
            if kind == "correction":
                correction_cases.append(case)
            elif kind == "ground_truth_event":
                gt_cases.append(case)
            elif kind == "label_csv_event":
                label_cases.append(case)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(cases),
        "correction_cases": len(correction_cases),
        "ground_truth_cases": len(gt_cases),
        "label_csv_cases": len(label_cases),
        "label_csv_files": label_stats["label_files"],
        # Tallies use .get throughout: preserved cases come from a possibly
        # older schema and need not carry every field this build emits.
        "label_csv_cases_without_detector_run": sum(
            1 for c in label_cases if c.get("no_detector_run")),
        "runs_with_corrections": runs_with_corrections,
        "runs_with_ground_truth": runs_with_ground_truth,
        "unreadable_runs": unreadable_runs,
        "duplicates_dropped": corr_dupes + gt_dupes + label_dupes,
        # Cases carried over from the committed set because this checkout has
        # no ui_runs/ for them. Non-zero is normal on a teammate's machine.
        "preserved_cases": preserved_count,
        "skipped_legacy_entries": skipped_legacy,
        "cases_without_video_sha": sum(
            1 for c in cases if c.get("video_sha") is None),
        "cases_without_calibration": sum(
            1 for c in correction_cases if c.get("calibration") is None),
        "types": {
            kind: sum(1 for c in correction_cases
                      if (c.get("human") or {}).get("type") == kind)
            for kind in sorted(CORRECTION_TYPES)
        },
        "calls": {
            call: sum(1 for c in correction_cases
                      if (c.get("human") or {}).get("call") == call)
            for call in ("IN", "OUT")
        },
        "gt_missed_at_build": sum(
            1 for c in gt_cases if c.get("matched_detected") is None),
        # Only scorable label cases count — a video that was never tracked
        # would otherwise read as a total detector failure.
        "label_missed_at_build": sum(
            1 for c in label_cases
            if not c.get("no_detector_run") and c.get("matched_detected") is None),
        "label_scorable_cases": sum(
            1 for c in label_cases if not c.get("no_detector_run")),
        "latest_correction_at": max(
            (c["recorded_at"] for c in correction_cases if c.get("recorded_at")),
            default=None),
    }
    return cases, manifest


def merge_with_existing(cases, out_dir):
    """Keep cases this machine cannot rebuild.

    eval_set/ is git-tracked but derived from gitignored ui_runs/, so a
    teammate's checkout can hold labels whose source runs live on someone
    else's disk. A plain overwrite deletes those and looks like an ordinary
    diff. Rebuilt cases win by case_id; everything else is carried forward.
    """
    existing_path = Path(out_dir) / "cases.jsonl"
    if not existing_path.exists():
        return cases, 0

    rebuilt_ids = {case["case_id"] for case in cases}
    preserved = []
    try:
        with existing_path.open(encoding="utf-8") as lines:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if case.get("case_id") not in rebuilt_ids:
                    preserved.append(case)
    except OSError:
        return cases, 0

    merged = sorted(cases + preserved,
                    key=lambda c: (c.get("kind", ""), c.get("run_id") or "",
                                   c.get("frame", 0)))
    return merged, len(preserved)


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
    parser.add_argument(
        "--labels-dir", type=Path, default=Path(__file__).parent,
        help="Directory scanned for label_hits.py CSVs and their .meta.json "
             "sidecars. Pass --no-labels to skip.")
    parser.add_argument("--no-labels", action="store_true",
                        help="Ignore label_hits.py CSVs entirely.")
    parser.add_argument(
        "--replace", action="store_true",
        help="Rebuild from scratch, dropping committed cases this machine "
             "cannot regenerate. Default is to merge into the existing set.")
    args = parser.parse_args()

    cases, manifest = build_eval_set(
        args.runs_dir,
        None if args.no_labels else args.labels_dir,
        merge_dir=None if args.replace else args.out,
    )
    lines_path = write_eval_set(cases, manifest, args.out)

    print(f"Eval set: {manifest['correction_cases']} correction + "
          f"{manifest['ground_truth_cases']} ground-truth + "
          f"{manifest['label_csv_cases']} label-CSV cases "
          f"from {manifest['runs_with_corrections']}/"
          f"{manifest['runs_with_ground_truth']} runs and "
          f"{manifest['label_csv_files']} label file(s) -> {lines_path}")
    if manifest["preserved_cases"]:
        print(f"  {manifest['preserved_cases']} case(s) carried over from the "
              "committed set (no ui_runs/ here to rebuild them)")
    if manifest["label_csv_cases_without_detector_run"]:
        print(f"  {manifest['label_csv_cases_without_detector_run']} label case(s) "
              "have no tracking run for their video — track it to score them")
    print(f"  types: {manifest['types']}  calls: {manifest['calls']}")
    if manifest["gt_missed_at_build"]:
        print(f"  {manifest['gt_missed_at_build']} labeled bounce(s) had no "
              "detected hit at build time (missed-bounce eval axis)")
    if manifest["duplicates_dropped"]:
        print(f"  {manifest['duplicates_dropped']} stale duplicate label(s) dropped")
    if manifest["skipped_legacy_entries"]:
        print(f"  {manifest['skipped_legacy_entries']} legacy/malformed "
              "correction(s) skipped — re-label them in the UI")
    if manifest["unreadable_runs"]:
        print(f"  WARNING unreadable corrections in: {manifest['unreadable_runs']}")
    if manifest["cases_without_calibration"]:
        print(f"  {manifest['cases_without_calibration']} case(s) lack calibration "
              "(kept for tier-2 replay; geometric axes will skip them)")


if __name__ == "__main__":
    main()
