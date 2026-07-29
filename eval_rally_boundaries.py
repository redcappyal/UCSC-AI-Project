"""Score the rally segmenter's boundaries against labels.

An analysis with no eval axis is an opinion. Every threshold in
`rally_segmenter` is currently a first guess, and this is the number that lets
them be tuned by evidence rather than by how the output looks.

Two pieces:

  * `score_rallies` -- pure, greedy one-to-one midpoint matching within a
    tolerance. What it measures is whether the same rallies were found, not
    whether their edges agree to the frame; predicted and labeled structures
    come from different signals and their borders legitimately differ.

  * the CLI -- re-runs the segmenter over each labeled video's real audio and
    motion and reports per-video and aggregate F1.

**Labels here are silver, not gold.** They are derived by clustering
*human-labeled hit frames* into rallies, so what the aggregate measures is
agreement with a human's hit labels under a clustering rule, not absolute
truth. They are human hits rather than detected ones on purpose: scoring
against detector output would measure the segmenter against the very recall
problem it exists to route around.

Usage:

    .venv/Scripts/python.exe eval_rally_boundaries.py \
        --labels eval_set/rally_labels.jsonl
"""

import argparse
import json
from pathlib import Path

# Default matching tolerance, from design spec §8: rally boundaries are useful
# to about a second and a half, which is also roughly how far apart two honest
# humans put the same boundary.
DEFAULT_TOLERANCE_S = 1.5

# Gap between human-labeled hits that separates one rally from the next when
# seeding silver labels. Deliberately the segmenter's own default rather than
# its adaptive inference: a label set whose boundaries were chosen by the
# algorithm being scored would be marking its own homework.
LABEL_GAP_S = 5.0

# A rally needs at least two contacts. One labeled hit in isolation is a knock,
# a warm-up, or a mislabel, and calling it a rally inflates both counts.
MIN_LABEL_HITS = 2


def _midpoint(rally):
    return (float(rally["start_s"]) + float(rally["end_s"])) / 2.0


def score_rallies(predicted, labeled, tol_s=DEFAULT_TOLERANCE_S):
    """Greedy one-to-one match of predicted rallies to labeled ones.

    One-to-one matters more than it looks: without it, over-segmenting one
    rally into five scores as five true positives, and the failure mode reads
    as a perfect score.
    """
    unclaimed = list(range(len(labeled)))
    true_positives = 0

    for rally in sorted(predicted, key=_midpoint):
        centre = _midpoint(rally)
        best = None
        for position, index in enumerate(unclaimed):
            distance = abs(centre - _midpoint(labeled[index]))
            if distance <= tol_s and (best is None or distance < best[0]):
                best = (distance, position)
        if best is not None:
            unclaimed.pop(best[1])
            true_positives += 1

    false_positives = len(predicted) - true_positives
    false_negatives = len(labeled) - true_positives

    if not predicted and not labeled:
        # A clip with no rallies, correctly reported as having none, is a
        # perfect answer -- not an undefined one.
        precision = recall = f1 = 1.0
    else:
        precision = (
            true_positives / len(predicted) if predicted else 0.0
        )
        recall = true_positives / len(labeled) if labeled else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_delta": len(predicted) - len(labeled),
    }


def cluster_hit_frames_into_rallies(hit_frames, fps, gap_s=LABEL_GAP_S,
                                    min_hits=MIN_LABEL_HITS):
    """Human-labeled hit frames -> silver rally spans.

    `fps` is required and must be positive. Frame numbers without a frame rate
    are anonymous integers -- the same point the README makes about the label
    CSVs themselves, which is why every one of them carries a `.meta.json`
    sidecar recording fps and the video sha.
    """
    if not fps or fps <= 0:
        raise ValueError("hit frames need a positive fps to become times")

    times = sorted(frame / float(fps) for frame in hit_frames)
    if not times:
        return []

    runs = [[times[0]]]
    for time in times[1:]:
        if time - runs[-1][-1] > gap_s:
            runs.append([time])
        else:
            runs[-1].append(time)

    return [
        {"start_s": run[0], "end_s": run[-1], "hit_count": len(run)}
        for run in runs
        if len(run) >= min_hits
    ]


def load_labels(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]


def rallies_for_video(video_path, duration_s=None):
    """Re-run the segmenter over a video's real audio and motion.

    Imported lazily so the scorer above stays usable -- and testable -- without
    the decode stack.
    """
    import cv2

    import job_runner
    from audio_events import extract_audio_candidates
    from rally_segmenter import build_rally_timeline

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    end_frame = max(0, frame_count - 1)
    motion = job_runner.accumulate_motion_only(video_path, 0, end_frame, fps)
    audio = extract_audio_candidates(
        video_path, 0, end_frame, fps,
        max_peaks=job_runner.AUDIO_TIMELINE_MAX_PEAKS,
    )
    impacts = (
        None if audio is None
        else [float(c["time_seconds"]) for c in audio if "time_seconds" in c]
    )
    timeline = build_rally_timeline(
        impacts, motion.series(), duration_s or (frame_count / fps), None
    )
    return timeline["rallies"], timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="eval_set/rally_labels.jsonl")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    print(f"Rally-boundary eval: {len(labels)} labeled video(s), "
          f"tolerance +/-{args.tolerance}s\n")

    scored = []
    skipped = []
    for entry in labels:
        video_path = Path(entry["video_path"])
        labeled = entry["rallies"]
        if not video_path.exists():
            # Counted and named, never silently dropped: a label set that
            # scores 1.0 because most of it was skipped is worse than no
            # number at all.
            skipped.append((entry, "source video not on this machine"))
            continue
        try:
            predicted, _ = rallies_for_video(video_path)
        except Exception as error:
            skipped.append((entry, str(error)))
            continue
        score = score_rallies(predicted, labeled, tol_s=args.tolerance)
        scored.append((entry, score))
        print(f"{video_path.name}: F1 {score['f1']:.3f} "
              f"(tp {score['tp']} fp {score['fp']} fn {score['fn']}, "
              f"labeled {len(labeled)}, predicted {len(predicted)})")

    if scored:
        total_tp = sum(s["tp"] for _, s in scored)
        total_fp = sum(s["fp"] for _, s in scored)
        total_fn = sum(s["fn"] for _, s in scored)
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        print(f"\nAggregate: F1 {f1:.3f}  precision {precision:.3f}  "
              f"recall {recall:.3f}  (tp {total_tp} fp {total_fp} fn {total_fn})")
    else:
        print("\nAggregate: NOT MEASURED - no labeled video was scoreable.")

    for entry, reason in skipped:
        print(f"  skipped {Path(entry['video_path']).name}: {reason}")


if __name__ == "__main__":
    main()
