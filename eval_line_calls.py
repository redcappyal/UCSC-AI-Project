"""Replay multi-axis bounce evals against the human-labeled eval set.

Reads `eval_set/cases.jsonl` (see build_eval_set.py). Correction cases
carry the human's hit type, ball position, and timing; ground-truth cases
carry labeling-mode events with their build-time detector match. No video,
no model, sub-second: this is the regression gate for changes to
judge_call.py, calibration handling, and impact estimation, and the
scoreboard for detector quality (type confusion, position error, timing,
missed bounces).

Axes:
- IN/OUT accuracy: judge replayed on the human-corrected ball vs the human
  call — geometry/calibration correctness, isolated from detector error.
  `--fail-under PCT` gates this axis.
- drift: judge replayed on the label-time predicted ball vs the label-time
  call — exactly what a judge/calibration code change flipped.
- type: detector's hit type vs human type, incl. false-positive rate.
- position: px distance between predicted and corrected ball, by source.
- timing: how often the detected frame was the true bounce frame.
- missed bounces: labeled events with no detected hit at build time.

Usage: python eval_line_calls.py [--eval-set eval_set/cases.jsonl]
       [--verbose] [--fail-under PCT]
"""

import argparse
import json
import sys
from pathlib import Path

from judge_call import Point, judge_ball, judge_margin_px, load_calibration_lines

HIT_TYPES = ("wall", "side_wall", "floor", "racket")


def replay_judge(ball, calibration):
    """-> (call, margin_px) or (None, reason) when unreplayable."""
    if ball is None:
        return None, "no ball point recorded"
    if calibration is None:
        return None, "no calibration in run dir"
    try:
        top_line, bottom_line = load_calibration_lines(calibration)
        point = Point(float(ball["x"]), float(ball["y"]))
        call, _reason, _top_y, _bottom_y = judge_ball(point, top_line, bottom_line)
        margin = judge_margin_px(point, top_line, bottom_line)
    except (ValueError, KeyError, TypeError) as error:
        return None, str(error)
    return call, margin


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def evaluate_cases(cases):
    corrections = [c for c in cases if c.get("kind") == "correction"]
    gt_events = [c for c in cases if c.get("kind") == "ground_truth_event"]

    report = {
        "total_cases": len(cases),
        "correction_cases": len(corrections),
        "gt_cases": len(gt_events),
        # axis 1: IN/OUT geometry vs the human call
        "judged": 0, "correct": 0, "accuracy": None,
        "confusion": {}, "mismatches": [], "unreplayable": [],
        # axis 2: drift vs label-time prediction
        "drift": [], "drift_checked": 0,
        # axis 3: type classification
        "type_confusion": {}, "type_checked": 0, "type_correct": 0,
        "type_accuracy": None, "false_positive_hits": [],
        # axis 4: position error
        "position": {"count": 0, "mean": None, "median": None,
                     "p90": None, "max": None, "max_case": None,
                     "by_source": {}},
        # axis 5: timing
        "timing": {"checked": 0, "frame_confirmed": 0, "offsets": []},
        # axis 6: missed bounces
        "missed": {"checked": len(gt_events), "missed": [],
                   "missed_by_type": {}, "matched_type_checked": 0,
                   "matched_type_correct": 0},
    }

    position_errors = []
    for case in corrections:
        human = case["human"]
        predicted = case.get("predicted_at_label_time") or {}

        # -- axis 3: type ------------------------------------------------
        predicted_type = predicted.get("type")
        if predicted_type is not None:
            report["type_checked"] += 1
            key = (predicted_type, human["type"])
            report["type_confusion"][key] = report["type_confusion"].get(key, 0) + 1
            if predicted_type == human["type"]:
                report["type_correct"] += 1
        if human["type"] == "none":
            report["false_positive_hits"].append(case["case_id"])

        # -- axis 4: position -------------------------------------------
        predicted_ball, human_ball = predicted.get("ball"), human.get("ball")
        if predicted_ball and human_ball and human["type"] != "none":
            error = ((predicted_ball["x"] - human_ball["x"]) ** 2
                     + (predicted_ball["y"] - human_ball["y"]) ** 2) ** 0.5
            source = predicted.get("source") or "unknown"
            position_errors.append((error, case["case_id"], source))

        # -- axis 5: timing ---------------------------------------------
        if isinstance(human.get("frame_is_bounce"), bool):
            report["timing"]["checked"] += 1
            if human["frame_is_bounce"]:
                report["timing"]["frame_confirmed"] += 1
            elif human.get("frame") is not None:
                report["timing"]["offsets"].append(
                    (human["frame"] - case["frame"], case["case_id"]))

        # -- axis 1: IN/OUT on the corrected ball -----------------------
        if human["type"] == "wall" and human.get("call") in ("IN", "OUT"):
            call, margin_or_reason = replay_judge(
                human_ball, case.get("calibration"))
            if call is None:
                report["unreplayable"].append((case["case_id"], margin_or_reason))
            else:
                report["judged"] += 1
                key = (call, human["call"])
                report["confusion"][key] = report["confusion"].get(key, 0) + 1
                if call == human["call"]:
                    report["correct"] += 1
                else:
                    report["mismatches"].append({
                        "case_id": case["case_id"], "replay": call,
                        "human": human["call"], "margin_px": margin_or_reason,
                        "source": predicted.get("source"),
                    })

        # -- axis 2: drift on the label-time ball -----------------------
        recorded = predicted.get("call")
        if recorded in ("IN", "OUT") and predicted_ball:
            call, _ = replay_judge(predicted_ball, case.get("calibration"))
            if call is not None:
                report["drift_checked"] += 1
                if call != recorded:
                    report["drift"].append({
                        "case_id": case["case_id"],
                        "at_label_time": recorded, "now": call,
                    })

    if report["judged"]:
        report["accuracy"] = report["correct"] / report["judged"]
    if report["type_checked"]:
        report["type_accuracy"] = report["type_correct"] / report["type_checked"]

    if position_errors:
        errors = [e for e, _, _ in position_errors]
        worst = max(position_errors)
        by_source = {}
        for error, _, source in position_errors:
            by_source.setdefault(source, []).append(error)
        report["position"] = {
            "count": len(errors),
            "mean": sum(errors) / len(errors),
            "median": percentile(errors, 50),
            "p90": percentile(errors, 90),
            "max": worst[0], "max_case": worst[1],
            "by_source": {s: {"count": len(v), "mean": sum(v) / len(v)}
                          for s, v in sorted(by_source.items())},
        }

    # -- axis 6: missed bounces -----------------------------------------
    missed = report["missed"]
    for case in gt_events:
        match = case.get("matched_detected")
        if match is None:
            missed["missed"].append((case["case_id"], case["human_type"]))
            missed["missed_by_type"][case["human_type"]] = (
                missed["missed_by_type"].get(case["human_type"], 0) + 1)
        elif match.get("event_type") is not None:
            missed["matched_type_checked"] += 1
            if match["event_type"] == case["human_type"]:
                missed["matched_type_correct"] += 1
    return report


def load_eval_set(path):
    cases = []
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def print_report(report, verbose=False):
    print(f"Cases: {report['total_cases']} total "
          f"({report['correction_cases']} corrections, "
          f"{report['gt_cases']} ground-truth events)")

    print("\n== IN/OUT vs human calls (judge on corrected ball) ==")
    if report["accuracy"] is not None:
        print(f"Accuracy: {report['correct']}/{report['judged']} "
              f"= {report['accuracy'] * 100:.1f}%")
        for (replay, human), count in sorted(report["confusion"].items()):
            marker = "" if replay == human else "   <-- wrong"
            print(f"  judge {replay} / human {human}: {count}{marker}")
        for miss in report["mismatches"]:
            margin = miss["margin_px"]
            margin_text = f"{margin:+.1f}px" if isinstance(margin, float) else "?"
            print(f"  {miss['case_id']}: judge says {miss['replay']} "
                  f"({margin_text}, {miss['source']}), human says {miss['human']}")
    else:
        print("No replayable wall calls yet - record corrections in the UI.")
    if report["unreplayable"] and verbose:
        for case_id, reason in report["unreplayable"]:
            print(f"  unreplayable {case_id}: {reason}")

    if report["drift"]:
        print(f"\nDRIFT: {len(report['drift'])}/{report['drift_checked']} "
              "call(s) differ from label-time predictions - judge logic "
              "changed since these were recorded:")
        for d in report["drift"]:
            print(f"  {d['case_id']}: was {d['at_label_time']}, now {d['now']}")
    elif report["drift_checked"]:
        print(f"\nDrift: 0/{report['drift_checked']} - judge output "
              "unchanged since labeling.")

    print("\n== Hit-type classification ==")
    if report["type_checked"]:
        print(f"Accuracy: {report['type_correct']}/{report['type_checked']} "
              f"= {report['type_accuracy'] * 100:.1f}%")
        for (pred, human), count in sorted(report["type_confusion"].items()):
            marker = "" if pred == human else "   <-- wrong"
            print(f"  detector {pred} / human {human}: {count}{marker}")
    else:
        print("No cases with a predicted type yet.")
    if report["false_positive_hits"]:
        print(f"Detector false-positive hits (human: not a hit): "
              f"{len(report['false_positive_hits'])}")
        if verbose:
            for case_id in report["false_positive_hits"]:
                print(f"  {case_id}")

    print("\n== Ball-position error (predicted vs corrected, px) ==")
    pos = report["position"]
    if pos["count"]:
        print(f"n={pos['count']}  mean {pos['mean']:.1f}  "
              f"median {pos['median']:.1f}  p90 {pos['p90']:.1f}  "
              f"max {pos['max']:.1f} ({pos['max_case']})")
        for source, stats in pos["by_source"].items():
            print(f"  {source}: n={stats['count']} mean {stats['mean']:.1f}px")
    else:
        print("No position pairs yet.")

    print("\n== Bounce timing ==")
    timing = report["timing"]
    if timing["checked"]:
        confirmed = timing["frame_confirmed"]
        print(f"Detected frame confirmed as the bounce: "
              f"{confirmed}/{timing['checked']}")
        if timing["offsets"]:
            offsets = [o for o, _ in timing["offsets"]]
            mean_abs = sum(abs(o) for o in offsets) / len(offsets)
            print(f"  corrected-frame offsets: mean |Δ| {mean_abs:.1f} frames, "
                  f"range [{min(offsets)}, {max(offsets)}]")
            if verbose:
                for offset, case_id in timing["offsets"]:
                    print(f"    {case_id}: {offset:+d} frames")
    else:
        print("No timing labels yet.")

    print("\n== Missed bounces (labeled, no detected hit at build time) ==")
    missed = report["missed"]
    if missed["checked"]:
        print(f"Missed: {len(missed['missed'])}/{missed['checked']}")
        for hit_type, count in sorted(missed["missed_by_type"].items()):
            print(f"  {hit_type}: {count}")
        if missed["matched_type_checked"]:
            print(f"Matched-pair type agreement: "
                  f"{missed['matched_type_correct']}"
                  f"/{missed['matched_type_checked']}")
        if verbose:
            for case_id, hit_type in missed["missed"]:
                print(f"  {case_id} ({hit_type})")
    else:
        print("No ground-truth events swept yet - label some in the UI.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-set", type=Path,
                        default=Path(__file__).with_name("eval_set") / "cases.jsonl")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-under", type=float, default=None, metavar="PCT",
                        help="exit 1 if IN/OUT accuracy falls below this percentage")
    args = parser.parse_args()

    if not args.eval_set.exists():
        sys.exit(f"No eval set at {args.eval_set} - run build_eval_set.py first.")

    report = evaluate_cases(load_eval_set(args.eval_set))
    print_report(report, verbose=args.verbose)

    if args.fail_under is not None and report["accuracy"] is not None:
        if report["accuracy"] * 100 < args.fail_under:
            sys.exit(f"FAIL: accuracy {report['accuracy'] * 100:.1f}% "
                     f"< required {args.fail_under:.1f}%")


if __name__ == "__main__":
    main()
