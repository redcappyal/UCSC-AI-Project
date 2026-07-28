"""Score observed serve attribution against human rally labels.

Usage:
    python eval_attribution.py --run-dir ui_runs/<id> \
        --labels eval_set/attribution-labels-SquashAnalytics.json \
        --output eval_set/BASELINE-ATTRIBUTION-<date>.md

The labels file is human-produced (see the template). This script is the
only path to claiming attribution "improved" (spec §5)."""

import argparse
import json
from pathlib import Path


def score_attribution(players_v1, labels):
    observed = {
        r["rally_number"]: r["server_player_number"]
        for r in players_v1.get("rallies", [])
        if r.get("server_source") == "observed"
    }
    labeled = {
        r["rally_number"]: r["server"]
        for r in labels.get("rallies", [])
        if r.get("server") in (1, 2)
    }
    all_labeled = [r for r in labels.get("rallies", [])]
    scored = [n for n in labeled if n in observed]
    correct = sum(1 for n in scored if observed[n] == labeled[n])
    return {
        "labeled_rallies": len(all_labeled),
        "observed_rallies": len(observed),
        "scored_rallies": len(scored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "observed_coverage": (
            len(scored) / len(all_labeled) if all_labeled else None
        ),
        "mismatches": sorted(n for n in scored if observed[n] != labeled[n]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    job = json.loads((args.run_dir / "job.json").read_text(encoding="utf-8"))
    players_v1 = job.get("players_v1") or {}
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    report = score_attribution(players_v1, labels)

    lines = [
        "# Serve-attribution baseline",
        "",
        f"- Run: `{args.run_dir}`",
        f"- Labels: `{args.labels}` ({report['labeled_rallies']} rallies)",
        f"- Detector backend: {players_v1.get('detector_backend')}",
        f"- Observed rallies: {report['observed_rallies']}",
        f"- Scored (observed AND labeled): {report['scored_rallies']}",
        f"- Correct: {report['correct']}",
        f"- Accuracy: {report['accuracy']}",
        f"- Observed coverage of labeled rallies: {report['observed_coverage']}",
        f"- Mismatched rally numbers: {report['mismatches']}",
        "",
        "Rallies without observed serves are excluded from accuracy —",
        "coverage reports them honestly (spec §7: no pre-hit ball track ->",
        "`server_track: null`, never a guess).",
    ]
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
