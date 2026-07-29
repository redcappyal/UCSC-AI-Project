"""Score deterministic player attribution against human rally labels.

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
    assigned = {
        r["rally_number"]: r["server_player_number"]
        for r in players_v1.get("rallies", [])
        if r.get("server_player_number") in (1, 2)
    }
    labeled = {
        r["rally_number"]: r["server"]
        for r in labels.get("rallies", [])
        if r.get("server") in (1, 2)
    }
    all_labeled = [r for r in labels.get("rallies", [])]
    scored = [n for n in labeled if n in assigned]
    correct = sum(1 for n in scored if assigned[n] == labeled[n])
    return {
        "labeled_rallies": len(all_labeled),
        "assigned_rallies": len(assigned),
        "scored_rallies": len(scored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "assigned_coverage": (
            len(scored) / len(all_labeled) if all_labeled else None
        ),
        "mismatches": sorted(n for n in scored if assigned[n] != labeled[n]),
    }


TEMPLATE_BANNER = (
    "> ## TEMPLATE LABELS — NOT A REAL ACCURACY\n"
    ">\n"
    "> This run was scored against a template / human-gate-pending labels\n"
    "> file, not human-verified rally labels. The numbers below only show\n"
    "> what the pipeline produced against placeholder values — they are not\n"
    "> an accuracy claim (spec §5: `eval_attribution.py` is the only path to\n"
    "> claiming attribution improved, and a template run doesn't qualify)."
)


def is_template_labels(labels_path, labels):
    """True when the labels file is the unverified template: either by
    filename convention (the human-gate step is copying it off
    `.template.json`, spec §5) or by the "HUMAN GATE" note the template
    itself carries, so a copy that kept the note without renaming is still
    caught."""
    if ".template." in Path(labels_path).name:
        return True
    return str(labels.get("note", "")).startswith("HUMAN GATE")


def render_report(run_dir, labels_path, players_v1, labels, report):
    lines = ["# Serve-attribution baseline", ""]
    if is_template_labels(labels_path, labels):
        lines += [TEMPLATE_BANNER, ""]
    lines += [
        f"- Run: `{run_dir}`",
        f"- Labels: `{labels_path}` ({report['labeled_rallies']} rallies)",
        f"- Detector backend: {players_v1.get('detector_backend')}",
        f"- Assigned rallies: {report['assigned_rallies']}",
        f"- Scored (assigned AND labeled): {report['scored_rallies']}",
        f"- Correct: {report['correct']}",
        f"- Accuracy: {report['accuracy']}",
        f"- Assigned coverage of labeled rallies: {report['assigned_coverage']}",
        f"- Mismatched rally numbers: {report['mismatches']}",
        "",
        "The production rule assumes Player A serves rally 1, alternates",
        "front-wall contacts, and makes each inferred rally winner the next",
        "server. Rallies without an assigned server are excluded from accuracy.",
    ]
    return "\n".join(lines) + "\n"


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

    output = render_report(args.run_dir, args.labels, players_v1, labels, report)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
