"""eval_attribution: scoring deterministic servers against human labels."""

from eval_attribution import render_report, score_attribution


def test_score_attribution_counts_matches_and_coverage():
    players_v1 = {"rallies": [
        {"rally_number": 1, "server_player_number": 1, "server_source": "observed"},
        {"rally_number": 2, "server_player_number": 2, "server_source": "observed"},
        {"rally_number": 3, "server_player_number": 1, "server_source": "propagated"},
    ]}
    labels = {"rallies": [
        {"rally_number": 1, "server": 1},
        {"rally_number": 2, "server": 1},
        {"rally_number": 3, "server": 1},
        {"rally_number": 4, "server": 2},
    ]}
    report = score_attribution(players_v1, labels)
    assert report["labeled_rallies"] == 4
    assert report["assigned_rallies"] == 3
    assert report["scored_rallies"] == 3       # assigned AND labeled
    assert report["correct"] == 2              # rallies 1 and 3 right
    assert report["accuracy"] == 2 / 3
    assert report["assigned_coverage"] == 0.75


def test_score_attribution_skips_null_labels():
    players_v1 = {"rallies": [
        {"rally_number": 1, "server_player_number": 1, "server_source": "observed"},
    ]}
    labels = {"rallies": [{"rally_number": 1, "server": None}]}
    report = score_attribution(players_v1, labels)
    assert report["scored_rallies"] == 0
    assert report["accuracy"] is None


def test_render_report_banners_template_labels_above_the_numbers():
    """A template labels file scores by construction (it's checked in with
    one placeholder rally, so 'accuracy' is trivially 1.0 for anyone who
    hasn't done the human labeling pass) -- eval_set/BASELINE-ATTRIBUTION-
    2026-07-27.md documents exactly that trap. The report has to say so
    loudly, above the numbers, or the numbers alone get treated as a real
    accuracy claim."""
    players_v1 = {"detector_backend": "rfdetr", "rallies": [
        {"rally_number": 1, "server_player_number": 1, "server_source": "observed"},
    ]}
    labels = {
        "note": "HUMAN GATE: watch the clip, fill server per rally, save without .template",
        "rallies": [{"rally_number": 1, "server": 1}],
    }
    report = score_attribution(players_v1, labels)

    # Trigger 1: ".template." in the labels filename.
    by_filename = render_report(
        "ui_runs/some-run",
        "eval_set/attribution-labels-x.template.json",
        players_v1, labels, report,
    )
    assert "TEMPLATE LABELS — NOT A REAL ACCURACY" in by_filename
    assert by_filename.index("TEMPLATE LABELS") < by_filename.index("- Run:")

    # Trigger 2: a renamed copy (no ".template." in the filename) that kept
    # the "HUMAN GATE" note must still be caught.
    by_note = render_report(
        "ui_runs/some-run",
        "eval_set/attribution-labels-x.json",
        players_v1, labels, report,
    )
    assert "TEMPLATE LABELS — NOT A REAL ACCURACY" in by_note

    # A real, human-verified labels file (neither trigger) must not banner.
    real_labels = {**labels, "note": "reviewed against the clip by Ian, 2026-07-27"}
    real = render_report(
        "ui_runs/some-run",
        "eval_set/attribution-labels-x.json",
        players_v1, real_labels, report,
    )
    assert "TEMPLATE LABELS" not in real
    # Existing behavior (the numbers themselves) is unchanged either way.
    assert "- Accuracy: 1.0" in real
    assert "- Accuracy: 1.0" in by_filename
