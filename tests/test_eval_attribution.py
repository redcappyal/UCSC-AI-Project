"""eval_attribution: scoring observed servers against human labels."""

from eval_attribution import score_attribution


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
    assert report["observed_rallies"] == 2
    assert report["scored_rallies"] == 2       # observed AND labeled
    assert report["correct"] == 1              # rally 1 right, rally 2 wrong
    assert report["accuracy"] == 0.5
    assert report["observed_coverage"] == 0.5  # 2 observed of 4 labeled


def test_score_attribution_skips_null_labels():
    players_v1 = {"rallies": [
        {"rally_number": 1, "server_player_number": 1, "server_source": "observed"},
    ]}
    labels = {"rallies": [{"rally_number": 1, "server": None}]}
    report = score_attribution(players_v1, labels)
    assert report["scored_rallies"] == 0
    assert report["accuracy"] is None
