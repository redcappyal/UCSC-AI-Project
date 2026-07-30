from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1] / "index.html"
).read_text(encoding="utf-8")


def test_rally_winner_is_visible_inside_each_rally_segment():
    assert "function rallyWinnerDecision(rallyNumber)" in INDEX_HTML
    assert "playerDisplayName(decision.playerNumber)" in INDEX_HTML
    assert 'class="rallyWinner"' in INDEX_HTML


def test_rally_winner_hover_text_exposes_provenance():
    assert "source: ${decision.source}" in INDEX_HTML
    assert "reason: ${decision.reason}" in INDEX_HTML
    assert "attribution: ${decision.attributionState}" in INDEX_HTML


def test_separate_winner_debug_row_is_not_rendered():
    assert 'id="rallyWinnerDebug"' not in INDEX_HTML
    assert "winnerButton.textContent" not in INDEX_HTML
