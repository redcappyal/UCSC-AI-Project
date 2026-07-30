"""Analysis identity selection feeds live Training and Progress views."""

from pathlib import Path


INDEX = (Path(__file__).resolve().parents[1] / "index.html").read_text(
    encoding="utf-8"
)


def test_analysis_cards_let_the_user_choose_their_player():
    assert 'data-self-player="1"' in INDEX
    assert 'data-self-player="2"' in INDEX
    assert "Which player are you?" in INDEX
    assert "/api/runs/${encodeURIComponent(runId)}/me" in INDEX
    assert "body:JSON.stringify({player_number:playerNumber})" in INDEX


def test_judging_players_panel_lets_the_user_choose_themselves():
    players_start = INDEX.index('<div class="targetZones hidden" id="playersCard">')
    players_end = INDEX.index("</div>\n      <button", players_start)
    section = INDEX[players_start:players_end]
    assert 'id="playerMeA"' in section
    assert 'id="playerMeB"' in section
    assert section.count("This is me") == 2
    assert "Choose yourself for personal stats" in section
    assert "await persistRunPlayerSelection(S.run.run_id, playerNumber)" in INDEX
    assert "S.run.user_player_number = playerNumber" in INDEX


def test_training_stats_are_live_not_a_coming_soon_placeholder():
    start = INDEX.index('<section class="hidden" id="p-stats">')
    end = INDEX.index("</section>", start)
    section = INDEX[start:end]
    assert 'id="trainingStatsBody"' in section
    assert "Coming soon!" not in section
    assert "function buildTrainingStatsView()" in INDEX
    assert "function renderTrainingStatsView()" in INDEX


def test_progress_uses_the_selected_player_from_each_run():
    assert "function personalMetrics(id)" in INDEX
    assert "const personalRuns = selectedRuns()" in INDEX
    assert "get:id=>(personalMetrics(id)||{}).unforcedPct" in INDEX
    assert "Progress only compares the player you identify as yourself." in INDEX


def test_changing_identity_invalidates_pooled_training_data():
    assert "ADVICE.data = null" in INDEX
    assert "if(run) run.user_player_number = playerNumber" in INDEX
