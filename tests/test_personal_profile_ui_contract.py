"""Analysis identity selection feeds live Training and Progress views."""

from pathlib import Path


INDEX = (Path(__file__).resolve().parents[1] / "index.html").read_text(
    encoding="utf-8"
)


def test_analysis_cards_do_not_show_player_identity_controls():
    assert 'data-self-selector=' not in INDEX
    assert 'data-self-player=' not in INDEX
    assert "Which player are you?" not in INDEX
    assert "Tracking your stats as" not in INDEX


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


def test_progress_uses_latest_available_value_for_each_player_metric():
    assert "function latestPersonalMetricPair(key)" in INDEX
    assert ".filter(value => Number.isFinite(value))" in INDEX
    assert "latestPersonalMetricPair('unforcedPct')" in INDEX
    assert "latestPersonalMetricPair('tPct')" in INDEX
    assert "latestPersonalMetricPair('wide')" in INDEX


def test_changing_identity_invalidates_pooled_training_data():
    assert "ADVICE.data = null" in INDEX
    assert "if(run) run.user_player_number = playerNumber" in INDEX


def test_failed_ollama_coaching_can_be_retried_without_reloading():
    assert 'id="coachRetryP1"' in INDEX
    assert 'id="coachRetryP2"' in INDEX
    assert "function retryOllamaCoachReport(playerNumber)" in INDEX
    assert "llm_status:'pending'" in INDEX
    assert "requestLlmCoachReport(runId)" in INDEX
