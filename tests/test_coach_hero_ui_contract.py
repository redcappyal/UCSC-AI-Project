"""The Training hub exposes one coaching destination: Your coach."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_training_hub_has_no_duplicate_coaching_hero():
    assert 'id="coachHeroBtn"' not in INDEX_HTML
    assert 'id="coachRing"' not in INDEX_HTML


def test_your_coach_card_opens_the_live_coaching_page():
    assert (
        '<button class="featureCard" type="button" '
        'data-phase="coach_advice">' in INDEX_HTML
    )
    assert "<strong>Your coach</strong>" in INDEX_HTML
    assert "<span class=\"fcTag\">Live</span>" in INDEX_HTML


def test_obsolete_sharing_placeholder_is_removed():
    assert 'id="p-sharing"' not in INDEX_HTML
    assert "sharing: {label:'Your coach'" not in INDEX_HTML


def test_shot_selection_placeholder_is_removed():
    """Dropped 2026-07-30. Every hook goes with the card — a phase left in
    ROADMAP_PHASES whose section no longer exists throws on $(id) in setPhase."""
    assert 'data-phase="shot_bot"' not in INDEX_HTML
    assert 'id="p-shot-bot"' not in INDEX_HTML
    assert "shot_bot:{label:" not in INDEX_HTML
    assert "const ROADMAP_PHASES = ['live','matches','coach','progress','stats'];" in INDEX_HTML
    assert "p-shot-bot" not in INDEX_HTML


def test_the_advice_page_is_a_scrollable_leaf_off_the_training_hub():
    assert "coach_advice:{label:'Your coach', instr:''}," in INDEX_HTML
    assert "const PAGE_PHASES = [...ROADMAP_PHASES, 'match', 'coach_advice'];" in INDEX_HTML
    assert "} else if(S.phase === 'coach_advice'){" in INDEX_HTML
    assert "setPhase('coach');" in INDEX_HTML


def test_the_advice_page_uses_ollama_history_output():
    assert "/api/coach/advice?sessions=${POOLED_SESSIONS}" in INDEX_HTML
    assert "const coach = ADVICE.data.coach;" in INDEX_HTML
    assert "'Ollama reviewed those matches from oldest to newest.'," in INDEX_HTML
    assert "no rule-based advice was substituted" in INDEX_HTML
    # The pooling caveat now rides in the §8.23 disclosure on the headline.
    assert "ADVICE.data.me_pooling_note || ''," in INDEX_HTML


def test_design_md_documents_the_single_your_coach_destination():
    design = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
    assert "including the live Your coach entry" in design
    assert "No deterministic coaching copy is substituted" in design
