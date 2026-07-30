"""The Training hub's Coaching hero is the entry to the coaching advice.

Everything else on that hub is a roadmap placeholder, so if this hero stops
being a button the page has no working destination at all — which is the state
it shipped in before 2026-07-30.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_the_hero_is_a_button_and_not_a_div():
    assert '<button class="hero" type="button" id="coachHeroBtn" disabled>' in INDEX_HTML


def test_the_hero_opens_the_advice_page_not_the_per_run_review():
    # openRunReview lands on the frame-by-frame call page, which is the wrong
    # first thing for a reader who tapped "Coaching".
    assert "$('coachHeroBtn').onclick = () => { if(LIVE.runs.length) setPhase('coach_advice'); };" in INDEX_HTML


def test_the_hero_is_disabled_until_a_run_exists():
    assert "$('coachHeroBtn').disabled = !count" in INDEX_HTML
    assert "'Analyze a clip to get your first drills'" in INDEX_HTML


def test_the_advice_page_is_a_scrollable_leaf_off_the_training_hub():
    assert "coach_advice:{label:'Coaching', instr:''}," in INDEX_HTML
    # PAGE_PHASES drives both the page background and main's own scroller.
    assert "const PAGE_PHASES = [...ROADMAP_PHASES, 'match', 'coach_advice'];" in INDEX_HTML
    assert "} else if(S.phase === 'coach_advice'){" in INDEX_HTML
    assert "setPhase('coach');" in INDEX_HTML


def test_the_advice_page_pools_sessions_and_shows_the_pooling_caveat():
    assert "/api/coach/advice?sessions=${POOLED_SESSIONS}" in INDEX_HTML
    assert "$('adviceCaveat').textContent = data.pooling_note" in INDEX_HTML


def test_the_button_hero_overrides_the_ua_flex_centering():
    # A bare <button class="hero"> shrink-wraps its .herotop row to the center;
    # the row has to span the card like it does in the <div> hero.
    assert "align-items:stretch" in INDEX_HTML


def test_design_md_documents_the_tappable_hero_and_the_advice_page():
    design = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
    assert "**Tappable ink hero.**" in design
    assert "p-coach-advice" in design
