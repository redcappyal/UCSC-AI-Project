"""Contract tests for the 2026-07-30 word-clutter pass.

The UI is one file of inline markup, so these read index.html as text. They
exist to stop the cut copy creeping back in: every assertion here is a line
someone deliberately removed, or a component that replaced one.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")
DESIGN_MD = (ROOT / "DESIGN.md").read_text(encoding="utf-8")


def test_the_disclosure_builder_escapes_server_text():
    # Detector names and capability reasons reach this builder from the
    # server, so it must escape rather than trusting its caller.
    assert "function whyDisclosure(text){" in INDEX_HTML
    assert "escapeHtml(text)" in INDEX_HTML


def test_the_disclosure_starts_collapsed_and_is_announced():
    assert 'class="whybtn"' in INDEX_HTML
    assert 'aria-expanded="false"' in INDEX_HTML
    assert 'aria-controls=' in INDEX_HTML
    assert 'aria-label="Why this number"' in INDEX_HTML
    assert 'class="whynote hidden"' in INDEX_HTML


def test_the_toggle_is_delegated_because_reports_rebuild_their_markup():
    # Every .whybtn is rendered into an innerHTML sink that is replaced on
    # each report load, so a per-button listener would be dropped.
    assert "e.target.closest('.whybtn')" in INDEX_HTML


def test_design_md_documents_the_disclosure():
    assert "### 8.23 Provenance disclosure" in DESIGN_MD
    assert ".whybtn" in DESIGN_MD
    assert ".whynote" in DESIGN_MD


def test_section_roots_do_not_repeat_the_header_title():
    # #stepLabel already names every one of these screens; the <h2> under it
    # said the same word ~50px lower.
    assert "<h2>Analysis</h2>" not in INDEX_HTML
    assert '<div class="viewhead"><h2>Training</h2></div>' not in INDEX_HTML
    assert '<div class="viewhead"><h2>Progress</h2></div>' not in INDEX_HTML
    assert '<h2 id="matchTitle">Match</h2>' not in INDEX_HTML
    assert "<h2>Your stats</h2>" not in INDEX_HTML


def test_the_header_label_is_the_pages_one_heading():
    assert '<h1 id="stepLabel">' in INDEX_HTML
    assert '<div id="stepLabel">' not in INDEX_HTML


def test_the_stats_screen_has_exactly_one_name():
    # bf15df4 made p-stats a live page titled "Your stats" but left the header
    # label as "Stats + trends", so the two chrome layers disagreed.
    assert "stats:   {label:'Your stats', instr:''}," in INDEX_HTML
    assert "Stats + trends" not in INDEX_HTML


def test_the_meta_spans_survive_because_they_carry_new_information():
    assert 'id="clipMeta"' in INDEX_HTML
    assert 'id="matchMeta"' in INDEX_HTML
    assert 'id="trainingStatsMeta"' in INDEX_HTML


def test_the_hero_does_not_repeat_the_coach_notes_below_it():
    # renderCoachNotes renders the same sentences from the same string.
    assert "if(note) $('heroNote').textContent = note;" not in INDEX_HTML
    assert "$('heroNote').classList.toggle('hidden', true);" in INDEX_HTML


def test_the_hero_note_survives_as_the_empty_state():
    # With no runs it is the only thing on the page that says what to do.
    assert 'id="heroNote"' in INDEX_HTML
    assert "the pipeline turns it into" not in INDEX_HTML


def test_the_analysis_list_counts_sessions_not_pipeline_runs():
    assert "runs · live pipeline" not in INDEX_HTML
    assert "session${LIVE.runs.length===1?'':'s'}`" in INDEX_HTML


def test_the_run_id_is_dev_only():
    # A 13-digit epoch id is for whoever is sitting at the Mac.
    assert '<div class="metaline devOnly">run ${escapeHtml(id)}</div>' in INDEX_HTML


def test_roadmap_cards_do_not_expose_internal_phase_numbers():
    # "Phase 5" went when main made "Your coach" Live; Phase 6 is the last one.
    assert "Phase 5" not in INDEX_HTML
    assert "Phase 6" not in INDEX_HTML
    assert '<span class="fcTag">Soon</span>' in INDEX_HTML


def test_the_stats_card_subtitle_fits_on_one_line():
    # "Your shots and patterns across identified matches" truncated mid-word
    # at 375px, which loses information rather than saving space.
    assert "Your shots and patterns across identified matches" not in INDEX_HTML
    assert "<strong>Your stats</strong><span>Shots and patterns</span>" in INDEX_HTML


def test_the_capability_card_is_titled_for_what_it_reports():
    assert "What this clip could measure" not in INDEX_HTML
    assert "Not measured" in INDEX_HTML


def test_the_capability_card_lists_only_the_tiers_that_did_not_run():
    assert "Object.keys(TIER_LABELS).filter(k => caps[k] && !caps[k].enabled)" in INDEX_HTML


def test_the_capability_card_vanishes_when_everything_ran():
    # Its whole purpose is explaining absence; with none there is nothing to say.
    assert "const capRows = capabilityRows(rep);" in INDEX_HTML
    assert "capRows ? `<div class=\"cardtitle\" style=\"font-size:15px\">Not measured</div>${capRows}` : ''" in INDEX_HTML


def test_capability_reasons_stay_visible_body_text():
    # They must never move behind the §8.23 disclosure.
    assert 'escapeHtml(t.reason||\'\')' in INDEX_HTML
    assert "whyDisclosure(t.reason" not in INDEX_HTML


def test_the_rally_and_movement_provenance_moved_behind_the_tap():
    # NB: assert on the metaline WRAPPER, not on the sentence. The rally line is
    # built as `<div class="metaline">${escapeHtml(tl.audio_available ? '...`,
    # so asserting `'<div class="metaline">From impact' not in ...` would pass
    # vacuously — that substring never existed.
    assert 'metaline">${escapeHtml(tl.audio_available' not in INDEX_HTML
    assert '<div class="metaline">Detector: ' not in INDEX_HTML
    assert "sectionHead('Rallies', " in INDEX_HTML
    assert "sectionHead('Movement', " in INDEX_HTML


def test_the_rally_provenance_drops_the_pipeline_jargon():
    assert "hit-derived rallies" not in INDEX_HTML
    assert "disagrees with the rallies counted from ball contacts" in INDEX_HTML


def test_the_heatmap_keeps_its_legend_and_hides_only_the_detector():
    assert "stronger color means more time" in INDEX_HTML
    assert "stronger color means more time. Detector:" not in INDEX_HTML


def test_the_coaching_screen_collapses_its_three_caveats_into_one():
    assert 'id="adviceMeta"' not in INDEX_HTML
    assert 'id="adviceCaveat"' not in INDEX_HTML
    assert 'id="adviceWhy"' in INDEX_HTML


def test_the_coaching_disclosure_only_exists_in_the_loaded_state():
    # renderAdvice also has error / loading / no-identified-player states,
    # none of which have provenance to offer.
    assert "$('adviceWhy').innerHTML = '';" in INDEX_HTML


def test_the_low_sample_note_is_provenance_not_body_copy():
    advice = (ROOT / "coaching_advice.py").read_text(encoding="utf-8")
    assert "treat this as a pointer" not in advice
    assert "low_sample_note" in advice
