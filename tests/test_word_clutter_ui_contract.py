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
