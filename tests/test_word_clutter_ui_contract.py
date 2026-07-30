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
