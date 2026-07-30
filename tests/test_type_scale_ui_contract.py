"""The type ladder, and the one thing that must not move with it.

Lettering inside the front-wall diagram is sized to the diagram, which has a
locked aspect ratio — scaling it with the app's type ladder clips the zone
percentages and slides the OUT LINE / SERVICE LINE chips over the zone numbers.
Raising the ladder on 2026-07-30 did exactly that before it was caught.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")
STYLE = INDEX_HTML.split("<style>", 1)[1].split("</style>", 1)[0]


def rule(selector):
    """The declaration block for one CSS rule in index.html's <style>."""
    match = re.search(re.escape(selector) + r"\{([^}]*)\}", STYLE)
    assert match, f"{selector} is no longer a rule in index.html"
    return match.group(1)


def font_size(selector):
    match = re.search(r"font-size:(\d+)px", rule(selector))
    assert match, f"{selector} declares no font-size"
    return int(match.group(1))


def test_the_court_diagram_lettering_stays_off_the_ladder():
    assert font_size(".targetPct") == 26
    assert font_size(".targetZoneNum") == 13
    assert font_size(".courtText") == 12


def test_the_body_base_matches_the_documented_ladder():
    assert font_size("body") == 17


def test_prose_sits_at_the_ladder_floor_or_above():
    """13 px sub-meta is the smallest running copy. The rules left below it are
    lettering wedged into a width-proportional rally segment or a 12 px legend
    square, which are sized by the space they sit in, not by readability."""
    assert font_size(".metaline") == 13
    assert font_size(".adviceMeta") == 14
    assert font_size(".drillSteps li") == 14


def test_the_verdict_box_still_reserves_its_height():
    assert "min-height:86px" in rule(".verdict")


def test_design_md_records_the_ladder_and_its_exemption():
    design = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
    assert "**Base:** 17 px / 1.4 line-height on `body`." in design
    assert "Exempt: lettering inside the court diagram" in design
