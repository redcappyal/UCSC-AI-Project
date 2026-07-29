from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_judging_stage_contains_transparent_front_wall_impact_map():
    assert 'id="impactMiniMap"' in INDEX_HTML
    assert 'viewBox="0 0 21 15"' in INDEX_HTML
    assert "rgba(8,9,11,.2)" in INDEX_HTML
    assert 'id="impactMiniMarkers"' in INDEX_HTML


def test_impact_map_only_uses_front_wall_hits_with_diagram_coordinates():
    assert "hit.event_type == null" in INDEX_HTML
    assert "hit.event_type === 'wall'" in INDEX_HTML
    assert "hit.event_type === 'unknown'" in INDEX_HTML
    assert "hit.wall_diagram" in INDEX_HTML
    assert "Number.isFinite(Number(hit.wall_diagram.x))" in INDEX_HTML
    assert "Number.isFinite(Number(hit.wall_diagram.y))" in INDEX_HTML
    assert "Number(hit.wall_diagram.x) >= 0" in INDEX_HTML
    assert "Number(hit.wall_diagram.x) <= 1" in INDEX_HTML
    assert "Number(hit.wall_diagram.y) >= 0" in INDEX_HTML
    assert "Number(hit.wall_diagram.y) <= 1" in INDEX_HTML
    assert "Math.min(1, Math.max(0, Number(hit.wall_diagram.x)))" not in INDEX_HTML


def test_impact_markers_follow_playhead_and_highlight_exact_hit():
    assert "Number(hit.frame) <= playhead" in INDEX_HTML
    assert "Number(hit.frame) === playhead" in INDEX_HTML
    assert "renderImpactMiniMap(frame);" in INDEX_HTML


def test_impact_map_resets_to_the_current_rally():
    assert "Number.isFinite(Number(hit.rally_number))" in INDEX_HTML
    assert "Number(hit.rally_number) === currentRallyNumber" in INDEX_HTML
    assert "every marker from the previous rally disappears" in INDEX_HTML
    assert "for rally ${currentRallyNumber}" in INDEX_HTML


def test_impact_map_is_positioned_inside_the_displayed_video():
    assert "const canvasRect = canvas.getBoundingClientRect();" in INDEX_HTML
    assert "canvasRect.top - stageRect.top + inset" in INDEX_HTML
    assert "stageRect.right - canvasRect.right + inset" in INDEX_HTML
