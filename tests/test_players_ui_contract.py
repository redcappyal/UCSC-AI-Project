"""Static contract for direct crop-to-name player identification."""

from pathlib import Path


INDEX = (Path(__file__).resolve().parents[1] / "index.html").read_text(
    encoding="utf-8"
)


def test_each_player_crop_is_paired_with_its_own_name_input():
    for track in ("A", "B"):
        assert f'id="playerCrop{track}"' in INDEX
        assert f'id="playerName{track}"' in INDEX
        crop_position = INDEX.index(f'id="playerCrop{track}"')
        name_position = INDEX.index(f'id="playerName{track}"')
        assert crop_position < name_position


def test_indirect_who_is_this_controls_are_removed():
    assert 'id="cropIsA"' not in INDEX
    assert 'id="cropIsB"' not in INDEX
    assert "Who is this?" not in INDEX
    assert "v1.player_crops" in INDEX


def test_player_mapping_is_direct_and_explains_a_first_rule():
    assert "Number(playerNumber) === 1 ? 'A' : 'B'" in INDEX
    assert "Player A is assumed to serve first" in INDEX
    assert "later servers follow the previous rally winner" in INDEX
