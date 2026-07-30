import csv

from label_hits import (
    LABEL_SHORTCUTS,
    load_labels,
    save_labels,
    toggle_label,
)


def test_shortcuts_map_to_canonical_event_types():
    assert LABEL_SHORTCUTS == {
        "h": "wall",
        "f": "floor",
        "p": "side_wall",
    }


def test_typed_labels_round_trip(tmp_path):
    path = tmp_path / "hits.csv"
    labels = {10: "wall", 20: "floor", 30: "side_wall"}

    save_labels(path, labels)

    with path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"hit_frame": "10", "event_type": "wall"},
            {"hit_frame": "20", "event_type": "floor"},
            {"hit_frame": "30", "event_type": "side_wall"},
        ]
    assert load_labels(path) == labels


def test_legacy_one_column_csv_defaults_to_wall(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text("hit_frame\n10\n20\n", encoding="utf-8")

    assert load_labels(path) == {10: "wall", 20: "wall"}


def test_toggle_replaces_a_different_type_and_removes_the_same_type():
    labels = {10: "wall"}

    assert toggle_label(labels, 10, "floor") == "floor"
    assert labels == {10: "floor"}
    assert toggle_label(labels, 10, "floor") is None
    assert labels == {}
