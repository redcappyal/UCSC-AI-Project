"""prepare_ball_dataset: geometry, splitting and crop planning (no cv2 needed)."""

import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_ball_dataset import (
    _clip_box, ascii_slug, burst_count, clip_and_frame, clip_scale_factors,
    crop_file_name, plan_crops, polygon_aabb, polygon_points, slugified_clips,
    split_by_clip, streak_metrics, thin_bursts,
)


def frame(clip="A", number=0, balls=(), width=1920, height=1080):
    return {"path": Path(f"{clip}_mov-{number:04d}_jpg.jpg"), "clip": clip,
            "frame": number, "width": width, "height": height,
            "balls": [{"bbox": list(b), "streak": None} for b in balls]}


def test_clip_and_frame_splits_roboflow_names():
    assert clip_and_frame("Bay-Club-1_mov-0042_jpg.rf.abc.jpg") == ("Bay-Club-1", 42)
    assert clip_and_frame("SQUASHCLIP_MOV-0158_jpg.rf.d.jpg") == ("SQUASHCLIP", 158)
    # Unparseable names still yield a stable clip so they can't silently
    # scatter across splits.
    assert clip_and_frame("loose.rf.hash.jpg") == ("loose", None)


def test_polygon_aabb_bounds_the_ring():
    points = polygon_points([[10, 20, 30, 20, 30, 50, 10, 50]])
    assert polygon_aabb(points) == (10, 20, 20, 30)


def test_streak_metrics_separates_round_from_streaked():
    round_ball = streak_metrics([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert round_ball["aspect"] == pytest.approx(1.0, abs=0.45)

    streak = streak_metrics([(0, 0), (100, 0), (100, 10), (0, 10)])
    assert streak["aspect"] > 4
    assert streak["length"] == pytest.approx(100.5, abs=1.0)
    # Endpoints are the exposure path, which is what judge_call.py wants. Both
    # diagonals of this rectangle are equally the major axis, so assert the
    # span rather than which one was picked.
    (ax, ay), (bx, by) = streak["endpoints"]
    assert {ax, bx} == {0, 100} and {ay, by} == {0, 10}


def test_streak_metrics_needs_two_points():
    assert streak_metrics([(1, 1)]) is None


def test_burst_count_collapses_consecutive_runs():
    # 20 consecutive frames are one moment, not twenty.
    run = [frame(number=n) for n in range(20)]
    assert burst_count(run) == 1
    spread = [frame(number=n * 30) for n in range(20)]
    assert burst_count(spread) == 20


def test_thin_bursts_keeps_one_frame_per_gap():
    kept, dropped = thin_bursts([frame(number=n) for n in range(10)], min_gap=5)
    assert [r["frame"] for r in kept] == [0, 5]
    assert dropped == 8


def test_thin_bursts_is_a_noop_at_gap_one():
    records = [frame(number=n) for n in range(4)]
    kept, dropped = thin_bursts(records, min_gap=1)
    assert kept == records and dropped == 0


def test_thin_bursts_counts_per_clip():
    records = [frame(clip="A", number=0), frame(clip="B", number=1),
               frame(clip="A", number=1)]
    kept, _ = thin_bursts(records, min_gap=5)
    # B's frame 1 is its own clip's first, so the gap is not shared with A.
    assert {(r["clip"], r["frame"]) for r in kept} == {("A", 0), ("B", 1)}


def test_split_by_clip_never_lets_a_clip_span_splits():
    records = [frame(clip="A"), frame(clip="B"), frame(clip="C")]
    splits = split_by_clip(records, val_clips=["B"], test_clips=["C"])
    assert [r["clip"] for r in splits["train"]] == ["A"]
    assert [r["clip"] for r in splits["val"]] == ["B"]
    assert [r["clip"] for r in splits["test"]] == ["C"]


def test_split_by_clip_rejects_a_clip_in_two_splits():
    with pytest.raises(SystemExit):
        split_by_clip([frame(clip="A")], val_clips=["A"], test_clips=["A"])


def test_clip_scale_factors_normalise_median_ball_and_clamp():
    records = [frame(clip="A", balls=[(0, 0, 10, 10)]),
               frame(clip="A", balls=[(0, 0, 20, 20)])]
    assert clip_scale_factors(records, target_ball_px=30)["A"] == pytest.approx(2.0)
    # A clip needing 100x to reach the target is out of domain, not rescalable.
    tiny = [frame(clip="B", balls=[(0, 0, 1, 1)])]
    assert clip_scale_factors(tiny, target_ball_px=100)["B"] == 4.0
    assert clip_scale_factors(records, target_ball_px=0) == {}


def test_clip_box_drops_a_ball_mostly_outside_the_window():
    # Fully inside.
    assert _clip_box([10, 10, 20, 20], 0, 0, 416, 0.6) == [10, 10, 20, 20]
    # Only a sliver survives -> dropped rather than taught as a whole ball.
    assert _clip_box([-18, 10, 20, 20], 0, 0, 416, 0.6) is None
    # Entirely outside.
    assert _clip_box([500, 500, 20, 20], 0, 0, 416, 0.6) is None


def test_plan_crops_keeps_the_ball_inside_every_positive():
    record = frame(balls=[(900, 500, 12, 12)])
    plans = plan_crops(record, crop=416, scale=1.0, rng=random.Random(0),
                       positives=8, negatives=0, jitter=0.8, min_visible=0.6)
    assert len(plans) == 8
    for plan in plans:
        assert plan["boxes"], "a positive crop must contain its ball"
        for box in plan["boxes"]:
            x, y, w, h = box["bbox"]
            assert 0 <= x and 0 <= y and x + w <= 416 and y + h <= 416


def test_plan_crops_jitter_moves_the_ball_off_centre():
    record = frame(balls=[(900, 500, 12, 12)])
    plans = plan_crops(record, crop=416, scale=1.0, rng=random.Random(1),
                       positives=12, negatives=0, jitter=0.8, min_visible=0.6)
    centres = {round(plan["boxes"][0]["bbox"][0], 1) for plan in plans}
    assert len(centres) > 1, "a centred ball teaches the head to predict the centre"


def test_plan_crops_zero_jitter_centres_every_crop():
    record = frame(balls=[(900, 500, 12, 12)])
    plans = plan_crops(record, crop=416, scale=1.0, rng=random.Random(2),
                       positives=3, negatives=0, jitter=0.0, min_visible=0.6)
    assert len({tuple(p["origin"]) for p in plans}) == 1


def test_plan_crops_negatives_contain_no_ball():
    record = frame(balls=[(900, 500, 12, 12)])
    plans = plan_crops(record, crop=416, scale=1.0, rng=random.Random(3),
                       positives=1, negatives=20, jitter=0.8, min_visible=0.6)
    negatives = [p for p in plans if not p["boxes"]]
    assert negatives, "hard negatives are where wall ball-marks come from"
    for plan in negatives:
        ox, oy = plan["origin"]
        assert _clip_box([900, 500, 12, 12], ox, oy, 416, 0.6) is None


def test_plan_crops_skips_frames_smaller_than_the_window():
    record = frame(balls=[(10, 10, 5, 5)], width=480, height=272)
    assert plan_crops(record, crop=416, scale=1.0, rng=random.Random(0),
                      positives=4, negatives=1, jitter=0.8, min_visible=0.6) == []


def test_plan_crops_scale_rescales_boxes_with_the_frame():
    record = frame(balls=[(900, 500, 10, 10)])
    plans = plan_crops(record, crop=416, scale=2.0, rng=random.Random(0),
                       positives=4, negatives=0, jitter=0.0, min_visible=0.6)
    for plan in plans:
        width = plan["boxes"][0]["bbox"][2]
        assert width == pytest.approx(20.0)


def test_plan_crops_is_deterministic_for_a_seed():
    record = frame(balls=[(900, 500, 12, 12)])
    args = dict(crop=416, scale=1.0, positives=5, negatives=3,
                jitter=0.8, min_visible=0.6)
    first = plan_crops(record, rng=random.Random(7), **args)
    second = plan_crops(record, rng=random.Random(7), **args)
    assert [p["origin"] for p in first] == [p["origin"] for p in second]


SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def test_ascii_slug_leaves_a_clean_roboflow_name_alone():
    # The 1,975 already-good crops must not be renamed by this change.
    name = "Bay-Club-1_mov-0042_jpg.rf.abc123"
    assert ascii_slug(name) == name


def test_ascii_slug_digests_a_name_the_charset_filter_ignores_but_strip_shortens():
    # "abc." and "abc" are the same file on Windows (trailing dots are
    # silently discarded there), and ".hidden" is a hidden file on Unix, so
    # strip() must still run on names the charset filter already accepts -
    # and because that strip is lossy, the digest must stay too, or the
    # stripped name would collide with the bare one below.
    for stem, bare in [("abc.", "abc"), (".hidden", "hidden")]:
        slug = ascii_slug(stem)
        assert slug != stem
        assert slug.startswith(f"{bare}-")
        assert len(slug) == len(bare) + 1 + 8
        assert slug != ascii_slug(bare)


def test_ascii_slug_replaces_the_fullwidth_bar_and_marks_the_change():
    # The production case: a YouTube title whose "|" was sanitised to U+FF5C.
    assert (ascii_slug("Squash Rally ｜ Best_mov-9_jpg.rf.d")
            == "Squash_Rally_Best_mov-9_jpg.rf.d-cc74d589")


def test_ascii_slug_transliterates_accents_rather_than_dropping_letters():
    # NFKD first: "café" is still recognisably café, not "caf".
    assert ascii_slug("café").startswith("cafe-")


def test_ascii_slug_falls_back_when_no_character_survives():
    # A wholly non-Latin title must still produce a usable, unique filename.
    slug = ascii_slug("スカッシュ")
    assert slug.startswith("clip-")
    assert len(slug) == len("clip-") + 8


def test_ascii_slug_handles_the_cp437_mojibake_form():
    # What the U+FF5C name became on disk after the bad unzip; recovering a
    # half-corrupted dataset must not trip over it either.
    assert SAFE_NAME.fullmatch(ascii_slug("clip∩╜£name_mov-1_jpg.rf.a"))


def test_ascii_slug_keeps_colliding_names_distinct():
    # Two different titles collapse onto one base; the digest is the only thing
    # stopping their crops from overwriting each other.
    bar, question = ascii_slug("Rally ｜ One"), ascii_slug("Rally ? One")
    assert bar.startswith("Rally_One-") and question.startswith("Rally_One-")
    assert bar != question


def test_ascii_slug_output_is_always_filename_safe():
    for name in ["Squash ｜ Rally", 'a/b\\c:d*e?f"g<h>i|j', "  ", "..",
                 "スカッシュ", "Bay-Club-1_mov-0042_jpg.rf.abc123"]:
        assert SAFE_NAME.fullmatch(ascii_slug(name)), name


def test_ascii_slug_is_deterministic():
    # Regenerating the dataset must not reshuffle filenames.
    assert ascii_slug("Squash ｜ Rally") == ascii_slug("Squash ｜ Rally")


def test_crop_file_name_indexes_crops_within_a_slugged_stem():
    assert (crop_file_name("Bay-Club-1_mov-0042_jpg.rf.abc", 3)
            == "Bay-Club-1_mov-0042_jpg.rf.abc_c3.jpg")


def test_crop_file_name_is_safe_even_when_the_stem_is_not():
    name = crop_file_name("Rally ｜ One_jpg.rf.d", 0)
    assert name == f"{ascii_slug('Rally ｜ One_jpg.rf.d')}_c0.jpg"
    assert SAFE_NAME.fullmatch(name)


def test_slugified_clips_lists_only_the_offenders():
    # The manifest entry exists so a digest suffix in a filename explains itself
    # without opening the COCO json.
    records = [frame(clip="Bay-Club-1"), frame(clip="Rally ｜ One"),
               frame(clip="Bay-Club-1")]
    assert slugified_clips(records) == ["Rally ｜ One"]
