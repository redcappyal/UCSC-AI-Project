"""prepare_ball_dataset: geometry, splitting and crop planning (no cv2 needed)."""

import json
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_ball_dataset
from prepare_ball_dataset import (
    _clip_box, _imread_unicode, _imwrite_unicode, ascii_slug, burst_count,
    clip_and_frame, clip_scale_factors, crop_file_name, plan_crops,
    polygon_aabb, polygon_points, render_split, slugified_clips,
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


def test_slugified_clips_tracks_stem_divergence_not_clip_divergence():
    # A prior implementation slugged r["clip"] directly, but crop filenames
    # are slugged from r["path"].stem. "abc." looks unsafe as a bare clip
    # name (a trailing dot is stripped), but once it is embedded in
    # "abc._mov-0000_jpg" the stem is already in canonical form and its crops
    # carry no digest at all - the old implementation listed it anyway.
    records = [frame(clip="abc.", number=0)]
    assert slugified_clips(records) == []


def test_unicode_path_survives_a_write_read_round_trip(tmp_path):
    # cv2.imwrite returns True while writing a mojibake filename on Windows, so
    # the COCO json ends up naming files that are not there. This is the guard.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    target = tmp_path / "Rally ｜ One_c0.jpg"
    _imwrite_unicode(target, np.full((16, 16, 3), 128, dtype=np.uint8),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # The mojibake sibling is what this assertion is really looking for.
    assert [p.name for p in tmp_path.iterdir()] == [target.name]
    assert _imread_unicode(target).shape == (16, 16, 3)


def test_imread_unicode_returns_none_for_a_missing_file(tmp_path):
    pytest.importorskip("cv2")
    # render_split skips frames it cannot read; that contract has to survive.
    assert _imread_unicode(tmp_path / "absent.jpg") is None


def test_render_split_raises_on_a_filename_collision_across_source_dirs(tmp_path):
    # crop_file_name keys off path.stem, dropping the parent directory. Two
    # source records with the same file name in different export split dirs
    # (as Roboflow can emit) therefore plan to the same crop filename; the
    # second write would silently clobber the first's pixels while the COCO
    # json still ends up with two distinct `images` entries.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    stem = "Bay-Club-1_mov-0001_jpg"
    train_dir, valid_dir = tmp_path / "train", tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    path_a, path_b = train_dir / f"{stem}.jpg", valid_dir / f"{stem}.jpg"
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    for path in (path_a, path_b):
        _imwrite_unicode(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def record(path):
        return {"path": path, "clip": "Bay-Club-1", "frame": 1,
                "width": 32, "height": 32,
                "balls": [{"bbox": [8, 8, 4, 4], "streak": None}]}

    record_a, record_b = record(path_a), record(path_b)
    args = dict(crop=16, scale=1.0, positives=1, negatives=0,
                jitter=0.0, min_visible=0.6)
    plans_by_record = {
        path_a: plan_crops(record_a, rng=random.Random(0), **args),
        path_b: plan_crops(record_b, rng=random.Random(0), **args),
    }

    with pytest.raises(SystemExit):
        render_split([record_a, record_b], plans_by_record, tmp_path, "train", 16, 95)


# --- Sequence crops (--seq-frames > 1) ---------------------------------

def _solid_frame(np_, value, size=64):
    return np_.full((size, size, 3), min(max(value, 0), 255), dtype="uint8")


def test_seq_frames_requires_clips_dir(tmp_path):
    with pytest.raises(SystemExit, match="clips-dir"):
        prepare_ball_dataset.build(
            source=tmp_path / "missing-source", out=tmp_path / "out", crop=416,
            positives=1, negatives=0, jitter=0.3, min_visible=0.5,
            min_source_width=0, min_frame_gap=1, target_ball_px=0,
            val_clips=[], test_clips=[], seed=0, quality=95,
            seq_frames=3, clips_dir=None)


def test_seq_frames_must_be_odd(tmp_path):
    with pytest.raises(SystemExit, match="odd"):
        prepare_ball_dataset.build(
            source=tmp_path / "missing-source", out=tmp_path / "out", crop=416,
            positives=1, negatives=0, jitter=0.3, min_visible=0.5,
            min_source_width=0, min_frame_gap=1, target_ball_px=0,
            val_clips=[], test_clips=[], seed=0, quality=95,
            seq_frames=2, clips_dir=tmp_path)


def test_find_clip_videos_matches_by_ascii_slug(tmp_path):
    # record["clip"] is the readable name load_export produces (spaces and
    # all), never a pre-slugged value -- matching has to slug both sides.
    (tmp_path / "Bay Club Rally 1.mp4").write_bytes(b"")
    records = [{"clip": "Bay Club Rally 1"}]
    found = prepare_ball_dataset.find_clip_videos(tmp_path, records)
    assert set(found) == {"Bay Club Rally 1"}
    assert found["Bay Club Rally 1"] == tmp_path / "Bay Club Rally 1.mp4"


def test_find_clip_videos_matches_non_ascii_clip_names(tmp_path):
    # The production case ascii_slug itself documents: a YouTube title
    # carrying a fullwidth bar. The video stem and the clip name are the
    # same non-ASCII string; the match must survive slugging both.
    (tmp_path / "Rally ｜ One.mp4").write_bytes(b"")
    records = [{"clip": "Rally ｜ One"}]
    found = prepare_ball_dataset.find_clip_videos(tmp_path, records)
    assert found["Rally ｜ One"] == tmp_path / "Rally ｜ One.mp4"


def test_find_clip_videos_missing_clip_is_fatal(tmp_path):
    records = [{"clip": "nonexistent-clip"}]
    with pytest.raises(SystemExit, match="nonexistent-clip"):
        prepare_ball_dataset.find_clip_videos(tmp_path, records)


def test_find_clip_videos_missing_clips_dir_is_fatal(tmp_path):
    # A raw FileNotFoundError from Path.iterdir() would be unreadable next
    # to this module's other loud, path-naming SystemExits.
    missing_dir = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit) as excinfo:
        prepare_ball_dataset.find_clip_videos(missing_dir, [{"clip": "Bay"}])
    assert str(missing_dir) in str(excinfo.value)


def test_sequence_render_writes_three_aligned_crops(tmp_path, monkeypatch):
    # monkeypatch decode_frames to return synthetic frames whose pixel values
    # encode the frame index; assert the .tm1/.tp1 files land beside the
    # anchor crop, contain the right frame's pixels (t-1 and t+1), and the
    # COCO "sequence" lists [tm1, anchor, tp1] oldest-first.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, i * 20) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0005_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 5 * 20),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 5, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    plans_by_record = {path: plans}

    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 20})

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    tm1_name, tp1_name = f"{stem}.tm1.jpg", f"{stem}.tp1.jpg"

    assert (train_dir / anchor_name).exists()
    assert (train_dir / tm1_name).exists()
    assert (train_dir / tp1_name).exists()

    # frame 4's pixels (t-1), not frame 5's or frame 6's.
    assert _imread_unicode(train_dir / tm1_name).mean() == pytest.approx(80, abs=5)
    # frame 6's pixels (t+1).
    assert _imread_unicode(train_dir / tp1_name).mean() == pytest.approx(120, abs=5)

    coco = json.loads((tmp_path / "annotations" / "instances_train.json").read_text())
    image = next(i for i in coco["images"] if i["file_name"] == anchor_name)
    assert image["sequence"] == [tm1_name, anchor_name, tp1_name]


def test_sequence_pads_at_clip_start(tmp_path, monkeypatch):
    # labeled frame index 0: tm1 must repeat frame 0 (matches the runtime
    # edge padding in ball_track_offline._centered_windows), not crash or
    # go negative; tp1 is the real frame 1.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, i * 20) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0000_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 0),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 0, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    plans_by_record = {path: plans}

    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 20})

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    tm1_name = f"{stem}.tm1.jpg"
    tp1_name = f"{stem}.tp1.jpg"

    # tm1 repeats frame 0 -- same content as the anchor.
    assert (_imread_unicode(train_dir / tm1_name).mean()
            == pytest.approx(_imread_unicode(train_dir / anchor_name).mean(), abs=2))
    # tp1 is the real frame 1.
    assert _imread_unicode(train_dir / tp1_name).mean() == pytest.approx(20, abs=5)


def test_sequence_pads_at_clip_end(tmp_path, monkeypatch):
    # labeled frame == last frame of the clip: tp1 repeats the last frame
    # (decode_frames raising "ended at frame N" must not happen for this).
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        for i in indices:
            if i > 8:
                raise SystemExit(f"{video_path} ended at frame 9; needed {i}")
        return {i: _solid_frame(np, i * 20) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0008_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 8 * 20),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 8, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    plans_by_record = {path: plans}

    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 8})

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    tp1_name = f"{stem}.tp1.jpg"

    # tp1 repeats the last frame -- same content as the anchor.
    assert (_imread_unicode(train_dir / tp1_name).mean()
            == pytest.approx(_imread_unicode(train_dir / anchor_name).mean(), abs=2))


def test_alignment_failure_is_fatal(tmp_path, monkeypatch):
    # decode_frames returns a frame that does NOT match the export image ->
    # SystemExit naming the clip; a silently shifted sequence poisons training.
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")

    def fake_decode(video_path, indices):
        # Every decoded frame is far from the export image, at every offset.
        return {i: _solid_frame(np, 250) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0005_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 5 * 20),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 5, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    plans_by_record = {path: plans}

    with pytest.raises(SystemExit, match="Bay"):
        render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                     seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                     clip_last_frame={"Bay": 20})


def test_sequence_alignment_detects_and_applies_a_consistent_offset(tmp_path, monkeypatch):
    # A clip-uniform off-by-one in the export (every label's export image
    # actually holds the video's *next* frame) must be both detected AND
    # propagated into which neighbour frames get decoded -- not just used to
    # pass the alignment check and then discarded.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    # decoded index i holds the content of true frame (i - 1): content(t)
    # therefore lives one index ahead of where a naive (offset=0) read would
    # look for it.
    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, (i - 1) * 20) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0005_jpg.jpg"
    # Export image for label 5 holds true frame 5's content (100) -- which
    # decode_frames only serves back at index 6 (offset +1).
    _imwrite_unicode(path, _solid_frame(np, 5 * 20),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 5, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    plans_by_record = {path: plans}

    # Must not raise: a consistent offset across the clip is recoverable.
    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 20})

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    tm1_name, tp1_name = f"{stem}.tm1.jpg", f"{stem}.tp1.jpg"

    # true_t = 5 + 1 = 6, so the window is (5, 6, 7): tm1 = content(5) = 80,
    # tp1 = content(7) = 120. An unshifted (offset=0) read would have used
    # window (4, 5, 6) and written tm1 = content(3) = 60, tp1 = content(5)
    # = 100 instead -- the corrected values below are not those.
    assert _imread_unicode(train_dir / tm1_name).mean() == pytest.approx(80, abs=5)
    assert _imread_unicode(train_dir / tp1_name).mean() == pytest.approx(120, abs=5)


def test_resolve_clip_offset_argmin_beats_first_pass_wins():
    # Offset 0 is under tolerance (5 < 12) but offset +1 is a strictly
    # better fit for every record -- argmin must prefer +1, not stop at the
    # first offset that merely clears the tolerance bar. This is the
    # discriminability gap IMPORTANT-3 exists to close: on static-camera
    # footage adjacent frames are near-duplicates, so a first-pass-wins rule
    # would silently keep offset 0 forever.
    np = pytest.importorskip("numpy")

    def solid(value):
        return np.full((4, 4, 3), value, dtype="uint8")

    export_images = {"a": solid(100), "b": solid(100)}
    decoded = {
        4: solid(93),    # offset -1 candidate for frame 5 -> diff 7
        5: solid(95),    # offset  0 candidate -> diff 5 (under tolerance=12)
        6: solid(99),    # offset +1 candidate -> diff 1 (best fit)
    }
    clip_records = [{"frame": 5, "path": "a"}, {"frame": 5, "path": "b"}]

    offset = prepare_ball_dataset._resolve_clip_offset(
        "clip", clip_records, decoded, export_images, last_frame=None,
        tolerance=12.0)
    assert offset == 1


def test_resolve_clip_offset_still_fails_loudly_when_every_candidate_is_bad():
    # argmin must not turn "best of three bad options" into a silent pass --
    # the winning candidate still has to clear `tolerance`.
    np = pytest.importorskip("numpy")

    def solid(value):
        return np.full((4, 4, 3), value, dtype="uint8")

    export_images = {"a": solid(100)}
    decoded = {4: solid(0), 5: solid(0), 6: solid(0)}
    clip_records = [{"frame": 5, "path": "a"}]

    with pytest.raises(SystemExit, match="clip-x"):
        prepare_ball_dataset._resolve_clip_offset(
            "clip-x", clip_records, decoded, export_images, last_frame=None,
            tolerance=12.0)


def test_sequence_edge_padding_repeats_anchor_under_positive_offset(
        tmp_path, monkeypatch):
    # With a detected +1 offset, the max-labelled record's true VIDEO-space
    # anchor index is last_frame + 1 -- one past anything `_needed_indices`
    # actually fetched. tp1 must repeat the ANCHOR (the record's own export
    # image, scaled/windowed exactly like every other frame) -- not some
    # other decoded frame, and not collapse onto tm1's value. tm1 is the
    # real predecessor, since decode index (true_t - 1) genuinely was
    # fetched. Offset detection itself is orthogonal to this -- forced via
    # monkeypatch.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, i * 10) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)
    monkeypatch.setattr(prepare_ball_dataset, "_resolve_clip_offset",
                        lambda *a, **k: 1)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0008_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 200),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # record["frame"] == clip_last_frame: the max-labelled/edge record.
    record = {"path": path, "clip": "Bay", "frame": 8, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    plans_by_record = {path: plans}

    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 8})

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    tm1_name, tp1_name = f"{stem}.tm1.jpg", f"{stem}.tp1.jpg"

    anchor_tile = _imread_unicode(train_dir / anchor_name)
    tm1_tile = _imread_unicode(train_dir / tm1_name)
    tp1_tile = _imread_unicode(train_dir / tp1_name)

    # true_t = 8 + 1 = 9: tm1's target (8) was actually fetched -> real
    # decoded content (content(8) = 80).
    assert tm1_tile.mean() == pytest.approx(80, abs=5)
    # tp1's target (10) was never fetched -- repeat-ANCHOR: tp1 must equal
    # the anchor crop itself.
    assert tp1_tile.mean() == pytest.approx(anchor_tile.mean(), abs=2)
    assert tm1_tile.mean() != pytest.approx(tp1_tile.mean(), abs=5)


def test_sequence_edge_padding_repeats_anchor_under_negative_offset(
        tmp_path, monkeypatch):
    # Mirror of the above at the opposite corner: frame 0 under a detected
    # -1 offset. true_t = -1 -- there is no video frame before the clip's
    # start -- so tm1 must repeat the ANCHOR; tp1 is the real successor
    # since decode index 0 genuinely was fetched.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, i * 10) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)
    monkeypatch.setattr(prepare_ball_dataset, "_resolve_clip_offset",
                        lambda *a, **k: -1)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0000_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 200),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # record["frame"] == 0: the min-labelled/edge record.
    record = {"path": path, "clip": "Bay", "frame": 0, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    plans_by_record = {path: plans}

    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 8})

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    tm1_name, tp1_name = f"{stem}.tm1.jpg", f"{stem}.tp1.jpg"

    anchor_tile = _imread_unicode(train_dir / anchor_name)
    tm1_tile = _imread_unicode(train_dir / tm1_name)
    tp1_tile = _imread_unicode(train_dir / tp1_name)

    # tm1's target (-2) was never fetched -- repeat-ANCHOR: tm1 must equal
    # the anchor crop itself.
    assert tm1_tile.mean() == pytest.approx(anchor_tile.mean(), abs=2)
    # true_t = 0 - 1 = -1: tp1's target (0) was actually fetched -> real
    # decoded content (content(0) = 0).
    assert tp1_tile.mean() == pytest.approx(0, abs=5)
    assert tm1_tile.mean() != pytest.approx(tp1_tile.mean(), abs=5)


def test_frame_map_overrides_offset_probe_and_uses_mapped_neighbours(
        tmp_path, monkeypatch):
    # A mapped clip must never enter the 0/-1/+1 probe: sampling was not 1:1,
    # so no single offset exists. Neighbours come from the mapped video index.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, i * 10) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    def probe_must_not_run(*args, **kwargs):
        raise AssertionError("offset probe ran for a mapped clip")
    monkeypatch.setattr(
        prepare_ball_dataset, "_resolve_clip_offset", probe_must_not_run)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0005_jpg.jpg"
    # Export image matches mapped video frame 17 (value 170), nowhere near
    # label-space index 5 -- exactly the case the offset probe cannot align.
    _imwrite_unicode(path, _solid_frame(np, 170),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 5, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans_by_record = {path: plan_crops(
        record, crop=16, scale=1.0, rng=random.Random(0),
        positives=1, negatives=0, jitter=0.0, min_visible=0.5)}

    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 20},
                 frame_maps={"Bay": {"video_frame_count": 30,
                                     "frames": {5: 17}}})

    stem = crop_file_name(path.stem, 0)[:-len(".jpg")]
    # tm1 = video frame 16, tp1 = video frame 18 -- mapped space, not 4/6.
    assert (_imread_unicode(train_dir / f"{stem}.tm1.jpg").mean()
            == pytest.approx(160, abs=4))
    assert (_imread_unicode(train_dir / f"{stem}.tp1.jpg").mean()
            == pytest.approx(180, abs=4))


def test_frame_map_alignment_mismatch_is_fatal(tmp_path, monkeypatch):
    # The map changes where to look, never whether alignment is checked: a
    # mapped frame that doesn't match the export image must die loudly.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, i * 10) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0005_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 0),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 5, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans_by_record = {path: plan_crops(
        record, crop=16, scale=1.0, rng=random.Random(0),
        positives=1, negatives=0, jitter=0.0, min_visible=0.5)}

    with pytest.raises(SystemExit, match="frame-map alignment failed"):
        render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                     seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                     clip_last_frame={"Bay": 20},
                     frame_maps={"Bay": {"video_frame_count": 30,
                                         "frames": {5: 17}}})


def test_frame_map_missing_labelled_frame_is_fatal(tmp_path, monkeypatch):
    # A partial map silently falling back to the offset probe would defeat
    # the point of supplying one; it must die naming the unmapped frames.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, i * 10) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0005_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 170),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 5, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans_by_record = {path: plan_crops(
        record, crop=16, scale=1.0, rng=random.Random(0),
        positives=1, negatives=0, jitter=0.0, min_visible=0.5)}

    with pytest.raises(SystemExit, match=r"lacks entries.*\[5\]"):
        render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                     seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                     clip_last_frame={"Bay": 20},
                     frame_maps={"Bay": {"video_frame_count": 30,
                                         "frames": {4: 16}}})


def test_frame_map_neighbour_past_video_end_pads_with_anchor(
        tmp_path, monkeypatch):
    # Mapped anchor at the video's last frame: tp1's target equals
    # video_frame_count, is never requested from decode_frames (which would
    # die there), and pads by repeating the anchor -- same convention as the
    # unmapped clip-end case.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def fake_decode(video_path, indices):
        assert max(indices) <= 29, "requested a frame past the video's end"
        return {i: _solid_frame(np, i * 8) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    path = train_dir / "Bay_mov-0005_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, 232),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": 5, "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans_by_record = {path: plan_crops(
        record, crop=16, scale=1.0, rng=random.Random(0),
        positives=1, negatives=0, jitter=0.0, min_visible=0.5)}

    render_split([record], plans_by_record, tmp_path, "train", 16, 95,
                 seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                 clip_last_frame={"Bay": 20},
                 frame_maps={"Bay": {"video_frame_count": 30,
                                     "frames": {5: 29}}})

    stem = crop_file_name(path.stem, 0)[:-len(".jpg")]
    anchor_mean = _imread_unicode(train_dir / crop_file_name(path.stem, 0)).mean()
    # tm1 = real video frame 28.
    assert (_imread_unicode(train_dir / f"{stem}.tm1.jpg").mean()
            == pytest.approx(224, abs=4))
    # tp1 padded with the anchor's own pixels.
    assert (_imread_unicode(train_dir / f"{stem}.tp1.jpg").mean()
            == pytest.approx(anchor_mean, abs=2))


def test_load_frame_map_parses_ints_and_rejects_malformed(tmp_path):
    good = tmp_path / "map.json"
    good.write_text(json.dumps(
        {"Bay": {"video_frame_count": 30, "frames": {"5": 17}}}))
    parsed = prepare_ball_dataset.load_frame_map(good)
    assert parsed == {"Bay": {"video_frame_count": 30, "frames": {5: 17}}}

    for bad_entry in ({"frames": {"5": 17}},                      # no count
                      {"video_frame_count": 30},                  # no frames
                      {"video_frame_count": 30, "frames": {"5": "x"}}):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"Bay": bad_entry}))
        with pytest.raises(SystemExit, match="Bay"):
            prepare_ball_dataset.load_frame_map(bad)


def test_frame_map_without_seq_frames_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="frame-map only applies"):
        prepare_ball_dataset.build(
            tmp_path, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
            [], [], 0, 95, seq_frames=1,
            frame_maps={"Bay": {"video_frame_count": 1, "frames": {}}})


# --- Clip exclusion (--exclude-clips) ----------------------------------

def _tiny_export(tmp_path, np, cv2, clips, name="export"):
    """Minimal Roboflow-shaped COCO export.

    `clips` maps a clip name to either `[frame_number, ...]` (every frame a
    solid 100) or `{frame_number: pixel_value}` when a test needs to tell the
    frames apart by content.
    """
    root = tmp_path / name
    split = root / "train"
    split.mkdir(parents=True)
    images, annotations = [], []
    for clip, frames in clips.items():
        values = frames if isinstance(frames, dict) else {f: 100 for f in frames}
        for number, value in sorted(values.items()):
            file_name = f"{clip}_mov-{number:04d}_jpg.jpg"
            _imwrite_unicode(split / file_name, _solid_frame(np, value),
                             [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            images.append({"id": len(images) + 1, "file_name": file_name,
                           "width": 64, "height": 64})
            annotations.append({"id": len(annotations) + 1,
                                "image_id": len(images), "category_id": 0,
                                "bbox": [24, 24, 8, 8]})
    (split / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations,
                    "categories": [{"id": 0, "name": "ball"}]}),
        encoding="utf-8")
    return root


def test_exclude_clips_drops_the_clip_and_records_the_cost(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = _tiny_export(tmp_path, np, cv2,
                          {"Keep": [0, 1, 2], "Tutorial": [0, 1]})

    baseline = prepare_ball_dataset.build(
        source, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
        [], [], 0, 95)
    assert baseline["excluded_clips"] == {}
    assert baseline["source_frames_kept"] == 5

    manifest = prepare_ball_dataset.build(
        source, tmp_path / "out2", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
        [], [], 0, 95, exclude_clips=["Tutorial"])

    # `seen` still counts the whole export while `kept` drops to 3, and the
    # difference is spelled out -- a shrunken dataset has to read as a
    # decision, not as unexplained data loss.
    assert manifest["source_frames_seen"] == 5
    assert manifest["source_frames_kept"] == 3
    assert manifest["excluded_clips"] == {"Tutorial": 2}
    assert manifest["splits"]["train"]["clips"] == ["Keep"]


def test_exclude_clips_rejects_an_unknown_name(tmp_path):
    # A typo'd clip name silently keeping the clip is the failure this
    # guards: the build would look like it succeeded.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = _tiny_export(tmp_path, np, cv2, {"Keep": [0, 1]})
    with pytest.raises(SystemExit, match="not in the export"):
        prepare_ball_dataset.build(
            source, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
            [], [], 0, 95, exclude_clips=["Tutorail"])


def test_exclude_clips_removing_everything_is_fatal(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = _tiny_export(tmp_path, np, cv2, {"Only": [0, 1]})
    with pytest.raises(SystemExit, match="every frame"):
        prepare_ball_dataset.build(
            source, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
            [], [], 0, 95, exclude_clips=["Only"])


# --- Sequence scene continuity (--seq-cut-threshold) -------------------

def _seq_record(tmp_path, np, cv2, anchor_frame, anchor_value):
    """One labelled record whose export image is a solid `anchor_value`."""
    train_dir = tmp_path / "train"
    train_dir.mkdir(exist_ok=True)
    path = train_dir / f"Bay_mov-{anchor_frame:04d}_jpg.jpg"
    _imwrite_unicode(path, _solid_frame(np, anchor_value),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    record = {"path": path, "clip": "Bay", "frame": anchor_frame,
              "width": 64, "height": 64,
              "balls": [{"bbox": [24, 24, 8, 8], "streak": None}]}
    plans = plan_crops(record, crop=16, scale=1.0, rng=random.Random(0),
                       positives=1, negatives=0, jitter=0.0, min_visible=0.5)
    return record, path, plans


def _decode_values(np, monkeypatch, values, default=100):
    def fake_decode(video_path, indices):
        return {i: _solid_frame(np, values.get(i, default)) for i in indices}
    monkeypatch.setattr(prepare_ball_dataset, "decode_frames", fake_decode)


def test_sequence_continuity_is_measured_without_a_threshold(tmp_path, monkeypatch):
    # Default build: tiles unchanged, but the anchor-vs-neighbour spread is
    # reported anyway. This is what makes --seq-cut-threshold a measurement
    # rather than a guess.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    _decode_values(np, monkeypatch, {4: 90, 5: 100, 6: 120})
    record, path, plans = _seq_record(tmp_path, np, cv2, 5, 100)

    slice_ = render_split([record], {path: plans}, tmp_path, "train", 16, 95,
                          seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                          clip_last_frame={"Bay": 20})

    cont = slice_["sequence_continuity"]["Bay"]
    assert cont["neighbours_measured"] == 2
    assert cont["cut_padded"] == 0
    assert cont["dropped_no_neighbour"] == 0
    # |90-100| = 10 and |120-100| = 20.
    assert cont["neighbour_diff"]["p50"] == pytest.approx(20, abs=2)
    assert cont["neighbour_diff"]["max"] == pytest.approx(20, abs=2)
    assert slice_["sequence_crops"] == 1


def test_sequence_cut_threshold_pads_only_the_discontinuous_side(tmp_path, monkeypatch):
    # tm1 is the same shot, tp1 is a cut. tp1 must fall back to repeating the
    # anchor -- identical to the existing clip-edge convention -- while tm1
    # stays the real neighbour, so the sample keeps its temporal signal.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    _decode_values(np, monkeypatch, {4: 90, 5: 100, 6: 250})
    record, path, plans = _seq_record(tmp_path, np, cv2, 5, 100)

    slice_ = render_split([record], {path: plans}, tmp_path, "train", 16, 95,
                          seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                          clip_last_frame={"Bay": 20}, cut_threshold=50.0)

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    train_dir = tmp_path / "train"
    assert (_imread_unicode(train_dir / f"{stem}.tm1.jpg").mean()
            == pytest.approx(90, abs=5))
    assert (_imread_unicode(train_dir / f"{stem}.tp1.jpg").mean()
            == pytest.approx(_imread_unicode(train_dir / anchor_name).mean(), abs=2))

    cont = slice_["sequence_continuity"]["Bay"]
    assert cont["cut_padded"] == 1
    assert cont["dropped_no_neighbour"] == 0
    assert slice_["sequence_crops"] == 1


def test_sequence_drops_the_anchor_when_both_neighbours_are_cuts(tmp_path, monkeypatch):
    # Both sides cut -> the stack would be three IDENTICAL frames carrying a
    # positive ball label, which is exactly what WASB-TRAIN.md 5(d) wants to
    # predict EMPTY. No tiles are written and the COCO entry carries no
    # "sequence", so the sequence dataset skips it; the anchor survives as a
    # plain single-frame crop.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    _decode_values(np, monkeypatch, {4: 250, 5: 100, 6: 250})
    record, path, plans = _seq_record(tmp_path, np, cv2, 5, 100)

    slice_ = render_split([record], {path: plans}, tmp_path, "train", 16, 95,
                          seq_frames=3, clip_videos={"Bay": Path("fake.mp4")},
                          clip_last_frame={"Bay": 20}, cut_threshold=50.0)

    anchor_name = crop_file_name(path.stem, 0)
    stem = anchor_name[:-len(".jpg")]
    train_dir = tmp_path / "train"
    assert (train_dir / anchor_name).exists()
    assert not (train_dir / f"{stem}.tm1.jpg").exists()
    assert not (train_dir / f"{stem}.tp1.jpg").exists()

    coco = json.loads(
        (tmp_path / "annotations" / "instances_train.json").read_text())
    image = next(i for i in coco["images"] if i["file_name"] == anchor_name)
    assert "sequence" not in image

    cont = slice_["sequence_continuity"]["Bay"]
    # cut_padded counts neighbour SIDES replaced; dropped counts ANCHORS.
    assert cont["cut_padded"] == 2
    assert cont["dropped_no_neighbour"] == 1
    assert slice_["sequence_crops"] == 0


def test_continuity_report_percentiles_are_ordered():
    stats = {"Bay": {"neighbour_diffs": [float(v) for v in range(100)],
                     "cut_padded": 3, "dropped_no_neighbour": 1}}
    report = prepare_ball_dataset._continuity_report(stats)["Bay"]
    assert report["neighbours_measured"] == 100
    assert report["cut_padded"] == 3
    assert report["dropped_no_neighbour"] == 1
    spread = report["neighbour_diff"]
    assert spread["p50"] <= spread["p90"] <= spread["p99"] <= spread["max"]
    assert spread["max"] == 99


# --- Export-sourced sequences (--seq-from-export) ----------------------

def _seq_names(tmp_path, split, source_frame):
    """(anchor, tm1, tp1) crop filenames for a source frame, from the COCO."""
    coco = json.loads(
        (tmp_path / "annotations" / f"instances_{split}.json").read_text())
    image = next(i for i in coco["images"] if i["source_frame"] == source_frame)
    return image


def test_seq_from_export_builds_sequences_with_no_video(tmp_path):
    # The whole point: a 3-frame dataset from the export alone. Frame values
    # differ per frame, so "which image landed in tm1" is checkable.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = _tiny_export(tmp_path, np, cv2, {"Bay": {0: 50, 1: 100, 2: 150}})
    out = tmp_path / "out"

    manifest = prepare_ball_dataset.build(
        source, out, 16, 1, 0, 0.0, 0.5, 0, 1, 0,
        [], [], 0, 95, seq_frames=3, seq_from_export=True)

    assert manifest["params"]["seq_neighbour_source"] == "export"
    assert manifest["schema_version"] == prepare_ball_dataset.SCHEMA_VERSION_SEQ

    image = _seq_names(out, "train", 1)
    tm1_name, anchor_name, tp1_name = image["sequence"]
    train_dir = out / "train"
    # Oldest-first: tm1 is frame 0's pixels, tp1 is frame 2's.
    assert _imread_unicode(train_dir / tm1_name).mean() == pytest.approx(50, abs=5)
    assert _imread_unicode(train_dir / anchor_name).mean() == pytest.approx(100, abs=5)
    assert _imread_unicode(train_dir / tp1_name).mean() == pytest.approx(150, abs=5)


def test_seq_from_export_repeats_the_anchor_at_a_run_boundary(tmp_path):
    # Frame 0 has no predecessor in the export. That side repeats the anchor
    # -- the same convention the video path uses at a clip edge -- rather than
    # dropping a usable sample.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = _tiny_export(tmp_path, np, cv2, {"Bay": {0: 50, 1: 100, 2: 150}})
    out = tmp_path / "out"

    prepare_ball_dataset.build(
        source, out, 16, 1, 0, 0.0, 0.5, 0, 1, 0,
        [], [], 0, 95, seq_frames=3, seq_from_export=True)

    image = _seq_names(out, "train", 0)
    tm1_name, anchor_name, tp1_name = image["sequence"]
    train_dir = out / "train"
    anchor_mean = _imread_unicode(train_dir / anchor_name).mean()
    assert _imread_unicode(train_dir / tm1_name).mean() == pytest.approx(anchor_mean, abs=2)
    assert _imread_unicode(train_dir / tp1_name).mean() == pytest.approx(100, abs=5)


def test_seq_from_export_drops_an_isolated_frame(tmp_path):
    # Neither neighbour exists -> three identical frames under a positive
    # label, which 5(d) wants to read as EMPTY. Dropped from the sequence set;
    # the anchor still ships as a plain single-frame crop.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = _tiny_export(tmp_path, np, cv2, {"Bay": {0: 50, 9: 150}})
    out = tmp_path / "out"

    manifest = prepare_ball_dataset.build(
        source, out, 16, 1, 0, 0.0, 0.5, 0, 1, 0,
        [], [], 0, 95, seq_frames=3, seq_from_export=True)

    train = manifest["splits"]["train"]
    assert train["sequence_crops"] == 0
    assert train["crops"] == 2               # both anchors still written
    assert train["sequence_continuity"]["Bay"]["dropped_no_neighbour"] == 2

    coco = json.loads(
        (out / "annotations" / "instances_train.json").read_text())
    assert all("sequence" not in i for i in coco["images"])


def test_seq_from_export_neighbours_survive_burst_thinning(tmp_path):
    # --min-frame-gap thins which frames become ANCHORS; it must not delete
    # frames from the neighbour index. With gap 2 the anchors are 0/2/4, and
    # anchor 2's neighbours are the thinned-away frames 1 and 3 -- which are
    # still on disk and still the correct t-1/t+1.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = _tiny_export(
        tmp_path, np, cv2, {"Bay": {0: 20, 1: 40, 2: 60, 3: 80, 4: 100}})
    out = tmp_path / "out"

    prepare_ball_dataset.build(
        source, out, 16, 1, 0, 0.0, 0.5, 0, 2, 0,
        [], [], 0, 95, seq_frames=3, seq_from_export=True)

    image = _seq_names(out, "train", 2)
    tm1_name, _, tp1_name = image["sequence"]
    train_dir = out / "train"
    assert _imread_unicode(train_dir / tm1_name).mean() == pytest.approx(40, abs=5)
    assert _imread_unicode(train_dir / tp1_name).mean() == pytest.approx(80, abs=5)


def test_seq_from_export_rejects_clips_dir(tmp_path):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        prepare_ball_dataset.build(
            tmp_path, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
            [], [], 0, 95, seq_frames=3, clips_dir=tmp_path,
            seq_from_export=True)


def test_seq_from_export_without_seq_frames_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="only applies"):
        prepare_ball_dataset.build(
            tmp_path, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
            [], [], 0, 95, seq_frames=1, seq_from_export=True)


def test_seq_from_export_rejects_a_frame_map(tmp_path):
    with pytest.raises(SystemExit, match="meaningless"):
        prepare_ball_dataset.build(
            tmp_path, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
            [], [], 0, 95, seq_frames=3, seq_from_export=True,
            frame_maps={"Bay": {"video_frame_count": 1, "frames": {}}})


def test_seq_frames_without_a_neighbour_source_names_both_options(tmp_path):
    # The error an operator with no footage hits first -- it has to point at
    # --seq-from-export, not just demand --clips-dir.
    with pytest.raises(SystemExit, match="seq-from-export"):
        prepare_ball_dataset.build(
            tmp_path, tmp_path / "out", 16, 1, 0, 0.0, 0.5, 0, 1, 0,
            [], [], 0, 95, seq_frames=3, clips_dir=None)


def test_continuity_report_survives_a_clip_with_no_measurements():
    # Every anchor dropped at a clip edge -> no diffs to summarise, but the
    # clip must still appear with its counts rather than vanishing.
    stats = {"Bay": {"neighbour_diffs": [], "cut_padded": 0,
                     "dropped_no_neighbour": 2}}
    report = prepare_ball_dataset._continuity_report(stats)["Bay"]
    assert report["neighbours_measured"] == 0
    assert report["dropped_no_neighbour"] == 2
    assert "neighbour_diff" not in report
