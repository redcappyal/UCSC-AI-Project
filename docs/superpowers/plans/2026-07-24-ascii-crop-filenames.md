# ASCII Crop Filenames + Unicode-Safe cv2 I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it impossible for `prepare_ball_dataset.py` to emit a non-ASCII crop filename, and fix the two OpenCV call sites that mishandle non-ASCII paths on Windows.

**Architecture:** A pure `ascii_slug()` transliterates each source frame stem to `[A-Za-z0-9._-]`, appending an 8-hex SHA-1 digest of the original only when the name actually changed — so already-clean names stay byte-identical and lossy collapses stay distinct. The human-readable clip name keeps living in the COCO per-image `clip` field. Separately, `_imread_unicode`/`_imwrite_unicode` replace `cv2.imread`/`cv2.imwrite`, routing bytes through `np.fromfile`/`ndarray.tofile` so the path never reaches OpenCV's ANSI file API.

**Tech Stack:** Python 3, stdlib `unicodedata`/`hashlib`/`re`, OpenCV (`cv2` 5.0.0) and NumPy at the I/O boundary only, pytest.

**Spec:** [docs/superpowers/specs/2026-07-24-ascii-crop-filenames-design.md](../specs/2026-07-24-ascii-crop-filenames-design.md)

## Global Constraints

- **`prepare_ball_dataset.py` must stay importable without cv2 or numpy.** `tests/test_prepare_ball_dataset.py` opens with `"""prepare_ball_dataset: geometry, splitting and crop planning (no cv2 needed)."""` and runs in a bare interpreter. All cv2/numpy imports stay function-local, matching the existing `import cv2  # lazy: geometry is testable without it` inside `render_split`.
- **Safe filename charset is exactly `[A-Za-z0-9._-]`.** Nothing else may appear in an emitted crop filename.
- **Only names already in canonical form skip the digest.** Canonical means safe charset *and* no leading/trailing `._-`; `Bay-Club-1_mov-0042_jpg.rf.abc123` → itself, no digest, because it already satisfies both. A name the charset filter leaves alone but stripping still shortens (e.g. `abc.` or `.hidden`) still earns the digest, since that strip is itself a lossy change. This is not a whole-dataset rename.
- **Digest is `hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]`**, computed on the *original* stem, appended after a `-`.
- **Do not touch `cv2.VideoCapture` / `cv2.VideoWriter` call sites.** Measured working with a `｜` path (FFmpeg converts UTF-8 to wide chars itself). Changing them adds risk for no benefit.
- **Test interpreter on this Windows box:** no single interpreter has both pytest and cv2. Use a pytest-only venv for the suite; `C:\Users\alann\Code\ball-detector-train\.venv` has cv2 5.0.0 + numpy 2.4.4 but no pytest. The `.venv` in CLAUDE.md does not exist here.
- Baseline before any change: `tests/test_prepare_ball_dataset.py` = **19 passed**.

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `prepare_ball_dataset.py` | Slug + crop naming + manifest + cv2 I/O helpers | Modify |
| `tests/test_prepare_ball_dataset.py` | Unit tests for all of the above | Modify |
| `CLAUDE.md` | Windows env + the cv2 rule | Modify |

No new files. The slug and I/O helpers belong beside their only consumer; extracting a
module would add an import for two call sites.

---

### Task 1: `ascii_slug()`

**Files:**
- Modify: `prepare_ball_dataset.py` (imports near line 32-39; new function after `clip_and_frame`, ~line 62)
- Test: `tests/test_prepare_ball_dataset.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ascii_slug(stem: str) -> str`. Total function, never raises, never returns `""`. Output always matches `^[A-Za-z0-9._-]+$`. Used by Task 2.

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `tests/test_prepare_ball_dataset.py` — it currently imports `_clip_box, burst_count, clip_and_frame, ...`; add `ascii_slug` in alphabetical position (first):

```python
import re

from prepare_ball_dataset import (
    _clip_box, ascii_slug, burst_count, clip_and_frame, clip_scale_factors,
    plan_crops, polygon_aabb, polygon_points, split_by_clip, streak_metrics,
    thin_bursts,
)
```

Append these tests to the end of the file:

```python
SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def test_ascii_slug_leaves_a_clean_roboflow_name_alone():
    # The 1,975 already-good crops must not be renamed by this change.
    name = "Bay-Club-1_mov-0042_jpg.rf.abc123"
    assert ascii_slug(name) == name


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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_prepare_ball_dataset.py -q
```

Expected: collection error — `ImportError: cannot import name 'ascii_slug' from 'prepare_ball_dataset'`.

- [ ] **Step 3: Write the implementation**

In `prepare_ball_dataset.py`, add `hashlib` and `unicodedata` to the import block (keep it alphabetical: `argparse, hashlib, json, math, random, re, statistics, unicodedata`).

Add beneath the `FRAME_RE` constant:

```python
# Anything outside this set breaks a crop filename somewhere in the chain: cv2
# on Windows, or a zip round-trip that loses the UTF-8 flag. See
# docs/superpowers/specs/2026-07-24-ascii-crop-filenames-design.md.
UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
```

Add after `clip_and_frame`:

```python
def ascii_slug(stem):
    """ASCII, filesystem-safe form of a source frame stem.

    One source clip is a YouTube title carrying U+FF5C (｜, a sanitised "|"), and
    a filename built from it is unusable on Windows twice over: cv2.imread
    returns None for it, and cv2.imwrite reports success while writing a
    mojibake name. Emitting ASCII is what makes the crop names portable.

    NFKD runs first so accents transliterate (é -> e) instead of vanishing, but
    it also turns U+FF5C into a literal "|" — valid ASCII, invalid in a Windows
    filename — so the charset filter runs after it, never instead of it.

    The transform is lossy, so a name that changed carries an 8-hex digest of
    the original: two clips that collapse onto one base stay distinct, and the
    suffix is stable per source name. A name already in the safe set is returned
    untouched, which is what keeps this from renaming the whole dataset.
    """
    folded = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    slug = UNSAFE_CHARS.sub("_", folded).strip("._-") or "clip"
    if slug == stem:
        return slug
    return f"{slug}-{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:8]}"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_prepare_ball_dataset.py -q
```

Expected: `27 passed` (19 baseline + 8 new).

- [ ] **Step 5: Commit**

```bash
git add prepare_ball_dataset.py tests/test_prepare_ball_dataset.py && git commit -m "feat: add ascii_slug for portable crop filenames"
```

---

### Task 2: Apply the slug to crop names, and surface it in the manifest

**Files:**
- Modify: `prepare_ball_dataset.py` (new helpers after `ascii_slug`; `render_split` line 314; `build` manifest dict ~line 391-416)
- Test: `tests/test_prepare_ball_dataset.py`

**Interfaces:**
- Consumes: `ascii_slug(stem) -> str` from Task 1.
- Produces: `crop_file_name(source_stem: str, index: int) -> str` and `slugified_clips(records: list[dict]) -> list[str]`. `records` are the dicts built by `load_export` — each has a `"clip"` key. Used by `render_split` and `build`; no later task depends on them.

- [ ] **Step 1: Write the failing tests**

Add `crop_file_name` and `slugified_clips` to the import block (alphabetical):

```python
from prepare_ball_dataset import (
    _clip_box, ascii_slug, burst_count, clip_and_frame, clip_scale_factors,
    crop_file_name, plan_crops, polygon_aabb, polygon_points, slugified_clips,
    split_by_clip, streak_metrics, thin_bursts,
)
```

Append:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_prepare_ball_dataset.py -q
```

Expected: collection error — `ImportError: cannot import name 'crop_file_name'`.

- [ ] **Step 3: Write the implementation**

Add after `ascii_slug` in `prepare_ball_dataset.py`:

```python
def crop_file_name(source_stem, index):
    """Filename for the `index`-th crop cut from a source frame."""
    return f"{ascii_slug(source_stem)}_c{index}.jpg"


def slugified_clips(records):
    """Clip names that could not be used verbatim in a filename.

    Reported in the manifest so the digest suffix on those crops is
    self-explaining; the readable name itself stays in each image's `clip`.
    """
    return sorted({r["clip"] for r in records if ascii_slug(r["clip"]) != r["clip"]})
```

In `render_split`, replace line 314:

```python
            name = f"{record['path'].stem}_c{index}.jpg"
```

with:

```python
            name = crop_file_name(record["path"].stem, index)
```

In `build`, add to the `manifest` dict — immediately after the `"clip_scale_factors"` entry, keeping the existing style:

```python
        # The readable name lives in each image's `clip`; this flags the clips
        # whose crops therefore carry a digest suffix.
        "slugified_clips": slugified_clips(kept),
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_prepare_ball_dataset.py -q
```

Expected: `30 passed`.

- [ ] **Step 5: Confirm the raw clip name is still the one recorded**

`render_split` already writes `"clip": record["clip"]` (line 320) from the unslugged value — verify by eye that this line is untouched, since it is the traceability anchor the whole design leans on.

```bash
grep -n '"clip": record\["clip"\]' prepare_ball_dataset.py
```

Expected: one hit, unchanged.

- [ ] **Step 6: Commit**

```bash
git add prepare_ball_dataset.py tests/test_prepare_ball_dataset.py && git commit -m "feat: slugify crop filenames, keep readable clip name in metadata"
```

---

### Task 3: Unicode-safe cv2 read/write

**Files:**
- Modify: `prepare_ball_dataset.py` (helpers before `render_split`; `render_split` lines 302 and 315-316)
- Test: `tests/test_prepare_ball_dataset.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_imread_unicode(path) -> ndarray | None` (`None` for missing/empty/undecodable, preserving `render_split`'s skip) and `_imwrite_unicode(path, image, params=None) -> None` (raises `OSError` on encode failure). `path` may be `str` or `Path`.

- [ ] **Step 1: Write the failing tests**

Add to the import block: `_imread_unicode, _imwrite_unicode` (alphabetical, before `_clip_box`... note `_clip_box` sorts first):

```python
from prepare_ball_dataset import (
    _clip_box, _imread_unicode, _imwrite_unicode, ascii_slug, burst_count,
    clip_and_frame, clip_scale_factors, crop_file_name, plan_crops,
    polygon_aabb, polygon_points, slugified_clips, split_by_clip,
    streak_metrics, thin_bursts,
)
```

Append:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_prepare_ball_dataset.py -q
```

Expected: collection error — `ImportError: cannot import name '_imread_unicode'`.

- [ ] **Step 3: Write the implementation**

Add immediately above `render_split` in `prepare_ball_dataset.py`:

```python
def _imread_unicode(path):
    """cv2.imread that survives a non-ASCII path on Windows.

    cv2 gets the UTF-8 bytes of a Python str and hands them to the CRT's
    non-Unicode file API, so on a cp1252 box every non-ASCII path misses and
    imread returns None with the file sitting right there. Reading the bytes in
    Python and decoding them in memory sidesteps the path entirely. Returns None
    for a missing or unreadable file, as imread did.
    """
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path, image, params=None):
    """cv2.imwrite that survives a non-ASCII path on Windows.

    The read-side failure is loud; this one is not. cv2.imwrite returns True and
    writes a real file whose name is the UTF-8 bytes reinterpreted as cp1252, so
    the COCO json ends up pointing at files that do not exist — how the
    2026-07-24 dataset lost 961 of its 2,936 train crops. Raising rather than
    returning a bool keeps a silent no-op off the table.
    """
    import cv2

    ok, buffer = cv2.imencode(Path(path).suffix, image, params or [])
    if not ok:
        raise OSError(f"cv2 could not encode {path}")
    buffer.tofile(str(path))
```

In `render_split`, replace line 302:

```python
        frame = cv2.imread(str(record["path"]))
```

with:

```python
        frame = _imread_unicode(record["path"])
```

and replace lines 315-316:

```python
            cv2.imwrite(str(images_dir / name), tile,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])
```

with:

```python
            _imwrite_unicode(images_dir / name, tile,
                             [int(cv2.IMWRITE_JPEG_QUALITY), quality])
```

Leave `render_split`'s own `import cv2` in place — `cv2.IMWRITE_JPEG_QUALITY` and `cv2.resize` still need it.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_prepare_ball_dataset.py -q
```

Expected under a pytest-only interpreter: `30 passed, 2 skipped` — the two cv2 tests skip. **That is not a pass for those two.** Verify them for real under the cv2 interpreter in Step 5.

- [ ] **Step 5: Verify the cv2 helpers against a real `｜` path**

No interpreter here has both pytest and cv2, so exercise the same assertions directly:

```bash
"C:/Users/alann/Code/ball-detector-train/.venv/Scripts/python.exe" -c "import tempfile,pathlib,sys; sys.path.insert(0,'.'); import numpy as np, cv2; from prepare_ball_dataset import _imread_unicode,_imwrite_unicode; d=pathlib.Path(tempfile.mkdtemp()); t=d/'Rally ｜ One_c0.jpg'; _imwrite_unicode(t,np.full((16,16,3),128,np.uint8),[int(cv2.IMWRITE_JPEG_QUALITY),95]); print('on disk:',[p.name for p in d.iterdir()]); print('shape:',_imread_unicode(t).shape); print('missing ->',_imread_unicode(d/'absent.jpg'))"
```

Expected: `on disk: ['Rally ｜ One_c0.jpg']` (exactly one file, no mojibake sibling), `shape: (16, 16, 3)`, `missing -> None`.

- [ ] **Step 6: Commit**

```bash
git add prepare_ball_dataset.py tests/test_prepare_ball_dataset.py && git commit -m "fix: route crop image I/O around cv2's ANSI path API on Windows"
```

---

### Task 4: Document the Windows box and the cv2 rule

**Files:**
- Modify: `CLAUDE.md` (Environment section)

**Interfaces:**
- Consumes: names `ascii_slug`, `_imread_unicode`, `_imwrite_unicode` from Tasks 1-3.
- Produces: nothing.

- [ ] **Step 1: Add the note**

In `CLAUDE.md`, after the paragraph ending "Failures come back as a *blocked edit*, not a warning.", add:

```markdown
On the Windows CUDA training box there is no `.venv` here; that environment lives at
`C:\Users\alann\Code\ball-detector-train\.venv` (cv2 + torch, no pytest or flask), and
the PostToolUse hook above is not configured there.

**Never pass a possibly-non-ASCII path to `cv2.imread`/`cv2.imwrite`.** On Windows both
reach the CRT's ANSI file API: reads return `None`, and writes return `True` while landing
under a mojibake filename — that is how `ball-crops-2026-07-24` lost 961 of 2,936 train
crops. Use `_imread_unicode`/`_imwrite_unicode` in `prepare_ball_dataset.py`. Crop
filenames are ASCII by construction via `ascii_slug()`, with the readable clip name kept
in the COCO per-image `clip` field. `cv2.VideoCapture`/`VideoWriter` are *not* affected —
FFmpeg does its own UTF-8 conversion — so leave those call sites alone.
```

- [ ] **Step 2: Verify the claims still hold**

```bash
grep -rn "cv2.imread\|cv2.imwrite" --include=*.py .
```

Expected: no hits (both call sites are now the helpers). If a hit appears, the doc is lying — fix the call site.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md && git commit -m "docs: record the Windows training env and the cv2 path rule"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
| --- | --- |
| §Design 1 `ascii_slug` | Task 1 |
| §Design 2 Unicode-safe I/O | Task 3 |
| §Design 3 COCO `clip` untouched + `slugified_clips` | Task 2 (Steps 3, 5) |
| §Design 4 Tests (8 slug cases + cv2 round-trip) | Tasks 1-3 |
| §Design 5 Documentation | Task 4 |
| §Audit (no other call sites) | Task 4 Step 2 asserts it mechanically |

**Type consistency:** `ascii_slug` is referenced with the same name and signature in Tasks 1, 2 and 4; `crop_file_name`/`slugified_clips` in Task 2 only; `_imread_unicode`/`_imwrite_unicode` in Tasks 3 and 4. The test-file import block is restated in full at each task because it grows — later tasks show the complete line, not a diff.

**Known non-verification:** the two cv2 tests skip under the pytest interpreter. Task 3 Step 5 covers them with a direct run and states the split explicitly rather than folding them into a pass count. CLAUDE.md's "259 tests" figure is left alone: the full suite needs flask, torch and cv2 together, which no interpreter here provides, so a new total cannot be measured on this machine.
