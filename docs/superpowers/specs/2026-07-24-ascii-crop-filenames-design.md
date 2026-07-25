# ASCII crop filenames + Unicode-safe cv2 I/O (2026-07-24)

`prepare_ball_dataset.py` derives each output crop's filename from its source clip name.
One source clip is a YouTube video whose title contains **U+FF5C** (｜, fullwidth vertical
line — a sanitized `|`), so 961 of the 2,936 train crops in `ball-crops-2026-07-24` carry
non-ASCII filenames. That cost two separate training failures on the Windows CUDA box.

This spec makes non-ASCII crop filenames impossible to emit, and fixes the OpenCV call
sites that mishandle non-ASCII paths regardless of what the generator names things.

## The two observed failures

**F1 — transport corruption.** The dataset was generated on macOS and unzipped on Windows
without the zip's UTF-8 filename flag being honoured. Each UTF-8 byte was decoded as
CP437: U+FF5C (`EF BD 9C`) became `∩╜£` (U+2229 U+255C U+00A3) on disk. The COCO JSON kept
the correct UTF-8 name, so 961 of 2,936 train entries pointed at files that did not exist.
Recovered by renaming with `name.encode('cp437').decode('utf-8')`.

**F2 — OpenCV cannot read the correct path either.** `cv2.imread` on Windows reaches the
CRT's non-Unicode file API, so it returns `None` for a path it cannot represent in the
active ANSI code page. Worked around by patching the local YOLOX clone's
`yolox/data/datasets/coco.py` `load_image` to use
`cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)`.

## Measured behaviour (this machine, 2026-07-24)

Probed directly rather than assumed. Environment: `cv2` 5.0.0, `numpy` 2.4.4,
`locale.getpreferredencoding()` = `cp1252`, `sys.getfilesystemencoding()` = `utf-8`,
interpreter `C:\Users\alann\Code\ball-detector-train\.venv`.

| path | `cv2.imread` | `cv2.imwrite` | `cv2.imdecode(np.fromfile(...))` |
| --- | --- | --- | --- |
| `plain_c0.jpg` | ✅ `(16,16,3)` | ✅ writes `w_plain_c0.jpg` | ✅ |
| `clip｜name_c0.jpg` | ❌ `None` | ⚠️ **returns `True`**, writes `w_clipï½œname_c0.jpg` | ✅ |
| `café_c0.jpg` | ❌ `None` | ⚠️ **returns `True`**, writes `w_cafÃ©_c0.jpg` | ✅ |

Three findings that change the diagnosis:

- **D1. `cv2.imwrite` fails silently and destructively.** It does not return `False`; it
  returns `True` having written a real file under a mojibake name — the UTF-8 bytes
  reinterpreted as cp1252 (`EF BD 9C` → `ï ½ œ`). **F1 therefore did not require the zip
  at all.** Regenerating the dataset natively on Windows produces the identical symptom —
  correct UTF-8 names in the COCO JSON, mojibake names on disk — through `cv2.imwrite`
  alone. The zip/CP437 path was one route in; `imwrite` is a second, independent one.
- **D2. The boundary is ASCII, not cp1252.** `café` also failed, though `é` *is*
  representable in cp1252, because Python hands OpenCV UTF-8 (`C3 A9`) and cp1252 wants
  `E9`. Any non-ASCII byte breaks it. This is why the fix is an ASCII slug rather than a
  codepage-aware encoding.
- **D3. `VideoCapture` / `VideoWriter` are unaffected.** `clip｜name.mp4` was written and
  read back successfully (`isOpened()` true both ways, first frame read OK) — FFmpeg does
  its own UTF-8→wide-char conversion on Windows. The ~12 video call sites across `app.py`,
  `job_runner.py`, `local_model_eval.py`, `stereo_offline.py`, `modelEval.py`,
  `yolo_model_eval.py`, `benchmark_tracking.py`, `label_hits.py` and
  `train_bounce_classifier.py` need no change. Recorded here so they are not "fixed" later.

## Audit of image-file I/O call sites

Repo-wide grep for `imread|imwrite|imdecode|imencode|fromfile|tofile`:

- `prepare_ball_dataset.py:302` — `cv2.imread(str(record["path"]))`. **Broken on Windows.**
  Source export paths keep the raw Roboflow names, so they still contain U+FF5C after this
  change. Must be fixed independently of the slug.
- `prepare_ball_dataset.py:315` — `cv2.imwrite(str(images_dir / name), ...)`. **Broken on
  Windows (silently, per D1).** The slug fixes the leaf name, but `--out` may itself sit
  under a non-ASCII directory, so this must be fixed too.

There are no other `imread`/`imwrite` calls in the repo. In particular `inference_engine.py`,
`local_model_eval.py` and `rerun_detection.py` have no image-file I/O; `rerun_detection.py`
does not import cv2 at all.

Runtime uploads are already safe by construction: `app.py` routes every user-supplied
filename through `werkzeug.utils.secure_filename`, which strips non-ASCII.

## Design

### 1. `ascii_slug(stem)` — new function in `prepare_ball_dataset.py`

```
NFKD-normalize
  → encode('ascii', 'ignore').decode('ascii')
  → re.sub(r"[^A-Za-z0-9._-]+", "_", ...)
  → strip leading/trailing "._-"
  → "clip" if empty
  → if result != input: result += "-" + sha1(original.encode('utf-8')).hexdigest()[:8]
```

Ordering matters. NFKD buys transliteration for free (`é` → `e` + combining acute, and
`ascii/ignore` drops the accent rather than the letter), but it also maps U+FF5C to
U+007C `|` — ASCII, yet illegal in a Windows filename. The charset filter therefore runs
*after* normalization, never instead of it. The `+` quantifier collapses runs, so
`Rally ｜ Best` yields `Rally_Best`, not `Rally___Best`.

The conditional hash restores what slugging destroys. The transform is lossy — two
distinct clips can collapse onto one base — so an 8-hex prefix of the SHA-1 of the
*original* stem makes the mapping injective in practice, and traceable: the same source
name always produces the same suffix. Gating it on `result != input` means the ~1,975
already-clean crops keep byte-identical filenames, so this is not a whole-dataset rename.

Applied at the crop-naming site (`prepare_ball_dataset.py:314`), replacing
`record['path'].stem`:

```
Bay-Club-1_mov-0042_jpg.rf.abc123_c0.jpg     ->  unchanged
Squash Rally ｜ Best_mov-9_jpg.rf.d_c0.jpg   ->  Squash_Rally_Best_mov-9_jpg.rf.d-cc74d589_c0.jpg
```

(Both lines computed with the reference implementation, not illustrative.)

### 2. Unicode-safe I/O helpers

`_imread_unicode(path)` — `np.fromfile` → `cv2.imdecode`. Returns `None` for a missing or
empty file, preserving today's `if frame is None: continue` at the call site.

`_imwrite_unicode(path, image, params)` — `cv2.imencode` → `ndarray.tofile`. **Raises**
on encode failure rather than returning a bool. Since the entire bug is that `imwrite`'s
`True` was a lie (D1), the replacement must not be capable of silently doing nothing.

Both do their cv2/numpy imports locally, keeping the module importable without cv2 — the
property that lets `tests/test_prepare_ball_dataset.py` run in a bare interpreter.

### 3. COCO / manifest metadata

The per-image `clip` field already carries the raw, human-readable clip name and is
untouched — it is the traceability anchor, and JSON is UTF-8 so it is safe there. One new
manifest key, `slugified_clips`, lists the clips whose names required mangling, so a hash
suffix in a filename explains itself without cross-referencing the COCO JSON.

### 4. Tests

Added to `tests/test_prepare_ball_dataset.py`, matching the file's existing conventions
(module-level import from `prepare_ball_dataset`, one behaviour per test, comments stating
*why* the case matters). Regenerating the dataset is impossible here — the source Roboflow
export `SquashAI.coco` is not on this machine — so the slug logic is validated over
representative names instead:

- ASCII Roboflow name passes through byte-identical, no hash suffix.
- U+FF5C fullwidth bar → `_`, hash appended (the actual production case).
- `café` → `cafe` (NFKD transliteration, not character loss).
- CJK / emoji-only stem → `clip-<hash>` fallback, never an empty filename.
- The CP437 mojibake form `∩╜£` (F1's on-disk artifact) also slugs safely.
- Two different originals that collapse to the same base stay distinct.
- Every output matches `^[A-Za-z0-9._-]+$`.
- Determinism: same input → same slug across calls.

Plus one `pytest.importorskip("cv2")`-guarded round-trip asserting `_imwrite_unicode` /
`_imread_unicode` survive a `｜` path — a regression test for D1 that runs on the training
box even though it skips here.

### 5. Documentation

`CLAUDE.md` gains a short Windows-training-box note: the `ball-detector-train` venv path,
and the rule that image I/O goes through the helpers while `VideoCapture`/`VideoWriter`
are fine as-is (with the reason, so the asymmetry does not read as an oversight).

## Verification plan and its limits

No interpreter on this machine has both pytest and cv2: `ball-detector-train\.venv` has
cv2 5.0.0 and numpy but no pytest; system Python 3.14.6 has neither. `CLAUDE.md`'s
documented `.venv` does not exist here at all, and the pytest PostToolUse hook it
describes is not configured in any `settings.json` in the repo or in `~/.claude`.

- Slug tests: run under a throwaway pytest venv in the session scratchpad (they need no
  cv2), leaving `ball-detector-train\.venv` untouched.
- cv2 round-trip: exercised as a standalone script under `ball-detector-train\.venv` for
  real evidence, and shipped as the `importorskip`-guarded test for future runs.
- The cv2-guarded test will report **skipped** in the pytest run. That is stated rather
  than counted as a pass.

## Out of scope

- Regenerating `ball-crops-2026-07-24` (source export absent from this machine).
- The YOLOX clone's `coco.py` patch — that lives in `ball-detector-train`, not this repo,
  and remains needed for any dataset whose *directory* path contains non-ASCII.
- Windows reserved device names (`CON`, `NUL`, `COM1`…). A Roboflow stem is always
  `<clip>_mov-NNNN_jpg.rf.<hash>`, so a bare reserved name cannot occur.
