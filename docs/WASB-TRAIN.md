# WASB fine-tune: data → adapter → train → smoke → export

Runbook for fine-tuning a WASB-style (HRNet, MIMO heatmap) ball detector on
squash footage, replacing the single-frame YOLOX detector's blind spot at
direction reversals (wall/floor contacts) with a 3-frame temporal input. See
`ios/MODEL.md` for the YOLOX path this supplements, not replaces yet — YOLOX
stays the shipped detector until this clears the recall gate in
`eval_set/RESULTS-*.md` (Task 8).

**Everything in this document runs on the CUDA training box**
(`C:\Users\alann\Code\ball-detector-train\.venv`), except §4a, which runs on
the Mac in this repo. WASB-SBDT is cloned *inside* the training repo and never
becomes a runtime dependency of this repo — same rule as YOLOX
(`ios/MODEL.md` §0), for the same reason: this repo's serving code
(`ball_model.py`, `ball_detector.py`) loads a traced TorchScript artifact and
needs only `torch`, never the training framework.

Facts below are marked verified (checked against the WASB-SBDT source at the
pinned commit, 2026-07-27) or `TODO(verify on clone)` where the source didn't
settle it. Nothing here is a guess presented as fact.

---

## 0. The one finding that changes the shape of this runbook

**WASB-SBDT's own training path is disabled in the public repo.** `README`'s
`GET_STARTED.md` says, verbatim, under "Training": `TBA`. Checked why: in
`src/runners/__init__.py`, the `Trainer` import and its `'train'` registry
entry are commented out —

```python
# from .train_and_test import Trainer
...
__runner_factory = {
    #'train': Trainer,
    'eval': VideosInferenceRunner,
    'extract_frame': ExtractFrameRunner,
}
```

The `Trainer` class itself (`src/runners/train_and_test.py`) still exists and
is basically complete, but it hard-fails if `runner.test.run` is true
(`assert 0, 'not yet (2023.4.23)'`) and calls `test_epoch(0, model,
test_loader, ...)` in a code path that references `model`/`test_loader`
without ever binding them (should be `self._model`/`self._test_loader`) — a
live bug in code the authors never finished wiring up. There's also no
`configs/runner/train.yaml` and no dataset config for anything but the five
shipped sports, so getting `Trainer` running through Hydra means writing a
runner config, fixing the bug above, and registering a `squash` dataset the
same way `datasets/tennis.py` etc. do.

**Decision: don't resurrect it.** Fighting an unmaintained, half-wired Hydra
runner buys nothing — we don't need Hydra's multi-sport config tree, only
three things from the clone:

1. `models/hrnet.py` — the `HRNet` class (the architecture).
2. `losses/heatmap.py` + `losses/wbce.py` (its default sub-loss) — the loss.
3. `pretrained_weights/wasb_*.pth.tar` — the checkpoint.

Everything else (§3) is a plain PyTorch training loop written against those
three imports, fed by `wasb_crops_dataset.py` (§2). This is the same shape as
this repo's own YOLOX integration boundary (`export_ball_model.py` imports
`yolox` only in the training env, never at serving time) — here we go one
step further and don't even import WASB-SBDT's own trainer, only its model
and loss modules.

---

## 1. Setup

**Clone at a pinned commit** (main HEAD as of 2026-07-27; re-verify this is
still HEAD before cloning — the repo is low-traffic, 6 commits total, last
pushed 2023-11-23, so drift is unlikely but check):

```
git clone https://github.com/nttcom/WASB-SBDT
cd WASB-SBDT
git checkout 923462cacdeb3353b84ddebdedb3f4b7a8553b0f
```

MIT license (`LICENSE.md`) — same posture as YOLOX's Apache-2.0, fine to
depend on for training-only tooling.

**Dependencies — do not follow the repo's `Dockerfile` pins.** It specifies
`torch==1.11.0+cu113` (2021-era, base image `nvcr.io/nvidia/tensorrt:21.06-py3`,
Python 3.8) and `hydra-core==1.2.0`. This is the exact problem `ios/MODEL.md`
§2 documents for YOLOX on the RTX 5060: pre-cu128 wheels ship no sm_120
kernels, so `torch==1.11.0+cu113` will not run on this box at all. Since §0
means we don't use Hydra or most of the repo's own plumbing, `hydra-core`
isn't needed either. Install into the existing `.venv`
(`C:\Users\alann\Code\ball-detector-train\.venv`, already has a working
cu128 torch per `ios/MODEL.md`) — no new environment needed.

**Verified on clone (2026-07-28), with three corrections to the above:**

1. **Configs live at `src/configs/`, not `configs/`.** Every config path in this
   document (`configs/model/wasb.yaml`, `configs/loss/hm_wbce.yaml`,
   `configs/dataloader/default.yaml`) is relative to `src/`. The repo root holds
   only `src/` and whatever you create beside it. Confirmed absent:
   `src/configs/runner/train.yaml` — §0's "training is unwired" finding holds.
2. **`cfg` is not a plain dict.** `models/hrnet.py` imports cleanly with only
   `torch`, but `HRNet(cfg)` *fails* on a `dict`: `__init__` subscripts
   (`cfg['MODEL']['EXTRA']`) while `_make_deconv_layers` uses attribute access
   (`cfg.MODEL.EXTRA`, `hrnet.py:334`). It is reached even with
   `DECONV.NUM_DECONVS: 0`, because the attribute read happens before the loop.
   Pass an `OmegaConf.load(...)` object — `pip install omegaconf` (standalone;
   still no `hydra-core`).
3. **`losses/` needs `pandas`.** `from losses.heatmap import HeatmapLoss` runs
   `losses/__init__.py`, which pulls the whole loss zoo and transitively
   `utils/utils.py` → `pandas`. Fails `ModuleNotFoundError` without it.

Full extra install on the box: `pip install gdown pandas omegaconf pyyaml`.

**Pretrained checkpoints.** Tennis and badminton are the closest regimes to
squash: both are small-fast-ball racquet sports shot at broadcast-ish scale,
versus soccer/volleyball/basketball's larger, slower balls. From
`MODEL_ZOO.md` / `setup_scripts/setup_weights.sh` (verified — these are the
repo's actual Google Drive file IDs, not invented):

```
pip install gdown   # the repo's own `wget https://drive.google.com/uc?id=...`
                     # trips Google Drive's virus-scan interstitial on files
                     # this size; gdown handles it. TODO(verify on clone):
                     # confirm gdown still pulls the real file, not the
                     # warning page, for these two IDs.
mkdir pretrained_weights
gdown 14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z -O pretrained_weights/wasb_tennis_best.pth.tar
gdown 17Ac0pO5oryh1JwgwTFQTjOKHY3umbDQu -O pretrained_weights/wasb_badminton_best.pth.tar
```

Despite the `.tar` extension these are plain `torch.save` objects, not literal
tar archives (verified: `utils/utils.py:save_checkpoint` is `torch.save(state,
model_path)`). The state dict is under the key `'model_state_dict'`, and the
same key is what `detectors/detector.py` reads back
(`checkpoint['model_state_dict']`) — this is the load key to use in §3, and
it's consistent between how the repo trains and how the repo itself loads a
checkpoint, which is the best evidence available that the released
`wasb_tennis_best.pth.tar` follows this format too.

**Verified on clone (2026-07-28)**: `wasb_tennis_best.pth.tar` is 6,102,633
bytes (6.1 MB — small, it is a 1.48 M-param network), `torch.load` yields a
`dict` whose *only* top-level key is `model_state_dict`, holding 428 entries.
So `torch.load(ckpt)['model_state_dict']` is correct with no fallback needed.
The stem is `conv1.weight` of shape `(64, 9, 3, 3)` — **9 input channels
already**, confirming §2.1: `frames_in=3` × BGR needs no conv surgery.

---

## 2. Dataset adapter contract (`wasb_crops_dataset.py`, lives in the training repo)

This script does not exist yet — this section is its contract, to be
implemented against `ball-crops-v2` output from `prepare_ball_dataset.py`
(this repo, already merged). It is *not* one of the five dataset classes in
WASB-SBDT's own `src/datasets/` (those are wired to Hydra configs we're not
using per §0) — it's a plain `torch.utils.data.Dataset` written from scratch
against the COCO layout `prepare_ball_dataset.py` already emits.

### 2.1 What one sample is

Each sample is one anchor annotation's 3-frame sequence, oldest-first:

- **Files**: for anchor image `<stem>.jpg` with a `"sequence"` field on its
  COCO image entry, load `<stem>.tm1.jpg`, `<stem>.jpg`, `<stem>.tp1.jpg` —
  in that order (`sequence: [tm1, anchor, tp1]`, verified in
  `prepare_ball_dataset.py` around `render_split`/`sequence_names`).
- **Pixel format**: BGR (`cv2.imread` default), raw `0–255` float32, **no
  mean/std normalization, no channel swap** — matches the convention
  `HeatmapRunner` in this repo's `ball_model.py` documents for serving
  (`"No mean/std, consistent with TorchScriptRunner's BGR-raw convention"`).
  Training and serving must agree on this or the fine-tune learns a
  distribution the traced model never sees.
- **Concatenation**: stack the 3 frames along the channel axis, oldest first
  → 9 channels total (`[tm1_B, tm1_G, tm1_R, t_B, t_G, t_R, tp1_B, tp1_G,
  tp1_R]`). This is the same ordering `detect_frame_stack` in
  `ball_detector.py` builds at serving time
  (`np.concatenate([...frames], axis=2)` over an oldest-first `frames`
  list) and the ordering `HRNet.__init__` expects
  (`conv1 = nn.Conv2d(3*frames_in, ...)`, verified in `models/hrnet.py` — with
  `frames_in=3` this is already a 9-channel stem, so **no conv-surgery is
  needed to adapt the pretrained stem** — confirms the brief's claim).

### 2.2 Targets

One Gaussian heatmap per input frame (3 target channels, matching the MIMO
output `[frames_out=3, Hh, Wh]`), at `heatmap_stride` resolution, σ = 2.0
heatmap px. The forward transform placing a Gaussian is the exact inverse of
`ball_model.decode_heatmap`'s `k * stride` peak-to-pixel scaling: heatmap
coords `x / stride, y / stride` from the pixel-space label, with no
half-pixel center-alignment offset. (WASB's own default is σ = 2.5, from
`configs/dataloader/default.yaml: heatmap.sigmas: [2.5]` — 2.0 is this
project's deliberate choice for a ~7–24 px ball, tighter than WASB's broadcast
targets; if you change `heatmap_stride` from whatever the exported config
ends up at, re-derive whether 2.0 heatmap-px is still an appropriate fraction
of the ball's size.)

**Only the middle frame `t` has a real label.** The three target channels are
built as:

| frame | has a real label? | target |
|---|---|---|
| `t` (middle) | always | Gaussian at the true center |
| `t-1`, `t+1`, from an **edge-padded** sequence (repeated first/last frame — `prepare_ball_dataset.py` pads clip edges by repeating) | no (it's a duplicate of the anchor frame) | Gaussian at the **anchor's** center — the neighbor pixels are literally the anchor frame, so its ball is really there too |
| `t-1`, `t+1`, from **real, distinct** footage | no (never labeled) | **masked out of the loss**, not an empty heatmap |
| any frame of a **negative** crop (no ball anywhere near this window) | — | empty heatmap on **all three** channels |

The mask distinction matters: an *unlabeled* neighbor is not evidence the ball
is absent there — for a real (non-padded) `t-1`/`t+1` we genuinely don't know,
because `prepare_ball_dataset.py` only carries annotations for the labeled
frame. Training against an empty target for those frames would actively
teach "no ball, even when there might be one" — the exact failure mode
`ios/MODEL.md` §1 documents for the old motion-blur augmentation ("erased the
ball ... while leaving the label attached"). Implement the mask as a per-frame
boolean fed into the loss (zero out that frame's contribution), not as an
empty target.

### 2.3 Hard negatives are mandatory, not a stretch goal

Wall and floor scuff marks are YOLOX's dominant false-positive class today
(`docs/annotation-guide.md`: "Squash walls are covered in ball marks - dark
smudges the exact size and colour of the ball ... This is the single biggest
false-positive source in this domain"). The whole
reason to spend a temporal channel is to teach the network that a *static*
mark, present identically in all 3 frames, is not the ball — but that lesson
only transfers if training actually shows static ball-lookalikes labeled
empty. `prepare_ball_dataset.py`'s existing `--negatives-per-frame` only
samples random ball-free windows near labeled frames; it has no concept of
"this window is where the current detector false-fires" or "this window is a
wall mark a player is about to walk away from." `wasb_crops_dataset.py` (or a
preprocessing pass ahead of it) needs to mine two specific negative sources,
both windows with **empty heatmap targets on all 3 channels**:

- **(a) RF-DETR false-fire locations on eval clips.** Run `local_model_eval.py`
  (this repo) over the eval clips, diff its predictions against the labeled
  ground truth, and keep detections with no matching annotation nearby — those
  pixel locations are exactly the marks the current detector already
  mistakes for a ball. There is no packaged script that does this diff today;
  it's new work for whoever builds `wasb_crops_dataset.py`. `TODO(verify on
  clone)`: confirm `local_model_eval.py`'s CSV output carries enough
  per-frame (x, y, confidence) detail to do this diff directly, or whether it
  needs a small companion script.
- **(b) Player-adjacent wall regions**, specifically the revealed-background
  ghosting case: a wall mark that looks like it "appears" as the player
  walks off it, because the region was occluded in the previous frame(s) and
  isn't in this one — that's real motion energy at the mark's location, which
  is exactly the pattern that could fool a temporal-difference-sensitive
  network if it's never shown as a negative. Mine crop windows centered on
  wall regions adjacent to player bounding boxes/silhouettes across a few
  consecutive frames of eval footage.

Both sources need to appear in training with empty heatmap targets on all
three channels (they are not "no ball in the middle frame" cases with a
label elsewhere — there's no ball in the window at all).

---

## 3. Fine-tune recipe

- **Init from the tennis checkpoint** (`wasb_tennis_best.pth.tar`, §1) via
  `model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])`. If the
  key errors, fall back to inspecting the raw dict — see the `TODO` in §1.
- **Input size 416×416**, not WASB's native `288×512` (`configs/model/wasb.yaml`,
  verified — a 16:9-ish broadcast frame size, non-square). Change this to
  match the crop tile `prepare_ball_dataset.py --crop 416` (default) produces.
  This is a pure size change, not a code change: `HRNet.forward` is fully
  convolutional (stem → 4 parallel-branch HRNet stages → final 1×1 conv per
  scale, no `Linear`/flatten anywhere — verified by reading the forward
  method), so a change in input H×W requires **no change to any operator's
  stride or shape** — every conv in the network is defined independent of
  spatial size. The only thing that changes is the config's declared
  bookkeeping: `inp_height`/`inp_width`/`out_height`/`out_width` in
  `configs/model/wasb.yaml` go from `288`/`512` to `416`/`416`. Pretrained
  weights transfer unchanged — there is nothing in the checkpoint that is
  shaped by the old input size.
- **Output stride**: WASB's shipped config has `STEM.STRIDES: [1,1]` and
  `DECONV.NUM_DECONVS: 0`, which — verified by reading `HRNet.forward`, which
  emits `y_list[scale]` straight from the last HRNet stage with no
  downsampling or upsampling op in between — means the *default* config's
  heatmap output is **stride 1** (same resolution as the input: `out_height
  == inp_height`, `out_width == inp_width` in `configs/model/wasb.yaml`,
  consistent with stride 1). If you keep this config as-is, `heatmap_stride`
  for the exported manifest (Task 7) is `1`, not the `4` a typical heatmap
  detector uses — don't assume the conventional value. Whatever the actual
  trained config ends up being, derive `heatmap_stride` from the real output
  tensor shape at export time (`input_H / output_H`), never hardcode it.

  **Measured on the box (2026-07-28)**, building `HRNet` from the shipped
  `wasb.yaml` with only `inp_/out_height` and `inp_/out_width` set to 416 and
  loading the tennis checkpoint: `load_state_dict` reports **0 missing and 0
  unexpected keys**, and a `(1, 9, 416, 416)` input yields `(1, 3, 416, 416)`
  — so **`heatmap_stride` is confirmed 1** (full-resolution heatmap) and the
  416 change really is pure bookkeeping. Network is 1,481,427 params.
- **Batch size on the 8 GB RTX 5060: use 4.** Measured peak allocation for a
  full forward+backward+step at 416×416: batch 2 → 2.36 GB, batch 4 → 4.67 GB,
  batch 8 → 9.32 GB, batch 16 → `OutOfMemoryError`. The card has ~8.55 GB, so
  batch 8's 9.32 GB only "succeeds" by spilling into WDDM shared host memory
  and will run far slower than its step count suggests — treat 8 as unusable,
  not as a working option. The memory cost is activations, not weights: a
  stride-1 network keeps full 416×416 feature maps through every stage, which
  is why a 1.5 M-param model needs more VRAM than the 5 M-param YOLOX-Tiny.
- **Loss**: `losses/heatmap.py`'s `HeatmapLoss` wrapping `losses/wbce.py`'s
  `WBCELoss` (`sub_name: wbce`) — this is the loss WASB's own model config
  pairs with in the shipped `hm_wbce.yaml`, i.e. the setup its released
  weights were actually optimized under. It's weighted BCE over
  `sigmoid(logits)` against the Gaussian targets from §2.2; masked frames
  (§2.2's `t±1` unlabeled case) must be excluded from the loss sum, not just
  zeroed in the target, or the network is still penalized for the wrong
  answer on channels we don't have ground truth for.
- **Freeze nothing.** The training set is small (per `ios/MODEL.md` §1's
  experience with a similarly-scoped squash dataset: 66 independent moments
  informed the ~1k-annotation YOLOX set) but the domain gap between broadcast
  tennis/badminton and a fin-mounted squash court is real enough that a
  frozen backbone likely can't close it. Fine-tune the whole network.
- **Early stop on val heatmap F1** (peak-detection precision/recall against
  the val split's labeled centers, at the same `conf_threshold` the export
  will use), not on the loss curve. `ios/MODEL.md`'s YOLOX run is the cautionary
  precedent: val AP peaked at epoch 100 of 300 and 200 more epochs bought
  nothing measurable — expect the same shape here given a comparably small
  dataset, and don't burn box time past the point val stops improving.

---

## 4. Commands

### 4a. Mac: regenerate crops with sequences

Real flags, checked against `prepare_ball_dataset.py`'s `parse_args()` (all
five below exist verbatim; `--seq-frames` and `--clips-dir` are the ones this
branch added — `--seq-frames` must be odd and requires `--clips-dir` or the
script exits with `SystemExit`):

**The source clips are gone (established 2026-07-28).** Nothing had needed them
before: the 2026-07-24 build was single-frame, and `--clips-dir` is only
required for `--seq-frames > 1`. A Spotlight sweep of the Mac turned up only the
export JPEGs. So the neighbour frames come from the export itself, via
`--seq-from-export`:

```bash
.venv/bin/python prepare_ball_dataset.py \
    --source "~/Desktop/Annotated Data/SquashAI.coco" \
    --seq-frames 3 \
    --seq-from-export \
    --out ball-crops-seq-v1 \
    --val-clips "ModelTrainTest3" \
    --test-clips "Bay Club Clip Compilation 1" \
    --exclude-clips "Squash ｜ 5 Easy ways to INSTANTLY IMPROVE Your Game ｜ For Beginners -RhVGSsfFmGg-_f399"
```

**Why this works at all.** Roboflow exports frames, and labelled frames arrive
in contiguous runs, so an anchor inside a run already has its `t±1` on disk as
plain export images. Measured on `ball-crops-2026-07-24`: both neighbours
present for **92.2%** of `ModelTrainTest2`'s anchors, 44.3% of
`ModelTrainTest3`'s, 64.1% of `Bay Club`'s. Anchors at a run boundary keep the
side they have (repeat-anchor, the same convention the video path uses at a clip
edge) and are dropped only when neither side exists.

Three consequences, all favourable:

- **No alignment probe and no `--frame-map`.** Both exist to translate export
  indices into video frames; this path never leaves export space, so the
  `Bay Club` variable-frame-rate problem in the note below simply does not
  arise. Passing `--frame-map` with `--seq-from-export` is rejected.
- **A cut cannot be sampled.** Shot boundaries fall at run boundaries, so the
  cross-shot frame is unreachable by construction rather than by threshold.
- **Neighbours often carry real labels**, unlike decoded video frames. Where
  they do, `wasb_crops_dataset.py` can use a true Gaussian on that channel
  instead of §2.2's mask — strictly more supervision than the video path
  offers. Confirm the labelled fraction against the real export before relying
  on it.

The cost: a clip whose export was **not** sampled 1:1 with video frames gets a
wider temporal baseline than the model sees at serving time — adjacent export
frames are more than one video frame apart. That is `Bay Club Clip Compilation
1` (export index 96 = video frame 163, 290 = 516), and it is the *test* split,
so it costs eval fidelity rather than training correctness. `ModelTrainTest2/3`
sampled 1:1 and are clean. Re-check this if `Bay Club` ever becomes a train
clip.

If the footage ever turns up, drop `--seq-from-export` and pass `--clips-dir`
instead (the two are mutually exclusive): the video path restores the run-
boundary anchors, which would roughly double the val split from 43 to 97
anchors, at the price of reinstating the alignment probe and the `--frame-map`
work.

**Why the tutorial clip is excluded (decided 2026-07-28).** Its labelled frames
are all court-view and perfectly good, but the surrounding video is an edited
YouTube tutorial — cuts, text cards, talking head. Labelling therefore stops at
every shot boundary, which puts a cut one frame from the last anchor of each
burst, and `--seq-frames 3` pulls `t±1` straight from the video. Measured burst
structure makes the difference stark: `ModelTrainTest2` is 412 labelled frames
in **16 bursts** (avg 26 consecutive), while the tutorial is 175 frames in
**51 bursts** (avg 3.4) — 53% of its anchors sit at a burst edge versus 7.8%
for `ModelTrainTest2`. A cross-shot neighbour is worse than a missing one: §2.2
masks the *target* for an unlabelled `t±1`, so the label side is safe, but the
*input* stack still shows maximal inter-frame change co-occurring with a ball
in the middle channel — training the exact inverse of §5(d)'s static-clutter
lesson. Cost of excluding it: 961 of 2,936 train crops (33%).

The two `｜` in that name are U+FF5C (fullwidth vertical line), not ASCII
pipes — copy it, don't retype it.

**Then read `splits.*.sequence_continuity` in the output manifest** before
trusting the remaining clips. `prepare_ball_dataset.py` measures
anchor-vs-neighbour mean|pixel diff| for every sequence it builds and reports
per-clip `p50/p90/p99/max`. A flat spread means continuous footage and needs
nothing; a long tail means shot boundaries next to labels, and
`--seq-cut-threshold <N>` then replaces those neighbours with a repeated anchor
(dropping the anchor entirely when neither side is usable, so three identical
frames never carry a positive label). Pick `N` from the clip's own
distribution — there is no good default, and a guessed one is worse than none.
**`Bay Club Clip Compilation 1` is the one to check**: "Compilation" implies
concatenated footage, and it is the test split, so a cut there corrupts the
eval metric rather than the training set.

Use the **same val/test clip split** as the last single-frame crop build
(`ios/MODEL.md` §1's `ModelTrainTest3` / `Bay Club Clip Compilation 1` shown
above as the precedent) — a different split makes any F1 comparison between
this model and the YOLOX baseline meaningless. Output `manifest.json` will
carry `"schema_version": "ball-crops-v2"` once any image has a `"sequence"`
field (verified: `SCHEMA_VERSION_SEQ` is set exactly when `seq_frames > 1`).

**If a clip fails the 0/±1 offset alignment** ("no single frame offset ...
aligns every export image"), its Roboflow upload was not sampled 1:1 with
video frames — measured on `Bay Club Clip Compilation 1.mov` (60 fps
variable-frame-rate iPhone footage: export index 96 is video frame 163, 290
is 516, non-uniformly). For such clips, pass `--frame-map <json>` mapping
each export index to its true video frame:
`{clip: {"video_frame_count": N, "frames": {"<export_index>": video_frame}}}`.
Build the map by brute-force matching every export image against every
decoded frame at thumbnail size (argmin of mean |diff|), and only trust it
when every match is unambiguous (clear margin over the best frame outside
the ±2 neighbourhood) and the mapping is strictly increasing in export
order. Mapped anchors are still alignment-verified per frame under the same
tolerance — the map changes where to look, never whether alignment is
checked.

### 4b. Box: environment + checkpoint (see §1 for the reasoning)

```
git clone https://github.com/nttcom/WASB-SBDT
cd WASB-SBDT && git checkout 923462cacdeb3353b84ddebdedb3f4b7a8553b0f
pip install gdown
mkdir pretrained_weights
gdown 14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z -O pretrained_weights/wasb_tennis_best.pth.tar
```

### 4c. Box: training

There is no ready-made training entry point (§0) — `train_wasb.py` is new
code to write in `C:\Users\alann\Code\ball-detector-train`, alongside
`wasb_crops_dataset.py` (§2), importing only `WASB-SBDT/src/models/hrnet.py`
and `WASB-SBDT/src/losses/`. Shape it around this contract:

```
.venv\Scripts\python train_wasb.py ^
    --data ball-crops-seq-v1 ^
    --init-checkpoint WASB-SBDT\pretrained_weights\wasb_tennis_best.pth.tar ^
    --input-size 416 ^
    --sigma 2.0 ^
    --out wasb_runs\crosscourt-ball-wasb-v1
```

Flag names above are illustrative, not a fixed CLI this doc mandates — match
whatever `wasb_crops_dataset.py` and `train_wasb.py` actually expose, but
keep the recipe in §3 (init checkpoint, input size, loss, no freezing, F1
early stop) as the acceptance contract regardless of exact flag spelling.

---

## 5. Smoke gates before a full run

Run these in order; each is cheap and each catches a different class of
silent failure. Don't start a multi-hour run past a failing gate.

**(a) One-batch overfit.** Take a single batch (a handful of positive samples
is enough), train on only that batch, and confirm loss drops to near-zero in
under 200 steps. If it doesn't, the model/loss/optimizer wiring is broken
before data quality is even a variable.

**(b) 16-sample val contact sheet.** Run the model over 16 val samples,
render predicted heatmaps as an image grid, and eyeball that the peak sits on
the ball in each. This is the same "looks locked-on" bar `ios/MODEL.md` §3
uses for the YOLOX acceptance gate — a visual check, not a substitute for the
numeric eval in Task 8.

**(c) Channel-order round-trip.** Construct (or find) a sequence where the
ball is visible only in the middle frame `t` — absent or clearly at a
different position in `t-1`/`t+1`. Run it through the model and confirm the
**middle** output channel (`output[..., 1, :, :]` for a 3-channel MIMO
output) responds and the outer two do not. If an outer channel fires instead,
the oldest-first concatenation order is inconsistent between training and
this check (or between training and serving — cross-check against
`detect_frame_stack`'s ordering in `ball_detector.py`), and every downstream
result is silently mislabeled.

**(d) Static-clutter check.** Build a sequence of 3 *identical* frames (a
literal repeat, simulating a static wall mark under zero motion) containing a
ball-sized dark mark and nothing else. Run it through the model and confirm
the **middle** heatmap is near-empty — this is the exact case the temporal
input exists to suppress, and if the network still fires on it, the hard-negative
mining in §2.3 hasn't taught it anything yet (check that those negatives
actually made it into this training run's data, not just into the crop
directory).

---

## 6. Export (Task 7, not this document)

Once a checkpoint passes §5 and clears whatever val F1 bar Task 8's baseline
sets, `export_wasb_model.py` (new script, not yet written) traces it to
TorchScript and writes a `ball-model-v2` manifest beside it — mirroring
`export_ball_model.py` for YOLOX (`ios/MODEL.md` §2b). The fields below are
the **delta** this v2 schema adds on top of what a manifest already carries
today (`name`, `version`, `nms_iou`, `class_names`, `tile_overlap_px`,
`max_batch_tiles`, `artifact_sha256`, ... — all still required; `ball_model.py`'s
`ModelManifest` loader raises `KeyError` on any missing field, v1 or v2, so
`export_wasb_model.py` must populate the full set, not just the new ones
listed here):

**Output-activation contract**: the checkpoint is logits-native (§3's loss is
BCE over `sigmoid(logits)`), but the **traced graph the manifest points at
must emit probabilities**, not logits — `export_wasb_model.py` traces
`_SigmoidWrapper(model)` (sigmoid folded into the graph at export time), and
asserts the traced output lands in `[0, 1]` before writing anything to disk.
Nothing about training changes: the checkpoint itself, and `train_wasb.py`'s
loss, stay logits-native throughout §3. Only the exported TorchScript artifact
differs from the checkpoint's raw forward pass.

- `schema_version`: `"ball-model-v2"`
- `input_size`: `[416, 416]`
- `frames_per_input`: `3` (must be odd — `ball_model.py` raises otherwise)
- `decode`: `"heatmap_peak"`
- `heatmap_stride`: derived from the trained model's actual output shape
  (§3 — do not assume 1 or 4, measure it)
- `nominal_ball_px`: sizes the reported detection box (see
  `ball_model.py:HeatmapRunner._decode_output`)
- `conf_threshold`: `0.1` — deliberately low, not a placeholder. Per the
  implementation plan
  (`docs/superpowers/plans/2026-07-27-wasb-temporal-ball-detector.md`, Task 7),
  this is a BYTE-style low-confidence rescue adapted to a single fast, tiny
  object: weak-but-real heatmap peaks are allowed through the manifest gate
  so they reach `tracking_common`'s motion-consistency scorer, which promotes
  moving candidates (+0.30) and suppresses stationary ones (−0.40) — the
  caller's actual selection bar stays at confidence 0.4
  (`detections_to_track_samples`), so a low manifest floor doesn't mean a low
  final acceptance bar. This deliberately differs from the YOLOX export's
  `--conf-threshold` default of `0.25` (`export_ball_model.py`), because YOLOX
  has no equivalent downstream motion rescue to lean on — its manifest
  threshold has to do the precision work alone. Still tunable per export via
  `export_wasb_model.py --conf-threshold`.

`BALL_MODEL_DIR` (env var, `ball_model.py`) then points the Flask pipeline at
the exported directory the same way it does for the YOLOX artifact today.
