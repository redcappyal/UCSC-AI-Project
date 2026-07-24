# Ball annotation guide

How to label squash footage for the CrossCourt line-calling detector, and why each
rule exists. Every number here was measured from the current dataset
(`SquashAI.coco`, 1,367 frames / 1,103 annotations) on 2026-07-24.

The detector has one job: find a ball that is 4-25 px wide in a 4K frame. At that
size the model is working near the limit of the available evidence, so labeling
mistakes that would be harmless on a car or a person are fatal here.

## 1. The rule that matters most: never label consecutive frames

This is the single biggest thing to change. An audit of the current dataset:

| Clip | Annotations | Median frame gap | Consecutive | Independent moments |
| --- | --- | --- | --- | --- |
| ModelTrainTest2 | 309 | 1 | 95% | 11 |
| Bay Club Compilation | 276 | 1 | 80% | 14 |
| SQUASHCLIP | 167 | 1 | 89% | 7 |
| YouTube clip | 161 | 1 | 67% | 33 |
| ModelTrainTest3 | 97 | 1 | 69% | 17 |
| IMG_3210 | 93 | 1 | 92% | 5 |
| **Total** | **1,103** | | | **87** |

At 60 fps, frame *n* and frame *n+1* share the same lighting, the same background,
the same player positions, and a ball that has moved a few pixels. The network
treats them as one example. **1,103 annotations are carrying roughly 87 samples'
worth of information.**

The same labeling effort spread one frame per 30 would have produced ~1,100
distinct moments instead of 87 - more than ten times the information for
identical cost.

**Rule: minimum 30 frames between labels within a clip (0.5 s at 60 fps).**

The only exception is the end-to-end eval set, where `eval_line_calls.py` needs
contiguous frames around an impact to score an IN/OUT call. Keep those as short
deliberate bursts, and keep them out of detector training data.

## 2. Label footage from the production rig

Camera and mount for anything destined for training or eval:

| Setting | Value |
| --- | --- |
| Resolution / rate | 4K, 60 fps, HEVC |
| Shutter | 1/1000 s |
| ISO | Locked |
| White balance | Locked, set per court |
| Stabilization | OFF |
| Lens | Ultrawide |
| Mount | Top of back wall glass fins |

Footage that does not match this is **pretraining data only**. It teaches the
generic "small dark sphere against a court" feature, which transfers, but it
cannot tell you how the product will behave.

**The eval set must be 100% from the production rig, and frozen.** You cannot
improve what you cannot measure, and a score measured on off-rig footage predicts
nothing. Aim for at least 3 different courts and 6 different sessions so you can
report per-court variance rather than a single number that hides it.

**Split by clip or session, never by frame.** Because neighbouring frames are
near-identical, a frame-level split puts copies of the same moment on both sides
and reports a score that has nothing to do with generalisation.

## 3. Drawing the label

- **Use a box for most annotations.** At the measured median aspect ratio of
  1.18 the ball is essentially round, and a polygon adds nothing over a box while
  taking several times longer to draw around a 9 px object.
- **Use a polygon only when the streak is visibly elongated** - roughly the 8%
  of annotations above aspect 2.0. There the major axis is real information: its
  two endpoints are where the ball was at the start and end of the exposure, which
  gives the tracker two timestamped positions from a single frame.
- **Draw tight to the ball**, and include the *whole* streak when there is one.
  Do not trim a streak back to a round blob - the extent is the measurement.
- **Label every ball you can actually see**, including partially occluded ones.
- **If you cannot see the ball, mark the frame as a negative.** Do not infer its
  position from the trajectory and place a box where it "must" be. A box on a
  patch of empty wall teaches the model to fire on empty wall - this is exactly
  what the old motion-blur augmentation was doing, and it cost roughly two thirds
  of the model's recall.
- **Never leave a frame unlabeled if a ball is visible.** An unlabeled visible
  ball is scored as a false positive during training, which actively teaches the
  model to suppress correct detections.

## 4. Negatives: budget 25-30% of your labeling time

Squash walls are covered in **ball marks** - dark smudges the exact size and
colour of the ball, sitting on white plaster, hundreds per court. A detector
trained to find "small dark blob on a light wall" will fire on every one of them.
This is the single biggest false-positive source in this domain, and the only fix
is showing the model a lot of them with no box attached.

Deliberately capture ball-free frames containing:

- Wall ball marks, especially dense clusters near the front wall
- Shoe soles and dark trainers
- Dark logos, sponsor boards, the tin
- Court line junctions and corners
- Shadows under players
- The ball held in a hand or pocket before a serve

## 5. Per-session diversity checklist

Spend the budget on variety, not volume. Tick each of these per session:

- [ ] **Court** - different wall colour, lighting, logos, floor tone
- [ ] **Ball depth** - front wall (smallest, hardest, and the one you must call),
      mid court, and near camera. Over-sample front-wall balls relative to how
      often they occur.
- [ ] **Background** - white wall, glass back wall, floor, tin, across a player's body
- [ ] **Ball condition** - fresh and scuffed
- [ ] **Occlusion** - partially hidden by a player or racket
- [ ] **Speed** - hard drives and soft lobs
- [ ] **Negatives** - 25-30% of frames, per section 4

## 6. Never do

- Do not label runs of consecutive frames
- Do not place a box where you believe the ball is but cannot see it
- Do not skip a frame that has a visible ball
- Do not label footage below ~960 px wide - the ball is 3-4 px and there is no
  signal to learn
- Do not split train/val/test by frame
- Do not apply preprocessing or augmentation in Roboflow (see section 7)

## 7. Roboflow export settings

Generate every version with:

**Preprocessing:** Auto-Orient only. **No resize.** Never "Stretch to" - from 4K
that is 7.5x horizontally against 4.2x vertically, which hands the network a round
ball deformed into an ellipse.

**Augmentation:** none, and versions per training image set to 1. All augmentation
belongs in the training loop where its magnitude can be scaled against the ball
size.

**Include unannotated images:** unchecked.

**Export format:** COCO JSON - it carries real `bbox` fields alongside
`segmentation`, so nothing has to infer boxes from polygons.

This is not a stylistic preference. The deployed v4 model was trained with
`resize 512x512 "Stretch to"` and `motion-blur: 100 pixels` on a canvas where the
median ball is 8.7 px. The blur erased the ball in about one of every three
training copies while leaving the label attached.

## 8. Where to spend the next annotation budget

In priority order:

1. **~400 sparse frames for a frozen eval set** from the production rig, across
   at least 3 courts and 6 sessions. Never train on these. Highest value per
   label of anything on this list, because nothing else can be measured without it.
2. **~1,200 sparse frames for fine-tuning**, from the rig, from *different*
   sessions than the eval set.
3. **Active-learning rounds of ~300.** After the first in-domain model exists,
   label where it fails rather than sampling randomly. Actively selected frames
   are worth several times their number in random ones.

## Quick reference

| | |
| --- | --- |
| Minimum gap between labels | 30 frames (0.5 s at 60 fps) |
| Box or polygon | Box, unless the streak is visibly elongated |
| Ball visible but hard | Label it |
| Ball not visible | Negative - never a guessed box |
| Negative budget | 25-30% of frames |
| Minimum source width | 960 px |
| Split unit | Clip or session, never frame |
| Roboflow preprocessing | Auto-orient only |
| Roboflow augmentation | None |
| Export format | COCO JSON |
