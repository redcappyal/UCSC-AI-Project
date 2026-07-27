"""Automatic squash-court detection from a handful of frames.

Finds the court's lines in an image and derives every calibration landmark as
an intersection of two of them, producing exactly the structures the manual
wizard's buildJson() produces (schema squash-calibration-v2). Design spec:
docs/superpowers/specs/2026-07-27-auto-court-detection-design.md

Two invariants this module is built around:

  * Never key on hue. Court lines are navy at Bay Club and red at
    SquashAnalytics; the property that holds across both is "a thin stripe
    darker and/or more chromatic than its local surround" (spec §4.1).
  * Never solve 3-D pose. Both fits are plane homographies, which keeps this
    clear of the solve_camera_model chirality defect that rejects every real
    stored calibration.

Pure functions over numpy arrays: no Flask, no disk access, no globals.
"""

import cv2
import numpy as np

# --- temporal median -------------------------------------------------------
MOTION_DELTA = 18        # grey levels; below this a pixel counts as unchanged
MOTION_FRACTION = 0.25   # fraction of changed pixels that means "the camera moved"


def median_frame(frames):
    """Temporal median of same-size BGR frames, plus a camera-motion verdict.

    Two players moving through a static court change a few percent of the
    pixels and vanish into the median. A pan changes most of them and medians
    into a ghost, so the caller must fall back to a single frame — which is
    exactly what SquashAnalytics.mp4 does and the product's fin mount does not.
    """
    stack = np.stack([np.asarray(frame) for frame in frames])
    if stack.shape[0] < 2:
        return stack[0].copy(), False

    median = np.median(stack, axis=0).astype(np.uint8)
    reference = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY).astype(np.int16)
    changed = [
        float(np.mean(
            np.abs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
                   - reference) > MOTION_DELTA))
        for frame in stack
    ]
    return median, bool(float(np.mean(changed)) > MOTION_FRACTION)
