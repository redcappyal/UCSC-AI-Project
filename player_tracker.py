"""Two-player tracking over person detections.

Anonymous tracks "A"/"B" — naming is a post-hoc relabel (spec §4.4). Pure
logic: no cv2/torch/rfdetr imports; PersonFramePass only counts frames and
delegates to the injected detector.
"""

from dataclasses import dataclass
import math

COAST_MAX_S = 1.0        # coast a dropped track at its last position this long
AMBIGUITY_MARGIN = 0.2   # pairings within 20% total cost are ambiguous
PERSON_DETECT_HZ = 4.0   # target person-detection cadence in video seconds


@dataclass
class TrackSample:
    t_s: float
    frame_idx: int
    foot_px: tuple
    bbox: tuple            # (x_center, y_center, width, height)
    confidence: float
    coasted: bool


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _Track:
    def __init__(self):
        self.samples = []
        self.last_live_t = None
        self.last_foot = None
        self.last_bbox = None

    def add_live(self, t_s, frame_idx, detection):
        foot = detection.foot_px
        bbox = (detection.x, detection.y, detection.width, detection.height)
        self.samples.append(TrackSample(t_s, frame_idx, foot, bbox,
                                        detection.confidence, coasted=False))
        self.last_live_t = t_s
        self.last_foot = foot
        self.last_bbox = bbox

    def add_coast(self, t_s, frame_idx):
        if self.last_live_t is None:
            return
        if t_s - self.last_live_t > COAST_MAX_S:
            return
        self.samples.append(TrackSample(t_s, frame_idx, self.last_foot,
                                        self.last_bbox, 0.0, coasted=True))


class TwoPlayerTracker:
    def __init__(self):
        self._tracks = {"A": _Track(), "B": _Track()}
        self._ambiguity_times = []
        self._updates = 0

    def update(self, t_s, frame_idx, detections):
        self._updates += 1
        top2 = sorted(detections, key=lambda d: -d.confidence)[:2]
        a, b = self._tracks["A"], self._tracks["B"]

        if a.last_foot is None:
            # Seed: leftmost -> "A", the other (if any) -> "B".
            ordered = sorted(top2, key=lambda d: d.x)
            if ordered:
                a.add_live(t_s, frame_idx, ordered[0])
            if len(ordered) > 1:
                b.add_live(t_s, frame_idx, ordered[1])
            return

        if b.last_foot is None and len(top2) > 1:
            # Second track seeds from the detection farther from A.
            ordered = sorted(top2, key=lambda d: _distance(d.foot_px, a.last_foot))
            a.add_live(t_s, frame_idx, ordered[0])
            b.add_live(t_s, frame_idx, ordered[1])
            return

        if len(top2) == 0:
            a.add_coast(t_s, frame_idx)
            b.add_coast(t_s, frame_idx)
            return

        if len(top2) == 1 or b.last_foot is None:
            detection = top2[0]
            if b.last_foot is None:
                a.add_live(t_s, frame_idx, detection)
                return
            cost_a = _distance(detection.foot_px, a.last_foot)
            cost_b = _distance(detection.foot_px, b.last_foot)
            if cost_a <= cost_b:
                a.add_live(t_s, frame_idx, detection)
                b.add_coast(t_s, frame_idx)
            else:
                b.add_live(t_s, frame_idx, detection)
                a.add_coast(t_s, frame_idx)
            return

        d1, d2 = top2
        straight = (_distance(d1.foot_px, a.last_foot)
                    + _distance(d2.foot_px, b.last_foot))
        crossed = (_distance(d2.foot_px, a.last_foot)
                   + _distance(d1.foot_px, b.last_foot))
        low, high = min(straight, crossed), max(straight, crossed)
        if high > 0 and low >= (1.0 - AMBIGUITY_MARGIN) * high:
            self._ambiguity_times.append(t_s)
        if straight <= crossed:
            a.add_live(t_s, frame_idx, d1)
            b.add_live(t_s, frame_idx, d2)
        else:
            a.add_live(t_s, frame_idx, d2)
            b.add_live(t_s, frame_idx, d1)

    def samples(self):
        return {key: list(track.samples) for key, track in self._tracks.items()}

    def ambiguity_times(self):
        return list(self._ambiguity_times)

    def stats(self):
        return {
            "updates": self._updates,
            "ambiguous_assignments": len(self._ambiguity_times),
        }


class PersonFramePass:
    """Coarse-pass frame observer: detect every Nth coarse frame, feed the
    tracker. job_runner passes .observe as track_segments' frame_observer.

    A detector failure at runtime (not just at load time) must not kill the
    tracking job: one bad frame permanently disables further detection for
    this pass, keeping whatever samples were already collected."""

    def __init__(self, detector, source_fps, frame_stride):
        self.detector = detector
        self.source_fps = float(source_fps) or 30.0
        coarse_hz = self.source_fps / max(1, int(frame_stride))
        self.detect_every = max(1, round(coarse_hz / PERSON_DETECT_HZ))
        self.tracker = TwoPlayerTracker()
        self._seen = 0
        self._disabled = False
        self.detect_failures = 0
        self.detect_error = None

    def observe(self, frame_idx, frame_bgr):
        if self._disabled:
            return
        index = self._seen
        self._seen += 1
        if index % self.detect_every != 0:
            return
        try:
            detections = self.detector.detect(frame_bgr)
            self.tracker.update(frame_idx / self.source_fps, frame_idx, detections)
        except Exception as error:
            self.detect_failures += 1
            if self.detect_error is None:
                self.detect_error = repr(error)
            self._disabled = True

    def stats(self):
        """Tracker stats plus this pass's own detect_failures count, for the
        job_runner payload (job_runner reads person_pass.stats(), not
        person_pass.tracker.stats(), so a runtime detector failure is always
        visible in players_v1)."""
        stats = dict(self.tracker.stats())
        stats["detect_failures"] = self.detect_failures
        return stats
