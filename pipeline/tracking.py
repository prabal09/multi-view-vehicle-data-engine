"""Step 3, piece 1: the tracking "brain" -- associate detections to tracks.

At each sampled frame we must decide, for every fresh detection, whether it is a
vehicle SAM2 is already tracking (skip) or a new one (seed a new track). That is
a matching problem between N detection boxes and M existing-track boxes, solved
with a pairwise IoU matrix.

This module is pure NumPy -- no torch -- so the matching logic is unit-testable
locally. SAM2 itself (the mask propagation) lives in the EC2-only engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """One vehicle's trajectory within a single clip (one camera)."""
    track_id: int
    camera: str
    clip: str
    seed_frame: int
    # frame_index -> box (x0,y0,x1,y1); masks are attached by the SAM2 engine.
    boxes: dict = field(default_factory=dict)

    @property
    def frame_start(self) -> int:
        return min(self.boxes) if self.boxes else self.seed_frame

    @property
    def frame_end(self) -> int:
        return max(self.boxes) if self.boxes else self.seed_frame


def pairwise_iou(a, b) -> np.ndarray:
    """IoU of every box in a (N,4) against every box in b (M,4) -> (N,M).

    Vectorized via broadcasting: an (N,1) column against a (1,M) row fills the
    whole (N,M) grid in one shot -- no Python loop over pairs.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    ax0, ay0, ax1, ay1 = (a[:, i][:, None] for i in range(4))   # each (N,1)
    bx0, by0, bx1, by1 = (b[:, i][None, :] for i in range(4))   # each (1,M)

    inter_w = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    inter_h = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = inter_w * inter_h                                   # (N,M)

    area_a = ((ax1 - ax0) * (ay1 - ay0))                       # (N,1)
    area_b = ((bx1 - bx0) * (by1 - by0))                       # (1,M)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def associate(det_boxes, track_boxes, iou_thresh: float = 0.3):
    """Greedy match detections to existing tracks by IoU.

    Returns (matches, unmatched_dets, unmatched_tracks) where
      matches          = [(det_index, track_index), ...]
      unmatched_dets   = detection indices that matched no track -> NEW tracks
      unmatched_tracks = track indices with no detection this frame
    """
    iou = pairwise_iou(det_boxes, track_boxes)              # (N,M)
    n, m = iou.shape

    # Greedy: take the highest-IoU pair above threshold, lock both out, repeat.
    pairs = sorted(
        ((iou[i, j], i, j) for i in range(n) for j in range(m)),
        key=lambda t: -t[0],
    )
    matches, used_d, used_t = [], set(), set()
    for score, i, j in pairs:
        if score < iou_thresh:
            break
        if i in used_d or j in used_t:
            continue
        matches.append((i, j))
        used_d.add(i)
        used_t.add(j)

    unmatched_dets = [i for i in range(n) if i not in used_d]
    unmatched_tracks = [j for j in range(m) if j not in used_t]
    return matches, unmatched_dets, unmatched_tracks


# --- local demo (no GPU): python -m pipeline.tracking ------------------------
if __name__ == "__main__":
    # 3 fresh detections this frame; 2 tracks SAM2 is already carrying.
    det_boxes = [
        [440, 320, 1260, 640],   # overlaps track 0  -> already tracked
        [100, 300, 300, 450],    # overlaps track 1  -> already tracked
        [1500, 200, 1600, 300],  # overlaps nothing  -> NEW vehicle
    ]
    track_boxes = [
        [445, 322, 1255, 638],   # track 0 (the Yukon)
        [105, 305, 295, 448],    # track 1
    ]

    matches, new_dets, lost_tracks = associate(det_boxes, track_boxes, iou_thresh=0.3)
    print("IoU matrix (dets x tracks):")
    print(np.round(pairwise_iou(det_boxes, track_boxes), 2))
    print("matches (det->track):", matches)
    print("new detections -> seed new tracks:", new_dets)
    print("tracks with no detection this frame:", lost_tracks)
