"""Step 2: turn raw Grounding DINO output into clean per-frame vehicle detections.

Grounding DINO returns, for one frame: boxes, scores, and (noisy, multi-word)
labels. Two problems we saw in the sanity runs:
  - the SAME vehicle gets several overlapping boxes (e.g. the Hummer had 3);
  - labels are a soup ("suv truck", "pickup truck van", ...).

`clean_detections` fixes both:
  1. score filter   -- drop weak boxes
  2. label-unify    -- every vehicle-ish label becomes the single class "vehicle"
  3. NMS            -- merge heavily-overlapping boxes into one

Order matters: we unify labels BEFORE NMS on purpose. The duplicate Hummer boxes
had *different* labels ("suv truck" vs "truck" vs "pickup truck van"); class-aware
NMS would not suppress across different classes, so they'd survive. Collapsing to
one class first lets single-class NMS merge them correctly.

Pure NumPy -- no torch -- so it runs and unit-tests locally without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VEHICLE_CLASS = "vehicle"


@dataclass
class Detection:
    """One clean vehicle detection in a frame."""
    box: tuple          # (x0, y0, x1, y1) in pixels
    score: float
    label: str = VEHICLE_CLASS
    raw_label: str | None = None   # original GDINO label, kept for traceability


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box (4,) against many boxes (N,4) -> (N,)."""
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy non-max suppression. Returns indices of kept boxes.

    Keep the highest-scoring box, drop everything that overlaps it more than
    `iou_thresh`, repeat on what's left.
    """
    order = np.argsort(scores)[::-1]      # indices, best score first
    keep: list[int] = []
    while order.size > 0:
        i = order[0]                      # best remaining box
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou(boxes[i], boxes[rest])
        order = rest[ious <= iou_thresh]  # keep only the non-overlapping rest
    return keep


def clean_detections(boxes, scores, labels=None, *,
                     score_thresh: float = 0.25,
                     iou_thresh: float = 0.5) -> list[Detection]:
    """Clean one frame's raw GDINO output into vehicle detections.

    boxes:  iterable of (x0,y0,x1,y1)
    scores: iterable of float
    labels: optional iterable of raw GDINO label strings (kept for traceability)
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels is None:
        labels = [None] * len(boxes)
    labels = list(labels)

    # 1. score filter
    keep_mask = scores >= score_thresh
    boxes, scores = boxes[keep_mask], scores[keep_mask]
    labels = [l for l, k in zip(labels, keep_mask) if k]
    if len(boxes) == 0:
        return []

    # 2. label-unify happens implicitly: we treat every surviving box as one
    #    class ("vehicle"), which is also what makes step 3 a single-class NMS.
    # 3. NMS
    kept = _nms(boxes, scores, iou_thresh)

    return [
        Detection(box=tuple(float(v) for v in boxes[i]),
                  score=float(scores[i]),
                  label=VEHICLE_CLASS,
                  raw_label=labels[i])
        for i in kept
    ]


# --- local demo on REAL sanity data (no GPU needed): run `python -m pipeline.detection`
if __name__ == "__main__":
    # The Hummer frame c4_3 from the sanity run: 3 boxes are the same Hummer,
    # 1 is a different small background object (top-right).
    raw = [
        ([309, 45, 858, 347], 0.55, "suv truck"),
        ([1435, 6, 1518, 58], 0.51, "car suv pickup truck"),
        ([312, 49, 856, 348], 0.38, "truck"),
        ([312, 48, 857, 348], 0.27, "pickup truck van"),
    ]
    boxes = [r[0] for r in raw]
    scores = [r[1] for r in raw]
    labels = [r[2] for r in raw]

    print(f"raw detections: {len(raw)}")
    for b, s, l in raw:
        print(f"   {s:.2f}  {l:<22} {b}")

    cleaned = clean_detections(boxes, scores, labels)
    print(f"\nafter clean_detections: {len(cleaned)}")
    for d in cleaned:
        print(f"   {d.score:.2f}  {d.label} (raw='{d.raw_label}')  {tuple(round(v) for v in d.box)}")
