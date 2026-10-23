"""Extract vehicle crops for the kept (moving) tracks, to test appearance models.

For each on-road moving track (per Step 4's filter), pull K crops spread across
the track from the LOCAL clips and save them. We then embed these crops with
different models (CLIP vs DINOv2 vs vehicle-ReID) on Colab and compare how well
each separates the two vehicles -- without re-running SAM2.

cv2 only, no torch -- runs locally. Filenames encode ground truth for eval:
    {label}__{camera}__{clip}__t{track_id}__f{frame}.jpg
"""

import glob
import json
import os

import cv2

from pipeline.bev import load_homographies, load_horizons, localize_tracks

CLIP_ROOT = r"C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/roadside_video_1600x900"
YML = r"C:/Users/praba/PycharmProjects/AvaCar/AvaCAR_code/localization/frameCalibration.yml"
RUNS_DIR = "runs"
OUT_DIR = os.path.join("sanity", "crops")
K = 4          # crops per track (the K LARGEST/clearest boxes)
PAD = 0.05
MIN_DIM = 50   # skip crops smaller than this (too small to embed meaningfully)

# ground-truth first-frames per vehicle per camera (for labeling the eval only)
GT = {"Prius": {"c0_5": 460, "c1_1": 7510, "c2_0": 5365, "c3_3": 7356, "c4_3": 8772},
      "Cadillac": {"c0_5": 590, "c1_1": 7640, "c2_0": 5495, "c3_3": 7486, "c4_2": 8902}}


def label_for(clip, mid):
    return min(GT, key=lambda v: abs(GT[v].get(clip, 1e9) - mid))


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def largest_frames(boxes, k, min_dim):
    """Frames of the K largest boxes, skipping any smaller than min_dim."""
    ranked = sorted(boxes, key=lambda fr: -_area(boxes[fr]))
    out = []
    for fr in ranked:
        b = boxes[fr]
        if min(b[2] - b[0], b[3] - b[1]) < min_dim:
            break                         # sorted desc -> rest are smaller too
        out.append(fr)
        if len(out) >= k:
            break
    return out


def crop_box(frame, box, pad=PAD):
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    xa, ya = max(0, int(x0 - dx)), max(0, int(y0 - dy))
    xb, yb = min(w, int(x1 + dx)), min(h, int(y1 + dy))
    c = frame[ya:yb, xa:xb]
    return c if c.size else None


def main():
    H = load_homographies(YML)
    hz = load_horizons(YML)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(RUNS_DIR, "*_tracks.json"))):
        tracks = json.load(open(f))
        by_id = {t["track_id"]: t for t in tracks}
        kept = localize_tracks(tracks, H, hz)
        for tid, traj in kept.items():
            t = by_id[tid]
            cam, clip = t["camera"], t["clip"]
            boxes = {int(fr): b for fr, b in t["boxes"].items()}
            mid = (min(traj) + max(traj)) / 2
            label = label_for(clip, mid)
            cap = cv2.VideoCapture(os.path.join(CLIP_ROOT, cam, clip + ".mp4"))
            for fr in largest_frames(boxes, K, MIN_DIM):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, frame = cap.read()
                if not ok:
                    continue
                c = crop_box(frame, boxes[fr])
                if c is None:
                    continue
                name = f"{label}__{cam}__{clip}__t{tid}__f{fr}.jpg"
                cv2.imwrite(os.path.join(OUT_DIR, name), c)
                n += 1
            cap.release()
    print(f"wrote {n} crops to {OUT_DIR}")


if __name__ == "__main__":
    main()
