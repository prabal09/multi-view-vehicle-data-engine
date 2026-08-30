"""Extract specific (clip, frame) stills from the 1600x900 clips.

Reusable helper for pulling sample frames to test on (e.g. in Colab). Edit the
FRAMES list and run:  python sanity/extract_frames.py

Uses OpenCV only -- no ML dependencies.
"""

import os
import cv2

# Root of the downscaled 1600x900 clips.
CLIP_ROOT = r"C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/roadside_video_1600x900"

# Where to drop the extracted stills.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")

# (camera, clip_basename, frame_index). These are the same vehicle
# ("Cadillac Escalade" row of VehicleSyncData.xlsx) seen from all 5 views,
# using mid-points of the frame ranges in test_1.py.
FRAMES = [
    ("c0", "c0_5", 592),
    ("c1", "c1_1", 7643),
    ("c2", "c2_0", 5502),
    ("c3", "c3_3", 5790),
    ("c4", "c4_2", 8925),
]


def extract(camera: str, clip: str, frame_idx: int) -> None:
    path = os.path.join(CLIP_ROOT, camera, clip + ".mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {path}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total:
        print(f"  ERROR: {clip} has {total} frames, asked for {frame_idx}")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  ERROR: failed to read frame {frame_idx} of {clip}")
        return
    out = os.path.join(OUT_DIR, f"{clip}_f{frame_idx}.jpg")
    cv2.imwrite(out, frame)
    print(f"  wrote {out}  ({frame.shape[1]}x{frame.shape[0]})")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Extracting {len(FRAMES)} frames -> {OUT_DIR}")
    for camera, clip, idx in FRAMES:
        extract(camera, clip, idx)


if __name__ == "__main__":
    main()
