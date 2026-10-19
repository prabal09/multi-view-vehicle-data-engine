"""Step 4: project per-camera tracks into the shared bird's-eye-view (BEV).

Reuses the existing homographies h_c0..h_c4 from frameCalibration.yml (built in
the prior AvaCar work), which map each camera's 1600x900 image plane to the
shared top-view -- validated to ~1px. Our clips are 1600x900, so no rescaling.

For each track box we take the ground-contact point (bottom-center of the box
= where the vehicle meets the road, v1) and project it through the camera's
homography to get a BEV (x, y). A track thus becomes a path in the common BEV
frame -- the input cross-view association (Step 5) needs.

Pure NumPy/OpenCV (cv2.FileStorage), no GPU.
"""

from __future__ import annotations

import cv2
import numpy as np

CAMERAS = ["c0", "c1", "c2", "c3", "c4"]


def load_homographies(yml_path: str) -> dict[str, np.ndarray]:
    """Load h_c0..h_c4 (3x3) from a frameCalibration.yml. Missing ones omitted."""
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    out = {}
    for cam in CAMERAS:
        node = fs.getNode(f"h_{cam}")
        mat = None if node.empty() else node.mat()
        if mat is not None and mat.shape == (3, 3):
            out[cam] = mat
    fs.release()
    return out


def contact_point(box) -> tuple[float, float]:
    """Ground-contact point = bottom-center of the box (v1)."""
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, float(y1))


def project_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply a homography to an image point -> BEV point."""
    v = H @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def project_track(boxes: dict[int, tuple], H: np.ndarray) -> dict[int, tuple]:
    """{frame: box} -> {frame: (bev_x, bev_y)} via bottom-center contact point."""
    return {f: project_point(H, *contact_point(b)) for f, b in boxes.items()}


def bev_motion(trajectory: dict[int, tuple]) -> tuple[float, float]:
    """(net_displacement, path_length) of a BEV trajectory. Parked -> ~0."""
    frames = sorted(trajectory)
    if len(frames) < 2:
        return 0.0, 0.0
    pts = np.array([trajectory[f] for f in frames])
    net = float(np.linalg.norm(pts[-1] - pts[0]))
    path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return net, path


# --- local demo: project the real c0_5 tracks into BEV --------------------
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--yml", default=r"C:/Users/praba/PycharmProjects/AvaCar/"
                                      r"AvaCAR_code/localization/frameCalibration.yml")
    ap.add_argument("--tracks", default="c0_5_tracks.json")
    args = ap.parse_args()

    H = load_homographies(args.yml)
    print("loaded homographies for:", sorted(H))

    tracks = json.load(open(args.tracks))
    print(f"\n{len(tracks)} tracks from {args.tracks}\n")
    print(f"{'id':>3} {'cam':>4} {'frames':>12} {'BEV net':>8} {'BEV path':>9}  kind")
    print("-" * 56)
    rows = []
    for t in tracks:
        cam = t["camera"]
        boxes = {int(f): tuple(b) for f, b in t["boxes"].items()}
        traj = project_track(boxes, H[cam])
        net, path = bev_motion(traj)
        rows.append((t["track_id"], cam, min(boxes), max(boxes), net, path))
    for tid, cam, f0, f1, net, path in sorted(rows, key=lambda r: -r[4]):
        kind = "MOVING (road)" if net > 50 else "stationary"
        print(f"{tid:>3} {cam:>4} {f0:>5}..{f1:<5} {net:>8.1f} {path:>9.1f}  {kind}")
