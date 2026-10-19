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


# --- validity / ROI filtering ------------------------------------------------
BEV_W, BEV_H = 1600, 900          # the shared top-view extent


def load_horizons(yml_path: str) -> dict[str, tuple]:
    """Load horizon_c0..c4 as ((x1,y1),(x2,y2)) line endpoints."""
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    out = {}
    for cam in CAMERAS:
        node = fs.getNode(f"horizon_{cam}")
        mat = None if node.empty() else node.mat()
        if mat is not None:
            v = mat.reshape(-1)
            out[cam] = ((float(v[0]), float(v[1])), (float(v[2]), float(v[3])))
    fs.release()
    return out


def horizon_y_at(line: tuple, x: float) -> float:
    """y of the horizon line at a given x."""
    (x1, y1), (x2, y2) = line
    if x2 == x1:
        return min(y1, y2)
    return y1 + (y2 - y1) / (x2 - x1) * (x - x1)


def in_bev_bounds(pt: tuple, margin: float = 200.0) -> bool:
    x, y = pt
    return -margin <= x <= BEV_W + margin and -margin <= y <= BEV_H + margin


def valid_contact(cam: str, box, H: np.ndarray, horizon=None,
                  horizon_margin: float = 15.0, bounds_margin: float = 200.0):
    """Project a box's contact point to BEV, or None if invalid.

    Invalid = contact point at/above the horizon (unstable projection), or the
    projected BEV point lands outside the top-view extent (near-horizon garbage).
    """
    cx, cy = contact_point(box)
    if horizon is not None and cy < horizon_y_at(horizon, cx) + horizon_margin:
        return None                      # too close to / above the horizon
    bev = project_point(H, cx, cy)
    return bev if in_bev_bounds(bev, bounds_margin) else None


def filter_track_to_bev(boxes: dict[int, tuple], H: np.ndarray, cam: str,
                        horizon=None, **kw) -> dict[int, tuple]:
    """{frame: box} -> {frame: bev_pt}, keeping only valid (on-ground) frames."""
    out = {}
    for f, b in boxes.items():
        bev = valid_contact(cam, b, H, horizon=horizon, **kw)
        if bev is not None:
            out[f] = bev
    return out


def localize_tracks(tracks, homographies: dict, horizons: dict | None = None, *,
                    min_valid: int = 5, speed_thresh: float = 1.0
                    ) -> dict[int, dict[int, tuple]]:
    """Step 4 output: on-road moving vehicles -> their BEV trajectories.

    tracks: list of {track_id, camera, boxes: {frame: box}}.
    Keeps a track only if it has >= min_valid on-ground frames AND its average
    BEV speed exceeds speed_thresh (drops parked/background/near-horizon).
    Returns {track_id: {frame: (bev_x, bev_y)}}.
    """
    horizons = horizons or {}
    kept = {}
    for t in tracks:
        cam = t["camera"]
        H = homographies.get(cam)
        if H is None:
            continue
        boxes = {int(f): tuple(b) for f, b in t["boxes"].items()}
        traj = filter_track_to_bev(boxes, H, cam, horizon=horizons.get(cam))
        if len(traj) < min_valid:
            continue
        net, _ = bev_motion(traj)
        if net / len(traj) <= speed_thresh:
            continue
        kept[t["track_id"]] = traj
    return kept


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
    horizons = load_horizons(args.yml)
    print("loaded homographies for:", sorted(H))

    MIN_VALID = 5        # need this many on-ground frames to trust a track
    SPEED_THRESH = 1.0   # BEV px/frame to count as moving (vs parked/drift)

    tracks = json.load(open(args.tracks))
    print(f"\n{len(tracks)} tracks from {args.tracks}\n")
    print(f"{'id':>3} {'cam':>4} {'frames':>12} {'valid':>7} {'BEV net':>8} {'px/frame':>8}  verdict")
    print("-" * 72)
    rows = []
    for t in tracks:
        cam = t["camera"]
        boxes = {int(f): tuple(b) for f, b in t["boxes"].items()}
        traj = filter_track_to_bev(boxes, H[cam], cam, horizon=horizons.get(cam))
        net, _ = bev_motion(traj)
        speed = net / len(traj) if traj else 0.0
        rows.append((t["track_id"], cam, min(boxes), max(boxes),
                     len(boxes), len(traj), net, speed))

    kept = 0
    for tid, cam, f0, f1, n_all, n_valid, net, speed in sorted(rows, key=lambda r: -r[7]):
        on_road = n_valid >= MIN_VALID
        moving = speed > SPEED_THRESH
        if on_road and moving:
            verdict = "KEEP (moving vehicle)"; kept += 1
        elif on_road:
            verdict = "drop (parked/stationary)"
        else:
            verdict = "drop (off-road/near-horizon)"
        print(f"{tid:>3} {cam:>4} {f0:>5}..{f1:<5} {n_valid:>3}/{n_all:<3} "
              f"{net:>8.1f} {speed:>8.2f}  {verdict}")
    print(f"\nkept {kept} moving vehicle(s) of {len(tracks)} tracks")
