"""Step 3, piece 3c: the windowed per-camera tracking driver.

SAM2 state grows with (frames x objects), so tracking a whole clip in one state
OOMs a 16 GB GPU. Fix: process the clip in OVERLAPPING windows, each with a fresh
SAM2 state (memory bounded by window length), then run `merge_tracks` once over
all windows. Because consecutive windows OVERLAP, a vehicle crossing a boundary
is tracked in both windows and shares frames in the overlap -- so `merge_tracks`
stitches the two halves into one track automatically. One mechanism handles both
within-window de-duplication and cross-window stitching.

`track_clip` takes the engine and a frame-extractor as arguments (dependency
injection), so the windowing/stitching logic is fully testable locally with
stubs; only `run_clip` touches the GPU and disk.
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, Optional

from .tracking import Track, merge_tracks, plan_seeds


def make_windows(start: int, end: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Split [start, end) into overlapping [ws, we) windows."""
    stride = max(1, window - overlap)
    windows = []
    ws = start
    while ws < end:
        windows.append((ws, min(ws + window, end)))
        if ws + window >= end:
            break
        ws += stride
    return windows


def track_clip(detections_by_frame, engine, extract_window: Callable[[int, int], str],
               *, camera: str, clip: str, end: int, start: int = 0,
               window: int = 120, overlap: int = 20,
               link_iou: float = 0.3, max_gap_frames: int = 30,
               merge_iou: float = 0.5, min_shared_frames: int = 3) -> list[Track]:
    """Windowed tracking over frames [start, end) of one clip.

    detections_by_frame: {global_frame: [Detection]}  (from detect_clip)
    engine:        has track_window(frames_dir, seeds)->{obj_id:{local_frame:box}}
    extract_window: (ws, we) -> frames_dir with that window's frames (0-based names)
    """
    all_raw: dict[int, dict[int, tuple]] = {}
    gid = 0

    for ws, we in make_windows(start, end, window, overlap):
        det_win = {f: d for f, d in detections_by_frame.items() if ws <= f < we}
        if not det_win:
            continue

        # seeds are global frames; SAM2 works in window-local frames
        seeds_global = plan_seeds(det_win, iou_thresh=link_iou,
                                  max_gap_frames=max_gap_frames)
        seeds_local = [(sf - ws, box) for sf, box in seeds_global]

        frames_dir = extract_window(ws, we)
        raw = engine.track_window(frames_dir, seeds_local)   # {obj:{local:box}}

        for fb in raw.values():                               # remap local -> global
            all_raw[gid] = {ws + lf: box for lf, box in fb.items()}
            gid += 1

    # one global pass: de-dup within windows + stitch across overlaps
    merged = merge_tracks(all_raw, iou_thresh=merge_iou,
                          min_shared_frames=min_shared_frames)
    return [Track(track_id=tid, camera=camera, clip=clip,
                 seed_frame=min(boxes), boxes=boxes)
            for tid, boxes in sorted(merged.items()) if boxes]


def run_clip(clip_path: str, *, camera: str, clip: str, frames_dir: str,
             step: int = 10, start: int = 0, end: Optional[int] = None,
             window: int = 120, overlap: int = 20,
             sam_model: str = "large", offload: bool = False,
             appearance: bool = True,
             detector=None, **track_kwargs) -> list[Track]:
    """Full windowed per-clip pipeline on the GPU box (lazy-builds GDINO + SAM2).

    Processes frames [start, end). `window`/`overlap` bound SAM2 memory per window.
    `sam_model` picks the SAM2 size; `offload` moves state to CPU RAM (slow).
    `appearance` adds a CLIP embedding per track (for cross-view association).
    """
    from .detect_clip import GroundingDinoDetector, detect_clip
    from .sampling import extract_all_frames, frame_count
    from .sam2_engine import Sam2ClipEngine, resolve_model

    detector = detector or GroundingDinoDetector()
    end = end if end is not None else frame_count(clip_path)

    # detect once over the sampled frames in [start, end)
    detections_by_frame = detect_clip(clip_path, detector, step=step, start=start, end=end)

    checkpoint, config = resolve_model(sam_model)
    engine = Sam2ClipEngine(checkpoint=checkpoint, config=config, offload_to_cpu=offload)

    def extract_window(ws: int, we: int) -> str:
        win_dir = os.path.join(frames_dir, f"win_{ws:06d}")
        if os.path.isdir(win_dir):
            shutil.rmtree(win_dir)
        extract_all_frames(clip_path, win_dir, start=ws, end=we)
        return win_dir

    tracks = track_clip(detections_by_frame, engine, extract_window,
                        camera=camera, clip=clip, start=start, end=end,
                        window=window, overlap=overlap, **track_kwargs)

    if appearance and tracks:
        from .appearance import AppearanceEmbedder, embed_tracks
        embed_tracks(clip_path, tracks, AppearanceEmbedder())

    return tracks
