"""Step 3, piece 3c: the per-camera tracking driver.

Ties the whole chain together for ONE clip:

  detect_clip -> plan_seeds -> engine.seed(each) -> engine.propagate -> merge_tracks -> Track[]

`track_clip` takes the SAM2 engine as an argument (dependency injection), so the
orchestration is fully testable locally with a stub engine; only the real
`Sam2ClipEngine` needs the GPU. `run_clip` is the EC2 entry point that wires up
the real GDINO + SAM2.
"""

from __future__ import annotations

from typing import Optional

from .tracking import Track, merge_tracks, plan_seeds


def track_clip(detections_by_frame, engine, *, camera: str, clip: str,
               link_iou: float = 0.3, max_gap_frames: int = 30,
               merge_iou: float = 0.5, min_shared_frames: int = 3) -> list[Track]:
    """Turn sparse per-frame detections into merged per-camera tracks.

    detections_by_frame: {frame_idx: [Detection]}  (from detect_clip)
    engine:              object with .seed(frame, box)->id and .propagate()->{id:{frame:box}}
    """
    # 1. decide one seed per provisional vehicle
    seeds = plan_seeds(detections_by_frame, iou_thresh=link_iou,
                       max_gap_frames=max_gap_frames)

    # 2. seed SAM2 (or stub) with each
    for seed_frame, seed_box in seeds:
        engine.seed(seed_frame, seed_box)

    # 3. one dense forward pass
    raw = engine.propagate()                      # {obj_id: {frame: box}}

    # 4. collapse duplicate tracks from any over-splits
    merged = merge_tracks(raw, iou_thresh=merge_iou,
                          min_shared_frames=min_shared_frames)

    # 5. wrap into Track objects carrying identity metadata
    tracks = []
    for tid, boxes in sorted(merged.items()):
        if not boxes:
            continue
        tracks.append(Track(track_id=tid, camera=camera, clip=clip,
                            seed_frame=min(boxes), boxes=boxes))
    return tracks


def run_clip(clip_path: str, *, camera: str, clip: str, frames_dir: str,
             step: int = 10, detector=None, **track_kwargs) -> list[Track]:
    """Full per-clip pipeline on the GPU box (lazy-builds GDINO + SAM2)."""
    from .detect_clip import GroundingDinoDetector, detect_clip
    from .sampling import extract_all_frames
    from .sam2_engine import Sam2ClipEngine

    # 1. detections on sampled frames
    detector = detector or GroundingDinoDetector()
    detections_by_frame = detect_clip(clip_path, detector, step=step)

    # 2. extract all frames for SAM2 (frame_idx aligns with global index)
    extract_all_frames(clip_path, frames_dir)

    # 3. SAM2 engine + track
    engine = Sam2ClipEngine(frames_dir)
    return track_clip(detections_by_frame, engine, camera=camera, clip=clip,
                      **track_kwargs)
