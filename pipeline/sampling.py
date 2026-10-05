"""Frame sampling for a clip.

We don't run detection on every frame -- we sample every `step` frames and let
SAM2 track between detections. `step` must be <= the shortest vehicle dwell time
(~21 frames for a fast foreground car, from the SAM2 sanity test), so ~10 is a
safe default: every vehicle gets caught at least once while visible.

Uses grab()+retrieve() so skipped frames are advanced cheaply without the full
decode/colour-convert that read() does.
"""

from __future__ import annotations

from typing import Iterator, Optional

import cv2


def sample_frames(clip_path: str, step: int = 10, start: int = 0,
                  end: Optional[int] = None) -> Iterator[tuple[int, "cv2.Mat"]]:
    """Yield (frame_index, bgr_frame) for every `step`-th frame in [start, end).

    frame_index is the GLOBAL index within the clip, so it stays aligned with
    the (camera, clip, local-frame) addressing used everywhere else.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open clip: {clip_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if end is None else min(end, total)

    idx = 0
    try:
        while idx < end:
            if not cap.grab():            # advance one frame (cheap)
                break
            if idx >= start and (idx - start) % step == 0:
                ok, frame = cap.retrieve()  # decode only the frames we keep
                if not ok:
                    break
                yield idx, frame
            idx += 1
    finally:
        cap.release()
