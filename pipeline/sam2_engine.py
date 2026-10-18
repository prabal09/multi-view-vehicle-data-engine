"""Step 3, piece 2: the SAM2 clip engine (mask propagation).

Wraps SAM2's video predictor in the mode our sanity test already proved:
  seed one or more boxes (each at the frame it was first detected) -> propagate
  once forward -> collect each object's mask (as a box) on every frame.

All SAM2/torch specifics live here and are lazy-imported, so importing the rest
of the pipeline locally never needs a GPU. This is the ONE piece that can only
be validated on the EC2 box; `mask_to_box` below is pure and locally testable.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import numpy as np

# Where the .pt checkpoints live (override with $SAM2_CHECKPOINT_DIR).
CHECKPOINT_DIR = os.environ.get("SAM2_CHECKPOINT_DIR",
                                os.path.expanduser("~/checkpoints"))

# model name -> (checkpoint filename, packaged hydra config path). Smaller models
# use far less memory and run much faster -- prefer "tiny"/"small" for tracking.
SAM2_MODELS = {
    "tiny":  ("sam2.1_hiera_tiny.pt",      "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt",     "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base":  ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt",     "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def resolve_model(name: str) -> tuple[str, str]:
    """model name -> (checkpoint_path, config path)."""
    fname, cfg = SAM2_MODELS[name]
    return os.path.join(CHECKPOINT_DIR, fname), cfg


def mask_to_box(mask) -> Optional[tuple]:
    """Tight (x0,y0,x1,y1) around a boolean mask, or None if the mask is empty.

    Accepts (H,W) or (1,H,W). Pure NumPy -- unit-testable without SAM2.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class Sam2ClipEngine:
    """SAM2 video predictor, built once and reused across windows (EC2-only).

    The model is loaded in __init__; `track_window` runs one window end-to-end
    (init_state -> seed -> propagate) with a FRESH state each call, so memory is
    bounded by the window length -- not the whole clip -- and is freed between
    windows. Frame indices in/out of `track_window` are LOCAL to the window.
    """

    def __init__(self,
                 checkpoint: Optional[str] = None,
                 config: Optional[str] = None,
                 device: Optional[str] = None,
                 offload_to_cpu: bool = False):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if checkpoint is None or config is None:
            checkpoint, config = resolve_model("large")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=self.device)
        self.offload = offload_to_cpu

    def _autocast(self):
        if self.device == "cuda":
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return self.torch.autocast("cpu", enabled=False)

    def track_window(self, frames_dir: str, seeds) -> dict[int, dict[int, tuple]]:
        """Track one window of frames.

        seeds: list of (local_frame_idx, box). Returns {obj_id: {local_frame: box}}.
        A fresh inference state is created and freed here, so repeated calls don't
        accumulate memory across windows.
        """
        t = self.torch
        out: dict[int, dict[int, tuple]] = defaultdict(dict)
        if not seeds:
            return dict(out)

        with t.inference_mode(), self._autocast():
            state = self.predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=self.offload,
                offload_state_to_cpu=self.offload,
            )
            for obj_id, (frame_idx, box) in enumerate(seeds):
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(frame_idx),
                    obj_id=obj_id,
                    box=np.asarray(box, dtype=np.float32),
                )
            for f_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                for k, oid in enumerate(obj_ids):
                    mask = (mask_logits[k] > 0.0).cpu().numpy()
                    box = mask_to_box(mask)
                    if box is not None:
                        out[int(oid)][int(f_idx)] = box

        del state                                    # free this window's state
        if self.device == "cuda":
            t.cuda.empty_cache()
        return dict(out)


# --- local test of the pure helper: python -m pipeline.sam2_engine -----------
if __name__ == "__main__":
    m = np.zeros((100, 200), dtype=bool)
    m[20:60, 50:150] = True          # a rectangle
    print("mask_to_box (H,W):   ", mask_to_box(m), "expected ~ (50, 20, 150, 60)")
    print("mask_to_box (1,H,W): ", mask_to_box(m[None]))
    print("empty mask:          ", mask_to_box(np.zeros((10, 10), bool)))
