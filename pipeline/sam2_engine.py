"""Step 3, piece 2: the SAM2 clip engine (mask propagation).

Wraps SAM2's video predictor in the mode our sanity test already proved:
  seed one or more boxes (each at the frame it was first detected) -> propagate
  once forward -> collect each object's mask (as a box) on every frame.

All SAM2/torch specifics live here and are lazy-imported, so importing the rest
of the pipeline locally never needs a GPU. This is the ONE piece that can only
be validated on the EC2 box; `mask_to_box` below is pure and locally testable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

# Defaults match the sanity test (and the weights baked into the AMI).
SAM_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


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
    """SAM2 video predictor over one clip's extracted frames (EC2-only)."""

    def __init__(self, frames_dir: str,
                 checkpoint: str = SAM_CHECKPOINT,
                 config: str = SAM_CONFIG,
                 device: Optional[str] = None,
                 offload_to_cpu: bool = True):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=self.device)
        # offload_*_to_cpu keeps long clips within T4 memory.
        self.state = self.predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=offload_to_cpu,
            offload_state_to_cpu=offload_to_cpu,
        )
        self._next_obj_id = 0

    def _autocast(self):
        if self.device == "cuda":
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return self.torch.autocast("cpu", enabled=False)

    def seed(self, frame_idx: int, box) -> int:
        """Add a new object from a box at `frame_idx`. Returns its obj_id."""
        obj_id = self._next_obj_id
        self._next_obj_id += 1
        with self.torch.inference_mode(), self._autocast():
            self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=int(frame_idx),
                obj_id=obj_id,
                box=np.asarray(box, dtype=np.float32),
            )
        return obj_id

    def propagate(self) -> dict[int, dict[int, tuple]]:
        """One forward pass. Returns {obj_id: {frame_idx: box}} for all seeds."""
        out: dict[int, dict[int, tuple]] = defaultdict(dict)
        with self.torch.inference_mode(), self._autocast():
            for f_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(self.state):
                for k, oid in enumerate(obj_ids):
                    mask = (mask_logits[k] > 0.0).cpu().numpy()
                    box = mask_to_box(mask)
                    if box is not None:
                        out[int(oid)][int(f_idx)] = box
        return dict(out)


# --- local test of the pure helper: python -m pipeline.sam2_engine -----------
if __name__ == "__main__":
    m = np.zeros((100, 200), dtype=bool)
    m[20:60, 50:150] = True          # a rectangle
    print("mask_to_box (H,W):   ", mask_to_box(m), "expected ~ (50, 20, 150, 60)")
    print("mask_to_box (1,H,W): ", mask_to_box(m[None]))
    print("empty mask:          ", mask_to_box(np.zeros((10, 10), bool)))
