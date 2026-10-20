"""Step 5 input: appearance embeddings per vehicle track.

One L2-normalized CLIP image embedding per track, taken from the frame where the
vehicle's box is largest (its clearest view). Cross-view association uses these
to tell vehicles apart when they share a BEV lane (cosine similarity on the
normalized vectors). EC2-only (CLIP needs torch); crop selection is plain cv2.

Reuses CLIP -- the same model family as the Step 5b type classifier.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

CLIP_MODEL = "openai/clip-vit-large-patch14"


class AppearanceEmbedder:
    """Wraps CLIP's image encoder (lazy torch import; GPU box only)."""

    def __init__(self, model_id: str = CLIP_MODEL, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def embed(self, pil_image: Image.Image) -> list[float]:
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feat = self.model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)   # L2-normalize -> cosine
        return feat[0].cpu().tolist()


def _area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def representative_frame(boxes: dict[int, tuple]) -> int:
    """The frame where the box is largest (clearest view of the vehicle)."""
    return max(boxes, key=lambda f: _area(boxes[f]))


def crop_box(frame, box, pad: float = 0.05):
    """Crop the (padded) box region from a BGR frame; None if empty."""
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    xa, ya = max(0, int(x0 - dx)), max(0, int(y0 - dy))
    xb, yb = min(w, int(x1 + dx)), min(h, int(y1 + dy))
    crop = frame[ya:yb, xa:xb]
    return crop if crop.size else None


def embed_tracks(clip_path: str, tracks, embedder: AppearanceEmbedder, pad: float = 0.05):
    """Set each track's .embedding from its clearest crop. Mutates + returns tracks."""
    cap = cv2.VideoCapture(clip_path)
    try:
        for t in tracks:
            if not t.boxes:
                continue
            f = representative_frame(t.boxes)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            crop = crop_box(frame, t.boxes[f], pad)
            if crop is None:
                continue
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            t.embedding = embedder.embed(pil)
    finally:
        cap.release()
    return tracks
