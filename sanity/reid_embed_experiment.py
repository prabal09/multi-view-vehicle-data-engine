"""Compare appearance embedders for cross-view vehicle matching. (Colab)

Embeds the crops from extract_crops.py with a chosen model, AVERAGES the crops
per track, and measures how well the embeddings separate the two vehicles
(same-vs-cross cosine + nearest-neighbour accuracy) -- directly comparable to
the CLIP baseline (which scored 8/10, failing both c4 tracks).

Filenames encode ground truth: {label}__{camera}__{clip}__t{tid}__f{frame}.jpg

------------------------------------------------------------------------------
COLAB
------------------------------------------------------------------------------
  !pip install -q --upgrade "transformers>=4.44" timm
  # upload the `crops/` folder, then:
  !python -u reid_embed_experiment.py --crops crops --model dinov2
  !python -u reid_embed_experiment.py --crops crops --model clip     # baseline
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

MODELS = {
    "dinov2": "facebook/dinov2-base",
    "clip": "openai/clip-vit-large-patch14",
}


def load_embedder(kind: str, device: str):
    if kind == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(MODELS["clip"]).to(device).eval()
        proc = CLIPProcessor.from_pretrained(MODELS["clip"])

        def embed(img):
            inputs = proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.get_image_features(**inputs)
            feat = out if torch.is_tensor(out) else (
                getattr(out, "image_embeds", None)
                if getattr(out, "image_embeds", None) is not None
                else out.pooler_output)
            return feat[0]
    else:  # dinov2
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(MODELS["dinov2"]).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(MODELS["dinov2"])

        def embed(img):
            inputs = proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            return out.pooler_output[0]       # CLS token
    return embed


def parse(fname: str):
    label, cam, clip, tid, _ = os.path.basename(fname)[:-4].split("__")
    return label, cam, clip, tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--model", default="dinov2", choices=list(MODELS))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={args.model}  device={device}")
    embed = load_embedder(args.model, device)

    # embed each crop, group by track, average -> one vector per track
    per_track = defaultdict(list)
    meta = {}
    for f in sorted(glob.glob(os.path.join(args.crops, "*.jpg"))):
        label, cam, clip, tid = parse(f)
        key = (cam, clip, tid)
        v = embed(Image.open(f).convert("RGB")).float().cpu().numpy()
        per_track[key].append(v)
        meta[key] = (label, cam, clip)

    keys = list(per_track)
    E = np.array([np.mean(per_track[k], axis=0) for k in keys])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    labels = [meta[k][0] for k in keys]
    S = E @ E.T

    same, cross = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (same if labels[i] == labels[j] else cross).append(S[i, j])
    print(f"\n{len(keys)} tracks | avg cosine  same: {np.mean(same):.3f}  "
          f"cross: {np.mean(cross):.3f}  (min-same {min(same):.3f}, max-cross {max(cross):.3f})")

    correct = 0
    print("\nnearest-neighbour per track:")
    for i, k in enumerate(keys):
        nn = max((j for j in range(len(keys)) if j != i), key=lambda j: S[i, j])
        ok = labels[nn] == labels[i]
        correct += ok
        print(f"  {'OK ' if ok else 'XX '}{labels[i]:<9}{meta[k][1]}/{meta[k][2]} "
              f"-> {labels[nn]:<9}{meta[keys[nn]][1]}/{meta[keys[nn]][2]} ({S[i,nn]:.3f})")
    print(f"\nnearest-neighbour accuracy: {correct}/{len(keys)}")


if __name__ == "__main__":
    main()
