"""Step 5b: coarse vehicle_type per track via CLIP zero-shot on the crops. (Colab)

Reads the crops from extract_crops.py, classifies each into a COARSE type
{suv, car, truck, bus} (CLIP comparative zero-shot -- strong on coarse types),
votes per track, and writes track_types.json: {"camera#clip#track_id": type}.
The ETL loader reads this and majority-votes a vehicle_type per vehicle.

  !pip install -q --upgrade "transformers>=4.44" timm
  !python classify_types.py --crops crops --out track_types.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-large-patch14"
TYPES = ["suv", "car", "truck", "bus"]        # coarse, well-separated by CLIP
TEMPLATE = "a photo of a {t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--out", default="track_types.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    texts = [TEMPLATE.format(t=t) for t in TYPES]

    per_track = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.crops, "*.jpg"))):
        # filename: {label}__{camera}__{clip}__t{tid}__f{frame}.jpg
        _, cam, clip, tid, _ = os.path.basename(f)[:-4].split("__")
        key = f"{cam}#{clip}#{int(tid[1:])}"       # 't5' -> 5, matches loader key
        img = Image.open(f).convert("RGB")
        inputs = proc(text=texts, images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=-1)[0]
        per_track[key].append(TYPES[int(probs.argmax())])

    # majority vote per track
    track_type = {k: Counter(v).most_common(1)[0][0] for k, v in per_track.items()}
    with open(args.out, "w") as fh:
        json.dump(track_type, fh, indent=2)

    print(f"wrote {args.out}")
    for k, t in sorted(track_type.items()):
        print(f"  {k:<18} {t}")


if __name__ == "__main__":
    main()
