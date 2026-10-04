"""Can CLIP (zero-shot) classify vehicle BODY-TYPE? -- same test as the GDINO one.

Flow mirrors the real pipeline and the GDINO experiment, so results compare 1:1:
  1. GDINO detects the vehicle -> largest box -> crop   (same crop as before).
  2. CLIP scores that crop against each candidate type.
  3. argmax = prediction; compare to the filename's ground-truth type.

Why CLIP should do better than GDINO here: CLIP embeds the image and ALL
candidate texts into one space and takes a softmax OVER the candidates -- a
comparative "which label is closest?", not GDINO's independent "is an X present?".

------------------------------------------------------------------------------
COLAB
------------------------------------------------------------------------------
  !pip install -q --upgrade "transformers>=4.44" timm
  !python -u clip_classify_experiment.py --dir typeset

  # experiment with prompt wording / candidates:
  !python -u clip_classify_experiment.py --dir typeset --template "a photo of a {t}"
  !python -u clip_classify_experiment.py --dir typeset \
      --candidates "red sedan,red suv,red hatchback,blue coupe,black suv"
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import inspect
import os

import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)

GDINO = "IDEA-Research/grounding-dino-base"
CLIP = "openai/clip-vit-large-patch14"
DETECT_PROMPT = "car. suv. truck. van. bus. pickup truck."
DEFAULT_TYPES = ["sedan", "suv", "pickup truck", "coupe", "hatchback", "minivan", "van", "bus"]


# ----------------------------- GDINO: make the crop --------------------------
def load_gdino(device):
    proc = AutoProcessor.from_pretrained(GDINO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(device).eval()
    return proc, model


@torch.no_grad()
def detect_crop(proc, model, image, device, pad=0.08):
    inputs = proc(images=image, text=DETECT_PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    fn = proc.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kw = {"target_sizes": [image.size[::-1]], "text_threshold": 0.20}
    kw["threshold" if "threshold" in params else "box_threshold"] = 0.25
    r = fn(outputs, inputs["input_ids"], **kw)[0]
    boxes = r["boxes"].tolist()
    if not boxes:
        return None
    i = max(range(len(boxes)), key=lambda k: (boxes[k][2]-boxes[k][0])*(boxes[k][3]-boxes[k][1]))
    x0, y0, x1, y1 = boxes[i]
    w, h = image.size
    dx, dy = (x1-x0)*pad, (y1-y0)*pad
    return image.crop((max(0, x0-dx), max(0, y0-dy), min(w, x1+dx), min(h, y1+dy)))


# ----------------------------- CLIP: classify the crop -----------------------
def load_clip(device):
    model = CLIPModel.from_pretrained(CLIP).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP)
    return proc, model


@torch.no_grad()
def classify(proc, model, crop, candidates, template, device):
    texts = [template.format(t=c) for c in candidates]
    inputs = proc(text=texts, images=crop, return_tensors="pt", padding=True).to(device)
    probs = model(**inputs).logits_per_image.softmax(dim=-1)[0]  # over candidates
    return sorted(zip(candidates, probs.tolist()), key=lambda cs: -cs[1])


def main():
    ap = argparse.ArgumentParser(description="CLIP zero-shot body-type experiment")
    ap.add_argument("--dir", default="typeset")
    ap.add_argument("--candidates", default=",".join(DEFAULT_TYPES))
    ap.add_argument("--template", default="a photo of a {t}")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    print(f"device={device}  template='{args.template}'")
    print(f"candidates: {candidates}\n")

    gproc, gmodel = load_gdino(device)
    cproc, cmodel = load_clip(device)
    known = set(DEFAULT_TYPES)
    correct = scored = 0

    paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
    if not paths:
        print(f"No .jpg files found in '{args.dir}' (cwd: {os.getcwd()})")
        return

    for path in paths:
        gt = os.path.basename(path).split("__")[0]
        image = Image.open(path).convert("RGB")
        crop = detect_crop(gproc, gmodel, image, device)
        if crop is None:
            print(f"{os.path.basename(path)}: no vehicle detected, skipping")
            continue

        ranked = classify(cproc, cmodel, crop, candidates, args.template, device)
        pred = ranked[0][0]
        mark = ""
        if gt in known and all(c in known for c in candidates):
            scored += 1
            ok = pred == gt
            correct += ok
            mark = "OK " if ok else "XX "
        scores_str = "  ".join(f"{c}:{s:.2f}" for c, s in ranked)
        print(f"{mark}{os.path.basename(path):<42} gt={gt:<10} pred={pred:<12} | {scores_str}")

    if scored:
        print(f"\naccuracy: {correct}/{scored} = {correct/scored:.0%}")


if __name__ == "__main__":
    main()
