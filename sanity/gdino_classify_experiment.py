"""Can Grounding DINO classify vehicle BODY-TYPE? -- an experiment.

Method (detector-as-classifier):
  1. Detect the vehicle in the frame (generic prompt) -> take the largest box.
  2. Crop to that box.
  3. For EACH candidate type, run Grounding DINO on the crop with just that one
     phrase and record its best score. The highest-scoring type = prediction.
  4. Compare to the ground-truth type encoded in the filename
     ("{gttype}__{name}__...jpg"), and report accuracy.

This is the right way to use a detector as a classifier (per-crop argmax over
mutually-exclusive candidates) rather than the multi-label soup you get from one
combined prompt.

Also test your color idea: pass colour-qualified candidates, e.g.
  --candidates "red sedan,red suv,red hatchback,blue coupe,black suv"
and read the ranked scores to see whether attributes sharpen the decision.

------------------------------------------------------------------------------
COLAB
------------------------------------------------------------------------------
  !pip install -q --upgrade "transformers>=4.44" timm
  !python gdino_classify_experiment.py --dir typeset

  # attribute test:
  !python gdino_classify_experiment.py --dir typeset --template "a {t}"
  !python gdino_classify_experiment.py --dir typeset \
      --candidates "sedan,suv,pickup truck,coupe,hatchback,minivan"
    OR
    !python -u gdino_classify_experiment.py --dir typeset

      ------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import inspect
import os

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL = "IDEA-Research/grounding-dino-base"
DETECT_PROMPT = "car. suv. truck. van. bus. pickup truck."
DEFAULT_TYPES = ["sedan", "suv", "pickup truck", "coupe", "hatchback", "minivan", "van", "bus"]


def load(device):
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(device).eval()
    return proc, model


@torch.no_grad()
def _run(proc, model, image, prompt, device, box_thr, text_thr):
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    fn = proc.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kw = {"target_sizes": [image.size[::-1]], "text_threshold": text_thr}
    kw["threshold" if "threshold" in params else "box_threshold"] = box_thr
    r = fn(outputs, inputs["input_ids"], **kw)[0]
    return r["boxes"].tolist(), r["scores"].tolist()


def detect_crop(proc, model, image, device, pad=0.08):
    boxes, scores = _run(proc, model, image, DETECT_PROMPT, device, 0.25, 0.20)
    if not boxes:
        return None
    # largest-area box = the foreground vehicle
    i = max(range(len(boxes)), key=lambda k: (boxes[k][2]-boxes[k][0])*(boxes[k][3]-boxes[k][1]))
    x0, y0, x1, y1 = boxes[i]
    w, h = image.size
    dx, dy = (x1-x0)*pad, (y1-y0)*pad
    box = (max(0, x0-dx), max(0, y0-dy), min(w, x1+dx), min(h, y1+dy))
    return image.crop(box)


def score_type(proc, model, crop, phrase, device):
    """Best grounding score for `phrase` on the crop (0 if nothing grounds)."""
    _, scores = _run(proc, model, crop, phrase + ".", device, 0.01, 0.01)
    return max(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser(description="GDINO body-type classification experiment")
    ap.add_argument("--dir", default="typeset", help="folder of {gttype}__name__*.jpg")
    ap.add_argument("--candidates", default=",".join(DEFAULT_TYPES),
                    help="comma-separated candidate types/phrases")
    ap.add_argument("--template", default="{t}", help="prompt template, e.g. 'a {t}'")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    print(f"device={device}  template='{args.template}'")
    print(f"candidates: {candidates}\n")

    proc, model = load(device)
    known_types = set(DEFAULT_TYPES)
    correct = scored = 0

    for path in sorted(glob.glob(os.path.join(args.dir, "*.jpg"))):
        gt = os.path.basename(path).split("__")[0]
        image = Image.open(path).convert("RGB")
        crop = detect_crop(proc, model, image, device)
        if crop is None:
            print(f"{os.path.basename(path)}: no vehicle detected, skipping")
            continue

        ranked = sorted(
            ((c, score_type(proc, model, crop, args.template.format(t=c), device))
             for c in candidates),
            key=lambda cs: -cs[1],
        )
        pred = ranked[0][0]
        mark = ""
        if gt in known_types and all(c in known_types for c in candidates):
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
