"""Grounding DINO sanity test -- open-vocabulary vehicle detection on stills.

Purpose: confirm, before spending on an EC2 GPU, that Grounding DINO detects
the vehicles in our roadside frames from a plain text prompt.

------------------------------------------------------------------------------
HOW TO RUN IN GOOGLE COLAB
------------------------------------------------------------------------------
1. Runtime -> Change runtime type -> GPU (T4 is fine).
2. Upload this file and the `frames/` folder (the extracted .jpg stills).
3. In a cell, install the one dependency Colab is missing, then run:

       !pip install -q --upgrade "transformers>=4.44" timm
       !python grounding_dino_sanity.py --frames frames --out annotated

   (torch + PIL are already in Colab.)
4. Open the images written to `annotated/` and read the printed summary.

Tweak the PROMPT / thresholds below and re-run to see how detections change.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import inspect
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# --- defaults (override via CLI flags) ---------------------------------------
# "-tiny" downloads faster; switch to "-base" for higher accuracy.
DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"

# Grounding DINO expects LOWERCASE phrases, separated by ". ", ending with ".".
# Each phrase is an independent open-vocabulary query.
DEFAULT_PROMPT = "a car. a truck. an suv. a van. a pickup truck. a bus. a person."

DEFAULT_BOX_THRESHOLD = 0.35    # min confidence to keep a box
DEFAULT_TEXT_THRESHOLD = 0.25   # min phrase-match strength for the label


def load_model(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def _post_process(processor, outputs, input_ids, target_size,
                  box_threshold, text_threshold):
    """Call post_process_grounded_object_detection robustly across the several
    transformers versions that renamed its keyword arguments."""
    fn = processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kwargs = {"target_sizes": [target_size], "text_threshold": text_threshold}
    if "threshold" in params:            # newer transformers
        kwargs["threshold"] = box_threshold
    elif "box_threshold" in params:      # older transformers
        kwargs["box_threshold"] = box_threshold
    return fn(outputs, input_ids, **kwargs)[0]


@torch.no_grad()
def detect(processor, model, image: Image.Image, prompt: str, device: str,
           box_threshold: float, text_threshold: float):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    result = _post_process(
        processor, outputs, inputs["input_ids"],
        target_size=image.size[::-1],   # (height, width)
        box_threshold=box_threshold, text_threshold=text_threshold,
    )
    # Label key was renamed across versions ('labels' -> 'text_labels').
    labels = result.get("text_labels", result.get("labels"))
    boxes = result["boxes"].tolist()
    scores = result["scores"].tolist()
    return list(zip(labels, scores, boxes))


def annotate(image: Image.Image, detections) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for label, score, (x0, y0, x1, y1) in detections:
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        tag = f"{label} {score:.2f}"
        ty = max(0, y0 - 20)
        draw.rectangle([x0, ty, x0 + 9 * len(tag), ty + 18], fill=(0, 128, 0))
        draw.text((x0 + 2, ty), tag, fill=(255, 255, 255), font=font)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounding DINO sanity test")
    ap.add_argument("--frames", default="frames", help="folder of .jpg stills")
    ap.add_argument("--out", default="annotated", help="output folder")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device        : {device}")
    print(f"model         : {args.model}")
    print(f"prompt        : {args.prompt}")
    print(f"thresholds    : box={args.box_threshold} text={args.text_threshold}\n")

    processor, model = load_model(args.model, device)
    os.makedirs(args.out, exist_ok=True)

    images = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if not images:
        print(f"No .jpg files found in '{args.frames}'.")
        return

    for path in images:
        image = Image.open(path).convert("RGB")
        dets = detect(processor, model, image, args.prompt, device,
                      args.box_threshold, args.text_threshold)
        out_path = os.path.join(args.out, os.path.basename(path))
        annotate(image, dets).save(out_path)

        print(f"{os.path.basename(path)}: {len(dets)} detections -> {out_path}")
        for label, score, box in sorted(dets, key=lambda d: -d[1]):
            xyxy = ", ".join(f"{v:.0f}" for v in box)
            print(f"    {label:<16} {score:.2f}   [{xyxy}]")
        print()


if __name__ == "__main__":
    main()
