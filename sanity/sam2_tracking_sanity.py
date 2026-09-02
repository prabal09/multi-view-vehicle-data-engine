"""SAM2 tracking sanity test -- detect once, track across frames.

Pipeline proven here (the core of Steps 2->3):
  1. Grounding DINO detects the vehicle box on ONE frame.
  2. That box is handed to SAM2 as a prompt.
  3. SAM2's video predictor propagates a mask for that same vehicle across all
     the following frames -- i.e. it tracks it.

Output: an overlay video (green mask + yellow box) so you can watch SAM2 stay
locked on the vehicle as it moves.

------------------------------------------------------------------------------
HOW TO RUN IN GOOGLE COLAB
------------------------------------------------------------------------------
1. Runtime -> Change runtime type -> GPU.
2. Upload this file and the clip `c0_5.mp4` (the 1600x900 version). (Or mount
   Drive and point --clip at it.)
3. Install deps + download the SAM2 checkpoint, then run:

     !pip install -q --upgrade "transformers>=4.44" timm
     !pip install -q "git+https://github.com/facebookresearch/sam2.git"
     !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
     !python sam2_tracking_sanity.py --clip c0_5.mp4 --start 590 --num 120 --out track_out

4. Download / preview `track_out/track.mp4`.

If VRAM is tight, use a smaller model:
     !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
     ... --sam-checkpoint sam2.1_hiera_tiny.pt --sam-config configs/sam2.1/sam2.1_hiera_t.yaml
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import inspect
import os
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

# --- detection (Grounding DINO) defaults -------------------------------------
# GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"   # loads the tiny version
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
GDINO_PROMPT = "car. suv. truck. pickup truck. van. bus."
GDINO_BOX_THRESHOLD = 0.25
GDINO_TEXT_THRESHOLD = 0.20

# --- SAM2 defaults -----------------------------------------------------------
SAM_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ============================ Grounding DINO =================================
def load_gdino(device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def detect_boxes(processor, model, image: Image.Image, device):
    inputs = processor(images=image, text=GDINO_PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    fn = processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kwargs = {"target_sizes": [image.size[::-1]], "text_threshold": GDINO_TEXT_THRESHOLD}
    kwargs["threshold" if "threshold" in params else "box_threshold"] = GDINO_BOX_THRESHOLD
    result = fn(outputs, inputs["input_ids"], **kwargs)[0]
    return result["boxes"].tolist(), result["scores"].tolist()


def pick_target_box(boxes, scores):
    """The on-road vehicle is the largest, foreground box -- pick max area."""
    if not boxes:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]


# ============================ frame extraction ===============================
def extract_frames(clip_path, start, num, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    for i in range(num):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), frame)
        written += 1
    cap.release()
    return written


# ================================ overlay ====================================
def overlay(frame_bgr, mask_bool, box=None):
    out = frame_bgr.copy()
    if mask_bool is not None and mask_bool.any():
        green = np.zeros_like(out)
        green[:] = (0, 255, 0)
        out[mask_bool] = (0.5 * out[mask_bool] + 0.5 * green[mask_bool]).astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return out


# ================================== main =====================================
def main():
    ap = argparse.ArgumentParser(description="SAM2 tracking sanity test")
    ap.add_argument("--clip", default="c0_5.mp4", help="path to the 1600x900 clip")
    ap.add_argument("--start", type=int, default=590, help="first frame to track from")
    ap.add_argument("--num", type=int, default=120, help="how many frames to track")
    ap.add_argument("--ann-idx", type=int, default=0,
                    help="frame (within the window) to detect+annotate on")
    ap.add_argument("--out", default="track_out")
    ap.add_argument("--sam-checkpoint", default=SAM_CHECKPOINT)
    ap.add_argument("--sam-config", default=SAM_CONFIG)
    ap.add_argument("--fps", type=int, default=15, help="output video fps")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. slice the clip into a frames folder (SAM2 video predictor wants this).
    frames_dir = os.path.join(tempfile.gettempdir(), "sam2_frames")
    if os.path.isdir(frames_dir):
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
    n = extract_frames(args.clip, args.start, args.num, frames_dir)
    print(f"extracted {n} frames (global {args.start}..{args.start + n - 1})")
    if n == 0:
        print("No frames extracted -- check --clip path.")
        return

    # 2. detect the vehicle box on the annotation frame.
    ann_path = os.path.join(frames_dir, f"{args.ann_idx:05d}.jpg")
    proc, gdino = load_gdino(device)
    boxes, scores = detect_boxes(proc, gdino, Image.open(ann_path).convert("RGB"), device)
    box = pick_target_box(boxes, scores)
    if box is None:
        print(f"No vehicle detected on frame {args.ann_idx}; try another --ann-idx.")
        return
    print(f"target box on frame {args.ann_idx}: {[round(v) for v in box]}")
    del gdino  # free VRAM before loading SAM2
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. SAM2: seed with the box, propagate across the window.
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam_config, args.sam_checkpoint, device=device)

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else torch.autocast("cpu", enabled=False))
    with torch.inference_mode(), autocast:
        state = predictor.init_state(video_path=frames_dir)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=args.ann_idx, obj_id=1,
            box=np.array(box, dtype=np.float32))

        masks = {}
        for out_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = (mask_logits[0] > 0.0).cpu().numpy()
            masks[out_idx] = m[0] if m.ndim == 3 else m

    # 4. write the overlay video.
    os.makedirs(args.out, exist_ok=True)
    h, w = cv2.imread(ann_path).shape[:2]
    writer = cv2.VideoWriter(os.path.join(args.out, "track.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    covered = 0
    for i in range(n):
        frame = cv2.imread(os.path.join(frames_dir, f"{i:05d}.jpg"))
        mask = masks.get(i)
        if mask is not None and mask.any():
            covered += 1
        writer.write(overlay(frame, mask, box if i == args.ann_idx else None))
    writer.release()

    print(f"tracked mask present in {covered}/{n} frames")
    print(f"wrote {os.path.join(args.out, 'track.mp4')}")


if __name__ == "__main__":
    main()
