"""Per-clip detection driver + the Grounding DINO detector it uses.

Design: the driver `detect_clip` is DETECTOR-AGNOSTIC -- it takes any callable
`detector(pil_image) -> (boxes, scores, labels)`. That lets us:
  - run the real `GroundingDinoDetector` on the GPU box, and
  - test the orchestration locally with a tiny stub detector (no torch needed).

Flow per clip:  sample frames -> detector -> clean_detections -> {frame_idx: [Detection]}
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import cv2
from PIL import Image

from .detection import Detection, clean_detections
from .sampling import sample_frames

# A detector is any callable: PIL.Image -> (boxes, scores, labels)
Detector = Callable[[Image.Image], tuple[list, list, list]]


class GroundingDinoDetector:
    """Grounding DINO wrapped as a detector callable.

    Requires torch + transformers, so this only runs on the GPU box. Loaded
    lazily in __init__ so importing this module locally (for the driver/stub)
    never needs torch.
    """

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base",
                 device: Optional[str] = None,
                 prompt: str = "car. suv. truck. van. bus. pickup truck.",
                 box_threshold: float = 0.20, text_threshold: float = 0.20):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(model_id).to(self.device).eval())
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def __call__(self, image: Image.Image):
        inputs = self.processor(images=image, text=self.prompt,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        fn = self.processor.post_process_grounded_object_detection
        params = inspect.signature(fn).parameters
        kw = {"target_sizes": [image.size[::-1]], "text_threshold": self.text_threshold}
        kw["threshold" if "threshold" in params else "box_threshold"] = self.box_threshold
        r = fn(outputs, inputs["input_ids"], **kw)[0]
        labels = r.get("text_labels", r.get("labels"))
        return r["boxes"].tolist(), r["scores"].tolist(), list(labels)


def detect_clip(clip_path: str, detector: Detector, *,
                step: int = 10, start: int = 0, end: Optional[int] = None,
                score_thresh: float = 0.25, iou_thresh: float = 0.5
                ) -> dict[int, list[Detection]]:
    """Run detection on sampled frames of one clip.

    Returns {global_frame_index: [clean Detection, ...]} -- the input Step 3's
    tracker consumes.
    """
    results: dict[int, list[Detection]] = {}
    for idx, frame_bgr in sample_frames(clip_path, step=step, start=start, end=end):
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, scores, labels = detector(image)
        results[idx] = clean_detections(boxes, scores, labels,
                                        score_thresh=score_thresh,
                                        iou_thresh=iou_thresh)
    return results
