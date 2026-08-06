"""
Open-vocabulary 2D object detection over the equirectangular camera image,
plus a coarse color-attribute estimate per detection crop.

Uses HF Owlv2 (portable, runs fine on the RTX 4090 eval box) rather than the
TensorRT NanoOWL engine from the TB4 stack -- that engine was hand-built for
the Orin's 8 SMs and won't transfer to a different GPU without a rebuild.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from config import DETECTION_SCORE_THRESHOLD, NAMED_COLORS_RGB, OPEN_VOCAB_MODEL_NAME


@dataclass
class Detection2D:
    label: str
    score: float
    box_xyxy: Tuple[int, int, int, int]  # pixel coords in the full equirect image
    color: str


class OpenVocabDetector:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(OPEN_VOCAB_MODEL_NAME)
        self.model = Owlv2ForObjectDetection.from_pretrained(OPEN_VOCAB_MODEL_NAME).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: Image.Image, vocabulary: List[str]) -> List[Detection2D]:
        """
        vocabulary: list of noun phrases to look for, e.g. ["a chair", "a trash can"].
        Prompts should read as noun phrases ("a chair"), not bare nouns --
        matches OWL-family conventions and what worked reliably in the
        NanoOWL landmark-tagging pipeline.
        """
        inputs = self.processor(text=[vocabulary], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=DETECTION_SCORE_THRESHOLD
        )[0]

        detections = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            label = vocabulary[int(label_idx)]
            crop = image.crop((x0, y0, x1, y1))
            color = estimate_dominant_color(crop)
            detections.append(Detection2D(label=label, score=float(score), box_xyxy=(x0, y0, x1, y1), color=color))
        return detections


def estimate_dominant_color(crop: Image.Image) -> str:
    """Very coarse: mean RGB of the crop, nearest-neighbor to a fixed palette.
    TODO: this washes out under mixed lighting / textured objects -- if this
    becomes a scoring bottleneck, replace with a small CLIP-text similarity
    over color+object prompts (e.g. "a blue chair" vs "a red chair") instead
    of raw pixel averaging, which tends to generalize better across lighting.
    """
    if crop.width == 0 or crop.height == 0:
        return "unknown"
    arr = np.asarray(crop.convert("RGB")).reshape(-1, 3).astype(np.float32)
    mean_rgb = arr.mean(axis=0)

    best_name, best_dist = "unknown", float("inf")
    for name, rgb in NAMED_COLORS_RGB.items():
        dist = np.linalg.norm(mean_rgb - np.array(rgb, dtype=np.float32))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name
