"""
Open-vocabulary 2D object detection over the equirectangular camera image,
plus a coarse color-attribute estimate per detection crop.

Uses HF Owlv2 (portable, runs fine on the RTX 4090 eval box) rather than the
TensorRT NanoOWL engine from the TB4 stack -- that engine was hand-built for
the Orin's 8 SMs and won't transfer to a different GPU without a rebuild.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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

        NOTE: Detection2D.label is the prompt verbatim, article included
        ("a television"). Strip the article before matching against scene-graph
        labels or question categories.
        """
        inputs = self.processor(text=[vocabulary], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=DETECTION_SCORE_THRESHOLD
        )[0]
        labels = results["labels"].unique()
        
        # if len(results["labels"].unique()) < len(vocabulary):
        #     return []
        # print(labels)
        detections = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            label = vocabulary[int(label_idx)]
            crop = image.crop((x0, y0, x1, y1))
            color = estimate_dominant_color(crop)
            detections.append(Detection2D(label=label, score=float(score), box_xyxy=(x0, y0, x1, y1), color=color))
        return detections

    def detect_and_save(self, image: Image.Image, vocabulary: List[str],
                        out_dir: str, tag: str = "") -> Tuple[List[Detection2D], Optional[str]]:
        """detect(), then write an annotated copy. Returns (detections, path).

        Convenience for debugging -- call detect() directly in the hot loop and
        save on a throttle, since a 10 Hz camera fills a disk quickly.
        """
        detections = self.detect(image, vocabulary)
        path = save_detections_debug(image, detections, out_dir, tag=tag, vocabulary=vocabulary)
        return detections, path


# Cycled per distinct label so multi-object frames stay readable.
_BOX_PALETTE = [
    (235, 64, 52), (52, 168, 235), (72, 200, 96), (245, 166, 35),
    (168, 92, 235), (235, 52, 168), (35, 210, 200), (200, 200, 60),
]


def save_detections_debug(image: Image.Image,
                          detections: Sequence[Detection2D],
                          out_dir: str,
                          tag: str = "",
                          vocabulary: Optional[Sequence[str]] = None,
                          max_side: int = 1920) -> Optional[str]:
    """Write `image` with detection boxes drawn to `out_dir`, return the path.

    Draws the label, score and estimated color per box. When `vocabulary` is
    given it is stamped in the corner -- worth doing, because the prompt set
    changes per question and a missed object usually means the prompt was
    wrong rather than the model being blind.

    Returns None (never raises) if anything goes wrong; debug output must not
    be able to take down the pipeline.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        img = image.convert("RGB").copy()

        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.BILINEAR)
        else:
            scale = 1.0

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(14)
        except TypeError:      # Pillow < 10.1 has no size argument
            font = ImageFont.load_default()

        label_colors = {}
        for det in detections:
            rgb = label_colors.setdefault(
                det.label, _BOX_PALETTE[len(label_colors) % len(_BOX_PALETTE)])

            x0, y0, x1, y1 = [int(v * scale) for v in det.box_xyxy]
            draw.rectangle([x0, y0, x1, y1], outline=rgb, width=3)

            caption = f"{det.label} {det.score:.2f}"
            if det.color and det.color != "unknown":
                caption += f" [{det.color}]"
            tw = draw.textlength(caption, font=font)
            ty = y0 - 16 if y0 >= 16 else y1
            draw.rectangle([x0, ty, x0 + tw + 6, ty + 16], fill=rgb)
            draw.text((x0 + 3, ty + 1), caption, fill=(255, 255, 255), font=font)

        banner = f"{len(detections)} det"
        if vocabulary:
            shown = ", ".join(vocabulary[:6])
            if len(vocabulary) > 6:
                shown += f", +{len(vocabulary) - 6}"
            banner += f" | prompts: {shown}"
        bw = draw.textlength(banner, font=font)
        draw.rectangle([0, 0, bw + 8, 18], fill=(0, 0, 0))
        draw.text((4, 2), banner, fill=(255, 255, 255), font=font)

        stamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:40]
        name = f"det_{stamp}_{len(detections):02d}{('_' + safe_tag) if safe_tag else ''}.jpg"
        path = os.path.join(out_dir, name)
        img.save(path, quality=85)
        return path
    except Exception:
        return None


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
