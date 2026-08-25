"""
Open-vocabulary 2D object detection over the equirectangular camera image,
plus a coarse color-attribute estimate per detection crop.

Uses Grounding DINO (IDEA-Research/grounding-dino-tiny) rather than the
TensorRT NanoOWL engine from the TB4 stack -- that engine was hand-built for
the Orin's 8 SMs and won't transfer to a different GPU without a rebuild.
Switched from OWL-ViT/Owlv2 on 2026-08-07 after live testing showed Owlv2's
false-positive rate was the actual precision bottleneck in the whole
pipeline -- see the MODEL SWAP note in config.py for the full reasoning.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .config import (
    GROUNDING_DINO_BOX_THRESHOLD,
    GROUNDING_DINO_TEXT_THRESHOLD,
    NAMED_COLORS_RGB,
    OPEN_VOCAB_MODEL_NAME,
    TILE_HFOV_DEG,
    TILE_OUTPUT_HEIGHT,
    TILE_OUTPUT_WIDTH,
    TILE_VFOV_DEG,
    TILE_YAW_CENTERS_DEG,
)
from .ros_utils import equirect_to_perspective_tile, perspective_box_to_equirect_box


@dataclass
class Detection2D:
    label: str
    score: float
    box_xyxy: Tuple[float, float, float, float]  # pixel coords in the full equirect image (may exceed IMAGE_WIDTH near the yaw seam -- see ros_utils.perspective_box_to_equirect_box)
    color: str


class OpenVocabDetector:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(OPEN_VOCAB_MODEL_NAME)
        # BF16 REVERTED 2026-08-17: bfloat16 model loading was tried for
        # memory savings (see history below), but confirmed live tonight
        # that it crashes the node entirely --
        # RuntimeError: expected scalar type BFloat16 but found Float,
        # inside Grounding DINO's deformable-attention grid_sample op.
        # Casting pixel_values to match model dtype (see the old DTYPE FIX
        # note, now removed below) only covers the INPUT tensor -- Grounding
        # DINO's deformable attention internally constructs sampling grids
        # procedurally in float32 regardless of model dtype, so the
        # mismatch happens deep inside the model's own forward pass, not
        # from anything this code controls. This is a known-bad model/dtype
        # combination -- do not reintroduce bf16 for this model. Reverted
        # to the default fp32 load; costs more resident memory (~440MB vs
        # ~220MB) but a stable node beats a faster-but-crashing one.
        #
        # Original memory-savings rationale, kept for context: there's no
        # working CUDA on this box (driver 535.309.01 < the 550+ CUDA
        # needs), so this model, Ollama, and the Unity sim all compete for
        # the same host RAM instead of the model living in separate GPU
        # memory. If memory pressure becomes a real problem again, look at
        # other mitigations (e.g. a smaller checkpoint, or int8
        # quantization via bitsandbytes) rather than bf16 for this
        # specific model.
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            OPEN_VOCAB_MODEL_NAME
        ).to(self.device)
        self.model.eval()

    def detect_360(self, equirect_image: Image.Image, vocabulary: List[str]) -> List[Detection2D]:
        """
        Runs detection across the full 360 view by splitting it into several
        normal-looking rectilinear sub-views (tiles) rather than feeding the
        raw equirectangular frame directly to the detector.

        WHY 2026-08-10: confirmed live (arabic_room, both Owlv2 and Grounding
        DINO) that feeding the raw equirect frame directly produced confident
        false-positive "sofa" detections landing specifically on bare walls
        -- no consistent other-object explanation, just wall regions,
        including at close range where localization error can't explain it.
        Equirectangular projection warps straight lines into curves,
        especially away from the horizontal center; a detector trained on
        normal photographs has essentially never seen this distortion and
        may be misreading warped wall geometry as furniture-like shapes.
        This reprojects each tile back to a normal, undistorted-looking crop
        the way a standard camera would see it before handing it to the
        detector, and maps detections back into full-equirect coordinates
        afterward so the rest of the pipeline (localize_detection_3d etc.)
        is unaffected.

        Tiles overlap (see TILE_YAW_CENTERS_DEG / TILE_HFOV_DEG in config.py)
        so an object near one tile's edge is very likely fully visible,
        un-truncated, in a neighboring tile -- the PARTIAL-VIEW REJECTION
        edge-check in detect() then does its job on genuinely better data
        instead of being the only defense against panorama-seam artifacts.
        No cross-tile duplicate suppression is done here; overlapping tiles
        can produce more than one detection of the same real object, but the
        existing 3D-distance-based scene-graph dedup (SceneGraph.add_or_merge
        / consolidate) already handles that at the point where it matters.
        """
        equirect_arr = np.asarray(equirect_image.convert("RGB"))
        all_detections: List[Detection2D] = []
        # MEMORY FIX 2026-08-11: tiles were already processed one at a time
        # (this loop, not a batch), which is good -- but the per-tile numpy
        # array, PIL image, and (inside detect()) tensor activations were
        # left for the garbage collector to reclaim on its own schedule,
        # rather than being released immediately once that tile's result is
        # in hand. Under real memory pressure (see the bfloat16 comment on
        # __init__ above for why that pressure exists on this machine) that
        # delay is exactly the kind of thing that turns "slow" into "OOM
        # crash" -- six tiles' worth of un-reclaimed intermediate arrays can
        # coexist in memory simultaneously if GC doesn't run between them.
        # Explicit del + a forced gc.collect() after each tile trades a
        # small, predictable CPU cost (gc.collect() is not free) for a hard
        # ceiling on how much intermediate memory can pile up at once --
        # worth it here since the crash is the actual problem being solved,
        # not raw throughput.
        import gc
        for yaw_center in TILE_YAW_CENTERS_DEG:
            tile_arr = equirect_to_perspective_tile(
                equirect_arr, yaw_center, TILE_HFOV_DEG, TILE_VFOV_DEG, TILE_OUTPUT_WIDTH, TILE_OUTPUT_HEIGHT
            )
            tile_image = Image.fromarray(tile_arr)
            tile_detections = self.detect(tile_image, vocabulary)
            for det in tile_detections:
                eq_box = perspective_box_to_equirect_box(
                    det.box_xyxy, yaw_center, TILE_HFOV_DEG, TILE_VFOV_DEG, TILE_OUTPUT_WIDTH, TILE_OUTPUT_HEIGHT
                )
                all_detections.append(Detection2D(label=det.label, score=det.score, box_xyxy=eq_box, color=det.color))
            del tile_arr, tile_image, tile_detections
            gc.collect()
        return all_detections

    def detect(self, image: Image.Image, vocabulary: List[str]) -> List[Detection2D]:
        """
        vocabulary: list of noun phrases to look for, e.g. ["a chair", "a trash can"].

        Grounding DINO's text interface differs from OWL-ViT's: instead of a
        list of separate prompts matched against fixed label indices, it
        takes ONE string with each phrase separated by ". " (lowercase,
        period-terminated is the documented convention) and returns the
        actual matched text span per detection rather than an index into
        the input list -- no manual vocabulary[label_idx] lookup needed.

        box_xyxy on the returned Detection2D is in THIS image's own pixel
        space -- when called per-tile via detect_360(), that's tile space,
        not full-equirect space; detect_360() remaps afterward.
        """
        text = ". ".join(v.lower() for v in vocabulary) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        # DTYPE CAST REMOVED 2026-08-17: was needed only while the model
        # loaded in bfloat16 (see the BF16 REVERTED note in __init__) --
        # now that the model is back to the processor's default fp32,
        # pixel_values already matches the model's dtype with no cast
        # needed.
        # torch.inference_mode() is stricter than no_grad() (also disables
        # version-counter tracking used for autograd correctness checks,
        # which this inference-only path never needs) and is the officially
        # recommended replacement for pure-inference code paths -- slightly
        # less overhead per call, which adds up over 6 tiles x every sampled
        # frame.
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = [image.size[::-1]]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # FIX 2026-08-10: confirmed live crash on real transformers
            # install -- the installed version's
            # post_process_grounded_object_detection() takes `threshold`,
            # not `box_threshold` (checked the actual installed signature
            # rather than assume from memory/docs, which is what caused
            # this in the first place -- HF appears to have renamed this
            # param at some point, likely to unify the API surface with
            # OWL-ViT's post-processing method, which also uses `threshold`).
            threshold=GROUNDING_DINO_BOX_THRESHOLD,
            text_threshold=GROUNDING_DINO_TEXT_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        # Different transformers versions have returned this matched-text
        # field under slightly different keys -- check both rather than
        # assuming one, so a version bump doesn't silently break detection.
        text_labels = results.get("text_labels") or results.get("labels")

        # FIX 2026-08-10: use the ACTUAL passed-in image's dimensions, not
        # the global IMAGE_WIDTH/IMAGE_HEIGHT constants -- this method is now
        # called per-tile (tile-sized images) via detect_360(), not just on
        # the full equirect frame, so a hardcoded global would silently
        # check edges against the wrong dimensions.
        img_w, img_h = image.size

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], text_labels):
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            edge_margin_px = 3
            touches_edge = (
                x0 <= edge_margin_px or x1 >= img_w - edge_margin_px or
                y0 <= edge_margin_px or y1 >= img_h - edge_margin_px
            )
            # TEMP DEBUG 2026-08-17: targeted diagnostic for "column" raw
            # detections, at the earliest possible point (straight out of
            # DINO, before ANY filtering). Columns are TALL -- likely to
            # span much of a tile's vertical extent -- making them
            # structurally prone to touching the top/bottom edge and
            # getting caught by PARTIAL-VIEW REJECTION below, unlike wider/
            # shorter furniture. This print shows the raw box, score, and
            # whether edge-rejection is about to discard it. REMOVE once
            # root cause of the missing-column detections is found.
            if "column" in str(label).lower() or "pillar" in str(label).lower():
                print(f"[DEBUG column RAW] label={label!r} score={score:.3f} box=({x0},{y0},{x1},{y1}) "
                      f"img=({img_w}x{img_h}) touches_edge={touches_edge}")
            if touches_edge:
                continue
            crop = image.crop((x0, y0, x1, y1))
            color = estimate_dominant_color(crop)
            detections.append(Detection2D(label=str(label), score=float(score), box_xyxy=(x0, y0, x1, y1), color=color))
        return detections


def estimate_dominant_color(crop: Image.Image) -> str:
    """Classify each pixel against the named palette individually, then take
    the most common result (mode), rather than averaging RGB first and
    classifying once.

    FIX 2026-08-10: confirmed live that a genuinely red-and-black two-tone
    sofa was being labeled "brown"/"black", never "red". Averaging RGB
    first is the root cause -- averaging a red region and a black region
    together produces a dark muted color that resembles neither original
    color, landing closer to "brown"/"black" in the palette than to either
    true color. Classifying per-pixel and taking the mode instead correctly
    identifies whichever true color covers the most area, since it never
    blends two real colors into a third, unreal one.
    """
    if crop.width == 0 or crop.height == 0:
        return "unknown"
    # Sample the center ~60% of the crop, not the full box -- edges are
    # where background/adjacent objects are most likely to bleed into a
    # loosely-fit detection box, which would otherwise get voted on too.
    w, h = crop.width, crop.height
    margin_x, margin_y = int(w * 0.2), int(h * 0.2)
    center_crop = crop.crop((margin_x, margin_y, w - margin_x, h - margin_y))
    if center_crop.width == 0 or center_crop.height == 0:
        center_crop = crop

    arr = np.asarray(center_crop.convert("RGB")).reshape(-1, 3).astype(np.float32)
    palette_names = list(NAMED_COLORS_RGB.keys())
    palette_rgb = np.array([NAMED_COLORS_RGB[n] for n in palette_names], dtype=np.float32)

    # Vectorized per-pixel nearest-color classification: (n_pixels, n_colors)
    # distance matrix, then argmin per pixel -- fast enough for a small crop.
    dists = np.linalg.norm(arr[:, None, :] - palette_rgb[None, :, :], axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    counts = np.bincount(nearest_idx, minlength=len(palette_names))
    return palette_names[int(np.argmax(counts))]
