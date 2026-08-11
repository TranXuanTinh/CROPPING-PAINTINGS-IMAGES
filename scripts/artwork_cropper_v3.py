#!/usr/bin/env python3
"""
artwork_cropper_v3.py
─────────────────────
Production artwork cropper for View at Home (VAH).
Crops raw artwork images from their backgrounds — removes walls, room scenes,
studio setups, frames, and mats — leaving only the artwork surface.

Built to handle wildly varied real-world inputs from hundreds of different
galleries, artists and e-commerce stores: studio mockups, room environment
photos, angled/leaning paintings, passepartout-matted prints, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE (per image)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Florence-2-large semantic detection
       Microsoft's Florence-2-large vision-language model understands what a
       "painting" or "artwork" IS in context.  It handles room scenes, easels,
       shelves, complex backgrounds, and angled shots that defeat pure
       pixel-statistics approaches.  Returns a bounding box.

       Retry-with-fallback: if the first detection pass (specific→general
       prompt order) finds nothing, a second pass is attempted with the
       prompts in reverse order (general→specific).  Only if both passes
       find nothing does the pipeline fall through to the OpenCV fallback.

  2. CLAHE edge refinement (always runs)
       CLAHE (Contrast Limited Adaptive Histogram Equalization) amplifies subtle
       luminance differences before Canny edge detection runs, allowing the
       boundary to be snapped to the true artwork edge even when Florence-2
       locked onto the content inside a mat rather than the mat itself.

       Two-step with an explicit gate:
         Step 1 (local, always)  — small outward search, safe for all scenes.
         Gate (background check) — Step 2 only fires on uniform backgrounds.
         Step 2 (full-image)     — finds wide mats/passepartouts on studio shots.
       The gate is the key architectural separation between studio crops
       (where expansion is correct) and room scene crops (where expansion
       would grab walls and furniture).

  3. Perspective correction
       Detects four-sided quadrilaterals within the detected bbox.  When a
       convincing quad is found (painting is angled, leaning, on easel),
       applies a four-point perspective warp to produce a front-facing crop.
       Falls back to a simple rectangular crop when the painting is face-on.

  4. Optional frame trim  (--frame-trim flag, off by default)
       Scans inward from each edge using per-row/column pixel variance.
       Frames are low-variance (uniform colour); artwork content is higher-
       variance.  Trims the uniform frame border to expose the canvas surface.
       VAH renders its own 3D frames, so physical frames must be excluded.

  5. OpenCV fallback
       When both Florence-2 passes find nothing plausible, falls back to the
       original rule-based artwork_cropper.py pipeline.  Never silently drops
       an image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCIES  (install before running)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pip install torch torchvision "transformers==4.48.0" einops timm pillow opencv-python numpy

  Exact versions that have been tested:
    torch            2.12.0
    torchvision      0.27.0
    transformers     4.48.0   ← pin this; 5.x breaks Florence-2's custom config
    einops           0.8.2
    timm             1.0.27
    pillow           12.2.0
    opencv-python    any recent (4.8+)
    numpy            any recent (1.24+)

  Model weights are downloaded automatically on first run from HuggingFace
  (~1.5 GB for Florence-2-large, cached at ~/.cache/huggingface/).
  Subsequent runs load from cache with no network required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARDWARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Runs on CPU only.  This is a deliberate deployment choice: CPU inference
  is fully deterministic across runs and across machines, which is required
  for a reproducible production pipeline on standard AWS EC2 instances.

  MPS (Apple Silicon) and CUDA (NVIDIA GPU) are intentionally not used.
  MPS produces non-deterministic bounding boxes due to floating-point
  reduction order in the Metal backend.  CUDA requires GPU instances.
  CPU inference on Florence-2-large takes ~15–40 s per image depending on
  instance size; this is acceptable for a batch processing pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  artwork_cropper_v3.py   — this file (primary script)
  artwork_cropper.py      — OpenCV fallback pipeline (must be in same directory)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLI USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Single image:
    python artwork_cropper_v3.py painting.jpg
    python artwork_cropper_v3.py painting.jpg --output cropped.jpg

  Batch (entire folder, recursively, preserves subfolder structure):
    python artwork_cropper_v3.py ./images/ --batch
    python artwork_cropper_v3.py ./images/ --batch --output ./out/

  Frame trimming (removes physical picture frame, exposes canvas surface):
    python artwork_cropper_v3.py painting.jpg --frame-trim
    python artwork_cropper_v3.py painting.jpg --frame-trim --frame-depth 0.15
    python artwork_cropper_v3.py painting.jpg --no-frame-trim   # explicit off
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ── Round-shape detection (reused from the OpenCV fallback pipeline) ──────────
# artwork_cropper.py contains robust geometric detection for circular / oval
# canvases (extent ≈ 0.785 across multiple thresholds).  We import the key
# functions so the v3 pipeline can short-circuit to ellipse-based cropping
# instead of forcing round paintings through the rectangular CLAHE / perspective
# / tightening pipeline, which mishandles them.
try:
    from artwork_cropper import (
        detect_round_shape as _detect_round_shape,
        crop_round as _crop_round,
    )
    _HAS_ROUND_DETECTION = True
except ImportError:
    _HAS_ROUND_DETECTION = False

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}

# Florence-2-large provides higher accuracy than the base model at ~2× memory
# (~1.5 GB) and ~1.5× inference time.  CPU inference is deterministic.
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"

# Detection prompts tried in order.  Iteration stops as soon as a high-scoring
# bbox is found.  Order matters: more specific → more general.
DETECTION_PROMPTS = ["painting", "artwork", "framed artwork", "canvas", "picture"]

# When Florence-2 detects only the CONTENT inside an artwork (e.g. the painted
# subject) rather than the physical artwork boundary (mat + frame), the detected
# bbox is typically < 20% of the image area.  In this case a second Florence-2
# pass runs with these frame-specific prompts, which ask the model to locate the
# physical frame rather than the subject inside it.  Common frame colours covered:
# wood (beige/tan/brown), black, white.
# Only fires for small-bbox detections — zero impact on all other image types.
SMALL_BBOX_THRESHOLD   = 0.20
FRAME_DETECTION_PROMPTS = [
    "picture frame", "frame", "wooden frame", "framed print", "framed photograph",
]

# Hard limits on what counts as a plausible detection.
BBOX_FRAC_MIN  = 0.025  # bbox must cover at least 2.5% of image area
                        # (lowered from 0.05 — small paintings in large room scenes
                        #  often occupy only ~3% of the image; Florence detects them
                        #  correctly but they were being rejected by the old threshold.
                        #  Safe because _score_bbox rewards area near 35%, so a 3%
                        #  bbox can only win if it's the sole detection anyway.)
BBOX_FRAC_MAX  = 0.999  # and at most 99.9%
                        # (raised from 0.95 — pre-cropped images and paintings that
                        #  fill the entire frame are legitimately 96-98% of the image.
                        #  Florence correctly detects them but they were being rejected.
                        #  Safe because Florence never returns a 97%+ bbox as a false
                        #  positive when a clearly bounded painting is visible.)
BBOX_ASPECT_MIN = 0.10  # width/height >= 0.10  (very tall thin strip = not art)
BBOX_ASPECT_MAX = 10.0  # width/height <= 10.0  (very wide thin strip = not art)

# Detections whose bbox area falls inside this "sweet spot" are treated as
# high-confidence.  Outside it → low-confidence → CLAHE refinement is applied.
BBOX_SWEETSPOT_MIN = 0.10
BBOX_SWEETSPOT_MAX = 0.85

# Early-exit threshold for the prompt loop: stop trying more prompts once a
# bbox scores above this.
PROMPT_EARLY_EXIT_SCORE = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Hardware detection
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Always returns "cpu".

    CPU is the only supported device for this deployment.  MPS (Apple Silicon)
    produces non-deterministic bounding boxes across runs; CUDA requires GPU
    instances.  CPU inference is fully deterministic on standard EC2 hardware.
    """
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model management — loaded once per process, reused across all images
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def load_model(device: Optional[str] = None) -> tuple:
    """
    Load the Florence-2-large model and processor, caching them for the
    lifetime of the process.  Safe to call multiple times; loading only
    happens once.

    The *device* argument is accepted for API compatibility but ignored —
    the model always loads on CPU for deterministic, reproducible inference.

    Returns:
        (model, processor, "cpu")
    """
    global _MODEL_CACHE
    if _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["processor"], _MODEL_CACHE["device"]

    # CPU is the only supported device (see module docstring).
    # Ignore any caller-supplied device override to prevent accidental GPU use.
    device = "cpu"

    # Use all available CPU cores for faster inference.
    # Previously pinned to 1 thread for byte-for-byte reproducibility across
    # EC2 instances, but the speed penalty (~4–8×) outweighs that benefit for
    # local batch processing.  Detection bboxes remain deterministic on the
    # same machine regardless of thread count.
    num_cores = os.cpu_count() or 1
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(max(1, num_cores // 2))

    print(f"[florence2] Loading {FLORENCE_MODEL_ID} on {device} …", flush=True)
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,   # float32 required for deterministic CPU math
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)

    _MODEL_CACHE["model"]     = model
    _MODEL_CACHE["processor"] = processor
    _MODEL_CACHE["device"]    = device

    print(f"[florence2] Ready in {time.time() - t0:.1f}s", flush=True)
    return model, processor, device


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Florence-2 detection
# ──────────────────────────────────────────────────────────────────────────────

# Maximum side length fed to Florence-2 for detection.  The model encodes
# images into a fixed spatial grid, so anything above ~800 px gives no
# extra detection quality but proportionally increases ViT patch encoding time.
# Bboxes are rescaled back to original coordinates after inference.
_FLORENCE_MAX_SIDE = 768


def _run_florence_detection(image: Image.Image, prompt_text: str,
                             model, processor, device: str) -> list[list[float]]:
    """
    Run one OPEN_VOCABULARY_DETECTION pass with *prompt_text* and return the
    raw list of bounding boxes (may be empty).

    Each bbox is [x1, y1, x2, y2] in image pixels (original resolution).
    Large images are downscaled to _FLORENCE_MAX_SIDE before inference and
    bboxes are rescaled back — this cuts ViT encoding time by 4–8× on
    typical artwork images without any loss in detection quality.
    """
    task   = "<OPEN_VOCABULARY_DETECTION>"
    prompt = f"{task}{prompt_text}"

    # ── Downscale large images before feeding to Florence-2 ────────────────
    orig_w, orig_h = image.width, image.height
    max_side = max(orig_w, orig_h)
    if max_side > _FLORENCE_MAX_SIDE:
        scale    = _FLORENCE_MAX_SIDE / max_side
        new_w    = max(1, int(orig_w * scale))
        new_h    = max(1, int(orig_h * scale))
        inf_image = image.resize((new_w, new_h), Image.BILINEAR)
    else:
        scale     = 1.0
        inf_image = image

    inputs = processor(text=prompt, images=inf_image, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,   # bbox outputs are ~10-20 tokens; 1024 is wasteful
            do_sample=False,
            num_beams=1,          # greedy decoding — 3× faster than beam=3 on CPU
            early_stopping=False, # suppress warning when num_beams=1
        )

    raw    = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # Decode against the *inference* image size — bboxes are in inf_image coords
    parsed = processor.post_process_generation(
        raw, task=task, image_size=(inf_image.width, inf_image.height)
    )
    bboxes = parsed.get(task, {}).get("bboxes", [])

    # Rescale bboxes back to original image coordinates
    if scale != 1.0 and bboxes:
        inv = 1.0 / scale
        bboxes = [
            [b[0] * inv, b[1] * inv, b[2] * inv, b[3] * inv]
            for b in bboxes
        ]
    return bboxes


def _score_bbox(bbox: list[float], img_w: int, img_h: int) -> float:
    """
    Score a bounding box on [0, 1] — used to select the best detection when
    Florence-2 returns multiple candidates.

    Scoring criteria:
      - Hard-rejected if area fraction or aspect ratio is outside limits.
      - Rewarded for area near 35% of the image (typical artwork-in-scene size).
      - Rewarded for being roughly centred (artworks are usually central).
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0

    frac   = (bw * bh) / (img_w * img_h)
    aspect = bw / bh

    if not (BBOX_FRAC_MIN < frac < BBOX_FRAC_MAX):
        return 0.0
    if not (BBOX_ASPECT_MIN < aspect < BBOX_ASPECT_MAX):
        return 0.0

    # Prefer fracs near 35% — penalise linearly as frac moves away
    frac_score   = max(0.0, 1.0 - abs(frac - 0.35) / 0.35)
    # Prefer bboxes whose centre is near the image centre
    cx, cy       = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
    center_score = max(0.0, 1.0 - (abs(cx - 0.5) + abs(cy - 0.5)))

    return (frac_score + center_score) / 2


def _detect_pass(
    image: Image.Image,
    prompts: list[str],
    model,
    processor,
    device: str,
) -> tuple[Optional[list[float]], float]:
    """
    Single detection pass: try each prompt in *prompts* order, return the
    (best_bbox, best_score) found.  Returns (None, 0.0) if nothing plausible
    was detected.
    """
    img_w, img_h = image.width, image.height
    best_bbox, best_score = None, 0.0

    for prompt in prompts:
        bboxes = _run_florence_detection(image, prompt, model, processor, device)
        for bbox in bboxes:
            score = _score_bbox(bbox, img_w, img_h)
            if score > best_score:
                best_score, best_bbox = score, bbox
        if best_score >= PROMPT_EARLY_EXIT_SCORE:
            break   # strong detection found — no need to try further prompts

    return best_bbox, best_score


def detect_artwork(
    image: Image.Image,
    model,
    processor,
    device: str,
) -> tuple[Optional[list[float]], str]:
    """
    Run Florence-2-large to locate the artwork in *image*.

    Pass 1 — DETECTION_PROMPTS in their defined order (specific → general).
    Pass 2 — if pass 1 finds nothing, retries with prompts in REVERSE order
              (general → specific).  A different ordering can unlock detections
              that the first pass missed due to early-exit behaviour.

    Only falls through to "none" if both passes find nothing plausible.

    Returns:
        (bbox, confidence)

        bbox:
            [x1, y1, x2, y2] in image pixels, or None if nothing plausible
            was found.

        confidence:
            "high"  — bbox area is in the sweet-spot (10–85%).
                      Proceed directly to perspective correction.
            "low"   — a bbox was found but its area is outside the sweet-spot.
                      Apply CLAHE refinement before perspective correction.
            "none"  — both detection passes found nothing plausible.
                      Fall back to the OpenCV rule-based pipeline.
    """
    img_w, img_h = image.width, image.height

    # Pass 1: specific → general (standard order)
    best_bbox, best_score = _detect_pass(image, DETECTION_PROMPTS, model, processor, device)

    if best_bbox is None:
        # Pass 2: general → specific (reversed order)
        print("  [florence2] Pass 1 found nothing — retrying with reversed prompts.",
              flush=True)
        best_bbox, best_score = _detect_pass(
            image, list(reversed(DETECTION_PROMPTS)), model, processor, device
        )

    if best_bbox is None:
        return None, "none"

    # ── Frame-specific pass for content-only detections ───────────────────
    # When Florence-2 found only the painted SUBJECT (e.g. flowers, faces,
    # objects) rather than the full artwork boundary, the bbox is much smaller
    # than the physical painting.  A third pass with frame-oriented prompts
    # asks Florence-2 to find the frame itself — wood, black, or white —
    # which it handles reliably via semantic understanding even on complex
    # backgrounds where CLAHE pixel-based edge detection fails.
    #
    # Guards prevent this from grabbing unrelated frames:
    #   1. The frame detection must be meaningfully LARGER than the content
    #      detection (frame_frac > content_frac × 1.5).
    #   2. The frame centre must be within 35 % of the content centre
    #      (normalised to image dimensions).
    #
    # Only fires when bbox_frac < SMALL_BBOX_THRESHOLD (20 %).
    # Has zero effect on all other image types.
    x1, y1, x2, y2 = best_bbox
    current_frac    = (x2 - x1) * (y2 - y1) / (img_w * img_h)
    if current_frac < SMALL_BBOX_THRESHOLD:
        print(f"  [florence2] Content-only detection ({current_frac:.2f} of image) "
              f"— running frame prompts …", flush=True)
        frame_bbox, frame_score = _detect_pass(
            image, FRAME_DETECTION_PROMPTS, model, processor, device
        )
        if frame_bbox is not None:
            fx1, fy1, fx2, fy2 = frame_bbox
            frame_frac  = (fx2 - fx1) * (fy2 - fy1) / (img_w * img_h)
            orig_cx     = (x1 + x2) / 2 / img_w
            orig_cy     = (y1 + y2) / 2 / img_h
            frame_cx    = (fx1 + fx2) / 2 / img_w
            frame_cy    = (fy1 + fy2) / 2 / img_h
            centre_dist = abs(frame_cx - orig_cx) + abs(frame_cy - orig_cy)
            if frame_frac > current_frac * 1.5 and centre_dist < 0.35:
                print(f"  [florence2] Frame found: {round(current_frac*100)}% content "
                      f"→ {round(frame_frac*100)}% frame  bbox={[round(v) for v in frame_bbox]}",
                      flush=True)
                best_bbox = frame_bbox
            else:
                print(f"  [florence2] Frame prompts returned bbox but failed guards "
                      f"(frame_frac={frame_frac:.2f} centre_dist={centre_dist:.2f}) — "
                      f"keeping content bbox", flush=True)
        else:
            print(f"  [florence2] Frame prompts found nothing — "
                  f"proceeding with content bbox", flush=True)

    x1, y1, x2, y2 = best_bbox
    frac = (x2 - x1) * (y2 - y1) / (img_w * img_h)
    confidence = "high" if BBOX_SWEETSPOT_MIN <= frac <= BBOX_SWEETSPOT_MAX else "low"
    return best_bbox, confidence


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CLAHE edge refinement
# ──────────────────────────────────────────────────────────────────────────────

# Uniformity threshold: background strips with pixel std below this value are
# considered simple enough (mat / studio / solid wall) to allow outward
# expansion.  Strips above this are considered complex (room scene, furniture,
# textured wall) and expansion is suppressed.
#
# Physical basis:
#   Solid white mat:          std  3–8
#   Light grey studio bg:     std  8–18
#   Off-white / cream mat:    std 10–20
#   Plain painted wall:       std 15–25   (borderline — F2 finds painting fine)
#   Wood-grain wall:          std 30–50
#   Room scene / furniture:   std 50–150+
CLAHE_BG_UNIFORM_THRESHOLD = 25.0


def _background_is_uniform(img_bgr: np.ndarray, bbox: list[float]) -> bool:
    """
    Return True when the background outside *bbox* is uniform enough for
    outward mat-expansion to be safe.

    Checks the four border strips (top / bottom / left / right) between the
    bbox edges and the image boundary.  If ANY strip has a grayscale std ≥
    CLAHE_BG_UNIFORM_THRESHOLD the background is considered complex (room
    scene, textured wall) and False is returned.  Strips thinner than 5 px
    are skipped — the bbox fills that edge of the image.

    This is the architectural gate that separates the two CLAHE use-cases:
      • Uniform background → studio / mockup / mat → expand to find mat edge.
      • Complex background → room / real scene → trust Florence-2, no expansion.

    The threshold is a physical property of what mats and room scenes look
    like, not a parameter tuned to the test set, so it generalises to future
    images without adjustment.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    MIN_STRIP = 5

    strips: list[np.ndarray] = []
    if y1 > MIN_STRIP:      strips.append(img_bgr[:y1, :])
    if h - y2 > MIN_STRIP:  strips.append(img_bgr[y2:, :])
    if x1 > MIN_STRIP:      strips.append(img_bgr[:, :x1])
    if w - x2 > MIN_STRIP:  strips.append(img_bgr[:, x2:])

    if not strips:
        return False   # bbox fills the image — no background to examine

    for strip in strips:
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) >= CLAHE_BG_UNIFORM_THRESHOLD:
            return False

    return True


def clahe_refine_boundary(
    img_bgr:    np.ndarray,
    bbox:       list[float],
    expand_pct: float = 0.15,
) -> list[float]:
    """
    Snap a bounding box to the true artwork edge using CLAHE-enhanced edges.

    Runs on EVERY detection (not only low-confidence ones) to fix:
      1. Tightening: bbox too loose (wall / shadow included) → trimmed inward.
      2. Expanding:  Florence-2 locks onto content inside a mat / passepartout
                     → expanded outward to the physical artwork boundary.

    Architecture — two steps with an explicit gate between them:

    Step 1 — local search (always runs):
        Searches a region expanded outward by *expand_pct* (15% default) from
        the Florence-2 bbox.  EDGE_MARGIN = 8 px keeps image-edge artefacts out.
        Accepts results that are ≥ 50% and ≤ 3× of the original bbox area.
        Safe for all image types including room scenes.

        Content-only detections:
        When Florence-2 detects only the subject inside the artwork (e.g. the
        flowers inside a framed print), those cases are handled upstream in
        detect_artwork() via frame-specific prompts before reaching this
        function.  By the time clahe_refine_boundary() is called, the bbox
        should already represent the full artwork boundary.

    Gate — _background_is_uniform:
        Step 2 only fires when every border strip outside the Florence-2 bbox
        has pixel std < CLAHE_BG_UNIFORM_THRESHOLD (25).  This separates:
          • Studio / mat shots  (uniform bg, std <25) → Step 2 runs → mat found.
          • Room / complex scenes (complex bg, std ≥25) → Step 2 skipped →
            Florence-2 bbox trusted, no risk of grabbing walls or furniture.

    Step 2 — full-image search (uniform backgrounds only):
        Searches the entire image with CLAHE.  Edge margin = 5% of min image
        dimension (minimum 3 px) to handle small images correctly.  Upper
        bound = 92% of full image area (replaces the 3× multiplier which broke
        when Florence-2 returned a small content-only bbox and the mat was
        large).  Only accepted if the result is meaningfully larger than Step 1
        and its centre stays within 30% of the original bbox centre.

    Args:
        img_bgr:    Full input image in BGR (OpenCV format).
        bbox:       [x1, y1, x2, y2] from Florence-2 (or previous step).
        expand_pct: Step 1 outward expansion as a fraction of bbox side length.

    Returns:
        Refined [x1, y1, x2, y2], or the original bbox unchanged.
    """
    h, w      = img_bgr.shape[:2]
    img_area  = h * w
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh    = x2 - x1, y2 - y1
    orig_area = max(1, bw * bh)

    # ── Shared CLAHE + Canny routine ──────────────────────────────────────
    def _search(region: np.ndarray, rx1: int, ry1: int,
                 edge_margin: int, area_upper: float) -> Optional[list[float]]:
        if region.size == 0:
            return None
        lab   = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        cl    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray    = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 20, 80)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        rh_r, rw_r  = region.shape[:2]
        region_area  = rh_r * rw_r
        best_rect, best_area = None, 0
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if not (0.05 < area / region_area < 0.97):
                continue
            if (cx < edge_margin or cy < edge_margin
                    or cx + cw > rw_r - edge_margin
                    or cy + ch > rh_r - edge_margin):
                continue
            if area > best_area:
                best_area = area
                best_rect = (cx, cy, cw, ch)
        if best_rect is None:
            return None
        cx, cy, cw, ch = best_rect
        result_area = cw * ch
        if result_area < orig_area * 0.50:    # guard: no over-shrinking
            return None
        if result_area > area_upper:          # guard: no runaway expansion
            return None
        # Guard: for large Florence-2 bboxes (>50% of image), reject CLAHE
        # results that are < 70% of the original bbox area.  Internal artwork
        # features (textures, edges inside the painting) can form contours
        # that are smaller than the true artwork boundary, particularly for
        # square paintings with uniform borders.
        orig_frac_of_img = orig_area / img_area
        if orig_frac_of_img > 0.50 and result_area < orig_area * 0.70:
            return None
        return [rx1 + cx, ry1 + cy, rx1 + cx + cw, ry1 + cy + ch]

    # ── Step 1: local search (always runs, safe for all image types) ──────
    pad_x = int(bw * expand_pct)
    pad_y = int(bh * expand_pct)
    rx1 = max(0, x1 - pad_x);  ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x);  ry2 = min(h, y2 + pad_y)

    result = _search(img_bgr[ry1:ry2, rx1:rx2], rx1, ry1,
                     edge_margin=8, area_upper=orig_area * 3.0)
    if result is None:
        result = list(bbox)

    # ── Gate: only proceed to Step 2 on uniform (studio / mat) backgrounds ─
    uniform = _background_is_uniform(img_bgr, bbox)
    print(f"  [clahe]     bg={'uniform' if uniform else 'complex '}  "
          f"step1={[round(v) for v in result]}", flush=True)

    # ── Early-exit for well-detected paintings ────────────────────────────
    # If Florence already found the painting at ≥ 20 % of the image area,
    # it captured the full boundary (canvas + any mat/passepartout).
    # Expanding further via wall-edge scan or Step 2 risks including
    # surrounding wall space, which caused Cat 1 regressions on
    # white/grey-wall images.  Only small (<20 %) content-only detections
    # (e.g. erishimatsuka floral print inside a large mat) need expansion.
    orig_frac = orig_area / img_area
    if orig_frac >= 0.20:
        # Clamp: Step 1 can only tighten (move edges inward), never expand.
        # CLAHE contours on pale pastel paintings can include the surrounding
        # wall shadow, producing a larger-than-correct bbox.
        result = [
            max(result[0], float(x1)),   # left  — only inward
            max(result[1], float(y1)),   # top   — only inward
            min(result[2], float(x2)),   # right — only inward
            min(result[3], float(y2)),   # bottom — only inward
        ]
        print(f"  [clahe]     frac={orig_frac:.2f} ≥ 0.20 — trusting Florence bbox, "
              f"no expansion", flush=True)
        return result   # Step 1 only — no wall-edge scan, no Step 2

    if not uniform:
        # ── Mat check: secondary pass on the immediate bbox vicinity ─────
        # The broad background strips can register as "complex" even for a
        # mat-on-wall setup: the strips span from the bbox edge all the way
        # to the image boundary, so they contain BOTH white mat AND grey wall,
        # producing high std even though the mat itself is perfectly uniform.
        #
        # Secondary test: sample only the 10-px band immediately outside each
        # bbox edge.  If every surrounding strip is uniform (std < threshold)
        # we have a mat/passepartout and Step 2 should run to find its outer
        # boundary.  Room scenes and complex backgrounds never have a uniform
        # band immediately around the detected painting bbox, so this check
        # does not regress cat 7 or other environment categories.
        MAT_SAMPLE_PX = max(8, int(min(bw, bh) * 0.05))
        imm_strips = []
        if y1 >= MAT_SAMPLE_PX:
            imm_strips.append(img_bgr[max(0, y1 - MAT_SAMPLE_PX):y1, x1:x2])
        if h - y2 >= MAT_SAMPLE_PX:
            imm_strips.append(img_bgr[y2:min(h, y2 + MAT_SAMPLE_PX), x1:x2])
        if x1 >= MAT_SAMPLE_PX:
            imm_strips.append(img_bgr[y1:y2, max(0, x1 - MAT_SAMPLE_PX):x1])
        if w - x2 >= MAT_SAMPLE_PX:
            imm_strips.append(img_bgr[y1:y2, x2:min(w, x2 + MAT_SAMPLE_PX)])

        has_mat = bool(imm_strips) and all(
            float(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).std()) < CLAHE_BG_UNIFORM_THRESHOLD
            for s in imm_strips if s.size > 0
        )

        if not has_mat:
            # Complex room scene — clamp Step 1 so it can only tighten the
            # Florence bbox, never expand it.  CLAHE contours in room scenes
            # sometimes include adjacent wall pixels that grow the bbox.
            result = [
                max(result[0], float(x1)),   # left  — only inward moves
                max(result[1], float(y1)),   # top   — only inward moves
                min(result[2], float(x2)),   # right — only inward moves
                min(result[3], float(y2)),   # bottom — only inward moves
            ]
            return result   # complex scene — trust Florence-2, don't expand

        # ── Brightness guard: white/off-white mat only ───────────────────
        # The immediate-strip check can also fire when the painting hangs on
        # a plain painted wall (the narrow strip is uniform wall colour).
        # A genuine passepartout/mat is white or near-white (median ≥ 195).
        # Plain walls are typically 155–190.  Skip the wall-edge scan for
        # low-brightness immediate strips — fall through to CLAHE Step 2
        # instead, which is the safe path for room/environment scenes.
        all_strip_pixels = np.concatenate([
            cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).flatten()
            for s in imm_strips if s.size > 0
        ])
        mat_median = float(np.median(all_strip_pixels))
        if mat_median < 195:
            print(f"  [clahe]     immediate strips uniform but dark (median={mat_median:.0f}) "
                  f"— likely plain wall, not mat → keeping Step 1 result", flush=True)
            # Clamp: plain wall on complex bg — Step 1 can only tighten, not expand
            result = [
                max(result[0], float(x1)),
                max(result[1], float(y1)),
                min(result[2], float(x2)),
                min(result[3], float(y2)),
            ]
            return result  # plain wall on complex background — don't expand with Step 2
        else:
            print(f"  [clahe]     mat detected around bbox (immediate strips uniform, "
                  f"median={mat_median:.0f}) → wall-edge scan for paper boundary", flush=True)

        if mat_median >= 195:
            # ── Row/column scan for white paper/mat outer edge ───────────────
            gray_full  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            WALL_UPPER = mat_median - 8
            WALL_LOWER = 120
            WALL_STD   = 14

            def _is_wall_row(row_idx: int) -> bool:
                seg = gray_full[row_idx, max(0, x1 - 20): min(w, x2 + 20)]
                if seg.size == 0:
                    return False
                return (WALL_LOWER < float(seg.mean()) < WALL_UPPER
                        and float(seg.std()) < WALL_STD)

            def _is_wall_col(col_idx: int) -> bool:
                seg = gray_full[max(0, y1 - 20): min(h, y2 + 20), col_idx]
                if seg.size == 0:
                    return False
                return (WALL_LOWER < float(seg.mean()) < WALL_UPPER
                        and float(seg.std()) < WALL_STD)

            # Per-side expansion cap: a mat/passepartout is at most 20 % of the
            # larger painting dimension on any side.  Wider expansions mean the
            # scan has walked into unrelated scene content (furniture, floor).
            MAX_EXPAND = int(max(bw, bh) * 0.20)

            top_bound = y1
            for row in range(y1 - 1, max(-1, y1 - 1 - MAX_EXPAND), -1):
                if _is_wall_row(row):
                    break
                top_bound = row

            bottom_bound = y2
            for row in range(y2, min(h, y2 + MAX_EXPAND)):
                if _is_wall_row(row):
                    break
                bottom_bound = row + 1

            left_bound = x1
            for col in range(x1 - 1, max(-1, x1 - 1 - MAX_EXPAND), -1):
                if _is_wall_col(col):
                    break
                left_bound = col

            right_bound = x2
            for col in range(x2, min(w, x2 + MAX_EXPAND)):
                if _is_wall_col(col):
                    break
                right_bound = col + 1

            def _trim_boundary(bound, step, axis, lo_ref, hi_ref):
                while True:
                    probe = bound - step
                    if not (0 <= probe < (h if axis == 0 else w)):
                        break
                    seg = (gray_full[probe, max(0, lo_ref):min(w, hi_ref)]
                           if axis == 0
                           else gray_full[max(0, lo_ref):min(h, hi_ref), probe])
                    if seg.size == 0:
                        break
                    m = float(seg.mean())
                    if m > 200 or m < 130:
                        break
                    bound -= step
                return bound

            bottom_bound = _trim_boundary(bottom_bound, 1, 0, x1, x2)
            top_bound    = _trim_boundary(top_bound,   -1, 0, x1, x2)
            right_bound  = _trim_boundary(right_bound,  1, 1, y1, y2)
            left_bound   = _trim_boundary(left_bound,  -1, 1, y1, y2)

            paper_bbox = [float(left_bound), float(top_bound),
                          float(right_bound), float(bottom_bound)]
            paper_area = (right_bound - left_bound) * (bottom_bound - top_bound)

            # Sanity guard: if the scan expanded the bbox by more than 5× the
            # original Florence bbox area, the scan walked into unrelated scene
            # content.  Reject the result and fall through to CLAHE Step 2.
            if paper_area > orig_area * 5.0:
                print(f"  [clahe]     wall-edge scan expansion too large "
                      f"({paper_area / orig_area:.1f}× orig_area) → CLAHE Step 2",
                      flush=True)
            elif paper_area > orig_area * 1.10:
                print(f"  [clahe]     wall-edge scan → paper bbox "
                      f"{[round(v) for v in paper_bbox]}", flush=True)
                return paper_bbox
            else:
                print(f"  [clahe]     wall-edge scan found no improvement → CLAHE Step 2",
                      flush=True)

    # ── Step 2: full-image search for mat / passepartout boundary ─────────
    # Only runs when Florence had a SMALL (content-only) detection — i.e. it
    # found only the subject inside the painting (flowers, faces) and missed the
    # surrounding mat/passepartout.  For larger detections (frac ≥ 20 %), Florence
    # already found the painting boundary well and Step 2 risks expanding into
    # surrounding wall space, which caused Cat 1 regressions on white-wall images.
    orig_frac = orig_area / img_area
    if orig_frac >= 0.20:
        return result   # Florence found the painting at 20 %+ of image — trust it

    # Trigger: Step 1 grew the bbox by less than 50% — may still be missing
    # a wide mat that lies beyond the local search region.
    res1_area = (result[2] - result[0]) * (result[3] - result[1])
    if res1_area > orig_area * 1.50:
        return result   # Step 1 already found significant expansion — done

    em2      = max(3, int(min(h, w) * 0.05))
    result_2 = _search(img_bgr, 0, 0,
                       edge_margin=em2, area_upper=img_area * 0.92)
    if result_2 is None:
        return result

    res2_area = (result_2[2] - result_2[0]) * (result_2[3] - result_2[1])
    if res2_area <= res1_area * 1.10:
        return result   # Step 2 not meaningfully larger — keep Step 1

    # Centre-proximity guard: the mat must be roughly centred on the artwork
    orig_cx = (x1 + x2) / 2 / w;  orig_cy = (y1 + y2) / 2 / h
    r2_cx   = (result_2[0] + result_2[2]) / 2 / w
    r2_cy   = (result_2[1] + result_2[3]) / 2 / h
    if abs(r2_cx - orig_cx) + abs(r2_cy - orig_cy) >= 0.30:
        return result   # too far from original — likely grabbed wrong object

    print(f"  [clahe]     step2 expanded to mat {[round(v) for v in result_2]}",
          flush=True)
    return result_2


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Perspective correction
# ──────────────────────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s        = pts.sum(axis=1)
    rect[0]  = pts[np.argmin(s)]
    rect[2]  = pts[np.argmax(s)]
    diff     = np.diff(pts, axis=1)
    rect[1]  = pts[np.argmin(diff)]
    rect[3]  = pts[np.argmax(diff)]
    return rect


def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective warp that maps *pts* to a front-facing rectangle."""
    rect       = _order_points(pts)
    tl, tr, br, bl = rect
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (W, H))


def perspective_correct(img_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Attempt to correct perspective distortion within the detected artwork bbox.

    For paintings that are angled, leaning, or shot from the side
    (categories 6, 6a–6g), the artwork's outer frame forms a quadrilateral
    that is not perfectly rectangular in the image.  Detecting that quad and
    applying a four-point warp produces a front-facing crop.

    Quad validation (all conditions must pass):
      - Exactly 4 vertices.
      - Convex hull — non-convex quads produce inverted/torn warps with large
        black fill areas, which was the root cause of Pattern A crops where the
        bbox overlay looked correct but the saved crop was a thin strip or had
        large black sections.
      - Covers 60–95% of the bbox region.  The lower bound is critical: without
        it, small spurious quads (desk corners, window frames, etc.) found inside
        a complex scene would pass and produce wildly wrong crops.  The original
        code only had an upper bound (110%), which allowed any small quad through.
      - The warp result has no more than 8% black pixels (catches bad warps from
        degenerate or nearly-degenerate quads that pass the area checks).
      - The warp does not expand the area by more than 10%.

    Falls back to a simple rectangular crop of the bbox when no valid quad is found.

    Args:
        img_bgr: Full input image in BGR.
        bbox:    [x1, y1, x2, y2] — the artwork region to correct within.

    Returns:
        A BGR crop: either perspective-corrected or simply rectangular.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    region          = img_bgr[y1:y2, x1:x2]
    rh, rw          = region.shape[:2]
    if rh == 0 or rw == 0:
        return region

    gray    = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 15, 50)
    edges   = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    region_area     = rh * rw
    best_pts, best_score = None, -1.0

    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4:
            continue

        pts  = approx.reshape(4, 2).astype("float32")

        # ── Convexity guard ───────────────────────────────────────────────
        # Non-convex quads produce inverted perspective warps with large black
        # regions (the warp maps output pixels outside the source image bounds).
        if not cv2.isContourConvex(pts.astype(np.int32)):
            continue

        area = cv2.contourArea(approx)
        frac = area / region_area

        # ── Coverage guard (60–95%) ───────────────────────────────────────
        # Lower bound raised from 0.15 → 0.60.  Any quad covering < 60% of
        # the region is a sub-region feature (furniture, window, desk corner)
        # rather than the painting boundary, and should not trigger a warp.
        if not (0.60 < frac < 0.95):
            continue

        rect = _order_points(pts)
        ev   = [rect[(i + 1) % 4] - rect[i] for i in range(4)]
        dots  = [
            abs(np.dot(ev[i], ev[(i + 1) % 4])) /
            (np.linalg.norm(ev[i]) * np.linalg.norm(ev[(i + 1) % 4]) + 1e-6)
            for i in range(4)
        ]
        score = area * (1.0 - np.mean(dots))
        if score > best_score:
            best_score, best_pts = score, pts

    if best_pts is not None:
        # ── Skewness guard ────────────────────────────────────────────────
        # Only apply perspective warp when the quad is genuinely tilted.
        # Face-on paintings and architectural rectangles have axis-aligned
        # edges (all edges within ~8° of horizontal or vertical).  Warping
        # those produces at best a no-op and at worst a distorted crop (e.g.
        # gallery dark-panel edges forming a rectangular quad inside the
        # Florence bbox get mapped to a tilted result).
        #
        # For a painting tilted at angle θ, ALL four edges deviate θ from
        # the nearest axis.  Only apply the warp when every edge deviates
        # more than 8° — i.e. the painting is genuinely off-axis.
        rect_ord = _order_points(best_pts)
        deviations = []
        for i in range(4):
            e   = rect_ord[(i + 1) % 4] - rect_ord[i]
            ang = float(np.degrees(np.arctan2(float(abs(e[1])),
                                              float(abs(e[0])) + 1e-6)))
            deviations.append(min(ang, 90.0 - ang))
        if min(deviations) < 8.0:
            return region   # near-axis-aligned — face-on or architectural

        warped     = _four_point_transform(region, best_pts)
        warp_area  = warped.shape[0] * warped.shape[1]

        # ── Area bounds ───────────────────────────────────────────────────
        if not (region_area * 0.60 <= warp_area <= region_area * 1.10):
            return region

        # ── Black-pixel guard ─────────────────────────────────────────────
        # A good warp has minimal black fill.  More than 8% black indicates
        # a degenerate mapping where the transform pulled in out-of-bounds
        # source pixels.
        black_ratio = float(np.mean(warped.sum(axis=2) == 0))
        if black_ratio > 0.08:
            return region

        return warped

    return region   # fallback: simple rectangular crop


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Frame border trim  (optional post-processing)
# ──────────────────────────────────────────────────────────────────────────────

def _find_trim_amount(stds: np.ndarray, max_trim: int, window: int = 3) -> int:
    """
    Scan *stds* (per-row or per-column standard deviations, ordered from the
    edge inward) and return how many pixels constitute the frame border.

    Logic:
      - The outermost pixels define the "frame baseline" variance.
      - If the edge itself is already high-variance (>25), the artwork starts
        at the edge and nothing is trimmed.
      - Otherwise, walk inward with a small sliding window until the mean std
        exceeds threshold = max(baseline × 3, 25).  That crossing point is the
        frame-to-artwork boundary.
    """
    if len(stds) < window + 1 or max_trim == 0:
        return 0

    outer_std = float(stds[:3].mean())

    # Edge is already artwork-like — no frame to trim.
    if outer_std > 25:
        return 0

    threshold = max(outer_std * 3.0, 25.0)

    for i in range(min(max_trim, len(stds) - window)):
        if float(stds[i : i + window].mean()) > threshold:
            return i

    return 0   # no frame edge found within the allowed depth


def trim_frame_border(img_bgr: np.ndarray, max_depth_pct: float = 0.12) -> np.ndarray:
    """
    Remove a uniform frame border from an already-cropped artwork image.

    Motivation:
        View at Home renders its own 3D frame around the artwork; physical
        frames captured in the photo must therefore be excluded from the crop.
        This function scans inward from each edge using per-row / per-column
        pixel standard deviation.  A frame is characterised by low variance
        (uniform colour or fine grain); artwork content has visibly higher
        variance.  The scan stops at the first layer whose variance jumps
        significantly above the edge baseline.

    Limitations (known):
        - Gallery-wrapped canvases (artwork extends to the stretcher edge) will
          not be trimmed because their outermost row/column already has high
          variance — the early-exit guard in _find_trim_amount handles this.
        - Very thin frames (< 5 px) may not yield enough pixels to measure.
        - White-on-white (frame colour == artwork margin colour) will not be
          trimmed; this is the correct behaviour since there is no visual signal.

    Args:
        img_bgr:       Input crop in BGR (OpenCV format).
        max_depth_pct: Maximum inward search depth on each side, expressed as
                       a fraction of the corresponding image dimension.
                       Default 0.12 (12 %).  Increase for thick ornate frames;
                       decrease if the function over-trims on your dataset.

    Returns:
        Trimmed BGR image, or the original array if no frame edge is detected.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    max_h = max(1, int(h * max_depth_pct))
    max_w = max(1, int(w * max_depth_pct))

    # Use the central 80 % of the perpendicular extent to compute per-row / per-
    # column std.  This avoids corner regions where two frame edges overlap and
    # artificially inflate variance, which would suppress trimming.
    cx1, cx2 = int(w * 0.10), int(w * 0.90)
    cy1, cy2 = int(h * 0.10), int(h * 0.90)

    # Per-row std (used for top / bottom detection): each value = spread of
    # pixel intensities across the central columns of that row.
    row_stds = gray[:, cx1:cx2].std(axis=1)   # shape (h,)

    # Per-column std (used for left / right detection).
    col_stds = gray[cy1:cy2, :].std(axis=0)   # shape (w,)

    top    = _find_trim_amount(row_stds[:max_h],          max_h)
    bottom = _find_trim_amount(row_stds[h - max_h:][::-1], max_h)
    left   = _find_trim_amount(col_stds[:max_w],          max_w)
    right  = _find_trim_amount(col_stds[w - max_w:][::-1], max_w)

    y1, y2 = top,        h - bottom
    x1, x2 = left,       w - right

    # Safety: never trim more than 50 % of any dimension (catches bad detections).
    if (y2 - y1) < h * 0.50 or (x2 - x1) < w * 0.50:
        print("  [frame-trim] trim would exceed 50% — skipping.", flush=True)
        return img_bgr

    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return img_bgr   # nothing to trim

    print(
        f"  [frame-trim] top={top}px  bottom={bottom}px  "
        f"left={left}px  right={right}px",
        flush=True,
    )
    return img_bgr[y1:y2, x1:x2]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  OpenCV fallback
# ──────────────────────────────────────────────────────────────────────────────

def _detect_by_bg_contrast(img_bgr: np.ndarray, margin_px: int = 5) -> Optional[list[float]]:
    """
    Detect the primary artwork object on a plain uniform background by finding
    all pixels that differ meaningfully from the background colour.

    This is a lightweight alternative to Florence-2 for the specific case of a
    small isolated painting on a plain wall — a case where the model sometimes
    returns nothing because the artwork is too small relative to the image.

    Returns:
        [x1, y1, x2, y2] bounding box in image pixels, or None if the background
        is not uniform enough or no distinct foreground object is found.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Sample background from the four corners (10 × 10 px each).
    cs = min(10, h // 6, w // 6)
    corners = np.concatenate([
        gray[:cs, :cs].flatten(),
        gray[:cs, -cs:].flatten(),
        gray[-cs:, :cs].flatten(),
        gray[-cs:, -cs:].flatten(),
    ])
    bg_mean = float(corners.mean())
    bg_std  = float(corners.std())

    # Only works on uniform backgrounds (plain walls / studio backdrops).
    if bg_std >= CLAHE_BG_UNIFORM_THRESHOLD:
        return None

    # Pixels that differ from the background by more than 20 grey levels.
    # 12 was too sensitive — it picked up faint canvas shadows that bleed
    # to the image edge.  20 catches visible painting edges reliably while
    # ignoring wall texture noise and compression artefacts.
    diff = cv2.absdiff(gray, np.full_like(gray, int(bg_mean)))
    _, fg_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Close small gaps so the artwork forms a single connected blob.
    # 11×11 kernel needed to bridge across the light centre of some paintings
    # (e.g. a white flower where the centre is nearly bg-coloured).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Find the largest contour — that should be the artwork.
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the bounding rect of the largest contour by area.
    largest = max(contours, key=cv2.contourArea)
    cx, cy, cw, ch = cv2.boundingRect(largest)

    # Sanity: object must cover at least 2% and at most 80% of the image.
    obj_frac = (cw * ch) / (w * h)
    if not (0.02 < obj_frac < 0.80):
        return None

    x1 = max(0, cx - margin_px)
    y1 = max(0, cy - margin_px)
    x2 = min(w, cx + cw + margin_px)
    y2 = min(h, cy + ch + margin_px)

    return [float(x1), float(y1), float(x2), float(y2)]


def _tighten_against_background(crop_bgr: np.ndarray, margin_px: int = 8) -> Optional[np.ndarray]:
    """
    Tighten a crop by finding all pixels that differ meaningfully from the
    background colour and returning a tight bounding rect around them.

    Used after the OpenCV fallback when the background is a plain uniform colour
    (white wall, grey studio) and the fallback left excess background around the
    main subject.  Has no effect when the background is complex (std ≥ 25) or
    when the tightened result is not meaningfully smaller than the input.

    Returns the tightened crop, or None if no meaningful improvement was found.
    """
    h, w = crop_bgr.shape[:2]
    if h < 40 or w < 40:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Sample background from the four corners (5 × 5 px each).
    corner_size = min(5, h // 4, w // 4)
    corners = np.concatenate([
        gray[:corner_size, :corner_size].flatten(),
        gray[:corner_size, -corner_size:].flatten(),
        gray[-corner_size:, :corner_size].flatten(),
        gray[-corner_size:, -corner_size:].flatten(),
    ])
    bg_mean = float(corners.mean())
    bg_std  = float(corners.std())

    # Only attempt tightening on uniform backgrounds (plain walls / studios).
    if bg_std >= CLAHE_BG_UNIFORM_THRESHOLD:
        return None

    # Pixels that differ from the background by more than 25 grey levels are
    # considered "foreground" (the artwork).
    diff = cv2.absdiff(gray, np.full_like(gray, int(bg_mean)))
    _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Close small gaps so the artwork forms a single connected blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    coords = np.argwhere(fg_mask)
    if coords.size == 0:
        return None

    ry_min, rx_min = coords.min(axis=0)
    ry_max, rx_max = coords.max(axis=0)

    # Add a small margin so the crop doesn't cut right at the artwork edge.
    rx_min = max(0, rx_min - margin_px)
    ry_min = max(0, ry_min - margin_px)
    rx_max = min(w - 1, rx_max + margin_px)
    ry_max = min(h - 1, ry_max + margin_px)

    tight_area  = (rx_max - rx_min) * (ry_max - ry_min)
    orig_area   = h * w

    # Only accept if the tightened result is at least 15 % smaller.
    if tight_area >= orig_area * 0.85:
        return None

    return crop_bgr[ry_min:ry_max + 1, rx_min:rx_max + 1]


def _tighten_bbox_by_bg(img_bgr: np.ndarray, bbox: list[float]) -> list[float]:
    """
    Tighten a bounding box to the true painting edge by sampling the background
    colour from strips *outside* the bbox, then finding all pixels inside the
    bbox that differ meaningfully from that background.

    This fixes the class of failure where Florence-2 returns a bbox that extends
    slightly past the painting edge onto the surrounding wall, and CLAHE Step 1
    does not snap back because it finds painting interior contours rather than
    the outer boundary.

    Background estimation:
        Each of the four outside strips (top / bottom / left / right) is evaluated
        independently.  Only *uniform* strips (pixel std < CLAHE_BG_UNIFORM_THRESHOLD)
        are used for background estimation — strips that contain furniture, plants, or
        other complex scene elements are silently excluded.  This lets the function
        work on room-scene images where only *some* outside strips are plain wall.

    Skips / returns original when:
      - no outside strip is uniform (all four sides are complex).
      - the tightened result is less than 8% smaller — no meaningful improvement.
      - the tightened result is less than 15% of the original area — over-tightened.

    Args:
        img_bgr:  Full input image in BGR.
        bbox:     [x1, y1, x2, y2] from CLAHE refinement.

    Returns:
        Tightened [x1, y1, x2, y2], or the original bbox unchanged.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return bbox

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Sample each outside strip independently; only keep uniform ones ────
    # Uniform = std < CLAHE_BG_UNIFORM_THRESHOLD (25).
    # Using centre 60 % of each perpendicular extent so that corners with
    # furniture / decorations don't contaminate the background sample.
    SAMPLE_W = max(5, min(20, int(min(bw, bh) * 0.04)))
    cx1 = x1 + int(bw * 0.20);  cx2 = x2 - int(bw * 0.20)   # centre x range
    cy1 = y1 + int(bh * 0.20);  cy2 = y2 - int(bh * 0.20)   # centre y range

    raw_strips: list[np.ndarray] = []
    if y1 >= SAMPLE_W:
        raw_strips.append(gray[max(0, y1 - SAMPLE_W):y1, cx1:cx2].flatten())
    if h - y2 >= SAMPLE_W:
        raw_strips.append(gray[y2:min(h, y2 + SAMPLE_W), cx1:cx2].flatten())
    if x1 >= SAMPLE_W:
        raw_strips.append(gray[cy1:cy2, max(0, x1 - SAMPLE_W):x1].flatten())
    if w - x2 >= SAMPLE_W:
        raw_strips.append(gray[cy1:cy2, x2:min(w, x2 + SAMPLE_W)].flatten())

    # Use only the uniform strips for background estimation.
    bg_samples = [s for s in raw_strips
                  if s.size > 0 and float(s.std()) < CLAHE_BG_UNIFORM_THRESHOLD]
    if not bg_samples:
        return bbox   # all surrounding strips are complex — can't estimate bg

    all_bg  = np.concatenate(bg_samples)
    bg_mean = float(np.median(all_bg))
    # Use the AVERAGE of within-strip stds (not the std of the concatenation).
    # Concatenating strips from different sides of the painting can give a
    # falsely high std even when every individual strip is uniform — e.g. a
    # bright-white top strip + dark-grey right strip concatenates to high std.
    # Average within-strip std correctly reflects local background noise.
    bg_std  = float(np.mean([float(s.std()) for s in bg_samples]))

    # Adaptive threshold: enough above noise to suppress background variation,
    # low enough to catch subtle wall→canvas colour transitions.
    threshold = max(10, bg_std * 1.5)

    # Pixels inside the bbox that differ from the background = painting pixels.
    region = gray[y1:y2, x1:x2]
    diff   = cv2.absdiff(region, np.full_like(region, int(bg_mean)))
    _, fg_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Close small gaps so the artwork forms one connected region.
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    coords = np.argwhere(fg_mask)
    if coords.size == 0:
        return bbox

    ry_min, rx_min = coords.min(axis=0)
    ry_max, rx_max = coords.max(axis=0)

    # Add margin so the crop doesn't shave off painting edges.
    MARGIN = 8
    rx_min = max(0, rx_min - MARGIN)
    ry_min = max(0, ry_min - MARGIN)
    rx_max = min(bw - 1, rx_max + MARGIN)
    ry_max = min(bh - 1, ry_max + MARGIN)

    new_area  = (rx_max - rx_min) * (ry_max - ry_min)
    orig_area = bw * bh

    # Reject if less than 8% smaller (no meaningful improvement).
    if new_area >= orig_area * 0.92:
        return bbox

    # Reject if suspiciously small (less than 40% of original — tightener is
    # designed for fine border trimming only; large reductions mean it couldn't
    # distinguish painting from background and should not be trusted).
    if new_area < orig_area * 0.40:
        return bbox

    new_bbox = [
        float(x1 + rx_min), float(y1 + ry_min),
        float(x1 + rx_max), float(y1 + ry_max),
    ]
    print(f"  [tighten]   bg_mean={bg_mean:.0f} thr={threshold:.1f} "
          f"uniform_strips={len(bg_samples)}/{len(raw_strips)} "
          f"→ {[round(v) for v in bbox]} → {[round(v) for v in new_bbox]}",
          flush=True)
    return new_bbox


def _opencv_fallback(input_path: str, output_path: str) -> str:
    """
    Delegate to the original rule-based artwork_cropper.py pipeline.

    Called when Florence-2 returns no plausible detection (confidence="none").
    artwork_cropper.py must be present in the same directory.  If it is not
    found, the original image is copied to *output_path* unchanged so the
    pipeline never silently drops an image.

    After the OpenCV result is saved, a background-contrast tightening pass is
    applied: when the background is a plain uniform colour (white wall, grey
    studio), any excess background margin left by the OpenCV pipeline is trimmed.
    This is particularly effective for small isolated paintings on plain walls
    where the OpenCV heuristics may leave a generous margin.
    """
    try:
        from artwork_cropper import crop_artwork as _ocv_crop  # noqa: PLC0415
        print("  [fallback] Florence-2 found nothing — using OpenCV pipeline.", flush=True)
        _ocv_crop(input_path, output_path)
    except ImportError:
        print("  [fallback] artwork_cropper.py not found — copying original.", flush=True)
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

    # ── Post-process: tighten against plain background ─────────────────────
    ocv_result = cv2.imread(output_path)
    if ocv_result is not None:
        tightened = _tighten_against_background(ocv_result)
        if tightened is not None:
            th, tw = tightened.shape[:2]
            print(f"  [fallback] Background tightening: {ocv_result.shape[1]}×{ocv_result.shape[0]}"
                  f" → {tw}×{th}px", flush=True)
            cv2.imwrite(output_path, tightened)

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main per-image pipeline
# ──────────────────────────────────────────────────────────────────────────────

def crop_artwork(
    input_path:   str,
    output_path:  Optional[str] = None,
    device:       Optional[str] = None,
    trim_frame:   bool = False,
    frame_depth:  float = 0.12,
    cutout:       bool = False,
) -> str:
    """
    Crop the artwork out of a single image using the full pipeline.

    Args:
        input_path:  Path to the source image file.
        output_path: Destination path for the cropped result.  When omitted,
                     the output is saved as <stem>_cropped<ext> alongside the
                     input file.
        device:      Override the compute device ("cuda", "mps", or "cpu").
                     Defaults to auto-detected best device.
        trim_frame:  When True, applies border-strip variance trimming after
                     the main crop to remove physical picture frames.
                     Off by default — enable per-client via --frame-trim.
        frame_depth: Maximum search depth for the frame trim, as a fraction
                     of the image dimension (default 0.12 = 12 %).
        cutout:      When True and a round shape is detected, returns a BGRA
                     PNG image with transparency outside the fitted ellipse.

    Returns:
        Absolute path to the saved output image.
    """
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_cropped{p.suffix}")

    # Load both PIL (Florence-2 expects RGB PIL) and BGR numpy (OpenCV steps)
    pil_image = Image.open(input_path).convert("RGB")
    img_bgr   = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    ih, iw = img_bgr.shape[:2]

    # ── Early round/oval shape detection ───────────────────────────────────
    # Circular and oval canvases need ellipse-based cropping, not the
    # rectangular CLAHE / perspective / tightening pipeline which mishandles
    # them (includes floor/shadow, attempts warp on round shapes, clips edges).
    # This check runs BEFORE Florence-2 detection so it catches round shapes
    # regardless of what the model returns.
    if _HAS_ROUND_DETECTION:
        round_cand = _detect_round_shape(img_bgr)
        if round_cand is not None:
            print(f"  [round] Circular/oval shape detected "
                  f"(extent={round_cand['extent']:.3f}, frac={round_cand['frac']:.0%}) "
                  f"→ using ellipse-based crop (cutout={cutout}).", flush=True)
            result = _crop_round(img_bgr, padding=3.0, cutout=cutout)
            rh, rw = result.shape[:2]
            if rh >= 50 and rw >= 50 and (rh * rw) >= (ih * iw * 0.03):
                if cutout and output_path.lower().endswith((".jpg", ".jpeg")):
                    output_path = str(Path(output_path).with_suffix(".png"))
                cv2.imwrite(output_path, result)
                print(f"  ✓ {Path(output_path).name}  ({rw}×{rh}px)", flush=True)
                return output_path
            # If crop_round returned a degenerate result, fall through to the
            # normal rectangular pipeline as a safety net.
            print("  [round] Ellipse crop degenerate — falling back to Florence-2 pipeline.",
                  flush=True)

    # ── Fast pre-screen: skip Florence-2 for plain-background images ────────
    # For artworks on a plain white/grey wall where the painting covers a
    # significant fraction of the image (category 1, 4, 9, most studio shots),
    # _detect_by_bg_contrast finds the bbox in <1 s by background subtraction —
    # no need for the 30-second Florence-2 inference.
    #
    # Safety gates (all must pass):
    #   1. Background must be uniform (corners std < CLAHE_BG_UNIFORM_THRESHOLD)
    #      — rules out room scenes, complex walls, and most non-studio shots.
    #   2. Detected bbox covers 30–90% of the image — rules out images where
    #      a tiny painting is lost in a large plain background (Florence handles
    #      those better) and full-bleed images that need no crop at all.
    #   3. The bbox is not already used for round-shape crops above.
    #
    # No accuracy regression: when the pre-screen fires the bbox is passed
    # through the same CLAHE-refine + tighten + perspective pipeline as normal
    # Florence detections, giving the same tight crop result.
    _fast_bbox = _detect_by_bg_contrast(img_bgr, margin_px=5)
    if _fast_bbox is not None:
        _fx1, _fy1, _fx2, _fy2 = _fast_bbox
        _fast_frac = (_fx2 - _fx1) * (_fy2 - _fy1) / (iw * ih)
        # Reject if bbox touches any image edge (within 10 px) — this means
        # the shadow/contour bled to the frame, making the bbox unreliable.
        _edge_margin = 10
        _touches_edge = (
            _fx1 < _edge_margin or _fy1 < _edge_margin
            or _fx2 > iw - _edge_margin or _fy2 > ih - _edge_margin
        )
        if 0.30 <= _fast_frac <= 0.90 and not _touches_edge:
            print(f"  [fast-path] Plain background — skipping Florence-2 "
                  f"(bbox_frac={_fast_frac:.2f})", flush=True)
            bbox       = _fast_bbox
            confidence = "high"
            # Jump directly to CLAHE refinement (Step 2 below).
            goto_refine = True
        else:
            goto_refine = False
    else:
        goto_refine = False

    if not goto_refine:
        # ── Step 1: Florence-2 semantic detection ──────────────────────────
        model, processor, device = load_model(device)
        bbox, confidence = detect_artwork(pil_image, model, processor, device)

        print(
            f"  [florence2] confidence={confidence}  "
            f"bbox={[round(v) for v in bbox] if bbox else None}",
            flush=True,
        )


    if confidence == "none":
        # Before running the full OpenCV pipeline, check whether the painting
        # already fills most of the frame (category 13 — nearly-cropped inputs).
        # In that case the correct action is a minimal border trim, not full
        # re-detection which tends to aggressively over-crop.
        gray_full   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        border_w    = max(1, int(iw * 0.06))
        border_h    = max(1, int(ih * 0.06))
        border_mean = np.mean([
            gray_full[:border_h, :].mean(),
            gray_full[-border_h:, :].mean(),
            gray_full[:, :border_w].mean(),
            gray_full[:, -border_w:].mean(),
        ])
        center_mean = float(gray_full[ih//4 : 3*ih//4, iw//4 : 3*iw//4].mean())
        # If the border colour differs from the centre by less than 20 and the
        # border is not very uniform, the painting almost certainly fills the frame.
        # Apply only a conservative trim via CLAHE rather than full OpenCV detection.
        border_contrast = abs(float(border_mean) - center_mean)
        # Threshold 45: catches nearly-cropped images (cat 13) where a thin
        # wall strip differs from the painting by up to ~37 intensity units,
        # while still excluding genuinely hard scenes where the painting is a
        # small object in a complex background (which need full re-detection).
        if border_contrast < 45:
            print("  [fallback] Nearly full-frame image — applying minimal CLAHE trim.",
                  flush=True)
            full_bbox  = [0.0, 0.0, float(iw), float(ih)]
            trimmed    = clahe_refine_boundary(img_bgr, full_bbox, expand_pct=0.0)
            rh, rw     = img_bgr[int(trimmed[1]):int(trimmed[3]),
                                  int(trimmed[0]):int(trimmed[2])].shape[:2]
            # Only accept the CLAHE result if it meaningfully reduced the image
            # (at least 10 % smaller by area).  When a tiny painting sits on a
            # large plain wall, border_contrast is small (both areas look similar
            # in brightness) so this path fires — but CLAHE finds no useful edge
            # and returns the full image unchanged.  Without this guard the full
            # image is saved as the "crop", silently discarding the OpenCV
            # fallback that would have correctly isolated the small painting.
            if rh > 50 and rw > 50 and (rh * rw) < (ih * iw * 0.90):
                result = img_bgr[int(trimmed[1]):int(trimmed[3]),
                                  int(trimmed[0]):int(trimmed[2])]
                cv2.imwrite(output_path, result)
                print(f"  ✓ {Path(output_path).name}  ({rw}×{rh}px)", flush=True)
                return output_path
            # CLAHE found no useful trim — fall through to OpenCV which handles
            # small isolated objects on plain backgrounds reliably.
            print("  [fallback] CLAHE trim found no useful boundary → OpenCV fallback.",
                  flush=True)

        # ── Background contrast detector: last resort before OpenCV ───────
        # For small isolated paintings on plain walls (e.g. a single canvas on
        # white drywall) Florence-2 sometimes finds nothing because the subject
        # is too small relative to the image.  A simple contrast-based detector
        # — find all pixels that differ from the uniform background colour —
        # works reliably in these cases and produces a tighter crop than the
        # general-purpose OpenCV fallback.
        contrast_bbox = _detect_by_bg_contrast(img_bgr)
        if contrast_bbox is not None:
            print(f"  [fallback] Background contrast detector → bbox={[round(v) for v in contrast_bbox]}",
                  flush=True)
            result = perspective_correct(img_bgr, contrast_bbox)
            rh, rw = result.shape[:2]
            if rh >= 50 and rw >= 50 and (rh * rw) >= (ih * iw * 0.02):
                cv2.imwrite(output_path, result)
                print(f"  ✓ {Path(output_path).name}  ({rw}×{rh}px)", flush=True)
                return output_path

        return _opencv_fallback(input_path, output_path)

    # ── Step 2: CLAHE edge refinement — always runs ────────────────────────
    # Runs regardless of confidence level.  Fixes two failure modes:
    #   HIGH confidence + content-only bbox: expands to include mat/passepartout.
    #   HIGH confidence + loose bbox: trims wall/shadow overhang.
    # Previously gated on confidence=="low", which meant high-confidence but
    # content-only detections (erishimatsuka series, HolidayBoy, etc.) were
    # never refined and produced crops missing the full artwork boundary.
    raw_bbox = bbox   # keep original for logging
    bbox = clahe_refine_boundary(img_bgr, bbox)
    if bbox != raw_bbox:
        print(f"  [clahe]     {[round(v) for v in raw_bbox]} → {[round(v) for v in bbox]}",
              flush=True)

    # ── Step 2b: Background-contrast bbox tightening ──────────────────────
    # Runs after CLAHE to snap loose bbox edges to the actual painting boundary.
    # Uses background colour sampled from strips *outside* the bbox so it works
    # even when the painting extends to some edges of the bbox.
    # Only fires on uniform-background images (plain walls / studio backdrops).
    bbox = _tighten_bbox_by_bg(img_bgr, bbox)

    # ── Step 3: Perspective correction ────────────────────────────────────
    result = perspective_correct(img_bgr, bbox)
    rh, rw = result.shape[:2]

    # Sanity check: reject degenerate crops
    if rh < 50 or rw < 50 or (rh * rw) < (ih * iw * 0.03):
        print("  [warn] Crop is degenerate — using OpenCV fallback.", flush=True)
        return _opencv_fallback(input_path, output_path)

    # ── Step 4 (optional): Frame border trim ──────────────────────────────
    if trim_frame:
        result = trim_frame_border(result, max_depth_pct=frame_depth)

    cv2.imwrite(output_path, result)
    print(f"  ✓ {Path(output_path).name}  ({rw}×{rh}px)", flush=True)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Batch processing
# ──────────────────────────────────────────────────────────────────────────────

def batch_crop(
    input_dir:   str,
    output_dir:  Optional[str] = None,
    device:      Optional[str] = None,
    trim_frame:  bool = False,
    frame_depth: float = 0.12,
    cutout:      bool = False,
) -> None:
    """
    Process all supported images in *input_dir*, saving results to *output_dir*.

    The Florence-2 model is loaded once and reused across all images, making
    batch processing efficient.  Subfolder structure is preserved in the output.

    Supported formats: JPG, PNG, WebP, TIFF, BMP.

    Args:
        input_dir:   Root directory to search for images (recursive).
        output_dir:  Directory for cropped outputs.  Defaults to
                     <input_dir>/cropped/.
        device:      Override compute device.  Defaults to auto-detect.
        trim_frame:  Enable physical frame trimming on every crop.
        frame_depth: Max search depth for frame trim (fraction, default 0.12).
    """
    in_path  = Path(input_dir)
    if not in_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    out_path = Path(output_dir) if output_dir else in_path / "cropped"
    out_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        f for f in in_path.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print("No supported images found.")
        return

    # Load the model before iterating so the first image isn't penalised by
    # the model-load latency in timing comparisons.
    load_model(device)

    print(f"Processing {len(images)} image(s) → {out_path}\n")
    for i, img_file in enumerate(images, 1):
        rel  = img_file.relative_to(in_path)
        dest = out_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(images)}] {rel}")
        try:
            crop_artwork(str(img_file), str(dest), device=device,
                         trim_frame=trim_frame, frame_depth=frame_depth, cutout=cutout)
        except Exception as exc:
            print(f"  ✗ {exc}", flush=True)

    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Florence-2 artwork cropper — semantic detection with OpenCV fallback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python artwork_cropper_v3.py painting.jpg
  python artwork_cropper_v3.py painting.jpg --output out.jpg
  python artwork_cropper_v3.py ./images/ --batch
  python artwork_cropper_v3.py ./images/ --batch --output ./cropped/
  python artwork_cropper_v3.py painting.jpg --frame-trim
  python artwork_cropper_v3.py oval_painting.jpg --cutout
        """,
    )
    parser.add_argument("input",
                        help="Image file path, or directory path when using --batch.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (single) or directory (batch). "
                             "Defaults to <stem>_cropped<ext> or <input>/cropped/.")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Process every image in the input directory recursively.")
    parser.add_argument("--cutout", action="store_true",
                        help="Round mode only: export a transparent die-cut PNG "
                             "(alpha matches fitted ellipse) instead of a rectangular crop.")
    parser.add_argument("--device", "-d", default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="Accepted for compatibility; this build always runs on CPU "
                             "for deterministic inference.  Passing any value is a no-op.")

    # Frame-trim flags — mutually exclusive pair so the developer can set a
    # platform-level default and individual clients can override it either way.
    parser.add_argument("--frame-trim", dest="frame_trim", action="store_true",
                        default=False,
                        help="Trim physical frame borders from the crop. "
                             "Off by default; enable per-client at the platform level.")
    parser.add_argument("--no-frame-trim", dest="frame_trim", action="store_false",
                        help="Disable frame trimming (overrides a platform-level default).")
    parser.add_argument("--frame-depth", type=float, default=0.12, metavar="FRAC",
                        help="Max inward search depth for frame trim, as a fraction of "
                             "image size (default: 0.12). Increase for very thick frames.")

    args = parser.parse_args()

    if args.batch:
        batch_crop(args.input, args.output, device=args.device,
                   trim_frame=args.frame_trim, frame_depth=args.frame_depth, cutout=args.cutout)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: '{args.input}' is not a file. Use --batch for directories.")
            sys.exit(1)
        crop_artwork(args.input, args.output, device=args.device,
                     trim_frame=args.frame_trim, frame_depth=args.frame_depth, cutout=args.cutout)


if __name__ == "__main__":
    main()
