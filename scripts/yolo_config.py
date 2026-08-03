#!/usr/bin/env python3
"""
yolo_config.py
──────────────
Central configuration for the YOLO-based artwork cropper pipeline.

All tuneable constants are collected here so they can be adjusted in one place
without touching the pipeline logic.  Each constant is documented with its
physical justification (what real-world property it corresponds to) so that
future adjustments are grounded in measurable image properties rather than
arbitrary tweaks.
"""

from __future__ import annotations

from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  File & format settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}

# Default output JPEG quality (0–100).  95 preserves visual fidelity while
# keeping file size reasonable for batch processing.
JPEG_QUALITY = 95

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  YOLO model settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Phase 1: YOLO-World (zero-shot, no training needed)
# yolov8s-worldv2 is the small variant (~47 MB) — best speed/accuracy on CPU.
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"

# Phase 2: Fine-tuned YOLOv8n (after manual annotation + training)
# Set this to your custom-trained weights path to use Phase 2.
YOLO_CUSTOM_MODEL: str | None = None  # e.g., "models/best.pt"

# Text prompts for YOLO-World open-vocabulary detection.
# Order matters: more specific prompts first for better precision.
# These prompts are semantically encoded once and cached by YOLO-World.
YOLO_WORLD_CLASSES = [
    "painting",
    "artwork",
    "picture",
    "canvas",
    "framed artwork",
    "picture frame",
    "framed picture",
    "wall art",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  Detection thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Minimum YOLO confidence score to accept a detection.
# 0.15 is deliberately low — we'd rather get a noisy detection and refine it
# with OpenCV than miss a painting entirely.  Post-processing filters handle
# false positives.
YOLO_CONFIDENCE = 0.15

# IoU threshold for Non-Maximum Suppression.
# 0.5 is standard — allows nearby but distinct paintings to survive.
YOLO_IOU = 0.5

# Inference image size.
# 1280px gives much better detection of small paintings in large room scenes
# vs the default 640px, at the cost of ~2× inference time on CPU.
# For Phase 2 (fine-tuned), 640 is usually sufficient.
YOLO_IMGSZ = 1280

# Device — always CPU.  See module docstring for rationale.
DEVICE = "cpu"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  Bounding box filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# A valid painting bbox must cover at least this fraction of the image.
# Paintings smaller than 2% of the image are likely false positives (a book
# spine, a decorative tile, etc.) — unless the image is a close-up crop.
BBOX_AREA_MIN = 0.02

# Maximum fraction.  Detections covering >99.5% are the model returning the
# whole image — not a useful crop.
BBOX_AREA_MAX = 0.995

# Aspect ratio limits.  Paintings are broadly rectangular or circular;
# extremely thin strips (aspect < 0.10 or > 10.0) are structural elements
# (baseboards, curtain rods) rather than artwork.
BBOX_ASPECT_MIN = 0.10
BBOX_ASPECT_MAX = 10.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  Detection scoring weights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# When YOLO returns multiple detections, we pick the best one using a weighted
# composite score.  These weights reflect the relative importance of each factor.

SCORE_WEIGHT_CONFIDENCE = 0.40   # Trust YOLO's own confidence
SCORE_WEIGHT_AREA       = 0.30   # Prefer paintings that occupy 10–80% of image
SCORE_WEIGHT_CENTER     = 0.20   # Prefer centered detections
SCORE_WEIGHT_ASPECT     = 0.10   # Penalise extreme aspect ratios

# "Ideal" area fraction — paintings in gallery/studio photos typically cover
# about 35% of the image.  Deviations are penalised linearly.
IDEAL_AREA_FRACTION = 0.35

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  Shape classification (rectangular vs oval/circular)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# "Extent" = contour_area / min_area_rect_area.
# A perfect circle has extent = π/4 ≈ 0.785.
# A perfect rectangle has extent = 1.0.
# Observed ranges in real photos:
#   Oval/circular:  0.70 – 0.86
#   Rectangular:    0.85 – 1.00
OVAL_EXTENT_MIN = 0.70
OVAL_EXTENT_MAX = 0.86

# Minimum solidity (contour_area / convex_hull_area) for a valid oval.
# Ovals have smooth, unnotched boundaries → high solidity.
OVAL_SOLIDITY_MIN = 0.96

# Minimum number of polygon vertices to qualify as "round".
# A polygon approximation of a rectangle has 4 vertices.
# An oval has many more (typically >10).
OVAL_MIN_VERTICES = 7

# Number of consistent threshold levels required (out of 10 total).
# Multiple thresholds agreeing on "oval" separates true round shapes from
# rectangles that happen to graze extent ≈ 0.785 at one specific threshold.
OVAL_MIN_CONSISTENT = 2

# Thresholds (background-difference levels) to scan for round shape detection.
ROUND_THRESHOLDS = (6, 8, 10, 12, 15, 18, 20, 25, 30, 35)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.  Crop refinement (CLAHE + background tightening)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CLAHE outward expansion limit as fraction of bbox side length.
# 15% is safe for all image types: enough to catch a 5–10% edge misalignment
# from YOLO, but not so large that it grabs adjacent furniture.
CLAHE_EXPAND_PCT = 0.15

# Background uniformity threshold.
# Pixel std < 25 in a border strip = uniform (mat / studio / plain wall).
# Pixel std >= 25 = complex (room scene, furniture, textured wall).
BG_UNIFORM_THRESHOLD = 25.0

# Minimum improvement (area reduction) for background tightening to be applied.
# Changes < 8% are within measurement noise and should not alter the crop.
TIGHTEN_MIN_IMPROVEMENT = 0.08

# Maximum allowed shrinkage from tightening.
# If the tightened result is < 40% of original, the tightener couldn't
# distinguish painting from background and should not be trusted.
TIGHTEN_MAX_SHRINKAGE = 0.40

# Final crop padding (fraction of crop dimension).
# 1% padding prevents edge clipping from JPEG compression artifacts.
FINAL_PADDING_PCT = 0.01

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8.  Perspective correction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Minimum quad coverage of the bbox region (area fraction).
# Below 60%, the quad is a sub-feature (desk corner, window) not the painting.
PERSPECTIVE_COVERAGE_MIN = 0.60

# Maximum quad coverage.  Above 95% the quad is the bbox itself (face-on).
PERSPECTIVE_COVERAGE_MAX = 0.95

# Minimum edge deviation from axis-aligned (degrees).
# Below 8°, the painting is essentially face-on — warping is a no-op or harmful.
PERSPECTIVE_MIN_ANGLE_DEV = 8.0

# Maximum black-pixel ratio in a warped result.
# >8% black indicates a degenerate perspective transform.
PERSPECTIVE_MAX_BLACK = 0.08

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9.  Frame trimming (optional post-processing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Maximum inward search depth for frame trim, as fraction of image dimension.
FRAME_TRIM_MAX_DEPTH = 0.12

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Test runner / comparison images
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Side-by-side comparison panel height (px).
COMPARE_PANEL_H = 560

# Header bar height (px).
COMPARE_HEADER_H = 46

# Gap between left and right panels (px).
COMPARE_GAP = 6

# Maximum panel width (px).
COMPARE_MAX_PANEL_W = 700
