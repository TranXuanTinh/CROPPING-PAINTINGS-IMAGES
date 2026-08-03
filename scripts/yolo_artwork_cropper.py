#!/usr/bin/env python3
"""
yolo_artwork_cropper.py
────────────────────────
YOLO-based artwork cropper — detects and crops paintings from images.

Handles both rectangular/square and oval/circular paintings using a
streamlined pipeline:

  1. YOLO Detection      — YOLO-World (zero-shot) or custom-trained YOLOv8n
  2. Shape Classification — OpenCV-based (extent ratio analysis)
  3. Crop Refinement      — CLAHE edge snapping + background tightening
  4. Perspective Correction — 4-point warp for angled shots (rectangular only)
  5. Oval Processing      — Ellipse fitting + optional alpha mask (oval only)
  6. Frame Trim           — Optional physical frame removal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARDWARE:  CPU only.  No GPU required.
MODEL:     YOLO-World-v2-S (~47 MB) — zero-shot, no training needed.
SPEED:     ~2–5 s/image on CPU (vs 15–40 s with Florence-2-large).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPENDENCIES:
  pip install -r requirements.txt

CLI USAGE:
  # Single image
  python yolo_artwork_cropper.py painting.jpg
  python yolo_artwork_cropper.py painting.jpg --output cropped.jpg

  # Batch (recursive, preserves subfolder structure)
  python yolo_artwork_cropper.py ./images/ --batch
  python yolo_artwork_cropper.py ./images/ --batch --output ./out/

  # Frame trimming
  python yolo_artwork_cropper.py painting.jpg --frame-trim

  # Force shape mode
  python yolo_artwork_cropper.py painting.jpg --mode oval
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
from PIL import Image

from yolo_config import (
    SUPPORTED_EXTENSIONS,
    JPEG_QUALITY,
    YOLO_WORLD_MODEL,
    YOLO_CUSTOM_MODEL,
    YOLO_WORLD_CLASSES,
    YOLO_CONFIDENCE,
    YOLO_IOU,
    YOLO_IMGSZ,
    DEVICE,
    BBOX_AREA_MIN,
    BBOX_AREA_MAX,
    BBOX_ASPECT_MIN,
    BBOX_ASPECT_MAX,
    SCORE_WEIGHT_CONFIDENCE,
    SCORE_WEIGHT_AREA,
    SCORE_WEIGHT_CENTER,
    SCORE_WEIGHT_ASPECT,
    IDEAL_AREA_FRACTION,
    FRAME_TRIM_MAX_DEPTH,
)
from shape_classifier import classify_shape, detect_round_shape
from perspective_correction import perspective_correct
from crop_refinement import (
    clahe_refine_boundary,
    tighten_bbox_by_background,
    add_padding,
    trim_frame_border,
)
from oval_cropper import crop_oval


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  Model management — loaded once, reused across all images
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MODEL_CACHE: dict = {}


def load_model(custom_model: Optional[str] = None) -> object:
    """
    Load the YOLO model (YOLO-World or custom-trained), caching it for the
    lifetime of the process.

    Phase 1 (default): YOLO-World with zero-shot open-vocabulary detection.
    Phase 2 (custom):  Fine-tuned YOLOv8n with domain-specific weights.

    Returns the loaded model object.
    """
    global _MODEL_CACHE

    model_path = custom_model or YOLO_CUSTOM_MODEL
    is_world = model_path is None

    cache_key = "world" if is_world else model_path
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    print(f"[yolo] Loading model...", flush=True)
    t0 = time.time()

    if is_world:
        # Phase 1: YOLO-World (zero-shot, open-vocabulary)
        from ultralytics import YOLOWorld
        model = YOLOWorld(YOLO_WORLD_MODEL)
        model.set_classes(YOLO_WORLD_CLASSES)
        print(
            f"[yolo] YOLO-World ({YOLO_WORLD_MODEL}) ready in "
            f"{time.time() - t0:.1f}s — {len(YOLO_WORLD_CLASSES)} classes",
            flush=True,
        )
    else:
        # Phase 2: Custom-trained YOLOv8
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(
            f"[yolo] Custom model ({model_path}) ready in "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )

    _MODEL_CACHE[cache_key] = model
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  YOLO detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _score_detection(
    bbox: list[float],
    confidence: float,
    img_w: int,
    img_h: int,
) -> float:
    """
    Score a YOLO detection on [0, 1] for best-bbox selection.

    Combines:
      - YOLO confidence (40%): trust the model's own assessment
      - Area fraction (30%): prefer paintings ~35% of image area
      - Centrality (20%): artworks are usually centered in the photo
      - Aspect ratio (10%): penalise extreme ratios (not paintings)
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0

    area_frac = (bw * bh) / (img_w * img_h)
    aspect = bw / bh

    # Hard reject: outside physical limits
    if not (BBOX_AREA_MIN < area_frac < BBOX_AREA_MAX):
        return 0.0
    if not (BBOX_ASPECT_MIN < aspect < BBOX_ASPECT_MAX):
        return 0.0

    # Component scores (each on [0, 1])
    conf_score = min(confidence, 1.0)
    area_score = max(0.0, 1.0 - abs(area_frac - IDEAL_AREA_FRACTION) / 0.50)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    center_score = max(0.0, 1.0 - (abs(cx - 0.5) + abs(cy - 0.5)))
    aspect_score = max(0.0, 1.0 - abs(np.log(aspect)) / 3.0)

    return (
        SCORE_WEIGHT_CONFIDENCE * conf_score
        + SCORE_WEIGHT_AREA * area_score
        + SCORE_WEIGHT_CENTER * center_score
        + SCORE_WEIGHT_ASPECT * aspect_score
    )


def detect_artwork(
    image_path: str,
    model: object,
) -> tuple[Optional[list[float]], float, str]:
    """
    Run YOLO detection to locate the artwork in the image.

    Returns:
        (bbox, confidence, method)

        bbox:        [x1, y1, x2, y2] in pixels, or None if nothing found.
        confidence:  YOLO confidence score of the selected detection.
        method:      "yolo" if detected, "opencv_fallback" if fallback used.
    """
    # Run YOLO inference
    results = model.predict(
        source=image_path,
        device=DEVICE,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return None, 0.0, "none"

    # Get image dimensions from the result
    img_h, img_w = results[0].orig_shape

    # Score all detections and pick the best one
    best_bbox, best_score, best_conf = None, -1.0, 0.0

    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0].cpu().numpy())
        score = _score_detection(xyxy, conf, img_w, img_h)
        if score > best_score:
            best_score = score
            best_bbox = xyxy
            best_conf = conf

    if best_bbox is None:
        return None, 0.0, "none"

    return best_bbox, best_conf, "yolo"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  OpenCV fallback detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _opencv_fallback_detect(img_bgr: np.ndarray) -> Optional[list[float]]:
    """
    Fallback detection when YOLO finds nothing.  Uses background contrast
    to find the primary object on a plain/uniform background.

    Returns [x1, y1, x2, y2] or None.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Sample background from corners
    cs = min(10, h // 6, w // 6)
    corners = np.concatenate([
        gray[:cs, :cs].flatten(),
        gray[:cs, -cs:].flatten(),
        gray[-cs:, :cs].flatten(),
        gray[-cs:, -cs:].flatten(),
    ])
    bg_mean = float(corners.mean())
    bg_std = float(corners.std())

    # Only works on uniform backgrounds
    if bg_std >= 25.0:
        return None

    # Foreground: pixels that differ from background
    diff = cv2.absdiff(gray, np.full_like(gray, int(bg_mean)))
    _, fg_mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    cx, cy, cw, ch = cv2.boundingRect(largest)
    obj_frac = (cw * ch) / (w * h)
    if obj_frac < 0.02:
        return None

    margin = 5
    return [
        float(max(0, cx - margin)),
        float(max(0, cy - margin)),
        float(min(w, cx + cw + margin)),
        float(min(h, cy + ch + margin)),
    ]


def _opencv_edge_detect(img_bgr: np.ndarray) -> Optional[list[float]]:
    """
    Edge-based fallback: finds the largest rectangular-ish contour using
    multi-strategy detection (Otsu, Canny, brightness threshold).

    For complex backgrounds where the simple contrast method fails.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    k_sm = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    candidates: list[tuple[float, tuple]] = []

    # Strategy A: Otsu, both polarities
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in [otsu, cv2.bitwise_not(otsu)]:
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_sm)
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid = [c for c in contours if cv2.contourArea(c) > h * w * 0.03]
        if valid:
            x, y, cw, ch = cv2.boundingRect(np.vstack(valid))
            frac = cw * ch / (h * w)
            if 0.06 < frac < 0.97:
                candidates.append((frac, (x, y, cw, ch)))

    # Strategy B: Canny edges
    edges = cv2.Canny(blur, 25, 80)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_sm)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, k_sm)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [c for c in contours if cv2.contourArea(c) > h * w * 0.03]
    if valid:
        x, y, cw, ch = cv2.boundingRect(np.vstack(valid))
        frac = cw * ch / (h * w)
        if 0.06 < frac < 0.97:
            candidates.append((frac, (x, y, cw, ch)))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    x, y, cw, ch = candidates[0][1]
    return [float(x), float(y), float(x + cw), float(y + ch)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  Main per-image pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def crop_artwork(
    input_path: str,
    output_path: Optional[str] = None,
    model: Optional[object] = None,
    custom_model: Optional[str] = None,
    mode: str = "auto",
    trim_frame: bool = False,
    frame_depth: float = FRAME_TRIM_MAX_DEPTH,
) -> str:
    """
    Crop the artwork out of a single image using the full YOLO pipeline.

    Args:
        input_path:    Path to the source image file.
        output_path:   Destination path for the cropped result.
        model:         Pre-loaded YOLO model (for batch efficiency).
        custom_model:  Path to custom-trained weights (Phase 2).
        mode:          'auto', 'rect', or 'oval'.
        trim_frame:    Enable physical frame border removal.
        frame_depth:   Max search depth for frame trim (fraction).

    Returns:
        Absolute path to the saved output image.
    """
    t0 = time.time()

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_cropped{p.suffix}")

    # Load image
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    ih, iw = img_bgr.shape[:2]

    # ── Step 1: YOLO detection ────────────────────────────────────────────
    if model is None:
        model = load_model(custom_model)

    bbox, confidence, method = detect_artwork(input_path, model)

    if bbox is not None:
        print(
            f"  [yolo] conf={confidence:.2f}  "
            f"bbox=[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]",
            flush=True,
        )
    else:
        # Fallback: OpenCV-based detection
        print("  [yolo] No detection — trying OpenCV fallback.", flush=True)
        bbox = _opencv_fallback_detect(img_bgr)
        if bbox is None:
            bbox = _opencv_edge_detect(img_bgr)
        if bbox is not None:
            method = "opencv_fallback"
            print(
                f"  [opencv] Fallback bbox="
                f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]",
                flush=True,
            )
        else:
            # Nothing found at all — return original image
            print(
                "  [warn] No painting detected — saving original image.",
                flush=True,
            )
            cv2.imwrite(output_path, img_bgr)
            return output_path

    # ── Step 2: Shape classification ──────────────────────────────────────
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(iw, x2)
    y2 = min(ih, y2)

    if mode == "auto":
        region = img_bgr[y1:y2, x1:x2]
        if region.size > 0:
            shape = classify_shape(region)
        else:
            shape = "rectangular"
    elif mode == "oval":
        shape = "oval"
    else:
        shape = "rectangular"

    print(f"  [shape] {shape}", flush=True)

    # ── Step 3: Process based on shape ────────────────────────────────────
    if shape == "oval":
        # Oval pipeline: use the detected region for ellipse fitting
        region = img_bgr[y1:y2, x1:x2]
        result = crop_oval(region, padding_pct=3.0, cutout=False)
    else:
        # Rectangular pipeline
        # Step 3a: CLAHE edge refinement
        refined_bbox = clahe_refine_boundary(img_bgr, list(bbox))
        if refined_bbox != list(bbox):
            print(
                f"  [clahe] refined → "
                f"[{int(refined_bbox[0])}, {int(refined_bbox[1])}, "
                f"{int(refined_bbox[2])}, {int(refined_bbox[3])}]",
                flush=True,
            )

        # Step 3b: Background contrast tightening
        tightened_bbox = tighten_bbox_by_background(img_bgr, refined_bbox)
        if tightened_bbox != refined_bbox:
            print(
                f"  [tighten] → "
                f"[{int(tightened_bbox[0])}, {int(tightened_bbox[1])}, "
                f"{int(tightened_bbox[2])}, {int(tightened_bbox[3])}]",
                flush=True,
            )

        # Step 3c: Add minimal padding
        final_bbox = add_padding(tightened_bbox, ih, iw)

        # Step 3d: Perspective correction
        result = perspective_correct(img_bgr, final_bbox)

    rh, rw = result.shape[:2]

    # Sanity check: reject degenerate crops
    if rh < 50 or rw < 50 or (rh * rw) < (ih * iw * 0.02):
        print(
            "  [warn] Degenerate crop — saving original image.",
            flush=True,
        )
        cv2.imwrite(output_path, img_bgr)
        return output_path

    # ── Step 4 (optional): Frame border trim ──────────────────────────────
    if trim_frame:
        result = trim_frame_border(result, max_depth_pct=frame_depth)

    # Save result
    rh, rw = result.shape[:2]
    cv2.imwrite(output_path, result)
    elapsed = time.time() - t0
    print(
        f"  ✓ {Path(output_path).name}  ({rw}×{rh}px)  "
        f"[{method}, {elapsed:.1f}s]",
        flush=True,
    )
    return output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  Batch processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def batch_crop(
    input_dir: str,
    output_dir: Optional[str] = None,
    custom_model: Optional[str] = None,
    mode: str = "auto",
    trim_frame: bool = False,
    frame_depth: float = FRAME_TRIM_MAX_DEPTH,
) -> None:
    """
    Process all supported images in *input_dir* recursively.

    The YOLO model is loaded once and reused across all images.
    Subfolder structure is preserved in the output.

    Args:
        input_dir:    Root directory to search for images.
        output_dir:   Directory for cropped outputs.
        custom_model: Path to custom-trained weights (Phase 2).
        mode:         'auto', 'rect', or 'oval'.
        trim_frame:   Enable frame trimming on every crop.
        frame_depth:  Max search depth for frame trim.
    """
    in_path = Path(input_dir)
    if not in_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    out_path = Path(output_dir) if output_dir else in_path / "cropped"
    out_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        f
        for f in in_path.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print("No supported images found.")
        return

    # Load model once before iterating
    model = load_model(custom_model)

    total = len(images)
    errors = 0
    print(f"Processing {total} image(s) → {out_path}\n")

    for i, img_file in enumerate(images, 1):
        rel = img_file.relative_to(in_path)
        dest = out_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{total}] {rel}")
        try:
            crop_artwork(
                str(img_file),
                str(dest),
                model=model,
                custom_model=custom_model,
                mode=mode,
                trim_frame=trim_frame,
                frame_depth=frame_depth,
            )
        except Exception as exc:
            print(f"  ✗ {exc}", flush=True)
            errors += 1

    print(f"\nDone.  {total - errors}/{total} succeeded, {errors} errors.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  Installation check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_installation() -> bool:
    """Verify that all required packages are installed and functional."""
    print("Checking installation...\n")
    ok = True

    # Check packages
    packages = {
        "ultralytics": "ultralytics",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "PIL": "Pillow",
    }
    for module, pip_name in packages.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✓ {pip_name:20s}  {ver}")
        except ImportError:
            print(f"  ✗ {pip_name:20s}  NOT INSTALLED")
            ok = False

    # Check YOLO-World model availability
    print()
    try:
        from ultralytics import YOLOWorld
        print(f"  ✓ YOLOWorld import OK")
        # Don't actually load the model here — it downloads on first use
        print(
            f"  ℹ Model '{YOLO_WORLD_MODEL}' will download automatically on first run (~47 MB)"
        )
    except Exception as e:
        print(f"  ✗ YOLOWorld import failed: {e}")
        ok = False

    print()
    if ok:
        print("All checks passed. Ready to crop paintings! 🎨")
    else:
        print("Some checks failed. Run: pip install -r requirements.txt")
    return ok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO artwork cropper — detects and crops paintings from images. "
            "Handles rectangular, square, oval, and circular paintings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python yolo_artwork_cropper.py painting.jpg
  python yolo_artwork_cropper.py painting.jpg --output cropped.jpg
  python yolo_artwork_cropper.py ./images/ --batch
  python yolo_artwork_cropper.py ./images/ --batch --output ./cropped/
  python yolo_artwork_cropper.py painting.jpg --frame-trim
  python yolo_artwork_cropper.py painting.jpg --mode oval
  python yolo_artwork_cropper.py --check
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Image file path, or directory path when using --batch.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output file (single) or directory (batch). "
            "Defaults to <stem>_cropped<ext> or <input>/cropped/."
        ),
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process every image in the input directory recursively.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="auto",
        choices=["auto", "rect", "oval"],
        help="Force shape detection mode (default: auto).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to custom-trained YOLO weights (Phase 2).",
    )
    parser.add_argument(
        "--frame-trim",
        dest="frame_trim",
        action="store_true",
        default=False,
        help="Trim physical frame borders from the crop.",
    )
    parser.add_argument(
        "--no-frame-trim",
        dest="frame_trim",
        action="store_false",
        help="Disable frame trimming.",
    )
    parser.add_argument(
        "--frame-depth",
        type=float,
        default=FRAME_TRIM_MAX_DEPTH,
        metavar="FRAC",
        help="Max inward search depth for frame trim (default: 0.12).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check installation and exit.",
    )

    args = parser.parse_args()

    if args.check:
        check_installation()
        return

    if args.input is None:
        parser.error("Input path is required (or use --check).")

    if args.batch:
        batch_crop(
            args.input,
            args.output,
            custom_model=args.model,
            mode=args.mode,
            trim_frame=args.frame_trim,
            frame_depth=args.frame_depth,
        )
    else:
        if not os.path.isfile(args.input):
            print(
                f"Error: '{args.input}' is not a file. Use --batch for directories."
            )
            sys.exit(1)
        crop_artwork(
            args.input,
            args.output,
            custom_model=args.model,
            mode=args.mode,
            trim_frame=args.frame_trim,
            frame_depth=args.frame_depth,
        )


if __name__ == "__main__":
    main()
