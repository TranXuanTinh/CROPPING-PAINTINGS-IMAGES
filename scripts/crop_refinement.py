#!/usr/bin/env python3
"""
crop_refinement.py
───────────────────
Post-detection crop refinement: snaps the YOLO bounding box to the true
painting edge using OpenCV-based techniques.

YOLO gives a good initial bbox (~90% accurate) but often includes a few
pixels of wall, shadow, or frame edge.  These refinement steps make the
crop pixel-precise:

  1. CLAHE Edge Snapping — Enhanced contrast reveals subtle luminance
     boundaries between painting and wall/mat.

  2. Background Contrast Tightening — Samples background colour from strips
     outside the bbox, then trims any painting-region pixels that match the
     background.

  3. Smart Padding — Adds a tiny (1%) margin to prevent JPEG edge artifacts.

All techniques are pure OpenCV (no ML).  The ±15% search window is
deliberately conservative: YOLO already provides a good starting bbox, so
we only need fine adjustments, not full re-detection.  This eliminates the
runaway-expansion bugs in the original Florence-2 pipeline.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from yolo_config import (
    CLAHE_EXPAND_PCT,
    BG_UNIFORM_THRESHOLD,
    TIGHTEN_MIN_IMPROVEMENT,
    TIGHTEN_MAX_SHRINKAGE,
    FINAL_PADDING_PCT,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CLAHE edge snapping
# ──────────────────────────────────────────────────────────────────────────────

def clahe_refine_boundary(
    img_bgr: np.ndarray,
    bbox: list[float],
    expand_pct: float = CLAHE_EXPAND_PCT,
) -> list[float]:
    """
    Snap a bounding box to the true artwork edge using CLAHE-enhanced edges.

    Searches a region expanded outward by *expand_pct* (15% default) from the
    bbox.  Uses CLAHE contrast enhancement → Canny edge detection → contour-based
    boundary snapping.

    Only accepts results that are ≥50% and ≤3× of the original bbox area.
    This prevents both over-shrinking and runaway expansion.

    Args:
        img_bgr:    Full input image in BGR (OpenCV format).
        bbox:       [x1, y1, x2, y2] from YOLO detection.
        expand_pct: Outward expansion fraction (default 15%).

    Returns:
        Refined [x1, y1, x2, y2], or the original bbox unchanged.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    orig_area = max(1, bw * bh)

    # Search region: bbox + padding
    pad_x = int(bw * expand_pct)
    pad_y = int(bh * expand_pct)
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x)
    ry2 = min(h, y2 + pad_y)

    region = img_bgr[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return list(bbox)

    # CLAHE enhancement
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Edge detection
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return list(bbox)

    # Find the best bounding rect among contours
    rh_r, rw_r = region.shape[:2]
    region_area = rh_r * rw_r
    edge_margin = 8  # pixels from region edge to ignore
    best_rect, best_area = None, 0

    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if not (0.05 < area / region_area < 0.97):
            continue
        if (
            cx < edge_margin
            or cy < edge_margin
            or cx + cw > rw_r - edge_margin
            or cy + ch > rh_r - edge_margin
        ):
            continue
        if area > best_area:
            best_area = area
            best_rect = (cx, cy, cw, ch)

    if best_rect is None:
        return list(bbox)

    cx, cy, cw, ch = best_rect
    result_area = cw * ch

    # Guard: no over-shrinking
    if result_area < orig_area * 0.50:
        return list(bbox)
    # Guard: no runaway expansion
    if result_area > orig_area * 3.0:
        return list(bbox)

    return [
        float(rx1 + cx),
        float(ry1 + cy),
        float(rx1 + cx + cw),
        float(ry1 + cy + ch),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Background contrast tightening
# ──────────────────────────────────────────────────────────────────────────────

def tighten_bbox_by_background(
    img_bgr: np.ndarray,
    bbox: list[float],
) -> list[float]:
    """
    Tighten a bounding box to the true painting edge by sampling the background
    colour from strips *outside* the bbox, then finding all pixels inside that
    differ from the background.

    Only fires on uniform-background images (plain walls, studio backdrops).
    Has no effect when the background is complex (room scenes with furniture).

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

    # Sample each outside strip independently; only keep uniform ones.
    sample_w = max(5, min(20, int(min(bw, bh) * 0.04)))
    cx1 = x1 + int(bw * 0.20)
    cx2 = x2 - int(bw * 0.20)
    cy1 = y1 + int(bh * 0.20)
    cy2 = y2 - int(bh * 0.20)

    raw_strips: list[np.ndarray] = []
    if y1 >= sample_w:
        raw_strips.append(gray[max(0, y1 - sample_w): y1, cx1:cx2].flatten())
    if h - y2 >= sample_w:
        raw_strips.append(gray[y2: min(h, y2 + sample_w), cx1:cx2].flatten())
    if x1 >= sample_w:
        raw_strips.append(gray[cy1:cy2, max(0, x1 - sample_w): x1].flatten())
    if w - x2 >= sample_w:
        raw_strips.append(gray[cy1:cy2, x2: min(w, x2 + sample_w)].flatten())

    # Use only uniform strips for background estimation.
    bg_samples = [
        s
        for s in raw_strips
        if s.size > 0 and float(s.std()) < BG_UNIFORM_THRESHOLD
    ]
    if not bg_samples:
        return bbox  # all complex — can't estimate bg

    all_bg = np.concatenate(bg_samples)
    bg_mean = float(np.median(all_bg))
    bg_std = float(np.mean([float(s.std()) for s in bg_samples]))

    # Adaptive threshold
    threshold = max(10, bg_std * 1.5)

    # Foreground mask: pixels inside bbox that differ from background
    region = gray[y1:y2, x1:x2]
    diff = cv2.absdiff(region, np.full_like(region, int(bg_mean)))
    _, fg_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    coords = np.argwhere(fg_mask)
    if coords.size == 0:
        return bbox

    ry_min, rx_min = coords.min(axis=0)
    ry_max, rx_max = coords.max(axis=0)

    margin = 8
    rx_min = max(0, rx_min - margin)
    ry_min = max(0, ry_min - margin)
    rx_max = min(bw - 1, rx_max + margin)
    ry_max = min(bh - 1, ry_max + margin)

    new_area = (rx_max - rx_min) * (ry_max - ry_min)
    orig_area = bw * bh

    # Reject if no meaningful improvement
    if new_area >= orig_area * (1.0 - TIGHTEN_MIN_IMPROVEMENT):
        return bbox

    # Reject if over-tightened
    if new_area < orig_area * TIGHTEN_MAX_SHRINKAGE:
        return bbox

    return [
        float(x1 + rx_min),
        float(y1 + ry_min),
        float(x1 + rx_max),
        float(y1 + ry_max),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Smart padding
# ──────────────────────────────────────────────────────────────────────────────

def add_padding(
    bbox: list[float],
    img_h: int,
    img_w: int,
    pad_pct: float = FINAL_PADDING_PCT,
) -> list[float]:
    """
    Add a small margin around the bbox to prevent edge clipping.

    Args:
        bbox:    [x1, y1, x2, y2]
        img_h:   Image height.
        img_w:   Image width.
        pad_pct: Padding as fraction of bbox dimensions (default 1%).

    Returns:
        Padded [x1, y1, x2, y2] clipped to image bounds.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad_pct)
    py = int(bh * pad_pct)
    return [
        float(max(0, x1 - px)),
        float(max(0, y1 - py)),
        float(min(img_w, x2 + px)),
        float(min(img_h, y2 + py)),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Frame border trim (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _find_trim_amount(stds: np.ndarray, max_trim: int, window: int = 3) -> int:
    """
    Scan per-row/column standard deviations (from edge inward) and return
    how many pixels constitute the frame border.

    Frame pixels have low variance (uniform colour).
    Artwork content has higher variance.
    """
    if len(stds) < window + 1 or max_trim == 0:
        return 0

    outer_std = float(stds[:3].mean())
    if outer_std > 25:
        return 0  # edge is already artwork-like

    threshold = max(outer_std * 3.0, 25.0)

    for i in range(min(max_trim, len(stds) - window)):
        if float(stds[i: i + window].mean()) > threshold:
            return i

    return 0


def trim_frame_border(
    img_bgr: np.ndarray,
    max_depth_pct: float = 0.12,
) -> np.ndarray:
    """
    Remove a uniform frame border from an already-cropped artwork image.

    Scans inward from each edge using per-row/column pixel standard deviation.
    A frame is characterised by low variance; artwork has visibly higher variance.

    Args:
        img_bgr:       Input crop in BGR.
        max_depth_pct: Maximum inward search depth per side (fraction of dim).

    Returns:
        Trimmed image, or original if no frame detected.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    max_h = max(1, int(h * max_depth_pct))
    max_w = max(1, int(w * max_depth_pct))

    # Central 80% to avoid corner overlap effects
    cx1, cx2 = int(w * 0.10), int(w * 0.90)
    cy1, cy2 = int(h * 0.10), int(h * 0.90)

    row_stds = gray[:, cx1:cx2].std(axis=1)
    col_stds = gray[cy1:cy2, :].std(axis=0)

    top = _find_trim_amount(row_stds[:max_h], max_h)
    bottom = _find_trim_amount(row_stds[h - max_h:][::-1], max_h)
    left = _find_trim_amount(col_stds[:max_w], max_w)
    right = _find_trim_amount(col_stds[w - max_w:][::-1], max_w)

    y1, y2 = top, h - bottom
    x1, x2 = left, w - right

    # Safety: never trim more than 50% of any dimension.
    if (y2 - y1) < h * 0.50 or (x2 - x1) < w * 0.50:
        return img_bgr

    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return img_bgr

    print(
        f"  [frame-trim] top={top}px  bottom={bottom}px  "
        f"left={left}px  right={right}px",
        flush=True,
    )
    return img_bgr[y1:y2, x1:x2]
