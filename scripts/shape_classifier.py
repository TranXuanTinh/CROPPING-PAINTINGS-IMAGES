#!/usr/bin/env python3
"""
shape_classifier.py
────────────────────
Determines whether a detected painting region is rectangular or oval/circular.

Pure OpenCV — no ML model needed.  Uses the geometric "extent" ratio
(contour_area / min_area_rect_area) which is a stable physical property of
the shape:
  - Circle:    extent = π/4 ≈ 0.785  (constant regardless of size/angle)
  - Rectangle: extent → 1.0          (drops slightly when angled)

The key insight is that a true ellipse maintains extent ≈ 0.785 across a RANGE
of background-difference thresholds, whereas a rectangle only grazes that value
at one incidental threshold.  Requiring consistency across multiple thresholds
is what separates the two reliably.
"""

from __future__ import annotations

import cv2
import numpy as np

from yolo_config import (
    OVAL_EXTENT_MIN,
    OVAL_EXTENT_MAX,
    OVAL_SOLIDITY_MIN,
    OVAL_MIN_VERTICES,
    OVAL_MIN_CONSISTENT,
    ROUND_THRESHOLDS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sample_corners(img_bgr: np.ndarray, border_px: int) -> np.ndarray:
    """Median BGR colour of the four corner patches."""
    b = border_px
    corners = [
        img_bgr[:b, :b],
        img_bgr[:b, -b:],
        img_bgr[-b:, :b],
        img_bgr[-b:, -b:],
    ]
    pixels = np.vstack([c.reshape(-1, 3) for c in corners]).astype(float)
    return np.median(pixels, axis=0)


def _round_shape_candidates(img_bgr: np.ndarray) -> list[dict]:
    """
    Scan across multiple background-difference thresholds and collect
    candidate contours that have ellipse-like extent, high solidity, and
    enough vertices to be non-rectangular.

    Returns a list of dicts with keys: t, contour, area, frac, extent,
    solidity, nverts.
    """
    h, w = img_bgr.shape[:2]
    b = max(15, int(min(h, w) * 0.03))
    bg = _sample_corners(img_bgr, b)
    diff = np.abs(img_bgr.astype(float) - bg).mean(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    candidates: list[dict] = []
    for t in ROUND_THRESHOLDS:
        mask = (diff > t).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        frac = area / (h * w)
        if not (0.10 < frac < 0.97) or len(c) < 5:
            continue

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        rect_area = rw * rh
        if rect_area <= 0:
            continue

        extent = area / rect_area
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        candidates.append({
            "t": t,
            "contour": c,
            "area": area,
            "frac": frac,
            "extent": extent,
            "solidity": solidity,
            "nverts": len(approx),
        })

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_round_shape(img_bgr: np.ndarray) -> dict | None:
    """
    Return the best-matching oval/circular candidate dict, or None if the
    painting looks rectangular.

    The candidate dict contains keys: t, contour, area, frac, extent,
    solidity, nverts.
    """
    candidates = _round_shape_candidates(img_bgr)
    if not candidates:
        return None

    # Filter for candidates that have ellipse-like geometry.
    round_like = [
        c for c in candidates
        if OVAL_EXTENT_MIN < c["extent"] < OVAL_EXTENT_MAX
        and c["solidity"] > OVAL_SOLIDITY_MIN
        and c["nverts"] > OVAL_MIN_VERTICES
    ]

    # Require consistency: enough thresholds must agree.
    if len(round_like) < max(OVAL_MIN_CONSISTENT, len(candidates) // 3):
        return None

    # Pick the most representative: closest to the median frac among round-like
    # hits (avoids threshold-noise outliers).
    fracs = sorted(c["frac"] for c in round_like)
    median_frac = fracs[len(fracs) // 2]
    return min(round_like, key=lambda c: abs(c["frac"] - median_frac))


def classify_shape(img_bgr: np.ndarray) -> str:
    """
    Classify the painting in *img_bgr* as 'oval' or 'rectangular'.

    Args:
        img_bgr: The detected painting region (cropped from the full image).

    Returns:
        'oval' if the shape is circular/oval, 'rectangular' otherwise.
    """
    result = detect_round_shape(img_bgr)
    if result is not None:
        return "oval"
    return "rectangular"
