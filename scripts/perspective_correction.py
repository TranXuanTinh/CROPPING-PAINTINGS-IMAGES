#!/usr/bin/env python3
"""
perspective_correction.py
──────────────────────────
Corrects perspective distortion for paintings photographed at an angle
(leaning on walls, on easels, shot from the side).

Finds the best 4-point quadrilateral within the detected painting bbox,
validates it (convex, correct coverage, genuinely angled), and applies a
perspective warp to produce a front-facing rectangular crop.

Falls back to a simple rectangular crop when the painting is face-on or
when no valid quad is found.

Extracted and cleaned from the existing artwork_cropper_v3.py, lines 824–975.
The algorithm is proven and well-tested — no changes to core logic needed.
"""

from __future__ import annotations

import cv2
import numpy as np

from yolo_config import (
    PERSPECTIVE_COVERAGE_MIN,
    PERSPECTIVE_COVERAGE_MAX,
    PERSPECTIVE_MIN_ANGLE_DEV,
    PERSPECTIVE_MAX_BLACK,
)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective warp that maps *pts* to a front-facing rectangle."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 10 or H < 10:
        return img  # degenerate transform
    dst = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (W, H))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def perspective_correct(img_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Attempt to correct perspective distortion within the detected artwork bbox.

    For paintings that are angled, leaning, or shot from the side, the artwork's
    outer frame forms a quadrilateral that is not perfectly rectangular.  Detecting
    that quad and applying a four-point warp produces a front-facing crop.

    Quad validation (all must pass):
      - Exactly 4 vertices.
      - Convex hull (non-convex quads produce torn warps with black fill).
      - Covers 60–95% of the bbox region.
      - Every edge deviates >8° from axis-aligned (painting is genuinely tilted).
      - Warped result has no more than 8% black pixels.
      - Warp does not expand area by more than 10%.

    Falls back to a simple rectangular crop when no valid quad is found.

    Args:
        img_bgr:  Full input image in BGR.
        bbox:     [x1, y1, x2, y2] — the painting region.

    Returns:
        BGR crop: either perspective-corrected or simple rectangle.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    region = img_bgr[y1:y2, x1:x2]
    rh, rw = region.shape[:2]
    if rh == 0 or rw == 0:
        return region

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 15, 50)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    region_area = rh * rw
    best_pts, best_score = None, -1.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype("float32")

        # Convexity guard — non-convex quads produce inverted warps.
        if not cv2.isContourConvex(pts.astype(np.int32)):
            continue

        area = cv2.contourArea(approx)
        frac = area / region_area

        # Coverage guard.
        if not (PERSPECTIVE_COVERAGE_MIN < frac < PERSPECTIVE_COVERAGE_MAX):
            continue

        # Score: larger quads with more right-angle corners are better.
        rect_pts = _order_points(pts)
        edge_vecs = [rect_pts[(i + 1) % 4] - rect_pts[i] for i in range(4)]
        dots = [
            abs(np.dot(edge_vecs[i], edge_vecs[(i + 1) % 4]))
            / (
                np.linalg.norm(edge_vecs[i])
                * np.linalg.norm(edge_vecs[(i + 1) % 4])
                + 1e-6
            )
            for i in range(4)
        ]
        score = area * (1.0 - np.mean(dots))
        if score > best_score:
            best_score, best_pts = score, pts

    if best_pts is not None:
        # Skewness guard — only warp when genuinely tilted.
        rect_ord = _order_points(best_pts)
        deviations = []
        for i in range(4):
            e = rect_ord[(i + 1) % 4] - rect_ord[i]
            ang = float(
                np.degrees(
                    np.arctan2(float(abs(e[1])), float(abs(e[0])) + 1e-6)
                )
            )
            deviations.append(min(ang, 90.0 - ang))
        if min(deviations) < PERSPECTIVE_MIN_ANGLE_DEV:
            return region  # face-on — no warp needed

        warped = _four_point_transform(region, best_pts)
        warp_area = warped.shape[0] * warped.shape[1]

        # Area bounds: warp should not drastically change area.
        if not (region_area * 0.60 <= warp_area <= region_area * 1.10):
            return region

        # Black-pixel guard: >8% black = degenerate transform.
        black_ratio = float(np.mean(warped.sum(axis=2) == 0))
        if black_ratio > PERSPECTIVE_MAX_BLACK:
            return region

        return warped

    return region  # no valid quad → simple rectangular crop
