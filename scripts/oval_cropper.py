#!/usr/bin/env python3
"""
oval_cropper.py
────────────────
Specialized handling for oval/circular paintings.

After YOLO detects the painting region and the shape classifier identifies
it as oval/circular, this module:

  1. Fits a coarse ellipse to the detected foreground contour
  2. Refines the ellipse using radial gradient search + RANSAC robust fitting
  3. Computes the precise axis-aligned bounding box from ellipse geometry
  4. Optionally generates an alpha mask for transparent-background output

The ellipse fitting code is extracted from the existing artwork_cropper.py
(lines 100–354) — it's well-engineered and handles edge cases like:
  - Fading canvas edges against near-white backgrounds
  - Hanging wire shadows that contaminate the contour at one spot
  - Slight camera angle distortion

The RANSAC refit is critical: it lets the majority of clean edge points
outvote the handful of contaminated ones (shadow, wire, wall mark).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from shape_classifier import detect_round_shape, _sample_corners
from yolo_config import ROUND_THRESHOLDS


# ──────────────────────────────────────────────────────────────────────────────
# Ellipse geometry
# ──────────────────────────────────────────────────────────────────────────────

def ellipse_aabb(
    ellipse: tuple,
    img_h: int,
    img_w: int,
) -> Optional[tuple[int, int, int, int]]:
    """
    Axis-aligned bounding box that exactly contains a (possibly rotated)
    ellipse, as returned by cv2.fitEllipse: ((cx, cy), (MA, ma), angle).

    Returns (x1, y1, x2, y2) clipped to the image, or None if degenerate.
    """
    (cx, cy), (ma_w, ma_h), angle = ellipse
    theta = np.radians(angle)
    a, b = ma_w / 2, ma_h / 2

    ux = float(np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2))
    uy = float(np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2))

    x1 = max(0, int(round(cx - ux)))
    y1 = max(0, int(round(cy - uy)))
    x2 = min(img_w, int(round(cx + ux)))
    y2 = min(img_h, int(round(cy + uy)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ──────────────────────────────────────────────────────────────────────────────
# Ellipse refinement (radial gradient search + RANSAC)
# ──────────────────────────────────────────────────────────────────────────────

def _point_ellipse_dist(pts: np.ndarray, ellipse: tuple) -> np.ndarray:
    """Approximate radial distance from each point to an ellipse boundary."""
    (cx, cy), (ma, mb), angle = ellipse
    theta = np.radians(angle)
    a, b = ma / 2.0, mb / 2.0
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2 + 1e-9)
    return np.abs(r - 1.0) * min(a, b)


def _radial_edge_points(
    gray: np.ndarray,
    ellipse: tuple,
    n_rays: int = 360,
    band_frac: float = 0.18,
    min_band: float = 18.0,
) -> np.ndarray:
    """
    Sample the strongest local gradient along many rays cast from the ellipse
    centre, searching a band around the current boundary estimate.

    This locates the true photometric edge directly instead of trusting the
    diff-threshold contour, which can be dragged by shadows or wall marks.
    """
    (cx, cy), (ma, mb), angle = ellipse
    theta = np.radians(angle)
    a, b = ma / 2.0, mb / 2.0
    band = max(min_band, band_frac * min(a, b))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (3, 3), 0)

    h, w = gray.shape[:2]
    ts = np.arange(-band, band + 1, 1.0)
    pts = []
    for i in range(n_rays):
        phi = 2 * np.pi * i / n_rays
        ex, ey = a * np.cos(phi), b * np.sin(phi)
        rx = ex * np.cos(theta) - ey * np.sin(theta)
        ry = ex * np.sin(theta) + ey * np.cos(theta)
        rlen = np.hypot(rx, ry)
        if rlen < 1e-3:
            continue
        dx, dy = rx / rlen, ry / rlen
        bx, by = cx + rx, cy + ry
        xs, ys = bx + dx * ts, by + dy * ts
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if valid.sum() < 3:
            continue
        xs_v, ys_v = xs[valid], ys[valid]
        gvals = grad[ys_v.astype(int), xs_v.astype(int)]
        j = int(np.argmax(gvals))
        if gvals[j] < 10:
            continue
        pts.append((xs_v[j], ys_v[j]))
    return np.array(pts, dtype=np.float32)


def _ransac_ellipse_fit(
    pts: np.ndarray,
    n_iter: int = 400,
    tol: float = 6.0,
    min_inlier_frac: float = 0.45,
) -> Optional[tuple]:
    """
    Robust ellipse fit: RANSAC with random subsets, keep hypothesis with
    most inliers, then refit using only those inliers.

    This lets the clean edge points outvote contamination artifacts.
    """
    n = len(pts)
    if n < 20:
        return None
    rng = np.random.default_rng(0)
    idx_all = np.arange(n)
    best_ellipse, best_inliers = None, -1

    for _ in range(n_iter):
        sample_idx = rng.choice(idx_all, size=min(12, n), replace=False)
        try:
            ell = cv2.fitEllipse(pts[sample_idx].reshape(-1, 1, 2))
        except cv2.error:
            continue
        inliers = int(np.sum(_point_ellipse_dist(pts, ell) < tol))
        if inliers > best_inliers:
            best_inliers, best_ellipse = inliers, ell

    if best_ellipse is None or best_inliers < min_inlier_frac * n:
        return best_ellipse

    inlier_pts = pts[
        _point_ellipse_dist(pts, best_ellipse) < tol
    ].reshape(-1, 1, 2)
    if len(inlier_pts) >= 10:
        try:
            return cv2.fitEllipse(inlier_pts)
        except cv2.error:
            pass
    return best_ellipse


def refine_ellipse(gray: np.ndarray, ellipse: tuple) -> tuple:
    """
    Snap an initial ellipse estimate onto the true photometric boundary
    via radial gradient search + robust RANSAC refit.

    Falls back to the original ellipse if refinement doesn't find enough
    reliable edge points.
    """
    pts = _radial_edge_points(gray, ellipse)
    refined = _ransac_ellipse_fit(pts)
    return refined if refined is not None else ellipse


# ──────────────────────────────────────────────────────────────────────────────
# Alpha mask for transparent-background output
# ──────────────────────────────────────────────────────────────────────────────

def _ellipse_alpha_mask(
    shape_hw: tuple[int, int],
    ellipse: tuple,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """
    Filled ellipse mask (0/255, lightly feathered) sized to *shape_hw*,
    with the ellipse re-centred to match a crop's local coordinate frame.
    """
    (cx, cy), (ma_w, ma_h), angle = ellipse
    local = ((cx - offset_x, cy - offset_y), (ma_w, ma_h), angle)
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.ellipse(mask, local, 255, -1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (5, 5), 0)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def crop_oval(
    img_bgr: np.ndarray,
    padding_pct: float = 3.0,
    cutout: bool = False,
) -> np.ndarray:
    """
    Crop an oval/circular painting from its background.

    Fits an ellipse to the detected boundary and derives the exact axis-aligned
    bounding box from the ellipse geometry — stays accurate even where the canvas
    edge fades into a near-white background.

    Args:
        img_bgr:     Full image in BGR.
        padding_pct: Padding around the ellipse bbox (percentage).
        cutout:      If True, return BGRA image with transparent background
                     outside the ellipse.  Requires saving as PNG.

    Returns:
        Cropped image (BGR or BGRA if cutout=True).
    """
    h, w = img_bgr.shape[:2]
    cand = detect_round_shape(img_bgr)

    if cand is None:
        print("  [oval] No oval shape detected — returning input region.")
        return img_bgr

    # Fit coarse ellipse to the detected contour
    coarse_ellipse = cv2.fitEllipse(cand["contour"])

    # Refine with radial gradient search + RANSAC
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ellipse = refine_ellipse(gray, coarse_ellipse)

    # Compute precise AABB from ellipse geometry
    bbox = ellipse_aabb(ellipse, h, w)
    if bbox is None:
        print("  [oval] Degenerate ellipse bbox — returning input region.")
        return img_bgr

    x1, y1, x2, y2 = bbox
    print(
        f"  [oval] t={cand['t']} extent={cand['extent']:.3f} "
        f"frac={cand['frac']:.0%} ellipse=({x2 - x1}×{y2 - y1}) "
        f"angle={ellipse[2]:.0f}°"
    )

    # Add padding
    if padding_pct > 0:
        pw = int((x2 - x1) * padding_pct / 100)
        ph = int((y2 - y1) * padding_pct / 100)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)

    crop = img_bgr[y1:y2, x1:x2]

    if not cutout:
        return crop

    # Transparent background mode
    mask = _ellipse_alpha_mask(crop.shape[:2], ellipse, x1, y1)
    b, g, r = cv2.split(crop)
    return cv2.merge([b, g, r, mask])
