#!/usr/bin/env python3
"""
artwork_cropper.py  v7
──────────────────────
Automatically crops artwork out of product images.

Four modes, auto-selected or manually specified:

  plain   — Canvas/artwork clearly contrasts its neutral background.
             Multi-strategy: Otsu, Canny, brightness threshold.
             Use for: unframed canvas on white/grey wall, dark textile on white wall.

  bgdiff  — Artwork on a background that is a DIFFERENT colour (even subtly).
             Samples the full border for reference, then finds the largest region
             that differs from it. Adaptive threshold selection ensures clean crops
             for: paper on blue-grey bg, watercolour on grey bg, canvas on coloured
             bg, matted prints (salmon/peach print on cream paper), small canvas on
             white wall where canvas colour differs from wall.

  pinned  — Paper/card clipped or pinned to a wall where paper ≈ wall colour.
             Finds INK CONTENT bbox, pads out to recover paper margin.
             Use for: sparse ink drawings on off-white paper on off-white wall.

  room    — Artwork in a room, on a wood/plank wall, leaning on a pedestal.
             Quad detection + perspective correction. Fallback: region segmentation.

  round   — Circular or oval-shaped canvas (not a rectangle). Detected via a
             geometric extent test, then cropped from a fitted ellipse so the
             tight bounding box hugs the true curved edge, not noisy pixels.

  auto    — Heuristic classifier picks the best mode (default).

Usage:
  python artwork_cropper.py image.jpg
  python artwork_cropper.py image.jpg --mode bgdiff
  python artwork_cropper.py image.jpg --mode pinned --padding 20
  python artwork_cropper.py ./folder/ --batch
  python artwork_cropper.py ./folder/ --batch --output ./out/ --mode bgdiff

Dependencies: opencv-python, numpy  (MIT/BSD — completely free)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def add_padding(x, y, w, h, pad_pct: float, img_h: int, img_w: int):
    px, py = int(w * pad_pct / 100), int(h * pad_pct / 100)
    return max(0, x - px), max(0, y - py), min(img_w, x + w + px), min(img_h, y + h + py)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    tl, tr, br, bl = rect
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (W, H))


def sample_border(img: np.ndarray, b: int) -> np.ndarray:
    """Median BGR colour of the full outer border strip (all 4 sides)."""
    strips = [img[:b, :], img[-b:, :], img[:, :b], img[:, -b:]]
    pixels = np.vstack([s.reshape(-1, 3) for s in strips]).astype(float)
    return np.median(pixels, axis=0)


def sample_corners(img: np.ndarray, b: int) -> np.ndarray:
    """Median BGR colour of the four corner patches."""
    corners = [img[:b, :b], img[:b, -b:], img[-b:, :b], img[-b:, -b:]]
    pixels = np.vstack([c.reshape(-1, 3) for c in corners]).astype(float)
    return np.median(pixels, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Mode: round  (circular / oval canvas — not a rectangle)
# ──────────────────────────────────────────────────────────────────────────────
#
# Key idea: a true ellipse (circle is just an ellipse with equal axes) has
# contour_area ≈ (π/4) × rotated-bounding-rect area, i.e. "extent" ≈ 0.785,
# and this ratio holds steady across a wide range of background-diff
# thresholds because the boundary is one smooth curve — growing/shrinking the
# threshold just grows/shrinks the ellipse a little, the ratio barely moves.
# A rectangular canvas (even photographed at a slight angle) only grazes
# extent≈0.785 by coincidence at one incidental threshold; neighbouring
# thresholds snap back toward 0.9–1.0 or fragment unpredictably. Requiring
# several thresholds to agree, plus high solidity (clean unnotched boundary)
# and a vertex count too high for a simple quad, separates true round/oval
# shapes from rectangles reliably.

_ROUND_THRESHOLDS = (6, 8, 10, 12, 15, 18, 20, 25, 30, 35)


def _round_shape_candidates(img: np.ndarray) -> list[dict]:
    h, w = img.shape[:2]
    b = max(15, int(min(h, w) * 0.03))
    bg = sample_corners(img, b)
    diff = np.abs(img.astype(float) - bg).mean(axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    candidates = []
    for t in _ROUND_THRESHOLDS:
        mask = (diff > t).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            "t": t, "contour": c, "area": area, "frac": frac,
            "extent": extent, "solidity": solidity, "nverts": len(approx),
        })
    return candidates


def detect_round_shape(img: np.ndarray) -> dict | None:
    """
    Return the best-matching round/oval candidate dict, or None if the
    artwork looks rectangular.
    """
    candidates = _round_shape_candidates(img)
    if not candidates:
        return None
    round_like = [c for c in candidates
                  if 0.70 < c["extent"] < 0.86 and c["solidity"] > 0.96 and c["nverts"] > 7]
    if len(round_like) < max(2, len(candidates) // 3):
        return None
    # Most representative candidate: the one whose frac is closest to the
    # median frac among round-like hits (avoids picking a threshold-noise outlier).
    fracs = sorted(c["frac"] for c in round_like)
    med = fracs[len(fracs) // 2]
    return min(round_like, key=lambda c: abs(c["frac"] - med))


def ellipse_aabb(ellipse, img_h: int, img_w: int):
    """
    Axis-aligned bounding box that exactly contains a (possibly rotated)
    ellipse, as returned by cv2.fitEllipse: ((cx, cy), (MA, ma), angle).
    Returns (x1, y1, x2, y2) clipped to the image, or None if degenerate.
    """
    (cx, cy), (ma_w, ma_h), angle = ellipse
    theta = np.radians(angle)
    a, bb = ma_w / 2, ma_h / 2
    ux = float(np.sqrt((a * np.cos(theta)) ** 2 + (bb * np.sin(theta)) ** 2))
    uy = float(np.sqrt((a * np.sin(theta)) ** 2 + (bb * np.cos(theta)) ** 2))

    x1, y1, x2, y2 = cx - ux, cy - uy, cx + ux, cy + uy
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(img_w, int(round(x2))), min(img_h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _point_ellipse_dist(pts: np.ndarray, ellipse) -> np.ndarray:
    """Approximate radial distance from each point to an ellipse's boundary."""
    (cx, cy), (ma, mb), angle = ellipse
    theta = np.radians(angle)
    a, b = ma / 2.0, mb / 2.0
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2 + 1e-9)
    return np.abs(r - 1.0) * min(a, b)


def _radial_edge_points(gray: np.ndarray, ellipse,
                         n_rays: int = 360, band_frac: float = 0.18,
                         min_band: float = 18.0) -> np.ndarray:
    """
    Sample the strongest local gradient (the real canvas/wall edge) along
    many rays cast from the ellipse centre, searching a band around the
    current boundary estimate. This locates the true photometric edge
    directly instead of trusting the diff-threshold mask's contour, which can
    be dragged off by a hanging wire, cast shadow, or wall mark that merges
    into the contour at one or two localised spots. The band has to be wide
    enough to reach past such an artifact, not just hug the (already biased)
    starting ellipse — too narrow and the search just re-confirms the artifact.
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


def _ransac_ellipse_fit(pts: np.ndarray, n_iter: int = 400, tol: float = 6.0,
                         min_inlier_frac: float = 0.45):
    """
    Robust ellipse fit over candidate edge points: repeatedly fit to small
    random subsets, keep the hypothesis with the most inliers, then refit
    using only those inliers. This is what lets the handful of points still
    stuck on a contamination artifact get outvoted by the majority that found
    the true edge, instead of dragging an ordinary least-squares fit off.
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
    inlier_pts = pts[_point_ellipse_dist(pts, best_ellipse) < tol].reshape(-1, 1, 2)
    if len(inlier_pts) >= 10:
        try:
            return cv2.fitEllipse(inlier_pts)
        except cv2.error:
            pass
    return best_ellipse


def refine_round_ellipse(gray: np.ndarray, ellipse):
    """
    Snap an initial ellipse estimate (from the diff-threshold contour) onto
    the true photometric boundary via radial gradient search + robust refit.
    Falls back to the original ellipse if refinement doesn't turn up enough
    reliable edge points.
    """
    pts = _radial_edge_points(gray, ellipse)
    refined = _ransac_ellipse_fit(pts)
    return refined if refined is not None else ellipse


def _ellipse_alpha_mask(shape_hw: tuple, ellipse, offset_x: int, offset_y: int) -> np.ndarray:
    """
    Filled ellipse mask (0/255, lightly feathered) sized to *shape_hw*, with
    the ellipse re-centred by (-offset_x, -offset_y) to match a crop's local
    coordinate frame.
    """
    (cx, cy), (ma_w, ma_h), angle = ellipse
    local = ((cx - offset_x, cy - offset_y), (ma_w, ma_h), angle)
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.ellipse(mask, local, 255, -1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def crop_round(img: np.ndarray, padding: float = 3.0, cutout: bool = False) -> np.ndarray:
    """
    Circular or oval canvas. Fits an ellipse to the detected boundary and
    derives the exact axis-aligned bounding box from the ellipse geometry
    (center, axes, rotation) rather than a noisy pixel bounding box — this
    stays accurate even where the canvas edge fades into a near-white
    background (the ellipse model bridges low-contrast arcs that a per-pixel
    threshold would miss).

    When cutout=True, returns a BGRA image with the region outside the fitted
    ellipse made transparent — a true die-cut shape instead of just a tight
    rectangle around it. Requires saving as PNG (JPEG has no alpha channel).
    """
    h, w = img.shape[:2]
    cand = detect_round_shape(img)
    if cand is None:
        print("  [round] No round shape found — falling back to bgdiff.")
        return crop_bgdiff(img, padding)

    coarse_ellipse = cv2.fitEllipse(cand["contour"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ellipse = refine_round_ellipse(gray, coarse_ellipse)
    bbox = ellipse_aabb(ellipse, h, w)
    if bbox is None:
        print("  [round] Degenerate ellipse bbox — falling back to bgdiff.")
        return crop_bgdiff(img, padding)

    x1, y1, x2, y2 = bbox
    print(f"  [round] t={cand['t']} extent={cand['extent']:.3f} "
          f"frac={cand['frac']:.0%} ellipse=({x2 - x1}x{y2 - y1}) angle={ellipse[2]:.0f}")

    if padding > 0:
        x1, y1, x2, y2 = add_padding(x1, y1, x2 - x1, y2 - y1, padding, h, w)
    crop = img[y1:y2, x1:x2]

    if not cutout:
        return crop

    mask = _ellipse_alpha_mask(crop.shape[:2], ellipse, x1, y1)
    b, g, r = cv2.split(crop)
    return cv2.merge([b, g, r, mask])


# ──────────────────────────────────────────────────────────────────────────────
# Mode 0: wall  (painting on a neutral plain wall / rendered mockup)
# ──────────────────────────────────────────────────────────────────────────────

def _bilinear_wall_model(img: np.ndarray, b: int) -> np.ndarray:
    """
    Bilinear interpolation of corner mean colours across the full image.
    Accounts for vignette / gradient effects in rendered product mockups
    (e.g. top-left corner ≈ 204, top-right ≈ 240 on the same grey wall).
    Returns float32 BGR array same shape as img.
    """
    h, w = img.shape[:2]
    tl = img[:b,  :b ].reshape(-1, 3).mean(axis=0).astype(np.float32)
    tr = img[:b,  -b:].reshape(-1, 3).mean(axis=0).astype(np.float32)
    bl = img[-b:, :b ].reshape(-1, 3).mean(axis=0).astype(np.float32)
    br = img[-b:, -b:].reshape(-1, 3).mean(axis=0).astype(np.float32)
    fy = np.linspace(0., 1., h, dtype=np.float32)[:, np.newaxis]   # (h,1)
    fx = np.linspace(0., 1., w, dtype=np.float32)[np.newaxis, :]   # (1,w)
    wall = np.stack(
        [(1-fy)*(1-fx)*tl[c] + (1-fy)*fx*tr[c] +
           fy *(1-fx)*bl[c] +  fy *fx*br[c]
         for c in range(3)],
        axis=2
    )
    return wall


def _fwd_min(arr: np.ndarray, win: int) -> np.ndarray:
    """Forward sliding minimum: fwd_min[i] = min(arr[i : i+win])."""
    n = len(arr)
    padded = np.pad(arr, (0, win), mode='edge')
    idx = np.arange(n)[:, None] + np.arange(win)
    return padded[idx].min(axis=1)


def _crop_painting_from_wall(img: np.ndarray, padding: float = 0.0) -> np.ndarray | None:
    """
    Find painting boundary by scanning row/column diff-from-background profiles.

    Two-pass design:
    Pass 1 (loose) — initial threshold finds the rough painting region.
        · Wall colour sampled from CORNER patches only (border strips get
          contaminated when painting extends to an image edge).
        · Starting threshold = 50% of centre-of-image diff, floor 12, cap 90.
        · If the loose region covers > 80% of the image (wall shadow leaked in),
          the threshold is raised by +2 until coverage drops below 80%.
          This handles low-contrast paintings on similar-tone walls where the
          minimum threshold at 12 would detect wall shadows as painting.

    Pass 2 (dense-core) — inner percentage profiles tighten the crop.
        · For rows: what fraction of *detected columns* has painting pixels?
          (canvas sides only cover part of painting height → lower inner pct)
        · For cols: what fraction of *detected rows* has painting pixels?
          (shadow columns beyond painting edge have partial vertical coverage)
        · Dense thresholds = 75th-percentile of painting region × multiplier:
          rows × 0.60 (looser — keeps painting bottom rows even when narrowing)
          cols × 0.75 (tighter — excludes canvas side & shadow columns)

    Returns None if no plausible crop is found.
    """
    h, w = img.shape[:2]
    c = max(20, int(min(h, w) * 0.05))

    # ── Step 1: wall colour from corners only ───────────────────────────────
    corners_px = np.vstack([
        img[:c, :c].reshape(-1, 3),
        img[:c, -c:].reshape(-1, 3),
        img[-c:, :c].reshape(-1, 3),
        img[-c:, -c:].reshape(-1, 3),
    ]).astype(float)
    bg_color = corners_px.mean(axis=0)
    diff = np.abs(img.astype(float) - bg_color).mean(axis=2)

    # ── Step 2: adaptive threshold with over-detection guard ────────────────
    cy, cx = h // 2, w // 2
    cp = max(10, int(min(h, w) * 0.05))
    center_diff = float(diff[cy - cp:cy + cp, cx - cp:cx + cp].mean())
    # 50% of centre diff (up from 30%): better excludes shadow/transition zones
    pixel_thresh = float(np.clip(center_diff * 0.50, 12, 90))

    # If threshold is too low (wall shadows leak in) raise it until coverage < 80%
    for _ in range(12):
        _rp = (diff > pixel_thresh).mean(axis=1)
        _cp = (diff > pixel_thresh).mean(axis=0)
        _lr = np.where(_rp > 0.05)[0]
        _lc = np.where(_cp > 0.05)[0]
        if len(_lr) == 0 or len(_lc) == 0:
            break
        _frac = (_lr[-1] - _lr[0] + 1) * (_lc[-1] - _lc[0] + 1) / (h * w)
        if _frac < 0.80:
            break
        pixel_thresh = min(pixel_thresh + 2, 90)

    # ── Step 3: Pass 1 — loose detection (any row/col with > 5% painting) ──
    row_pct = (diff > pixel_thresh).mean(axis=1)
    col_pct = (diff > pixel_thresh).mean(axis=0)
    loose_rows = np.where(row_pct > 0.05)[0]
    loose_cols = np.where(col_pct > 0.05)[0]

    # Fallback: if nothing found, try lower thresholds
    if len(loose_rows) == 0 or len(loose_cols) == 0:
        for fallback_t in [10, 8, 6]:
            rp = (diff > fallback_t).mean(axis=1)
            cp2 = (diff > fallback_t).mean(axis=0)
            loose_rows = np.where(rp > 0.05)[0]
            loose_cols = np.where(cp2 > 0.05)[0]
            if len(loose_rows) > 0 and len(loose_cols) > 0:
                pixel_thresh = fallback_t
                break

    if len(loose_rows) == 0 or len(loose_cols) == 0:
        return None

    y1l, y2l = int(loose_rows[0]), int(loose_rows[-1]) + 1
    x1l, x2l = int(loose_cols[0]), int(loose_cols[-1]) + 1
    frac_l = (y2l - y1l) * (x2l - x1l) / (h * w)

    # ── Step 4: Pass 2 — dense-core detection using inner percentages ───────
    # row_pct_inner[r] = fraction of detected columns where row r has painting
    # col_pct_inner[c] = fraction of detected rows where col c has painting
    row_pct_inner = (diff[:, x1l:x2l] > pixel_thresh).mean(axis=1)
    col_pct_inner = (diff[y1l:y2l, :] > pixel_thresh).mean(axis=0)

    p75_row = float(np.percentile(row_pct_inner[y1l:y2l], 75))
    p75_col = float(np.percentile(col_pct_inner[x1l:x2l], 75))
    # Rows: 60% of typical → keeps painting bottom even if it narrows there
    # Cols: 75% of typical → excludes canvas sides & shadow beyond painting edge
    row_dense = max(0.08, p75_row * 0.60)
    col_dense = max(0.08, p75_col * 0.75)

    dense_rows = np.where(row_pct_inner > row_dense)[0]
    dense_cols = np.where(col_pct_inner > col_dense)[0]

    y1 = int(dense_rows[0]) if len(dense_rows) > 0 else y1l
    y2 = int(dense_rows[-1]) + 1 if len(dense_rows) > 0 else y2l
    x1 = int(dense_cols[0]) if len(dense_cols) > 0 else x1l
    x2 = int(dense_cols[-1]) + 1 if len(dense_cols) > 0 else x2l
    frac = (y2 - y1) * (x2 - x1) / (h * w)

    if not (0.05 < frac < 0.95):
        # Dense result invalid, fall back to loose
        if not (0.05 < frac_l < 0.95):
            return None
        y1, y2, x1, x2, frac = y1l, y2l, x1l, x2l, frac_l

    print(f"  [wall_profile] thresh={pixel_thresh:.0f} frac={frac:.0%}")
    cw, ch2 = x2 - x1, y2 - y1
    if padding > 0:
        px = int(cw * padding / 100)
        py = int(ch2 * padding / 100)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return img[y1:y2, x1:x2]


def crop_wall(img: np.ndarray, padding: float = 0.0) -> np.ndarray:
    """
    Painting/canvas on a plain neutral wall (grey rendered mockup or real wall).

    Strategy:
    1. Build a bilinear wall model from the four corners to handle vignette.
    2. Mark "definite painting" pixels: BOTH high colour diff AND high texture.
       Shadows have colour diff but near-zero texture, so they're excluded.
    3. Find a quick bounding box with a low density threshold (catches everything).
    4. For each of the 4 edges, try to trim shadow zone by scanning inward with a
       forward-looking minimum window.  If the trim would exceed 5% of the image
       dimension the shadow-trim is discarded and the quick edge is kept — this
       prevents over-cropping smooth paintings that have low texture.
    5. The result is the tightest crop that doesn't eat into painting content.

    For real gallery walls (non-studio), we first try _crop_painting_from_wall
    which uses row/column background-diff profiles and is more reliable than
    bilinear texture modelling on real-wall images.
    """
    h, w = img.shape[:2]

    # ── Profile-based crop (primary for real gallery walls) ──────────────────
    profile_result = _crop_painting_from_wall(img, padding)
    if profile_result is not None:
        pfrac = (profile_result.shape[0] * profile_result.shape[1]) / (h * w)
        if 0.08 < pfrac < 0.92:
            return profile_result
    # ── Texture-based crop (bilinear wall model, best for studio mockups) ────
    b = max(25, int(min(h, w) * 0.05))

    wall_model = _bilinear_wall_model(img, b)
    cdiff = np.abs(img.astype(np.float32) - wall_model).mean(axis=2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = np.abs(cv2.Laplacian(blurred, cv2.CV_32F))
    corner_tex = (lap[:b, :b].mean() + lap[:b, -b:].mean() +
                  lap[-b:, :b].mean() + lap[-b:, -b:].mean()) / 4
    tex = np.maximum(lap - (corner_tex * 1.5 + 2.0), 0.0)

    painting = ((cdiff > 12) & (tex > 3)).astype(np.float32)
    row_density = painting.mean(axis=1)
    col_density = painting.mean(axis=0)

    row_cdiff_mean = cdiff.mean(axis=1)
    col_cdiff_mean = cdiff.mean(axis=0)

    # Quick bbox: any row/col with ≥ 3% definite-painting pixels
    y_q = np.where(row_density > 0.03)[0]
    x_q = np.where(col_density > 0.03)[0]
    if len(y_q) == 0 or len(x_q) == 0:
        print("  [wall] No painting found — trying bgdiff fallback.")
        r = _try_bgdiff_validated(img, padding, min_frac=0.04, max_frac=0.95)
        if r is not None:
            return r
        print("  [wall] bgdiff fallback failed — returning original.")
        return img

    y1_q, y2_q = int(y_q[0]), int(y_q[-1])
    x1_q, x2_q = int(x_q[0]), int(x_q[-1])

    # Shadow-trim: scan inward from each quick edge.
    # We look for the first position where the NEXT win rows/cols are
    # ALL above a sustained threshold. Cap trim at 5% of image dimension.
    win_r = max(10, int(h * 0.015))
    win_c = max(10, int(w * 0.015))
    max_trim_r = int(h * 0.05)
    max_trim_c = int(w * 0.05)
    thr = 0.05

    fwd_row = _fwd_min(row_density, win_r)
    bwd_row = _fwd_min(row_density[::-1], win_r)[::-1]
    fwd_col = _fwd_min(col_density, win_c)
    bwd_col = _fwd_min(col_density[::-1], win_c)[::-1]

    def trim_edge(fwd_arr, quick_edge, direction, max_trim):
        """Scan from quick_edge inward (direction=+1 for top/left, -1 for bot/right)."""
        step = direction
        pos = quick_edge
        limit = quick_edge + direction * max_trim
        while pos != limit and 0 <= pos < len(fwd_arr):
            if fwd_arr[pos] > thr:
                return pos
            pos += step
        return quick_edge  # can't trim enough → keep quick edge

    y1 = trim_edge(fwd_row,  y1_q, +1, max_trim_r)
    y2 = trim_edge(bwd_row,  y2_q, -1, max_trim_r)
    x1 = trim_edge(fwd_col,  x1_q, +1, max_trim_c)
    x2 = trim_edge(bwd_col,  x2_q, -1, max_trim_c)

    # Mean-cdiff snap: handles smooth-canvas paintings whose edges have low texture
    # but clear colour difference from wall.  The border patch gives the wall baseline;
    # any row/col whose full-width mean_cdiff exceeds 2× that baseline is painting.
    border_cdiff = np.concatenate([row_cdiff_mean[:b], row_cdiff_mean[-b:]])
    wall_cdiff_base = np.percentile(border_cdiff, 75)
    snap_thr = max(wall_cdiff_base * 2.0, 8.0)
    max_snap = min(30, int(min(h, w) * 0.015))

    def cdiff_snap(cdiff_arr, edge, direction):
        for delta in range(max_snap + 1):
            pos = edge + direction * delta
            if pos < 0 or pos >= len(cdiff_arr):
                break
            if cdiff_arr[pos] > snap_thr:
                return pos
        return edge

    y1 = cdiff_snap(row_cdiff_mean, y1, +1)
    y2 = cdiff_snap(row_cdiff_mean, y2, -1)
    x1 = cdiff_snap(col_cdiff_mean, x1, +1)
    x2 = cdiff_snap(col_cdiff_mean, x2, -1)

    frac = (x2 - x1) * (y2 - y1) / (h * w)
    if frac < 0.05 or frac > 0.97:
        print(f"  [wall] frac={frac:.0%} out of range — trying bgdiff fallback.")
        r = _try_bgdiff_validated(img, padding, min_frac=0.04, max_frac=0.95)
        if r is not None:
            print(f"  [wall] bgdiff fallback succeeded.")
            return r
        print(f"  [wall] bgdiff fallback failed — returning original.")
        return img

    print(f"  [wall] frac={frac:.0%}")

    if padding > 0:
        x1, y1, x2, y2 = add_padding(x1, y1, x2 - x1, y2 - y1, padding, h, w)
    return img[y1:y2, x1:x2]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: plain
# ──────────────────────────────────────────────────────────────────────────────

def crop_plain(img: np.ndarray, padding: float = 2.0) -> np.ndarray:
    """
    Artwork clearly contrasts a neutral solid background.
    Tries Otsu (both polarities), Canny edges, and explicit brightness thresholds.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    k_sm = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    k_lg = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    candidates = []

    # A: Otsu, both polarities
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in [otsu, cv2.bitwise_not(otsu)]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_sm)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > h * w * 0.03]
        if valid:
            x, y, cw, ch = cv2.boundingRect(np.vstack(valid))
            frac = cw * ch / (h * w)
            if 0.06 < frac < 0.97:
                candidates.append((frac, (x, y, cw, ch)))

    # B: Canny edges
    edges = cv2.Canny(blur, 25, 80)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_sm)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, k_sm)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > h * w * 0.03]
    if valid:
        x, y, cw, ch = cv2.boundingRect(np.vstack(valid))
        frac = cw * ch / (h * w)
        if 0.06 < frac < 0.97:
            candidates.append((frac, (x, y, cw, ch)))

    # C: Explicit brightness threshold vs background
    bg_mean = int(np.median(blur[: h // 8, :]))
    for tv, inv in [(max(bg_mean - 25, 30), False), (min(bg_mean + 25, 225), True)]:
        flag = cv2.THRESH_BINARY_INV if not inv else cv2.THRESH_BINARY
        _, mask = cv2.threshold(blur, tv, 255, flag)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_lg)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_lg)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:1]:
            x, y, cw, ch = cv2.boundingRect(c)
            frac = cw * ch / (h * w)
            if 0.06 < frac < 0.97:
                candidates.append((frac, (x, y, cw, ch)))

    if not candidates:
        print("  [plain] No clear artwork found — returning original.")
        return img

    candidates.sort(key=lambda c: c[0], reverse=True)
    x, y, cw, ch = candidates[0][1]
    x1, y1, x2, y2 = add_padding(x, y, cw, ch, padding, h, w)
    return img[y1:y2, x1:x2]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: bgdiff
# ──────────────────────────────────────────────────────────────────────────────

def crop_bgdiff(img: np.ndarray, padding: float = 2.0) -> np.ndarray:
    """
    Crops artwork from a background that differs in colour (even subtly).

    Samples the full outer border for background colour reference, then finds
    the largest region that differs from it using adaptive threshold selection.

    The "preferred zone" (15–80% of image area) ensures we neither include the
    full image nor over-crop to just the artwork content. This handles:
    - Paper on distinctly coloured bg (blue-grey, grey)
    - Canvas on slightly different-toned wall
    - Matted prints (salmon/coloured print rect on cream paper on cream bg)
    - Small canvas on white wall with visible canvas edge colour
    """
    h, w = img.shape[:2]
    b = max(20, int(min(h, w) * 0.04))
    bg = sample_border(img, b)
    diff = np.abs(img.astype(float) - bg).mean(axis=2)

    candidates = []  # (thresh, frac, bbox)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    # Full-bleed check: artwork fills the canvas edge-to-edge (no white margin).
    # Detected by: corners are pure white (bg median > 240) but the full border
    # strip mean is lower (< 232) because the artwork's colours bleed to the edge.
    b_strips = max(20, int(min(h, w) * 0.04))
    full_strip = np.vstack([
        img[:b_strips, :].reshape(-1, 3),
        img[-b_strips:, :].reshape(-1, 3),
        img[:, :b_strips].reshape(-1, 3),
        img[:, -b_strips:].reshape(-1, 3)
    ]).astype(float)
    strip_mean = full_strip.mean()
    if bg.mean() > 240 and np.std(bg) < 5 and strip_mean < 232:
        pct_diff = np.mean(diff > 15)
        if pct_diff > 0.55:
            print('  [bgdiff] Full-bleed artwork detected — returning original.')
            return img

    for thresh in [5, 6, 8, 10, 12, 15, 18, 20, 25, 30]:
        mask = (diff > thresh).astype(np.uint8) * 255
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
            x, y, cw, ch = cv2.boundingRect(c)
            frac = cw * ch / (h * w)
            if 0.08 < frac < 0.92:
                candidates.append((thresh, frac, (x, y, cw, ch)))
                break

    if not candidates:
        print("  [bgdiff] No region found — falling back to plain mode.")
        return crop_plain(img, padding)

    # Prefer results in the 15–80% range (clearly cropped, not too small)
    preferred = [(t, f, b) for t, f, b in candidates if 0.12 < f < 0.88]
    pool = preferred if preferred else candidates
    # Among the preferred, take the largest fraction (most generous crop)
    pool.sort(key=lambda x: x[1], reverse=True)
    _, frac, (x, y, cw, ch) = pool[0]

    # If bgdiff result is very generous (>75%), also try the profile-based crop.
    # The profile approach uses higher pixel-diff thresholds (≥15) which avoids
    # bridging wall noise into the painting region via morphological closing.
    if frac > 0.75:
        profile = _crop_painting_from_wall(img, padding)
        if profile is not None:
            pfrac = (profile.shape[0] * profile.shape[1]) / (h * w)
            if 0.08 < pfrac < frac - 0.05:  # meaningfully tighter, still valid
                print(f"  [bgdiff] Profile crop ({pfrac:.0%}) tighter than bgdiff ({frac:.0%}) — using profile.")
                return profile

    x1, y1, x2, y2 = add_padding(x, y, cw, ch, padding, h, w)
    return img[y1:y2, x1:x2]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: pinned
# ──────────────────────────────────────────────────────────────────────────────

def crop_pinned(img: np.ndarray, padding: float = 18.0) -> np.ndarray:
    """
    Paper/card pinned or clipped to a wall where paper ≈ wall colour.
    Detects the INK/CONTENT bounding box, pads generously to recover paper margin.
    Default 18% padding matches typical artist paper margins.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > h * w * 0.005]

    if not valid:
        print("  [pinned] No content — falling back to plain mode.")
        return crop_plain(img, 2.0)

    x, y, cw, ch = cv2.boundingRect(np.vstack(valid))
    if cw * ch / (h * w) > 0.85:
        print("  [pinned] Content fills frame — returning original.")
        return img

    x1, y1, x2, y2 = add_padding(x, y, cw, ch, padding, h, w)
    return img[y1:y2, x1:x2]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 4: room
# ──────────────────────────────────────────────────────────────────────────────

def _find_best_quad(gray: np.ndarray, img_h: int, img_w: int):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 15, 50)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_pts, best_score = None, -1.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        frac = area / (img_h * img_w)
        if not (0.08 < frac < 0.88):
            continue
        pts = approx.reshape(4, 2).astype("float32")
        rect = order_points(pts)
        ev = [rect[(i + 1) % 4] - rect[i] for i in range(4)]
        dots = [abs(np.dot(ev[i], ev[(i + 1) % 4])) /
                (np.linalg.norm(ev[i]) * np.linalg.norm(ev[(i + 1) % 4]) + 1e-6)
                for i in range(4)]
        score = area * (1.0 - np.mean(dots))
        if score > best_score:
            best_score, best_pts = score, pts

    return best_pts, best_score


def crop_room(img: np.ndarray, padding: float = 0.0) -> np.ndarray:
    """
    Artwork in a room, on a wood/plank wall, or leaning on a pedestal.
    Quad detection + perspective correction; falls back to region segmentation.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pts, score = _find_best_quad(gray, h, w)
    if pts is not None and score > 0:
        cropped = four_point_transform(img, pts)
        if padding > 0:
            ch2, cw2 = cropped.shape[:2]
            px, py = int(cw2 * padding / 100), int(ch2 * padding / 100)
            cropped = cropped[py: ch2 - py, px: cw2 - px]
        # Reject if perspective warp expanded the image beyond the original size
        frac_q = (cropped.shape[0] * cropped.shape[1]) / (h * w)
        if frac_q <= 0.97:
            return cropped
        print(f"  [room] Quad result too large (frac={frac_q:.0%}) — falling back.")

    # Quad inconclusive — try bgdiff (painting differs from room border),
    # then region segmentation, then plain.
    print("  [room] Quad inconclusive — trying bgdiff then region segmentation.")
    r = _try_bgdiff_validated(img, padding)
    if r is not None:
        return r

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best_region, best_score = None, -1.0

    for tv in [60, 80, 100, 120, 150]:
        for inv in [False, True]:
            flag = cv2.THRESH_BINARY if not inv else cv2.THRESH_BINARY_INV
            _, mask = cv2.threshold(blur, tv, 255, flag)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                x, y, cw, ch = cv2.boundingRect(c)
                frac = cw * ch / (h * w)
                if not (0.08 < frac < 0.88):
                    continue
                aspect = min(cw, ch) / max(cw, ch)
                cx2, cy2 = x + cw / 2, y + ch / 2
                center_s = 1 - abs(cx2 / w - 0.5) - abs(cy2 / h - 0.5)
                score = aspect * frac * max(0, center_s)
                if score > best_score:
                    best_score, best_region = score, (x, y, cw, ch)

    if best_region and best_score > 0.05:
        x, y, cw, ch = best_region
        # Validate aspect ratio — reject window-like (very narrow) strips
        aspect_ratio = cw / (ch + 1e-6)
        if 0.12 < aspect_ratio < 8.0:
            x1, y1, x2, y2 = add_padding(x, y, cw, ch, padding, h, w)
            return img[y1:y2, x1:x2]

    print("  [room] Falling back to plain mode.")
    return crop_plain(img, padding if padding > 0 else 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared by room_scene / crop_room
# ──────────────────────────────────────────────────────────────────────────────

def _valid_art_crop(crop: np.ndarray, orig: np.ndarray,
                    min_frac: float = 0.04, max_frac: float = 0.85) -> bool:
    """True if *crop* is a plausible artwork region of *orig*."""
    h_o, w_o = orig.shape[:2]
    h_c, w_c = crop.shape[:2]
    if h_c == 0 or w_c == 0:
        return False
    frac   = (h_c * w_c) / (h_o * w_o)
    aspect = w_c / h_c          # width / height
    return (min_frac < frac < max_frac) and (0.12 < aspect < 8.0)


def _try_bgdiff_validated(img: np.ndarray, padding: float,
                           min_frac: float = 0.04,
                           max_frac: float = 0.80) -> np.ndarray | None:
    """
    Run crop_bgdiff and return the result only when it looks like a valid
    artwork crop (reasonable size, not a narrow window-like strip).
    Returns None when the result fails validation.
    """
    try:
        result = crop_bgdiff(img, padding)
    except Exception:
        return None
    if _valid_art_crop(result, img, min_frac, max_frac):
        return result
    return None


def _try_quad_validated(img: np.ndarray, padding: float) -> np.ndarray | None:
    """Run crop_room quad detection; return result only when valid."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts, score = _find_best_quad(gray, h, w)
    if pts is None or score <= 0:
        return None
    result = four_point_transform(img, pts)
    if padding > 0:
        ch2, cw2 = result.shape[:2]
        px, py = int(cw2 * padding / 100), int(ch2 * padding / 100)
        result = result[py: ch2 - py, px: cw2 - px]
    if _valid_art_crop(result, img):
        return result
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Mode 5: room_scene  (paintings hanging in a furnished room)
# ──────────────────────────────────────────────────────────────────────────────

def crop_room_scene(img: np.ndarray, padding: float = 0.0) -> np.ndarray:
    """
    One or two paintings hanging on a wall in a furnished room.

    Strategy:
    1. For neutral scenes (max_mid_sat ≤ 40): try bgdiff first — border pixels
       represent the room, the painting is the distinct region.
    2. Signal: HSV saturation for colourful paintings; Laplacian texture for
       neutral/grey paintings.  If neither gives usable signal, fall back to
       quad edge detection.
    3. Find painting rows via the first contiguous high-signal band, then
       extend UPWARD only with a lower threshold to recover smooth canvas
       edges (e.g. white canvas border with near-zero texture).
    4. Thin-strip guard: if the detected band is < 8% of image height, fall
       back to quad then bgdiff.
    5. Locate a diptych gap: minimum-signal column in the middle half of the
       painting column span.
    6. Validate gap: below 20% of the 75th-percentile painting signal, each
       half ≥ 15% of image width.
    7. Score each half: col_signal when one side clearly dominates (wall vs.
       painting); bottom brightness when both sides look like paintings (picks
       the less obstructed one).
    8. Final fallback: diffuse signal (>70% frac) → quad first, then bgdiff
       (with aspect-ratio guard to reject window-like strips).
    """
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = np.abs(cv2.Laplacian(blurred, cv2.CV_32F))

    mid_c = w // 3
    max_mid_sat = float(sat[:, mid_c:w - mid_c].mean(axis=0).max())

    # ── Neutral scene: try bgdiff before the signal approach.
    # Room walls / floors form the border → the painting is the "different" region.
    if max_mid_sat <= 40:
        result = _try_bgdiff_validated(img, padding)
        if result is not None:
            frac = (result.shape[0] * result.shape[1]) / (h * w)
            print(f"  [room_scene] bgdiff-first found painting, frac={frac:.0%}")
            return result

    if max_mid_sat > 40:
        signal = sat
        thr_signal = 30.0
    else:
        signal = lap
        thr_signal = 2.0

    # Smooth row signal with a 20-row moving average
    row_signal = signal.mean(axis=1)
    kernel_size = 20
    smoothed = np.convolve(row_signal, np.ones(kernel_size) / kernel_size, mode='same')

    # Find py1: first row above main threshold
    paint_start = -1
    for i in range(len(smoothed)):
        if smoothed[i] > thr_signal:
            paint_start = i
            break
    if paint_start == -1:
        print("  [room_scene] No painting rows — falling back to quad detection.")
        return crop_room(img, padding)

    # Find py2: stop at first long gap (≥1% image height) of below-threshold rows
    gap_limit = max(20, int(h * 0.01))
    gap_count = 0
    last_above = paint_start
    paint_end = len(smoothed) - 1
    for i in range(paint_start, len(smoothed)):
        if smoothed[i] > thr_signal:
            last_above = i
            gap_count = 0
        else:
            gap_count += 1
            if gap_count > gap_limit:
                paint_end = last_above
                break

    # Extend UPWARD only — recover smooth canvas edges that fall below the
    # main threshold (e.g. white top border of a grey/neutral painting).
    # Do NOT extend downward to avoid pulling in furniture below the paintings.
    thr_extend = thr_signal * 0.5
    small_gap  = max(5, int(h * 0.003))
    max_ext    = int(h * 0.20)

    new_py1 = paint_start
    gap_count = 0
    for i in range(paint_start - 1, max(0, paint_start - max_ext), -1):
        if smoothed[i] > thr_extend:
            new_py1 = i
            gap_count = 0
        else:
            gap_count += 1
            if gap_count > small_gap:
                break
    py1 = int(new_py1)
    py2 = int(paint_end) + 1   # exclusive slice end

    if py2 - py1 < max(50, int(h * 0.08)):
        print("  [room_scene] Painting band too thin — trying quad then bgdiff.")
        r = _try_quad_validated(img, padding)
        if r is not None:
            return r
        r = _try_bgdiff_validated(img, padding)
        if r is not None:
            return r
        return crop_room(img, padding)   # full quad+segmentation pipeline

    # Column signal over the painting row band
    col_signal = signal[py1:py2, :].mean(axis=0)

    # Find the span of all painting-signal columns
    all_cols = np.where(col_signal > thr_signal * 0.5)[0]
    if len(all_cols) == 0:
        return crop_room(img, padding)

    px_span_l = int(all_cols[0])
    px_span_r = int(all_cols[-1])
    span_len  = px_span_r - px_span_l + 1

    # Locate diptych gap within the middle half of the painting span —
    # prevents the wall beside an off-centre painting from appearing as a gap.
    s_l = px_span_l + span_len // 4
    s_r = px_span_l + 3 * span_len // 4
    if s_r > s_l:
        gap_col = int(s_l + np.argmin(col_signal[s_l:s_r]))
    else:
        gap_col = (px_span_l + px_span_r) // 2

    # Gap is real only if the minimum signal is well below the typical painting
    # level (75th percentile) in the span.
    p75 = float(np.percentile(col_signal[px_span_l:px_span_r + 1], 75))
    gap_is_real = col_signal[gap_col] < p75 * 0.2

    min_painting_w = int(w * 0.15)  # a real painting must be at least 15% wide

    if gap_is_real:
        left_cols  = np.where(col_signal[:gap_col] > thr_signal * 0.5)[0]
        right_cols = np.where(col_signal[gap_col:] > thr_signal * 0.5)[0]

        left_x1  = int(left_cols[0])  if len(left_cols) > 0 else 0
        left_x2  = int(left_cols[-1]) + 1 if len(left_cols) > 0 else gap_col
        right_x1 = int(gap_col + right_cols[0])  if len(right_cols) > 0 else gap_col
        right_x2 = int(gap_col + right_cols[-1]) + 1 if len(right_cols) > 0 else w

        left_real  = (left_x2  - left_x1)  >= min_painting_w
        right_real = (right_x2 - right_x1) >= min_painting_w

        if not left_real and not right_real:
            gap_is_real = False   # neither side is a real painting → single
        elif not left_real:
            # Only right side is a real painting — left was background/furniture
            gap_is_real = False
            px_span_l, px_span_r = right_x1, right_x2 - 1
        elif not right_real:
            gap_is_real = False
            px_span_l, px_span_r = left_x1, left_x2 - 1

    if not gap_is_real:
        # Single painting — crop to column span
        px1 = px_span_l
        px2 = px_span_r + 1
        frac_single = (px2 - px1) * (py2 - py1) / (h * w)
        print(f"  [room_scene] Single painting, rows {py1}-{py2}, cols {px1}-{px2}, frac={frac_single:.0%}")
        if frac_single > 0.70:
            print("  [room_scene] Signal too diffuse — trying quad then bgdiff.")
            r = _try_quad_validated(img, padding)
            if r is not None:
                return r
            r = _try_bgdiff_validated(img, padding)
            if r is not None:
                return r
            # last resort: full crop_room pipeline (includes region segmentation)
            return crop_room(img, padding)

        # Aspect-ratio guard: extremely wide/narrow strips are not paintings.
        # Try quad then bgdiff before accepting an implausible crop.
        _crop_w, _crop_h = px2 - px1, py2 - py1
        _asp = _crop_w / (_crop_h + 1e-6)
        if _asp > 3.0 or _asp < 0.22:
            print(f"  [room_scene] Crop aspect {_asp:.2f} implausible — trying quad then bgdiff.")
            r = _try_quad_validated(img, padding)
            if r is not None:
                return r
            r = _try_bgdiff_validated(img, padding)
            if r is not None:
                return r
            return crop_room(img, padding)

        if padding > 0:
            px1, py1, px2, py2 = add_padding(px1, py1, px2 - px1, py2 - py1, padding, h, w)
        return img[py1:py2, px1:px2]

    # Diptych: pick the painting using a two-stage score.
    # Stage 1 — col_signal (sat or tex): plain walls near zero, paintings high.
    #   If scores differ by >2×, one side is clearly a wall → pick higher signal.
    # Stage 2 — bottom brightness: when both sides look painting-like (similar signal),
    #   choose the unobstructed one (plant/obstruction darkens the bottom of a painting).
    left_sig  = float(col_signal[left_x1:left_x2].mean())  if left_x2 > left_x1  else 0.0
    right_sig = float(col_signal[right_x1:right_x2].mean()) if right_x2 > right_x1 else 0.0
    ratio = max(left_sig, right_sig) / max(min(left_sig, right_sig), 0.1)
    if ratio > 2.0:
        left_score, right_score, score_label = left_sig, right_sig, "sig"
    else:
        bottom_start = int(py2 - (py2 - py1) * 0.3)
        left_score  = float(img[bottom_start:py2, left_x1:left_x2].mean())  if (left_x2 > left_x1)  else 0.0
        right_score = float(img[bottom_start:py2, right_x1:right_x2].mean()) if (right_x2 > right_x1) else 0.0
        score_label = "brt"

    if right_score >= left_score:
        bx1, bx2, side = right_x1, right_x2, "right"
    else:
        bx1, bx2, side = left_x1, left_x2, "left"

    frac = (bx2 - bx1) * (py2 - py1) / (h * w)
    print(f"  [room_scene] {side} painting chosen [{score_label}] "
          f"(L={left_score:.1f} R={right_score:.1f}), "
          f"rows {py1}-{py2}, cols {bx1}-{bx2}, frac={frac:.0%}")

    if frac < 0.05 or bx2 <= bx1 or py2 <= py1:
        print("  [room_scene] Degenerate crop — falling back to quad detection.")
        return crop_room(img, padding)

    # Aspect-ratio guard on diptych result too
    _dp_asp = (bx2 - bx1) / (py2 - py1 + 1e-6)
    if _dp_asp > 3.0 or _dp_asp < 0.22:
        print(f"  [room_scene] Diptych result aspect {_dp_asp:.2f} implausible — trying quad then bgdiff.")
        r = _try_quad_validated(img, padding)
        if r is not None:
            return r
        r = _try_bgdiff_validated(img, padding)
        if r is not None:
            return r
        return crop_room(img, padding)

    if padding > 0:
        bx1, py1, bx2, py2 = add_padding(bx1, py1, bx2 - bx1, py2 - py1, padding, h, w)
    return img[py1:py2, bx1:bx2]


# ──────────────────────────────────────────────────────────────────────────────
# Auto-mode classifier
# ──────────────────────────────────────────────────────────────────────────────

def detect_mode(img: np.ndarray) -> str:
    """
    Classify image into round / plain / bgdiff / pinned / room.

    Decision tree:
    0. Circular/oval canvas (geometric extent test) → round
    1. Non-uniform border → room
    2. Uniform bright border:
       a. Background has colour (chroma > 5) → bgdiff
       b. Background clearly differs from image content → bgdiff
       c. Very low dark content, neutral bg → pinned
       d. Otherwise → plain
    3. Uniform dark border → plain
    """
    if detect_round_shape(img) is not None:
        return "round"

    h, w = img.shape[:2]
    b = max(12, int(min(h, w) * 0.05))

    strips = [img[:b, :], img[-b:, :], img[:, :b], img[:, -b:]]
    strip = np.vstack([s.reshape(-1, 3) for s in strips]).astype(float)
    border_std = np.mean(np.std(strip, axis=0))
    border_mean = strip.mean()

    corners = [img[:b, :b], img[:b, -b:], img[-b:, :b], img[-b:, -b:]]
    corner_means = np.array([c.reshape(-1, 3).mean(axis=0) for c in corners])
    corner_spread = np.std(corner_means, axis=0).mean()

    is_uniform = border_std < 22 and corner_spread < 30

    if not is_uniform:
        corner_brightness = np.array([c.reshape(-1, 3).mean() for c in corners])
        bg_check = sample_border(img, b)
        # All bright/white corners → full-bleed painting, not a room scene.
        if corner_brightness.min() > 220 and np.std(bg_check) < 8:
            is_uniform = True
        # At least 2 corners are grey/light (>130) AND tightly clustered → painting
        # on a grey wall mockup where some corners may be inside the painting area.
        elif np.sum(corner_brightness > 130) >= 2:
            light_corners = corner_brightness[corner_brightness > 130]
            if light_corners.std() < 30:
                # Grey wall mockups have uniformly bright corners (range ≈ 20-40).
                # Real room scenes have at least one dark corner (furniture/floor),
                # creating a wider brightness range.
                corner_range = corner_brightness.max() - corner_brightness.min()
                if corner_range > 50:
                    return "room_scene"
                # Even with tight corners, check if the bottom third has furniture
                # (high pixel variation), indicating a real room scene.
                _bt = img[2 * h // 3:, :].astype(float)
                _bt_std = float(np.std(_bt.mean(axis=2)))
                if _bt_std > 40:
                    return "room_scene"
                return "wall"
            else:
                return "room"
        else:
            return "room"

    bg = sample_border(img, b)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_pct = np.mean(gray < 180)

    # Check how "coloured" the background is (non-grey)
    bg_chroma = np.std(bg)      # 0 = perfect grey, higher = coloured
    bg_sat = (bg.max() - bg.min()) / (bg.max() + 1e-6)

    if border_mean > 140:
        # Room-scene check even when border looks uniform:
        # Use the OUTER 30px bottom strip (not the full bottom third), because a
        # painting on a plain wall has uniform wall at its very bottom edge (low std),
        # whereas a room photo has furniture/floor at the edge (high std).
        # Using the full bottom third would falsely trigger on paintings whose content
        # fills the lower part of the frame.
        bottom_strip = img[max(0, h - 30):, :].astype(float)
        top_third    = img[:h // 3, :].astype(float)
        bottom_strip_std  = float(np.std(bottom_strip.mean(axis=2)))
        top_mean          = float(top_third.mean())
        bottom_strip_mean = float(bottom_strip.mean())
        if bottom_strip_std > 20 or (top_mean > 200 and bottom_strip_mean < 175):
            return "room_scene"

        # Coloured background → bgdiff
        if bg_chroma > 5 or bg_sat > 0.05:
            return "bgdiff"

        # Neutral grey / off-white wall → wall mode (texture + corner-colour diff).
        # Covers rendered mockup backgrounds (~220-240) and real plain walls.
        # Pure white backgrounds (>248) fall through to bgdiff/plain below.
        if bg_chroma <= 5 and 160 < border_mean < 248:
            return "wall"

        # Neutral bg but clearly different from image content → bgdiff
        diff = np.abs(img.astype(float) - bg).mean(axis=2)
        if diff.max() > 25 and np.mean(diff > 12) > 0.04:
            return "bgdiff"

        # Very bright, mostly light, neutral bg → pinned (paper ≈ wall)
        if dark_pct < 0.20 and border_mean > 180:
            return "pinned"

    return "plain"


# ──────────────────────────────────────────────────────────────────────────────
# Core entry point
# ──────────────────────────────────────────────────────────────────────────────

_MODE_DEFAULTS = {"plain": 2.0, "bgdiff": 2.0, "pinned": 18.0, "room": 0.0, "wall": 0.0, "room_scene": 0.0, "round": 3.0}


def crop_artwork(input_path: str,
                 output_path: str | None = None,
                 mode: str = "auto",
                 padding: float | None = None,
                 cutout: bool = False) -> str:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    h, w = img.shape[:2]

    if mode == "auto":
        mode = detect_mode(img)
        print(f"  [auto] → {mode}")

    eff_pad = padding if padding is not None else _MODE_DEFAULTS.get(mode, 2.0)

    if cutout and mode != "round":
        print(f"  [warn] --cutout only applies to round mode (this image is '{mode}') — ignoring.")

    if mode == "round":
        result = crop_round(img, eff_pad, cutout=cutout)
    elif mode == "wall":
        result = crop_wall(img, eff_pad)
    elif mode == "plain":
        result = crop_plain(img, eff_pad)
    elif mode == "bgdiff":
        result = crop_bgdiff(img, eff_pad)
    elif mode == "pinned":
        result = crop_pinned(img, eff_pad)
    elif mode == "room":
        result = crop_room(img, eff_pad)
    elif mode == "room_scene":
        result = crop_room_scene(img, eff_pad)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    rh, rw = result.shape[:2]
    if rh < 50 or rw < 50 or rh * rw < h * w * 0.04:
        print("  [warn] Crop too small — returning original.")
        result = img

    has_alpha = result.ndim == 3 and result.shape[2] == 4

    if output_path is None:
        p = Path(input_path)
        suffix = ".png" if has_alpha else p.suffix
        output_path = str(p.parent / f"{p.stem}_cropped{suffix}")
    elif has_alpha and not output_path.lower().endswith(".png"):
        # JPEG has no alpha channel — force PNG so the cutout isn't silently flattened.
        output_path = str(Path(output_path).with_suffix(".png"))
        print(f"  [warn] Cutout has transparency — forcing .png output: {output_path}")

    cv2.imwrite(output_path, result)
    print(f"  ✓ {output_path.split('/')[-1]}  ({result.shape[1]}×{result.shape[0]}px)")
    return output_path


def batch_crop(input_dir: str, output_dir: str | None = None,
               mode: str = "auto", padding: float | None = None,
               cutout: bool = False):
    in_path = Path(input_dir)
    if not in_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    out_path = Path(output_dir) if output_dir else in_path / "cropped"
    out_path.mkdir(parents=True, exist_ok=True)
    images = sorted(f for f in in_path.rglob("*") if f.suffix.lower() in SUPPORTED)
    if not images:
        print("No supported images found.")
        return
    print(f"Processing {len(images)} images → {out_path}\n")
    for i, img_file in enumerate(images, 1):
        # Preserve subfolder structure in output
        rel = img_file.relative_to(in_path)
        dest = out_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(images)}] {rel}")
        try:
            crop_artwork(str(img_file), str(dest),
                         mode=mode, padding=padding, cutout=cutout)
        except Exception as e:
            print(f"  ✗ {e}")
    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crop artwork from product images — zero cost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  plain    Canvas/artwork on a neutral solid background.
  bgdiff   Artwork on a distinctly coloured or subtly different background.
           Also handles matted prints, canvas on slightly different-toned wall.
  pinned   Sparse ink/watercolour on paper pinned to a matching-colour wall.
           Detects content, pads out by --padding %% (default 18).
  room     Framed artwork in a room, on wood wall, leaning on pedestal.
  round    Circular or oval-shaped canvas (auto-detected geometrically).
           Add --cutout for a transparent die-cut PNG instead of a rectangle.
  auto     Auto-detect (default).

Examples:
  python artwork_cropper.py painting.jpg
  python artwork_cropper.py drawing.jpg --mode pinned --padding 20
  python artwork_cropper.py print.jpg --mode bgdiff
  python artwork_cropper.py oval_canvas.jpg --mode round
  python artwork_cropper.py oval_canvas.jpg --mode round --cutout
  python artwork_cropper.py ./images/ --batch
  python artwork_cropper.py ./images/ --batch --output ./out/ --mode bgdiff
        """,
    )
    parser.add_argument("input")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--mode", "-m",
                        choices=["auto", "wall", "plain", "bgdiff", "pinned", "room", "room_scene", "round"],
                        default="auto")
    parser.add_argument("--padding", "-p", type=float, default=None,
                        help="Override padding %%. Defaults: plain=2, bgdiff=2, pinned=18, room=0, round=3")
    parser.add_argument("--batch", "-b", action="store_true")
    parser.add_argument("--cutout", action="store_true",
                        help="Round mode only: export a transparent die-cut PNG "
                             "(alpha matches the fitted ellipse) instead of a rectangle.")
    args = parser.parse_args()

    if args.batch:
        batch_crop(args.input, args.output, mode=args.mode, padding=args.padding, cutout=args.cutout)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: '{args.input}' is not a file. Use --batch for directories.")
            sys.exit(1)
        crop_artwork(args.input, args.output, mode=args.mode, padding=args.padding, cutout=args.cutout)


if __name__ == "__main__":
    main()
