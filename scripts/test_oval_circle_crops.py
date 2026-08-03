#!/usr/bin/env python3
"""
test_oval_circle_crops.py
──────────────────────────
Runs artwork_cropper.py's round/oval detection over every image in
oval-circle-shapes/ and produces annotated side-by-side comparison images
for manual review — same format used for the earlier rectangular-artwork
test passes.

Comparison image layout (per image):
  ┌──────────────────────────────────────────────────────────┐
  │ HEADER: filename | mode | extent/frac | orig→crop size   │
  ├─────────────────────────┬────────────────────────────────┤
  │  ORIGINAL               │  CROPPED RESULT                │
  │  + fitted ellipse (cyan)│                                │
  │  + bounding box (yellow)│                                │
  ├─────────────────────────┴────────────────────────────────┤
  │ "DETECTION"  label            "CROP RESULT"  label        │
  └──────────────────────────────────────────────────────────┘

Box colours:
  Green bbox   — round shape detected, ellipse fit succeeded
  Red bbox     — no round shape found, fell back to another mode

Usage:
  python test_oval_circle_crops.py
  python test_oval_circle_crops.py --input oval-circle-shapes --output oval_circle_test_results
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from artwork_cropper import (
    SUPPORTED,
    detect_mode,
    detect_round_shape,
    ellipse_aabb,
    refine_round_ellipse,
    crop_artwork,
    _MODE_DEFAULTS,
)

PANEL_H = 560
HEADER_H = 46
LABEL_H = 26
GAP = 6
MAX_PANEL_W = 700
JPEG_Q = 90

COLOUR_DETECTED = (70, 200, 90)     # green  — round shape found
COLOUR_FALLBACK = (220, 70, 70)     # red    — no round shape, fallback mode
COLOUR_ELLIPSE = (255, 210, 40)     # cyan/yellow ellipse outline (BGR: yellow-cyan)
COLOUR_BBOX = (60, 220, 255)        # amber bbox (BGR)
COLOUR_PANEL_BG = (210, 210, 210)
COLOUR_BG = (28, 28, 28)
COLOUR_TEXT = (230, 230, 230)
COLOUR_DIM = (150, 150, 150)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


FONT_HEADER = _load_font(15)
FONT_LABEL = _load_font(14)


def _fit_panel(img: np.ndarray, panel_h: int, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = panel_h / h
    new_w = min(int(w * scale), max_w)
    scale_w = new_w / w
    final_scale = min(scale, scale_w) if new_w == max_w else scale
    out_w, out_h = int(w * final_scale), int(h * final_scale)
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA), final_scale


def build_comparison(img_path: Path, out_path: Path) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": img_path.name, "error": "unreadable"}
    h, w = img.shape[:2]

    mode = detect_mode(img)
    cand = detect_round_shape(img)

    annotated = img.copy()
    detected = cand is not None
    box_colour = COLOUR_DETECTED if detected else COLOUR_FALLBACK
    stats = ""

    if cand is not None:
        coarse_ellipse = cv2.fitEllipse(cand["contour"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ellipse = refine_round_ellipse(gray, coarse_ellipse)
        cv2.ellipse(annotated, ellipse, COLOUR_ELLIPSE, max(2, w // 300))
        bbox = ellipse_aabb(ellipse, h, w)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOUR_BBOX, max(2, w // 300))
        stats = f"extent={cand['extent']:.3f}  frac={cand['frac']:.0%}  t={cand['t']}"

    # Border so the detection panel reads clearly against the label bar.
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), box_colour, max(3, w // 200))

    try:
        crop_path = out_path.parent / f"_tmp_{img_path.stem}_crop{img_path.suffix}"
        crop_artwork(str(img_path), str(crop_path), mode="auto")
        cropped = cv2.imread(str(crop_path))
        crop_path.unlink(missing_ok=True)
    except Exception as e:
        cropped = np.full((h, w, 3), 255, np.uint8)
        stats += f"  [crop error: {e}]"

    ch, cw = cropped.shape[:2]

    left, _ = _fit_panel(annotated, PANEL_H, MAX_PANEL_W)
    right, _ = _fit_panel(cropped, PANEL_H, MAX_PANEL_W)

    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    panel_h = max(lh, rh)

    def pad(panel, target_h):
        ph, pw = panel.shape[:2]
        canvas = np.full((target_h, pw, 3), COLOUR_PANEL_BG, np.uint8)
        y0 = (target_h - ph) // 2
        canvas[y0:y0 + ph, :] = panel
        return canvas

    left = pad(left, panel_h)
    right = pad(right, panel_h)

    total_w = lw + GAP + rw
    total_h = HEADER_H + panel_h + LABEL_H
    canvas = np.full((total_h, total_w, 3), COLOUR_BG, np.uint8)
    canvas[HEADER_H:HEADER_H + panel_h, 0:lw] = left
    canvas[HEADER_H:HEADER_H + panel_h, lw + GAP:lw + GAP + rw] = right

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    header = f"{img_path.name}   mode={mode}   {stats}   {w}x{h} -> {cw}x{ch}"
    draw.text((10, 14), header, font=FONT_HEADER, fill=COLOUR_TEXT)
    draw.text((10, HEADER_H + panel_h + 5), "DETECTION", font=FONT_LABEL, fill=COLOUR_DIM)
    draw.text((lw + GAP + 10, HEADER_H + panel_h + 5), "CROP RESULT", font=FONT_LABEL, fill=COLOUR_DIM)

    pil.save(out_path, quality=JPEG_Q)

    return {
        "file": img_path.name, "mode": mode, "detected": detected,
        "orig_size": f"{w}x{h}", "crop_size": f"{cw}x{ch}", "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="oval-circle-shapes")
    parser.add_argument("--output", default="oval_circle_test_results")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in SUPPORTED)
    print(f"Processing {len(images)} images → {out_dir}\n")

    results = []
    for i, f in enumerate(images, 1):
        out_path = out_dir / f"{f.stem}_compare.jpg"
        print(f"[{i}/{len(images)}] {f.name}")
        r = build_comparison(f, out_path)
        results.append(r)
        status = "round detected" if r.get("detected") else "FALLBACK (no round shape)"
        print(f"  {status} → {out_path.name}")

    print("\nSummary:")
    for r in results:
        if "error" in r:
            print(f"  {r['file']:45s} ERROR: {r['error']}")
        else:
            print(f"  {r['file']:45s} mode={r['mode']:6s} {r['orig_size']:>10s} → {r['crop_size']:>10s}")


if __name__ == "__main__":
    main()
