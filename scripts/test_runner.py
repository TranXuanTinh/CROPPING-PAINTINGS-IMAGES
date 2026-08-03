#!/usr/bin/env python3
"""
test_runner.py
───────────────
Batch testing with side-by-side comparison image generation.

For each input image, generates a comparison image showing:
  Left:   Original with YOLO bounding box overlay
  Right:  Cropped result
  Header: filename, confidence, method, sizes, timing

Also generates:
  - summary.csv with per-image metrics
  - accuracy_report.md with per-category statistics

Usage:
  python test_runner.py
  python test_runner.py --input "input images/" --output yolo_test_results/
  python test_runner.py --model models/best.pt  # Phase 2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure this script can import siblings
sys.path.insert(0, str(Path(__file__).parent))

from yolo_config import (
    SUPPORTED_EXTENSIONS,
    JPEG_QUALITY,
    COMPARE_PANEL_H,
    COMPARE_HEADER_H,
    COMPARE_GAP,
    COMPARE_MAX_PANEL_W,
)
from yolo_artwork_cropper import load_model, detect_artwork, crop_artwork


# ──────────────────────────────────────────────────────────────────────────────
# Visual comparison helpers
# ──────────────────────────────────────────────────────────────────────────────

COLOUR_YOLO = (70, 220, 70)          # green bbox for YOLO detection
COLOUR_FALLBACK = (70, 70, 220)      # red bbox for fallback detection
COLOUR_NONE = (70, 70, 220)          # red bbox for no detection
COLOUR_BG = (28, 28, 28)             # dark background
COLOUR_PANEL_BG = (210, 210, 210)    # panel padding colour
COLOUR_TEXT = (230, 230, 230)        # header text
COLOUR_DIM = (150, 150, 150)         # label text
LABEL_H = 26


def _load_font(size: int):
    """Load a system font, falling back to default if none found."""
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


def _fit_panel(
    img: np.ndarray,
    panel_h: int,
    max_w: int,
) -> tuple[np.ndarray, float]:
    """Scale an image to fit within panel_h × max_w, preserving aspect."""
    h, w = img.shape[:2]
    scale = panel_h / h
    new_w = min(int(w * scale), max_w)
    scale_w = new_w / w
    final_scale = min(scale, scale_w) if new_w == max_w else scale
    out_w = int(w * final_scale)
    out_h = int(h * final_scale)
    resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized, final_scale


def _pad_panel(panel: np.ndarray, target_h: int) -> np.ndarray:
    """Vertically centre-pad a panel to target_h."""
    ph, pw = panel.shape[:2]
    canvas = np.full((target_h, pw, 3), COLOUR_PANEL_BG, np.uint8)
    y0 = (target_h - ph) // 2
    canvas[y0: y0 + ph, :] = panel
    return canvas


def build_comparison(
    img_path: Path,
    out_path: Path,
    model: object,
    custom_model: str | None = None,
    mode: str = "auto",
) -> dict:
    """
    Process one image and generate a side-by-side comparison.

    Returns a dict with metadata (file, method, confidence, sizes, timing).
    """
    t0 = time.time()

    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": img_path.name, "error": "unreadable"}
    h, w = img.shape[:2]

    # Run detection
    bbox, conf, method = detect_artwork(str(img_path), model)

    # Annotate original
    annotated = img.copy()
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        colour = COLOUR_YOLO if method == "yolo" else COLOUR_FALLBACK
        thickness = max(2, w // 300)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)
        # Draw confidence text
        label = f"{method} {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )
    else:
        # Red border for no detection
        cv2.rectangle(
            annotated, (0, 0), (w - 1, h - 1), COLOUR_NONE, max(3, w // 200)
        )

    # Run crop
    crop_method = method
    try:
        crop_tmp = out_path.parent / f"_tmp_{img_path.stem}_crop{img_path.suffix}"
        crop_artwork(
            str(img_path),
            str(crop_tmp),
            model=model,
            custom_model=custom_model,
            mode=mode,
        )
        cropped = cv2.imread(str(crop_tmp))
        crop_tmp.unlink(missing_ok=True)
        if method == "none" and cropped is not None:
            # If YOLO found nothing but crop succeeded, it must have used the fallback
            crop_method = "opencv_fallback"
    except Exception as e:
        cropped = np.full((h, w, 3), 200, np.uint8)
        cv2.putText(
            cropped,
            f"Error: {e}",
            (10, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    ch, cw = cropped.shape[:2]
    elapsed = time.time() - t0

    # Build side-by-side comparison
    left, _ = _fit_panel(annotated, COMPARE_PANEL_H, COMPARE_MAX_PANEL_W)
    right, _ = _fit_panel(cropped, COMPARE_PANEL_H, COMPARE_MAX_PANEL_W)

    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    panel_h = max(lh, rh)

    left = _pad_panel(left, panel_h)
    right = _pad_panel(right, panel_h)

    total_w = lw + COMPARE_GAP + rw
    total_h = COMPARE_HEADER_H + panel_h + LABEL_H
    canvas = np.full((total_h, total_w, 3), COLOUR_BG, np.uint8)
    canvas[COMPARE_HEADER_H: COMPARE_HEADER_H + panel_h, 0:lw] = left
    canvas[
        COMPARE_HEADER_H: COMPARE_HEADER_H + panel_h,
        lw + COMPARE_GAP: lw + COMPARE_GAP + rw,
    ] = right

    # Draw header and labels
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Get category from parent directory name
    category = img_path.parent.name

    header = (
        f"{category}   |   {img_path.name}   |   "
        f"{crop_method.upper()}: {'conf=' + f'{conf:.2f}' if conf > 0 else 'N/A'}   |   "
        f"{w}×{h} → {cw}×{ch}   |   {elapsed:.1f}s"
    )
    draw.text((10, 14), header, font=FONT_HEADER, fill=COLOUR_TEXT)
    draw.text(
        (10, COMPARE_HEADER_H + panel_h + 5),
        "ORIGINAL",
        font=FONT_LABEL,
        fill=COLOUR_DIM,
    )
    draw.text(
        (lw + COMPARE_GAP + 10, COMPARE_HEADER_H + panel_h + 5),
        "CROPPED RESULT",
        font=FONT_LABEL,
        fill=COLOUR_DIM,
    )

    # Save comparison
    comparison_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), comparison_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    return {
        "file": img_path.name,
        "category": category,
        "method": crop_method,
        "confidence": f"{conf:.2f}",
        "orig_size": f"{w}×{h}",
        "crop_size": f"{cw}×{ch}",
        "time_s": f"{elapsed:.2f}",
    }



# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_report(
    results: list[dict],
    output_dir: Path,
) -> None:
    """Generate summary.csv and accuracy_report.md."""

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "file",
                "method",
                "confidence",
                "orig_size",
                "crop_size",
                "time_s",
            ],
        )
        writer.writeheader()
        for r in results:
            if "error" not in r:
                writer.writerow(r)
    print(f"\n📊 Summary: {csv_path}")

    # ── Markdown report ───────────────────────────────────────────────────
    report_path = output_dir / "accuracy_report.md"

    # Group by category
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    yolo_count = sum(1 for r in results if r.get("method") == "yolo")
    fallback_count = sum(
        1 for r in results if r.get("method") == "opencv_fallback"
    )
    none_count = sum(1 for r in results if r.get("method") == "none")

    lines = [
        "# YOLO Artwork Cropper — Test Results\n",
        f"**Total images:** {total}  |  "
        f"**YOLO detections:** {yolo_count}  |  "
        f"**OpenCV fallback:** {fallback_count}  |  "
        f"**No detection:** {none_count}  |  "
        f"**Errors:** {errors}\n",
        f"**YOLO detection rate:** {yolo_count / max(1, total):.1%}\n",
        "",
        "## Per-Category Breakdown\n",
        "| Category | Total | YOLO | Fallback | None | Detection Rate |",
        "|---|---|---|---|---|---|",
    ]

    for cat in sorted(categories.keys()):
        items = categories[cat]
        n = len(items)
        y = sum(1 for r in items if r.get("method") == "yolo")
        fb = sum(1 for r in items if r.get("method") == "opencv_fallback")
        no = sum(1 for r in items if r.get("method") == "none")
        rate = (y + fb) / max(1, n)
        lines.append(
            f"| {cat} | {n} | {y} | {fb} | {no} | {rate:.0%} |"
        )

    lines.extend([
        "",
        "## Timing\n",
    ])

    times = [
        float(r["time_s"])
        for r in results
        if "time_s" in r and "error" not in r
    ]
    if times:
        lines.extend([
            f"- **Mean:** {np.mean(times):.1f}s/image",
            f"- **Median:** {np.median(times):.1f}s/image",
            f"- **Min:** {min(times):.1f}s  |  **Max:** {max(times):.1f}s",
        ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"📋 Report: {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the YOLO artwork cropper on all images and generate "
            "side-by-side comparison results."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default="input images/",
        help="Input directory containing images (default: 'input images/').",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="yolo_test_results/",
        help="Output directory for comparison images (default: 'yolo_test_results/').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to custom-trained YOLO weights (Phase 2).",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "rect", "oval"],
        help="Force shape mode (default: auto).",
    )

    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = sorted(
        f
        for f in in_dir.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"No supported images found in {in_dir}")
        return

    # Load model once
    model = load_model(args.model)

    print(f"Processing {len(images)} images → {out_dir}\n")

    results = []
    for i, img_file in enumerate(images, 1):
        rel = img_file.relative_to(in_dir)
        out_file = out_dir / rel.parent / f"{img_file.stem}_compare.jpg"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(images)}] {rel}")
        r = build_comparison(
            img_file, out_file, model, custom_model=args.model, mode=args.mode
        )
        results.append(r)

        status = r.get("method", "error")
        if "error" in r:
            status = f"ERROR: {r['error']}"
        print(f"  → {status}")

    # Generate reports
    _generate_report(results, out_dir)


if __name__ == "__main__":
    main()
