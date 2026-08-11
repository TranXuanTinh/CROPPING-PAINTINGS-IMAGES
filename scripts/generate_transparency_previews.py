#!/usr/bin/env python3
"""
generate_transparency_previews.py
──────────────────────────────────
Generates transparent PNG die-cut crops for circular/oval images and creates
a `transparency_previews` folder containing `*_on_checkerboard.jpg` images
for visual verification of the alpha channel.

CLI Usage:
  python generate_transparency_previews.py
  python generate_transparency_previews.py \
    --input "../input_images/oval_and_circles/oval-circle-shapes" \
    --output "../test_results_new/oval_circle_cutout_results"
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from artwork_cropper import (
    SUPPORTED,
    crop_artwork,
)


def generate_checkerboard(h: int, w: int, square_size: int = 20) -> np.ndarray:
    """
    Generate an alternating light-gray and white checkerboard background.
    """
    c_white = np.array([255, 255, 255], dtype=np.uint8)
    c_gray = np.array([210, 210, 210], dtype=np.uint8)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            use_white = ((x // square_size) + (y // square_size)) % 2 == 0
            bg[y:y + square_size, x:x + square_size] = c_white if use_white else c_gray
    return bg


def process_transparency_previews(input_dir: str, output_dir: str) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    preview_path = out_path / "transparency_previews"

    out_path.mkdir(parents=True, exist_ok=True)
    preview_path.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in in_path.iterdir() if f.suffix.lower() in SUPPORTED and not f.name.endswith("_cropped.jpeg"))
    print(f"Processing {len(images)} images for transparent PNGs & checkerboard previews...\n")
    print(f"  Input dir:   {in_path}")
    print(f"  Output dir:  {out_path}")
    print(f"  Preview dir: {preview_path}\n")

    for i, img_file in enumerate(images, 1):
        png_out = out_path / f"{img_file.stem}.png"
        preview_out = preview_path / f"{img_file.stem}_on_checkerboard.jpg"

        print(f"[{i}/{len(images)}] {img_file.name}")

        # 1. Crop artwork with cutout=True to get transparent BGRA PNG
        crop_artwork(str(img_file), str(png_out), mode="round", cutout=True)

        # 2. Read generated PNG and composite onto checkerboard
        img_bgra = cv2.imread(str(png_out), cv2.IMREAD_UNCHANGED)
        if img_bgra is not None and img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
            h, w = img_bgra.shape[:2]
            bgr = img_bgra[:, :, :3]
            alpha = (img_bgra[:, :, 3] / 255.0)[:, :, np.newaxis]
            bg = generate_checkerboard(h, w)
            blended = (bgr * alpha + bg * (1.0 - alpha)).astype(np.uint8)

            cv2.imwrite(str(preview_out), blended, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  ✓ PNG cutout: {png_out.name} ({w}×{h}px)")
            print(f"  ✓ Checkerboard preview: {preview_out.name}")
        else:
            print(f"  ✗ Failed to generate alpha preview for {img_file.name}")

    print("\nAll transparent PNG cutouts and previews generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate transparent PNG cutouts and checkerboard previews for round/oval images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i",
                        default="../input_images/oval_and_circles/oval-circle-shapes",
                        help="Input directory containing oval/circle images.")
    parser.add_argument("--output", "-o",
                        default="../test_results_new/oval_circle_cutout_results",
                        help="Output directory for transparent PNGs & transparency_previews folder.")

    args = parser.parse_args()
    process_transparency_previews(args.input, args.output)


if __name__ == "__main__":
    main()
