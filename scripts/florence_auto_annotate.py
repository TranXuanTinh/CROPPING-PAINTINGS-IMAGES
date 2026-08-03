#!/usr/bin/env python3
"""
florence_auto_annotate.py
─────────────────────────
Phase 2 auto-annotation script using Florence-2.
Automatically processes all images in "input images/", runs Florence-2 detection,
and writes YOLO-format label files (.txt) to build a training dataset.

This eliminates the need for manual annotation of 465+ images.
Once this runs, the user has a fully labeled dataset ready to fine-tune YOLOv8.
"""

import os
import sys
import random
import shutil
import time
from pathlib import Path
from PIL import Image
import torch

# Ensure we can load Florence-2 from the v3 script configuration
sys.path.insert(0, str(Path(__file__).parent))
from artwork_cropper_v3 import load_model as load_florence_model, detect_artwork as detect_florence_artwork

# Supported formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Automatically generate YOLO annotations using Florence-2."
    )
    parser.add_argument(
        "--input", "-i",
        default="/media/tinhtran/01D85BC599D1D460/FreeLancer/ComputerVision/input images",
        help="Input root directory containing images."
    )
    parser.add_argument(
        "--output", "-o",
        default="/media/tinhtran/01D85BC599D1D460/FreeLancer/ComputerVision/dataset",
        help="Output directory for the generated YOLO dataset."
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.85,
        help="Train/val split ratio (default: 0.85 = 85% train)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split."
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    # Find all images recursively
    images = sorted(
        f for f in in_dir.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    
    if not images:
        print(f"No supported images found in {in_dir}")
        return

    print(f"Found {len(images)} images total.")

    # Load Florence-2 model
    print("Loading Florence-2 model on CPU (this may take a moment)...")
    model, processor, device = load_florence_model("cpu")

    # Shuffle and split train/val
    random.seed(args.seed)
    shuffled = list(images)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * args.split)
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]

    print(f"Split: {len(train_images)} train, {len(val_images)} val")

    # Process and copy files
    for split_name, split_images in [("train", train_images), ("val", val_images)]:
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_path in enumerate(split_images, 1):
            print(f"[{split_name}] [{idx}/{len(split_images)}] {img_path.name}")
            
            # Destination image path
            dst_img_path = img_dir / img_path.name
            shutil.copy2(str(img_path), str(dst_img_path))

            # Run Florence-2 detection
            try:
                pil_img = Image.open(img_path).convert("RGB")
                w, h = pil_img.size
                bbox, confidence = detect_florence_artwork(pil_img, model, processor, device)
                
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if bbox is not None:
                    # bbox format: [x1, y1, x2, y2] in pixels
                    x1, y1, x2, y2 = bbox
                    
                    # Convert to normalized YOLO format: class cx cy bw bh
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    # Clip coordinates to [0.0, 1.0]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))

                    with open(lbl_path, "w") as f:
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    print(f"  → Saved bbox: cx={cx:.3f}, cy={cy:.3f}, w={bw:.3f}, h={bh:.3f} (conf={confidence})")
                else:
                    # Write empty label file (background class)
                    with open(lbl_path, "w") as f:
                        pass
                    print("  → No bbox found (saved empty background label)")
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")

    # Generate data.yaml
    yaml_content = f"""# YOLOv8 custom dataset for painting detection
path: {out_dir.resolve()}
train: train/images
val: val/images

# Classes
names:
  0: painting
"""
    with open(out_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("\n✓ Auto-annotation complete!")
    print(f"Dataset generated at: {out_dir}")
    print(f"data.yaml generated at: {out_dir / 'data.yaml'}")

if __name__ == "__main__":
    main()
