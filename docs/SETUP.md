# Installation & Setup Guide

## Prerequisites

- **Python**: 3.9 or higher
- **OS**: Linux, macOS, or Windows
- **Hardware**: CPU only (no GPU required)
- **Disk**: ~200 MB for dependencies + model weights
- **RAM**: 4 GB minimum, 8 GB recommended

## Step 1: Activate Miniconda base environment

Make sure your Miniconda base environment is activated:

```bash
cd /path/to/ComputerVision

# Activate conda base
conda activate base
```

## Step 2: Install Dependencies

Use pip inside your active conda base environment to install the required packages:

```bash
pip install -r scripts/requirements.txt
```

This installs:
- `ultralytics` (includes PyTorch, torchvision) — YOLO detection
- `opencv-python` — image processing
- `numpy` — array operations
- `Pillow` — image loading

**Total download:** ~150 MB (vs ~1.5 GB for the Florence-2 stack)

## Step 3: Verify Installation

```bash
python scripts/yolo_artwork_cropper.py --check
```

Expected output:
```
Checking installation...

  ✓ opencv-python         4.x.x
  ✓ numpy                 1.x.x
  ✓ Pillow                10.x.x
  ✓ ultralytics           8.x.x

  ✓ YOLOWorld import OK
  ℹ Model 'yolov8s-worldv2.pt' will download automatically on first run (~47 MB)

All checks passed. Ready to crop paintings! 🎨
```

## Step 4: First Run (Model Download)

On the first run, the YOLO-World model weights (~47 MB) are automatically
downloaded from Ultralytics and cached at `~/.cache/ultralytics/`.
Subsequent runs load from cache with no network required.

```bash
# Test with a single image
python scripts/yolo_artwork_cropper.py \
    "input images/oval_and_circles/oval-circle-shapes/m5y8oc0nqfzs6lquduss.jpeg"
```

## Step 5: Batch Processing

```bash
# Process all images (recursive), save to output directory
python scripts/yolo_artwork_cropper.py \
    "input images/" \
    --batch \
    --output "yolo_output/"
```

## Step 6: Run Tests with Comparisons

```bash
# Generate side-by-side comparison images for review
python scripts/test_runner.py \
    --input "input images/" \
    --output "yolo_test_results/"
```

This creates:
- `yolo_test_results/` — comparison images (left: original + bbox, right: crop)
- `yolo_test_results/summary.csv` — per-image metrics
- `yolo_test_results/accuracy_report.md` — per-category statistics

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install ultralytics>=8.3.0
```

### "Model download fails"
The model is downloaded from `https://github.com/ultralytics/assets/releases/`.
If behind a firewall, download manually and place at `~/.cache/ultralytics/`.

### "Slow inference (>10 s/image)"
- Check CPU load: `top` or `htop`
- Try reducing image size: add `--imgsz 640` to the command
- Export to ONNX for faster CPU inference (see TRAINING.md)

### "Out of memory"
- Reduce `YOLO_IMGSZ` in `yolo_config.py` from 1280 to 640
- Close other memory-intensive applications
