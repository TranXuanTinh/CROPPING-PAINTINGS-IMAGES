# CLI Usage Guide

## Quick Start

```bash
# Crop a single painting
python scripts/yolo_artwork_cropper.py photo.jpg

# Crop all paintings in a directory
python scripts/yolo_artwork_cropper.py ./photos/ --batch
```

## Command Reference

```
python yolo_artwork_cropper.py <input> [options]

Positional Arguments:
  input                   Image file path, or directory path with --batch

Options:
  --output, -o PATH       Output file (single) or directory (batch)
                          Default: <stem>_cropped<ext> or <input>/cropped/

  --batch, -b             Process every image in the directory recursively

  --mode, -m MODE         Force shape detection mode:
                            auto  — automatic detection (default)
                            rect  — treat all paintings as rectangular
                            oval  — treat all paintings as oval/circular

  --model PATH            Path to custom-trained YOLO weights (Phase 2)
                          Default: YOLO-World zero-shot model

  --frame-trim            Trim physical picture frame borders from the crop
  --no-frame-trim         Disable frame trimming (default)
  --frame-depth FRAC      Max inward search depth for frame trim
                          Default: 0.12 (12% of image dimension)

  --check                 Verify installation and exit
```

## Examples

### Single Image

```bash
# Basic crop (output: photo_cropped.jpg alongside original)
python scripts/yolo_artwork_cropper.py photo.jpg

# Specify output path
python scripts/yolo_artwork_cropper.py photo.jpg --output cropped/result.jpg

# Force oval mode (for circular paintings)
python scripts/yolo_artwork_cropper.py circle_painting.jpg --mode oval

# Remove physical frame from crop
python scripts/yolo_artwork_cropper.py framed_painting.jpg --frame-trim
```

### Batch Processing

```bash
# Process all images recursively, preserve folder structure
python scripts/yolo_artwork_cropper.py "./input images/" --batch

# Specify output directory
python scripts/yolo_artwork_cropper.py "./input images/" --batch --output ./cropped/

# Batch with frame trimming
python scripts/yolo_artwork_cropper.py "./input images/" --batch --frame-trim
```

### Using Custom-Trained Model (Phase 2)

```bash
# After training with train_yolo.py:
python scripts/yolo_artwork_cropper.py photo.jpg --model models/best.pt
python scripts/yolo_artwork_cropper.py ./photos/ --batch --model models/best.pt
```

### Test Runner (Comparison Images)

```bash
# Generate side-by-side comparisons for all categories
python scripts/test_runner.py \
    --input "input images/" \
    --output "yolo_test_results/"

# Test only oval images
python scripts/test_runner.py \
    --input "input images/oval_and_circles/" \
    --output "yolo_test_results/ovals/" \
    --mode oval
```

## Output Format

- **Rectangular paintings**: JPEG crop of the painting region
- **Oval paintings**: JPEG tight bounding box around the oval
- **Comparison images**: Side-by-side JPEG (original + bbox on left, crop on right)

## Pipeline Stages (Logged to Console)

```
[1/456] rectangular_and_squares/1. Canvas or Paperworks on a wall/painting.jpg
  [yolo] conf=0.85  bbox=[120, 80, 430, 540]     # YOLO detection
  [shape] rectangular                               # Shape classification
  [clahe] refined → [118, 78, 432, 542]            # Edge refinement
  [tighten] → [122, 82, 428, 538]                  # Background tightening
  ✓ painting_cropped.jpg  (306×456px)  [yolo, 3.2s] # Final result
```

## Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.webp`, `.tiff`, `.bmp`
