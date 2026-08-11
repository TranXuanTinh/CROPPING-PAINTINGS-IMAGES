# Artwork Cropper — Image Processing & Cropping Pipeline

Production artwork cropping pipeline for View at Home (VAH). Automatically isolates artwork (paintings, paper works, prints, circular/oval canvases) from complex backgrounds — removing background walls, room environments, studio setups, frames, and mats — leaving only the artwork surface.

---

## Table of Contents

- [Requirements & Installation](#requirements--installation)
- [Quick Start](#quick-start)
- [Testing Individual Files](#testing-individual-files)
  - [Using Primary AI Pipeline (`artwork_cropper_v3.py`)](#using-primary-ai-pipeline-artwork_cropper_v3py)
  - [Using Fast Rule-Based Pipeline (`artwork_cropper.py`)](#using-fast-rule-based-pipeline-artwork_cropperpy)
  - [Individual File Examples & Advanced Flags](#individual-file-examples--advanced-flags)
- [Running Full Tests on All Inputs](#running-full-tests-on-all-inputs)
  - [1. Full Batch Crop Generation (AI Pipeline)](#1-full-batch-crop-generation-ai-pipeline)
  - [2. Full Batch Crop Generation (Rule-Based Pipeline)](#2-full-batch-crop-generation-rule-based-pipeline)
  - [3. Visual Oval & Circle Comparison Suite](#3-visual-oval--circle-comparison-suite)
  - [4. Generating Transparent PNG Cutouts & `transparency_previews`](#4-generating-transparent-png-cutouts--transparency_previews)
  - [5. Speed Optimization Summary](#5-speed-optimization-summary)
- [Project Directory Structure](#project-directory-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Supported Categories](#supported-categories)
- [Troubleshooting](#troubleshooting)

---

## Requirements & Installation

### Requirements

- **Python**: 3.10+
- **OS**: Linux, macOS, or Windows
- **RAM**: 4 GB minimum (Florence-2 model uses ~1.5 GB)
- **Disk**: ~1.5 GB (for Florence-2 model weights, cached locally on first run)
- **Hardware**: CPU-only (deterministic, reproducible inference)

### Setup & Dependencies

```bash
cd CroptingImage/scripts

# Create & activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages (Pin transformers==4.48.0 for Florence-2 compatibility)
pip install torch torchvision "transformers==4.48.0" einops timm pillow opencv-python numpy
```

---

## Quick Start

```bash
cd CroptingImage/scripts

# Test a single rectangular image
python artwork_cropper_v3.py "../input_images/rectangular_and_squares/images different categories/1. Canvas or Paperworks on a wall/1.+£100.+WCS+only_Untitled+106.+42x30cm1.webp"

# Test a single circular / oval image
python artwork_cropper_v3.py "../input_images/oval_and_circles/oval-circle-shapes/dl4wxya9p07jx04ng821.jpeg"

# Run batch processing on all input_images
python artwork_cropper_v3.py "../input_images" --batch --output "../test_results_new/all_cropped_outputs"
```

---

## Testing Individual Files

### Using Primary AI Pipeline (`artwork_cropper_v3.py`)

The primary pipeline uses **Microsoft Florence-2-large** for semantic detection combined with OpenCV CLAHE edge refinement, quad perspective correction, and automatic geometric round/oval shape detection.

#### Command Syntax
```bash
python artwork_cropper_v3.py <path_to_image> [options]
```

#### Common Examples

1. **Basic single image crop**:
   ```bash
   python artwork_cropper_v3.py /path/to/image.jpg
   ```
   *Saves output as `<stem>_cropped.<ext>` in the same directory as the source image.*

2. **Specify output path**:
   ```bash
   python artwork_cropper_v3.py /path/to/image.jpg --output /path/to/output_cropped.jpg
   ```

3. **Enable physical frame trimming** (removes outer picture frame borders):
   ```bash
   python artwork_cropper_v3.py /path/to/framed_painting.jpg --frame-trim
   ```

4. **Adjust frame trim depth** (for thick frames, e.g. 15% search depth):
   ```bash
   python artwork_cropper_v3.py /path/to/framed_painting.jpg --frame-trim --frame-depth 0.15
   ```

---

### Using Fast Rule-Based Pipeline (`artwork_cropper.py`)

The legacy pipeline uses pure OpenCV pixel-statistics (color difference, background estimation, contours, ellipse fitting). It requires no neural network models or PyTorch.

#### Command Syntax
```bash
python artwork_cropper.py <path_to_image> [options]
```

#### Common Examples

1. **Auto-detect mode**:
   ```bash
   python artwork_cropper.py /path/to/image.jpg
   ```

2. **Force round canvas mode**:
   ```bash
   python artwork_cropper.py /path/to/oval_painting.jpg --mode round
   ```

3. **Export transparent die-cut PNG** (removes background around circular canvas):
   ```bash
   python artwork_cropper.py /path/to/oval_painting.jpg --mode round --cutout --output /path/to/output.png
   ```

4. **Force specific detection mode**:
   - `--mode plain`: Artwork on neutral solid background.
   - `--mode bgdiff`: Artwork on distinct/different colored background.
   - `--mode pinned`: Paper/ink artwork pinned to wall with padding.
   - `--mode room`: Framed artwork in complex room scene.

---

## Running Full Tests on All Inputs

To run processing across all images in the `input_images/` directory and generate complete outputs in a target directory, follow these instructions.

### 1. Full Batch Crop Generation (AI Pipeline)

Processes all rectangular, square, circular, and oval images recursively from `input_images/` using the Florence-2 + OpenCV pipeline (`artwork_cropper_v3.py`).

```bash
cd CroptingImage/scripts

# Run batch crop on all inputs, saving outputs to test_results_new/all_cropped_outputs/
python artwork_cropper_v3.py "../input_images" --batch --output "../test_results_new/all_cropped_outputs"
```

* **Behavior**: Preserves exact input subfolder hierarchy in the output folder.
* **Speed**: ~15–30 seconds per image on CPU.

---

### 2. Full Batch Crop Generation (Rule-Based Pipeline)

To run a fast batch processing test across all input_images using only OpenCV (`artwork_cropper.py`):

```bash
cd CroptingImage/scripts

# Run fast OpenCV batch crop on all inputs
python artwork_cropper.py "../input_images" --batch --output "../test_results_new/opencv_batch_outputs"
```

---

### 3. Visual Oval & Circle Comparison Suite

Generates side-by-side visual comparison images (Original with fitted ellipse/bbox on left panel, cropped result on right panel) for all circular and oval shapes.

```bash
cd CroptingImage/scripts

# Generate visual test comparison images
python test_oval_circle_crops.py \
  --input "../input_images/oval_and_circles/oval-circle-shapes" \
  --output "../test_results_new/oval_and_cricles/oval_circle_test_results"
```

* **Output**: Generates `*_compare.jpg` files visualizing detection accuracy, fitted ellipse parameters, and bounding box extents.

---

### 4. Generating Transparent PNG Cutouts & `transparency_previews`

Generates transparent die-cut PNG crops for circular/oval canvases, and renders visual preview images (`*_on_checkerboard.jpg`) inside the `transparency_previews/` directory by compositing the alpha channel onto a checkerboard background pattern.

```bash
cd CroptingImage/scripts

# Generate transparent PNG cutouts and transparency_previews folder
python generate_transparency_previews.py \
  --input "../input_images/oval_and_circles/oval-circle-shapes" \
  --output "../test_results_new/oval_circle_cutout_results"
```

* **Output Files**:
  - `*.png` — True die-cut PNG image with 4-channel BGRA alpha transparency.
  - `transparency_previews/*_on_checkerboard.jpg` — Composited JPEG preview rendered on an alternating checkerboard pattern for easy visual review in standard image viewers.

---

### 5. Speed Optimization Summary

CPU inference performance in `artwork_cropper_v3.py` has been optimized for high-speed batch processing on multi-core CPUs while maintaining full detection accuracy:

| Optimization | Technical Implementation | Speedup Impact |
|---|---|---|
| **Multi-Core Threading** | Dynamically utilizes all CPU cores (`torch.set_num_threads`) for PyTorch model tensor operations. | ~4× multi-core throughput |
| **Greedy Decoding** | Switched from beam search (`num_beams=3`) to greedy decoding (`num_beams=1`). | ~3× faster token generation |
| **Input Image Rescaling** | Downscales high-resolution images to `768px` max side before Florence-2 ViT encoding, then rescales bounding boxes back. | Reduces ViT patch encoding overhead |
| **Capped Token Generation** | Set `max_new_tokens=128` (bounding box coordinate outputs require only ~10–20 tokens). | Eliminates redundant token generation |
| **Fast-Path Background Pre-Screen** | Detects bounded artworks on plain backgrounds via OpenCV background subtraction, skipping Florence-2 entirely when safe. | **~6–8× faster** for eligible images (~6s vs ~40s) |

#### Benchmark Performance

| Image Category / Scenario | Baseline Speed | Optimized Speed | Performance Gain |
|---|---|---|---|
| **Plain Background** (Fast-Path Bypasses Florence-2) | ~49.0s / image | **~6.2s / image** | **~8× Faster** |
| **Complex Background** (Florence-2 Inference Required) | ~49.0s / image | **~38.7s / image** | **~21% Faster** |

---

## Project Directory Structure

```
CroptingImage/
├── README.md                              # Documentation & setup instructions
├── scripts/
│   ├── artwork_cropper_v3.py              # Primary script (Florence-2 AI + OpenCV)
│   ├── artwork_cropper.py                 # Legacy script (Rule-based OpenCV & round engine)
│   ├── test_oval_circle_crops.py          # Visual test generator for circular/oval shapes
│   └── generate_transparency_previews.py  # Transparent PNG & checkerboard preview generator
│
├── input_images/
│   ├── oval_and_circles/
│   │   └── oval-circle-shapes/            # Oval and circular test canvases
│   └── rectangular_and_squares/
│       └── images different categories/
│           ├── 1. Canvas or Paperworks on a wall/
│           ├── 3. Painting on a shelf/
│           ├── 6. Picture taken from the side.../
│           ├── 11. Circle paintings or Non-rectangular paintings/
│           └── ... (21 categories total)
│
└── test_results_new/                      # Output folder for generated results
    ├── all_cropped_outputs/               # Full batch crop outputs
    └── oval_circle_cutout_results/        # Transparent PNG cutouts
        └── transparency_previews/         # Composited checkerboard preview JPEGs
```

---

## Pipeline Architecture

```
                      Input Image
                           │
                           ▼
              ┌─────────────────────────┐
              │  Round Shape Detection  │ ◄── Geometric Ellipse Engine
              │  (ellipse-based crop)   │     (Handles circles & ovals)
              └────────────┬────────────┘
                           │ not round
                           ▼
              ┌─────────────────────────┐
              │  Florence-2 Detection   │ ◄── Semantic Vision-Language AI
              │  (bbox + confidence)    │     (Handles rooms, easels, shelves)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  CLAHE Edge Refinement  │ ◄── Adaptive histogram equalization
              │  (local + full-image)   │     (Snaps boundary to artwork/mat edge)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Perspective Correction  │ ◄── Quad detection & warp
              │ (for angled paintings)  │     (Corrects skewed perspective)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Optional Frame Trim    │ ◄── Edge variance scanner
              │  (--frame-trim flag)    │     (Trims physical frame borders)
              └────────────┬────────────┘
                           │
                           ▼
                    Cropped Output
```

---

## Supported Categories

| # | Category | Key Feature |
|---|----------|-------------|
| 1 | Canvas on wall | Standard wall-mounted rectangular canvases |
| 3 | Painting on shelf | Art resting on shelf/ledge |
| 4 | Paper works | Paper artworks on matching wall background |
| 4a | Paper with clip | Pinned paper works with clips |
| 6 | Side angle | High perspective skew / angled photos |
| 6a–g | Leaning / Easel | Art on floor, sofa, stool, easel |
| 7 | Room environment | Full room context |
| 7a | Room with person | People in frame |
| 9 | Frame matches wall | Low contrast frame vs wall |
| 10 | Frame differs from wall | High contrast frame vs wall |
| 11 | Circle / non-rectangular | Circular, round, and oval canvases |
| 12 | Multiple paintings | Multi-artwork photos |
| 13 | Nearly cropped | High fill factor inputs |
| 14 | Irregular edges | Non-straight edge artwork |

---

## Troubleshooting

1. **`ModuleNotFoundError: No module named 'transformers'`**:
   Ensure you run commands using the python environment where dependencies were installed (e.g. `python3` or `/path/to/venv/bin/python`).

2. **`transformers` 5.x Incompatibility**:
   Florence-2 requires `transformers==4.48.0`. Re-install with:
   ```bash
   pip install "transformers==4.48.0"
   ```

3. **Input File Not Found in Batch Mode**:
   Make sure to pass `--batch` when specifying a folder path:
   ```bash
   python artwork_cropper_v3.py "../input_images" --batch
   ```
