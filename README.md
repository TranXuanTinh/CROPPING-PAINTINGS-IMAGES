# YOLO Painting Cropper 🎨✂️

An optimized, CPU-only Computer Vision pipeline designed to detect and crop paintings (rectangular, square, oval, or circular) from photographs with extremely high accuracy and sub-second performance.

This project implements a hybrid architecture combining **YOLO Object Detection** (for localization) with **advanced OpenCV algorithms** (for shape classification, perspective correction, sub-pixel edge alignment, and robust ellipse fitting).

---

## 🌟 Key Features

*   **⚡ Sub-Second CPU Inference:** Runs in `<0.1s` with a fine-tuned model (or `2-5s` zero-shot) on standard CPU hardware — replacing heavy transformer models (Florence-2) which required up to 40s per image.
*   **📐 Shape-Aware Processing:**
    *   **Rectangular Paintings:** Detects perspective skew, applies a 4-point perspective warp for angled shots, and snaps to edges with sub-pixel precision.
    *   **Oval & Circular Paintings:** Employs radial gradient search and robust RANSAC ellipse-fitting to isolate round canvas boundaries cleanly even against low-contrast backgrounds.
*   **🛠️ Robust Post-Detection Refinement:**
    *   *CLAHE Edge Snapping:* Enhances subtle contrast boundaries to snap boundaries precisely to the true canvas border.
    *   *Background Contrast Tightening:* Trims background wall margins automatically on plain/studio walls.
    *   *Frame Trimming:* Automatically detects and removes physical picture frames if requested.
*   **🔄 Hybrid Fallback Pipeline:** If YOLO fails to detect an object, the pipeline falls back to multi-strategy OpenCV contour and edge detection, guaranteeing that no image is dropped.

---

## 🏗️ Two-Phase Architecture

We provide a flexible two-stage pipeline:

| Phase | Strategy | Training Needed? | Model Size | CPU Speed | Target Accuracy |
|---|---|---|---|---|---|
| **Phase 1** | **YOLO-World (Zero-Shot)** | ❌ No | ~47 MB | ~2–5 s | ~85–95% |
| **Phase 2** | **Fine-Tuned YOLOv8n** | ✅ Yes | **~6 MB** | **<0.1 s** | **95–99%** |

*To avoid manual annotation for Phase 2, we include a Florence-2 auto-annotation script that generates high-quality labels automatically from your existing images.*

---

## 📁 Repository Structure

```
ComputerVision/
├── README.md                   # ★ You are here
│
├── scripts/
│   ├── yolo_artwork_cropper.py # Main crop execution pipeline
│   ├── yolo_config.py          # Central configuration & thresholds
│   ├── shape_classifier.py     # Rectangular vs Oval classifier
│   ├── perspective_correction.py # 4-point perspective warper
│   ├── crop_refinement.py      # CLAHE alignment, padding, and frame trimming
│   ├── oval_cropper.py         # RANSAC ellipse fitting & masking
│   ├── test_runner.py          # Batch evaluation & side-by-side generator
│   │
│   # Phase 2 Fine-Tuning Tools
│   ├── florence_auto_annotate.py # Automatic Florence-2 dataset labeler
│   ├── train_yolo.py           # YOLOv8n fine-tuning script
│   └── requirements.txt        # Dependency declaration
│
├── docs/
│   ├── DESIGN.md               # Pipeline architecture & design decisions
│   ├── SETUP.md                # Environment & installation instructions
│   ├── USAGE.md                # CLI command reference & options
│   └── TRAINING.md             # Custom dataset fine-tuning guide
│
├── models/                     # Custom trained weights (Phase 2 best.pt)
└── yolo_test_results/          # Evaluation report outputs
```

---

## 🚀 Quick Start

### 1. Installation

Ensure your **Miniconda base environment** is active, then install the dependencies:

```bash
# Navigate to the repository
cd /path/to/ComputerVision

# Install dependencies
pip install -r scripts/requirements.txt
```

### 2. Verify Setup

```bash
python scripts/yolo_artwork_cropper.py --check
```

### 3. Basic Usage

**Crop a single image:**
```bash
python scripts/yolo_artwork_cropper.py photo.jpg
```

**Crop a batch of images (recursively, preserving folder structure):**
```bash
python scripts/yolo_artwork_cropper.py "input images/" --batch --output "output_dir/"
```

**Trim physical frames from cropped images:**
```bash
python scripts/yolo_artwork_cropper.py photo.jpg --frame-trim
```

---

## 📊 Evaluation & Testing

Run the side-by-side comparison generator to test the pipeline on your dataset and generate timing/accuracy reports:

```bash
python scripts/test_runner.py --input "input images/" --output "yolo_test_results/"
```

This creates side-by-side images (Original + Detected Box vs Crop) and generates:
*   `yolo_test_results/summary.csv` — Per-image dimensions and processing metrics.
*   `yolo_test_results/accuracy_report.md` — Per-category detection breakdowns and speed metrics.

---

## 📘 Detailed Documentation

Please refer to the following guides in the `docs/` folder for deeper information:
*   **[Setup & Environment](docs/SETUP.md):** Detailed setup steps, environment configuration, and troubleshooting.
*   **[Usage Guide](docs/USAGE.md):** Full CLI arguments, shape overrides, and frame trimming options.
*   **[Design Document](docs/DESIGN.md):** Math/logic explanation of shape classification, RANSAC fitting, and fallback systems.
*   **[Fine-Tuning Guide](docs/TRAINING.md):** How to run automated dataset labeling, build local datasets, and train a 6MB custom YOLOv8 model.
