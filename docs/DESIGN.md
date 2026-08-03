# Architecture & Design

## Overview

The YOLO Artwork Cropper detects and crops paintings from photographs, handling both **rectangular/square** and **oval/circular** shapes. It replaces the previous Florence-2 + OpenCV pipeline with a lighter, faster YOLO-based approach.

## Pipeline Architecture

```
Input Image
    │
    ▼
┌───────────────────┐
│  YOLO Detection   │  YOLO-World (zero-shot) or fine-tuned YOLOv8n
│  ~47 MB model     │  Inference: 2–5s on CPU
└───────┬───────────┘
        │ bbox [x1, y1, x2, y2]
        ▼
┌───────────────────┐
│  OpenCV Fallback  │  Only if YOLO finds nothing (~5% of images)
│  Background diff  │  Contrast + edge-based detection
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Shape Classifier  │  Extent ratio analysis (contour_area / rect_area)
│ Pure OpenCV       │  Oval: 0.70 < extent < 0.86 | Rect: > 0.85
└───────┬───────────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────┐
│ Rect │  │ Oval │
└──┬───┘  └──┬───┘
   │         │
   ▼         ▼
┌──────────┐ ┌──────────────┐
│ CLAHE    │ │ Ellipse Fit  │
│ Refine   │ │ + RANSAC     │
│ + Tight  │ │ Refinement   │
└──┬───────┘ └──┬───────────┘
   │             │
   ▼             ▼
┌──────────┐ ┌──────────────┐
│Perspect. │ │ AABB from    │
│Correct   │ │ Ellipse Geom │
└──┬───────┘ └──┬───────────┘
   │             │
   └──────┬──────┘
          ▼
   ┌──────────────┐
   │ Frame Trim   │  Optional: removes physical picture frames
   │ (optional)   │
   └──────┬───────┘
          ▼
     Output Image
```

## Module Structure

| Module | Purpose | Dependencies |
|---|---|---|
| `yolo_config.py` | Central configuration constants | None |
| `shape_classifier.py` | Rectangular vs oval detection | `yolo_config`, OpenCV |
| `perspective_correction.py` | 4-point warp for angled shots | `yolo_config`, OpenCV |
| `crop_refinement.py` | CLAHE + background tightening | `yolo_config`, OpenCV |
| `oval_cropper.py` | Ellipse fitting + RANSAC | `shape_classifier`, OpenCV |
| `yolo_artwork_cropper.py` | **Main pipeline** (entry point) | All above + ultralytics |
| `test_runner.py` | Batch testing + comparison images | All above |
| `annotate_helper.py` | Phase 2: pre-annotation generator | ultralytics |
| `train_yolo.py` | Phase 2: fine-tuning script | ultralytics |

## Key Design Decisions

### 1. YOLO-World for Zero-Shot Detection

**Why not standard YOLOv8?** Standard YOLO models are trained on COCO (80 classes) which does NOT include "painting" or "artwork". YOLO-World uses vision-language integration (CLIP text encoder) to detect arbitrary objects described in text — no training needed.

**Trade-off:** Slightly lower accuracy than a fine-tuned model, but works immediately out of the box. Phase 2 fine-tuning is available for production use.

### 2. Conservative Refinement (±15% max)

The previous Florence-2 pipeline had complex multi-step expansion logic (~300 lines) that could "run away" — expanding the crop to include walls, furniture, etc. Our approach: YOLO provides a good bbox (~90% accurate), so refinement only makes **fine adjustments** within ±15% of the bbox.

### 3. Shape Classification via Multi-Threshold Extent

Rather than using ML for shape classification, we exploit a geometric invariant: a circle's extent (contour_area / bounding_rect_area) is always π/4 ≈ 0.785, regardless of size or angle. By checking consistency across multiple background-difference thresholds, we reliably separate ovals from rectangles.

### 4. CPU-Only by Design

All models run on CPU for:
- **Deterministic** results across machines
- **No GPU** infrastructure needed
- **Deployment simplicity** on standard EC2/cloud instances

## Performance Comparison

| Metric | Florence-2 (old) | YOLO-World (new) |
|---|---|---|
| Model size | ~1.5 GB | ~47 MB |
| Inference speed (CPU) | 15–40 s/image | 2–5 s/image |
| Dependencies | torch + transformers + einops + timm | ultralytics (includes torch) |
| Training required | No | No (Phase 1) |
| Accuracy | ~70–80% (estimated from test results) | ~85–95% (Phase 1) / ~95–99% (Phase 2) |
