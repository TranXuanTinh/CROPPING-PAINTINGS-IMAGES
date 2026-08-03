# Phase 2: Fine-Tuning Guide

Phase 2 replaces the zero-shot YOLO-World model with a custom-trained YOLOv8n
model for maximum accuracy. This guide walks through the full process.

## When to Use Phase 2

- Phase 1 (YOLO-World) achieves ~85–90% accuracy out of the box
- Phase 2 (fine-tuned) targets ~95–99% accuracy
- **Recommended** if you process paintings regularly and need production-grade results

## Overview

```
Step 1: Generate pre-annotations (automated, ~2 min)
Step 2: Review & correct annotations (manual, ~30–60 min for 50 images)
Step 3: Train the model (automated, ~1–4 hours on CPU)
Step 4: Use the trained model (drop-in replacement)
```

## Step 1: Generate Pre-Annotations

The `annotate_helper.py` script runs YOLO-World on all your images and saves
the predictions as YOLO-format label files. These serve as starting annotations
that you then review and correct — much faster than annotating from scratch.

```bash
python scripts/annotate_helper.py \
    --input "input images/" \
    --output dataset/ \
    --split 0.85
```

This creates:
```
dataset/
├── train/
│   ├── images/     # 85% of images
│   └── labels/     # Pre-generated YOLO labels
├── val/
│   ├── images/     # 15% of images
│   └── labels/
└── data.yaml       # Dataset config
```

## Step 2: Review & Correct Annotations

**CRITICAL**: The pre-annotations are NOT perfect. You must review and correct
them before training. A model trained on bad labels will produce bad detections.

### Recommended Tools (Free)

1. **CVAT** (Computer Vision Annotation Tool)
   - Web-based, free: https://www.cvat.ai/
   - Upload your `dataset/train/images/` directory
   - Import labels from `dataset/train/labels/` (YOLO format)
   - Review each image, adjust bounding boxes
   - Export corrected labels back in YOLO format

2. **Label Studio**
   - Web-based, free: https://labelstud.io/
   - `pip install label-studio && label-studio`
   - Import with YOLO format connector

3. **LabelImg** (Simple desktop app)
   ```bash
   pip install labelImg
   labelImg dataset/train/images/ dataset/train/labels/ dataset/classes.txt
   ```
   Create `dataset/classes.txt` with one line: `painting`

### What to Check

For each image, verify:
- ✅ The bounding box tightly encloses the painting (not too loose, not too tight)
- ✅ The box includes the full painting surface (not just the content inside a frame)
- ✅ No extra objects are detected (furniture, windows, etc.)
- ✅ Paintings that YOLO missed have been manually annotated
- ✅ Multiple paintings in one image each have their own box

### Tips for Quality

- **Tight boxes**: The box should touch the painting edges on all 4 sides
- **Include frames**: If the painting has a frame, include it in the box
- **Background images**: Keep 5–10 images with NO labels (empty .txt files) — these teach the model what is NOT a painting
- **Minimum images**: 50 annotated images is the minimum; 100+ is better

## Step 3: Train the Model

```bash
python scripts/train_yolo.py \
    --data dataset/data.yaml \
    --epochs 100 \
    --device cpu
```

### Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 100 | Maximum training epochs |
| `--imgsz` | 640 | Training image size |
| `--batch` | 4 | Batch size (keep small for CPU) |
| `--freeze` | 10 | Backbone layers to freeze (prevents overfitting) |
| `--patience` | 20 | Early stopping patience |
| `--base-model` | yolov8n.pt | Pretrained weights to start from |

### Expected Training Time

| Hardware | ~50 images | ~200 images | ~500 images |
|---|---|---|---|
| Modern CPU (8 cores) | ~1 hour | ~3 hours | ~6 hours |
| Laptop CPU (4 cores) | ~2 hours | ~6 hours | ~12 hours |

### Monitoring Training

Training creates results in `models/painting_detector/`:
- `results.png` — loss/metric curves
- `confusion_matrix.png` — classification accuracy
- `val_batch*.jpg` — validation predictions (check visually!)
- `weights/best.pt` — best model weights

## Step 4: Use the Trained Model

The best model is automatically saved to `models/best.pt`.

```bash
# Single image
python scripts/yolo_artwork_cropper.py painting.jpg --model models/best.pt

# Batch
python scripts/yolo_artwork_cropper.py "./input images/" --batch --model models/best.pt

# Test with comparisons
python scripts/test_runner.py \
    --input "input images/" \
    --output "yolo_test_results_v2/" \
    --model models/best.pt
```

### Updating Configuration

For the fine-tuned model, you can adjust `yolo_config.py`:
```python
# Switch to Phase 2 as default
YOLO_CUSTOM_MODEL = "models/best.pt"

# Higher confidence threshold (fine-tuned model is more precise)
YOLO_CONFIDENCE = 0.40

# Can use smaller image size (fine-tuned model is optimised)
YOLO_IMGSZ = 640
```

## Step 5: Export for Production (Optional)

For faster CPU inference, export the model to ONNX or OpenVINO format:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")

# ONNX (portable, fast)
model.export(format="onnx")

# OpenVINO (fastest on Intel CPUs)
model.export(format="openvino")
```

Then use the exported model:
```bash
python scripts/yolo_artwork_cropper.py painting.jpg --model models/best.onnx
```

## Troubleshooting

### "Loss not decreasing"
- Check that labels are correct (Step 2 is critical!)
- Try unfreezing more layers: `--freeze 5` or `--freeze 0`
- Increase training images

### "Too many false positives"
- Add background images (images with no paintings, empty labels)
- Increase `YOLO_CONFIDENCE` threshold
- Train with more epochs

### "Slow training"
- Reduce `--imgsz` to 416
- Reduce `--epochs` to 50
- Consider using Google Colab (free GPU) for training only
