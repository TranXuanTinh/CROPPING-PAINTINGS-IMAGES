import sys
import cv2
from pathlib import Path
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes([
    "painting",
    "artwork",
    "picture",
    "canvas",
    "framed artwork",
    "picture frame",
    "framed picture",
    "wall art",
])

img_path = "/media/tinhtran/01D85BC599D1D460/FreeLancer/ComputerVision/input images/rectangular_and_squares/images different categories/3. Painting on a shelf/imgi_3_jannasyvanoja_painting_artwork_sommarstilleben_lokalhelsinki2025-7-550x682.jpg"

results = model.predict(source=img_path, conf=0.01, device="cpu", imgsz=1280)
print(f"Total detected boxes: {len(results[0].boxes)}")
for i, box in enumerate(results[0].boxes):
    xyxy = box.xyxy[0].cpu().numpy().tolist()
    conf = float(box.conf[0].cpu().numpy())
    cls = int(box.cls[0].cpu().numpy())
    class_name = results[0].names[cls]
    print(f"Box {i}: class={class_name} ({cls}), conf={conf:.3f}, xyxy={[int(v) for v in xyxy]}")
