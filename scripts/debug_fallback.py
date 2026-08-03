import cv2
import numpy as np

img_path = "/media/tinhtran/01D85BC599D1D460/FreeLancer/ComputerVision/input images/oval_and_circles/oval-circle-shapes/m5y8oc0nqfzs6lquduss.jpeg"
img_bgr = cv2.imread(img_path)
h, w = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

cs = min(10, h // 6, w // 6)
corners = np.concatenate([
    gray[:cs, :cs].flatten(),
    gray[:cs, -cs:].flatten(),
    gray[-cs:, :cs].flatten(),
    gray[-cs:, -cs:].flatten(),
])
bg_mean = float(corners.mean())
bg_std = float(corners.std())

print(f"bg_mean={bg_mean:.3f}, bg_std={bg_std:.3f}")

diff = cv2.absdiff(gray, np.full_like(gray, int(bg_mean)))
_, fg_mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(
    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"Number of contours: {len(contours)}")

if contours:
    largest = max(contours, key=cv2.contourArea)
    cx, cy, cw, ch = cv2.boundingRect(largest)
    obj_frac = (cw * ch) / (w * h)
    print(f"Largest contour: cx={cx}, cy={cy}, cw={cw}, ch={ch}, obj_frac={obj_frac:.3f}")
