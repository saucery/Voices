import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
b, g, r = cv2.split(img)

# 1. Dark Olive Background Mask
olive_bg_mask = (
    (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 55) &
    (hsv[:, :, 1] >= 20) &
    (hsv[:, :, 2] >= 15) & (hsv[:, :, 2] <= 135)
).astype(np.uint8) * 255

kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
bg_closed = cv2.morphologyEx(olive_bg_mask, cv2.MORPH_CLOSE, kernel_close)

contours, _ = cv2.findContours(bg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours)}")

for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    # Check if contour contains (590, 450) where Sapphire is
    if x <= 590 <= x + w and y <= 450 <= y + h:
        print(f"Sapphire contour #{i}: rect=({x}, {y}, {w}x{h})")
