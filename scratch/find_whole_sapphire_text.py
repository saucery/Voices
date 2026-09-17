import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
b, g, r = cv2.split(img)

gem_text_border_mask = (
    (r >= 140) & (g >= 125) &
    (b < 120) &
    (g.astype(np.int16) >= b.astype(np.int16) + 25) &
    (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 40) &
    (hsv[:, :, 1] >= 60) &
    (hsv[:, :, 2] >= 140)
).astype(np.uint8) * 255

k_text_cluster = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
text_dilated = cv2.dilate(gem_text_border_mask, k_text_cluster)
t_contours, _ = cv2.findContours(text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total dilated text contours on whole screen: {len(t_contours)}")
for i, cnt in enumerate(t_contours):
    rx, ry, rw, rh = cv2.boundingRect(cnt)
    if (rx <= 600 <= rx + rw) and (ry <= 460 <= ry + rh):
        print(f"Match for Sapphire: #{i}: rect=({rx}, {ry}, {rw}x{rh})")
