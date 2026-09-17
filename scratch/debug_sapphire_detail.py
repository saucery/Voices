import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
sapphire_crop = img[440:485, 580:685]
hsv = cv2.cvtColor(sapphire_crop, cv2.COLOR_BGR2HSV)
b, g, r = cv2.split(sapphire_crop)

olive_gem_bg_mask = (
    (g >= 36) & (g <= 125) &
    (r >= 42) & (r <= 135) &
    (b < 55) &
    (g.astype(np.int16) >= b.astype(np.int16) + 10) &
    (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 42) &
    (hsv[:, :, 1] >= 60) &
    (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] <= 135)
).astype(np.uint8) * 255

gem_text_border_mask = (
    (r >= 140) & (g >= 125) &
    (b < 120) &
    (g.astype(np.int16) >= b.astype(np.int16) + 25) &
    (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 40) &
    (hsv[:, :, 1] >= 60) &
    (hsv[:, :, 2] >= 140)
).astype(np.uint8) * 255

print(f"Sapphire crop shape: {sapphire_crop.shape}")
print(f"Olive mask mean: {np.mean(olive_gem_bg_mask)*100:.1f}%")
print(f"Text mask count: {np.count_nonzero(gem_text_border_mask)}")

# Let's check text dilation
k_text_cluster = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
text_dilated = cv2.dilate(gem_text_border_mask, k_text_cluster)
t_contours, _ = cv2.findContours(text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Text contours count: {len(t_contours)}")
for cnt in t_contours:
    rx, ry, rw, rh = cv2.boundingRect(cnt)
    print(f"  Contour: ({rx}, {ry}, {rw}x{rh})")
