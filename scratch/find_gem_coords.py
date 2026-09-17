import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")

# Let's save a grid overlay of gems_screenshot so we can see exact pixel coords
grid = img.copy()
for y in range(0, img.shape[0], 50):
    cv2.line(grid, (0, y), (img.shape[1], y), (0, 255, 0), 1)
    cv2.putText(grid, str(y), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

for x in range(0, img.shape[1], 50):
    cv2.line(grid, (x, 0), (x, img.shape[0]), (0, 255, 0), 1)
    cv2.putText(grid, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\grid_screenshot.png", grid)

# Ruby is near x: 550-620, y: 180-230
ruby_crop = img[180:240, 540:630]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\ruby_crop.png", ruby_crop)

# Sapphire is near x: 570-690, y: 520-580
sapphire_crop = img[510:570, 570:690]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\sapphire_crop.png", sapphire_crop)

print("Saved ruby and sapphire crops")
