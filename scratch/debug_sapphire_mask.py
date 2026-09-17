import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
sapphire_crop = img[440:485, 580:685]

hsv = cv2.cvtColor(sapphire_crop, cv2.COLOR_BGR2HSV)
b, g, r = cv2.split(sapphire_crop)

print(f"Sapphire crop size: {sapphire_crop.shape}")

# Inspect BGR and HSV of sapphire background
# Background is dark pixels (Value < 120)
bg_pixels = sapphire_crop[hsv[:,:,2] < 120]
print(f"BG pixels count: {len(bg_pixels)}")
if len(bg_pixels) > 0:
    hsv_bg = cv2.cvtColor(bg_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    print(f"BG Hue: min={hsv_bg[:,0].min()}, mean={hsv_bg[:,0].mean():.1f}, max={hsv_bg[:,0].max()}")
    print(f"BG Sat: min={hsv_bg[:,1].min()}, mean={hsv_bg[:,1].mean():.1f}, max={hsv_bg[:,1].max()}")
    print(f"BG Val: min={hsv_bg[:,2].min()}, mean={hsv_bg[:,2].mean():.1f}, max={hsv_bg[:,2].max()}")

# Inspect text pixels
text_pixels = sapphire_crop[hsv[:,:,2] >= 140]
print(f"\nText pixels count: {len(text_pixels)}")
if len(text_pixels) > 0:
    hsv_txt = cv2.cvtColor(text_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    print(f"Text Hue: min={hsv_txt[:,0].min()}, mean={hsv_txt[:,0].mean():.1f}, max={hsv_txt[:,0].max()}")
    print(f"Text Sat: min={hsv_txt[:,1].min()}, mean={hsv_txt[:,1].mean():.1f}, max={hsv_txt[:,1].max()}")
    print(f"Text Val: min={hsv_txt[:,2].min()}, mean={hsv_txt[:,2].mean():.1f}, max={hsv_txt[:,2].max()}")

# Let's check why findContours didn't find it
# 1. Olive mask
olive_bg_mask = (
    (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 55) &
    (hsv[:, :, 1] >= 20) &
    (hsv[:, :, 2] >= 15) & (hsv[:, :, 2] <= 135)
).astype(np.uint8) * 255

print(f"\nOlive mask match percentage in sapphire crop: {np.mean(olive_bg_mask)*100:.1f}%")

# Let's save sapphire mask image
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\sapphire_mask.png", olive_bg_mask)
