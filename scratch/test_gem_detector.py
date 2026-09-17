import cv2
import numpy as np
import os
import sys

def detect_gems(screen):
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(screen)
    sw, sh = screen.shape[1], screen.shape[0]

    # 1. Dark Olive/Greenish-Brown Background Mask
    # PoE Gem filter style: dark olive background (Hue ~ 15-48, Saturation >= 25, Value in [18, 120])
    olive_bg_mask = (
        (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 50) &
        (hsv[:, :, 1] >= 25) &
        (hsv[:, :, 2] >= 18) & (hsv[:, :, 2] <= 125) &
        (g.astype(np.int16) >= b.astype(np.int16) - 5)
    ).astype(np.uint8) * 255

    # 2. Yellow / Lime / Chartreuse Text & Border Mask
    # Bright yellow/lime text: Hue ~ 18-42, Saturation >= 80, Value >= 150
    yellow_lime_mask = (
        (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) &
        (hsv[:, :, 1] >= 80) &
        (hsv[:, :, 2] >= 150) &
        (g > 110) & (r > 130)
    ).astype(np.uint8) * 255

    # Morphology to find solid gem background boxes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    bg_closed = cv2.morphologyEx(olive_bg_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(bg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []

    min_w = 45
    max_w = 260
    min_h = 16
    max_h = 55
    min_ar = 1.3
    min_bg_frac = 0.35
    min_text_px = 12

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Dimensions check (gems are compact: Ruby ~ 65px, Sapphire ~ 100px, Time-Lost Diamond ~ 180px)
        if w < min_w or w > max_w or h < min_h or h > max_h:
            continue
        
        ar = w / max(1.0, float(h))
        if ar < min_ar:
            continue

        box_olive = olive_bg_mask[y:y+h, x:x+w]
        box_lime = yellow_lime_mask[y:y+h, x:x+w]
        bg_frac = np.mean(box_olive > 0)
        lime_pixels = int(np.count_nonzero(box_lime > 0))

        if bg_frac >= min_bg_frac and lime_pixels >= min_text_px:
            conf = min(1.0, 0.60 + (lime_pixels / 120.0) + (bg_frac * 0.3))
            items.append(("gem_uncut_cut", x, y, w, h, conf, bg_frac, lime_pixels))

    return items

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
res = detect_gems(img)

print(f"Total gems detected: {len(res)}")
for r in res:
    print(f"  -> {r[0]} at ({r[1]}, {r[2]}, {r[3]}x{r[4]}) conf={r[5]:.2f} bg_frac={r[6]:.2f} text_px={r[7]}")

# Save visualization overlay
vis = img.copy()
for r in res:
    x, y, w, h = r[1], r[2], r[3], r[4]
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(vis, "GEM", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\detected_gems_overlay.png", vis)
print("Saved detected_gems_overlay.png")
