import cv2
import numpy as np

def detect_gems_v3(screen):
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(screen)
    sw, sh = screen.shape[1], screen.shape[0]

    # 1. Dark Olive/Greenish-Brown Background Mask (PoE Uncut/Cut Gems)
    olive_bg = (
        (g >= 35) & (g <= 125) &
        (r >= 40) & (r <= 135) &
        (b < 55) &
        (g.astype(np.int16) >= b.astype(np.int16) + 10) &
        (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 42) &
        (hsv[:, :, 1] >= 60) &
        (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] <= 135)
    ).astype(np.uint8) * 255

    # 2. Gem Text & Border Mask (Yellow/Lime)
    gem_text = (
        (r >= 140) & (g >= 125) &
        (b < 120) &
        (g.astype(np.int16) >= b.astype(np.int16) + 20) &
        (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 38) &
        (hsv[:, :, 1] >= 65) &
        (hsv[:, :, 2] >= 140)
    ).astype(np.uint8) * 255

    # Combine text only where it touches or is inside olive background (dilate olive bg by 2px)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    olive_dilated = cv2.dilate(olive_bg, k_dilate)
    gem_text_isolated = cv2.bitwise_and(gem_text, olive_dilated)

    # 1D Horizontal closing of olive background + isolated text
    combined_gem = cv2.bitwise_or(olive_bg, gem_text_isolated)
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    gem_closed = cv2.morphologyEx(combined_gem, cv2.MORPH_CLOSE, k_horiz)

    contours, _ = cv2.findContours(gem_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []

    min_w = 40
    max_w = 240
    min_h = 16
    max_h = 55
    min_ar = 1.3
    min_bg_frac = 0.35
    min_text_px = 15

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_w or w > max_w or h < min_h or h > max_h:
            continue
        if (w / max(1.0, float(h))) < min_ar:
            continue

        box_olive = olive_bg[y:y+h, x:x+w]
        box_text = gem_text[y:y+h, x:x+w]
        bg_frac = np.mean(box_olive > 0)
        text_pixels = int(np.count_nonzero(box_text > 0))

        if bg_frac >= min_bg_frac and text_pixels >= min_text_px:
            conf = min(1.0, 0.65 + (text_pixels / 120.0) + (bg_frac * 0.25))
            items.append(("gem_uncut_cut", x, y, w, h, conf, bg_frac, text_pixels))

    return items

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
res = detect_gems_v3(img)
print(f"Total gems detected: {len(res)}")
for r in res:
    print(f"  -> {r[0]} at ({r[1]}, {r[2]}, {r[3]}x{r[4]}) conf={r[5]:.2f} bg_frac={r[6]:.2f} text_px={r[7]}")

vis = img.copy()
for r in res:
    x, y, w, h = r[1], r[2], r[3], r[4]
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(vis, "GEM", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_v3_result.png", vis)
print("Saved gems_v3_result.png")
