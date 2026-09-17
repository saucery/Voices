import cv2
import numpy as np
import os
import glob
import sys

def is_ui_zone(x, y, w, h, sw, sh):
    if sw < 600 or sh < 400:
        return False
    cx = x + w // 2
    cy = y + h // 2
    # Top bar / latency graph
    if y < 45 or cy < 45:
        return True
    # Chat window (bottom-left)
    if cx < int(sw * 0.28) and cy > int(sh * 0.58):
        return True
    # Life globe
    if cx < 240 and cy > (sh - 240):
        return True
    # Mana globe
    if cx > (sw - 240) and cy > (sh - 240):
        return True
    # Bottom skill bar
    if cy > (sh - 110):
        return True
    # Minimap (top-right)
    if cx > (sw - 380) and cy < 380:
        return True
    # Buff bar (top-left)
    if cx < 350 and cy < 85:
        return True
    return False

def detect_white_box_red_text_v2(screen):
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(screen)
    sw, sh = screen.shape[1], screen.shape[0]

    white_mask = (
        (r > 175) & (g > 170) & (b > 170) &
        (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 170)
    ).astype(np.uint8) * 255

    red_text_mask = (
        (r > 155) &
        (r.astype(np.int16) - g.astype(np.int16) > 55) &
        (r.astype(np.int16) - b.astype(np.int16) > 55) &
        (g < 110) & (b < 110)
    ).astype(np.uint8) * 255

    min_w = 55
    max_w = 500
    min_h = 16
    max_h = 65
    min_ar = 1.6
    min_bg_frac = 0.40
    min_text_px = 16

    items = []

    # Pass 1: Horizontal White Box Contours
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    white_closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_horiz)
    w_contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in w_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if is_ui_zone(x, y, w, h, sw, sh):
            continue
        if w < min_w or w > max_w or h < min_h or h > max_h:
            continue
        if (w / max(1.0, float(h))) < min_ar:
            continue

        box_white = white_mask[y:y+h, x:x+w]
        box_red = red_text_mask[y:y+h, x:x+w]
        bg_frac = np.mean(box_white > 0)
        red_frac = np.mean(box_red > 0)
        red_pixels = int(np.count_nonzero(box_red > 0))

        # Must have dominant white background over red text
        if bg_frac >= min_bg_frac and red_pixels >= min_text_px and bg_frac > red_frac * 1.5:
            conf = min(1.0, 0.5 + (red_pixels / 80.0) + (bg_frac * 0.3))
            items.append(("P1_white_red", x, y, w, h, conf))

    # Pass 2: 1D Horizontal Red Text Clusters
    k_remove_beams = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    red_no_beams = cv2.morphologyEx(red_text_mask, cv2.MORPH_OPEN, k_remove_beams)
    kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    red_dilated = cv2.dilate(red_no_beams, kernel_red)
    r_contours, _ = cv2.findContours(red_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in r_contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if is_ui_zone(rx, ry, rw, rh, sw, sh):
            continue
        if rw < 30 or rh < 6:
            continue
        raw_red_pixels = int(np.count_nonzero(red_no_beams[ry:ry+rh, rx:rx+rw] > 0))
        if raw_red_pixels < min_text_px:
            continue

        pad_x, pad_y = 8, 6
        bx = max(0, rx - pad_x)
        by = max(0, ry - pad_y)
        bw = min(sw - bx, rw + 2 * pad_x)
        bh = min(sh - by, rh + 2 * pad_y)

        if bw < min_w or bh < min_h:
            continue

        box_white = white_mask[by:by+bh, bx:bx+bw]
        box_red = red_text_mask[by:by+bh, bx:bx+bw]
        bg_frac = np.mean(box_white > 0)
        red_frac = np.mean(box_red > 0)

        if bg_frac >= min_bg_frac and bg_frac > red_frac * 1.5:
            conf = min(1.0, 0.5 + (raw_red_pixels / 80.0) + (bg_frac * 0.3))
            items.append(("P1_white_red", bx, by, bw, bh, conf))

    return items

def test_v2():
    test_files = [
        # Ground truth positives
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102718_567_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-16 144110.png",
        r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-16 165905.png",
        r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\.user_uploaded\media_1789633348578.png",
        
        # False positives to eliminate
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102722_890_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102721_099_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102531_344_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102533_920_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102538_532_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102540_964_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102543_442_P1_tier1_white_box_red_text_full.png",
    ]

    for p in test_files:
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        res = detect_white_box_red_text_v2(img)
        print(f"{os.path.basename(p)} -> {len(res)} item(s)")
        for r in res:
            print(f"   {r}")

if __name__ == "__main__":
    test_v2()
