import cv2
import numpy as np

def detect_gems_precise(screen):
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(screen)
    sw, sh = screen.shape[1], screen.shape[0]

    # 1. Dark Olive Background Mask for Gems
    olive_gem_bg_mask = (
        (g >= 36) & (g <= 125) &
        (r >= 42) & (r <= 135) &
        (b < 55) &
        (g.astype(np.int16) >= b.astype(np.int16) + 10) &
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 42) &
        (hsv[:, :, 1] >= 60) &
        (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] <= 135)
    ).astype(np.uint8) * 255

    # 2. Gem Text & Border Mask
    gem_text_border_mask = (
        (r >= 140) & (g >= 125) &
        (b < 120) &
        (g.astype(np.int16) >= b.astype(np.int16) + 25) &
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 40) &
        (hsv[:, :, 1] >= 60) &
        (hsv[:, :, 2] >= 140)
    ).astype(np.uint8) * 255

    min_w = 40
    max_w = 260
    min_h = 16
    max_h = 55
    min_ar = 1.3
    min_bg_frac = 0.35
    min_text_px = 12

    items = []

    # Pass 1: Horizontal Olive Box Contours (1D horizontal closing)
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    bg_closed = cv2.morphologyEx(olive_gem_bg_mask, cv2.MORPH_CLOSE, k_horiz)
    contours, _ = cv2.findContours(bg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_w or w > max_w or h < min_h or h > max_h:
            continue
        if (w / max(1.0, float(h))) < min_ar:
            continue

        box_olive = olive_gem_bg_mask[y:y+h, x:x+w]
        box_text = gem_text_border_mask[y:y+h, x:x+w]
        bg_frac = np.mean(box_olive > 0)
        text_pixels = int(np.count_nonzero(box_text > 0))

        if bg_frac >= min_bg_frac and text_pixels >= min_text_px:
            conf = min(1.0, 0.65 + (text_pixels / 120.0) + (bg_frac * 0.25))
            items.append(("gem_uncut_cut", x, y, w, h, conf))

    # Pass 2: 1D Horizontal Text Cluster Projection
    k_text_cluster = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    text_dilated = cv2.dilate(gem_text_border_mask, k_text_cluster)
    t_contours, _ = cv2.findContours(text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in t_contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw < 20 or rh < 6:
            continue
        raw_text_px = int(np.count_nonzero(gem_text_border_mask[ry:ry+rh, rx:rx+rw] > 0))
        if raw_text_px < min_text_px:
            continue

        pad_x = 6
        pad_y = 6
        bx = max(0, rx - pad_x)
        by = max(0, ry - pad_y)
        bw = min(sw - bx, rw + 2 * pad_x)
        bh = min(sh - by, rh + 2 * pad_y)

        if bw < min_w or bw > max_w or bh < min_h or bh > max_h:
            continue

        box_olive = olive_gem_bg_mask[by:by+bh, bx:bx+bw]
        bg_frac = np.mean(box_olive > 0)

        if bg_frac >= min_bg_frac:
            conf = min(1.0, 0.65 + (raw_text_px / 120.0) + (bg_frac * 0.25))
            items.append(("gem_uncut_cut", bx, by, bw, bh, conf))

    # NMS
    final_items = []
    items.sort(key=lambda it: -it[4])
    for it in items:
        x, y, w, h, conf = it[1], it[2], it[3], it[4], it[5]
        overlap = False
        for f in final_items:
            fx, fy, fw, fh = f[1], f[2], f[3], f[4]
            # IoU
            ix1 = max(x, fx)
            iy1 = max(y, fy)
            ix2 = min(x + w, fx + fw)
            iy2 = min(y + h, fy + fh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (w * h) + (fw * fh) - inter
                if inter / union > 0.30:
                    overlap = True
                    break
        if not overlap:
            final_items.append(it)

    return final_items

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
res = detect_gems_precise(img)
print(f"Total gems detected: {len(res)}")
for r in res:
    print(f"  -> {r[0]} at ({r[1]}, {r[2]}, {r[3]}x{r[4]}) conf={r[5]:.2f}")

vis = img.copy()
for r in res:
    x, y, w, h = r[1], r[2], r[3], r[4]
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(vis, "GEM", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\exact_gems_result.png", vis)
print("Saved exact_gems_result.png")
