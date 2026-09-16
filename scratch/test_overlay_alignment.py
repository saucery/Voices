"""
Overlay Minimap Crop on Reference Map at Matched Position
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from src.map_localizer import MapLocalizer
from src.minimap_extractor import MinimapExtractor

def test_overlay_alignment():
    capture_path = "debug_output/live_monitor2_capture.png"
    if not os.path.exists(capture_path):
        print("No live capture found.")
        return

    full_img = cv2.imread(capture_path)
    extractor = MinimapExtractor()
    crop = extractor.extract_roi(full_img)

    localizer = MapLocalizer("templates/full_map_reference.png")
    c_h, c_w = crop.shape[:2]
    crop_edges = localizer.extract_crop_features(crop)

    ref_bgr = cv2.imread("templates/full_map_reference.png")
    ref_h, ref_w = ref_bgr.shape[:2]

    # Test scales
    best_score = -1.0
    best_loc = (0, 0)
    best_target_wh = (0, 0)

    min_x, max_x, min_y, max_y = localizer.red_zone_bounds or (0, ref_w, 0, ref_h)

    for scale in np.linspace(0.18, 0.55, 30):
        target_w = int(c_w * scale)
        target_h = int(c_h * scale)

        if target_h < ref_h and target_w < ref_w:
            resized_crop = cv2.resize(crop_edges, (target_w, target_h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(localizer.ref_edges, resized_crop, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            center_x = max_loc[0] + target_w // 2
            center_y = max_loc[1] + target_h // 2

            in_zone = (min_x - 30 <= center_x <= max_x + 30) and (min_y - 30 <= center_y <= max_y + 30)

            if in_zone and max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_target_wh = (target_w, target_h)

    print(f"Best Matched Top-Left: {best_loc}, Size: {best_target_wh}, Score: {best_score:.4f}")

    top_left_x, top_left_y = best_loc
    tw, th = best_target_wh
    player_x = top_left_x + tw // 2
    player_y = top_left_y + th // 2

    print(f"Calculated Player Center: ({player_x}, {player_y})")

    # Draw overlay of crop on reference map
    canvas = ref_bgr.copy()
    resized_color_crop = cv2.resize(crop, (tw, th))

    # Blend cropped minimap onto reference map at matched location
    roi = canvas[top_left_y:top_left_y+th, top_left_x:top_left_x+tw]
    blended = cv2.addWeighted(roi, 0.4, resized_color_crop, 0.6, 0)
    canvas[top_left_y:top_left_y+th, top_left_x:top_left_x+tw] = blended

    # Draw player center cross
    cv2.circle(canvas, (player_x, player_y), 10, (0, 255, 255), 2)
    cv2.circle(canvas, (player_x, player_y), 4, (0, 0, 255), -1)

    cv2.imwrite("debug_output/overlay_minimap_alignment.png", canvas)
    print("Saved overlay image to debug_output/overlay_minimap_alignment.png")

test_overlay_alignment()
