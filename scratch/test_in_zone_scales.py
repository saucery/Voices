"""
Test Realistic Scale Matching within Red Zone
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from src.map_localizer import MapLocalizer

def test_realistic_scales():
    crop = cv2.imread("debug_output/live_minimap_crop.png")
    localizer = MapLocalizer("templates/full_map_reference.png")
    crop_edges = localizer.extract_crop_features(crop)
    c_h, c_w = crop_edges.shape[:2]

    min_x, max_x, min_y, max_y = localizer.red_zone_bounds or (0, 273, 0, 270)

    best_in_zone_score = -1.0
    best_in_zone_loc = None
    best_in_zone_scale = None

    print(f"Searching within / near Red Zone: X [{min_x}..{max_x}], Y [{min_y}..{max_y}]")

    for scale in np.linspace(0.18, 0.40, 20):
        target_w = int(c_w * scale)
        target_h = int(c_h * scale)

        if target_h < localizer.ref_h and target_w < localizer.ref_w:
            resized_crop = cv2.resize(crop_edges, (target_w, target_h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(localizer.ref_edges, resized_crop, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            center_x = max_loc[0] + target_w // 2
            center_y = max_loc[1] + target_h // 2

            in_zone = (min_x - 30 <= center_x <= max_x + 30) and (min_y - 30 <= center_y <= max_y + 30)

            print(f"Scale {scale:.2f} ({target_w}x{target_h}) | Match ({center_x}, {center_y}) | Score: {max_val:.4f} | In Zone: {in_zone}")

            if in_zone and max_val > best_in_zone_score:
                best_in_zone_score = max_val
                best_in_zone_loc = (center_x, center_y)
                best_in_zone_scale = scale

    print(f"\n---> BEST IN-ZONE LOCALIZATION: Pos {best_in_zone_loc} | Score: {best_in_zone_score:.4f} | Scale: {best_in_zone_scale:.2f}")

test_realistic_scales()
