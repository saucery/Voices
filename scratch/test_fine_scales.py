"""
Test Detailed Scales for Minimap Matching
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from src.map_localizer import MapLocalizer

def test_scales():
    crop = cv2.imread("debug_output/live_minimap_crop.png")
    localizer = MapLocalizer("templates/full_map_reference.png")

    crop_edges = localizer.extract_crop_features(crop)
    c_h, c_w = crop_edges.shape[:2]

    best_score = -1.0
    best_scale = 1.0
    best_loc = (0, 0)

    # Test fine scales from 0.15 to 0.75 in steps of 0.02
    for scale in np.linspace(0.12, 0.75, 35):
        target_w = int(c_w * scale)
        target_h = int(c_h * scale)

        if target_h < localizer.ref_h and target_w < localizer.ref_w and target_h > 10 and target_w > 10:
            resized_crop = cv2.resize(crop_edges, (target_w, target_h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(localizer.ref_edges, resized_crop, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            center_x = max_loc[0] + target_w // 2
            center_y = max_loc[1] + target_h // 2

            if max_val > 0.15:
                print(f"Scale: {scale:.3f} | Target: ({target_w}x{target_h}) | Score: {max_val:.4f} | Center: ({center_x}, {center_y})")

            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                best_loc = (center_x, center_y)

    print(f"\n---> BEST MATCH: Scale: {best_scale:.3f} | Score: {best_score:.4f} | Center: {best_loc}")

test_scales()
