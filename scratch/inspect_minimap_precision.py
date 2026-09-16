"""
Inspect Minimap Center & ROI Precision
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np

def inspect_minimap_center():
    capture_path = "debug_output/live_monitor2_capture.png"
    if not os.path.exists(capture_path):
        print("No live capture found.")
        return

    full_img = cv2.imread(capture_path)
    print(f"Full Screenshot Shape: {full_img.shape}")

    from src.minimap_extractor import MinimapExtractor
    extractor = MinimapExtractor()
    crop = extractor.extract_roi(full_img)

    cv2.imwrite("debug_output/inspect_minimap_crop_raw.png", crop)
    print(f"Raw Crop Shape: {crop.shape}")

    # Draw crosshair at exact center of crop
    c_h, c_w = crop.shape[:2]
    center_img = crop.copy()
    cv2.drawMarker(center_img, (c_w // 2, c_h // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.imwrite("debug_output/inspect_minimap_center_crosshair.png", center_img)
    print(f"Center pixel of minimap crop: ({c_w // 2}, {c_h // 2})")

inspect_minimap_center()
