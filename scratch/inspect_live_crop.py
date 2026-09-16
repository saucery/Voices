"""
Inspect Live Capture & Minimap Crop
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
import os
from src.minimap_extractor import MinimapExtractor
from src.map_localizer import MapLocalizer

def inspect_live_minimap():
    img_path = "debug_output/live_monitor2_capture.png"
    if not os.path.exists(img_path):
        print("No live capture found.")
        return

    full_img = cv2.imread(img_path)
    extractor = MinimapExtractor()
    crop = extractor.extract_roi(full_img)
    cv2.imwrite("debug_output/live_minimap_crop.png", crop)
    print(f"Saved minimap crop to debug_output/live_minimap_crop.png, shape: {crop.shape}")

    localizer = MapLocalizer("templates/full_map_reference.png")
    crop_edges = localizer.extract_crop_features(crop)
    cv2.imwrite("debug_output/live_crop_edges.png", crop_edges)
    cv2.imwrite("debug_output/ref_map_edges.png", localizer.ref_edges)
    print("Saved edges to debug_output/live_crop_edges.png and debug_output/ref_map_edges.png")

inspect_live_minimap()
