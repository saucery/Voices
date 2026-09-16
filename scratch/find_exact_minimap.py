"""
Find Exact Minimap Box on 1080p Screenshot
"""
import cv2
import numpy as np
import os

def find_minimap_box():
    capture_path = "debug_output/live_monitor2_capture.png"
    if not os.path.exists(capture_path):
        print("No live capture found.")
        return

    full_img = cv2.imread(capture_path)
    h, w = full_img.shape[:2]

    # Crop top-right quarter of screen (top 0..400px, left 1400..1920px)
    top_right = full_img[0:400, 1400:w]
    cv2.imwrite("debug_output/top_right_quarter.png", top_right)
    print(f"Top-Right Quarter shape: {top_right.shape}")

    # Inspect cyan/blue wall color or minimap borders in top right
    # Path of Exile 2 minimap walls are bright Cyan / Blue (B > 150, G > 150, R < 100 or high contrast)
    hsv = cv2.cvtColor(top_right, cv2.COLOR_BGR2HSV)

    # Cyan/Blue mask
    mask_cyan = cv2.inRange(hsv, np.array([85, 100, 100]), np.array([105, 255, 255]))
    cv2.imwrite("debug_output/top_right_cyan_mask.png", mask_cyan)

    # Let's also check Canny edges on top_right
    gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite("debug_output/top_right_edges.png", edges)
    print("Saved top-right inspection images.")

find_minimap_box()
