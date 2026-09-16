"""
Inspect Red Square on Full Map Reference
"""
import cv2
import numpy as np
import os

map_path = "templates/full_map_reference.png"

if os.path.exists(map_path):
    img = cv2.imread(map_path)
    print(f"Loaded {map_path} shape: {img.shape}")

    # Convert to HSV or RGB to detect Red color
    # Red in BGR: B low, G low, R high
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    # Red mask condition: High Red, low Green & Blue
    red_mask = (r > 160) & (g < 80) & (b < 80)

    # Also check HSV red range (0-10 & 170-180)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    combined_red_mask = cv2.bitwise_or(mask1, mask2)

    red_y, red_x = np.where(combined_red_mask > 0)

    if len(red_x) > 0:
        min_x, max_x = int(np.min(red_x)), int(np.max(red_x))
        min_y, max_y = int(np.min(red_y)), int(np.max(red_y))
        print(f"[RED SQUARE DETECTED] Bounds: X [{min_x}..{max_x}], Y [{min_y}..{max_y}] | Width: {max_x - min_x}px, Height: {max_y - min_y}px")
    else:
        print("[WARNING] No red pixels detected with HSV mask. Testing BGR condition...")
        red_y, red_x = np.where(red_mask)
        if len(red_x) > 0:
            min_x, max_x = int(np.min(red_x)), int(np.max(red_x))
            min_y, max_y = int(np.min(red_y)), int(np.max(red_y))
            print(f"[RED SQUARE DETECTED] Bounds: X [{min_x}..{max_x}], Y [{min_y}..{max_y}] | Width: {max_x - min_x}px, Height: {max_y - min_y}px")
        else:
            print("[ERROR] No red square detected in image.")
else:
    print(f"Error: {map_path} does not exist.")
