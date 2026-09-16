"""
Inspect Top Right Quarter Minimap Content
"""
import cv2
import numpy as np

def analyze_top_right():
    tr = cv2.imread("debug_output/top_right_quarter.png")
    h, w = tr.shape[:2]

    # Find bounding box of active minimap region in top_right
    gray = cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY)
    # Threshold for dark UI vs minimap content
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Find contours in top_right
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 100 and ch > 100:
            print(f"Found Minimap Contour: Box ({x}, {y}, {cw}, {ch}) | Center: ({x + cw//2}, {y + ch//2})")

            # Crop exact minimap box
            minimap_box = tr[y:y+ch, x:x+cw]
            cv2.imwrite("debug_output/exact_minimap_box.png", minimap_box)

            # Draw center cross on exact minimap box
            box_cross = minimap_box.copy()
            cv2.drawMarker(box_cross, (cw // 2, ch // 2), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.imwrite("debug_output/exact_minimap_center.png", box_cross)

analyze_top_right()
