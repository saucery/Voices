import cv2
import numpy as np
import os
import glob
import sys
sys.path.insert(0, ".")
from src.loot_detector import LootDetector

def inspect_image(img_path):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
    img = cv2.imread(img_path)
    detector = LootDetector()
    items = detector.detect_loot(img)
    print(f"\n--- Inspecting {os.path.basename(img_path)} (size: {img.shape}) ---")
    print(f"Total detected items: {len(items)}")
    for i, it in enumerate(items):
        print(f"  #{i+1}: [P{it.priority}] {it.rule_id} at rect=({it.x}, {it.y}, {it.w}, {it.h}) conf={it.confidence:.2f}")
        # inspect what pixels are inside this rect
        crop = img[it.y:it.y+it.h, it.x:it.x+it.w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(crop)
        white_cnt = np.count_nonzero((r > 175) & (g > 170) & (b > 170) & (hsv[:,:,1] < 65) & (hsv[:,:,2] > 170))
        red_cnt = np.count_nonzero((r > 155) & (r.astype(np.int16) - g.astype(np.int16) > 55) & (r.astype(np.int16) - b.astype(np.int16) > 55) & (g < 110) & (b < 110))
        total = crop.shape[0] * crop.shape[1]
        print(f"       Crop info: white_px={white_cnt}/{total} ({white_cnt/total*100:.1f}%), red_px={red_cnt}/{total} ({red_cnt/total*100:.1f}%)")

if __name__ == "__main__":
    inspect_image(r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102722_890_P1_tier1_white_box_red_text_full.png")
    inspect_image(r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102718_567_P1_tier1_white_box_red_text_full.png")
    inspect_image(r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102721_099_P1_tier1_white_box_red_text_full.png")
    inspect_image(r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102531_344_P1_tier1_white_box_red_text_full.png")
