import cv2
import numpy as np
import os
import glob
import sys
sys.path.insert(0, ".")
from src.loot_detector import LootDetector

def run_test():
    detector = LootDetector()
    all_debug_files = glob.glob(r"C:\Users\gregg\Voices\loot_debug\*_full.png")
    all_debug_files.extend(glob.glob(r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot*.png"))
    
    # Also user uploaded images
    user_up = glob.glob(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\.user_uploaded\*.png")
    all_debug_files.extend(user_up)

    print(f"Total test images found: {len(all_debug_files)}")
    for f in all_debug_files:
        if not os.path.exists(f):
            continue
        img = cv2.imread(f)
        if img is None:
            continue
        items = detector.detect_loot(img)
        print(f"\nImage: {os.path.basename(f)}")
        for it in items:
            print(f"  -> [P{it.priority}] {it.rule_id} at ({it.x}, {it.y}, {it.w}x{it.h}) conf={it.confidence:.2f}")

if __name__ == "__main__":
    run_test()
