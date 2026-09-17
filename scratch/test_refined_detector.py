import cv2
import numpy as np
import os
import glob
import sys

sys.path.insert(0, ".")

def is_ui_zone(x, y, w, h, sw, sh):
    if sw < 600 or sh < 400:
        return False
    cx = x + w // 2
    cy = y + h // 2
    # Top bar / latency graph
    if y < 45 or cy < 45:
        return True
    # Chat window (bottom-left)
    if cx < int(sw * 0.28) and cy > int(sh * 0.58):
        return True
    # Life globe
    if cx < 240 and cy > (sh - 240):
        return True
    # Mana globe
    if cx > (sw - 240) and cy > (sh - 240):
        return True
    # Bottom skill bar
    if cy > (sh - 110):
        return True
    # Minimap (top-right)
    if cx > (sw - 380) and cy < 380:
        return True
    # Buff bar (top-left)
    if cx < 350 and cy < 85:
        return True
    return False

def test_on_all():
    from src.loot_detector import LootDetector
    detector = LootDetector()
    
    test_files = [
        # Crucial ground truth positive cases
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102718_567_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-16 144110.png",
        r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-16 165905.png",
        r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-15 214538.png",
        r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-15 210942.png",
        r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\.user_uploaded\media_1789633348578.png",
        
        # Crucial false positive cases that MUST NOT be detected
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102722_890_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102721_099_P1_tier1_white_box_red_text_full.png",
        r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102531_344_P1_tier1_white_box_red_text_full.png",
    ]

    for p in test_files:
        if not os.path.exists(p):
            print(f"Skipping missing: {p}")
            continue
        img = cv2.imread(p)
        detected = detector.detect_loot(img)
        print(f"\nFile: {os.path.basename(p)}")
        print(f"  Detected count: {len(detected)}")
        for it in detected:
            print(f"    -> [P{it.priority}] {it.rule_id} at ({it.x}, {it.y}, {it.w}x{it.h}) conf={it.confidence:.2f}")

if __name__ == "__main__":
    test_on_all()
