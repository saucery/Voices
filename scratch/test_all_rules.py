import cv2
import numpy as np
import os
import sys

sys.path.insert(0, ".")
from src.loot_detector import LootDetector

def run_suite():
    detector = LootDetector(config_file="routines/loot_filter.json")
    
    test_files = [
        ("Gems Screenshot (Ruby / Sapphire)", r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png", 2),
        ("Potent Liquid Ferocity", r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102718_567_P1_tier1_white_box_red_text_full.png", 1),
        ("Concentrated Liquid Isolation", r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-16 144110.png", 1),
        ("Liquid Isolation + Divine Orb", r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-16 165905.png", 2),
        ("Raven's Reflection Purple", r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-15 214538.png", 2),
        ("False Pos 1: Lava Spark / Div Chat", r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102722_890_P1_tier1_white_box_red_text_full.png", 0),
        ("False Pos 2: Chance Shards Red Box", r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102721_099_P1_tier1_white_box_red_text_full.png", 0),
        ("False Pos 3: Chat window text", r"C:\Users\gregg\Voices\loot_debug\loot_20260917_102531_344_P1_tier1_white_box_red_text_full.png", 0),
    ]

    all_passed = True
    for label, path, expected_min in test_files:
        if not os.path.exists(path):
            print(f"[SKIP] File not found: {path}")
            continue
        img = cv2.imread(path)
        items = detector.detect_loot(img)
        print(f"\nTest: {label}")
        print(f"  Found {len(items)} items (expected at least {expected_min}):")
        for it in items:
            print(f"    -> [P{it.priority}] {it.rule_id} at ({it.x}, {it.y}, {it.w}x{it.h}) conf={it.confidence:.2f}")

        if expected_min == 0 and len(items) != 0:
            print(f"  [FAIL] Expected 0 items, got {len(items)}")
            all_passed = False
        elif expected_min > 0 and len(items) < expected_min:
            print(f"  [FAIL] Expected at least {expected_min} items, got {len(items)}")
            all_passed = False
        else:
            print("  [PASS]")

    print(f"\nOverall Result: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    run_suite()
