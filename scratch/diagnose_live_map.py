"""
Diagnose Live Map Localization & Red Zone Boundaries
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
from src.screen_capturer import ScreenCapturer
from src.map_localizer import MapLocalizer

def diagnose_live_map():
    capturer = ScreenCapturer(monitor_idx=2)
    localizer = MapLocalizer("templates/full_map_reference.png")

    print(f"Red Zone Bounds: {localizer.red_zone_bounds}")

    screenshot = capturer.capture()
    if screenshot is None:
        print("[ERROR] Could not capture Monitor 2 screenshot.")
        return

    # Save screenshot for inspection
    cv2.imwrite("debug_output/live_monitor2_capture.png", screenshot)

    # Localize player
    res = localizer.localize_player(screenshot)
    print(f"\nLocalization Result: {res}")

    if localizer.ref_img is not None and res.get("player_position"):
        out_path = localizer.render_player_location(res, "debug_output/live_player_location.png")
        print(f"Rendered player location map to: {out_path}")

diagnose_live_map()
