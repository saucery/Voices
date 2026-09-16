"""
Test Cyan Wall Masking for High Precision Map Localization
"""
import cv2
import numpy as np
import os

def extract_cyan_wall_mask(img_bgr):
    """Extracts clean cyan/blue minimap wall lines."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Cyan/Blue HSV range (H: 80..120, S: 50..255, V: 50..255)
    mask1 = cv2.inRange(hsv, np.array([80, 50, 50]), np.array([125, 255, 255]))

    # Also check high blue vs red/green channel condition
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    mask2 = ((b > 120) & (g > 100) & (r < 120)).astype(np.uint8) * 255

    combined = cv2.bitwise_or(mask1, mask2)
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return cleaned

def test_cyan_wall_matching():
    capture_path = "debug_output/live_monitor2_capture.png"
    ref_path = "templates/full_map_reference.png"

    if not os.path.exists(capture_path) or not os.path.exists(ref_path):
        print("Missing capture or reference map.")
        return

    full_img = cv2.imread(capture_path)
    ref_bgr = cv2.imread(ref_path)

    from src.minimap_extractor import MinimapExtractor
    extractor = MinimapExtractor()
    crop = extractor.extract_roi(full_img)

    crop_walls = extract_cyan_wall_mask(crop)
    ref_walls = extract_cyan_wall_mask(ref_bgr)

    cv2.imwrite("debug_output/crop_cyan_walls.png", crop_walls)
    cv2.imwrite("debug_output/ref_cyan_walls.png", ref_walls)

    c_h, c_w = crop_walls.shape[:2]
    ref_h, ref_w = ref_walls.shape[:2]

    best_score = -1.0
    best_loc = (0, 0)
    best_target_wh = (0, 0)

    for scale in np.linspace(0.18, 0.55, 30):
        target_w = int(c_w * scale)
        target_h = int(c_h * scale)

        if target_h < ref_h and target_w < ref_w and target_h > 10 and target_w > 10:
            resized_crop = cv2.resize(crop_walls, (target_w, target_h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(ref_walls, resized_crop, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_target_wh = (target_w, target_h)

    top_left_x, top_left_y = best_loc
    tw, th = best_target_wh
    player_x = top_left_x + tw // 2
    player_y = top_left_y + th // 2

    print(f"--> CYAN WALL MATCH RESULT: Score: {best_score:.4f} | Size: ({tw}x{th}) | Player Center: ({player_x}, {player_y})")

test_cyan_wall_matching()
