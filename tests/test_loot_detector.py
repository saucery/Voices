import os
import sys
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loot_detector import LootDetector, LootItem
from src.route_navigator import RouteNavigator
from src.movement_path import MovementPath


def test_loot_detector_loads_config():
    """Verifies that LootDetector loads routines/loot_filter.json correctly."""
    detector = LootDetector(config_file="routines/loot_filter.json")
    assert detector.enabled is True
    assert len(detector.rules) >= 2
    rule_ids = [r["id"] for r in detector.rules]
    assert "tier1_white_box_red_text" in rule_ids
    assert "ravens_reflection_purple" in rule_ids


def test_loot_detector_white_red_synthetic():
    """Verifies detection of white box with red text on a synthetic test canvas."""
    detector = LootDetector()
    # Create 500x500 dark background image
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw white box at (100, 100, 150, 30)
    canvas[100:130, 100:250] = (240, 240, 240)
    # Draw red text inside white box
    cv2.putText(canvas, "DIVINE ORB", (110, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)
    
    detected = detector.detect_loot(canvas)
    assert len(detected) >= 1
    t1 = detected[0]
    assert t1.rule_id == "tier1_white_box_red_text"
    assert t1.priority == 1
    assert 95 <= t1.x <= 105
    assert 95 <= t1.y <= 105


def test_loot_detector_purple_synthetic():
    """Verifies detection of purple box (Raven's Reflection) on a synthetic test canvas."""
    detector = LootDetector()
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw purple/magenta box at (200, 200, 180, 32)
    # BGR for purple: B=200, G=20, R=200
    canvas[200:232, 200:380] = (200, 20, 200)
    # Draw white text inside
    cv2.putText(canvas, "RAVEN'S REFLECTION", (205, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    detected = detector.detect_loot(canvas)
    assert len(detected) >= 1
    purp = [d for d in detected if d.rule_id == "ravens_reflection_purple"]
    assert len(purp) == 1
    assert purp[0].priority == 3


def test_loot_detector_user_uploaded_samples_if_present():
    """Tests detection on user uploaded sample screenshots if they exist in workspace."""
    sample_crop = "C:/Users/gregg/.gemini/antigravity-ide/brain/bc444d8e-d778-41d6-a9f9-4d86757c5719/.user_uploaded/media_1789633348578.png"
    if os.path.exists(sample_crop):
        img = cv2.imread(sample_crop)
        detector = LootDetector()
        items = detector.detect_loot(img)
        assert len(items) >= 1
        assert items[0].rule_id == "tier1_white_box_red_text"


def test_loot_detector_priority_sorting():
    """Verifies that P1 White+Red items are sorted before P2 Gems and P3 Purple items."""
    detector = LootDetector()
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Draw purple item higher up on screen at y=100
    canvas[100:130, 100:280] = (200, 20, 200)
    cv2.putText(canvas, "RAVEN'S REFLECTION", (105, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw gem item in middle at y=200
    canvas[200:232, 100:200] = (20, 75, 85)
    cv2.putText(canvas, "RUBY", (105, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 210, 230), 2)

    # Draw white/red item lower on screen at y=300
    canvas[300:330, 100:250] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (110, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    detected = detector.detect_loot(canvas)
    assert len(detected) == 3
    # Tier 1 (priority 1) should be first despite being lowest on screen
    assert detected[0].priority == 1
    assert detected[0].rule_id == "tier1_white_box_red_text"
    assert detected[1].priority == 2
    assert detected[1].rule_id == "gems_uncut_cut_olive"
    assert detected[2].priority == 3
    assert detected[2].rule_id == "ravens_reflection_purple"


def test_route_navigator_collect_loot_integration():
    """Verifies that RouteNavigator.collect_loot() clicks detected items using LootDetector."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    # Create dummy canvas with white/red item
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    canvas[100:130, 100:250] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (110, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    with patch.object(nav, "_get_capturer") as mock_get_cap, \
         patch("src.route_navigator.pydirectinput") as mock_pdi:
        mock_cap = MagicMock()
        mock_cap.capture.return_value = canvas
        mock_get_cap.return_value = mock_cap

        # Call locate_loot
        pos = nav.locate_loot()
        assert pos is not None
        assert 100 <= pos[0] <= 250
        assert 100 <= pos[1] <= 130

        # Call collect_loot with limit=1 and no wait
        picked = nav.collect_loot(max_pickups=1, approach_wait=0.0)
        assert picked == 1
        assert mock_pdi.click.call_count == 1
    nav.stop()


def test_loot_detector_save_debug_screenshot(tmp_path):
    """Verifies that LootDetector.save_debug_screenshot creates full and crop image files in target dir."""
    detector = LootDetector()
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    # Draw white box with red text
    canvas[100:130, 150:300] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (160, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    detected = detector.detect_loot(canvas)
    assert len(detected) >= 1
    item = detected[0]

    out_dir = str(tmp_path / "test_loot_debug")
    full_path, crop_path = detector.save_debug_screenshot(canvas, item, all_items=detected, output_dir=out_dir)

    assert os.path.exists(full_path)
    assert os.path.exists(crop_path)
    assert full_path.endswith("_full.png")
    assert crop_path.endswith("_crop.png")

    full_img = cv2.imread(full_path)
    crop_img = cv2.imread(crop_path)
    assert full_img is not None
    assert full_img.shape == canvas.shape
    assert crop_img is not None
    assert crop_img.shape[0] > 0 and crop_img.shape[1] > 0


def test_route_navigator_saves_loot_debug_screenshot(tmp_path):
    """Verifies that RouteNavigator.locate_loot triggers save_debug_screenshot into configured debug dir."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.save_loot_debug_screenshots = True
    out_dir = str(tmp_path / "nav_loot_debug")
    nav.loot_debug_dir = out_dir

    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas[100:130, 150:300] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (160, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    with patch.object(nav, "_get_capturer") as mock_get_cap:
        mock_cap = MagicMock()
        mock_cap.capture.return_value = canvas
        mock_get_cap.return_value = mock_cap

        pos = nav.locate_loot()
        assert pos is not None

        # Verify debug files created in out_dir
        files = os.listdir(out_dir)
        assert len(files) == 2  # 1 full, 1 crop
        assert any(f.endswith("_full.png") for f in files)
        assert any(f.endswith("_crop.png") for f in files)
    nav.stop()


def test_loot_detector_rejects_red_background_boxes():
    """Verifies that currency boxes with RED background and WHITE text (e.g. Chance Shards) are NOT detected as Tier 1."""
    detector = LootDetector()
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    # Draw red background box at (200, 200, 120, 25) - BGR for salmon/red: (60, 70, 220)
    canvas[200:225, 200:320] = (60, 70, 220)
    # Draw white text inside
    cv2.putText(canvas, "2x CHANCE SHARD", (205, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    detected = detector.detect_loot(canvas)
    # Should NOT be classified as Tier 1 White Box / Red Text
    t1_items = [d for d in detected if d.rule_id == "tier1_white_box_red_text"]
    assert len(t1_items) == 0


def test_loot_detector_ui_exclusion_zones():
    """Verifies that text/boxes in chat, minimap, or status bars are ignored."""
    detector = LootDetector()
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Draw white box with red text inside Chat window area (x=100, y=800)
    canvas[800:830, 100:250] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (110, 822), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    # Draw white box with red text inside Bottom Mana Globe area (x=1750, y=980)
    canvas[980:1010, 1750:1880] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (1760, 1000), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    detected = detector.detect_loot(canvas)
    assert len(detected) == 0

    # Draw white box with red text in play area (x=800, y=400)
    canvas[400:430, 800:950] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (810, 422), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    detected_play = detector.detect_loot(canvas)
    assert len(detected_play) == 1
    assert detected_play[0].x >= 790
    assert detected_play[0].y >= 390


def test_loot_detector_gems_synthetic():
    """Verifies detection of uncut and cut gems with olive background and yellow/lime text & border."""
    detector = LootDetector(config_file="routines/loot_filter.json")
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)

    # Draw olive gem box at (150, 150, 90, 32)
    # BGR for dark olive: (20, 75, 85)
    canvas[150:182, 150:240] = (20, 75, 85)
    # Draw yellow/lime text "RUBY" inside
    cv2.putText(canvas, "RUBY", (160, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 210, 230), 2)

    detected = detector.detect_loot(canvas)
    assert len(detected) >= 1
    gem = detected[0]
    assert gem.rule_id == "gems_uncut_cut_olive"
    assert gem.priority == 2
    assert 140 <= gem.x <= 160
    assert 140 <= gem.y <= 160


def test_loot_detector_gems_screenshot_if_present():
    """Verifies detection of Ruby and Sapphire on the gems sample screenshot if present."""
    sample_path = r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png"
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        detector = LootDetector(config_file="routines/loot_filter.json")
        items = detector.detect_loot(img)
        gems = [it for it in items if it.rule_id == "gems_uncut_cut_olive"]
        assert len(gems) >= 2
        # Check that both Ruby (upper) and Sapphire (lower) were found
        y_coords = [g.y for g in gems]
        assert any(y < 200 for y in y_coords)   # Ruby
        assert any(y > 400 for y in y_coords)   # Sapphire



