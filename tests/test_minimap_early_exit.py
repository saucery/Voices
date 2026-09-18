"""
Unit tests for Minimap Loot Icon Detection & Early Encounter Exit.
"""

import json
import os
import time
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loot_detector import LootDetector
from src.route_navigator import RouteNavigator
from src.movement_path import MovementPath
from src.player_tracker_visualizer import PlayerTrackerVisualizer


def test_detect_minimap_loot_icons_real_samples():
    """Test detect_minimap_loot_icons detects stars, diamonds, and circles on real samples."""
    detector = LootDetector()
    
    # 1. User uploaded minimap screenshot
    sample1_path = "C:/Users/gregg/.gemini/antigravity-ide/brain/bc444d8e-d778-41d6-a9f9-4d86757c5719/.user_uploaded/media_1789746758361.png"
    if os.path.exists(sample1_path):
        crop1 = cv2.imread(sample1_path)
        count1, icons1 = detector.detect_minimap_loot_icons(crop1)
        assert count1 >= 1
        assert len(icons1) == count1

    # 2. Synthetic empty minimap (dark terrain with player circle)
    empty_crop = np.zeros((280, 288, 3), dtype=np.uint8)
    empty_crop[:] = (15, 18, 20)
    # Draw player arrow at center
    cv2.circle(empty_crop, (144, 140), 6, (255, 200, 100), -1)
    # Draw gray terrain walls
    cv2.line(empty_crop, (20, 20), (260, 20), (120, 100, 110), 2)
    count_empty, icons_empty = detector.detect_minimap_loot_icons(empty_crop)
    assert count_empty == 0
    assert icons_empty == []

    # 3. Synthetic minimap with canonical templates pasted
    with_loot = empty_crop.copy()
    tpl_gold = cv2.imread("templates/minimap_icons/star_gold.png")
    tpl_silver = cv2.imread("templates/minimap_icons/star_silver.png")
    if tpl_gold is not None:
        gh, gw = tpl_gold.shape[:2]
        with_loot[180:180+gh, 200:200+gw] = tpl_gold
    if tpl_silver is not None:
        sh, sw = tpl_silver.shape[:2]
        with_loot[70:70+sh, 80:80+sw] = tpl_silver
        
    count_loot, icons_loot = detector.detect_minimap_loot_icons(with_loot)
    assert count_loot >= 2


def test_false_positive_minimap_screenshots_rejected():
    """Verify live run screenshots that previously triggered false positives return 0 icons."""
    detector = LootDetector()
    from src.minimap_extractor import MinimapExtractor
    extractor = MinimapExtractor()

    fp_files = [
        "loot_debug/minimap_early_exit_20260918_170953_fullscreen.png",
        "loot_debug/minimap_early_exit_20260918_171329_fullscreen.png",
        "loot_debug/minimap_early_exit_20260918_193907_fullscreen.png",
        "loot_debug/minimap_early_exit_20260918_194102_fullscreen.png",
        "loot_debug/minimap_early_exit_20260918_200319_fullscreen.png",
    ]
    for fp_path in fp_files:
        if os.path.exists(fp_path):
            full_img = cv2.imread(fp_path)
            mm_crop = extractor.extract_roi(full_img)
            count, icons = detector.detect_minimap_loot_icons(mm_crop)
            assert count == 0, f"False positive detected on {fp_path}: {icons}"


def test_live_loot_drop_screenshots_detected():
    """Verify real screenshots with dropped loot reliably detect loot icons."""
    detector = LootDetector()
    from src.minimap_extractor import MinimapExtractor
    extractor = MinimapExtractor()

    tp_files = [
        "loot_debug/minimap_early_exit_20260918_194308_fullscreen.png",
        "loot_debug/minimap_early_exit_20260918_194500_fullscreen.png",
        "loot_debug/minimap_early_exit_20260918_194707_fullscreen.png",
        "loot_debug/minimap_early_exit_20260918_200443_fullscreen.png",
        "loot_debug/loot_20260918_162730_495_P1_template_exaltedorb_4777_full.png",
        "loot_debug/loot_20260918_171003_643_P1_template_vaalorb_4814_full.png",
    ]
    for tp_path in tp_files:
        if os.path.exists(tp_path):
            full_img = cv2.imread(tp_path)
            mm_crop = extractor.extract_roi(full_img)
            count, icons = detector.detect_minimap_loot_icons(mm_crop)
            assert count >= 1, f"Expected loot icons on {tp_path}, found {count}"



def test_toggle_minimap_early_exit_and_persistence(tmp_path):
    """Test toggle_minimap_early_exit flips state and updates config.json."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"autopilot": {"minimap_early_exit_enabled": True}}), encoding="utf-8")

    nav = RouteNavigator(config_path=str(cfg_file), movement_path=MovementPath())
    assert nav.minimap_early_exit_enabled is True

    # Toggle OFF
    new_state = nav.toggle_minimap_early_exit(save_to_config=True)
    assert new_state is False
    assert nav.minimap_early_exit_enabled is False

    with open(str(cfg_file), "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["autopilot"]["minimap_early_exit_enabled"] is False

    # Toggle ON
    new_state2 = nav.toggle_minimap_early_exit(save_to_config=True)
    assert new_state2 is True
    assert nav.minimap_early_exit_enabled is True


def test_check_minimap_loot_drop():
    """Test check_minimap_loot_drop captures screen, extracts minimap, and calls detector."""
    nav = RouteNavigator(movement_path=MovementPath())
    
    mock_capt = MagicMock()
    mock_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Add gold star inside minimap ROI (top right)
    cv2.circle(mock_screen, (1800, 100), 6, (0, 215, 255), -1)
    mock_capt.capture.return_value = mock_screen
    nav.capturer = mock_capt

    loot_count, loot_icons = nav.check_minimap_loot_drop()
    assert loot_count >= 1
    assert len(loot_icons) == loot_count


def test_run_orbit_loop_early_exit_when_loot_detected():
    """Test _run_orbit_loop terminates early when t >= 25s and minimap loot is detected."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.minimap_early_exit_enabled = True
    nav.minimap_early_exit_min_seconds = 25.0
    nav.minimap_early_exit_check_interval = 0.001
    nav.step_duration = 0.001

    zone = {"id": "zone_1", "center": [100.0, 100.0], "perimeter_points": [[100, 100], [120, 100], [120, 120]]}
    nav.latest_pos = (100.0, 100.0)

    # Mock clock to simulate jumping from 0 to 26 seconds
    sim_time = [0.0]

    def fake_time():
        t = sim_time[0]
        sim_time[0] += 1.0  # Advance time by 1s on each check
        return t

    # Simulate minimap returning 2 loot icons after 25s
    nav.check_minimap_loot_drop = MagicMock(return_value=(2, [{"type": "star_gold"}]))
    nav.move_mouse_inside_game = MagicMock()
    nav.release_all_keys = MagicMock()

    with patch("time.time", side_effect=fake_time), \
         patch("time.sleep", return_value=None), \
         patch("src.route_navigator.pydirectinput", None):
        success = nav._run_orbit_loop(duration=50.0, best_zone=zone)

    assert success is True
    # Verify check_minimap_loot_drop was called once time reached >= 25s
    assert nav.check_minimap_loot_drop.call_count >= 1
    # Verify orbit finished before reaching the full 50s duration
    assert sim_time[0] < 50.0


def test_run_orbit_loop_does_not_early_exit_when_disabled():
    """Test _run_orbit_loop runs full duration when minimap_early_exit_enabled is False."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.minimap_early_exit_enabled = False
    nav.step_duration = 0.001

    zone = {"id": "zone_1", "center": [100.0, 100.0], "perimeter_points": []}
    nav.latest_pos = (100.0, 100.0)

    sim_time = [0.0]

    def fake_time():
        t = sim_time[0]
        sim_time[0] += 5.0
        return t

    nav.check_minimap_loot_drop = MagicMock(return_value=(5, []))
    nav.move_mouse_inside_game = MagicMock()
    nav.release_all_keys = MagicMock()

    with patch("time.time", side_effect=fake_time), \
         patch("time.sleep", return_value=None), \
         patch("src.route_navigator.pydirectinput", None):
        success = nav._run_orbit_loop(duration=50.0, best_zone=zone)

    assert success is True
    # check_minimap_loot_drop should NOT be called since feature is disabled
    assert nav.check_minimap_loot_drop.call_count == 0
    assert sim_time[0] >= 50.0


def test_visualizer_early_exit_button_click_and_render():
    """Test PlayerTrackerVisualizer renders [E] EARLY EXIT button and handles mouse clicks."""
    viz = PlayerTrackerVisualizer()
    viz.navigator.is_active = True
    viz.navigator.minimap_early_exit_enabled = True

    dummy_crop = np.zeros((280, 288, 3), dtype=np.uint8)
    dummy_result = {
        "navigation": viz.navigator.get_telemetry(None, 0.0, []),
        "room": {},
        "world_map": {},
        "reference_map": {},
    }

    # Render dashboard
    dashboard = viz.render_dashboard(dummy_crop, dummy_result)
    assert dashboard is not None
    assert viz.btn_early_exit_rect[2] > 0  # Width > 0

    ee_x, ee_y, ee_w, ee_h = viz.btn_early_exit_rect
    click_x = ee_x + ee_w // 2
    click_y = ee_y + ee_h // 2

    # Click button to toggle OFF
    viz._on_mouse(cv2.EVENT_LBUTTONDOWN, click_x, click_y, 0, None)
    assert viz.navigator.minimap_early_exit_enabled is False
    assert "MINIMAP EARLY EXIT: DISABLED" in viz.notification_msg

    # Click button to toggle ON
    viz._on_mouse(cv2.EVENT_LBUTTONDOWN, click_x, click_y, 0, None)
    assert viz.navigator.minimap_early_exit_enabled is True
    assert "MINIMAP EARLY EXIT: ENABLED" in viz.notification_msg
