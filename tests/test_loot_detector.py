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


def test_loot_detector_white_red_synthetic():
    """Verifies detection of white box with red text on a synthetic test canvas."""
    detector = LootDetector()
    detector._set_default_rules()
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
    detector._set_default_rules()
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
        detector._set_default_rules()
        items = detector.detect_loot(img)
        assert len(items) >= 1
        assert items[0].rule_id == "tier1_white_box_red_text"


def test_loot_detector_priority_sorting():
    """Verifies that P1 White+Red items are sorted before P2 Gems and P3 Purple items."""
    detector = LootDetector()
    detector._set_default_rules()
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
    detector = LootDetector(config_file=None)
    detector._set_default_rules()
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
        detector = LootDetector(config_file=None)
        detector._set_default_rules()
        items = detector.detect_loot(img)
        gems = [it for it in items if it.rule_id == "gems_uncut_cut_olive"]
        assert len(gems) >= 2
        # Check that both Ruby (upper) and Sapphire (lower) were found
        y_coords = [g.y for g in gems]
        assert any(y < 200 for y in y_coords)   # Ruby
        assert any(y > 400 for y in y_coords)   # Sapphire


def test_loot_detector_save_pre_pickup_screenshot(tmp_path):
    """Verifies that LootDetector.save_pre_pickup_screenshot creates raw and annotated full-screen screenshots."""
    detector = LootDetector()
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas[100:130, 150:300] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (160, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    detected = detector.detect_loot(canvas)
    out_dir = str(tmp_path / "pre_pickup_test")

    raw_path = detector.save_pre_pickup_screenshot(canvas, all_items=detected, output_dir=out_dir)

    assert os.path.exists(raw_path)
    assert raw_path.endswith("_raw.png")
    annotated_path = raw_path.replace("_raw.png", "_annotated.png")
    assert os.path.exists(annotated_path)

    raw_img = cv2.imread(raw_path)
    ann_img = cv2.imread(annotated_path)
    assert raw_img is not None and raw_img.shape == (400, 600, 3)
    assert ann_img is not None and ann_img.shape == (400, 600, 3)


def test_loot_detector_add_template_item_rule(tmp_path):
    """Verifies that add_template_item_rule writes image template and adds template rule."""
    cfg_file = str(tmp_path / "loot_filter.json")
    detector = LootDetector()
    detector.config_file = cfg_file
    detector.save_config()

    crop = np.full((30, 100, 3), 200, dtype=np.uint8)
    cv2.putText(crop, "EXALT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    tmpl_dir = str(tmp_path / "loot_templates")
    rule = detector.add_template_item_rule(
        name="Exalted Orb",
        crop_img=crop,
        priority=1,
        threshold=0.55,
        templates_dir=tmpl_dir,
        save=True
    )

    assert rule["type"] == "template"
    assert rule["name"] == "Exalted Orb"
    assert os.path.exists(rule["template_file"])
    assert any(r["name"] == "Exalted Orb" for r in detector.rules)

    # Test reload from saved config
    reloaded = LootDetector(config_file=cfg_file)
    assert any(r["name"] == "Exalted Orb" for r in reloaded.rules)

    # Test updating / replacing the template crop image
    new_crop = np.full((35, 120, 3), 220, dtype=np.uint8)
    cv2.putText(new_crop, "EXALT V2", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    updated = reloaded.update_template_image(rule["id"], new_crop, save=True)
    assert updated is True

    # Verify updated image on disk
    saved_img = cv2.imread(rule["template_file"])
    assert saved_img is not None
    assert saved_img.shape == (35, 120, 3)

    # Test removing rule
    removed = reloaded.remove_rule(rule["id"], save=True)
    assert removed is True


def test_crop_color_auto_detection():
    """Verifies that LootDetector.analyze_crop_colors_and_geometry accurately classifies colors and geometry."""
    # 1. Orange background (BGR: 20, 100, 200) with black text (BGR: 10, 10, 10)
    orange_crop = np.full((32, 140, 3), (20, 100, 200), dtype=np.uint8)
    cv2.putText(orange_crop, "HEADHUNTER", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 2)
    res_orange = LootDetector.analyze_crop_colors_and_geometry(orange_crop)
    assert res_orange["bg_color"] == "orange"
    assert res_orange["text_color"] == "black"
    assert res_orange["width"] == 140
    assert res_orange["height"] == 32
    assert res_orange["aspect_ratio"] > 3.5

    # 2. White background (BGR: 240, 240, 240) with red text (BGR: 20, 20, 220)
    white_crop = np.full((30, 120, 3), (240, 240, 240), dtype=np.uint8)
    cv2.putText(white_crop, "DIVINE ORB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 220), 2)
    res_white = LootDetector.analyze_crop_colors_and_geometry(white_crop)
    assert res_white["bg_color"] == "white"
    assert res_white["text_color"] == "red"

    # 3. Purple background (BGR: 140, 20, 130) with white text (BGR: 240, 240, 240)
    purple_crop = np.full((28, 130, 3), (140, 20, 130), dtype=np.uint8)
    cv2.putText(purple_crop, "RAVEN REFLECTION", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 2)
    res_purple = LootDetector.analyze_crop_colors_and_geometry(purple_crop)
    assert res_purple["bg_color"] == "purple"
    assert res_purple["text_color"] == "white"

    # 4. Olive background (BGR: 20, 75, 85) with yellow text (BGR: 30, 210, 230)
    olive_crop = np.full((26, 110, 3), (20, 75, 85), dtype=np.uint8)
    cv2.putText(olive_crop, "RUBY GEM", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 210, 230), 2)
    res_olive = LootDetector.analyze_crop_colors_and_geometry(olive_crop)
    assert res_olive["bg_color"] == "olive"
    assert res_olive["text_color"] == "yellow"


def test_orange_box_detection():
    """Verifies that LootDetector detects orange unique loot boxes."""
    detector = LootDetector(config_file=None)
    rule = {
        "id": "test_unique_orange",
        "name": "Unique Orange Box",
        "enabled": True,
        "priority": 1,
        "type": "color_box",
        "bg_color": "orange",
        "text_color": "black",
        "min_width": 30,
        "max_width": 400,
        "min_height": 14,
        "max_height": 70,
        "min_aspect_ratio": 1.2,
        "min_bg_fraction": 0.30,
        "min_text_pixels": 10,
    }
    detector.rules = [rule]

    canvas = np.zeros((400, 500, 3), dtype=np.uint8)
    # Draw orange box at (120, 100, 150, 32) -> BGR: (20, 100, 200)
    canvas[100:132, 120:270] = (20, 100, 200)
    cv2.putText(canvas, "MAGEBLOOD", (130, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 2)

    items = detector.detect_loot(canvas)
    assert len(items) >= 1
    matched = items[0]
    assert matched.rule_id == "test_unique_orange"
    assert matched.priority == 1
    assert 115 <= matched.x <= 125
    assert 95 <= matched.y <= 105


def test_collect_loot_z_toggle_first_and_subsequent_cycles():
    """
    Verifies that collect_loot():
    1. First cycle: presses 'Z' twice before looting, then 'Z' once after looting (hiding labels).
    2. Subsequent cycle: presses 'Z' 3 times before looting (unhide, hide, unhide), then 'Z' once after looting.
    """
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.loot_z_toggle_delay_seconds = 0.001  # fast in tests

    z_presses: List[str] = []

    def mock_press_z():
        z_presses.append("z")

    nav._press_z_key = MagicMock(side_effect=mock_press_z)
    nav.locate_loot = MagicMock(return_value=None)  # No loot on screen

    # Cycle 1 (Initial: _loot_labels_hidden is False)
    assert nav._loot_labels_hidden is False
    picked1 = nav.collect_loot()
    assert picked1 == 0
    # Pre-loot: 2 presses (Z -> Z), Post-loot: 1 press (Z) -> Total 3 presses
    assert len(z_presses) == 3
    assert nav._loot_labels_hidden is True

    # Cycle 2 (Subsequent: _loot_labels_hidden is True)
    z_presses.clear()
    picked2 = nav.collect_loot()
    assert picked2 == 0
    # Pre-loot: 3 presses (Z -> Z -> Z), Post-loot: 1 press (Z) -> Total 4 presses
    assert len(z_presses) == 4
    assert nav._loot_labels_hidden is True


def test_collect_loot_approach_wait_default_1_1s():
    """Verifies that collect_loot defaults to 1.1s approach wait when picking up items."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.loot_z_toggle_delay_seconds = 0.001

    assert nav.loot_approach_wait_seconds == 1.1
    assert nav.loot_detector.approach_wait_seconds == 1.1

    # Simulate 1 loot item
    loot_positions = [(200, 200), None]
    nav.locate_loot = MagicMock(side_effect=lambda **kw: loot_positions.pop(0) if loot_positions else None)
    nav.move_mouse_inside_game = MagicMock(return_value=(200, 200))
    nav._wait_for_approach = MagicMock()
    nav._press_z_key = MagicMock()

    with patch("src.route_navigator.pydirectinput.click"), \
         patch("time.sleep", return_value=None):
        picked = nav.collect_loot()

    assert picked == 1
    nav._wait_for_approach.assert_called_once_with(1.1, reason="LOOT #1")


def test_ensure_loot_labels_visible_and_hide_loot_labels():
    """Verifies that ensure_loot_labels_visible unhides labels and hide_loot_labels hides them."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.loot_z_toggle_delay_seconds = 0.001

    z_presses: List[str] = []
    nav._press_z_key = MagicMock(side_effect=lambda: z_presses.append("z"))

    # Case 1: Already visible -> ensure_loot_labels_visible does nothing
    nav._loot_labels_hidden = False
    nav.ensure_loot_labels_visible()
    assert len(z_presses) == 0
    assert nav._loot_labels_hidden is False

    # Case 2: Hide labels -> presses Z, becomes True
    nav.hide_loot_labels()
    assert len(z_presses) == 1
    assert nav._loot_labels_hidden is True

    # Case 3: Redundant hide -> does not send duplicate Z
    nav.hide_loot_labels()
    assert len(z_presses) == 1
    assert nav._loot_labels_hidden is True

    # Case 4: Unhide labels -> presses Z, becomes False
    nav.ensure_loot_labels_visible()
    assert len(z_presses) == 2
    assert nav._loot_labels_hidden is False


def test_encounter_banner_and_sims_toggle_loot_visibility():
    """Verifies that zone routine steps unhide loot labels before search and hide them after."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.loot_z_toggle_delay_seconds = 0.001
    nav.banner_search_attempts = 1
    nav.banner_approach_wait_seconds = 0.0

    z_actions: List[str] = []

    def mock_ensure_visible():
        z_actions.append("ensure_visible")
        nav._loot_labels_hidden = False

    def mock_hide():
        z_actions.append("hide")
        nav._loot_labels_hidden = True

    nav.ensure_loot_labels_visible = MagicMock(side_effect=mock_ensure_visible)
    nav.hide_loot_labels = MagicMock(side_effect=mock_hide)
    nav.locate_encounter_banner = MagicMock(return_value=(500, 500))
    nav.move_mouse_inside_game = MagicMock(return_value=(500, 500))

    # Test click_encounter_banner step
    step = {
        "action": "click_encounter_banner",
        "approach_wait": 0.0,
        "search_attempts": 1,
        "verify_click": False,
        "reclick": False,
    }
    context = {}
    with patch("src.route_navigator.pydirectinput.click"):
        success = nav._execute_zone_routine_step(step, context, zone_label="TEST")

    assert success is True
    assert "ensure_visible" in z_actions
    assert "hide" in z_actions
    # Ensure visible was called before hide
    assert z_actions.index("ensure_visible") < z_actions.index("hide")



