import os
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
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
    assert purp[0].priority == 2


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
    """Verifies that P1 White+Red items are sorted before P2 Purple items."""
    detector = LootDetector()
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Draw purple item higher up on screen at y=100
    canvas[100:130, 100:280] = (200, 20, 200)
    cv2.putText(canvas, "RAVEN'S REFLECTION", (105, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw white/red item lower on screen at y=300
    canvas[300:330, 100:250] = (240, 240, 240)
    cv2.putText(canvas, "DIVINE ORB", (110, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)

    detected = detector.detect_loot(canvas)
    assert len(detected) == 2
    # Tier 1 (priority 1) should be first despite being lower on screen
    assert detected[0].priority == 1
    assert detected[0].rule_id == "tier1_white_box_red_text"
    assert detected[1].priority == 2
    assert detected[1].rule_id == "ravens_reflection_purple"


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
