"""
Tests for EnemyDetector
Verifies red enemy health bar detection and proximity calculation.
"""

import os
import cv2
import numpy as np
import pytest

from src.enemy_detector import EnemyDetector


def test_enemy_detector_on_sample_image():
    """Verifies that enemy health bars are correctly detected on the user sample image."""
    user_img_path = r"C:\Users\gregg\.gemini\antigravity-ide\brain\6021aaf6-ceab-42b6-b014-ed48115be00b\.user_uploaded\media_1789995385256.png"
    if not os.path.exists(user_img_path):
        pytest.skip("User sample image not found at expected path")

    detector = EnemyDetector()
    img = cv2.imread(user_img_path)
    # Character in this crop is roughly around (210, 260)
    char_pos = (210.0, 260.0)

    result = detector.detect(img, character_center=char_pos, near_threshold_px=100.0)

    assert result["detected"] is True
    assert result["count"] >= 2
    # Check that the near enemy is detected and marked as near
    assert result["has_enemy_near"] is True
    assert result["nearest_distance"] < 80.0
    assert result["nearest_enemy"] is not None
    assert result["nearest_enemy"]["is_near"] is True


def test_enemy_detector_synthetic_bars():
    """Tests detection with synthetically generated enemy health bars at known coordinates."""
    # Create dark game-like canvas (600x800)
    canvas = np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)

    # Character at center (400, 300)
    char_center = (400.0, 300.0)

    # Enemy 1: Distant enemy at (100, 100), distance ~360px
    # Draw red health bar: width=40, height=5, BGR = (25, 25, 160)
    canvas[98:103, 80:120] = (25, 25, 160)

    detector = EnemyDetector()
    res1 = detector.detect(canvas, character_center=char_center, near_threshold_px=200.0)
    assert res1["detected"] is True
    assert res1["count"] == 1
    assert res1["has_enemy_near"] is False
    assert res1["nearest_distance"] > 250.0

    # Enemy 2: Close enemy at (420, 340), distance ~45px
    # Draw red health bar: width=45, height=5, BGR = (28, 30, 165)
    canvas[338:343, 398:443] = (28, 30, 165)

    res2 = detector.detect(canvas, character_center=char_center, near_threshold_px=200.0)
    assert res2["detected"] is True
    assert res2["count"] == 2
    assert res2["has_enemy_near"] is True
    assert res2["nearest_distance"] < 60.0


def test_enemy_detector_rejects_non_bars():
    """Ensures non-bar shapes or non-red colors are not detected as enemies."""
    canvas = np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)

    # Add a square red box (height=30, width=30) -> should be rejected by aspect ratio and height
    canvas[100:130, 100:130] = (20, 20, 160)

    # Add a green bar (width=50, height=5) -> should be rejected by red color filter
    canvas[200:205, 100:150] = (20, 160, 20)

    # Add a blue bar (width=50, height=5)
    canvas[250:255, 100:150] = (160, 20, 20)

    detector = EnemyDetector()
    res = detector.detect(canvas)
    assert res["detected"] is False
    assert res["count"] == 0
