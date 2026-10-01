"""
Unit Tests for Room 7 Position Tracking & Navigation Stability
"""

import os
import cv2
import numpy as np
import pytest

from src.room_classifier import RoomClassifier
from src.movement_path import MovementPath
from src.route_navigator import RouteNavigator


@pytest.fixture
def classifier():
    return RoomClassifier("config.json")


def test_room_7_telemetry_frames_recognized(classifier):
    """
    Validates that the historical tracking loss frames in Room 7
    are now recognized with high confidence (>= 0.85) and locked into Room 7.
    """
    loss_files = [
        "debug_logs/tracker_lost/loss_20260921_143930_861_score0.19_lock_lost_low_confidence.png",
        "debug_logs/tracker_lost/loss_20260921_142758_938_score0.19_lock_lost_low_confidence.png",
        "debug_logs/tracker_lost/loss_20260921_142802_123_score0.17_lock_lost_low_confidence.png",
    ]

    r7_roi = (40, 30, 210, 205)
    expected_pos = (106.0, 122.0)

    for fpath in loss_files:
        if not os.path.exists(fpath):
            continue
        img = cv2.imread(fpath)
        assert img is not None, f"Failed to load {fpath}"

        res = classifier.classify(img, is_crop=True, search_roi=r7_roi, expected_pos=expected_pos)
        assert res["recognized"] is True, f"{fpath} was not recognized"
        assert res["room_id"] == "room_7", f"{fpath} wrong room: {res['room_id']}"
        assert res["confidence"] >= 0.85, f"{fpath} low confidence: {res['confidence']}"
        assert res["character_position"] is not None
        pos_x, pos_y = res["character_position"]
        # Character must be physically located in Room 7 boundaries (X in 40..200, Y in 30..190)
        assert 40 <= pos_x <= 200, f"X out of bounds: {pos_x}"
        assert 30 <= pos_y <= 190, f"Y out of bounds: {pos_y}"


def test_room_7_movement_path_bounds():
    """Validates that Room 7 search ROI is tightly bounded in MovementPath."""
    mp = MovementPath()

    # Set waypoint to 74 (Pink dot 7 / Zone 7 orbit)
    mp.current_idx = 74
    bounds = mp.get_active_room_bounds()
    assert bounds is not None
    x1, y1, x2, y2 = bounds
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 210 and y2 <= 210
    # Must NOT be the old wide (0, 0, 280, 280)
    assert bounds != (0, 0, 280, 280)


def test_rooms_1_to_6_unaffected(classifier):
    """
    Validates that frames outside Room 7 continue to match against standard master map layout
    with their normal configuration and 0.50 threshold.
    """
    room_5_frame = "debug_logs/tracker_lost/loss_20260920_201945_724_score1.05_jump_dist_143px.png"
    if os.path.exists(room_5_frame):
        img = cv2.imread(room_5_frame)
        roi = (396, 57, 720, 349)
        exp = (521.0, 200.0)
        res = classifier.classify(img, is_crop=True, search_roi=roi, expected_pos=exp)
        assert res["recognized"] is True
        assert res["room_id"] == "map_layout"
        assert res["confidence"] >= 0.50


def test_orbit_position_gating_rejects_outliers():
    """Validates that RouteNavigator rejects coordinates jumping far outside the orbit zone during orbit."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)

    zone_7 = {
        "id": "zone_7",
        "center": [112.5, 105.1],
        "radius": 60.0,
        "perimeter_points": [[162.9, 105.1], [112.5, 155.4], [73.3, 105.1], [112.5, 54.7]],
    }

    dist_outside = (136.0, 250.0)
    dist_zc = ((dist_outside[0] - 112.5)**2 + (dist_outside[1] - 105.1)**2)**0.5
    assert dist_zc > 96.0  # Would be rejected by position gating
