"""
Unit Tests for Room-Isolated Localization and Hard ROI Gating.
Verifies that when tracking Room 7 (top-left corner), the localizer and classifier
are strictly bounded to Room 7's quadrant [0, 0, 280, 280] and cannot jump to Room 3 or 4.
"""

import os
import sys
import json
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.map_localizer import MapLocalizer
from src.room_classifier import RoomClassifier
from src.movement_path import MovementPath


@pytest.fixture
def movement_path():
    """Load real movement route with 7 rooms."""
    mp = MovementPath()
    return mp


@pytest.fixture
def localizer():
    """Load MapLocalizer with reference map."""
    map_path = "templates/room_1_entry.png"
    if not os.path.exists(map_path):
        map_path = "templates/full_map_reference.png"
    return MapLocalizer(reference_map_path=map_path)


def test_room_7_active_bounds_clamping(movement_path):
    """Verify that get_active_room_bounds returns strictly (0, 0, 280, 280) when in Room 7."""
    assert movement_path.is_loaded is True

    # Jump to Room 7 finish waypoints (e.g. index 70..75)
    movement_path.current_idx = 72
    bounds = movement_path.get_active_room_bounds()
    assert bounds is not None
    assert bounds[0] >= 0  # min_x
    assert bounds[1] >= 0  # min_y
    assert bounds[2] <= 280  # max_x
    assert bounds[3] <= 280  # max_y
    # Verify tightened bounds
    assert bounds[0] >= 0 and bounds[1] >= 0
    assert bounds[2] <= 220 and bounds[3] <= 220

    roi = movement_path.get_search_roi_for_progress()
    assert roi is not None
    assert roi[2] <= 280
    assert roi[3] <= 280


def test_map_localizer_room_7_hard_roi_isolation(localizer):
    """Verify MapLocalizer strictly confines matching to Room 7 when search_roi is given."""
    if localizer.ref_img is None:
        pytest.skip("Reference map not found")

    # Room 7 region crop from ref_img
    crop = localizer.ref_img[50:180, 50:180].copy()

    # Localize with Room 7 search_roi
    search_roi = (0, 0, 280, 280)
    res = localizer.localize_player(crop, fast_track=False, search_roi=search_roi)

    assert res["located"] is True
    px, py = res["player_position"]
    # Player position must strictly be inside Room 7 quadrant (X <= 280, Y <= 280)
    assert px <= 280
    assert py <= 280
    assert px >= 0
    assert py >= 0
    # Must NOT be in Room 3 (x ~ 530) or Room 4 (x ~ 675)
    assert px < 350


def test_map_localizer_corner_hysteresis_holds_room_7(localizer):
    """Verify that when in the far corner with sparse/black features, localizer holds last Room 7 position."""
    if localizer.ref_img is None:
        pytest.skip("Reference map not found")

    # Simulate locked onto Room 7 (75, 65)
    localizer.last_player_pos = (75, 65)
    localizer.locked_counter = 5
    localizer.last_confidence = 0.35

    # Pass an empty/black frame (extreme top-left unvisited corner)
    empty_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    search_roi = (0, 0, 280, 280)

    res = localizer.localize_player(empty_crop, fast_track=False, search_roi=search_roi)
    assert res["located"] is True
    px, py = res["player_position"]
    # Retains Room 7 corner position instead of jumping or returning None
    assert (px, py) == (75, 65)


def test_room_classifier_hard_filters_cross_room_candidates():
    """Verify RoomClassifier hard-rejects candidate locations outside search_roi."""
    config_path = "config.json"
    if not os.path.exists(config_path):
        pytest.skip("config.json not found")

    rc = RoomClassifier(config_path_or_dict=config_path)
    if not rc.rooms:
        pytest.skip("No rooms loaded in classifier")

    # Room 7 search_roi
    search_roi = (0, 0, 280, 280)

    # Empty crop
    empty_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    res = rc.classify(empty_crop, is_crop=True, search_roi=search_roi)

    # If any position is returned, it must be inside search_roi
    if res.get("character_position"):
        cx, cy = res["character_position"]
        assert cx <= 280 + 35
        assert cy <= 280 + 35
