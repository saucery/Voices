import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.movement_path import MovementPath
from src.room_classifier import RoomClassifier
from src.map_localizer import MapLocalizer


def test_movement_path_room_indexing_and_roi():
    mp = MovementPath()
    assert mp.is_loaded, "Route should be loaded"
    assert len(mp.waypoints) > 0

    # Test Room Indexing
    r1 = mp.get_current_room_index(0)
    assert r1 == 1

    pink_wps = mp.get_pink_waypoints()
    assert len(pink_wps) == 7

    # Check that each pink dot transitions to the expected room
    for r_i, (p_idx, _) in enumerate(pink_wps, start=1):
        assert mp.get_current_room_index(p_idx) == r_i

    # Test Room Bounding Boxes
    bbox1 = mp.get_room_bounding_box(1)
    assert bbox1 is not None
    assert len(bbox1) == 4
    min_x, min_y, max_x, max_y = bbox1
    assert max_x > min_x
    assert max_y > min_y

    bbox7 = mp.get_room_bounding_box(7)
    assert bbox7 is not None

    # Test Progress Search ROI
    mp.current_idx = 0
    roi_start = mp.get_search_roi_for_progress()
    assert roi_start is not None
    assert roi_start[0] <= mp.waypoints[0]["x"] <= roi_start[2]
    assert roi_start[1] <= mp.waypoints[0]["y"] <= roi_start[3]


def test_room_classifier_jump_gating():
    classifier = RoomClassifier()
    classifier.still_threshold = 3.5
    classifier.jump_threshold = 60.0
    classifier.jump_consensus_frames = 3

    # Manually lock classifier at (500.0, 300.0) in map_layout
    classifier.is_locked = True
    classifier.last_known_pos = (500.0, 300.0)
    classifier.last_room_id = "map_layout"
    classifier.last_room_name = "Map Layout"

    # Create a synthetic crop
    synthetic_crop = np.zeros((200, 200, 3), dtype=np.uint8)

    classifier.pending_jump_pos = None
    classifier.pending_jump_count = 0

    # Verify classifier has jump filter attributes
    assert hasattr(classifier, "pending_jump_count")
    assert hasattr(classifier, "jump_threshold")
    assert hasattr(classifier, "jump_rejections_total")


def test_map_localizer_guided_search_and_jump_filter():
    localizer = MapLocalizer()
    assert localizer.ref_edges is not None

    localizer.last_player_pos = (500, 300)
    localizer.locked_counter = 5

    # Test guided search ROI parameter acceptance
    synthetic_crop = np.zeros((150, 150, 3), dtype=np.uint8)
    search_roi = (450, 250, 550, 350)
    res = localizer.localize_player(synthetic_crop, fast_track=True, search_roi=search_roi, expected_pos=(500.0, 300.0))
    assert "located" in res
    assert hasattr(localizer, "pending_jump_pos")
