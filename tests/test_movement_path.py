"""
Unit Tests for MovementPath Module
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.movement_path import MovementPath


def test_movement_path_missing_file():
    path_mgr = MovementPath(movement_file_path="paths/non_existent.json")
    assert path_mgr.is_configured is False
    assert len(path_mgr.get_waypoints()) == 0
    assert path_mgr.get_current_target() is None


def test_movement_path_sample_creation_and_loading(tmp_path):
    sample_file = str(tmp_path / "test_route.json")
    MovementPath.create_sample_file(sample_file)

    assert os.path.exists(sample_file)

    path_mgr = MovementPath(movement_file_path=sample_file)
    assert path_mgr.is_configured is True
    assert len(path_mgr.get_waypoints()) == 3

    # Check first target
    target = path_mgr.get_current_target()
    assert target is not None
    assert target["x"] == 200.0
    assert target["y"] == 300.0

    # Distance calculation
    dist = path_mgr.distance_to_target((200.0, 300.0))
    assert dist == 0.0
    assert path_mgr.is_target_reached((200.0, 300.0)) is True

    # Advance target
    next_target = path_mgr.advance()
    assert next_target is not None
    assert next_target["x"] == 350.0
    assert next_target["y"] == 280.0


def test_painted_route_extraction(tmp_path):
    import cv2
    import numpy as np

    # Create synthetic painted route image: 200x200
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)

    # Blue Start dot at (30, 30) (BGR: 255, 0, 0)
    cv2.circle(canvas, (30, 30), 5, (255, 0, 0), -1)

    # Green line from (30, 30) to (150, 150) (BGR: 0, 255, 0)
    cv2.line(canvas, (30, 30), (150, 150), (0, 255, 0), 3)

    # Red Finish dot at (150, 150) (BGR: 0, 0, 255)
    cv2.circle(canvas, (150, 150), 5, (0, 0, 255), -1)

    img_path = str(tmp_path / "synthetic_route.png")
    json_path = str(tmp_path / "extracted_route.json")
    cv2.imwrite(img_path, canvas)

    path_mgr = MovementPath(movement_file_path=img_path)
    assert path_mgr.is_configured is True
    wps = path_mgr.get_waypoints()
    assert len(wps) >= 5

    # Verify Start is near (30, 30)
    assert abs(wps[0]["x"] - 30) <= 2
    assert abs(wps[0]["y"] - 30) <= 2
    assert wps[0]["name"] == "Start"

    # Verify Finish is near (150, 150)
    assert abs(wps[-1]["x"] - 150) <= 2
    assert abs(wps[-1]["y"] - 150) <= 2
    assert wps[-1]["name"] == "Finish"


def test_gap_bridging_and_reload(tmp_path):
    import cv2
    import numpy as np

    # Create synthetic painted route with a 10px gap between green line and red finish
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(canvas, (20, 20), 5, (255, 0, 0), -1)  # Blue start
    cv2.line(canvas, (20, 20), (130, 130), (0, 255, 0), 3)  # Green line stops at (130, 130)
    cv2.circle(canvas, (142, 142), 5, (0, 0, 255), -1)  # Red finish at (142, 142) - 17px gap!

    img_path = str(tmp_path / "gap_route.png")
    json_path = str(tmp_path / "gap_route.json")
    cv2.imwrite(img_path, canvas)

    path_mgr = MovementPath(movement_file_path=json_path)
    success = path_mgr.load_from_painted_image(img_path, save_json_path=json_path)
    assert success is True
    assert path_mgr.is_configured is True
    assert len(path_mgr.get_waypoints()) >= 4

    # Test on-demand reload
    reload_ok = path_mgr.reload(image_path=img_path)
    assert reload_ok is True
    assert len(path_mgr.get_waypoints()) >= 4


