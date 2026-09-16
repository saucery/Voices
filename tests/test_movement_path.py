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


def test_cyan_and_white_dot_extraction_and_linking(tmp_path):
    """
    Verifies that:
    1. Cyan dots (SIM locations) and White dots (Loot locations) are detected.
    2. Blue start dot is not confused with Cyan dot.
    3. Red finish dot is not confused with Pink dot.
    4. Pink encounter waypoints correctly link nearest Cyan (sim_pos) and White (loot_pos).
    5. JSON export and import preserve sim_pos and loot_pos.
    """
    import cv2
    import numpy as np

    canvas = np.zeros((300, 300, 3), dtype=np.uint8)

    # Blue Start dot at (30, 30) (BGR: 255, 0, 0)
    cv2.circle(canvas, (30, 30), 6, (255, 0, 0), -1)

    # Green path from (30, 30) through (100, 100) to (250, 250)
    cv2.line(canvas, (30, 30), (100, 100), (0, 255, 0), 3)
    cv2.line(canvas, (100, 100), (250, 250), (0, 255, 0), 3)

    # Pink encounter dot at (100, 100) (BGR: 201, 174, 255)
    cv2.circle(canvas, (100, 100), 6, (201, 174, 255), -1)

    # Cyan SIM dot near pink at (115, 110) (BGR: 255, 255, 0)
    cv2.circle(canvas, (115, 110), 6, (255, 255, 0), -1)

    # White Loot dot near pink at (120, 95) (BGR: 255, 255, 255)
    cv2.circle(canvas, (120, 95), 6, (255, 255, 255), -1)

    # Red Finish dot at (250, 250) (BGR: 0, 0, 255)
    cv2.circle(canvas, (250, 250), 6, (0, 0, 255), -1)

    img_path = str(tmp_path / "triplet_route.png")
    json_path = str(tmp_path / "triplet_route.json")
    cv2.imwrite(img_path, canvas)

    path_mgr = MovementPath(movement_file_path=json_path)
    ok = path_mgr.load_from_painted_image(img_path, save_json_path=json_path)
    assert ok is True

    # Check detected dots
    assert len(path_mgr.get_pink_zones()) == 1
    assert len(path_mgr.get_cyan_zones()) == 1
    assert len(path_mgr.get_loot_zones()) == 1

    pink_wps = path_mgr.get_pink_waypoints()
    assert len(pink_wps) == 1
    idx, wp = pink_wps[0]

    # Verify positions linked to waypoint
    assert wp.get("pink_pos") is not None
    assert abs(wp["pink_pos"][0] - 100) <= 2
    assert abs(wp["pink_pos"][1] - 100) <= 2

    assert wp.get("sim_pos") is not None
    assert abs(wp["sim_pos"][0] - 115) <= 2
    assert abs(wp["sim_pos"][1] - 110) <= 2

    assert wp.get("loot_pos") is not None
    assert abs(wp["loot_pos"][0] - 120) <= 2
    assert abs(wp["loot_pos"][1] - 95) <= 2

    # Verify JSON roundtrip
    reloaded_mgr = MovementPath(movement_file_path=json_path)
    assert reloaded_mgr.is_configured is True
    reloaded_pinks = reloaded_mgr.get_pink_waypoints()
    assert len(reloaded_pinks) == 1
    _, r_wp = reloaded_pinks[0]
    assert r_wp.get("sim_pos") is not None
    assert r_wp.get("loot_pos") is not None
    assert abs(r_wp["sim_pos"][0] - 115) <= 2
    assert abs(r_wp["loot_pos"][0] - 120) <= 2



