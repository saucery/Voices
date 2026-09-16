"""
Unit Tests for RouteNavigator Module
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.movement_path import MovementPath
from src.route_navigator import RouteNavigator


def test_compute_wasd_keys_cardinal():
    # Target is directly East (Right)
    assert RouteNavigator.compute_wasd_keys((100, 100), (150, 100)) == ["d"]
    # Target is directly West (Left)
    assert RouteNavigator.compute_wasd_keys((100, 100), (50, 100)) == ["a"]
    # Target is directly South (Down)
    assert RouteNavigator.compute_wasd_keys((100, 100), (100, 150)) == ["s"]
    # Target is directly North (Up)
    assert RouteNavigator.compute_wasd_keys((100, 100), (100, 50)) == ["w"]


def test_compute_wasd_keys_diagonal():
    # Target is North-East
    assert sorted(RouteNavigator.compute_wasd_keys((100, 100), (150, 50))) == ["d", "w"]
    # Target is South-East
    assert sorted(RouteNavigator.compute_wasd_keys((100, 100), (150, 150))) == ["d", "s"]
    # Target is South-West
    assert sorted(RouteNavigator.compute_wasd_keys((100, 100), (50, 150))) == ["a", "s"]
    # Target is North-West
    assert sorted(RouteNavigator.compute_wasd_keys((100, 100), (50, 50))) == ["a", "w"]


def test_navigator_waypoint_advancement(tmp_path):
    sample_file = str(tmp_path / "nav_route.json")
    MovementPath.create_sample_file(sample_file)

    path_mgr = MovementPath(movement_file_path=sample_file)
    navigator = RouteNavigator(movement_path=path_mgr, arrival_threshold=15.0)

    navigator.is_active = True

    # Waypoint 0 is at (200, 300)
    # Character is far away at (100, 300) -> should move 'd'
    telemetry = navigator.update((100.0, 300.0))
    assert telemetry["is_active"] is True
    assert telemetry["target_index"] == 0
    assert "d" in telemetry["held_keys"]

    # Character arrives close at (195.0, 300.0) (within 15px) -> advances to Waypoint 1 (350, 280)
    telemetry2 = navigator.update((195.0, 300.0))
    assert telemetry2["target_index"] == 1
    assert telemetry2["distance_to_target"] > 100.0

    # Stop navigator
    navigator.stop()
    assert navigator.is_active is False
    assert len(navigator.held_keys) == 0


def test_dynamic_resync_when_ahead():
    # Route with 5 sequential waypoints
    path_mgr = MovementPath()
    path_mgr.waypoints = [
        {"index": 0, "name": "WP 0", "x": 100.0, "y": 100.0, "action": "walk"},
        {"index": 1, "name": "WP 1", "x": 150.0, "y": 100.0, "action": "walk"},
        {"index": 2, "name": "WP 2", "x": 200.0, "y": 100.0, "action": "walk"},
        {"index": 3, "name": "WP 3", "x": 250.0, "y": 100.0, "action": "walk"},
        {"index": 4, "name": "WP 4", "x": 300.0, "y": 100.0, "action": "walk"},
    ]
    path_mgr.current_idx = 0
    path_mgr.is_loaded = True
    path_mgr.arrival_distance = 15.0

    navigator = RouteNavigator(movement_path=path_mgr, arrival_threshold=15.0)
    navigator.is_active = True

    # Character starts near WP 0
    t0 = navigator.update((95.0, 100.0))
    # Arrived at WP 0, targeting WP 1
    assert t0["target_index"] == 1

    # Character jumped or ran ahead to WP 3 (at x=252, y=100)
    # The navigator must NOT get stuck at WP 1! It must resync to WP 3 / 4!
    t_ahead = navigator.update((252.0, 100.0))
    assert t_ahead["target_index"] in [3, 4]
    assert t_ahead["target_index"] > 1


def test_stuck_recovery_and_skip():
    path_mgr = MovementPath()
    path_mgr.waypoints = [
        {"index": 0, "name": "Start", "x": 100.0, "y": 100.0},
        {"index": 1, "name": "Corner Stalker", "x": 150.0, "y": 100.0},
        {"index": 2, "name": "Open Hall", "x": 200.0, "y": 100.0},
        {"index": 3, "name": "Finish", "x": 250.0, "y": 100.0},
    ]
    navigator = RouteNavigator(movement_path=path_mgr)
    navigator.start_at_pink_dot = 0
    path_mgr.current_idx = 1
    navigator.is_active = True
    navigator.last_known_pos = (110.0, 100.0)
    navigator.latest_pos = (145.0, 100.0)
    navigator.last_held_keys = ["d"]

    # Trigger stuck recovery
    navigator._execute_stuck_recovery(reason="Stuck at Corner Stalker")

    # Current WP must have been skipped from 1 -> 2
    assert navigator.movement_path.current_idx == 2
    assert navigator.latest_recovery_event is not None
    assert "Skipped WP #1" in navigator.latest_recovery_event
    assert navigator.movement_path.get_current_target()["name"] == "Open Hall"

    # Telemetry should expose recovery_event
    telem = navigator.get_telemetry(navigator.movement_path.get_current_target(), 50.0, ["d"])
    assert telem.get("recovery_event") == navigator.latest_recovery_event


def test_manual_skip_waypoint():
    path_mgr = MovementPath()
    path_mgr.waypoints = [
        {"index": 0, "name": "WP 0", "x": 10.0, "y": 10.0},
        {"index": 1, "name": "WP 1", "x": 20.0, "y": 20.0},
        {"index": 2, "name": "WP 2", "x": 30.0, "y": 30.0},
    ]
    path_mgr.current_idx = 0
    path_mgr.is_loaded = True

    navigator = RouteNavigator(movement_path=path_mgr)
    navigator.is_active = True

    next_wp = navigator.skip_current_waypoint()
    assert next_wp is not None
    assert next_wp["index"] == 1
    assert navigator.movement_path.current_idx == 1


def test_f4_pause_and_resume_preserves_position():
    """Verifies that pressing F4 pauses autopilot (releases keys, preserves waypoint) and resumes upon second press."""
    path_mgr = MovementPath()
    path_mgr.waypoints = [
        {"index": 0, "name": "Start", "x": 10.0, "y": 10.0},
        {"index": 1, "name": "Midpoint", "x": 50.0, "y": 10.0},
        {"index": 2, "name": "Finish", "x": 100.0, "y": 10.0},
    ]
    path_mgr.is_loaded = True

    navigator = RouteNavigator(movement_path=path_mgr)
    navigator.start_at_pink_dot = 0
    path_mgr.current_idx = 1
    navigator.is_active = True
    navigator.held_keys = {"d"}

    # 1. First F4 press -> Pause
    is_paused = navigator.toggle_pause()
    assert is_paused is True
    assert navigator.is_paused is True
    assert navigator.is_active is True
    assert len(navigator.held_keys) == 0  # Keys released
    assert navigator.movement_path.current_idx == 1  # Waypoint preserved

    # Telemetry should reflect paused state
    telem = navigator.update(current_pos=(25.0, 10.0))
    assert telem["is_paused"] is True
    assert telem["held_keys"] == []
    assert "PAUSED" in telem["status_message"]
    assert "F4" in telem["status_message"]

    # 2. Advance time past debounce and press F4 again -> Resume
    navigator._last_f4_time = 0.0
    is_paused = navigator.toggle_pause()
    assert is_paused is False
    assert navigator.is_paused is False
    assert navigator.is_active is True
    # Still targeting Midpoint (index 1)
    assert navigator.movement_path.current_idx == 1

    # Navigation tick should now compute keys to WP 1 (x=50, character at x=25 -> 'd')
    telem = navigator.update(current_pos=(25.0, 10.0))
    assert telem["is_paused"] is False
    assert "d" in telem["held_keys"]



