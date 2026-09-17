"""
Unit and integration tests for Pink Dot Encounter and Sim Selection Navigation.
Verifies:
1. Pink Dot color/shape extraction from route image and waypoint action tagging.
2. Character movement pause for pink_dot_stop_seconds upon reaching pink dot.
3. Sim selection sequence: sim1 -> wait 2s -> sim3 -> wait 2s -> sim2.
4. Fallback sequence when no sims available: click encounter_banner -> click right mouse -> hold middle mouse 4 seconds.
5. Navigation integration: waypoint arrival triggers interaction once and advances to next waypoint.
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.movement_path import MovementPath
from src.route_navigator import RouteNavigator
from src.stop_handler import stop_handler


@pytest.fixture
def synthetic_route_with_pink_dot(tmp_path):
    """Creates a synthetic route image with Blue start, Green line, Pink dot, and Red finish."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # Blue Start dot at (50, 100) (BGR: (255, 0, 0))
    cv2.circle(img, (50, 100), 7, (255, 0, 0), -1)

    # Green line from (50, 100) to (250, 100) (BGR: (0, 255, 0))
    cv2.line(img, (50, 100), (250, 100), (0, 255, 0), 5)

    # Pink dot at (150, 100) - HSV ~ (150, 180, 220) -> BGR: (180, 50, 220)
    # Magenta/Pink: R high, B high, G low
    cv2.circle(img, (150, 100), 8, (200, 40, 220), -1)

    # Red Finish dot at (250, 100) (BGR: (0, 0, 255))
    cv2.circle(img, (250, 100), 7, (0, 0, 255), -1)

    img_path = str(tmp_path / "route_with_pink.png")
    cv2.imwrite(img_path, img)
    return img_path


def test_pink_dot_extraction_from_image(synthetic_route_with_pink_dot, tmp_path):
    """Verifies that pink dots on the route are detected and waypoints are tagged with action 'pink_encounter'."""
    out_json = str(tmp_path / "pink_route.json")
    path_mgr = MovementPath()
    success = path_mgr.load_from_painted_image(
        image_path=synthetic_route_with_pink_dot,
        spacing=20.0,
        save_json_path=out_json,
    )

    assert success is True
    assert len(path_mgr.waypoints) >= 4

    # Verify pink zones metadata
    pinks = path_mgr.get_pink_zones()
    assert len(pinks) == 1
    pz = pinks[0]
    assert abs(pz["x"] - 150) <= 5
    assert abs(pz["y"] - 100) <= 5
    assert pz["area"] >= 8.0

    # Verify waypoint action tagging
    pink_wps = [wp for wp in path_mgr.waypoints if wp.get("action") == "pink_encounter"]
    assert len(pink_wps) == 1
    assert "Pink" in pink_wps[0]["name"]
    assert pink_wps[0]["pink_pos"] is not None

    # Verify JSON persistence
    assert os.path.exists(out_json)
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "pink_zones" in data
    assert len(data["pink_zones"]) == 1


def test_pink_dot_sim_priority_clicks(tmp_path):
    """
    Tests the sim selection sequence when sims are detected:
    Character stops -> checks sim1 -> clicks -> waits 2s -> checks sim3 -> clicks -> waits 2s -> checks sim2 -> clicks.
    Encounter banner fallback should NOT be triggered.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.1,  # short for fast testing
            "sim_match_threshold": 0.50,
            "middle_click_hold_seconds": 0.1,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    mock_movement_path.waypoints = [{"index": 0, "name": "Pink", "x": 150, "y": 100, "action": "pink_encounter"}]

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True

    click_log = []
    seen = set()

    def mock_locate_sim(sim_key):
        # Found on initial query for each sim, then disappears on verification
        if sim_key not in seen:
            seen.add(sim_key)
            return (300, 400)
        return None

    def mock_click():
        click_log.append("left_click")

    navigator.locate_sim_template = MagicMock(side_effect=mock_locate_sim)
    navigator.locate_encounter_banner = MagicMock(return_value=(500, 500))
    navigator.collect_loot = MagicMock(return_value=0)
    navigator.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 100, y or 100))

    with patch("src.route_navigator.pydirectinput.click", side_effect=mock_click), \
         patch("src.route_navigator.pydirectinput.rightClick") as mock_rclick, \
         patch("src.route_navigator.pydirectinput.mouseDown") as mock_mdown, \
         patch("time.sleep", return_value=None):

        result = navigator.execute_pink_dot_interaction(target=mock_movement_path.waypoints[0])

    assert result is True
    # Verify sim query sequence: sim1 -> sim3 -> sim2
    sim_calls = [c[0][0] for c in navigator.locate_sim_template.call_args_list if c[0][0] in ["sim1", "sim3", "sim2"]]
    assert sim_calls == ["sim1", "sim1", "sim3", "sim3", "sim2", "sim2"]

    # 3 clicks performed (one for each sim, verified confirmed)
    assert len(click_log) == 3

    # Fallback should NOT be triggered
    navigator.locate_encounter_banner.assert_not_called()
    mock_rclick.assert_not_called()
    mock_mdown.assert_not_called()


def test_pink_dot_sim_double_check_when_still_visible(tmp_path):
    """
    Verifies that during pink dot encounter, if a sim is still visible after clicking,
    the double-check logic re-clicks it.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.01,
            "sim_approach_wait_seconds": 0.0,
            "sim_verify_delay_seconds": 0.01,
            "sim_max_click_attempts": 2,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    mock_movement_path.waypoints = [{"index": 0, "name": "Pink", "x": 150, "y": 100, "action": "pink_encounter"}]

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True

    click_count = 0
    def mock_click():
        nonlocal click_count
        click_count += 1

    sim1_calls = 0
    def mock_locate(key, threshold=None):
        nonlocal sim1_calls
        if key == "sim1":
            sim1_calls += 1
            if sim1_calls <= 2:
                return (350, 250)  # Found on 1st search AND 2nd verification check (still visible!)
            return None  # Gone on 3rd check
        return None

    navigator.locate_sim_template = MagicMock(side_effect=mock_locate)
    navigator.collect_loot = MagicMock(return_value=0)
    navigator.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 100, y or 100))

    with patch("src.route_navigator.pydirectinput.click", side_effect=mock_click), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):

        result = navigator.execute_pink_dot_interaction(target=mock_movement_path.waypoints[0])

    assert result is True
    # Initial click + double-check re-click = 2 clicks
    assert click_count == 2


def test_pink_dot_no_sims_fallback_to_banner_and_middle_click(tmp_path):
    """
    Tests fallback sequence when no sims are detected:
    Character stops -> checks sims (none found) -> clicks encounter banner -> clicks right mouse button ->
    presses and holds middle mouse button for configured seconds (4.0s) -> releases.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.05,
            "middle_click_hold_seconds": 0.1,  # short for fast testing
            "banner_search_attempts": 1,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    mock_movement_path.waypoints = [{"index": 0, "name": "Pink", "x": 150, "y": 100, "action": "pink_encounter"}]

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True

    navigator.locate_sim_template = MagicMock(return_value=None)  # No sims available
    navigator.locate_encounter_banner = MagicMock(return_value=(600, 300))
    navigator.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 200, y or 200))

    actions = []

    def log_click():
        actions.append("banner_click")

    def log_rclick():
        actions.append("right_click")

    def log_mdown(button=None):
        actions.append(f"down_{button}")

    def log_mup(button=None):
        actions.append(f"up_{button}")

    with patch("src.route_navigator.pydirectinput.click", side_effect=log_click), \
         patch("src.route_navigator.pydirectinput.rightClick", side_effect=log_rclick), \
         patch("src.route_navigator.pydirectinput.mouseDown", side_effect=log_mdown), \
         patch("src.route_navigator.pydirectinput.mouseUp", side_effect=log_mup), \
         patch("time.sleep", return_value=None):

        result = navigator.execute_pink_dot_interaction(target=mock_movement_path.waypoints[0])

    assert result is True
    # Encounter banner searched and clicked
    assert navigator.locate_encounter_banner.called
    assert "banner_click" in actions

    # Right mouse button clicked once
    assert "right_click" in actions

    # Middle button pressed down and up
    assert "down_middle" in actions
    assert "up_middle" in actions

    # Verify execution order: banner_click -> right_click -> down_middle -> up_middle
    banner_i = actions.index("banner_click")
    rclick_i = actions.index("right_click")
    mdown_i = actions.index("down_middle")
    mup_i = actions.index("up_middle")
    assert banner_i < rclick_i < mdown_i < mup_i


def test_navigator_triggers_pink_interaction_at_pink_waypoint(tmp_path):
    """Verifies that navigator.update() triggers pink dot encounter upon arriving at a pink_encounter waypoint."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.01,
            "middle_click_hold_seconds": 0.01,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    wp0 = {"index": 0, "name": "Start", "x": 50.0, "y": 100.0, "action": "walk"}
    wp1 = {"index": 1, "name": "Pink Marker", "x": 150.0, "y": 100.0, "action": "pink_encounter"}
    wp2 = {"index": 2, "name": "Finish", "x": 250.0, "y": 100.0, "action": "interact"}
    mock_movement_path.waypoints = [wp0, wp1, wp2]
    mock_movement_path.current_idx = 1
    mock_movement_path.get_current_target.return_value = wp1
    mock_movement_path.update_to_nearest.return_value = wp1
    mock_movement_path.advance.return_value = wp2

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        arrival_threshold=15.0,
        config_path=str(cfg_file),
    )
    navigator.is_active = True
    navigator.execute_pink_dot_interaction = MagicMock(return_value=True)

    # Player arrives at WP1 (pink dot at 150, 100)
    telem = navigator.update(current_pos=(150.0, 100.0))

    # Interaction should have fired
    assert navigator.execute_pink_dot_interaction.call_count == 1
    assert 1 in navigator.interacted_pink_dots
    # Route should advance to next waypoint
    mock_movement_path.advance.assert_called_once()


def test_pink_dot_fallback_initiates_yellow_orbit(tmp_path):
    """Verifies that after holding middle mouse for 4s, the bot initiates inside-orbit around the nearest yellow marker."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.01,
            "middle_click_hold_seconds": 0.01,
            "banner_search_attempts": 1,
            "orbit_duration_seconds": 50.0,
        }
    }))

    yellow_zone = {
        "id": "zone_1",
        "center": [160.0, 110.0],
        "radius": 25.0,
        "duration": 50.0,
        "perimeter_points": [[160.0, 95.0], [175.0, 110.0], [160.0, 125.0], [145.0, 110.0]],
    }

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    mock_movement_path.get_orbit_zones.return_value = [yellow_zone]
    mock_movement_path.waypoints = [{"index": 0, "name": "Pink", "x": 150.0, "y": 100.0, "action": "pink_encounter"}]

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True
    navigator.latest_pos = (150.0, 100.0)

    navigator.locate_sim_template = MagicMock(return_value=None)
    navigator.locate_encounter_banner = MagicMock(return_value=(500, 300))
    navigator.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 200, y or 200))

    with patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.rightClick"), \
         patch("src.route_navigator.pydirectinput.mouseDown"), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):

        result = navigator.execute_pink_dot_interaction(target=mock_movement_path.waypoints[0])

    assert result is True
    # Character must enter orbiting state inside yellow zone
    assert navigator.is_orbiting is True
    assert navigator.current_orbit_zone["id"] == "zone_1"
    assert navigator.orbit_duration == 50.0
    assert len(navigator.orbit_perimeter_pts) == 4


def test_collect_loot_multi_pickup(tmp_path):
    """Verifies that collect_loot() scans and clicks on multiple detected loot items until none remain."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "loot_pickup_wait_seconds": 0.01,
            "max_loot_pickups": 5,
        }
    }))

    navigator = RouteNavigator(config_path=str(cfg_file))
    navigator.is_active = True

    # Simulate finding loot 1, loot 2, then no more loot
    loot_coords = [(400, 300), (550, 420), None]
    navigator.locate_loot = MagicMock(side_effect=loot_coords)
    navigator.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 100, y or 100))

    clicked_count = 0

    def mock_click():
        nonlocal clicked_count
        clicked_count += 1

    with patch("src.route_navigator.pydirectinput.click", side_effect=mock_click), \
         patch("time.sleep", return_value=None):

        picked = navigator.collect_loot()

    assert picked == 2
    assert clicked_count == 2
    assert navigator.locate_loot.call_count == 3


def test_orbit_completion_triggers_loot_and_advancement(tmp_path):
    """Verifies that when yellow zone orbiting completes, collect_loot() is invoked and green light is awaited before advancing route."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"autopilot": {"wait_for_loot_confirmation": True}}))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    next_wp = {"index": 2, "name": "Next Pink Dot", "x": 250.0, "y": 100.0, "action": "pink_encounter"}
    mock_movement_path.advance.return_value = next_wp

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True
    navigator.is_orbiting = True
    navigator.orbit_duration = 10.0
    navigator.orbit_start_time = time.time() - 15.0  # Duration expired!
    navigator.collect_loot = MagicMock(return_value=1)

    telem = navigator.update(current_pos=(160.0, 100.0))

    # Orbit should have finished
    assert navigator.is_orbiting is False
    # Loot collection should have run
    navigator.collect_loot.assert_called_once()
    # Bot should be stopped waiting for green light verification
    assert navigator.waiting_for_green_light is True
    assert telem.get("waiting_for_green_light") is True
    mock_movement_path.advance.assert_not_called()

    # Now simulate user giving GREEN LIGHT:
    consumed = navigator.give_green_light()
    assert consumed is True
    assert navigator.waiting_for_green_light is False
    mock_movement_path.advance.assert_called_once()


def test_loot_confirmation_disabled_advances_immediately(tmp_path):
    """Verifies that when wait_for_loot_confirmation is False, bot advances immediately after loot collection."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"autopilot": {"wait_for_loot_confirmation": False}}))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    next_wp = {"index": 2, "name": "Next Pink Dot", "x": 250.0, "y": 100.0, "action": "pink_encounter"}
    mock_movement_path.advance.return_value = next_wp

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True
    navigator.is_orbiting = True
    navigator.orbit_duration = 10.0
    navigator.orbit_start_time = time.time() - 15.0
    navigator.collect_loot = MagicMock(return_value=1)

    telem = navigator.update(current_pos=(160.0, 100.0))

    assert navigator.is_orbiting is False
    navigator.collect_loot.assert_called_once()
    assert navigator.waiting_for_green_light is False
    mock_movement_path.advance.assert_called_once()
    assert telem["target"] == next_wp


def test_anti_stuck_disabled_during_pink_dot_interaction(tmp_path):
    """
    Verifies that while the character stops at a pink dot and searches for sims/banner,
    anti-stuck recovery is strictly disabled and stuck counter remains 0.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.1,
            "banner_search_attempts": 1,
            "middle_click_hold_seconds": 0.05,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    wp8 = {"index": 8, "name": "Pink Dot #8", "x": 134.0, "y": 371.0, "action": "pink_encounter"}
    wp9 = {"index": 9, "name": "Green Waypoint #9", "x": 180.0, "y": 371.0, "action": "walk"}
    mock_movement_path.waypoints = [wp8, wp9]
    mock_movement_path.current_idx = 8
    mock_movement_path.get_current_target.return_value = wp8
    mock_movement_path.get_orbit_zones.return_value = []

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True
    navigator.latest_pos = (134.0, 371.0)
    navigator.last_progress_pos = (134.0, 371.0)
    navigator.last_progress_time = time.time() - 10.0  # Would trigger stuck if not interacting!
    navigator.stuck_counter = 5

    # Mock template searches so it runs fallback sequence
    navigator.locate_sim_template = MagicMock(return_value=None)
    navigator.locate_encounter_banner = MagicMock(return_value=(200, 200))
    navigator.move_mouse_inside_game = MagicMock(return_value=(200, 200))
    navigator._execute_stuck_recovery = MagicMock()

    was_interacting_states = []

    def track_sleep(duration):
        # Record state during simulated waits
        was_interacting_states.append(navigator.is_interacting)
        assert navigator.stuck_counter == 0

    with patch("time.sleep", side_effect=track_sleep), \
         patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.rightClick"), \
         patch("src.route_navigator.pydirectinput.mouseDown"), \
         patch("src.route_navigator.pydirectinput.mouseUp"):

        result = navigator.execute_pink_dot_interaction(target=wp8)

    assert result is True
    # Verify is_interacting was True during all waits/searches
    assert len(was_interacting_states) > 0
    assert all(was_interacting_states)

    # After completion, is_interacting is reset to False and stuck_counter is cleared
    assert navigator.is_interacting is False
    assert navigator.stuck_counter == 0
    # Stuck recovery should NEVER have been called!
    navigator._execute_stuck_recovery.assert_not_called()
    assert mock_movement_path.current_idx == 8


def test_anti_stuck_active_during_green_path_and_orbit():
    """
    Verifies that anti-stuck is active when traveling along green path or orbiting in yellow area:
    - On green path: skips waypoint and backtracks.
    - In yellow orbit: switches orbit perimeter point without skipping route waypoint.
    """
    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    wp0 = {"index": 0, "name": "Path WP 0", "x": 50.0, "y": 100.0, "action": "walk"}
    wp1 = {"index": 1, "name": "Path WP 1", "x": 100.0, "y": 100.0, "action": "walk"}
    wp2 = {"index": 2, "name": "Path WP 2", "x": 200.0, "y": 100.0, "action": "walk"}
    mock_movement_path.waypoints = [wp0, wp1, wp2]
    mock_movement_path.current_idx = 1
    mock_movement_path.get_current_target.return_value = wp1

    navigator = RouteNavigator(movement_path=mock_movement_path)
    navigator.start_at_pink_dot = 0
    mock_movement_path.current_idx = 1
    mock_movement_path.get_current_target.return_value = wp1
    navigator.is_active = True
    navigator.is_interacting = False
    navigator.latest_pos = (100.0, 100.0)
    navigator.last_progress_pos = (100.0, 100.0)

    # 1. Test green path stuck recovery
    navigator.is_orbiting = False
    navigator._execute_stuck_recovery(reason="Stuck on Green Path")

    # Should have skipped WP 1 -> target now WP 2
    assert mock_movement_path.current_idx == 2
    assert "Skipped WP #1" in navigator.latest_recovery_event

    # 2. Test yellow area orbit stuck recovery
    navigator.is_orbiting = True
    navigator.orbit_perimeter_pts = [[100.0, 100.0], [150.0, 100.0], [150.0, 150.0]]
    navigator.orbit_point_idx = 0
    mock_movement_path.current_idx = 2  # Keep at WP 2

    navigator._execute_stuck_recovery(reason="Stuck in Yellow Area")

    # Should advance orbit perimeter point (0 -> 1) WITHOUT advancing movement_path.current_idx
    assert navigator.orbit_point_idx == 1
    assert mock_movement_path.current_idx == 2  # Route waypoint untouched!
    assert "ORBIT RECOVERY: Unstuck in yellow zone" in navigator.latest_recovery_event


def test_update_to_nearest_never_jumps_over_pink_dot():
    """
    Verifies that dynamic resynchronization in MovementPath NEVER jumps over
    an unvisited pink_encounter waypoint, even when the character's physical position
    is closer to a waypoint ahead of it.
    """
    path = MovementPath()
    path.arrival_distance = 25.0
    wp16 = {"index": 16, "name": "Waypoint 16", "x": 286.0, "y": 284.0, "action": "walk"}
    wp17 = {"index": 17, "name": "Waypoint 17", "x": 301.0, "y": 271.0, "action": "walk"}
    wp18 = {"index": 18, "name": "Pink Marker (pink_4)", "x": 316.0, "y": 257.0, "action": "pink_encounter"}
    wp19 = {"index": 19, "name": "Orbit Zone (zone_4)", "x": 336.0, "y": 259.0, "action": "orbit"}
    path.waypoints = [wp16, wp17, wp18, wp19]
    path.current_idx = 1  # At WP 17

    # Character position at (327, 273):
    # dist to WP 18 (316, 257) = 19.41 px
    # dist to WP 19 (336, 259) = 16.64 px (physically closer!)
    char_pos = (327.0, 273.0)

    target = path.update_to_nearest(char_pos)

    # Must clamp to WP 18 (Pink Marker) and NEVER jump directly to WP 19!
    assert target is not None
    assert target["index"] == 18
    assert target["action"] == "pink_encounter"
    assert path.current_idx == 2


def test_yellow_zone_checks_sims_first():
    """
    Verifies that execute_yellow_zone_interaction attempts to find and select Sims first
    before attempting the banner click sequence.
    """
    navigator = RouteNavigator()
    navigator.is_active = True

    clicked_actions = []

    def mock_detect_sims(prefix="[SIM]"):
        clicked_actions.append("sims_selected")
        return ["sim1"]

    navigator._detect_and_click_sims = mock_detect_sims
    navigator.locate_encounter_banner = MagicMock()

    res = navigator.execute_yellow_zone_interaction()

    assert res is True
    assert "sims_selected" in clicked_actions
    # Banner check should NOT be called if Sims were detected and handled
    navigator.locate_encounter_banner.assert_not_called()


def test_yellow_zone_orbit_not_duplicated_after_pink_dot():
    """
    Verifies that once a yellow zone has been orbited from a pink dot interaction,
    arriving at the subsequent yellow orbit waypoint does not re-trigger orbiting.
    """
    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    wp18 = {"index": 18, "name": "Pink Marker", "x": 316.0, "y": 257.0, "action": "pink_encounter"}
    wp19 = {
        "index": 19,
        "name": "Orbit Zone (zone_4)",
        "x": 336.0,
        "y": 259.0,
        "action": "orbit",
        "orbit_zone": {"id": "zone_4", "duration": 50.0, "perimeter_points": [[336.0, 259.0]]},
    }
    wp20 = {"index": 20, "name": "Waypoint 20", "x": 360.0, "y": 270.0, "action": "walk"}
    mock_movement_path.waypoints = [wp18, wp19, wp20]
    mock_movement_path.current_idx = 1
    mock_movement_path.get_current_target.return_value = wp19
    mock_movement_path.advance.return_value = wp20
    mock_movement_path.update_to_nearest.return_value = wp19

    navigator = RouteNavigator(movement_path=mock_movement_path)
    navigator.is_active = True
    navigator.execute_yellow_zone_interaction = MagicMock()

    # Precondition: zone_4 was already orbited during pink dot interaction
    navigator.interacted_zones.add("zone_4")

    # Arrive at WP 19
    telemetry = navigator.update((336.0, 259.0))

    # Should NOT trigger yellow zone interaction or set is_orbiting to True
    navigator.execute_yellow_zone_interaction.assert_not_called()
    assert not navigator.is_orbiting
    # Should advance to next waypoint
    mock_movement_path.advance.assert_called_once()


def test_visualizer_green_light_indicator_and_click():
    """Verifies that PlayerTrackerVisualizer renders the green light button and graphic indicator when waiting."""
    from src.player_tracker_visualizer import PlayerTrackerVisualizer

    viz = PlayerTrackerVisualizer()
    viz.navigator.waiting_for_green_light = True

    # Render dashboard
    dummy_crop = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_result = {
        "minimap_player": {"found": True, "x": 100, "y": 100, "box": (90, 90, 20, 20)},
        "room": {"room_id": "room_1", "confidence": 0.9},
        "world_map": {},
        "reference_map": {},
        "navigation": viz.navigator.get_telemetry(None, 0.0, []),
    }

    dashboard = viz.render_dashboard(dummy_crop, dummy_result)

    # Button rect must be set
    assert viz.btn_green_light_rect != (0, 0, 0, 0)
    bx, by, bw, bh = viz.btn_green_light_rect
    assert bw > 0 and bh > 0

    # Simulate mouse click on the green light button
    click_x = bx + bw // 2
    click_y = by + bh // 2
    viz._on_mouse(cv2.EVENT_LBUTTONDOWN, click_x, click_y, 0, None)

    # Must have given the green light!
    assert viz.navigator.waiting_for_green_light is False
    assert "GREEN LIGHT GIVEN" in viz.notification_msg


def test_sim_click_offset_configuration(tmp_path):
    """
    Verifies that when a SIM template is detected, the bot does not click directly on it,
    but clicks 30-40px below it as configured by sim_click_y_offset_px.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.05,
            "sim_click_y_offset_px": 38,
            "sim_click_x_offset_px": 5,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    wp = {"index": 0, "name": "Pink", "x": 100, "y": 100, "action": "pink_encounter"}
    mock_movement_path.waypoints = [wp]

    navigator = RouteNavigator(
        movement_path=mock_movement_path,
        config_path=str(cfg_file),
    )
    navigator.is_active = True

    assert navigator.sim_click_y_offset_px == 38
    assert navigator.sim_click_x_offset_px == 5

    # Mock detection of sim1 at (450, 200)
    navigator.locate_sim_template = MagicMock(side_effect=lambda k: (450, 200) if k == "sim1" else None)
    mouse_moves = []
    navigator.move_mouse_inside_game = MagicMock(side_effect=lambda x, y: mouse_moves.append((x, y)) or (x, y))

    with patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):
        clicked = navigator._detect_and_click_sims()

    assert "sim1" in clicked
    # Verify move_mouse_inside_game was called with offset: (450 + 5, 200 + 38) = (455, 238)
    assert (455, 238) in mouse_moves


def test_start_at_pink_dot_configuration_and_targeting(tmp_path):
    """
    Verifies that setting start_at_pink_dot (e.g. 3) properly:
    1. Marks all preceding pink dots (1 and 2) as completed in interacted_pink_dots.
    2. Marks all preceding orbit zones as completed in interacted_zones.
    3. Positions movement_path.current_idx at the green path segment leading to Pink Dot 3.
    4. Allows character stopped at Pink Dot 2 to navigate along green route to Pink Dot 3.
    """
    from src.movement_path import MovementPath

    # Create a realistic route with 4 pink dots and green path waypoints
    waypoints = []
    # 0..4: Start to before Pink 1
    for i in range(5):
        waypoints.append({"index": i, "name": f"WP #{i}", "x": 100.0 + i * 10, "y": 100.0, "action": "walk"})
    # 5: Pink Dot 1
    waypoints.append({"index": 5, "name": "Pink Marker 1", "x": 150.0, "y": 100.0, "action": "pink_encounter", "pink_pos": [150, 100]})
    # 6..11: Green path between Pink 1 and Pink 2
    for i in range(6, 12):
        waypoints.append({"index": i, "name": f"WP #{i}", "x": 150.0 + (i - 5) * 10, "y": 120.0, "action": "walk"})
    # 12: Pink Dot 2
    waypoints.append({"index": 12, "name": "Pink Marker 2", "x": 220.0, "y": 120.0, "action": "pink_encounter", "pink_pos": [220, 120]})
    # 13..19: Green path between Pink 2 and Pink 3
    for i in range(13, 20):
        waypoints.append({"index": i, "name": f"WP #{i}", "x": 220.0 + (i - 12) * 10, "y": 150.0, "action": "walk"})
    # 20: Pink Dot 3
    waypoints.append({"index": 20, "name": "Pink Marker 3", "x": 300.0, "y": 150.0, "action": "pink_encounter", "pink_pos": [300, 150]})
    # 21..25: Green path after Pink 3
    for i in range(21, 26):
        waypoints.append({"index": i, "name": f"WP #{i}", "x": 300.0 + (i - 20) * 10, "y": 180.0, "action": "walk"})
    # 26: Pink Dot 4
    waypoints.append({"index": 26, "name": "Pink Marker 4", "x": 360.0, "y": 180.0, "action": "pink_encounter", "pink_pos": [360, 180]})

    mp = MovementPath()
    mp.waypoints = waypoints
    mp.is_loaded = True
    mp.arrival_distance = 15.0

    # Write config with start_at_pink_dot = 3
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "start_at_pink_dot": 3,
        }
    }))

    navigator = RouteNavigator(
        movement_path=mp,
        config_path=str(cfg_file),
    )

    # Config must parse start_at_pink_dot = 3
    assert navigator.start_at_pink_dot == 3

    # Check pink waypoints list
    pink_wps = mp.get_pink_waypoints()
    assert len(pink_wps) == 4
    assert [idx for idx, wp in pink_wps] == [5, 12, 20, 26]

    # Pink Dots 1 and 2 must be marked as visited
    assert 5 in navigator.interacted_pink_dots
    assert 12 in navigator.interacted_pink_dots
    assert 20 not in navigator.interacted_pink_dots

    # Starting waypoint should be WP 13 (the green path starting right after Pink Dot 2!)
    assert mp.current_idx == 13
    curr_target = mp.get_current_target()
    assert curr_target["index"] == 13

    # Now verify update_to_nearest with character at Pink Dot 2 position (220.0, 120.0)
    # It must NOT backtrack to Pink Dot 1 (WP 5) or re-trigger Pink Dot 2 (WP 12)
    target = mp.update_to_nearest((220.0, 120.0))
    assert mp.current_idx >= 13  # Never backtracks to 5 or 12!

    # Simulate character moving along green route: at WP 16
    mp.update_to_nearest((260.0, 150.0))
    assert mp.current_idx >= 16

    # Test cycling starting pink dot: 3 -> 4 -> 0 (All) -> 1 -> 2 -> 3
    assert navigator.cycle_start_pink_dot() == 4
    assert navigator.start_at_pink_dot == 4
    assert 20 in navigator.interacted_pink_dots
    assert mp.current_idx == 21  # Right after Pink 3

    assert navigator.cycle_start_pink_dot() == 0
    assert navigator.start_at_pink_dot == 0
    assert len(navigator.interacted_pink_dots) == 0
    assert mp.current_idx == 0


def test_visualizer_pink_dot_button_and_p_key():
    """Verifies that visualizer renders [P] PINK button, handles mouse clicks, and cycles correctly."""
    from src.player_tracker_visualizer import PlayerTrackerVisualizer
    from src.movement_path import MovementPath

    mp = MovementPath()
    mp.waypoints = [
        {"index": 0, "name": "Start", "x": 0, "y": 0, "action": "walk"},
        {"index": 1, "name": "Pink 1", "x": 10, "y": 10, "action": "pink_encounter"},
        {"index": 2, "name": "Walk", "x": 20, "y": 20, "action": "walk"},
        {"index": 3, "name": "Pink 2", "x": 30, "y": 30, "action": "pink_encounter"},
    ]
    mp.is_loaded = True

    viz = PlayerTrackerVisualizer(movement_path=mp)
    viz.navigator.movement_path = mp
    viz.navigator.start_at_pink_dot = 0

    # Render dashboard
    dummy_crop = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_result = {
        "minimap_player": {"found": True, "x": 100, "y": 100, "box": (90, 90, 20, 20)},
        "room": {"room_id": "room_1", "confidence": 0.9},
        "world_map": {},
        "reference_map": {},
        "navigation": viz.navigator.get_telemetry(None, 0.0, []),
    }

    dashboard = viz.render_dashboard(dummy_crop, dummy_result)

    # Verify button rect is set
    assert viz.btn_pink_dot_rect != (0, 0, 0, 0)
    px, py, pw, ph = viz.btn_pink_dot_rect
    assert pw > 0 and ph > 0

    # Initial pink target is 0
    assert viz.navigator.start_at_pink_dot == 0

    # Simulate click on [P] PINK button
    click_x = px + pw // 2
    click_y = py + ph // 2
    viz._on_mouse(cv2.EVENT_LBUTTONDOWN, click_x, click_y, 0, None)

    # Must have cycled to Pink #1
    assert viz.navigator.start_at_pink_dot == 1
    assert "TARGETING PINK DOT #1" in viz.notification_msg

    # Simulate second click
    viz._on_mouse(cv2.EVENT_LBUTTONDOWN, click_x, click_y, 0, None)
    assert viz.navigator.start_at_pink_dot == 2
    assert "TARGETING PINK DOT #2" in viz.notification_msg


def test_banner_approach_wait_before_right_and_middle_click(tmp_path):
    """
    Verifies that when banner is clicked, _wait_for_approach is executed with
    banner_approach_wait_seconds BEFORE right click and middle mouse hold are triggered.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.01,
            "middle_click_hold_seconds": 0.05,
            "banner_search_attempts": 1,
            "banner_approach_wait_seconds": 2.5,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    mock_movement_path.get_orbit_zones.return_value = []
    mock_movement_path.waypoints = [{"index": 0, "name": "Pink", "x": 150.0, "y": 100.0, "action": "pink_encounter"}]

    nav = RouteNavigator(movement_path=mock_movement_path, config_path=str(cfg_file))
    nav.is_active = True
    nav.wait_for_loot_confirmation = False

    nav.locate_sim_template = MagicMock(return_value=None)
    nav.locate_encounter_banner = MagicMock(return_value=(500, 300))
    nav.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 200, y or 200))

    event_order = []

    def mock_click():
        event_order.append("banner_click")

    def mock_approach(seconds, reason=""):
        event_order.append(f"approach_wait_{seconds}")

    def mock_rclick():
        event_order.append("right_click")

    def mock_mdown(button=None):
        event_order.append("middle_down")

    def mock_mup(button=None):
        event_order.append("middle_up")

    nav._wait_for_approach = MagicMock(side_effect=mock_approach)

    with patch("src.route_navigator.pydirectinput.click", side_effect=mock_click), \
         patch("src.route_navigator.pydirectinput.rightClick", side_effect=mock_rclick), \
         patch("src.route_navigator.pydirectinput.mouseDown", side_effect=mock_mdown), \
         patch("src.route_navigator.pydirectinput.mouseUp", side_effect=mock_mup), \
         patch("time.sleep", return_value=None):

        res = nav.execute_pink_dot_interaction(target=mock_movement_path.waypoints[0])

    assert res is True
    assert "banner_click" in event_order
    assert "approach_wait_2.5" in event_order
    assert "right_click" in event_order
    assert "middle_down" in event_order

    # Check strict sequence: banner_click -> approach_wait_2.5 -> right_click -> middle_down
    idx_click = event_order.index("banner_click")
    idx_wait = event_order.index("approach_wait_2.5")
    idx_rclick = event_order.index("right_click")
    idx_mdown = event_order.index("middle_down")

    assert idx_click < idx_wait < idx_rclick < idx_mdown


def test_banner_reclick_after_approach_when_enabled(tmp_path):
    """
    Verifies that when reclick_after_approach is enabled, the bot clicks the banner
    initially, waits for approach, and then re-clicks the banner in-range.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.01,
            "middle_click_hold_seconds": 0.05,
            "banner_search_attempts": 1,
            "banner_approach_wait_seconds": 1.0,
            "reclick_after_approach": True,
        }
    }))

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    mock_movement_path.get_orbit_zones.return_value = []
    mock_movement_path.waypoints = [{"index": 0, "name": "Pink", "x": 150.0, "y": 100.0, "action": "pink_encounter"}]

    nav = RouteNavigator(movement_path=mock_movement_path, config_path=str(cfg_file))
    nav.is_active = True
    nav.wait_for_loot_confirmation = False

    nav.locate_sim_template = MagicMock(return_value=None)
    nav.locate_encounter_banner = MagicMock(return_value=(500, 300))
    nav.collect_loot = MagicMock(return_value=0)
    nav.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 200, y or 200))
    nav._wait_for_approach = MagicMock()

    click_count = 0
    def mock_click():
        nonlocal click_count
        click_count += 1

    with patch("src.route_navigator.pydirectinput.click", side_effect=mock_click), \
         patch("src.route_navigator.pydirectinput.rightClick"), \
         patch("src.route_navigator.pydirectinput.mouseDown"), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):

        res = nav.execute_pink_dot_interaction(target=mock_movement_path.waypoints[0])

    assert res is True
    # Initial banner click + in-range reclick = 2 left clicks
    assert click_count == 2
    nav._wait_for_approach.assert_called_once_with(1.0, reason="PINK DOT BANNER")


def test_sim_approach_wait_between_sims(tmp_path):
    """
    Verifies that clicking a sim invokes _wait_for_approach with sim_approach_wait_seconds.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.01,
            "middle_click_hold_seconds": 0.05,
            "sim_approach_wait_seconds": 1.5,
        }
    }))

    nav = RouteNavigator(config_path=str(cfg_file))
    nav.is_active = True

    nav.locate_sim_template = MagicMock(side_effect=lambda k: (300, 400) if k == "sim1" else None)
    nav.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 100, y or 100))
    nav._wait_for_approach = MagicMock()

    with patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):

        clicked = nav._detect_and_click_sims(prefix="[TEST]")

    assert clicked == ["sim1"]
    nav._wait_for_approach.assert_called_once_with(1.5, reason="[TEST] Sim 1")


def test_wait_for_approach_movement_settling():
    """
    Tests that _wait_for_approach detects movement from latest_pos and exits early
    when the character settles (stops moving for >= 0.4s).
    """
    nav = RouteNavigator()
    nav.is_active = True
    nav.latest_pos = (100.0, 100.0)

    # Call _wait_for_approach with 0s max wait -> returns immediately
    nav._wait_for_approach(0.0)

    # Test with positive duration and simulated position settling
    positions = [
        (100.0, 100.0),
        (105.0, 100.0),
        (115.0, 100.0),
        (120.0, 100.0),
        (120.0, 100.0),
        (120.0, 100.0),
    ]
    def mock_sleep(sec):
        if positions:
            nav.latest_pos = positions.pop(0)

    with patch("time.sleep", side_effect=mock_sleep):
        nav._wait_for_approach(max_wait_seconds=5.0, reason="TEST SETTLE")

    assert nav.latest_pos == (120.0, 100.0)


def test_orbit_startup_grace_period_prevents_false_stuck_triggers(tmp_path):
    """
    Verifies that when entering orbit (after banner/pink dot interaction),
    a grace period is active so that momentary lack of movement (e.g. while starting
    to run or casting skills) does NOT trigger stuck recovery immediately.
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "autopilot": {
            "pink_dot_stop_seconds": 0.01,
            "middle_click_hold_seconds": 0.05,
            "banner_search_attempts": 1,
            "orbit_duration_seconds": 50.0,
        }
    }))

    yellow_zone = {
        "id": "zone_2",
        "center": [156.9, 360.6],
        "radius": 51.0,
        "duration": 50.0,
        "perimeter_points": [
            [203.9, 360.6], [170.2, 373.9], [156.9, 374.7], [140.3, 377.3],
            [109.9, 360.6], [137.0, 340.7], [156.9, 337.2], [180.1, 337.4],
        ],
    }

    mock_movement_path = MagicMock()
    mock_movement_path.is_configured = True
    mock_movement_path.get_orbit_zones.return_value = [yellow_zone]
    mock_movement_path.waypoints = [{"index": 7, "name": "Pink", "x": 148.0, "y": 376.0, "action": "pink_encounter"}]

    nav = RouteNavigator(movement_path=mock_movement_path, config_path=str(cfg_file))
    nav.is_active = True
    nav.latest_pos = (112.0, 340.0)

    nav.locate_sim_template = MagicMock(return_value=None)
    nav.locate_encounter_banner = MagicMock(return_value=(788, 174))
    nav.move_mouse_inside_game = MagicMock(side_effect=lambda x=None, y=None: (x or 200, y or 200))

    with patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.rightClick"), \
         patch("src.route_navigator.pydirectinput.mouseDown"), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):

        result = nav.execute_pink_dot_interaction(target=mock_movement_path.waypoints[0])

    assert result is True
    assert nav.is_orbiting is True
    # Orbit grace period must be set into the future
    assert nav.orbit_grace_until > time.time()
    assert nav.stuck_counter == 0
    assert nav.orbit_stuck_step_limit == 16
    assert nav.orbit_stuck_timeout_sec == 6.0


def test_multi_shade_pink_dot_extraction(tmp_path):
    """
    Verifies that various shades of pink (light pink, pastel pink, magenta, hot pink)
    are all correctly detected and tagged as pink encounter markers.
    """
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Background gray
    img[:] = (50, 50, 50)
    # Blue Start dot at (50, 50)
    cv2.circle(img, (50, 50), 6, (220, 50, 20), -1)
    # Green route line from (50, 50) to (250, 50)
    cv2.line(img, (50, 50), (250, 50), (40, 220, 40), 4)
    # Red Finish dot at (250, 50)
    cv2.circle(img, (250, 50), 6, (20, 20, 220), -1)

    # Place multiple different pink shades along the route:
    # 1. Light Pink (BGR: 193, 182, 255) at (100, 50)
    cv2.circle(img, (100, 50), 6, (193, 182, 255), -1)
    # 2. Magenta (BGR: 255, 0, 255) at (150, 50)
    cv2.circle(img, (150, 50), 6, (255, 0, 255), -1)
    # 3. Pastel Pink (BGR: 220, 209, 255) at (200, 50)
    cv2.circle(img, (200, 50), 6, (220, 209, 255), -1)

    img_file = str(tmp_path / "multi_shade_pink.png")
    cv2.imwrite(img_file, img)

    path_mgr = MovementPath()
    success = path_mgr.load_from_painted_image(img_file, spacing=20.0)
    assert success is True

    # Must find all 3 pink markers
    pink_zones = path_mgr.get_pink_zones()
    assert len(pink_zones) == 3, f"Expected 3 pink zones, got {len(pink_zones)}"

    pink_wps = [wp for wp in path_mgr.waypoints if wp.get("action") == "pink_encounter"]
    assert len(pink_wps) == 3, f"Expected 3 pink waypoints, got {len(pink_wps)}"


def test_pink_dot_arrival_detection_with_offset_pos(tmp_path):
    """
    Verifies that when a character approaches the actual pink marker position (pink_pos),
    the navigator triggers arrival even if the character is slightly offset from the green waypoint.
    """
    cfg_file = tmp_path / "test_config.json"
    with open(cfg_file, "w") as f:
        json.dump({"autopilot": {"arrival_threshold": 18.0}}, f)

    # Waypoint coordinate on green line is (316.0, 257.0), but pink dot is at (312.0, 231.0) (26px away)
    target_wp = {
        "index": 18,
        "name": "Pink Marker (pink_4)",
        "x": 316.0,
        "y": 257.0,
        "action": "pink_encounter",
        "pink_pos": [312.0, 231.0],
    }

    mock_path = MagicMock()
    mock_path.is_configured = True
    mock_path.waypoints = [target_wp]
    mock_path.get_current_target.return_value = target_wp
    mock_path.update_to_nearest.return_value = target_wp
    mock_path.advance.return_value = None

    nav = RouteNavigator(movement_path=mock_path, config_path=str(cfg_file))
    nav.is_active = True
    nav.execute_pink_dot_interaction = MagicMock()

    # Character is right next to the pink dot at (313.0, 232.0).
    # Distance to (316, 257) is ~25.2px (exceeding standard 18px arrival threshold).
    # But distance to pink_pos (312, 231) is only ~1.4px!
    current_pos = (313.0, 232.0)
    telemetry = nav.update(current_pos)

    # execute_pink_dot_interaction must have been called!
    assert nav.execute_pink_dot_interaction.called is True
    assert 18 in nav.interacted_pink_dots









