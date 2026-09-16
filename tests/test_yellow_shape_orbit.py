"""
Unit tests for Yellow Shape Detection & Configurable Orbit Navigation.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.movement_path import MovementPath
from src.route_navigator import RouteNavigator


@pytest.fixture
def synthetic_route_with_yellow_shape(tmp_path):
    """
    Creates a synthetic route image with:
    - Blue Start Dot at (50, 100)
    - Green Route Line from (50, 100) to (150, 100)
    - Yellow Circle at (150, 100) with radius 20
    - Green Route Line continuing from (170, 100) to (250, 100)
    - Red Finish Dot at (250, 100)
    """
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # 1. Green line
    cv2.line(img, (50, 100), (250, 100), (0, 255, 0), 5)

    # 2. Yellow Shape (Circle) at (150, 100)
    # BGR for Yellow is (0, 255, 255)
    cv2.circle(img, (150, 100), 20, (0, 255, 255), -1)

    # 3. Blue Start Dot at (50, 100) - BGR: (255, 0, 0)
    cv2.circle(img, (50, 100), 7, (255, 0, 0), -1)

    # 4. Red Finish Dot at (250, 100) - BGR: (0, 0, 255)
    cv2.circle(img, (250, 100), 7, (0, 0, 255), -1)

    img_path = str(tmp_path / "route_with_yellow.png")
    cv2.imwrite(img_path, img)
    return img_path


def test_yellow_shape_extraction(synthetic_route_with_yellow_shape, tmp_path):
    """Test extracting route with yellow shape and generating orbit zone metadata."""
    out_json = str(tmp_path / "extracted_route.json")
    path_mgr = MovementPath()
    success = path_mgr.load_from_painted_image(
        image_path=synthetic_route_with_yellow_shape,
        spacing=20.0,
        save_json_path=out_json,
    )

    assert success is True
    assert len(path_mgr.waypoints) >= 4
    zones = path_mgr.get_orbit_zones()
    assert len(zones) == 1

    zone = zones[0]
    assert abs(zone["center"][0] - 150) <= 5
    assert abs(zone["center"][1] - 100) <= 5
    assert zone["radius"] >= 15.0
    assert zone["orbit_radius"] < zone["radius"]  # Verified INSIDE
    assert len(zone["perimeter_points"]) == 8

    # All perimeter points should be strictly inside the circle (dist < radius)
    for px, py in zone["perimeter_points"]:
        d_from_center = np.hypot(px - zone["center"][0], py - zone["center"][1])
        assert d_from_center < zone["radius"]

    # Find the waypoint flagged with action 'orbit'
    orbit_wps = [wp for wp in path_mgr.waypoints if wp.get("action") == "orbit"]
    assert len(orbit_wps) == 1
    assert orbit_wps[0]["orbit_zone"]["id"] == zone["id"]

    # Verify JSON persistence
    assert os.path.exists(out_json)
    with open(out_json, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert len(saved_data.get("orbit_zones", [])) == 1
    assert saved_data["settings"]["orbit_duration_seconds"] in (10.0, 50.0)
    assert saved_data["settings"]["orbit_mode"] == "inside"


def test_yellow_square_extraction(tmp_path):
    """Test that yellow square shapes are also properly detected and orbited inside."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.line(img, (40, 80), (260, 80), (0, 255, 0), 5)
    # Yellow square from (130, 60) to (170, 100)
    cv2.rectangle(img, (130, 60), (170, 100), (0, 255, 255), -1)
    # Blue start
    cv2.circle(img, (40, 80), 6, (255, 0, 0), -1)
    # Red finish
    cv2.circle(img, (260, 80), 6, (0, 0, 255), -1)

    img_path = str(tmp_path / "route_yellow_square.png")
    cv2.imwrite(img_path, img)

    path_mgr = MovementPath()
    success = path_mgr.load_from_painted_image(img_path, spacing=20.0)
    assert success is True
    zones = path_mgr.get_orbit_zones()
    assert len(zones) == 1
    assert abs(zones[0]["center"][0] - 150) <= 5
    assert abs(zones[0]["center"][1] - 80) <= 5

    # Verify all perimeter points lie strictly inside the rectangle bounds [130, 170] x [60, 100]
    for px, py in zones[0]["perimeter_points"]:
        assert 130 <= px <= 170
        assert 60 <= py <= 100


def test_orbit_execution_and_timeout(synthetic_route_with_yellow_shape, tmp_path, monkeypatch):
    """Test that navigator enters orbit around yellow shape and resumes after duration."""
    path_mgr = MovementPath()
    path_mgr.load_from_painted_image(synthetic_route_with_yellow_shape, spacing=25.0)

    # Use a short duration for testing (0.3s)
    for z in path_mgr.orbit_zones:
        z["duration"] = 0.3
    for wp in path_mgr.waypoints:
        if wp.get("action") == "orbit":
            wp["orbit_zone"]["duration"] = 0.3

    nav = RouteNavigator(movement_path=path_mgr)
    nav.wait_for_loot_confirmation = False
    monkeypatch.setattr(nav, "execute_yellow_zone_interaction", lambda *a, **kw: True)
    nav.start()
    assert nav.is_active is True
    assert nav.is_orbiting is False

    # Find the orbit waypoint
    orbit_wp = next(wp for wp in path_mgr.waypoints if wp.get("action") == "orbit")
    orbit_idx = orbit_wp["index"]

    # Set navigator to the waypoint right before orbit
    path_mgr.current_idx = orbit_idx

    # Simulate player stepping onto the orbit waypoint
    telem = nav.update((orbit_wp["x"], orbit_wp["y"]))
    assert nav.is_orbiting is True
    assert telem["is_orbiting"] is True
    assert telem["orbit_remaining_sec"] > 0.0

    # Wait for the duration to elapse
    time.sleep(0.35)

    # Next update should automatically complete the orbit and advance to the next waypoint
    telem2 = nav.update((orbit_wp["x"], orbit_wp["y"]))
    assert nav.is_orbiting is False
    assert path_mgr.current_idx > orbit_idx
    nav.stop()


def test_orbit_manual_skip(synthetic_route_with_yellow_shape, monkeypatch):
    """Test that pressing skip [N] while orbiting immediately cancels orbit and moves to next WP."""
    path_mgr = MovementPath()
    path_mgr.load_from_painted_image(synthetic_route_with_yellow_shape, spacing=25.0)

    nav = RouteNavigator(movement_path=path_mgr)
    monkeypatch.setattr(nav, "execute_yellow_zone_interaction", lambda *a, **kw: True)
    nav.start()

    orbit_wp = next(wp for wp in path_mgr.waypoints if wp.get("action") == "orbit")
    path_mgr.current_idx = orbit_wp["index"]

    # Trigger orbit
    nav.update((orbit_wp["x"], orbit_wp["y"]))
    assert nav.is_orbiting is True

    # User manually skips WP
    next_wp = nav.skip_current_waypoint()
    assert nav.is_orbiting is False
    assert next_wp["index"] > orbit_wp["index"]
    nav.stop()


def test_locate_encounter_banner(tmp_path):
    """Test locating the encounter banner on a captured screen using template matching."""
    banner = np.zeros((100, 200, 3), dtype=np.uint8)
    banner[20:80, 30:170] = [200, 50, 30]  # distinct pattern
    banner_file = str(tmp_path / "test_banner.png")
    cv2.imwrite(banner_file, banner)

    # Synthetic full screen with banner placed at (400, 300)
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    screen[300:400, 400:600] = banner

    class MockCapturer:
        def __init__(self):
            self._sct = None
        def capture(self, bbox=None):
            return screen.copy()

    nav = RouteNavigator(capturer=MockCapturer())
    nav.encounter_banner_file = banner_file
    nav.encounter_banner_img = banner

    coords = nav.locate_encounter_banner()
    assert coords is not None
    # Center should be at (400 + 100, 300 + 50) = (500, 350)
    assert abs(coords[0] - 500) <= 2
    assert abs(coords[1] - 350) <= 2


def test_execute_yellow_zone_interaction_sequence(tmp_path, monkeypatch):
    """Test full interaction sequence: left-click banner -> right-click -> middle hold -> release."""
    actions = []

    # Mock pydirectinput actions
    class MockDirectInput:
        FAILSAFE = False
        PAUSE = 0.01

        @staticmethod
        def moveTo(x, y):
            actions.append(("moveTo", x, y))

        @staticmethod
        def click():
            actions.append(("click_left",))

        @staticmethod
        def mouseUp(button="left"):
            actions.append((f"mouseUp_{button}",))

        @staticmethod
        def rightClick():
            actions.append(("rightClick",))

        @staticmethod
        def mouseDown(button="middle"):
            actions.append((f"mouseDown_{button}",))

    import src.route_navigator as rn
    monkeypatch.setattr(rn, "pydirectinput", MockDirectInput)

    # Fast hold for unit test
    nav = RouteNavigator()
    nav.middle_click_hold_seconds = 0.1
    nav.banner_search_attempts = 1
    nav.is_active = True

    # Mock banner location
    monkeypatch.setattr(nav, "locate_encounter_banner", lambda: (500, 350))

    success = nav.execute_yellow_zone_interaction()
    assert success is True

    action_names = [a[0] for a in actions]
    assert "moveTo" in action_names
    assert "click_left" in action_names
    assert "rightClick" in action_names
    assert "mouseDown_middle" in action_names
    assert "mouseUp_middle" in action_names

    # Check order: left click -> right click -> middle down -> middle up
    idx_left = action_names.index("click_left")
    idx_right = action_names.index("rightClick")
    idx_m_down = action_names.index("mouseDown_middle")
    idx_m_up = action_names.index("mouseUp_middle")

    assert idx_left < idx_right < idx_m_down < idx_m_up


def test_yellow_zone_interaction_before_orbit(synthetic_route_with_yellow_shape, monkeypatch):
    """Test that arriving at yellow zone triggers interaction exactly once before orbiting."""
    interaction_called = []

    path_mgr = MovementPath()
    path_mgr.load_from_painted_image(synthetic_route_with_yellow_shape, spacing=25.0)

    for z in path_mgr.orbit_zones:
        z["duration"] = 0.2
    for wp in path_mgr.waypoints:
        if wp.get("action") == "orbit":
            wp["orbit_zone"]["duration"] = 0.2

    nav = RouteNavigator(movement_path=path_mgr)
    nav.middle_click_hold_seconds = 0.05
    nav.banner_search_attempts = 1

    def mock_interaction(orbit_zone=None):
        interaction_called.append(True)
        return True

    monkeypatch.setattr(nav, "execute_yellow_zone_interaction", mock_interaction)

    nav.start()
    orbit_wp = next(wp for wp in path_mgr.waypoints if wp.get("action") == "orbit")
    path_mgr.current_idx = orbit_wp["index"]

    # First arrival at yellow zone
    nav.update((orbit_wp["x"], orbit_wp["y"]))
    assert len(interaction_called) == 1
    assert nav.is_orbiting is True

    # Updating while already inside the orbit does not re-trigger interaction
    nav.update((orbit_wp["x"], orbit_wp["y"]))
    assert len(interaction_called) == 1
    nav.stop()


def test_mouse_movement_clamped_inside_game_window(monkeypatch):
    """Test that move_mouse_inside_game strictly clamps coordinates inside the game window."""
    nav = RouteNavigator()

    import src.route_navigator as rn
    # Mock window bounds: left=100, top=100, right=1000, bottom=800
    monkeypatch.setattr(rn.window_focuser, "get_game_window_bounds", lambda: (100, 100, 1000, 800))

    # Point outside window to the top-left (-500, -200)
    clamped_x, clamped_y = nav.clamp_coords_to_game_window(-500, -200)
    assert clamped_x >= 100 + 60
    assert clamped_y >= 100 + 60

    # Point outside window to the bottom-right (3000, 2000)
    clamped_x2, clamped_y2 = nav.clamp_coords_to_game_window(3000, 2000)
    assert clamped_x2 <= 1000 - 60
    assert clamped_y2 <= 800 - 60

    # Default move with no coords moves to center
    cx, cy = nav.get_game_center_coords()
    assert cx == (100 + 1000) // 2
    assert cy == (100 + 800) // 2


def test_config_toggles_disable_banner_or_middle_click(monkeypatch):
    """Test that disabling banner or middle click via config bypasses those actions."""
    actions = []

    class MockDirectInput:
        FAILSAFE = False
        PAUSE = 0.01

        @staticmethod
        def moveTo(x, y):
            actions.append(("moveTo", x, y))

        @staticmethod
        def click():
            actions.append(("click_left",))

        @staticmethod
        def mouseUp(button="left"):
            actions.append((f"mouseUp_{button}",))

        @staticmethod
        def rightClick():
            actions.append(("rightClick",))

        @staticmethod
        def mouseDown(button="middle"):
            actions.append((f"mouseDown_{button}",))

    import src.route_navigator as rn
    monkeypatch.setattr(rn, "pydirectinput", MockDirectInput)
    monkeypatch.setattr(rn.window_focuser, "get_game_window_bounds", lambda: (0, 0, 1920, 1080))

    nav = RouteNavigator()
    nav.is_active = True
    nav.click_banner_enabled = False       # Banner disabled
    nav.right_click_after_banner_enabled = True
    nav.middle_click_hold_enabled = False   # Middle hold disabled

    success = nav.execute_yellow_zone_interaction()
    assert success is True

    action_names = [a[0] for a in actions]
    assert "click_left" not in action_names
    assert "mouseDown_middle" not in action_names
    assert "rightClick" in action_names


def test_orbit_yellow_zone_disabled_toggle(synthetic_route_with_yellow_shape):
    """Test that orbit_yellow_zone_enabled=False bypasses orbiting on arrival and continues route."""
    path_mgr = MovementPath()
    path_mgr.load_from_painted_image(synthetic_route_with_yellow_shape, spacing=25.0)

    nav = RouteNavigator(movement_path=path_mgr)
    nav.orbit_yellow_zone_enabled = False  # Disabled!

    orbit_wp = next(wp for wp in path_mgr.waypoints if wp.get("action") == "orbit")
    orbit_idx = orbit_wp["index"]
    path_mgr.current_idx = orbit_idx

    # Stepping onto yellow zone waypoint
    telem = nav.update((orbit_wp["x"], orbit_wp["y"]))
    # Should NOT enter orbit mode, but advance to next waypoint
    assert nav.is_orbiting is False
    assert telem["is_orbiting"] is False


def test_orbit_constant_right_click(synthetic_route_with_yellow_shape, monkeypatch):
    """Test that constant right-clicking fires periodically while orbiting."""
    right_clicks = []

    class MockDirectInput:
        FAILSAFE = False
        PAUSE = 0.01

        @staticmethod
        def rightClick():
            right_clicks.append(time.time())

        @staticmethod
        def mouseUp(button="right"):
            pass

    import src.route_navigator as rn
    monkeypatch.setattr(rn, "pydirectinput", MockDirectInput)
    monkeypatch.setattr(rn.window_focuser, "get_game_window_bounds", lambda: (0, 0, 1920, 1080))

    path_mgr = MovementPath()
    path_mgr.load_from_painted_image(synthetic_route_with_yellow_shape, spacing=25.0)

    nav = RouteNavigator(movement_path=path_mgr)
    nav.orbit_constant_right_click_enabled = True
    nav.orbit_right_click_interval_seconds = 0.05
    nav.is_orbiting = True
    nav.orbit_start_time = time.time()
    nav.orbit_duration = 5.0
    nav.last_orbit_right_click = 0.0
    nav.is_active = True

    # Call update twice with delay
    nav.update((150.0, 100.0))
    time.sleep(0.06)
    nav.update((150.0, 100.0))

    assert len(right_clicks) >= 2


