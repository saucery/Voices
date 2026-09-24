import json
import os
import sys
import time
from unittest.mock import MagicMock, patch
import pytest
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.route_navigator import RouteNavigator
from src.movement_path import MovementPath


def test_zone_routines_file_loads_on_init():
    """Verifies that RouteNavigator automatically loads routines/zone_routines.json on init."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)

    assert hasattr(nav, "zone_routines")
    assert isinstance(nav.zone_routines, dict)
    assert "pink_zones" in nav.zone_routines
    assert "pink_1" in nav.zone_routines["pink_zones"]
    assert "pink_2" in nav.zone_routines["pink_zones"]
    assert "pink_3" in nav.zone_routines["pink_zones"]
    assert "pink_4" in nav.zone_routines["pink_zones"]

    # Verify per-zone middle click durations configured
    pink1_steps = nav.zone_routines["pink_zones"]["pink_1"]["steps"]
    pink2_steps = nav.zone_routines["pink_zones"]["pink_2"]["steps"]
    pink3_steps = nav.zone_routines["pink_zones"]["pink_3"]["steps"]
    pink4_steps = nav.zone_routines["pink_zones"]["pink_4"]["steps"]

    hold_p1 = next(s["duration"] for s in pink1_steps if s["action"] in ("hold_mouse", "hold_key"))
    hold_p2 = next(s["duration"] for s in pink2_steps if s["action"] in ("hold_mouse", "hold_key"))
    hold_p3 = next(s["duration"] for s in pink3_steps if s["action"] in ("hold_mouse", "hold_key"))
    hold_p4 = next(s["duration"] for s in pink4_steps if s["action"] in ("hold_mouse", "hold_key"))

    assert isinstance(hold_p1, (int, float)) and hold_p1 > 0
    assert isinstance(hold_p2, (int, float)) and hold_p2 > 0
    assert isinstance(hold_p3, (int, float)) and hold_p3 > 0
    assert isinstance(hold_p4, (int, float)) and hold_p4 > 0


def test_custom_pink_zone_routine_resolution():
    """Verifies that _get_pink_zone_routine resolves routines by sequential pink index, wp index, and name."""
    mp = MovementPath()
    mp.waypoints = [
        {"index": 0, "name": "Start", "x": 100, "y": 100, "action": "walk"},
        {"index": 8, "name": "Pink 1", "x": 150, "y": 100, "action": "pink_encounter"},
        {"index": 18, "name": "Pink 2", "x": 250, "y": 100, "action": "pink_encounter"},
    ]
    mp.is_loaded = True
    nav = RouteNavigator(movement_path=mp)

    # Resolution by pink target #1 (index 8)
    r1 = nav._get_pink_zone_routine(mp.waypoints[1])
    assert r1 is not None
    assert "Pink Dot #1" in r1.get("name", "")

    # Resolution by pink target #2 (index 18)
    r2 = nav._get_pink_zone_routine(mp.waypoints[2])
    assert r2 is not None
    assert "Pink Dot #2" in r2.get("name", "")


def test_custom_yellow_zone_routine_resolution():
    """Verifies that _get_yellow_zone_routine resolves routines by zone id."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)

    zone1 = {"id": "zone_1", "center": [100, 100], "radius": 50}
    r = nav._get_yellow_zone_routine(zone1)
    assert r is not None
    assert "Yellow Zone #1" in r.get("name", "")


def test_zone_routine_step_execution_hold_mouse():
    """Verifies that step 'hold_mouse' respects the exact duration configured in the step."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True

    step = {
        "action": "hold_mouse",
        "button": "middle",
        "duration": 0.15,
    }
    context = {}

    start_t = time.time()
    success = nav._execute_zone_routine_step(step, context, zone_label="TEST")
    elapsed = time.time() - start_t

    assert success is True
    assert elapsed >= 0.14


def test_zone_routine_step_execution_click_mouse():
    """Verifies that step 'click_mouse' executes properly."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True

    step = {
        "action": "click_mouse",
        "button": "right",
        "clicks": 2,
        "delay": 0.05,
    }
    context = {}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST")
    assert success is True


def test_zone_routine_step_execution_press_key():
    """Verifies that step 'press_key' executes key presses."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True

    step = {
        "action": "press_key",
        "key": "q",
        "duration": 0.05,
    }
    context = {}

    start_t = time.time()
    success = nav._execute_zone_routine_step(step, context, zone_label="TEST")
    elapsed = time.time() - start_t

    assert success is True
    assert elapsed >= 0.04


def test_zone_routine_step_execution_detect_sims_and_banner_skip():
    """
    Verifies that if detect_and_click_sims finds a sim, context['sims_clicked'] is True
    and subsequent click_encounter_banner skips banner clicking.
    """
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True

    # Mock locating sim1
    nav.locate_sim_template = MagicMock(side_effect=lambda key: (500, 300) if key == "sim1" else None)
    nav.locate_encounter_banner = MagicMock(return_value=(600, 400))

    sim_step = {
        "action": "detect_and_click_sims",
        "priority": ["sim1"],
        "approach_wait": 0.01,
    }
    banner_step = {
        "action": "click_encounter_banner",
        "only_if_no_sims": True,
        "approach_wait": 0.01,
    }

    context = {}
    nav._execute_zone_routine_step(sim_step, context, zone_label="TEST")
    assert context["sims_clicked"] is True

    # Now execute banner step -> should skip because sim was clicked
    nav._execute_zone_routine_step(banner_step, context, zone_label="TEST")
    assert context.get("banner_clicked", False) is False
    nav.locate_encounter_banner.assert_not_called()


def test_banner_always_clicked_even_if_sims_clicked():
    """
    Verifies that by default, clicking the encounter banner is ALWAYS performed
    even after sims are successfully detected and clicked.
    """
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True

    nav.locate_sim_template = MagicMock(side_effect=lambda key: (500, 300) if key == "sim1" else None)
    nav.locate_encounter_banner = MagicMock(return_value=(600, 400))
    nav.move_mouse_inside_game = MagicMock(return_value=(600, 400))

    sim_step = {
        "action": "detect_and_click_sims",
        "priority": ["sim1"],
        "approach_wait": 0.01,
    }
    banner_step = {
        "action": "click_encounter_banner",
        "approach_wait": 0.01,
    }

    context = {}
    nav._execute_zone_routine_step(sim_step, context, zone_label="TEST")
    assert context["sims_clicked"] is True

    # Now execute banner step -> must NOT skip, must click encounter banner
    nav._execute_zone_routine_step(banner_step, context, zone_label="TEST")
    assert context.get("banner_clicked") is True
    nav.locate_encounter_banner.assert_called()


def test_zone_routine_encounter_banner_double_check_when_still_present():
    """
    Verifies that if encounter banner remains detected after clicking in a zone routine step,
    the double-check verification automatically re-clicks the banner in-range.
    """
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True
    nav.banner_approach_wait_seconds = 0.0

    call_count = 0
    click_count = 0

    def mock_locate_banner():
        nonlocal call_count
        call_count += 1
        # 1st call: initial search -> found at (500, 300)
        # 2nd call: double-check after 1s -> STILL found at (510, 310) (missed initial click!)
        # 3rd call: double-check after re-click -> None (confirmed gone)
        if call_count == 1:
            return (500, 300)
        elif call_count == 2:
            return (510, 310)
        return None

    def mock_click():
        nonlocal click_count
        click_count += 1

    nav.locate_encounter_banner = MagicMock(side_effect=mock_locate_banner)
    moves = []
    nav.move_mouse_inside_game = MagicMock(side_effect=lambda x, y: moves.append((x, y)) or (x, y))

    banner_step = {
        "action": "click_encounter_banner",
        "approach_wait": 0.0,
        "verify_click": True,
        "verify_delay": 0.05,
        "max_click_attempts": 2,
    }
    context = {}

    with patch("src.route_navigator.pydirectinput.click", side_effect=mock_click), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):
        nav._execute_zone_routine_step(banner_step, context, zone_label="TEST")

    assert context.get("banner_clicked") is True
    assert click_count == 2
    assert (500, 300) in moves
    assert (510, 310) in moves
    assert call_count == 2


def test_reload_route_reloads_zone_routines(tmp_path):
    """Verifies that calling reload_route reloads zone_routines.json on the fly."""
    mp = MovementPath()
    mp.reload = MagicMock(return_value=True)

    routines_file = tmp_path / "zone_routines.json"
    routines_file.write_text(json.dumps({
        "pink_zones": {
            "pink_1": {
                "name": "Modified Pink 1",
                "steps": [{"action": "hold_mouse", "duration": 9.9}]
            }
        }
    }))

    nav = RouteNavigator(movement_path=mp)
    nav.zone_routines_file = str(routines_file)

    # Initial load of custom file
    nav.load_zone_routines(str(routines_file))
    assert nav.zone_routines["pink_zones"]["pink_1"]["name"] == "Modified Pink 1"
    assert nav.zone_routines["pink_zones"]["pink_1"]["steps"][0]["duration"] == 9.9

    # Modify file
    routines_file.write_text(json.dumps({
        "pink_zones": {
            "pink_1": {
                "name": "Reloaded Pink 1",
                "steps": [{"action": "hold_mouse", "duration": 12.5}]
            }
        }
    }))

    # Trigger reload_route
    nav.reload_route()
    assert nav.zone_routines["pink_zones"]["pink_1"]["name"] == "Reloaded Pink 1"
    assert nav.zone_routines["pink_zones"]["pink_1"]["steps"][0]["duration"] == 12.5


def test_pickup_loot_executes_approach_wait():
    """Verifies that pickup_loot passes approach_wait to collect_loot and executes _wait_for_approach."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True

    # Return 1 loot item then None
    loot_positions = [(300, 200), None]
    nav.locate_loot = MagicMock(side_effect=lambda: loot_positions.pop(0) if loot_positions else None)
    nav.move_mouse_inside_game = MagicMock(return_value=(300, 200))
    nav._wait_for_approach = MagicMock()

    step = {
        "action": "pickup_loot",
        "max_items": 5,
        "pickup_delay": 0.05,
        "approach_wait": 2.2,
        "wait_for_green_light": False,
    }
    context = {}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST")
    assert success is True
    nav._wait_for_approach.assert_called_once_with(2.2, reason="LOOT #1")


def test_zone_routine_step_execution_orbit_yellow_zone():
    """Verifies that step 'orbit_yellow_zone' executes the orbit loop and sets _routine_did_orbit."""
    mp = MovementPath()
    mp.orbit_zones = [
        {
            "id": "zone_1",
            "center": [100.0, 100.0],
            "radius": 50.0,
            "perimeter_points": [[90.0, 90.0], [110.0, 90.0], [110.0, 110.0], [90.0, 110.0]],
        }
    ]
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True
    nav.latest_pos = (100.0, 100.0)

    # Mock _run_orbit_loop to verify it is called with correct parameters
    nav._run_orbit_loop = MagicMock(return_value=True)

    step = {
        "action": "orbit_yellow_zone",
        "duration": 45.0,
        "right_click_interval": 0.5,
    }
    context = {"target": {"x": 100.0, "y": 100.0}}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST_ZONE")
    assert success is True
    assert context.get("orbited") is True
    assert nav._routine_did_orbit is True
    nav._run_orbit_loop.assert_called_once_with(
        duration=45.0,
        best_zone=mp.orbit_zones[0],
        right_click_interval=0.5,
        zone_label="TEST_ZONE",
        rolling_enabled=False,
        rolling_interval=5.0,
        portal_early_exit=False,
        portal_early_exit_min_seconds=30.0,
    )


def test_route_navigator_logs_have_timestamps(capsys):
    """Verifies that logs output from route_navigator include [HH:MM:SS] timestamps."""
    from src.route_navigator import _log
    import re

    _log("Test log message for verification")
    captured = capsys.readouterr()
    # Should match pattern [HH:MM:SS] Test log message
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] Test log message for verification", captured.out) is not None


def test_update_does_not_block_ui_when_worker_thread_alive():
    """Verifies that navigator.update() immediately returns telemetry and does not run blocking routines when worker thread is active."""
    mp = MovementPath()
    mp.waypoints = [
        {"index": 0, "name": "Start", "x": 100, "y": 100, "action": "walk"},
        {"index": 1, "name": "Pink Target", "x": 100, "y": 100, "action": "pink_encounter"},
    ]
    nav = RouteNavigator(movement_path=mp)
    nav.is_active = True
    nav.execute_pink_dot_interaction = MagicMock()

    # Simulate active background worker thread
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    nav._worker_thread = mock_thread

    # Call update with character arriving exactly at the pink target
    res = nav.update((100.0, 100.0))

    # Should update latest_pos and return telemetry immediately WITHOUT calling execute_pink_dot_interaction
    assert nav.latest_pos == (100.0, 100.0)
    assert res is not None
    assert isinstance(res, dict)
    assert "status_message" in res
    nav.execute_pink_dot_interaction.assert_not_called()


def test_run_orbit_loop_unsets_is_interacting_during_orbit():
    """Verifies that is_interacting is False during orbit so UI telemetry and worker loop reflect active movement."""
    mp = MovementPath()
    nav = RouteNavigator(movement_path=mp)
    nav.is_interacting = True
    best_zone = {
        "id": "zone_1",
        "center": [100.0, 100.0],
        "perimeter_points": [[100.0, 90.0], [110.0, 100.0]],
    }

    # Run for 0.05 seconds
    nav.compute_wasd_keys = MagicMock(return_value=["w"])
    nav.step_duration = 0.01

    def fake_wasd(*args, **kwargs):
        # Inside orbit loop, is_interacting should be False
        assert nav.is_interacting is False
        assert nav.is_orbiting is True
        return ["w"]

    nav.compute_wasd_keys = fake_wasd
    nav._run_orbit_loop(duration=0.04, best_zone=best_zone, right_click_interval=999.0)

    # After orbit loop finishes, is_interacting is restored
    assert nav.is_interacting is True
    assert nav.is_orbiting is False


def test_navigate_to_sim_location_executes_walk():
    """Verifies that navigate_to_sim_location delegates to _walk_to_coordinate with target's sim_pos."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._walk_to_coordinate = MagicMock(return_value=True)

    step = {"action": "navigate_to_sim_location", "timeout": 6.0}
    context = {"target": {"x": 100.0, "y": 100.0, "sim_pos": [115.0, 110.0]}}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST_ZONE")
    assert success is True
    nav._walk_to_coordinate.assert_called_once_with((115.0, 110.0), label="NAV→SIM", timeout=6.0, arrival_threshold=nav.arrival_threshold)


def test_navigate_to_sim_location_skipped_when_no_sim_pos():
    """Verifies that navigate_to_sim_location returns True gracefully if sim_pos is not available."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._walk_to_coordinate = MagicMock(return_value=True)

    step = {"action": "navigate_to_sim_location"}
    context = {"target": {"x": 100.0, "y": 100.0, "sim_pos": None}}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST_ZONE")
    assert success is True
    nav._walk_to_coordinate.assert_not_called()


def test_navigate_to_pink_location_executes_walk():
    """Verifies that navigate_to_pink_location delegates to _walk_to_coordinate with target's pink_pos."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._walk_to_coordinate = MagicMock(return_value=True)

    step = {"action": "navigate_to_pink_location", "timeout": 7.0}
    context = {"target": {"x": 100.0, "y": 100.0, "pink_pos": [98.0, 102.0]}}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST_ZONE")
    assert success is True
    nav._walk_to_coordinate.assert_called_once_with((98.0, 102.0), label="NAV→BANNER", timeout=7.0, arrival_threshold=nav.arrival_threshold)


def test_navigate_to_loot_location_executes_walk():
    """Verifies that navigate_to_loot_location delegates to _walk_to_coordinate with target's loot_pos."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._walk_to_coordinate = MagicMock(return_value=True)

    step = {"action": "navigate_to_loot_location", "timeout": 9.0}
    context = {"target": {"x": 100.0, "y": 100.0, "loot_pos": [130.0, 90.0]}}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST_ZONE")
    assert success is True
    nav._walk_to_coordinate.assert_called_once_with((130.0, 90.0), label="NAV→LOOT", timeout=9.0, arrival_threshold=nav.arrival_threshold)


def test_navigate_to_loot_location_skipped_when_no_loot_pos():
    """Verifies that navigate_to_loot_location returns True gracefully if loot_pos is not available."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._walk_to_coordinate = MagicMock(return_value=True)

    step = {"action": "navigate_to_loot_location"}
    context = {"target": {"x": 100.0, "y": 100.0}}

    success = nav._execute_zone_routine_step(step, context, zone_label="TEST_ZONE")
    assert success is True
    nav._walk_to_coordinate.assert_not_called()


def test_walk_to_coordinate_movement_loop():
    """Verifies that _walk_to_coordinate moves toward target until arrival threshold."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.step_duration = 0.01

    # Start character at (100.0, 100.0), target at (150.0, 100.0)
    # Character advances closer on each step
    positions = [(100.0, 100.0), (120.0, 100.0), (145.0, 100.0)]
    pos_idx = 0

    def mock_pos_getter():
        nonlocal pos_idx
        p = positions[min(pos_idx, len(positions) - 1)]
        pos_idx += 1
        return p

    orig_prop = getattr(type(nav), "latest_pos", None)
    type(nav).latest_pos = property(lambda self: mock_pos_getter())

    try:
        nav.compute_wasd_keys = MagicMock(return_value=["d"])
        reached = nav._walk_to_coordinate((150.0, 100.0), label="TEST_WALK", timeout=5.0, arrival_threshold=10.0)
        assert reached is True
        assert nav.held_keys == set()  # Keys released after walk
    finally:
        del type(nav).latest_pos
        if orig_prop is not None:
            type(nav).latest_pos = orig_prop


def test_detect_sims_with_retries_and_diagnostics():
    """Verifies that _detect_and_click_sims retries up to search_attempts and succeeds on later attempt."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.sim_approach_wait_seconds = 0.0

    call_count = 0

    def mock_locate(key, threshold=None):
        nonlocal call_count
        call_count += 1
        # Succeed on 3rd attempt for sim1, then disappear on verification
        if key == "sim1" and call_count == 3:
            return (400, 300)
        return None

    nav.locate_sim_template = MagicMock(side_effect=mock_locate)
    nav.move_mouse_inside_game = MagicMock(return_value=(400, 300))

    with patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):
        clicked = nav._detect_and_click_sims(
            sim_order=["sim1"],
            search_attempts=4,
            settle_wait=0.01,
        )

    assert clicked == ["sim1"]
    # 3 search attempts (found on 3rd) + 1 verification check (None -> confirmed) = 4 calls
    assert call_count == 4


def test_detect_sims_double_check_and_reclick_when_still_present():
    """
    Verifies that if a sim remains detected on screen after the initial click,
    the double-check logic automatically re-clicks the sim in-range.
    """
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.sim_approach_wait_seconds = 0.0

    call_count = 0
    click_count = 0

    def mock_locate(key, threshold=None):
        nonlocal call_count
        call_count += 1
        # 1st call: initial search -> found at (400, 300)
        # 2nd call: double check after 1s -> STILL found at (410, 310) (missed initial click!)
        # 3rd call: double check after re-click -> None (confirmed clicked/gone!)
        if key == "sim1":
            if call_count == 1:
                return (400, 300)
            elif call_count == 2:
                return (410, 310)
        return None

    def mock_click():
        nonlocal click_count
        click_count += 1

    nav.locate_sim_template = MagicMock(side_effect=mock_locate)
    moves = []
    nav.move_mouse_inside_game = MagicMock(side_effect=lambda x, y: moves.append((x, y)) or (x, y))

    with patch("src.route_navigator.pydirectinput.click", side_effect=mock_click), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("time.sleep", return_value=None):
        clicked = nav._detect_and_click_sims(
            sim_order=["sim1"],
            y_offset_px=35,
            x_offset_px=0,
            search_attempts=1,
            verify_click=True,
            verify_delay=0.1,
            max_click_attempts=2,
        )

    assert clicked == ["sim1"]
    # 2 clicks executed: 1 initial click + 1 double-check re-click!
    assert click_count == 2
    # First click at (400, 335), second click at (410, 345)
    assert (400, 335) in moves
    assert (410, 345) in moves
    assert call_count == 2


def test_collect_loot_concurrency_guard():
    """Verifies that collect_loot does not run concurrently if already active."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav._is_collecting_loot = True  # Simulate active loot collection

    with patch.object(nav, "locate_loot") as mock_locate:
        count = nav.collect_loot()

    assert count == 0
    mock_locate.assert_not_called()


def test_update_orbit_does_not_trigger_loot_when_worker_thread_alive():
    """Verifies that update() does not trigger collect_loot() or advance waypoints when worker thread is alive."""
    mock_path = MagicMock()
    mock_path.is_configured = True
    mock_path.current_idx = 5
    mock_path.get_current_target.return_value = {"x": 100, "y": 100, "name": "WP 5"}
    mock_path.distance_to_target.return_value = 10.0

    nav = RouteNavigator(movement_path=mock_path)
    nav.is_active = True
    nav.is_interacting = True  # Routine actively executing
    nav.is_orbiting = True
    nav.orbit_start_time = time.time() - 60.0  # Expired
    nav.orbit_duration = 50.0

    # Simulate alive background worker thread
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    nav._worker_thread = mock_thread

    with patch.object(nav, "collect_loot") as mock_loot:
        telemetry = nav.update(current_pos=(100.0, 100.0))

    mock_loot.assert_not_called()
    mock_path.advance.assert_not_called()
    assert telemetry is not None


def test_persistent_right_click_activation_and_reset():
    """Verifies that persistent right-click activates on hold_mouse and resets on stop/destination."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    assert nav.persistent_right_click_active is False

    # Simulate hold_mouse execution
    step = {"action": "hold_mouse", "button": "middle", "duration": 0.05, "right_click_interval": 0.65}
    with patch("src.route_navigator.pydirectinput"):
        nav._execute_zone_routine_step(step, {}, zone_label="Pink 1")

    assert nav.persistent_right_click_active is True
    assert nav.persistent_right_click_interval == 0.65

    # Test stop() resets it
    nav.stop()
    assert nav.persistent_right_click_active is False


def test_persistent_combat_during_all_events_key_mode():
    """Verifies that pressing letter 't' continues during waiting_for_green_light, stop, and update ticks."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.persistent_combat_action = "key"
    nav.persistent_combat_key = "t"
    nav.enable_persistent_combat(interval=0.05)
    assert nav.persistent_combat_active is True

    with patch("src.route_navigator.pydirectinput") as mock_pdi:
        # 1. Test update() tick triggers key 't'
        nav.last_orbit_right_click = time.time() - 1.0
        nav.update((100, 100))
        mock_pdi.keyDown.assert_any_call("t")
        mock_pdi.keyUp.assert_any_call("t")

        # 2. Test during stop step
        mock_pdi.keyDown.reset_mock()
        mock_pdi.keyUp.reset_mock()
        nav.last_orbit_right_click = time.time() - 1.0
        stop_step = {"action": "stop", "duration": 0.1}
        nav._execute_zone_routine_step(stop_step, {}, zone_label="Test")
        mock_pdi.keyDown.assert_any_call("t")
        mock_pdi.keyUp.assert_any_call("t")

        # 3. Test during waiting_for_green_light
        mock_pdi.keyDown.reset_mock()
        mock_pdi.keyUp.reset_mock()
        nav.last_orbit_right_click = time.time() - 1.0
        nav.waiting_for_green_light = True
        triggered = nav._trigger_persistent_combat_if_due()
        assert triggered is True
        mock_pdi.keyDown.assert_any_call("t")
        mock_pdi.keyUp.assert_any_call("t")

    nav.stop()
    assert nav.persistent_combat_active is False


def test_persistent_combat_during_all_events_right_click_mode():
    """Verifies that combat key 't' continues when persistent_combat_action is 'right_click'."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.persistent_combat_action = "right_click"
    nav.enable_persistent_combat(interval=0.05)
    assert nav.persistent_combat_active is True

    with patch("src.route_navigator.pydirectinput") as mock_pdi:
        nav.last_orbit_right_click = time.time() - 1.0
        nav.update((100, 100))
        mock_pdi.keyDown.assert_any_call("t")
        mock_pdi.keyUp.assert_any_call("t")

    nav.stop()
    assert nav.persistent_combat_active is False


def test_suppress_combat_during_approach():
    """Verifies that combat attacks are suppressed when approaching interactables if suppress_combat_during_approach is True."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.suppress_combat_during_approach = True
    nav.is_approaching_interactable = True
    nav.enable_persistent_combat(interval=0.05)

    with patch("src.route_navigator.pydirectinput") as mock_pdi:
        nav.last_orbit_right_click = time.time() - 1.0
        triggered = nav._trigger_persistent_combat_if_due()
        assert triggered is False
        mock_pdi.keyDown.assert_not_called()
        mock_pdi.rightClick.assert_not_called()

        # When not approaching, combat fires normally
        nav.is_approaching_interactable = False
        triggered = nav._trigger_persistent_combat_if_due()
        assert triggered is True

    nav.stop()


def test_hold_mouse_and_click_mouse_only_once_on_first_pink_dot():
    """Verifies that hold_mouse and click_mouse only run ONCE on first pink encounter and are skipped on subsequent ones."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.hold_q_enemy_reactive_enabled = False
    assert nav.has_executed_initial_hold is False

    step_hold = {"action": "hold_mouse", "button": "middle", "duration": 0.05, "combat_interval": 0.65, "combat_action": "key", "combat_key": "t"}
    step_click = {"action": "click_mouse", "button": "right", "clicks": 1}

    with patch("src.route_navigator.pydirectinput") as mock_pdi:
        # First pink encounter: executes key 't' and key 'q' hold
        nav._execute_zone_routine_step(step_click, {}, zone_label="Pink 1")
        mock_pdi.keyDown.assert_any_call("t")
        mock_pdi.keyUp.assert_any_call("t")

        mock_pdi.reset_mock()
        nav._execute_zone_routine_step(step_hold, {}, zone_label="Pink 1")
        mock_pdi.keyDown.assert_any_call("q")
        mock_pdi.keyUp.assert_any_call("q")
        assert nav.has_executed_initial_hold is True
        assert nav.persistent_combat_active is True
        assert nav.persistent_combat_action == "key"
        assert nav.persistent_combat_key == "t"

        # Second pink encounter: skips click and hold
        mock_pdi.reset_mock()
        nav._execute_zone_routine_step(step_click, {}, zone_label="Pink 2")
        mock_pdi.keyDown.assert_not_called()

        nav._execute_zone_routine_step(step_hold, {}, zone_label="Pink 2")
        mock_pdi.keyDown.assert_not_called()

    nav.stop()


def test_toggle_persistent_combat_hotkey():
    """Verifies that F3 / toggle_persistent_combat turns combat attacking on/off."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    assert nav.persistent_combat_active is False

    # Toggle ON
    active = nav.toggle_persistent_combat()
    assert active is True
    assert nav.persistent_combat_active is True

    # Toggle OFF
    active = nav.toggle_persistent_combat()
    assert active is False
    assert nav.persistent_combat_active is False
    nav.stop()


def test_click_mouse_near_character():
    """Verifies that click_mouse_near_character moves cursor next to game center, clicks, and presses follow-up key."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.monitor_idx = 0
    with patch.object(nav, "get_game_center_coords", return_value=(960, 540)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(1020, 580)) as mock_move, \
         patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch("src.route_navigator.window_focuser"):

        mx, my = nav.click_mouse_near_character(
            button="left",
            offset_x=60,
            offset_y=40,
            follow_up_key="r",
            follow_up_delay=0.01,
            label="Test Frost Bomb",
        )

        assert mx == 1020
        assert my == 580
        mock_move.assert_called_once_with(1020, 580)
        mock_pdi.click.assert_called_once()
        mock_pdi.mouseUp.assert_called_with(button="left")
        mock_pdi.keyDown.assert_any_call("r")
        mock_pdi.keyUp.assert_any_call("r")
    nav.stop()


def test_pink_1_frost_bomb_repeat_until_enemies_detected_then_hold_q_1_6s():
    """
    Verifies Room 1 flow:
    1. Encounter banner click has activate_combat: False.
    2. hold_key triggers Frost Bomb (L-click near char) + follow-up 'R'.
    3. Repeats Frost Bomb + 'R' every 4.0s (tested with fast interval) if no enemies detected.
    4. Upon enemy detection, holds 'Q' for 1.6s then releases.
    5. Transitions to persistent 'T' attack and yellow zone orbit.
    """
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    with patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch("src.route_navigator.window_focuser"), \
         patch.object(nav, "move_mouse_inside_game", return_value=(1020, 580)):

        # 1. Click encounter banner step (activate_combat: False)
        with patch.object(nav, "locate_encounter_banner", return_value=(960, 540)):
            step_banner = {
                "action": "click_encounter_banner",
                "approach_wait": 0.0,
                "activate_combat": False,
            }
            context = {"target": {"pink_pos": (200, 200)}}
            res = nav._execute_zone_routine_step(step_banner, context, zone_label="PINK DOT")
            assert res is True
            assert nav.persistent_combat_active is False

        # 2. Hold Q step with pre_cast_button (left), follow_up_key (r), repeat interval
        step_hold = {
            "action": "hold_key",
            "key": "q",
            "duration": 0.1,  # Fast duration for test speed
            "min_hold_seconds": 0.1,
            "max_hold_seconds": 0.1,
            "pre_cast_button": "left",
            "pre_cast_interval": 0.06,  # Fast repeat to test retry loop
            "follow_up_key": "r",
            "follow_up_delay": 0.01,
            "click_offset": [60, 40],
            "combat_key": "t",
            "combat_interval": 0.65,
            "detect_timeout": 5.0,
        }

        # Mock detector: undetected for 3 cycles (triggering pre-cast repeat), then detected
        detect_calls = 0
        def mock_detect(screen, character_center=None, near_threshold_px=200.0):
            nonlocal detect_calls
            detect_calls += 1
            if detect_calls < 4:
                return {"detected": False, "count": 0, "nearest_distance": 999.0, "has_enemy_near": False}
            return {"detected": True, "count": 2, "nearest_distance": 120.0, "has_enemy_near": True}

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = mock_detect

        mock_capt = MagicMock()
        import numpy as np
        mock_capt.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(nav, "_get_enemy_detector", return_value=mock_detector), \
             patch.object(nav, "_get_capturer", return_value=mock_capt):

            t0 = time.time()
            res_hold = nav._execute_zone_routine_step(step_hold, context, zone_label="PINK DOT")
            elapsed = time.time() - t0

            assert res_hold is True
            # Left click and follow-up 'r' were called multiple times due to retry
            assert mock_pdi.click.call_count >= 2
            mock_pdi.keyDown.assert_any_call("r")
            mock_pdi.keyUp.assert_any_call("r")

            # Once enemies detected, 'q' was pressed and released
            mock_pdi.keyDown.assert_any_call("q")
            mock_pdi.keyUp.assert_any_call("q")

            # After step completes, persistent combat 't' is activated
            assert nav.persistent_combat_active is True
            assert nav.persistent_combat_key == "t"

    nav.stop()


def test_pink_1_zone_routines_json_configuration():
    """Verify that routines/zone_routines.json pink_1 matches all user requirements."""
    nav = RouteNavigator(movement_path=MovementPath())
    p1 = nav.zone_routines["pink_zones"]["pink_1"]
    steps = p1["steps"]

    banner_step = next(s for s in steps if s["action"] == "click_encounter_banner")
    assert banner_step.get("activate_combat") is False

    hold_step = next(s for s in steps if s["action"] in ("hold_key", "hold_mouse"))
    assert hold_step["key"] == "q"
    assert hold_step["duration"] == 1.6
    assert hold_step["min_hold_seconds"] == 1.6
    assert hold_step["max_hold_seconds"] == 1.6
    assert hold_step["pre_cast_button"] == "left"
    assert hold_step["pre_cast_interval"] == 4.0
    assert hold_step["follow_up_key"] == "r"
    assert hold_step["follow_up_delay"] == 2.0
    assert hold_step["click_offset"] == [60, 40]
    assert hold_step["combat_key"] == "t"

    orbit_step = next(s for s in steps if s["action"] == "orbit_yellow_zone")
    assert orbit_step["duration"] == 50.0


def test_locate_portal_detection():
    """Verifies that locate_portal correctly detects the portal with high confidence."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.monitor_idx = 0

    fire_path = r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-22 191708.png"
    nofire_path = r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-22 191732.png"

    if os.path.exists(fire_path) and os.path.exists(nofire_path):
        img_fire = cv2.imread(fire_path)
        img_nofire = cv2.imread(nofire_path)

        pos_f = nav.locate_portal(screen=img_fire)
        assert pos_f is not None
        assert 1400 <= pos_f[0] <= 1700
        assert 350 <= pos_f[1] <= 650

        pos_nf = nav.locate_portal(screen=img_nofire)
        assert pos_nf is not None
        assert 1400 <= pos_nf[0] <= 1700
        assert 350 <= pos_nf[1] <= 650

    # Negative test on empty screen
    zero_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert nav.locate_portal(screen=zero_screen) is None
    nav.stop()


def test_locate_delirium_statue_detection():
    """Verifies that locate_delirium_statue detects the statue in both fire and clean ground."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.monitor_idx = 0

    fire_path = r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-22 191708.png"
    nofire_path = r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-22 191732.png"

    if os.path.exists(fire_path) and os.path.exists(nofire_path):
        img_fire = cv2.imread(fire_path)
        img_nofire = cv2.imread(nofire_path)

        pos_f = nav.locate_delirium_statue(screen=img_fire)
        assert pos_f is not None
        assert 900 <= pos_f[0] <= 1250
        assert 300 <= pos_f[1] <= 600

        pos_nf = nav.locate_delirium_statue(screen=img_nofire)
        assert pos_nf is not None
        assert 900 <= pos_nf[0] <= 1250
        assert 300 <= pos_nf[1] <= 600

    # Test on user run screenshot where statue is on left of portal
    user_shot_path = r"debug_logs\delirium_detected_20260923_105309_conf66.png"
    if os.path.exists(user_shot_path):
        img_user = cv2.imread(user_shot_path)
        pos_u = nav.locate_delirium_statue(screen=img_user)
        assert pos_u is not None
        # Must detect genuine statue on the left (x < 800), NOT false match on right (x > 1150)
        assert 600 <= pos_u[0] <= 800
        assert 250 <= pos_u[1] <= 400

    # Test on user run screenshot 134031 where burning ground previously pulled detection to the right
    user_shot_134031 = r"debug_logs\delirium_detected_20260923_134031_conf72.png"
    if os.path.exists(user_shot_134031):
        img_user2 = cv2.imread(user_shot_134031)
        pos_u2 = nav.locate_delirium_statue(screen=img_user2)
        assert pos_u2 is not None
        # Must detect genuine statue on the left (x in [600, 720]), NOT missclick into fire (x > 750)
        assert 600 <= pos_u2[0] <= 720
        assert 250 <= pos_u2[1] <= 450

    # Negative test on empty screen
    zero_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert nav.locate_delirium_statue(screen=zero_screen) is None
    nav.stop()


def test_click_delirium_statue_step_execution():
    """Verifies that click_delirium_statue locates statue, clicks, and waits loot_drop_delay."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    with patch.object(nav, "locate_delirium_statue", return_value=(1080, 420)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(1080, 420)) as mock_move, \
         patch("src.route_navigator.pydirectinput") as mock_pdi:

        step = {
            "action": "click_delirium_statue",
            "search_attempts": 3,
            "approach_wait": 0.0,
            "loot_drop_delay": 0.05,
        }
        res = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
        assert res is True
        mock_move.assert_called_once_with(1080, 420)
        mock_pdi.click.assert_called_once()
        mock_pdi.mouseUp.assert_called_with(button="left")

    nav.stop()


def test_orbit_yellow_zone_portal_early_exit():
    """Verifies that orbit_yellow_zone checks portal after 30s min duration and exits early when detected."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.monitor_idx = 0

    best_zone = {
        "id": "zone_7",
        "center": [112.5, 105.1],
        "radius": 60.0,
        "perimeter_points": [[112.5, 165.1], [172.5, 105.1]],
        "associated_pink": "pink_7",
    }

    # Simulate orbit: portal not detected initially (< portal_early_exit_min_seconds), then detected
    portal_calls = 0
    def mock_locate_portal(threshold=None, screen=None):
        nonlocal portal_calls
        portal_calls += 1
        return (1550, 490)

    with patch.object(nav, "locate_portal", side_effect=mock_locate_portal), \
         patch("src.route_navigator.window_focuser"), \
         patch.object(nav, "move_mouse_inside_game"):

        # Test with portal_early_exit=True and min_seconds=0.05 for fast test execution
        t0 = time.time()
        res = nav._run_orbit_loop(
            duration=10.0,
            best_zone=best_zone,
            right_click_interval=1.0,
            zone_label="PINK DOT #7",
            portal_early_exit=True,
            portal_early_exit_min_seconds=0.05,
        )
        elapsed = time.time() - t0

        assert res is True
        assert elapsed < 5.0  # Exited early before 10.0s duration
        assert portal_calls >= 1

    nav.stop()


def test_pink_7_zone_routines_json_configuration():
    """Verify that routines/zone_routines.json pink_7 matches all Room 7 requirements."""
    nav = RouteNavigator(movement_path=MovementPath())
    p7 = nav.zone_routines["pink_zones"]["pink_7"]
    steps = p7["steps"]

    orbit_step = next(s for s in steps if s["action"] == "orbit_yellow_zone")
    assert orbit_step["portal_early_exit"] is True
    assert orbit_step["portal_early_exit_min_seconds"] == 30.0

    statue_step = next(s for s in steps if s["action"] == "click_delirium_statue")
    assert statue_step["loot_drop_delay"] == 2.0
    assert statue_step.get("require_loot_proximity") is True
    assert statue_step.get("max_loot_distance", 35.0) == 35.0

    # Ensure correct ordering: orbit -> navigate_to_loot_location -> click_delirium_statue -> pickup_loot
    actions = [s["action"] for s in steps]
    orbit_idx = actions.index("orbit_yellow_zone")
    loot_nav_idx = actions.index("navigate_to_loot_location")
    statue_idx = actions.index("click_delirium_statue")
    pickup_idx = actions.index("pickup_loot")

    assert orbit_idx < loot_nav_idx < statue_idx < pickup_idx
    nav.stop()


def test_click_delirium_statue_proximity_guard_at_loot_dot():
    """Verifies that click_delirium_statue proceeds when character is within max_loot_distance of LOOT dot."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.latest_pos = (130.0, 61.0)  # ~1.4px away from (131.0, 62.0)

    with patch.object(nav, "locate_delirium_statue", return_value=(1080, 420)) as mock_locate, \
         patch.object(nav, "move_mouse_inside_game", return_value=(1080, 420)), \
         patch("src.route_navigator.pydirectinput") as mock_pdi:

        step = {
            "action": "click_delirium_statue",
            "search_attempts": 2,
            "approach_wait": 0.0,
            "loot_drop_delay": 0.05,
            "require_loot_proximity": True,
            "max_loot_distance": 35.0,
            "loot_pos": [131.0, 62.0],
        }
        res = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
        assert res is True
        mock_locate.assert_called()
        mock_pdi.click.assert_called_once()
    nav.stop()


def test_click_delirium_statue_proximity_guard_walks_to_loot_if_far():
    """Verifies that if character is far from LOOT dot, click_delirium_statue walks to LOOT dot first."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.latest_pos = (50.0, 50.0)  # ~82px away from (131.0, 62.0)

    def mock_walk(target_pos, **kwargs):
        nav.latest_pos = (131.0, 62.0)
        return True

    with patch.object(nav, "_walk_to_coordinate", side_effect=mock_walk) as mock_walk_fn, \
         patch.object(nav, "locate_delirium_statue", return_value=(1080, 420)) as mock_locate, \
         patch.object(nav, "move_mouse_inside_game", return_value=(1080, 420)), \
         patch("src.route_navigator.pydirectinput") as mock_pdi:

        step = {
            "action": "click_delirium_statue",
            "search_attempts": 2,
            "approach_wait": 0.0,
            "loot_drop_delay": 0.05,
            "require_loot_proximity": True,
            "max_loot_distance": 35.0,
            "walk_to_loot_if_far": True,
            "loot_pos": [131.0, 62.0],
        }
        res = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
        assert res is True
        mock_walk_fn.assert_called_once()
        mock_locate.assert_called()
        mock_pdi.click.assert_called_once()
    nav.stop()


def test_click_delirium_statue_proximity_guard_skips_when_far_and_no_walk():
    """Verifies that if character is far and walk is disabled, click_delirium_statue skips detection."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.latest_pos = (50.0, 50.0)  # ~82px away from (131.0, 62.0)

    with patch.object(nav, "locate_delirium_statue") as mock_locate, \
         patch("src.route_navigator.pydirectinput") as mock_pdi:

        step = {
            "action": "click_delirium_statue",
            "search_attempts": 2,
            "approach_wait": 0.0,
            "loot_drop_delay": 0.05,
            "require_loot_proximity": True,
            "max_loot_distance": 35.0,
            "walk_to_loot_if_far": False,
            "loot_pos": [131.0, 62.0],
        }
        res = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
        assert res is True
        mock_locate.assert_not_called()
        mock_pdi.click.assert_not_called()
    nav.stop()


def test_loot_hidden_after_sims_and_before_encounter_banner():
    """Verifies that loot labels are hidden immediately after SIMs are pressed, and stay hidden during banner click."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.hide_loot_labels = MagicMock()
    nav.ensure_loot_labels_visible = MagicMock()
    nav._detect_and_click_sims = MagicMock(return_value=["sim1"])
    nav.locate_encounter_banner = MagicMock(return_value=(500, 300))
    nav.move_mouse_inside_game = MagicMock(return_value=(500, 300))

    context = {}

    # 1. Step: detect_and_click_sims
    sim_step = {"action": "detect_and_click_sims", "priority": ["sim1"]}
    nav._execute_zone_routine_step(sim_step, context, zone_label="TEST_ZONE")

    # hide_loot_labels MUST be called after SIM detection completes
    nav.hide_loot_labels.assert_called()
    assert context["sims_clicked"] is True

    # 2. Step: click_encounter_banner
    nav.ensure_loot_labels_visible.reset_mock()
    nav.hide_loot_labels.reset_mock()

    banner_step = {"action": "click_encounter_banner", "approach_wait": 0.0}
    with patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.mouseUp"):
        nav._execute_zone_routine_step(banner_step, context, zone_label="TEST_ZONE")

    # click_encounter_banner must NOT call ensure_loot_labels_visible, and must call hide_loot_labels
    nav.ensure_loot_labels_visible.assert_not_called()
    nav.hide_loot_labels.assert_called()
    nav.stop()


def test_delirium_statue_fresh_screen_detection_and_hidden_loot_labels():
    """Verifies that click_delirium_statue locates statue fresh on screen and keeps loot labels hidden."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    nav.ensure_loot_labels_visible = MagicMock()
    nav.locate_delirium_statue = MagicMock(return_value=(800, 350))
    nav.move_mouse_inside_game = MagicMock(return_value=(800, 350))

    with patch("src.route_navigator.pydirectinput.click"), \
         patch("src.route_navigator.pydirectinput.mouseUp"), \
         patch("src.route_navigator.window_focuser"):

        step = {
            "action": "click_delirium_statue",
            "search_attempts": 2,
            "approach_wait": 0.0,
            "loot_drop_delay": 0.01,
            "require_loot_proximity": False,
        }
        context = {}
        res = nav._execute_zone_routine_step(step, context, zone_label="PINK DOT #7")

    assert res is True
    # locate_delirium_statue should be called fresh with portal_pos=None
    nav.locate_delirium_statue.assert_called()
    # Loot labels must NOT be unhidden before clicking Delirium statue
    nav.ensure_loot_labels_visible.assert_not_called()
    nav.stop()


def test_wait_for_user_key_f5():
    """Verifies that wait_for_user_key sets status and waits until key pressed or confirmed."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    # 1. Test timeout path
    res = nav.wait_for_user_key(key="f5", prompt="Press F5", timeout=0.05)
    assert res is True
    assert nav.waiting_for_user_key is False

    # 2. Test manual confirmation via confirm_user_key()
    nav.waiting_for_user_key = True
    assert nav.confirm_user_key() is True
    assert nav.waiting_for_user_key is False

    # 3. Test step execution
    step = {"action": "wait_for_user_key", "key": "f5", "timeout": 0.05}
    res_step = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
    assert res_step is True
    nav.stop()


def test_locate_stash_template_matching():
    """Verifies that locate_stash matches templates on user uploaded screenshot."""
    nav = RouteNavigator(movement_path=MovementPath())
    import cv2
    stash_crop = cv2.imread(r"templates/ui/stash_full.png")
    assert stash_crop is not None

    pos = nav.locate_stash(screen=stash_crop)
    assert pos is not None
    assert isinstance(pos, tuple)
    assert len(pos) == 2
    # Target should be inside crop
    assert 0 <= pos[0] <= stash_crop.shape[1]
    assert 0 <= pos[1] <= stash_crop.shape[0]
    nav.stop()


def test_is_inventory_open_matching():
    """Verifies that is_inventory_open returns True when inventory title banner is present."""
    nav = RouteNavigator(movement_path=MovementPath())
    import cv2
    title_img = cv2.imread(r"templates/ui/inventory_title.png")
    assert title_img is not None

    # Embed title inside a blank canvas
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    th, tw = title_img.shape[:2]
    canvas[50:50+th, 200:200+tw] = title_img

    assert nav.is_inventory_open(screen=canvas) is True

    # Blank canvas without inventory should return False
    blank = np.zeros((600, 800, 3), dtype=np.uint8)
    assert nav.is_inventory_open(screen=blank) is False
    nav.stop()


def test_click_exit_portal_routine_step():
    """Verifies click_exit_portal locates portal and dispatches left click."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    with patch.object(nav, "locate_portal", return_value=(1140, 344)) as mock_loc, \
         patch.object(nav, "move_mouse_inside_game", return_value=(1140, 344)), \
         patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "_wait_for_approach"):

        step = {"action": "click_exit_portal", "search_attempts": 2, "approach_wait": 0.0}
        res = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
        assert res is True
        mock_loc.assert_called()
        mock_pdi.click.assert_called_once()
        mock_pdi.mouseUp.assert_called_with(button="left")
    nav.stop()


def test_click_stash_routine_step():
    """Verifies click_stash locates stash, clicks it, and confirms inventory is open."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    with patch.object(nav, "locate_stash", return_value=(960, 480)) as mock_loc, \
         patch.object(nav, "move_mouse_inside_game", return_value=(960, 480)), \
         patch.object(nav, "is_inventory_open", side_effect=[False, True, True]) as mock_inv, \
         patch("src.route_navigator.pydirectinput") as mock_pdi:

        step = {
            "action": "click_stash",
            "search_attempts": 2,
            "timeout": 2.0,
            "verify_inventory_open": True,
        }
        res = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
        assert res is True
        mock_loc.assert_called()
        assert (mock_pdi.mouseDown.call_count == 1 and mock_pdi.mouseUp.call_count == 1) or mock_pdi.click.call_count == 1
        mock_inv.assert_called()
    nav.stop()


def test_click_stash_skips_click_when_inventory_already_open():
    """Verifies click_stash skips clicking when inventory is already open."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    with patch.object(nav, "locate_stash") as mock_loc, \
         patch.object(nav, "is_inventory_open", return_value=True), \
         patch("src.route_navigator.pydirectinput") as mock_pdi:

        res = nav.click_stash(timeout=2.0)
        assert res is True
        mock_loc.assert_not_called()
        assert mock_pdi.mouseDown.call_count == 0
        assert mock_pdi.click.call_count == 0
    nav.stop()


def test_pink_7_full_routine_includes_portal_and_stash():
    """Verifies that routines/zone_routines.json pink_7 includes the complete sequence through stash."""
    nav = RouteNavigator(movement_path=MovementPath())
    p7 = nav.zone_routines["pink_zones"]["pink_7"]
    actions = [s["action"] for s in p7["steps"]]

    assert "wait_for_user_key" in actions
    assert "click_exit_portal" in actions
    assert "click_stash" in actions
    assert "stash_inventory_items" in actions

    idx_loot = actions.index("pickup_loot")
    idx_loot_return = [i for i, a in enumerate(actions) if a == "navigate_to_loot_location"][-1]
    idx_f5 = actions.index("wait_for_user_key")
    idx_portal = actions.index("click_exit_portal")
    idx_stash = actions.index("click_stash")
    idx_stash_items = actions.index("stash_inventory_items")

    assert idx_loot < idx_loot_return < idx_f5 < idx_portal < idx_stash < idx_stash_items
    nav.stop()


def test_is_in_hideout_matching():
    """Verifies that is_in_hideout matches hideout_layout.png against screen."""
    nav = RouteNavigator(movement_path=MovementPath())
    import cv2
    layout = cv2.imread(r"templates/ui/hideout_layout.png")
    assert layout is not None

    # Place layout in upper-right quadrant of a 1080p canvas
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    lh, lw = layout.shape[:2]
    canvas[20:20+lh, 1920-lw-20:1920-20] = layout

    assert nav.is_in_hideout(screen=canvas) is True

    # Blank screen should return False
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert nav.is_in_hideout(screen=blank) is False
    nav.stop()


def test_save_inventory_screenshot():
    """Verifies that save_inventory_screenshot writes an image file to inventory_screenshots."""
    nav = RouteNavigator(movement_path=MovementPath())
    dummy_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    path = nav.save_inventory_screenshot(screen=dummy_screen)
    assert path is not None
    assert os.path.exists(path)
    # Clean up test artifact
    try:
        os.remove(path)
    except Exception:
        pass
    nav.stop()


def test_stash_inventory_items_detection_and_ctrl_click():
    """Verifies stash_inventory_items detects items in columns 0..9 and ctrl+clicks them, excluding cols 10..11."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    import cv2
    title_img = cv2.imread(r"templates/ui/inventory_title.png")
    assert title_img is not None

    # Construct test screen with inventory title at (1200, 100)
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    th, tw = title_img.shape[:2]
    screen[100:100+th, 1200:1200+tw] = title_img

    # Locate inventory to get exact slot coordinates
    inv_info = nav.locate_inventory_window(screen=screen)
    assert inv_info is not None

    # Paint an item in row 0, col 2 (should be stashed)
    c2_x = int(round(inv_info["col_centers"][2]))
    c2_y = int(round(inv_info["row_centers"][0]))
    screen[c2_y-10:c2_y+10, c2_x-10:c2_x+10] = 180

    # Paint an item in row 0, col 9 (10th column, should be EXCLUDED with exclude_last_columns=3)
    c9_x = int(round(inv_info["col_centers"][9]))
    c9_y = int(round(inv_info["row_centers"][0]))
    screen[c9_y-10:c9_y+10, c9_x-10:c9_x+10] = 200

    # Paint an item in row 0, col 11 (last column, should be EXCLUDED)
    c11_x = int(round(inv_info["col_centers"][11]))
    c11_y = int(round(inv_info["row_centers"][0]))
    screen[c11_y-10:c11_y+10, c11_x-10:c11_x+10] = 220

    clicked_positions = []
    def mock_move(x, y):
        clicked_positions.append((x, y))
        return (x, y)

    with patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "move_mouse_inside_game", side_effect=mock_move):

        stashed = nav.stash_inventory_items(screen=screen, exclude_last_columns=3)

        assert stashed == 1  # Only col 2 was stashed, col 9 and 11 were excluded!
        mock_pdi.keyDown.assert_any_call("ctrl")
        mock_pdi.click.assert_called_once()
        mock_pdi.keyUp.assert_any_call("ctrl")
        assert len(clicked_positions) == 1
        assert clicked_positions[0] == (c2_x, c2_y)

    nav.stop()


def test_locate_inventory_window_dynamic_row_detection():
    """Verifies that horizontal grid lines trigger dynamic edge detection and accurate Row 5 centering."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    import cv2
    title_img = cv2.imread(r"templates/ui/inventory_title.png")
    assert title_img is not None

    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    th, tw = title_img.shape[:2]
    screen[100:100+th, 1200:1200+tw] = title_img

    inv_origin_y = 100 - 45
    inv_origin_x = 1200 - 200

    # Draw 6 strong horizontal lines for the grid rows (PoE grid lines: ~54px spacing)
    line_ys = [580, 638, 692, 745, 798, 854]
    for rel_y in line_ys:
        gy = inv_origin_y + rel_y
        screen[gy-1:gy+2, inv_origin_x+20:inv_origin_x+600] = 220

    inv_info = nav.locate_inventory_window(screen=screen)
    assert inv_info is not None
    assert inv_info["found"] is True
    assert inv_info["row_method"] == "dynamic_edges"
    assert len(inv_info["row_centers"]) == 5

    # Verify row 5 center is midpoint of lines 798 and 854 -> 826.0 (approx inv_origin_y + 826)
    expected_row5_y = inv_origin_y + (798 + 854) / 2.0
    assert abs(inv_info["row_centers"][4] - expected_row5_y) <= 1.5

    nav.stop()


def test_locate_inventory_window_multi_resolution_screen1_1800p():
    """Verifies multi-scale inventory window detection and scaling on Screen 1 (2880x1800 OLED)."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    import cv2
    title_img = cv2.imread(r"templates/ui/inventory_title.png")
    assert title_img is not None

    # Screen 1 is 2880x1800 -> scale factor ~1.667
    target_h, target_w = 1800, 2880
    scale = target_h / 1080.0  # 1.6666...
    scaled_title = cv2.resize(title_img, (int(title_img.shape[1] * scale), int(title_img.shape[0] * scale)))

    screen_1800p = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    sth, stw = scaled_title.shape[:2]
    # Place on right side of 1800p screen
    t_y = int(100 * scale)
    t_x = int(2000)
    screen_1800p[t_y:t_y+sth, t_x:t_x+stw] = scaled_title

    inv_info = nav.locate_inventory_window(screen=screen_1800p)
    assert inv_info is not None
    assert inv_info["found"] is True
    assert abs(inv_info["scale"] - scale) <= 0.05
    assert len(inv_info["row_centers"]) == 5
    assert len(inv_info["col_centers"]) == 12

    # Check that Row 5 center is scaled by ~scale
    inv_origin_y = t_y - int(45 * inv_info["scale"])
    expected_row5_y = inv_origin_y + 825.5 * inv_info["scale"]
    assert abs(inv_info["row_centers"][4] - expected_row5_y) <= 5.0

    nav.stop()


def test_click_exit_portal_loot_dot_retry():
    """Verifies click_exit_portal walks to LOOT dot if portal not initially on screen."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    locate_attempts = 0
    def mock_locate(screen=None):
        nonlocal locate_attempts
        locate_attempts += 1
        # Fails on initial attempts, succeeds after walk to LOOT dot
        if locate_attempts > 2:
            return (1140, 344)
        return None

    with patch.object(nav, "locate_portal", side_effect=mock_locate), \
         patch.object(nav, "_resolve_target_loot_pos", return_value=(131.0, 62.0)), \
         patch.object(nav, "_walk_to_coordinate", return_value=True) as mock_walk, \
         patch.object(nav, "move_mouse_inside_game", return_value=(1140, 344)), \
         patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "_wait_for_approach"):

        step = {"action": "click_exit_portal", "search_attempts": 2, "approach_wait": 0.0}
        res = nav._execute_zone_routine_step(step, {}, zone_label="PINK DOT #7")
        assert res is True
        mock_walk.assert_called_once()
        mock_pdi.click.assert_called_once()
    nav.stop()


def test_hideout_sequence_live_and_dry_run():
    """Verifies test_hideout_sequence runs the complete hideout flow in live and dry_run modes."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    dummy_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with patch.object(nav, "is_in_hideout", return_value=True), \
         patch.object(nav, "locate_stash", return_value=(850, 420)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(850, 420)), \
         patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "is_inventory_open", return_value=True), \
         patch.object(nav, "save_inventory_screenshot", return_value="inventory_screenshots/test.png"), \
         patch.object(nav, "stash_inventory_items", return_value=4):

        # 1. Test Dry Run mode (no clicks, items_stashed is 0)
        rep_dry = nav.test_hideout_sequence(dry_run=True, screen=dummy_screen)
        assert rep_dry["success"] is True
        assert rep_dry["in_hideout"] is True
        assert rep_dry["stash_pos"] == (850, 420)
        assert rep_dry["stash_clicked"] is False
        assert rep_dry["inventory_open"] is True
        assert rep_dry["screenshot_path"] == "inventory_screenshots/test.png"
        assert rep_dry["items_detected"] == 4
        assert rep_dry["items_stashed"] == 0
        mock_pdi.click.assert_not_called()

        # 2. Test Live Execution mode
        rep_live = nav.test_hideout_sequence(dry_run=False, screen=dummy_screen)
        assert rep_live["success"] is True
        assert rep_live["stash_clicked"] is True
        assert rep_live["items_stashed"] == 4
        assert (mock_pdi.mouseDown.call_count == 1 and mock_pdi.mouseUp.call_count == 1) or mock_pdi.click.call_count == 1

    nav.stop()


def test_hideout_sequence_deposit_only():
    """Verifies deposit_only=True bypasses hideout minimap and stash clicking."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    dummy_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with patch.object(nav, "is_in_hideout") as mock_ho, \
         patch.object(nav, "locate_stash") as mock_locate, \
         patch.object(nav, "is_inventory_open", return_value=True), \
         patch.object(nav, "save_inventory_screenshot", return_value="inventory_screenshots/test.png"), \
         patch.object(nav, "stash_inventory_items", return_value=3):

        rep = nav.test_hideout_sequence(dry_run=False, deposit_only=True, screen=dummy_screen)
        assert rep["success"] is True
        assert rep["items_stashed"] == 3
    nav.stop()


def test_combat_stops_when_portal_detected_or_delirium_statue_pressed():
    """Verifies that persistent combat attack ('T') stops as soon as exit portal is detected or Delirium statue is clicked."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True
    nav.enable_persistent_combat(interval=0.05, key="t")
    assert nav.persistent_combat_active is True

    # 1. When Delirium statue is clicked and exit portal is NOT yet detected, combat remains ACTIVE for looting
    with patch.object(nav, "locate_delirium_statue", return_value=(600, 300)), \
         patch.object(nav, "locate_portal", return_value=None), \
         patch.object(nav, "move_mouse_inside_game", return_value=(600, 300)), \
         patch("src.route_navigator.pydirectinput"), \
         patch.object(nav, "_wait_for_approach"):

        nav.click_delirium_statue(loot_drop_delay=0.0, require_loot_proximity=False)
        assert nav.persistent_combat_active is True

    # 2. When exit portal in Room 7 is detected, persistent combat must be stopped
    with patch.object(nav, "locate_delirium_statue", return_value=(600, 300)), \
         patch.object(nav, "locate_portal", return_value=(1140, 344)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(600, 300)), \
         patch("src.route_navigator.pydirectinput"), \
         patch.object(nav, "_wait_for_approach"):

        nav.click_delirium_statue(loot_drop_delay=0.0, require_loot_proximity=False)
        assert nav.persistent_combat_active is False

    # 3. When orbit loop detects portal (early exit), persistent combat must also be stopped
    nav.enable_persistent_combat(interval=0.05, key="t")
    assert nav.persistent_combat_active is True

    zone = {"id": "zone_7", "center": [100.0, 100.0], "radius": 50.0}
    with patch.object(nav, "locate_portal", return_value=(1140, 344)):
        nav._run_orbit_loop(
            duration=5.0,
            best_zone=zone,
            portal_early_exit=True,
            portal_early_exit_min_seconds=0.0,
            zone_label="pink_7",
        )
        assert nav.persistent_combat_active is False

    nav.stop()


def test_visualizer_trigger_hideout_test():
    """Verifies that PlayerTrackerVisualizer.trigger_hideout_test launches without NameError and calls test_hideout_sequence."""
    from src.player_tracker_visualizer import PlayerTrackerVisualizer

    nav = RouteNavigator(movement_path=MovementPath())
    vis = PlayerTrackerVisualizer()
    vis.navigator = nav

    with patch.object(nav, "test_hideout_sequence", return_value={"success": True, "items_stashed": 5}) as mock_hideout:
        vis.trigger_hideout_test(dry_run=True)
        # Wait briefly for worker thread to finish
        time.sleep(0.15)
        mock_hideout.assert_called_once()
        assert "STASHED 5 ITEMS" in vis.notification_msg

    nav.stop()


def test_is_in_hideout_minimap_and_fallback():
    """Verifies is_in_hideout detection: minimap match, map device confirmation, and prevention of false-positives from stash in enemy areas."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.hideout_layout_tpl = np.zeros((100, 100, 3), dtype=np.uint8)

    # 1. Minimap matches directly
    screen_match = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with patch("src.route_navigator.cv2.matchTemplate", return_value=np.array([[0.85]])):
        assert nav.is_in_hideout(screen=screen_match) is True

    # 2. Minimap below threshold, and STASH alone is visible in enemy area (Simulacrum) -> MUST NOT be hideout!
    screen_enemy_stash = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with patch("src.route_navigator.cv2.matchTemplate", return_value=np.array([[0.30]])), \
         patch.object(nav, "locate_stash", return_value=(800, 450)), \
         patch.object(nav, "locate_map_device", return_value=None):
        assert nav.is_in_hideout(screen=screen_enemy_stash, check_stash_fallback=True) is False

    # 3. Minimap borderline (0.38) and Map Device is visible -> Confirmed hideout
    screen_map_device = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with patch("src.route_navigator.cv2.matchTemplate", return_value=np.array([[0.38]])), \
         patch.object(nav, "locate_map_device", return_value=(960, 540)):
        assert nav.is_in_hideout(screen=screen_map_device) is True

    # 4. Neither minimap nor Map Device matches
    screen_fail = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with patch("src.route_navigator.cv2.matchTemplate", return_value=np.array([[0.20]])), \
         patch.object(nav, "locate_map_device", return_value=None), \
         patch.object(nav, "locate_stash", return_value=None):
        assert nav.is_in_hideout(screen=screen_fail) is False

    # 5. Dual confirmation mode (require_map_device=True)
    with patch("src.route_navigator.cv2.matchTemplate", return_value=np.array([[0.85]])), \
         patch.object(nav, "locate_map_device", return_value=None):
        assert nav.is_in_hideout(screen=screen_match, require_map_device=True) is False

    with patch("src.route_navigator.cv2.matchTemplate", return_value=np.array([[0.85]])), \
         patch.object(nav, "locate_map_device", return_value=(960, 540)):
        assert nav.is_in_hideout(screen=screen_match, require_map_device=True) is True

    nav.stop()


def test_hideout_safety_guards_prevent_combat_and_routines():
    """Verifies that when in hideout (in_hideout=True), combat attacks, pink dot routines,
    zone routines, and navigation are strictly blocked and disarmed."""
    mp = MovementPath()
    mp.waypoints = [
        {"index": 0, "name": "Start", "x": 100, "y": 100, "action": "walk"},
        {"index": 1, "name": "Pink Target", "x": 100, "y": 100, "action": "pink_encounter"},
    ]
    nav = RouteNavigator(movement_path=mp)
    nav.in_hideout = True
    nav.persistent_combat_active = True
    nav.held_keys = {"w", "d"}

    # 1. Combat pulse must return False immediately
    with patch("src.route_navigator.pydirectinput.keyDown") as mock_down:
        assert nav._trigger_persistent_combat_if_due() is False
        mock_down.assert_not_called()

    # 2. Pink dot interaction must abort immediately
    assert nav.execute_pink_dot_interaction(mp.waypoints[1]) is False

    # 3. Zone routine must abort immediately
    dummy_routine = {"name": "Test Routine", "steps": [{"action": "stop", "duration": 1.0}]}
    assert nav._execute_zone_routine(dummy_routine, zone_label="PINK DOT") is False

    # 4. Starting route navigation must be refused
    with patch.object(nav, "is_in_hideout", return_value=True):
        nav.start()
        assert nav.is_active is False
        assert "Cannot start in Hideout" in nav.status_message

    # 5. update() in hideout disarms combat, releases keys, and keeps is_active=False
    telemetry = nav.update((100.0, 100.0))
    assert telemetry["in_hideout"] is True
    assert telemetry["is_active"] is False
    assert nav.is_active is False
    assert len(nav.held_keys) == 0
    assert nav.persistent_combat_active is False
    assert "HIDEOUT" in telemetry["status_message"]

    nav.stop()


def test_close_all_hideout_windows_escape():
    """Verifies close_all_hideout_windows issues Escape keypress and confirms window closure."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    with patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "is_inventory_open", return_value=False):
        closed = nav.close_all_hideout_windows(wait_seconds=0.01, verify_close=True)
        assert closed is True
        mock_pdi.keyDown.assert_any_call("escape")
        mock_pdi.keyUp.assert_any_call("escape")

    nav.stop()


def test_detect_simulacrum_map_nodes_and_circle_offset():
    """Verifies that Simulacrum icons are detected and circle coordinates are computed using configurable simulacrum_click_y_offset."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._load_stash_and_inventory_templates(force_reload=True)

    # Use existing medal or icon template
    tpl = nav.simulacrum_medal_tpl if nav.simulacrum_medal_tpl is not None else nav.simulacrum_icon_tpl
    assert tpl is not None, "Simulacrum medal/icon template must be available in templates/ui/"

    th, tw = tpl.shape[:2]
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    place_x = 800
    place_y = 450
    screen[place_y:place_y + th, place_x:place_x + tw] = tpl

    # 1. Test 15.0 px offset
    nav.simulacrum_click_y_offset = 15.0
    nodes = nav.detect_simulacrum_map_nodes(screen=screen, threshold=0.50)
    assert len(nodes) >= 1

    detected = nodes[0]
    expected_medal_x = place_x + tw // 2
    expected_medal_y = place_y + th // 2
    expected_circle_y = int(round(expected_medal_y + nav.simulacrum_click_y_offset * detected["scale"]))

    assert abs(detected["screen_medal_pos"][0] - expected_medal_x) <= 2
    assert abs(detected["screen_medal_pos"][1] - expected_medal_y) <= 2
    assert abs(detected["screen_circle_pos"][0] - expected_medal_x) <= 2
    assert abs(detected["screen_circle_pos"][1] - expected_circle_y) <= 2

    # 2. Test custom configurable offset (e.g. 25.0 px)
    nav.simulacrum_click_y_offset = 25.0
    nodes_custom = nav.detect_simulacrum_map_nodes(screen=screen, threshold=0.50)
    assert len(nodes_custom) >= 1
    expected_custom_circle_y = int(round(expected_medal_y + 25.0 * nodes_custom[0]["scale"]))
    assert abs(nodes_custom[0]["screen_circle_pos"][1] - expected_custom_circle_y) <= 2

    nav.stop()


def test_select_accessible_simulacrum_map_iteration():
    """Verifies that select_accessible_simulacrum_map iterates through candidate circles until popup is confirmed."""
    nav = RouteNavigator(movement_path=MovementPath())

    cand1 = {
        "screen_medal_pos": (400, 300),
        "screen_circle_pos": (400, 326),
        "medal_pos": (400, 300),
        "circle_pos": (400, 326),
        "confidence": 0.85,
        "scale": 1.0,
    }
    cand2 = {
        "screen_medal_pos": (700, 500),
        "screen_circle_pos": (700, 526),
        "medal_pos": (700, 500),
        "circle_pos": (700, 526),
        "confidence": 0.90,
        "scale": 1.0,
    }

    clicked_targets = []
    def mock_move(x, y):
        clicked_targets.append((x, y))
        return (x, y)

    # First candidate is inaccessible (popup=False), second candidate is accessible (popup=True)
    popup_sequence = [False, True]
    def mock_popup_visible(*args, **kwargs):
        return popup_sequence.pop(0) if popup_sequence else True

    with patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "move_mouse_inside_game", side_effect=mock_move), \
         patch.object(nav, "is_simulacrum_popup_visible", side_effect=mock_popup_visible):

        selected = nav.select_accessible_simulacrum_map(candidates=[cand1, cand2], max_attempts=5)
        assert selected is not None
        assert selected == cand2
        # Verify both circles were clicked in sequence
        assert (400, 326) in clicked_targets
        assert (mock_pdi.mouseDown.call_count == 2 and mock_pdi.mouseUp.call_count == 2) or mock_pdi.click.call_count == 2

    nav.stop()


def test_select_accessible_simulacrum_map_never_clicks_inaccessible_when_accessible_exists():
    """Verifies that select_accessible_simulacrum_map NEVER clicks inaccessible nodes when accessible nodes exist."""
    nav = RouteNavigator(movement_path=MovementPath())

    cand_accessible = {
        "screen_medal_pos": (500, 300),
        "screen_circle_pos": (500, 315),
        "medal_pos": (500, 300),
        "circle_pos": (500, 315),
        "confidence": 0.88,
        "scale": 1.0,
        "is_accessible": True,
        "acc_score": 2,
    }
    cand_inaccessible = {
        "screen_medal_pos": (800, 600),
        "screen_circle_pos": (800, 615),
        "medal_pos": (800, 600),
        "circle_pos": (800, 615),
        "confidence": 0.95,
        "scale": 1.0,
        "is_accessible": False,
        "acc_score": 0,
    }

    clicked_targets = []
    def mock_move(x, y):
        clicked_targets.append((x, y))
        return (x, y)

    with patch("src.route_navigator.pydirectinput"), \
         patch.object(nav, "move_mouse_inside_game", side_effect=mock_move), \
         patch.object(nav, "is_simulacrum_popup_visible", return_value=True):

        selected = nav.select_accessible_simulacrum_map(candidates=[cand_accessible, cand_inaccessible], max_attempts=3)
        assert selected == cand_accessible
        # Ensure only the accessible node was clicked, never the inaccessible one
        assert (500, 315) in clicked_targets
        assert (800, 615) not in clicked_targets

    nav.stop()


def test_locate_tier15_maps_in_inventory_and_transfer():
    """Verifies Tier 15 map detection in columns 10..12 and drag transfer into Delusion popup."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._load_stash_and_inventory_templates(force_reload=True)

    title_img = nav.inventory_title_tpl
    map_img = nav.tier15_map_tpl
    assert title_img is not None
    assert map_img is not None

    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    th, tw = title_img.shape[:2]
    screen[100:100 + th, 1200:1200 + tw] = title_img

    inv_info = nav.locate_inventory_window(screen=screen)
    assert inv_info is not None

    # Place Tier 15 map in Row 1, Column 10 (0-indexed col 9)
    col9_x = int(round(inv_info["col_centers"][9]))
    row1_y = int(round(inv_info["row_centers"][1]))
    mh, mw = map_img.shape[:2]
    screen[row1_y - mh//2 : row1_y - mh//2 + mh, col9_x - mw//2 : col9_x - mw//2 + mw] = map_img

    # 1. Test locate_tier15_maps_in_inventory
    found_maps = nav.locate_tier15_maps_in_inventory(screen=screen, threshold=0.50)
    assert len(found_maps) >= 1
    assert any(m["col"] == 9 and m["row"] == 1 for m in found_maps)

    # 2. Test transfer to popup slot 0 via drag
    with patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "is_inventory_open", return_value=True):
        ok = nav.insert_map_into_simulacrum_popup(screen=screen, target_slot_idx=0, method="drag")
        assert ok is True
        mock_pdi.mouseDown.assert_called_once_with(button="left")
        mock_pdi.mouseUp.assert_called_once_with(button="left")

    nav.stop()


def test_run_hideout_full_cycle_success():
    """Verifies run_hideout_full_cycle executes all 7 steps end-to-end."""
    nav = RouteNavigator(movement_path=MovementPath())

    with patch.object(nav, "is_in_hideout", return_value=True), \
         patch.object(nav, "click_stash", return_value=True), \
         patch.object(nav, "stash_inventory_items", return_value=4), \
         patch.object(nav, "close_all_hideout_windows", return_value=True), \
         patch.object(nav, "click_map_device", return_value=True), \
         patch.object(nav, "select_accessible_simulacrum_map", return_value={"screen_circle_pos": (600, 450)}), \
         patch.object(nav, "insert_map_into_simulacrum_popup", return_value=True):

        report = nav.run_hideout_full_cycle(dry_run=False, screen=np.zeros((1080, 1920, 3), dtype=np.uint8))
        assert report["success"] is True
        assert report["in_hideout"] is True
        assert report["stash_clicked"] is True
        assert report["inventory_stashed"] is True
        assert report["windows_closed"] is True
        assert report["map_device_clicked"] is True
        assert report["simulacrum_selected"] is True
        assert report["map_transferred"] is True

    nav.stop()


def test_is_simulacrum_popup_visible_with_4square_slots_and_title_only():
    """Verifies that is_simulacrum_popup_visible confirms 4-square slots and rejects title-only unavailable maps."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._load_stash_and_inventory_templates(force_reload=True)

    img_full = nav.delusion_popup_tpl
    img_title = getattr(nav, "delusion_title_tpl", None)
    assert img_full is not None, "delusion_popup_full template must be available"
    assert img_title is not None, "delusion_title_banner template must be available"

    # 1. Screen with full popup (available map with 4-square grid)
    screen_avail = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ph, pw = img_full.shape[:2]
    pos_x = (1920 - pw) // 2
    pos_y = (1080 - ph) // 2
    screen_avail[pos_y:pos_y + ph, pos_x:pos_x + pw] = img_full

    res_avail = nav.is_simulacrum_popup_visible(screen=screen_avail, save_debug=False)
    assert res_avail is True
    assert len(nav.delusion_detected_slots) == 4
    assert nav.delusion_detected_traverse is not None

    # 2. Screen with ONLY title banner (unavailable map without 4-square grid below)
    screen_unavail = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bh, bw = img_title.shape[:2]
    screen_unavail[pos_y:pos_y + bh, pos_x:pos_x + bw] = img_title

    res_unavail = nav.is_simulacrum_popup_visible(screen=screen_unavail, save_debug=False)
    assert res_unavail is False

    # 3. Blank screen
    screen_blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert nav.is_simulacrum_popup_visible(screen=screen_blank, save_debug=False) is False

    nav.stop()


def test_simulacrum_accessibility_via_green_node_connection():
    """Verifies that Simulacrum nodes connected by dashed lines to green completed nodes are recognized as accessible."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._load_stash_and_inventory_templates(force_reload=True)

    # Synthetic canvas with:
    # 1. A completed green map node at (400, 300)
    # 2. An accessible Simulacrum circle at (500, 300) linked by a dashed line to the green node
    # 3. An inaccessible Simulacrum circle at (800, 300) with no dashed line
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Draw green completed node at (400, 300):
    # Emerald green center (BGR: [60, 220, 70]) with dark/gold border
    cv2.circle(screen, (400, 300), 12, (30, 100, 160), -1)  # outer gold border
    cv2.circle(screen, (400, 300), 8, (60, 230, 70), -1)   # inner emerald green core

    # Draw dashed yellow line from (400, 300) to (500, 300)
    # Dash pattern: 5px on, 4px off
    for x in range(415, 485, 9):
        cv2.line(screen, (x, 300), (min(485, x + 5), 300), (40, 200, 240), 2)  # yellow/gold dash

    # Draw blue accessible circle at (500, 300) with white core and blue halo
    cv2.circle(screen, (500, 300), 12, (220, 140, 40), 2)  # blue ring
    cv2.circle(screen, (500, 300), 5, (255, 255, 255), -1) # white core

    # Draw inaccessible circle at (800, 300) (dark grey circle, no dashed line)
    cv2.circle(screen, (800, 300), 12, (60, 60, 60), 2)
    cv2.circle(screen, (800, 300), 5, (40, 40, 40), -1)

    # 1. Test detect_green_completed_nodes
    greens = nav.detect_green_completed_nodes(screen=screen)
    assert len(greens) >= 1
    assert any(np.hypot(g["center"][0] - 400, g["center"][1] - 300) < 5 for g in greens)

    # 2. Test check_node_accessibility
    acc_linked = nav.check_node_accessibility((500, 300), greens, screen=screen)
    assert acc_linked["is_accessible"] is True
    assert len(acc_linked["connected_greens"]) >= 1

    acc_isolated = nav.check_node_accessibility((800, 300), greens, screen=screen)
    assert acc_isolated["is_accessible"] is False
    assert len(acc_isolated["connected_greens"]) == 0

    nav.stop()


def test_simulacrum_real_atlas_screen_ranking_and_accessibility():
    """Verifies accessible vs inaccessible Simulacrum ranking on real 1080p Atlas screen."""
    screenshot_path = "inventory_screenshots/sim_circle_attempt_1_CLOSED_20260924_100159.png"
    if not os.path.exists(screenshot_path):
        pytest.skip("Real atlas screenshot not found")

    img = cv2.imread(screenshot_path)
    assert img is not None

    nav = RouteNavigator(movement_path=MovementPath())
    nodes = nav.detect_simulacrum_map_nodes(screen=img, threshold=0.72)
    assert len(nodes) >= 2, "Expected at least 2 detected Simulacrum nodes"

    # Candidate #1 must be the accessible Simulacrum map circle
    top_cand = nodes[0]
    assert top_cand["is_accessible"] is True
    assert top_cand.get("acc_score", 0) >= 2
    assert top_cand["has_blue_glow"] is True
    assert len(top_cand["connected_greens"]) >= 1
    assert np.hypot(top_cand["screen_circle_pos"][0] - 858, top_cand["screen_circle_pos"][1] - 432) < 15

    # Second candidate must be marked INACCESSIBLE
    second_cand = nodes[1]
    assert second_cand["is_accessible"] is False
    assert second_cand.get("acc_score", 0) == 0
    assert np.hypot(second_cand["screen_circle_pos"][0] - 710, second_cand["screen_circle_pos"][1] - 543) < 15

    # Dry-run selection must select candidate #1
    selected = nav.select_accessible_simulacrum_map(candidates=nodes, dry_run=True)
    assert selected is not None
    assert selected == top_cand

    nav.stop()


def test_locate_traverse_button_on_real_and_synthetic_screens():
    """Verifies that locate_traverse_button identifies the button on real and synthetic screens."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._load_stash_and_inventory_templates(force_reload=True)

    # 1. Real screenshot test
    screenshot_path = "inventory_screenshots/sim_circle_attempt_1_OPEN_20260924_105433.png"
    if os.path.exists(screenshot_path):
        img = cv2.imread(screenshot_path)
        pos = nav.locate_traverse_button(screen=img)
        assert pos is not None
        assert np.hypot(pos[0] - 880, pos[1] - 803) < 10

    # 2. Synthetic screen test with placed template
    btn_tpl = nav.delusion_traverse_tpl
    assert btn_tpl is not None, "traverse_button template must be loaded"
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bh, bw = btn_tpl.shape[:2]
    canvas[500:500 + bh, 700:700 + bw] = btn_tpl

    nav.delusion_detected_traverse = None
    pos_synth = nav.locate_traverse_button(screen=canvas)
    assert pos_synth is not None
    assert pos_synth == (700 + bw // 2, 500 + bh // 2)

    nav.stop()


def test_click_traverse_button_dry_run_and_live():
    """Verifies click_traverse_button in dry_run and live execution modes."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._load_stash_and_inventory_templates(force_reload=True)

    # 1. Dry run mode
    with patch.object(nav, "locate_traverse_button", return_value=(880, 803)):
        nav.delusion_detected_slots = [(100, 100), (200, 200)]
        nav.delusion_detected_traverse = (880, 803)
        ok_dry = nav.click_traverse_button(dry_run=True)
        assert ok_dry is True
        assert nav.delusion_detected_traverse is None
        assert nav.delusion_detected_slots == []

    # 2. Live mode with mocked inputs
    with patch.object(nav, "locate_traverse_button", return_value=(880, 803)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(880, 803)), \
         patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "is_simulacrum_popup_visible", return_value=False), \
         patch.object(nav, "is_inventory_open", return_value=False):

        ok_live = nav.click_traverse_button(dry_run=False, verify_close=True)
        assert ok_live is True
        mock_pdi.mouseDown.assert_called_once_with(button="left")
        mock_pdi.mouseUp.assert_called_once_with(button="left")

    nav.stop()


def test_locate_hideout_portal_detection():
    """Verifies that locate_hideout_portal detects portals on real and synthetic screens."""
    nav = RouteNavigator(movement_path=MovementPath())

    # 1. Real screenshot test
    screenshot_path = "debug_logs/delirium_detected_20260924_110348_conf82.png"
    if os.path.exists(screenshot_path):
        img = cv2.imread(screenshot_path)
        pos = nav.locate_hideout_portal(screen=img)
        assert pos is not None
        assert np.hypot(pos[0] - 1549, pos[1] - 487) < 15

    # 2. Synthetic canvas test with clamped SQDIFF matching
    canvas = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    p_img = nav.portal_template_img
    p_mask = nav.portal_mask
    assert p_img is not None and p_mask is not None
    ph, pw = p_img.shape[:2]
    canvas[400:400 + ph, 900:900 + pw] = np.where(p_mask[:, :, None] > 0, p_img, canvas[400:400 + ph, 900:900 + pw])

    pos_synth = nav.locate_hideout_portal(screen=canvas)
    assert pos_synth is not None
    assert np.hypot(pos_synth[0] - (900 + pw // 2), pos_synth[1] - (400 + ph // 2)) < 5

    nav.stop()


def test_click_hideout_portal_and_zone_transition():
    """Verifies that click_hideout_portal enters portal, verifies transition, and starts routine to Pink Dot #1."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.in_hideout = True

    # 1. Dry run test
    with patch.object(nav, "locate_hideout_portal", return_value=(950, 480)):
        ok_dry = nav.click_hideout_portal(dry_run=True, auto_start_route=True, start_pink_dot=1)
        assert ok_dry is True
        assert nav.start_at_pink_dot == 1

    # 2. Live mode test
    # Mock is_in_hideout to return True first (in hideout), then False (entered Simulacrum)
    hideout_states = [True, False]
    with patch.object(nav, "locate_hideout_portal", return_value=(950, 480)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(950, 480)), \
         patch("src.route_navigator.pydirectinput") as mock_pdi, \
         patch.object(nav, "is_in_hideout", side_effect=lambda: hideout_states.pop(0) if hideout_states else False), \
         patch.object(nav, "set_start_pink_dot") as mock_set_pink, \
         patch.object(nav, "start") as mock_start:

        ok_live = nav.click_hideout_portal(
            dry_run=False,
            approach_wait=0.01,
            verify_transition=True,
            auto_start_route=True,
            start_pink_dot=1,
            settle_wait=0.01,
            hold_w_seconds=0.01,
        )
        assert ok_live is True
        assert nav.in_hideout is False
        mock_pdi.mouseDown.assert_called_once_with(button="left")
        mock_pdi.mouseUp.assert_called_once_with(button="left")
        mock_pdi.keyDown.assert_any_call("w")
        mock_pdi.keyUp.assert_any_call("w")
        mock_set_pink.assert_called_once_with(1)
        mock_start.assert_called_once()

    nav.stop()


def test_execute_custom_zone_action_traverse_and_portal():
    """Verifies that _execute_zone_routine_step dispatches click_traverse_button and click_hideout_portal."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True

    with patch.object(nav, "click_traverse_button", return_value=True) as mock_trav, \
         patch.object(nav, "click_hideout_portal", return_value=True) as mock_portal:

        # Step 1: click_traverse_button
        s1 = {"action": "click_traverse_button", "timeout": 4.0, "verify_close": True}
        res1 = nav._execute_zone_routine_step(s1, context={}, zone_label="TEST")
        assert res1 is True
        mock_trav.assert_called_once_with(timeout=4.0, verify_close=True)

        # Step 2: click_hideout_portal
        s2 = {
            "action": "click_hideout_portal",
            "search_attempts": 10,
            "timeout": 8.0,
            "approach_wait": 2.0,
            "auto_start_route": True,
            "start_pink_dot": 1,
        }
        res2 = nav._execute_zone_routine_step(s2, context={}, zone_label="TEST")
        assert res2 is True
        mock_portal.assert_called_once_with(
            search_attempts=10,
            timeout=8.0,
            approach_wait=2.0,
            verify_transition=True,
            auto_start_route=True,
            start_pink_dot=1,
        )

    nav.stop()


def test_run_hideout_full_cycle_with_traverse_and_portal():
    """Verifies that run_hideout_full_cycle executes traverse and portal steps when traverse_and_enter is True."""
    nav = RouteNavigator(movement_path=MovementPath())

    with patch.object(nav, "is_in_hideout", return_value=True), \
         patch.object(nav, "click_stash", return_value=True), \
         patch.object(nav, "stash_inventory_items", return_value=3), \
         patch.object(nav, "close_all_hideout_windows", return_value=True), \
         patch.object(nav, "click_map_device", return_value=True), \
         patch.object(nav, "select_accessible_simulacrum_map", return_value={"screen_circle_pos": (600, 450)}), \
         patch.object(nav, "insert_map_into_simulacrum_popup", return_value=True), \
         patch.object(nav, "click_traverse_button", return_value=True) as mock_trav, \
         patch.object(nav, "click_hideout_portal", return_value=True) as mock_portal:

        report = nav.run_hideout_full_cycle(
            dry_run=False,
            screen=np.zeros((1080, 1920, 3), dtype=np.uint8),
            traverse_and_enter=True,
            auto_start_route=True,
            start_pink_dot=1,
        )
        assert report["success"] is True
        assert report["simulacrum_selected"] is True
        assert report["map_transferred"] is True
        assert report["traverse_clicked"] is True
        assert report["portal_clicked"] is True
        mock_trav.assert_called_once()
        mock_portal.assert_called_once_with(dry_run=False, auto_start_route=True, start_pink_dot=1)

    nav.stop()


def test_click_stash_unhides_labels_with_z_in_hideout():
    """Verifies that arriving in hideout ensures loot/object labels are unhidden with 'Z' key,
    and fallback toggles 'Z' if stash is not immediately detected."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav._loot_labels_hidden = True  # Simulating post-loot state from Room 7

    z_presses = []
    with patch.object(nav, "_press_z_key", side_effect=lambda: z_presses.append("z")), \
         patch.object(nav, "is_in_hideout", return_value=True), \
         patch.object(nav, "locate_stash", return_value=(800, 450)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(800, 450)), \
         patch.object(nav, "is_inventory_open", return_value=True), \
         patch.object(nav, "save_inventory_screenshot", return_value="dummy.png"), \
         patch("src.route_navigator.pydirectinput.click"):

        ok = nav.click_stash(search_attempts=3, timeout=5.0, verify_inventory=True)
        assert ok is True
        assert len(z_presses) >= 1
        assert nav._loot_labels_hidden is False

    nav.stop()


def test_hideout_arrival_preserves_is_active_for_routine_continuation():
    """Verifies that confirming hideout arrival and executing click_stash does NOT reset
    nav.is_active to False, ensuring step 14 (stash_inventory_items) and subsequent steps
    execute without being aborted."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.is_active = True  # Bot is actively running autopilot / routine

    with patch.object(nav, "is_in_hideout", return_value=True), \
         patch.object(nav, "locate_stash", return_value=(800, 450)), \
         patch.object(nav, "move_mouse_inside_game", return_value=(800, 450)), \
         patch.object(nav, "is_inventory_open", return_value=True), \
         patch.object(nav, "save_inventory_screenshot", return_value="dummy.png"), \
         patch("src.route_navigator.pydirectinput.click"):

        ok = nav.click_stash(search_attempts=3, timeout=5.0, verify_inventory=True)
        assert ok is True
        # nav.is_active MUST remain True so routine steps 14..20 are not aborted!
        assert nav.is_active is True
        assert nav.in_hideout is True

    nav.stop()


def test_select_accessible_simulacrum_map_fresh_capture_resync():
    """Verifies that when selecting Simulacrum map dynamically, if the first click fails to open
    the popup (e.g. map slightly moved), the bot captures a fresh screen, re-detects the live
    coordinates, and retries the updated coordinates instead of using stale positions."""
    nav = RouteNavigator(movement_path=MovementPath())

    # Simulated screenshots: img1 has node at (922, 435), img2 has node shifted to (832, 379)
    node1_initial = {
        "medal_pos": (922, 409),
        "circle_pos": (922, 435),
        "screen_circle_pos": (922, 435),
        "confidence": 0.88,
        "scale": 1.0,
        "is_accessible": True,
    }
    node1_shifted = {
        "medal_pos": (832, 353),
        "circle_pos": (832, 379),
        "screen_circle_pos": (832, 379),
        "confidence": 0.91,
        "scale": 1.0,
        "is_accessible": True,
    }

    detection_sequence = [[node1_initial], [node1_shifted]]
    def mock_detect(*args, **kwargs):
        return detection_sequence.pop(0) if detection_sequence else [node1_shifted]

    attempt = [0]
    clicked_targets = []
    def mock_move(x, y):
        attempt[0] += 1
        clicked_targets.append((x, y))
        return (x, y)

    def mock_popup_visible(*args, **kwargs):
        # Attempt 1 returns False, Attempt 2 returns True
        return attempt[0] >= 2

    capt_mock = MagicMock()
    capt_mock.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with patch.object(nav, "_get_capturer", return_value=capt_mock), \
         patch.object(nav, "detect_simulacrum_map_nodes", side_effect=mock_detect), \
         patch.object(nav, "move_mouse_inside_game", side_effect=mock_move), \
         patch.object(nav, "is_simulacrum_popup_visible", side_effect=mock_popup_visible), \
         patch("src.route_navigator.pydirectinput"):

        selected = nav.select_accessible_simulacrum_map(candidates=None, max_attempts=4)
        assert selected is not None
        # Must return the shifted node from fresh capture
        assert selected["circle_pos"] == (832, 379)
        # Click 1 was at initial position, Click 2 resynchronized to the new position
        assert (922, 435) in clicked_targets
        assert any(pt[0] == 832 for pt in clicked_targets)

    nav.stop()


def test_start_hideout_full_routine_ui_integration():
    """Verifies that the UI button and key [H] start the full hideout routine
    (unload, open map device, insert Tier 15 map, traverse, enter portal, and start bot navigation at Pink Dot #1)."""
    from src.player_tracker_visualizer import PlayerTrackerVisualizer

    nav = RouteNavigator(movement_path=MovementPath())
    vis = PlayerTrackerVisualizer(navigator=nav)

    # 1. Test start_hideout_full_routine calls run_hideout_full_cycle with full cycle parameters
    with patch.object(nav, "run_hideout_full_cycle", return_value={"success": True}) as mock_cycle:
        vis.start_hideout_full_routine(dry_run=True, start_pink_dot=1)
        time.sleep(0.1)  # Allow daemon thread worker to run
        mock_cycle.assert_called_once_with(
            dry_run=True,
            traverse_and_enter=True,
            auto_start_route=True,
            start_pink_dot=1,
        )

    # 2. Test start_hideout_full_routine does not re-enter if already active
    vis.hideout_routine_active = True
    vis.start_hideout_full_routine()
    assert "ALREADY IN PROGRESS" in vis.notification_msg
    vis.hideout_routine_active = False

    # 3. Test dashboard rendering of [H] START BOT ROUTINE button
    dummy_crop = np.zeros((150, 150, 3), dtype=np.uint8)
    dummy_result = {
        "area": {"is_hideout": True, "label": "HIDEOUT (SAFE ZONE)"},
        "room": {"recognized": False},
        "navigation": {"is_active": False, "waiting_for_green_light": False},
    }
    dashboard = vis.render_dashboard(dummy_crop, dummy_result)
    assert dashboard is not None
    bx, by, bw, bh = vis.btn_hideout_rect
    assert bw == 185
    assert bh == 30
    assert bx > 172 + 215  # Does not overlap area badge

    # 4. Test mouse click on [H] button triggers start_hideout_full_routine
    with patch.object(vis, "start_hideout_full_routine") as mock_start_btn:
        vis._on_mouse(cv2.EVENT_LBUTTONDOWN, bx + 10, by + 10, 0, None)
        mock_start_btn.assert_called_once()

    nav.stop()


















