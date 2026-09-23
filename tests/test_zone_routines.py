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




