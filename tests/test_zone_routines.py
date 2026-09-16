import json
import os
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

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

    hold_p1 = next(s["duration"] for s in pink1_steps if s["action"] == "hold_mouse")
    hold_p2 = next(s["duration"] for s in pink2_steps if s["action"] == "hold_mouse")
    hold_p3 = next(s["duration"] for s in pink3_steps if s["action"] == "hold_mouse")
    hold_p4 = next(s["duration"] for s in pink4_steps if s["action"] == "hold_mouse")

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



