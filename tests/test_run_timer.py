"""
Unit tests for Bot Run Timer and SIM Tracking Persistence.
Tests session start, live duration calculation, SIM click recording per room,
room clearance tracking, last room detection, and JSON/TXT file persistence.
"""

import json
import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.route_navigator import RouteNavigator
from src.movement_path import MovementPath


@pytest.fixture
def mock_movement_path():
    path = MagicMock(spec=MovementPath)
    path.is_configured = True
    path.waypoints = [
        {"index": 0, "name": "Start", "action": "walk", "x": 100, "y": 100},
        {"index": 1, "name": "Pink Marker (pink_1)", "action": "pink_encounter", "x": 200, "y": 200},
        {"index": 2, "name": "Waypoint 2", "action": "walk", "x": 300, "y": 300},
        {"index": 74, "name": "Pink Marker (pink_7)", "action": "pink_encounter", "x": 400, "y": 400},
        {"index": 75, "name": "Finish", "action": "interact", "x": 500, "y": 500},
    ]
    path.current_idx = 0
    path.get_current_target.return_value = path.waypoints[0]
    path.get_pink_waypoints.return_value = [(1, path.waypoints[1]), (74, path.waypoints[3])]
    return path


@pytest.fixture
def temp_log_dir(tmp_path):
    log_dir = tmp_path / "debug_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def test_format_duration():
    """Verify duration formatting across various second intervals."""
    assert RouteNavigator._format_duration(0.0) == "0.0s"
    assert RouteNavigator._format_duration(15.4) == "15.4s"
    assert RouteNavigator._format_duration(59.9) == "59.9s"
    assert RouteNavigator._format_duration(60.0) == "1m 00.0s"
    assert RouteNavigator._format_duration(65.5) == "1m 05.5s"
    assert RouteNavigator._format_duration(272.4) == "4m 32.4s"
    assert RouteNavigator._format_duration(-5.0) == "0.0s"


def test_run_session_start_and_telemetry(mock_movement_path, temp_log_dir):
    """Verify that starting navigator starts the run timer and telemetry exposes run state."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = str(temp_log_dir / "run_history.json")
    nav.run_summary_file = str(temp_log_dir / "run_history.txt")

    assert nav.run_start_time is None
    assert nav.run_completed is False

    with patch("threading.Thread"):  # Prevent background navigation loop thread
        with patch.object(nav, "move_mouse_inside_game"):
            nav.start()

    assert nav.is_active is True
    assert nav.run_start_time is not None
    assert nav.run_id is not None
    assert nav.run_id.startswith("run_")

    telem = nav.get_telemetry(target=None, dist=0.0, keys=[])
    assert telem["run_id"] == nav.run_id
    assert telem["run_active"] is True
    assert telem["run_completed"] is False
    assert telem["run_elapsed_sec"] >= 0.0
    assert "s" in telem["run_elapsed_str"]
    assert telem["run_sims_count"] == 0
    assert telem["run_loot_count"] == 0
    assert telem["run_last_room_cleared"] is None


def test_sim_click_recording(mock_movement_path, temp_log_dir):
    """Verify that SIM clicks are tracked with timestamps, room IDs, and run elapsed times."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = str(temp_log_dir / "run_history.json")
    nav.run_summary_file = str(temp_log_dir / "run_history.txt")
    nav._start_new_run()

    # Record SIMs in Room 1
    nav.current_room_key = "pink_1"
    nav._record_sim_clicked("sim1")

    # Record SIMs in Room 2
    nav.current_room_key = "pink_2"
    nav._record_sim_clicked("sim1")
    nav._record_sim_clicked("sim3")

    assert len(nav.run_sims_clicked) == 3
    assert nav.run_sims_clicked[0]["sim"] == "sim1"
    assert nav.run_sims_clicked[0]["room"] == "pink_1"
    assert nav.run_sims_clicked[1]["sim"] == "sim1"
    assert nav.run_sims_clicked[1]["room"] == "pink_2"
    assert nav.run_sims_clicked[2]["sim"] == "sim3"
    assert nav.run_sims_clicked[2]["room"] == "pink_2"


def test_room_clearance_recording(mock_movement_path, temp_log_dir):
    """Verify room clearance tracking, split calculation, and room mapping."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = str(temp_log_dir / "run_history.json")
    nav.run_summary_file = str(temp_log_dir / "run_history.txt")
    nav._start_new_run()

    nav.current_room_key = "pink_1"
    nav._record_sim_clicked("sim1")
    nav._record_room_cleared("pink_1", routine_duration=32.5)

    assert nav.run_last_room_cleared == "pink_1"
    assert len(nav.run_rooms_cleared) == 1
    r1 = nav.run_rooms_cleared[0]
    assert r1["room"] == "pink_1"
    assert r1["routine_duration_sec"] == 32.5
    assert r1["sims_clicked"] == ["sim1"]


def test_last_room_detection(mock_movement_path):
    """Verify accurate detection of the final room in a run."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.zone_routines = {
        "pink_zones": {
            f"pink_{i}": {"name": f"Pink Dot #{i}"} for i in range(1, 8)
        }
    }

    assert nav._get_last_pink_room_key() == "pink_7"
    assert nav._is_last_room("pink_7") is True
    assert nav._is_last_room("Pink Dot #7") is True
    assert nav._is_last_room("pink_1") is False
    assert nav._is_last_room("pink_6") is False

    # Test with custom 5-room setup
    nav.zone_routines = {
        "pink_zones": {
            f"pink_{i}": {"name": f"Pink Dot #{i}"} for i in range(1, 6)
        }
    }
    assert nav._get_last_pink_room_key() == "pink_5"
    assert nav._is_last_room("pink_5") is True
    assert nav._is_last_room("pink_7") is False


def test_run_finalization_and_persistence(mock_movement_path, temp_log_dir):
    """Verify that run completion writes structured JSON and human-readable TXT to disk."""
    json_path = str(temp_log_dir / "run_history.json")
    txt_path = str(temp_log_dir / "run_history.txt")
    last_run_path = str(temp_log_dir / "last_run.json")

    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = json_path
    nav.run_summary_file = txt_path

    nav._start_new_run()
    # Room 1
    nav.current_room_key = "pink_1"
    nav._record_sim_clicked("sim1")
    nav._record_room_cleared("pink_1", routine_duration=25.0)

    # Room 7 (Last room)
    nav.current_room_key = "pink_7"
    nav._record_sim_clicked("sim1")
    nav._record_sim_clicked("sim3")
    nav._record_room_cleared("pink_7", routine_duration=30.0)

    # Finalize run
    summary = nav._finalize_run(last_room="pink_7", reason="last_room_cleared")

    assert summary["completed"] is True
    assert summary["completion_reason"] == "last_room_cleared"
    assert summary["last_room_cleared"] == "pink_7"
    assert summary["total_rooms_cleared"] == 2
    assert summary["rooms_cleared"] == ["pink_1", "pink_7"]
    assert summary["sims_total_count"] == 3
    assert summary["sims_clicked_by_room"] == {
        "pink_1": ["sim1"],
        "pink_7": ["sim1", "sim3"],
    }
    assert summary["sims_counts"]["sim1"] == 2
    assert summary["sims_counts"]["sim3"] == 1
    assert summary["sims_counts"]["total"] == 3

    # Check JSON history file exists and contains valid data
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["run_id"] == summary["run_id"]
    assert data[0]["last_room_cleared"] == "pink_7"
    assert data[0]["sims_clicked_by_room"]["pink_7"] == ["sim1", "sim3"]

    # Check TXT summary file exists and contains formatted block
    assert os.path.exists(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()
    assert summary["run_id"] in txt_content
    assert "Last Room Cleared: pink_7" in txt_content
    assert "pink_1: sim1" in txt_content
    assert "pink_7: sim1, sim3" in txt_content

    # Check last_run.json exists
    assert os.path.exists(last_run_path)
    with open(last_run_path, "r", encoding="utf-8") as f:
        last_data = json.load(f)
    assert last_data["run_id"] == summary["run_id"]

    # Verify append behavior: Run a second session
    nav._start_new_run()
    nav.current_room_key = "pink_1"
    nav._record_sim_clicked("sim1")
    nav._record_room_cleared("pink_1", routine_duration=20.0)
    nav._finalize_run(last_room="pink_1", reason="stopped_by_user")

    with open(json_path, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    assert len(data2) == 2
    assert data2[1]["completion_reason"] == "stopped_by_user"


def test_pause_active_duration_tracking(mock_movement_path, temp_log_dir):
    """Verify that pausing tracks pause duration and calculates active duration separately."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = str(temp_log_dir / "run_history.json")
    nav.run_summary_file = str(temp_log_dir / "run_history.txt")

    nav.is_active = True
    nav._start_new_run()

    # Simulate pause of 0.1s
    nav.pause()
    assert nav.is_paused is True
    assert nav.run_pause_time is not None
    time.sleep(0.1)

    with patch.object(nav, "move_mouse_inside_game"):
        nav.resume()
    assert nav.is_paused is False
    assert nav.run_pause_time is None
    assert nav.total_paused_duration >= 0.08

    summary = nav._finalize_run(last_room="pink_1", reason="test")
    assert summary["paused_seconds"] >= 0.08
    assert summary["active_duration_seconds"] <= summary["duration_seconds"]


def test_zone_routine_auto_finalizes_on_last_room(mock_movement_path, temp_log_dir):
    """Verify that completing the last room routine automatically finalizes the run."""
    json_path = str(temp_log_dir / "run_history.json")
    txt_path = str(temp_log_dir / "run_history.txt")

    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = json_path
    nav.run_summary_file = txt_path
    nav.zone_routines = {
        "pink_zones": {
            "pink_7": {
                "name": "Pink Dot #7 (WP #74 - Room 7)",
                "steps": [
                    {"action": "stop", "duration": 0.01, "description": "Halt"}
                ]
            }
        }
    }

    nav.is_active = True
    nav._start_new_run()

    with patch.object(nav, "_execute_zone_routine_step", return_value=True):
        res = nav._execute_zone_routine(
            routine=nav.zone_routines["pink_zones"]["pink_7"],
            zone_label="PINK DOT",
            target=mock_movement_path.waypoints[3]  # WP 74 (pink_7)
        )

    assert res is True
    assert nav.run_completed is True
    assert nav.run_last_room_cleared == "pink_7"
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["last_room_cleared"] == "pink_7"
    assert data[0]["completed"] is True


def test_loot_recording_and_room_grouping(mock_movement_path, temp_log_dir):
    """Verify that loot pickups are tracked with room keys, grouped per room, and serialized to JSON and TXT."""
    from src.loot_detector import LootItem

    json_path = str(temp_log_dir / "run_history.json")
    txt_path = str(temp_log_dir / "run_history.txt")

    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = json_path
    nav.run_summary_file = txt_path
    nav._start_new_run()

    item_divine = LootItem(
        x=100, y=120, w=150, h=35, center_x=175, center_y=137,
        rule_id="tier1_divine", rule_name="Divine Orb", priority=1, confidence=0.96
    )
    item_annul = LootItem(
        x=200, y=250, w=130, h=30, center_x=265, center_y=265,
        rule_id="tier2_annul", rule_name="Orb of Annulment", priority=2, confidence=0.91
    )

    # Room 1: 1 Divine Orb
    nav.current_room_key = "pink_1"
    nav._record_loot_picked(item_divine, pos=(175, 137), room_key="pink_1")
    nav._record_room_cleared("pink_1", routine_duration=28.0)

    # Room 2: 1 Divine Orb + 1 Orb of Annulment
    nav.current_room_key = "pink_2"
    nav._record_loot_picked(item_divine, pos=(300, 400), room_key="pink_2")
    nav._record_loot_picked(item_annul, pos=(350, 420), room_key="pink_2")
    nav._record_room_cleared("pink_2", routine_duration=35.0)

    # Verify clearance records
    r1 = nav.run_rooms_cleared[0]
    assert r1["loot_count"] == 1
    assert r1["loot_collected"] == ["Divine Orb"]

    r2 = nav.run_rooms_cleared[1]
    assert r2["loot_count"] == 2
    assert "Orb of Annulment" in r2["loot_collected"]
    assert "Divine Orb" in r2["loot_collected"]

    # Finalize run
    summary = nav._finalize_run(last_room="pink_2", reason="last_room_cleared")

    assert summary["total_loot_collected"] == 3
    assert summary["loot_counts"]["Divine Orb"] == 2
    assert summary["loot_counts"]["Orb of Annulment"] == 1
    assert summary["loot_counts"]["total"] == 3
    assert summary["loot_by_room"]["pink_1"] == ["Divine Orb"]
    assert len(summary["loot_by_room"]["pink_2"]) == 2
    assert len(summary["loot_log"]) == 3

    # Check JSON
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert json_data[0]["total_loot_collected"] == 3
    assert json_data[0]["loot_counts"]["Divine Orb"] == 2
    assert json_data[0]["loot_by_room"]["pink_1"] == ["Divine Orb"]

    # Check TXT
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()
    assert "Loot Collected (3 total):" in txt_content
    assert "pink_1 (1 items): Divine Orb" in txt_content
    assert "pink_2 (2 items):" in txt_content
    assert "[Loot: 1 item]" in txt_content
    assert "[Loot: 2 items]" in txt_content


def test_collect_loot_integration_records_item(mock_movement_path, temp_log_dir):
    """Verify that calling collect_loot records the item and updates run telemetry."""
    from src.loot_detector import LootItem

    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = str(temp_log_dir / "run_history.json")
    nav.run_summary_file = str(temp_log_dir / "run_history.txt")
    nav.is_active = True
    nav.loot_z_toggle_enabled = False  # Speed up test by skipping Z sleeps
    nav.loot_approach_wait_seconds = 0.0
    nav._start_new_run()
    nav.current_room_key = "pink_3"

    mock_item = LootItem(
        x=500, y=500, w=100, h=30, center_x=550, center_y=515,
        rule_id="tier1_mirror", rule_name="Mirror of Kalandra", priority=1, confidence=0.99
    )

    call_count = 0
    def mock_locate_loot(exclude_positions=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            nav._last_clicked_loot_item = mock_item
            return (550, 515)
        return None

    with patch.object(nav, "locate_loot", side_effect=mock_locate_loot):
        with patch.object(nav, "move_mouse_inside_game", return_value=(550, 515)):
            with patch("src.route_navigator.pydirectinput.click"):
                with patch("src.route_navigator.pydirectinput.mouseUp"):
                    picked = nav.collect_loot(max_pickups=1, approach_wait=0.0)

    assert picked == 1
    assert len(nav.run_loot_picked) == 1
    entry = nav.run_loot_picked[0]
    assert entry["name"] == "Mirror of Kalandra"
    assert entry["room"] == "pink_3"
    assert entry["screen_pos"] == [550, 515]

    telem = nav.get_telemetry(target=None, dist=0.0, keys=[])
    assert telem["run_loot_count"] == 1


def test_click_hideout_portal_starts_timer_and_resets_to_pink_1(mock_movement_path, temp_log_dir):
    """Verify that entering hideout portal starts run timer and resets route to always start at Pink Dot 1 (WP 0)."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    nav.run_history_file = str(temp_log_dir / "run_history.json")
    nav.run_summary_file = str(temp_log_dir / "run_history.txt")

    # Simulate dirty state from a previous map
    nav.interacted_pink_dots.add(1)
    nav.interacted_pink_dots.add(74)
    nav.interacted_zones.add("zone_1")
    nav.movement_path.current_idx = 74
    nav.start_at_pink_dot = 3
    nav.run_completed = True
    nav.latest_pos = (400, 400)
    nav.last_known_pos = (400, 400)

    with patch.object(nav, "locate_hideout_portal", return_value=(960, 540)), \
         patch.object(nav, "is_in_hideout", return_value=False), \
         patch.object(nav, "move_mouse_inside_game"), \
         patch("src.hideout.map_traverse.pydirectinput"):

        ok = nav.click_hideout_portal(
            timeout=1.0,
            approach_wait=0.0,
            verify_transition=True,
            auto_start_route=True,
            start_pink_dot=1,
            hold_w_seconds=0.0,
            settle_wait=0.0,
        )

    assert ok is True
    # Verify timer started immediately upon entering portal
    assert nav.run_start_time is not None
    assert nav.run_completed is False
    assert nav.run_id is not None

    # Verify complete reset to start from first pink dot (WP 0)
    assert len(nav.interacted_pink_dots) == 0
    assert len(nav.interacted_zones) == 0
    assert nav.movement_path.current_idx == 0
    nav.stop()


def test_room_7_looting_stops_timer_and_records_history(mock_movement_path, temp_log_dir):
    """Verify that completing pickup_loot in Room 7 immediately stops the run timer and appends to run history."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    json_path = str(temp_log_dir / "run_history.json")
    txt_path = str(temp_log_dir / "run_history.txt")
    nav.run_history_file = json_path
    nav.run_summary_file = txt_path
    nav.is_active = True
    nav.loot_z_toggle_enabled = False
    nav.wait_for_loot_confirmation = False

    # Start run
    nav._start_new_run()
    assert nav.run_completed is False

    # Set up Room 7 context
    nav.current_room_key = "pink_7"
    nav.zone_routines = {
        "pink_zones": {
            f"pink_{i}": {"name": f"Pink Dot #{i}"} for i in range(1, 8)
        }
    }

    # Execute pickup_loot step in Room 7
    loot_step = {
        "action": "pickup_loot",
        "max_items": 5,
        "approach_wait": 0.0,
        "wait_for_green_light": False,
    }

    with patch.object(nav, "collect_loot", return_value=2):
        success = nav._execute_zone_routine_step(loot_step, context={}, zone_label="Pink Dot #7")

    assert success is True
    # Timer must have stopped
    assert nav.run_completed is True
    assert nav.run_end_time is not None
    assert nav.run_last_room_cleared == "pink_7"

    # Verify JSON file has the completed run
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["completed"] is True
    assert data[0]["completion_reason"] == "room_7_looting_completed"
    assert data[0]["last_room_cleared"] == "pink_7"

    # Verify TXT file contains summary
    assert os.path.exists(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        txt = f.read()
    assert "Last Room Cleared: pink_7" in txt
    nav.stop()


def test_multi_map_timer_lifecycle(mock_movement_path, temp_log_dir):
    """Verify that multiple successive maps cleanly start on portal entry and finalize on Room 7 looting."""
    nav = RouteNavigator(movement_path=mock_movement_path)
    json_path = str(temp_log_dir / "run_history.json")
    txt_path = str(temp_log_dir / "run_history.txt")
    nav.run_history_file = json_path
    nav.run_summary_file = txt_path
    nav.is_active = True
    nav.loot_z_toggle_enabled = False
    nav.wait_for_loot_confirmation = False
    nav.zone_routines = {
        "pink_zones": {
            f"pink_{i}": {"name": f"Pink Dot #{i}"} for i in range(1, 8)
        }
    }

    loot_step = {"action": "pickup_loot", "max_items": 5, "approach_wait": 0.0, "wait_for_green_light": False}

    # === MAP 1 ===
    with patch.object(nav, "locate_hideout_portal", return_value=(960, 540)), \
         patch.object(nav, "is_in_hideout", return_value=False), \
         patch.object(nav, "move_mouse_inside_game"), \
         patch("src.hideout.map_traverse.pydirectinput"):
        nav.click_hideout_portal(timeout=1.0, approach_wait=0.0, auto_start_route=True, hold_w_seconds=0.0, settle_wait=0.0)

    run_1_id = nav.run_id
    assert nav.run_completed is False
    assert nav.movement_path.current_idx == 0

    # Loot Room 7
    nav.current_room_key = "pink_7"
    with patch.object(nav, "collect_loot", return_value=1):
        nav._execute_zone_routine_step(loot_step, context={}, zone_label="Pink Dot #7")
    assert nav.run_completed is True

    # === MAP 2 ===
    time.sleep(0.02)
    with patch.object(nav, "locate_hideout_portal", return_value=(960, 540)), \
         patch.object(nav, "is_in_hideout", return_value=False), \
         patch.object(nav, "move_mouse_inside_game"), \
         patch("src.hideout.map_traverse.pydirectinput"):
        nav.click_hideout_portal(timeout=1.0, approach_wait=0.0, auto_start_route=True, hold_w_seconds=0.0, settle_wait=0.0)

    run_2_id = nav.run_id
    assert run_2_id != run_1_id
    assert nav.run_completed is False
    assert nav.movement_path.current_idx == 0

    # Loot Room 7 in Map 2
    nav.current_room_key = "pink_7"
    with patch.object(nav, "collect_loot", return_value=1):
        nav._execute_zone_routine_step(loot_step, context={}, zone_label="Pink Dot #7")
    assert nav.run_completed is True

    # Verify both runs persisted in run_history.json
    with open(json_path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    assert len(runs) == 2
    assert runs[0]["run_id"] == run_1_id
    assert runs[1]["run_id"] == run_2_id
    assert runs[0]["completed"] is True
    assert runs[1]["completed"] is True
    nav.stop()


