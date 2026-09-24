"""
High-level pink marker and yellow zone interaction lifecycle orchestration.
"""
from __future__ import annotations

import os
import time
import math
import json
import threading
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np

from src.movement_path import MovementPath
from src.screen_capturer import ScreenCapturer
from src.loot_detector import LootDetector, LootItem
from src.minimap_extractor import MinimapExtractor
from src.enemy_detector import EnemyDetector
from src.navigator_base import DynamicModuleProxy, log_msg

_log = log_msg
pydirectinput = DynamicModuleProxy("pydirectinput")
window_focuser = DynamicModuleProxy("window_focuser")
stop_handler = DynamicModuleProxy("stop_handler")
keyboard = DynamicModuleProxy("keyboard")


class ZoneInteractionsMixin:
    """High-level pink marker and yellow zone interaction lifecycle orchestration."""

    def execute_yellow_zone_interaction(self, orbit_zone: Optional[Dict[str, Any]] = None) -> bool:
        """
        Executes sequence upon arriving at yellow zone based on configured options:
        1. Halt WASD movement & ensure game window focus.
        2. If click_banner_enabled: Locate and click on encounter banner inside game window.
        3. If right_click_after_banner_enabled: Click right mouse button once inside game window.
        4. If middle_click_hold_enabled: Press and hold middle mouse button for configured duration inside game window.
        """
        self.is_interacting = True
        self.stuck_counter = 0
        self.release_all_keys()
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        try:
            self._routine_did_orbit = False
            # Check for custom zone routine first
            custom_routine = self._get_yellow_zone_routine(orbit_zone)
            if custom_routine and custom_routine.get("steps"):
                z_id = orbit_zone.get("id", "zone") if isinstance(orbit_zone, dict) else "zone"
                return self._execute_zone_routine(custom_routine, zone_label=f"YELLOW ZONE ({z_id})", zone=orbit_zone)

            # Ensure cursor starts inside the game window and loot labels are visible for SIM / banner detection
            self.move_mouse_inside_game()
            self.ensure_loot_labels_visible()

            # Check if all interactions are disabled
            if not self.click_banner_enabled and not self.right_click_after_banner_enabled and not self.middle_click_hold_enabled:
                return True

            self.status_message = "[YELLOW ZONE] Interacting..."
            _log("\n[AUTOPILOT] >>> REACHED YELLOW ZONE! Executing encounter activation sequence...")

            # 0. Check for Sims first! (Fallback safety in case dynamic resync landed on yellow zone directly)
            sims = self._detect_and_click_sims(prefix="[YELLOW ZONE]")
            self.hide_loot_labels()
            if sims:
                _log(f"  [YELLOW ZONE] Sims selected ({', '.join(sims)}). Proceeding to locate Encounter Banner...")

            # 1. Locate and click banner if enabled
            if self.click_banner_enabled:
                self.status_message = "[YELLOW ZONE] Finding Encounter Banner..."
                banner_pos = None
                for attempt in range(1, self.banner_search_attempts + 1):
                    if stop_handler.is_stopped() or not self.is_active:
                        return False
                    banner_pos = self.locate_encounter_banner()
                    if banner_pos is not None:
                        break
                    time.sleep(0.35)

                # Retry: If banner not found, walk closer to pink dot again and re-check
                if banner_pos is None and not stop_handler.is_stopped() and self.is_active:
                    pink_pos = None
                    if orbit_zone and isinstance(orbit_zone, dict):
                        pink_pos = orbit_zone.get("pink_pos") or orbit_zone.get("center")
                    if pink_pos:
                        _log(f"  [BANNER RETRY] Banner not detected on screen. Walking closer to pink dot at ({pink_pos[0]:.1f}, {pink_pos[1]:.1f}) and re-searching...")
                        self.status_message = "[YELLOW ZONE] Walking to Pink Dot Retry..."
                        self._walk_to_coordinate(
                            (float(pink_pos[0]), float(pink_pos[1])),
                            label="NAV→BANNER-RETRY",
                            timeout=5.0,
                            arrival_threshold=8.0,
                        )
                        time.sleep(0.4)
                        for retry_attempt in range(1, self.banner_search_attempts + 1):
                            if stop_handler.is_stopped() or not self.is_active:
                                return False
                            banner_pos = self.locate_encounter_banner()
                            if banner_pos is not None:
                                _log(f"  [BANNER RETRY SUCCESS] Encounter banner detected on retry at ({banner_pos[0]}, {banner_pos[1]})!")
                                break
                            time.sleep(0.35)

                if banner_pos is not None:
                    bx, by = self.move_mouse_inside_game(banner_pos[0], banner_pos[1])
                    _log(f"  [ACTION 1/3] Clicking encounter banner at ({bx}, {by})...")
                    self.status_message = f"[YELLOW ZONE] Clicking Banner ({bx}, {by})"
                    if pydirectinput:
                        pydirectinput.click()
                        time.sleep(0.08)
                        pydirectinput.mouseUp(button="left")
                    time.sleep(0.15)

                    # Approach wait: allow character to walk closer to banner before proceeding to next action
                    if self.banner_approach_wait_seconds > 0:
                        self._wait_for_approach(self.banner_approach_wait_seconds, reason="YELLOW ZONE BANNER")
                        if stop_handler.is_stopped() or not self.is_active:
                            return False
                        if self.reclick_after_approach:
                            in_range_pos = self.locate_encounter_banner()
                            if in_range_pos is not None:
                                rx, ry = self.move_mouse_inside_game(in_range_pos[0], in_range_pos[1])
                                _log(f"  [ACTION 1/3] Re-clicking encounter banner in-range at ({rx}, {ry})...")
                                if pydirectinput:
                                    pydirectinput.click()
                                    time.sleep(0.08)
                                    pydirectinput.mouseUp(button="left")
                                time.sleep(0.15)
                else:
                    _log(f"  [WARNING] Encounter banner not detected on screen after {self.banner_search_attempts} attempts. Cursor positioned inside game.")
                    self.move_mouse_inside_game()
            else:
                _log("  [CONFIG] Banner clicking disabled (click_banner_enabled=false). Skipping...")
                self.move_mouse_inside_game()

            # Hide loot labels after sims/banner interaction before combat/orbit
            self.hide_loot_labels()

            if stop_handler.is_stopped() or not self.is_active:
                return False

            # 2. Press 'T' key skill once if enabled
            if self.right_click_after_banner_enabled:
                _log("  [ACTION 2/3] Pressing 'T' key skill once...")
                self.status_message = "[YELLOW ZONE] Pressing 'T' Skill..."
                if pydirectinput:
                    pydirectinput.keyDown("t")
                    time.sleep(0.04)
                    pydirectinput.keyUp("t")
                time.sleep(0.15)
            else:
                _log("  [CONFIG] Skill after banner disabled. Skipping...")

            if stop_handler.is_stopped() or not self.is_active:
                return False

            # 3. Press and hold 'Q' key for configured duration or enemy proximity if enabled
            if self.middle_click_hold_enabled:
                hold_sec = self.middle_click_hold_seconds
                self.execute_hold_q("YELLOW ZONE", hold_sec)
            else:
                _log("  [CONFIG] Q key hold disabled (middle_click_hold_enabled=false). Skipping...")

            return True
        finally:
            self.is_interacting = False
            now = time.time()
            self.last_progress_pos = self.latest_pos
            self.last_progress_time = now
            self.last_known_time = now
            self.stuck_counter = 0


    def execute_pink_dot_interaction(self, target: Optional[Dict[str, Any]] = None) -> bool:
        """
        Executes sequence upon arriving at a PINK dot on the route:
        1. Halt character movement for pink_dot_stop_seconds.
        2. Ensure game window focus and move mouse inside the game window.
        3. Attempt to detect and select sim types (sim1 -> wait 2s -> sim3 -> wait 2s -> sim2).
        4. If no sims are available:
           - Click encounter_banner inside game window.
           - Click right mouse button once.
           - Press and hold middle mouse button for 4 seconds (middle_click_hold_seconds), then release.
        """
        if getattr(self, "in_hideout", False):
            _log("  [AUTOPILOT] Ignoring pink dot interaction: Character is in Hideout (Safe Zone).")
            return False

        self.is_interacting = True
        self.stuck_counter = 0
        self.release_all_keys()
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        try:
            self._routine_did_orbit = False
            return self._do_execute_pink_dot_interaction(target)
        finally:
            self.is_interacting = False
            now = time.time()
            self.last_progress_pos = self.latest_pos
            self.last_progress_time = now + (4.0 if self.is_orbiting else 0.0)
            self.last_known_time = now
            self.stuck_counter = 0


    def _do_execute_pink_dot_interaction(self, target: Optional[Dict[str, Any]] = None) -> bool:
        # Check for custom zone routine first
        custom_routine = self._get_pink_zone_routine(target)
        if custom_routine and custom_routine.get("steps"):
            return self._execute_zone_routine(custom_routine, zone_label="PINK DOT", target=target)

        fallback_room_key = self._resolve_room_key({}, zone_label="PINK DOT", target=target)
        prev_room_key = self.current_room_key
        self.current_room_key = fallback_room_key
        fallback_start_time = time.time()

        self.release_all_keys()
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.move_mouse_inside_game()
        self.ensure_loot_labels_visible()

        self.status_message = "[PINK DOT] Arrived! Stopping movement..."
        _log(f"\n[AUTOPILOT] >>> REACHED PINK DOT! Stopping character for {self.pink_dot_stop_seconds:.1f}s...")

        # 1. Halt movement for configured seconds
        stop_start = time.time()
        while (time.time() - stop_start) < self.pink_dot_stop_seconds:
            if stop_handler.is_stopped() or not self.is_active:
                return False
            rem_stop = max(0.0, self.pink_dot_stop_seconds - (time.time() - stop_start))
            self.status_message = f"[PINK DOT] Stopped ({rem_stop:.1f}s)..."
            time.sleep(0.05)

        if stop_handler.is_stopped() or not self.is_active:
            return False

        # 2. Attempt to detect and click Sims: sim1 -> wait 2s -> sim3 -> wait 2s -> sim2
        sims_clicked = self._detect_and_click_sims(prefix="[PINK DOT]")
        self.hide_loot_labels()

        if stop_handler.is_stopped() or not self.is_active:
            return False

        pink_pos = None
        if target and isinstance(target, dict):
            pink_pos = target.get("pink_pos") or (target.get("x"), target.get("y"))

        # If character moved or sims were clicked, walk back to the pink dot location to ensure banner is in range
        if sims_clicked and pink_pos:
            self._walk_to_coordinate(
                (float(pink_pos[0]), float(pink_pos[1])),
                label="NAV→BANNER",
                timeout=5.0,
                arrival_threshold=15.0,
            )
            time.sleep(0.3)

        if stop_handler.is_stopped() or not self.is_active:
            return False

        # 3. Locate and click Encounter Banner
        if self.click_banner_enabled:
            self.status_message = "[PINK DOT] Finding Encounter Banner..."
            banner_pos = None
            for attempt in range(1, self.banner_search_attempts + 1):
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                banner_pos = self.locate_encounter_banner()
                if banner_pos is not None:
                    break
                time.sleep(0.25)

            # Retry: If banner not found, walk closer to pink dot again and re-check
            if banner_pos is None and not stop_handler.is_stopped() and self.is_active:
                if pink_pos:
                    _log(f"  [BANNER RETRY] Banner not detected on screen. Walking closer to pink dot at ({pink_pos[0]:.1f}, {pink_pos[1]:.1f}) and re-searching...")
                    self.status_message = "[PINK DOT] Walking to Pink Dot Retry..."
                    self._walk_to_coordinate(
                        (float(pink_pos[0]), float(pink_pos[1])),
                        label="NAV→BANNER-RETRY",
                        timeout=5.0,
                        arrival_threshold=8.0,
                    )
                    time.sleep(0.4)
                    for retry_attempt in range(1, self.banner_search_attempts + 1):
                        if stop_handler.is_stopped() or not self.is_active:
                            return False
                        banner_pos = self.locate_encounter_banner()
                        if banner_pos is not None:
                            _log(f"  [BANNER RETRY SUCCESS] Encounter banner detected on retry at ({banner_pos[0]}, {banner_pos[1]})!")
                            break
                        time.sleep(0.25)

            if banner_pos is not None:
                bx, by = self.move_mouse_inside_game(banner_pos[0], banner_pos[1])
                _log(f"  [ACTION 1/3] Clicking encounter banner at ({bx}, {by})...")
                self.status_message = f"[PINK DOT] Clicking Banner ({bx}, {by})"
                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.08)
                    pydirectinput.mouseUp(button="left")
                time.sleep(0.15)

                # Approach wait: allow character to walk closer to banner before proceeding to next action
                if self.banner_approach_wait_seconds > 0:
                    self._wait_for_approach(self.banner_approach_wait_seconds, reason="PINK DOT BANNER")
                    if stop_handler.is_stopped() or not self.is_active:
                        return False

                # Double-check / Verification: Check after approach/wait if encounter banner is still visible on screen
                if self.banner_verify_click_enabled and not stop_handler.is_stopped() and self.is_active:
                    if self.banner_approach_wait_seconds <= 0 and self.banner_verify_delay_seconds > 0:
                        time.sleep(self.banner_verify_delay_seconds)

                    for attempt_num in range(1, max(1, self.banner_max_click_attempts)):
                        if stop_handler.is_stopped() or not self.is_active:
                            break
                        recheck_pos = self.locate_encounter_banner()
                        if recheck_pos is not None:
                            rx, ry = self.move_mouse_inside_game(recheck_pos[0], recheck_pos[1])
                            _log(f"  [BANNER DOUBLE-CHECK] Encounter banner is STILL visible on screen after approach. Re-clicking in-range at ({rx}, {ry}) (attempt {attempt_num + 1}/{self.banner_max_click_attempts})...")
                            self.status_message = f"[PINK DOT] Re-clicking Banner ({rx}, {ry})"
                            if pydirectinput:
                                pydirectinput.click()
                                time.sleep(0.08)
                                pydirectinput.mouseUp(button="left")
                            time.sleep(max(0.4, self.banner_verify_delay_seconds))
                        else:
                            _log(f"  [BANNER VERIFIED] Encounter banner click confirmed (no longer visible on screen).")
                            break
                elif self.reclick_after_approach:
                    in_range_pos = self.locate_encounter_banner()
                    if in_range_pos is not None:
                        rx, ry = self.move_mouse_inside_game(in_range_pos[0], in_range_pos[1])
                        _log(f"  [ACTION 1/3] Re-clicking encounter banner in-range at ({rx}, {ry})...")
                        if pydirectinput:
                            pydirectinput.click()
                            time.sleep(0.08)
                            pydirectinput.mouseUp(button="left")
                        time.sleep(0.15)
            else:
                _log(f"  [WARNING] Encounter banner not detected on screen after {self.banner_search_attempts} attempts. Positioning cursor inside game.")
                self.move_mouse_inside_game()
        else:
            _log("  [CONFIG] Banner clicking disabled (click_banner_enabled=false). Skipping...")
            self.move_mouse_inside_game()

        # Hide loot labels after sims/banner interaction before combat/orbit
        self.hide_loot_labels()

        if stop_handler.is_stopped() or not self.is_active:
            return False

        # Initial skills & hold actions (run once on first encounter)
        if not self.has_executed_initial_hold:
            # Press 'T' key skill once if enabled
            if getattr(self, "right_click_after_banner_enabled", True):
                _log("  [ACTION 2/3] Pressing 'T' key skill once...")
                self.status_message = "[PINK DOT] Pressing 'T' Skill..."
                if pydirectinput:
                    pydirectinput.keyDown("t")
                    time.sleep(0.04)
                    pydirectinput.keyUp("t")
                time.sleep(0.15)

            if stop_handler.is_stopped() or not self.is_active:
                return False

            # Press and hold 'Q' key for configured seconds or enemy proximity if enabled
            if getattr(self, "middle_click_hold_enabled", True):
                hold_sec = self.middle_click_hold_seconds
                self.execute_hold_q("PINK DOT", hold_sec)

            self.has_executed_initial_hold = True
            self.enable_persistent_combat()

        # 4. Start orbiting inside yellow marker for defined duration (e.g. 50s)
        orbit_zones = self.movement_path.get_orbit_zones()
        c_pos = self.latest_pos or ((target["x"], target["y"]) if target else (0.0, 0.0))
        best_zone = None
        best_d = float("inf")
        for z in orbit_zones:
            zc = z.get("center", [0, 0])
            d = math.hypot(zc[0] - c_pos[0], zc[1] - c_pos[1])
            if d < best_d:
                best_d = d
                best_zone = z

        if best_zone is not None:
            zone_id = best_zone.get("id")
            if zone_id:
                self.interacted_zones.add(zone_id)
            self.is_orbiting = True
            self.orbit_start_time = time.time()
            self.last_orbit_right_click = time.time()
            self.orbit_duration = float(best_zone.get("duration", getattr(self.movement_path, "orbit_duration_seconds", 50.0)))
            self.current_orbit_zone = best_zone
            self.orbit_perimeter_pts = best_zone.get("perimeter_points", [])
            if self.orbit_perimeter_pts:
                best_p_idx = 0
                best_p_dist = float("inf")
                for p_i, p_pt in enumerate(self.orbit_perimeter_pts):
                    d = math.hypot(p_pt[0] - c_pos[0], p_pt[1] - c_pos[1])
                    if d < best_p_dist:
                        best_p_dist = d
                        best_p_idx = p_i
                self.orbit_point_idx = best_p_idx
            self.orbit_grace_until = time.time() + 4.0
            self.stuck_counter = 0
            self.last_progress_pos = self.latest_pos or c_pos
            self.last_progress_time = time.time() + 4.0
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            self.move_mouse_inside_game()
            _log(f"  [AUTOPILOT] Pink dot encounter complete -> Starting orbit inside yellow shape ({best_zone.get('id', 'zone')}) for {self.orbit_duration:.1f}s...")
            self.status_message = f"Orbiting Yellow Zone ({self.orbit_duration:.1f}s left)"
        else:
            _log("  [PINK DOT] No yellow orbit shape found on route. Scanning for loot...")
            self.collect_loot()
            if self.wait_for_loot_confirmation:
                self.release_all_keys()
                self.waiting_for_green_light = True
                self.status_message = "WAITING FOR GREEN LIGHT (Verify Loot Pickup - Press 'G' to Resume)"
                _log("\n[AUTOPILOT] >>> LOOT PICKUP FINISHED! Waiting for GREEN LIGHT to continue...")

        if not self.is_orbiting:
            self.status_message = "[PINK DOT] Sequence completed. Resuming route..."
            _log("  [AUTOPILOT] Finished Pink Dot interaction! Resuming green route navigation...")
        else:
            self.status_message = f"Orbiting Yellow Zone ({self.orbit_duration:.1f}s left)"

        if fallback_room_key:
            self._record_room_cleared(fallback_room_key, time.time() - fallback_start_time)
            if self._is_last_room(fallback_room_key):
                self._finalize_run(last_room=fallback_room_key, reason="last_room_cleared")
        self.current_room_key = prev_room_key
        return True

