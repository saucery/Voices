"""
Stuck detection, inverse movement backtracking, un-stuck maneuvers, and waypoint skipping.
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


class StuckRecoveryMixin:
    """Stuck detection, inverse movement backtracking, un-stuck maneuvers, and waypoint skipping."""

    def skip_current_waypoint(self) -> Optional[Dict[str, Any]]:
        """Skips the current target waypoint and advances immediately to the next one."""
        if not self.movement_path.is_configured:
            return None
        self.is_orbiting = False
        self.current_orbit_zone = None
        self.orbit_perimeter_pts = []
        curr_idx = self.movement_path.current_idx
        if curr_idx < len(self.movement_path.waypoints) - 1:
            self.movement_path.current_idx = curr_idx + 1
            next_wp = self.movement_path.get_current_target()
            name = next_wp.get("name", f"WP #{self.movement_path.current_idx}") if next_wp else f"WP #{self.movement_path.current_idx}"
            msg = f"SKIPPED: WP #{curr_idx} -> Target is {name}"
            self.latest_recovery_event = msg
            self.status_message = msg
            _log(f"\n[NAVIGATOR] Skipped WP #{curr_idx} -> Target is now {name} (WP #{self.movement_path.current_idx})")
            return next_wp
        else:
            self.is_completed = True
            self.is_active = False
            self.release_all_keys()
            self.status_message = "Route Finished!"
            _log(f"\n[NAVIGATOR] Skipped final waypoint. Destination reached.")
            return None


    def _execute_inverse_movement_backtrack(self, pulses: int = 2, reason: str = "Stuck") -> List[str]:
        """
        Executes an inverse movement sequence to back out of obstacles or retrace steps:
        - Inverts recent movement keys (w <-> s, a <-> d).
        - Sends reverse pulses with optional space roll.
        - Pauses briefly so minimap/camera can stabilize and re-localize.
        """
        self.release_all_keys()
        opp = {"w": "s", "s": "w", "a": "d", "d": "a", "space": "space"}

        # 1. Inspect recent movements to find keys to invert
        backtrack_keys = []
        if self.recent_movements:
            for keys, _, _, _ in reversed(list(self.recent_movements)):
                inv_for_step = [opp[k] for k in keys if k in opp]
                if inv_for_step:
                    backtrack_keys = inv_for_step
                    break

        if not backtrack_keys and self.last_held_keys:
            backtrack_keys = [opp[k] for k in self.last_held_keys if k in opp]

        if not backtrack_keys and self.latest_pos and self.last_known_pos:
            dist_to_last = math.hypot(self.last_known_pos[0] - self.latest_pos[0], self.last_known_pos[1] - self.latest_pos[1])
            if dist_to_last > 4.0:
                backtrack_keys = self.compute_wasd_keys(self.latest_pos, self.last_known_pos)

        if not backtrack_keys:
            backtrack_keys = ["s"]

        _log(f"[AUTOPILOT RECOVERY] Executing inverse movement backtrack (keys: {backtrack_keys}, pulses: {pulses}) for {reason}...")

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.is_simulating_key = True
        self.held_keys = set(backtrack_keys)
        try:
            for p_i in range(pulses):
                if not self.is_active or stop_handler.is_stopped():
                    break
                if pydirectinput:
                    for k in backtrack_keys:
                        try:
                            pydirectinput.keyDown(k)
                        except Exception:
                            pass
                    time.sleep(self.step_duration)
                    for k in backtrack_keys:
                        try:
                            pydirectinput.keyUp(k)
                        except Exception:
                            pass
                    time.sleep(0.06)
                else:
                    time.sleep(self.step_duration)
        finally:
            self.is_simulating_key = False
            self.held_keys.clear()
            time.sleep(0.20)  # Camera/minimap settle pause

        return backtrack_keys


    def _execute_stuck_recovery(self, reason: str = "Stuck"):
        """
        Executes stuck / lost-location recovery maneuver:
        1. Releases current movement keys.
        2. Backtracks in reverse direction of recent movement.
        3. If orbiting yellow zone: switches to next perimeter point in yellow zone.
           If traveling on green route: retries waypoint with backtrack before advancing/skipping.
        4. Resumes navigation towards the active target.
        """
        if self.is_orbiting:
            _log(f"\n[AUTOPILOT RECOVERY] {reason} during Yellow Zone Orbit.")
            _log(f"[AUTOPILOT RECOVERY] Unsticking from geometry inside yellow area...")
            self._execute_inverse_movement_backtrack(pulses=1, reason="Orbit Stuck")

            if self.orbit_perimeter_pts:
                self.orbit_point_idx = (self.orbit_point_idx + 1) % len(self.orbit_perimeter_pts)
            msg = f"ORBIT RECOVERY: Unstuck in yellow zone -> Next point #{self.orbit_point_idx}"
            self.latest_recovery_event = msg
            self.status_message = msg
            _log(f"[AUTOPILOT RECOVERY] Unstuck! Switched to next yellow orbit point #{self.orbit_point_idx}")

            now = time.time()
            self.stuck_counter = 0
            self.orbit_grace_until = now + 4.0
            self.last_progress_pos = self.latest_pos
            self.last_progress_time = now + 4.0
            self.last_known_time = now
            return

        self.is_orbiting = False
        self.current_orbit_zone = None
        self.orbit_perimeter_pts = []
        curr_wp = self.movement_path.get_current_target()
        curr_idx = self.movement_path.current_idx if curr_wp else 0
        curr_name = curr_wp.get("name", f"WP #{curr_idx}") if curr_wp else f"WP #{curr_idx}"

        _log(f"\n[AUTOPILOT RECOVERY] {reason} at {curr_name} (WP #{curr_idx}).")

        # 1. Execute inverse movement backtrack
        self._execute_inverse_movement_backtrack(pulses=2, reason=f"Stuck at WP #{curr_idx}")

        # 2. Skip the waypoint where we got stuck / lost (or trigger encounter if at pink dot)
        if curr_wp and curr_wp.get("action") == "pink_encounter" and curr_idx not in self.interacted_pink_dots:
            _log(f"[AUTOPILOT RECOVERY] Character arrived near Pink Encounter (WP #{curr_idx}). Triggering encounter routine instead of skipping.")
            self.interacted_pink_dots.add(curr_idx)
            self.execute_pink_dot_interaction(curr_wp)
        elif curr_idx < len(self.movement_path.waypoints) - 1:
            self.movement_path.current_idx = curr_idx + 1
            next_wp = self.movement_path.get_current_target()
            next_name = next_wp.get("name", f"WP #{self.movement_path.current_idx}") if next_wp else f"WP #{self.movement_path.current_idx}"
            msg = f"RECOVERY: Skipped WP #{curr_idx} -> Target: {next_name} (WP #{self.movement_path.current_idx})"
            self.latest_recovery_event = msg
            self.status_message = msg
            _log(f"[AUTOPILOT RECOVERY] Backtracked! Skipped WP #{curr_idx} -> Target is now {next_name} (WP #{self.movement_path.current_idx})")
        else:
            self.is_completed = True
            self.is_active = False
            self.status_message = "Route Finished after final recovery!"
            _log("[AUTOPILOT RECOVERY] Final waypoint reached.")

        # Reset stuck detection timers
        now = time.time()
        self.stuck_counter = 0
        self.last_progress_pos = self.latest_pos
        self.last_progress_time = now
        self.last_known_time = now

