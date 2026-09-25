"""
Navigator lifecycle controls: start, stop, pause, resume, toggle, green light, route reload, and pink dot target configuration.
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


class NavigatorControlsMixin:
    """Navigator lifecycle controls: start, stop, pause, resume, toggle, green light, route reload, and pink dot target configuration."""

    def start(self, monitor_idx: Optional[int] = None):
        """Enables autopilot navigation."""
        if not self.movement_path.is_configured:
            self.status_message = "No route waypoints loaded"
            _log("[NAVIGATOR] Cannot start: No route waypoints loaded.")
            return

        if self.in_hideout or self.is_in_hideout():
            self.in_hideout = True
            self.status_message = "Cannot start in Hideout (Safe Zone)"
            _log("[NAVIGATOR] Character is in Hideout (safe space). Cannot start route navigation.")
            return
        else:
            self.in_hideout = False

        if monitor_idx is not None:
            self.monitor_idx = monitor_idx

        # If previously completed or at the end, reset back to starting waypoint (WP #0)
        if self.is_completed or self.movement_path.current_idx >= len(self.movement_path.waypoints) - 1:
            self.is_completed = False
            self.movement_path.current_idx = 0

        now = time.time()
        self.last_known_pos = self.latest_pos
        self.last_known_time = now
        self.last_progress_pos = self.latest_pos
        self.last_progress_time = now
        self.stuck_counter = 0
        self.latest_recovery_event = None

        # Reset orbit state
        self.is_orbiting = False
        self.is_paused = False
        self.current_orbit_zone = None
        self.orbit_perimeter_pts = []
        self.orbit_point_idx = 0

        self.is_active = True
        self.is_completed = False
        self.has_executed_initial_hold = False
        self.status_message = "Autopilot Active (Press 'A' to stop | 'F4' to pause)"
        _log(f"[NAVIGATOR] Autopilot Navigation ACTIVATED. Press 'A' to stop | 'F4' to pause.")

        # Initialize or reset run timer and SIM tracking if starting fresh
        if self.run_start_time is None or self.run_completed:
            self._start_new_run()

        if self.start_at_pink_dot > 0 and self.movement_path.is_configured:
            self.set_start_pink_dot(self.start_at_pink_dot)

        # Bring game to foreground so WASD inputs are directly received by Path Of Exile 2
        window_focuser.focus_game_window(monitor_idx=self.monitor_idx)

        # Launch background navigation worker thread
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._run_navigation_loop, daemon=True)
            self._worker_thread.start()


    def stop(self):
        """Disables autopilot navigation and releases all held keys."""
        self.is_active = False
        self.is_paused = False
        self.waiting_for_green_light = False
        self.waiting_for_user_key = False
        self.is_orbiting = False
        self.current_orbit_zone = None
        self.disable_persistent_right_click()
        self.interacted_zones.clear()
        self.interacted_pink_dots.clear()
        self.release_all_keys()
        self.status_message = "Autopilot Paused (Press 'A' to resume)"
        _log("[NAVIGATOR] Autopilot Navigation STOPPED.")
        if self._worker_thread and self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
            try:
                self._worker_thread.join(timeout=0.5)
            except Exception:
                pass
            self._worker_thread = None
        if self.run_start_time is not None and not self.run_completed and (self.run_rooms_cleared or self.run_sims_clicked):
            self._finalize_run(reason="stopped_by_user")


    def pause(self):
        """Pauses navigation, releases all movement keys, and holds current waypoint position."""
        if not self.is_active:
            return
        if self.run_start_time is not None and self.run_pause_time is None:
            self.run_pause_time = time.time()
        self.is_paused = True
        self.release_all_keys()
        self.status_message = "Autopilot PAUSED (Press 'F4' to resume)"
        curr_wp = self.movement_path.get_current_target()
        wp_name = curr_wp.get("name") if curr_wp else f"WP #{self.movement_path.current_idx}"
        _log(f"\n[AUTOPILOT] >>> PAUSED at WP #{self.movement_path.current_idx} ({wp_name}). Press 'F4' to resume.")


    def resume(self):
        """Resumes navigation towards current target waypoint from where it was paused."""
        if not self.is_active:
            return
        if self.run_pause_time is not None:
            self.total_paused_duration += max(0.0, time.time() - self.run_pause_time)
            self.run_pause_time = None
        self.is_paused = False
        window_focuser.focus_game_window(monitor_idx=self.monitor_idx)
        now = time.time()
        self.last_progress_pos = self.latest_pos
        self.last_progress_time = now
        self.last_known_time = now
        self.stuck_counter = 0
        curr_wp = self.movement_path.get_current_target()
        wp_name = curr_wp.get("name") if curr_wp else f"WP #{self.movement_path.current_idx}"
        self.status_message = f"Autopilot Resumed -> {wp_name} (Press 'F4' to pause)"
        _log(f"\n[AUTOPILOT] >>> RESUMED navigation towards WP #{self.movement_path.current_idx} ({wp_name}).")


    def toggle_pause(self) -> bool:
        """Toggles autopilot pause state on/off via F4."""
        now = time.time()
        if (now - self._last_f4_time) < 0.35:
            return self.is_paused
        self._last_f4_time = now

        if not self.is_active:
            _log("[NAVIGATOR] Autopilot is not currently running. Press 'A' or 'G' to start.")
            return False

        if self.waiting_for_green_light:
            self.give_green_light()
            return False

        if self.is_paused:
            self.resume()
        else:
            self.pause()
        return self.is_paused


    def give_green_light(self) -> bool:
        """
        Gives the green light to resume autopilot navigation after loot pickup verification.
        Returns True if green light was consumed, False if was not waiting.
        """
        if self.waiting_for_green_light:
            self.waiting_for_green_light = False
            self.status_message = "GREEN LIGHT GIVEN! Advancing to next waypoint..."
            _log("\n[AUTOPILOT] >>> GREEN LIGHT RECEIVED! Advancing to next waypoint...")
            # If in single-step/update mode (no background worker thread), advance waypoint now:
            if not (self._worker_thread and self._worker_thread.is_alive()):
                self.movement_path.advance()
            return True
        return False


    def toggle(self, monitor_idx: Optional[int] = None) -> bool:
        """Toggles autopilot navigation state on/off."""
        if self.is_active:
            self.stop()
        else:
            self.start(monitor_idx=monitor_idx)
        return self.is_active


    def reload_route(self) -> bool:
        """Reloads waypoints on demand and resynchronizes with character's current position."""
        success = self.movement_path.reload()
        if success:
            self.in_hideout = False
            self.is_completed = False
            self.is_paused = False
            self.waiting_for_green_light = False
            self.is_orbiting = False
            self.current_orbit_zone = None
            self.disable_persistent_combat()
            self.has_executed_initial_hold = False
            self.interacted_zones.clear()
            self.interacted_pink_dots.clear()
            if self.start_at_pink_dot > 0 and self.movement_path.is_configured:
                self.set_start_pink_dot(self.start_at_pink_dot)
            elif self.latest_pos is not None and self.movement_path.is_configured:
                self.movement_path.update_to_nearest(self.latest_pos)
            else:
                self.movement_path.current_idx = 0
            self.load_zone_routines()
            self.status_message = f"Route Reloaded ({len(self.movement_path.waypoints)} waypoints)"
            _log(f"[NAVIGATOR] Route reloaded. Total waypoints: {len(self.movement_path.waypoints)}")
        return success


    def set_start_pink_dot(self, pink_number: int, save_to_config: bool = False) -> bool:
        """
        Sets the starting point of route navigation targeting a specific Pink Dot (1-indexed).
        - pink_number == 0: Resets to route start (WP 0), clears skipped encounters.
        - pink_number >= 1: Targets Pink Dot #(pink_number):
          * Marks all preceding pink encounters as completed in self.interacted_pink_dots.
          * Marks all preceding yellow orbit zones as completed in self.interacted_zones.
          * Sets current_idx to the start of the route segment leading to the target pink dot
            (e.g., if pink_number=3, starts at WP after Pink Dot #2 and follows the green line).
          * Resets pause, waiting_for_green_light, and stuck recovery tracking.
        """
        if not getattr(self.movement_path, "is_configured", False):
            self.start_at_pink_dot = max(0, pink_number)
            return False

        # Safely resolve pink waypoints
        pink_wps = []
        if hasattr(self.movement_path, "get_pink_waypoints") and callable(self.movement_path.get_pink_waypoints):
            try:
                res = self.movement_path.get_pink_waypoints()
                if isinstance(res, (list, tuple)):
                    for item in res:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            pink_wps.append((item[0], item[1]))
                        elif isinstance(item, dict):
                            pink_wps.append((item.get("index", 0), item))
            except Exception:
                pass

        if not pink_wps and getattr(self.movement_path, "waypoints", None):
            wps = self.movement_path.waypoints
            if isinstance(wps, (list, tuple)):
                for idx, wp in enumerate(wps):
                    if isinstance(wp, dict) and wp.get("action") == "pink_encounter":
                        pink_wps.append((idx, wp))

        if not pink_wps or pink_number <= 0:
            self.start_at_pink_dot = 0
            self.target_pink_wp_idx = None
            self.target_pink_name = None
            self.target_pink_pos = None
            if hasattr(self.movement_path, "current_idx"):
                self.movement_path.current_idx = 0
            self.interacted_pink_dots.clear()
            self.interacted_zones.clear()
            self.is_completed = False
            self.is_orbiting = False
            self.waiting_for_green_light = False
            self.release_all_keys()
            self.stuck_counter = 0
            target = self.movement_path.get_current_target() if hasattr(self.movement_path, "get_current_target") else None
            t_name = target.get("name", "WP #0") if isinstance(target, dict) else "WP #0"
            self.status_message = f"Route Target: Start ({t_name})"
            if save_to_config:
                self._save_start_pink_dot_to_config(0)
            _log(f"[AUTOPILOT] Starting route from beginning (WP #0: {t_name}). All encounters active.")
            return True

        # Clamp pink_number to valid range
        pink_idx = max(1, min(pink_number, len(pink_wps)))
        self.start_at_pink_dot = pink_idx

        target_wp_idx, target_wp = pink_wps[pink_idx - 1]
        target_name = target_wp.get("name", f"Pink #{pink_idx}") if isinstance(target_wp, dict) else f"Pink #{pink_idx}"
        self.target_pink_wp_idx = target_wp_idx
        self.target_pink_name = target_name
        self.target_pink_pos = (target_wp.get("pink_pos") or [target_wp.get("x", 0), target_wp.get("y", 0)]) if isinstance(target_wp, dict) else None

        # Clear and mark all preceding pink encounters as visited
        self.interacted_pink_dots.clear()
        for p_i in range(pink_idx - 1):
            prev_wp_idx, _ = pink_wps[p_i]
            self.interacted_pink_dots.add(prev_wp_idx)

        # Mark all preceding orbit zones as visited
        self.interacted_zones.clear()
        orbit_zones = self.movement_path.get_orbit_zones() if hasattr(self.movement_path, "get_orbit_zones") else []
        if isinstance(orbit_zones, (list, tuple)):
            for zone in orbit_zones:
                if isinstance(zone, dict):
                    entry_idx = zone.get("entry_waypoint_index", float("inf"))
                    if entry_idx < target_wp_idx:
                        zone_id = zone.get("id")
                        if zone_id:
                            self.interacted_zones.add(zone_id)

        # Determine route segment starting waypoint
        if pink_idx == 1:
            start_wp_idx = 0
        else:
            prev_wp_idx, _ = pink_wps[pink_idx - 2]
            start_wp_idx = min(prev_wp_idx + 1, target_wp_idx)

        # If live player position is known and pink_idx > 1, find closest waypoint on the segment towards target
        wps = getattr(self.movement_path, "waypoints", None)
        if pink_idx > 1 and self.latest_pos is not None and isinstance(wps, (list, tuple)):
            min_s = prev_wp_idx
            best_s_idx = start_wp_idx
            best_s_d = float("inf")
            for s_i in range(min_s, min(len(wps), target_wp_idx + 1)):
                wp = wps[s_i]
                if isinstance(wp, dict) and "x" in wp and "y" in wp:
                    d = math.hypot(wp["x"] - self.latest_pos[0], wp["y"] - self.latest_pos[1])
                    if d < best_s_d:
                        best_s_d = d
                        best_s_idx = s_i
            if best_s_d <= 120.0:
                start_wp_idx = best_s_idx

        if hasattr(self.movement_path, "current_idx"):
            self.movement_path.current_idx = start_wp_idx
        self.is_completed = False
        self.is_orbiting = False
        self.current_orbit_zone = None
        self.waiting_for_green_light = False
        self.has_executed_initial_hold = False
        self.release_all_keys()

        # Reset stuck detection
        now = time.time()
        self.last_progress_pos = self.latest_pos
        self.last_progress_time = now
        self.last_known_time = now
        self.stuck_counter = 0

        curr_target = self.movement_path.get_current_target() if hasattr(self.movement_path, "get_current_target") else None
        curr_name = curr_target.get("name", f"WP #{start_wp_idx}") if isinstance(curr_target, dict) else f"WP #{start_wp_idx}"
        self.status_message = f"Next Target: Pink #{pink_idx} (from WP #{start_wp_idx}: {curr_name})"
        if save_to_config:
            self._save_start_pink_dot_to_config(pink_idx)
        _log(f"\n[AUTOPILOT] >>> ROUTE TARGET SET: Pink Dot #{pink_idx} ({target_name} at WP #{target_wp_idx})")
        _log(f"  Starting navigation at WP #{start_wp_idx} ({curr_name})")
        _log(f"  Preceding pink dots skipped/completed: {len(self.interacted_pink_dots)}")
        return True


    def _save_start_pink_dot_to_config(self, pink_number: int):
        """Persists the selected starting pink dot index into config.json."""
        if "PYTEST_CURRENT_TEST" in os.environ and (self.config_path == "config.json" or not os.path.isabs(self.config_path)):
            return
        if not self.config_path or not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "autopilot" not in cfg:
                cfg["autopilot"] = {}
            cfg["autopilot"]["start_at_pink_dot"] = int(pink_number)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass


    def cycle_start_pink_dot(self, save_to_config: bool = False) -> int:
        """
        Cycles starting pink dot to next available:
        0 (All) -> 1 -> 2 -> ... -> N -> 0 (All)
        Returns new start_at_pink_dot value.
        """
        pink_wps = []
        if hasattr(self.movement_path, "get_pink_waypoints") and callable(self.movement_path.get_pink_waypoints):
            try:
                res = self.movement_path.get_pink_waypoints()
                if isinstance(res, (list, tuple)):
                    pink_wps = res
            except Exception:
                pass
        if not pink_wps and getattr(self.movement_path, "waypoints", None):
            wps = self.movement_path.waypoints
            if isinstance(wps, (list, tuple)):
                pink_wps = [wp for wp in wps if isinstance(wp, dict) and wp.get("action") == "pink_encounter"]

        total_pinks = len(pink_wps)
        if total_pinks == 0:
            self.set_start_pink_dot(0, save_to_config=save_to_config)
            return 0

        next_pink = (self.start_at_pink_dot + 1) % (total_pinks + 1)
        self.set_start_pink_dot(next_pink, save_to_config=save_to_config)
        return next_pink


    def _get_capturer(self) -> ScreenCapturer:
        """Returns or lazily creates a ScreenCapturer instance for the target monitor."""
        if self.capturer is None:
            self.capturer = ScreenCapturer(monitor_idx=self.monitor_idx)
        return self.capturer


    def _get_enemy_detector(self) -> EnemyDetector:
        """Returns or lazily creates an EnemyDetector instance."""
        if self.enemy_detector is None:
            self.enemy_detector = EnemyDetector()
        return self.enemy_detector

