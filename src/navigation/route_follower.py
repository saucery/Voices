"""
Autonomous route navigation loop, real-time position updates, and telemetry reporting.
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


class RouteFollowerMixin:
    """Autonomous route navigation loop, real-time position updates, and telemetry reporting."""

    def _run_navigation_loop(self):
        """Background thread executing sustained WASD pulses into the game."""
        step_count = 0
        while self.is_active and not stop_handler.is_stopped():
            if getattr(self, "in_hideout", False):
                self.release_all_keys()
                time.sleep(0.1)
                continue

            now = time.time()

            # 0. Check if paused (F4 hotkey)
            if self.is_paused:
                self.release_all_keys()
                self.last_known_time = now
                self.last_progress_time = now
                self.stuck_counter = 0
                if self.is_orbiting:
                    self.orbit_start_time += 0.05
                time.sleep(0.05)
                continue

            # 0b. Check if actively interacting (pink dot stop/sims/banner/loot)
            if self.is_interacting:
                self.release_all_keys()
                self.last_known_time = now
                self.last_progress_time = now
                self.stuck_counter = 0
                time.sleep(0.05)
                continue

            # 0c. Check if waiting for green light (loot confirmation)
            if self.waiting_for_green_light:
                self.release_all_keys()
                self.last_known_time = now
                self.last_progress_time = now
                self.stuck_counter = 0
                self._trigger_persistent_right_click_if_due(now)
                if keyboard:
                    try:
                        if keyboard.is_pressed('g') or keyboard.is_pressed('G') or keyboard.is_pressed('enter'):
                            self.give_green_light()
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            current_pos = self.latest_pos

            # 1. Check if tracking / location is lost
            if current_pos is None:
                if not self.is_interacting and not self.is_paused:
                    time_lost = now - self.last_known_time
                    if time_lost > self.tracking_lost_timeout:
                        self._execute_stuck_recovery(reason=f"Location tracking lost for {time_lost:.1f}s")
                time.sleep(0.04)
                continue

            # Valid position: update last_known_pos & timestamp
            self.last_known_pos = current_pos
            self.last_known_time = now

            # 2. Waypoint & Orbit Handling
            if self.is_orbiting:
                elapsed = now - self.orbit_start_time
                rem = max(0.0, self.orbit_duration - elapsed)
                if elapsed >= self.orbit_duration:
                    _log(f"\n[AUTOPILOT] Finished orbiting yellow shape ({self.orbit_duration:.1f}s elapsed)! Scanning for loot...")
                    self.is_orbiting = False
                    self.current_orbit_zone = None
                    self.orbit_perimeter_pts = []
                    self.collect_loot()

                    if self.wait_for_loot_confirmation:
                        self.release_all_keys()
                        self.waiting_for_green_light = True
                        self.status_message = "WAITING FOR GREEN LIGHT (Verify Loot Pickup - Press 'G' to Resume)"
                        _log("\n[AUTOPILOT] >>> LOOT PICKUP FINISHED! Waiting for GREEN LIGHT to continue...")
                        time.sleep(0.04)
                        continue

                    next_target = self.movement_path.advance()
                    if next_target is None:
                        self.is_completed = True
                        self.is_active = False
                        self.release_all_keys()
                        self.status_message = "Route Completed!"
                        break
                    target = next_target
                    target_pos = (target["x"], target["y"])
                    dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
                    self.last_distance = dist
                    self.last_target = target
                    self.stuck_counter = 0
                    self.last_progress_pos = current_pos
                    self.last_progress_time = now
                else:
                    # Periodic combat attack while orbiting (key 't')
                    if self.orbit_constant_right_click_enabled:
                        self._trigger_persistent_combat_if_due(now)

                    # Orbiting around perimeter of yellow shape
                    if self.orbit_perimeter_pts:
                        opt = self.orbit_perimeter_pts[self.orbit_point_idx % len(self.orbit_perimeter_pts)]
                        dist_opt = math.hypot(opt[0] - current_pos[0], opt[1] - current_pos[1])
                        orbit_reach_dist = min(13.0, max(9.0, self.arrival_threshold * 0.5))
                        if dist_opt <= orbit_reach_dist:
                            self.orbit_point_idx = (self.orbit_point_idx + 1) % len(self.orbit_perimeter_pts)
                            opt = self.orbit_perimeter_pts[self.orbit_point_idx]
                            dist_opt = math.hypot(opt[0] - current_pos[0], opt[1] - current_pos[1])
                        target_pos = (opt[0], opt[1])
                        dist = dist_opt
                        target = {
                            "index": self.movement_path.current_idx,
                            "name": f"Yellow Orbit ({rem:.1f}s left)",
                            "x": opt[0],
                            "y": opt[1],
                            "action": "orbit",
                        }
                        self.last_distance = dist
                        self.last_target = target
            else:
                # Closed-loop verification: Resync target waypoint based on live character position
                target = self.movement_path.update_to_nearest(current_pos)
                if target is None:
                    self.is_completed = True
                    self.is_active = False
                    self.persistent_right_click_active = False
                    self.release_all_keys()
                    self.status_message = "Route Completed!"
                    _log("\n[AUTOPILOT] >>> ALL WAYPOINTS COMPLETED! Reached destination (Red Dot).")
                    if not self.run_completed and self.run_start_time is not None:
                        self._finalize_run(reason="route_completed")
                    break

                target_pos = (target["x"], target["y"])
                dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
                self.last_distance = dist
                self.last_target = target

                # 3. Waypoint arrival check
                is_reached = dist <= self.arrival_threshold
                if not is_reached and target.get("action") == "pink_encounter":
                    if target.get("pink_pos"):
                        pink_p = target["pink_pos"]
                        dist_to_pink = math.hypot(pink_p[0] - current_pos[0], pink_p[1] - current_pos[1])
                        if dist_to_pink <= max(self.arrival_threshold, 60.0):
                            is_reached = True
                    # Also check if inside yellow orbit zone of this encounter
                    orbit_z = target.get("orbit_zone")
                    if not is_reached and orbit_z and orbit_z.get("center"):
                        zc = orbit_z["center"]
                        zr = float(orbit_z.get("radius", 45.0))
                        d_to_orbit = math.hypot(zc[0] - current_pos[0], zc[1] - current_pos[1])
                        if d_to_orbit <= (zr + 15.0):
                            is_reached = True

                if is_reached:
                    # Check if this waypoint triggers Yellow Shape Orbit
                    if target.get("action") == "orbit":
                        orbit_zone = target.get("orbit_zone") or self.movement_path.get_orbit_zone_for_waypoint(target.get("index", 0))
                        if orbit_zone and not self.orbit_yellow_zone_enabled:
                            _log(f"\n[AUTOPILOT] Reached Yellow Shape ({orbit_zone.get('id', 'zone')}) but orbit is disabled in config. Advancing route...")
                        elif orbit_zone:
                            zone_id = orbit_zone.get("id") or f"zone_{target.get('index', 0)}"
                            if zone_id in self.interacted_zones:
                                _log(f"\n[AUTOPILOT] Yellow Zone ({zone_id}) already completed. Advancing...")
                            else:
                                self.interacted_zones.add(zone_id)
                                self.execute_yellow_zone_interaction(orbit_zone)
                                if stop_handler.is_stopped() or not self.is_active:
                                    break
                                if not getattr(self, "_routine_did_orbit", False):
                                    self.is_orbiting = True
                                    self.orbit_start_time = time.time()
                                    self.last_orbit_right_click = time.time()
                                    self.orbit_duration = float(orbit_zone.get("duration", getattr(self.movement_path, "orbit_duration_seconds", 10.0)))
                                    self.current_orbit_zone = orbit_zone
                                    self.orbit_perimeter_pts = orbit_zone.get("perimeter_points", [])
                                    if self.orbit_perimeter_pts:
                                        best_p_idx = 0
                                        best_p_dist = float("inf")
                                        for p_i, p_pt in enumerate(self.orbit_perimeter_pts):
                                            d = math.hypot(p_pt[0] - current_pos[0], p_pt[1] - current_pos[1])
                                            if d < best_p_dist:
                                                best_p_dist = d
                                                best_p_idx = p_i
                                        self.orbit_point_idx = best_p_idx
                                    self.orbit_grace_until = time.time() + 4.0
                                    self.stuck_counter = 0
                                    self.last_progress_pos = current_pos
                                    self.last_progress_time = time.time() + 4.0
                                    window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
                                    self.move_mouse_inside_game()
                                    _log(f"\n[AUTOPILOT] Reached Yellow Shape ({orbit_zone.get('id', 'zone')})! Running around shape for {self.orbit_duration:.1f}s...")
                                    self.status_message = f"Orbiting Yellow Zone ({self.orbit_duration:.1f}s left)"
                                    time.sleep(0.04)
                                    continue

                    elif target.get("action") == "pink_encounter":
                        wp_idx = target.get("index", 0)
                        if wp_idx not in self.interacted_pink_dots:
                            self.interacted_pink_dots.add(wp_idx)
                            self.execute_pink_dot_interaction(target)
                            if stop_handler.is_stopped() or not self.is_active:
                                break
                            if self.is_orbiting:
                                self.orbit_grace_until = time.time() + 4.0
                                self.stuck_counter = 0
                                self.last_progress_time = time.time() + 4.0
                                time.sleep(0.04)
                                continue

                    _log(f"\n[AUTOPILOT] Reached WP #{target.get('index', 0)} ({target.get('name', 'WP')}) at ({current_pos[0]:.0f}, {current_pos[1]:.0f})! Advancing...")
                    next_target = self.movement_path.advance()
                    if next_target is None:
                        self.is_completed = True
                        self.is_active = False
                        self.persistent_right_click_active = False
                        self.release_all_keys()
                        self.status_message = "Route Finished!"
                        _log("\n[AUTOPILOT] >>> DESTINATION REACHED (Red Dot)!")
                        if not self.run_completed and self.run_start_time is not None:
                            self._finalize_run(reason="route_completed")
                        break
                    target = next_target
                    target_pos = (target["x"], target["y"])
                    dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
                    self.last_distance = dist
                    self.last_target = target
                    # Reset stuck counters on normal waypoint arrival
                    self.stuck_counter = 0
                    self.backtrack_attempts_at_wp = 0
                    self.last_progress_pos = current_pos
                    self.last_progress_time = now

            # 4. Check for physical stuck (not moving towards target despite sending keys)
            # Strictly active ONLY when traveling on green path or orbiting inside yellow area, NEVER when interacting or paused
            if not self.is_interacting and not self.is_paused:
                if self.is_orbiting and now < self.orbit_grace_until:
                    # Within orbit startup/recovery grace period: skip stuck check to let character start movement
                    self.stuck_counter = 0
                    self.last_progress_pos = current_pos
                    self.last_progress_time = now
                elif self.last_progress_pos is not None:
                    dist_moved = math.hypot(current_pos[0] - self.last_progress_pos[0], current_pos[1] - self.last_progress_pos[1])
                    if dist_moved < 4.5:
                        self.stuck_counter += 1
                        limit_steps = self.orbit_stuck_step_limit if self.is_orbiting else self.stuck_step_limit
                        limit_sec = self.orbit_stuck_timeout_sec if self.is_orbiting else 2.5
                        if self.stuck_counter >= limit_steps or (now - self.last_progress_time) > limit_sec:
                            mode_name = "Orbit" if self.is_orbiting else f"Waypoint {target.get('index', 0)}"
                            self._execute_stuck_recovery(reason=f"Stuck at ({current_pos[0]:.0f}, {current_pos[1]:.0f}) - no movement for {self.stuck_counter} steps at {mode_name}")
                            continue
                    else:
                        self.stuck_counter = 0
                        self.last_progress_pos = current_pos
                        self.last_progress_time = now
                else:
                    self.last_progress_pos = current_pos
                    self.last_progress_time = now

            needed_keys = self.compute_wasd_keys(current_pos, target_pos)
            if not needed_keys:
                time.sleep(0.04)
                continue

            self.last_held_keys = list(needed_keys)
            self.recent_movements.append((list(needed_keys), time.time(), self.step_duration, current_pos))

            # Ensure game window is focused
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

            self.held_keys = set(needed_keys)
            key_str = "+".join(k.upper() for k in sorted(needed_keys))
            step_count += 1
            if self.is_orbiting:
                rem = max(0.0, self.orbit_duration - (time.time() - self.orbit_start_time))
                self.status_message = f"Orbiting Shape [{key_str}] ({rem:.1f}s left)"
            else:
                self.status_message = f"WP #{target.get('index', 0)} ({target.get('name', 'WP')}) | [{key_str}] ({dist:.0f}px)"

            # Periodic combat attack during transit (key 't')
            if (self.persistent_right_click_active or self.persistent_combat_active) and not self.is_orbiting:
                self._trigger_persistent_combat_if_due(now)

            # Sustained step pulse directly into game
            self.is_simulating_key = True
            try:
                if pydirectinput:
                    for k in needed_keys:
                        try:
                            pydirectinput.keyDown(k)
                        except Exception:
                            pass
                    time.sleep(self.step_duration)
                    for k in needed_keys:
                        try:
                            pydirectinput.keyUp(k)
                        except Exception:
                            pass
                else:
                    time.sleep(self.step_duration)
            finally:
                self.is_simulating_key = False
                self.held_keys.clear()

            time.sleep(0.02)  # Brief pause between steps

        self.release_all_keys()


    def update(self, current_pos: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Executes one navigation control tick using current live character position.

        :param current_pos: (X, Y) tuple of character on reference map layout.
        :return: Navigation telemetry dictionary.
        """
        self.latest_pos = current_pos
        now = time.time()
        if current_pos is not None:
            self.last_known_pos = current_pos
            self.last_known_time = now

        # Hideout Safety: Hideout is a safe zone. Never execute waypoint navigation, routines, or combat attacks
        if self.in_hideout:
            self.disable_persistent_combat()
            self.release_all_keys()
            return self.get_telemetry(None, 0.0, [])

        # Periodic right-click attack check on every live frame tick
        if self.persistent_right_click_active:
            self._trigger_persistent_right_click_if_due(now)

        # Safety: F1 emergency stop takes immediate priority
        if stop_handler.is_stopped():
            if self.is_active:
                _log("[NAVIGATOR] Emergency stop detected! Halting navigation.")
                self.stop()
            return self.get_telemetry(None, 0.0, [])

        if not self.is_active or not self.movement_path.is_configured:
            self.release_all_keys()
            target = self.movement_path.update_to_nearest(current_pos) if current_pos else self.movement_path.get_current_target()
            dist = self.movement_path.distance_to_target(current_pos) if current_pos and target else 0.0
            return self.get_telemetry(target, dist, [])

        # Check if paused (F4 hotkey)
        if self.is_paused:
            self.release_all_keys()
            target = self.movement_path.get_current_target()
            dist = self.movement_path.distance_to_target(current_pos) if current_pos and target else 0.0
            return self.get_telemetry(target, dist, [], is_paused=True)

        # Check if actively interacting (pink dot stop/sims/banner/loot)
        if self.is_interacting:
            target = self.movement_path.get_current_target()
            dist = self.movement_path.distance_to_target(current_pos) if current_pos and target else 0.0
            return self.get_telemetry(target, dist, list(self.held_keys))

        # Check if waiting for green light (loot confirmation)
        if self.waiting_for_green_light:
            self.release_all_keys()
            target = self.movement_path.get_current_target()
            dist = self.movement_path.distance_to_target(current_pos) if current_pos and target else 0.0
            return self.get_telemetry(target, dist, [])

        if current_pos is None:
            return self.get_telemetry(self.movement_path.get_current_target(), 0.0, list(self.held_keys), tracking_lost=True)

        now = time.time()

        if self.is_orbiting:
            elapsed = now - self.orbit_start_time
            rem = max(0.0, self.orbit_duration - elapsed)
            if elapsed >= self.orbit_duration:
                self.is_orbiting = False
                self.current_orbit_zone = None
                self.orbit_perimeter_pts = []
                self.collect_loot()

                if self.wait_for_loot_confirmation:
                    self.release_all_keys()
                    self.waiting_for_green_light = True
                    self.status_message = "WAITING FOR GREEN LIGHT (Verify Loot Pickup - Press 'G' to Resume)"
                    _log("\n[AUTOPILOT] >>> LOOT PICKUP FINISHED! Waiting for GREEN LIGHT to continue...")
                    target = self.movement_path.get_current_target()
                    dist = self.movement_path.distance_to_target(current_pos) if current_pos and target else 0.0
                    return self.get_telemetry(target, dist, [])

                target = self.movement_path.advance()
                if target is None:
                    self.release_all_keys()
                    self.is_active = False
                    self.is_completed = True
                    self.persistent_right_click_active = False
                    self.status_message = "Route Completed!"
                    return self.get_telemetry(None, 0.0, [])
                target_pos = (target["x"], target["y"])
                dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
            else:
                # If background worker thread is driving the orbit, update() only reports telemetry without simulating keys
                if self._worker_thread and self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
                    if self.orbit_perimeter_pts:
                        opt = self.orbit_perimeter_pts[self.orbit_point_idx % len(self.orbit_perimeter_pts)]
                        dist = math.hypot(opt[0] - current_pos[0], opt[1] - current_pos[1]) if current_pos else 0.0
                        target = {
                            "index": self.movement_path.current_idx,
                            "name": f"Yellow Orbit ({rem:.1f}s left)",
                            "x": opt[0],
                            "y": opt[1],
                            "action": "orbit",
                        }
                    else:
                        target = self.movement_path.get_current_target()
                        dist = self.movement_path.distance_to_target(current_pos) if current_pos and target else 0.0
                    return self.get_telemetry(target, dist, list(self.held_keys), tracking_lost=(current_pos is None))

                if self.orbit_constant_right_click_enabled:
                    self._trigger_persistent_combat_if_due(now)

                if self.orbit_perimeter_pts:
                    opt = self.orbit_perimeter_pts[self.orbit_point_idx % len(self.orbit_perimeter_pts)]
                    dist_opt = math.hypot(opt[0] - current_pos[0], opt[1] - current_pos[1])
                    if dist_opt <= max(14.0, self.arrival_threshold * 0.8):
                        self.orbit_point_idx = (self.orbit_point_idx + 1) % len(self.orbit_perimeter_pts)
                        opt = self.orbit_perimeter_pts[self.orbit_point_idx]
                        dist_opt = math.hypot(opt[0] - current_pos[0], opt[1] - current_pos[1])
                    target_pos = (opt[0], opt[1])
                    dist = dist_opt
                    target = {
                        "index": self.movement_path.current_idx,
                        "name": f"Yellow Orbit ({rem:.1f}s left)",
                        "x": opt[0],
                        "y": opt[1],
                        "action": "orbit",
                    }
                else:
                    target = self.movement_path.get_current_target()
                    target_pos = (target["x"], target["y"]) if target else (0.0, 0.0)
                    dist = 0.0
        else:
            # Closed-loop verification: Resync target waypoint based on live character position
            target = self.movement_path.update_to_nearest(current_pos)
            if target is None:
                self.release_all_keys()
                self.is_active = False
                self.is_completed = True
                self.status_message = "Route Completed!"
                return self.get_telemetry(None, 0.0, [])

            target_pos = (target["x"], target["y"])
            dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
            self.last_distance = dist
            self.last_target = target

            # Check Arrival at Current Waypoint
            is_reached = dist <= self.arrival_threshold
            if not is_reached and target.get("action") == "pink_encounter" and target.get("pink_pos"):
                pink_p = target["pink_pos"]
                dist_to_pink = math.hypot(pink_p[0] - current_pos[0], pink_p[1] - current_pos[1])
                if dist_to_pink <= max(self.arrival_threshold, 30.0):
                    is_reached = True

            if is_reached:
                if target.get("action") == "orbit":
                    orbit_zone = target.get("orbit_zone") or self.movement_path.get_orbit_zone_for_waypoint(target.get("index", 0))
                    if orbit_zone and not self.orbit_yellow_zone_enabled:
                        _log(f"\n[AUTOPILOT] Reached Yellow Shape ({orbit_zone.get('id', 'zone')}) but orbit is disabled in config. Advancing route...")
                    elif orbit_zone:
                        zone_id = orbit_zone.get("id") or f"zone_{target.get('index', 0)}"
                        if zone_id in self.interacted_zones:
                            _log(f"\n[AUTOPILOT] Yellow Zone ({zone_id}) already completed. Advancing...")
                        else:
                            self.interacted_zones.add(zone_id)
                            self.execute_yellow_zone_interaction(orbit_zone)
                            if stop_handler.is_stopped() or not self.is_active:
                                return self.get_telemetry(None, 0.0, [])
                            if not getattr(self, "_routine_did_orbit", False):
                                self.is_orbiting = True
                                self.orbit_start_time = time.time()
                                self.last_orbit_right_click = time.time()
                                self.orbit_duration = float(orbit_zone.get("duration", getattr(self.movement_path, "orbit_duration_seconds", 10.0)))
                                self.current_orbit_zone = orbit_zone
                                self.orbit_perimeter_pts = orbit_zone.get("perimeter_points", [])
                                if self.orbit_perimeter_pts:
                                    best_p_idx = 0
                                    best_p_dist = float("inf")
                                    for p_i, p_pt in enumerate(self.orbit_perimeter_pts):
                                        d = math.hypot(p_pt[0] - current_pos[0], p_pt[1] - current_pos[1])
                                        if d < best_p_dist:
                                            best_p_dist = d
                                            best_p_idx = p_i
                                    self.orbit_point_idx = best_p_idx
                                self.status_message = f"Orbiting Yellow Zone ({self.orbit_duration:.1f}s left)"
                                return self.get_telemetry(target, dist, list(self.held_keys))

                elif target.get("action") == "pink_encounter":
                    wp_idx = target.get("index", 0)
                    if wp_idx not in self.interacted_pink_dots:
                        if self._worker_thread and self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
                            # Let background worker thread handle the blocking pink encounter routine
                            target = self.movement_path.get_current_target()
                            dist = self.movement_path.distance_to_target(current_pos) if current_pos and target else 0.0
                            return self.get_telemetry(target, dist, list(self.held_keys))
                        self.interacted_pink_dots.add(wp_idx)
                        self.execute_pink_dot_interaction(target)
                        if stop_handler.is_stopped() or not self.is_active:
                            return self.get_telemetry(None, 0.0, [])
                        if self.is_orbiting:
                            return self.get_telemetry(target, dist, list(self.held_keys))

                next_target = self.movement_path.advance()
                if next_target is None:
                    self.release_all_keys()
                    self.is_active = False
                    self.is_completed = True
                    self.persistent_right_click_active = False
                    self.status_message = "Route Finished!"
                    return self.get_telemetry(None, 0.0, [])
                target = next_target
                target_pos = (target["x"], target["y"])
                dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])

        needed_keys = set(self.compute_wasd_keys(current_pos, target_pos))
        # Keep held_keys populated if worker thread is off (e.g. testing)
        if not (self._worker_thread and self._worker_thread.is_alive()):
            self.held_keys = needed_keys

        return self.get_telemetry(target, dist, list(needed_keys or self.held_keys))


    def get_telemetry(
        self,
        target: Optional[Dict[str, Any]],
        dist: float,
        keys: List[str],
        tracking_lost: bool = False,
        is_paused: bool = False,
    ) -> Dict[str, Any]:
        """Returns structured navigation status."""
        rem_sec = 0.0
        if self.is_orbiting:
            rem_sec = max(0.0, self.orbit_duration - (time.time() - self.orbit_start_time))

        status_msg = self.status_message
        if getattr(self, "in_hideout", False):
            status_msg = "HIDEOUT (Safe Zone - Navigation & Combat Disabled)"
        elif self.waiting_for_green_light:
            status_msg = "WAITING FOR GREEN LIGHT (Verify Loot - Press 'G' to Resume)"
        elif tracking_lost:
            status_msg = "Tracking Lost (Holding Position)"
        elif is_paused or self.is_paused:
            status_msg = "Autopilot PAUSED (Press 'F4' to resume)"

        now = time.time()
        elapsed_sec = 0.0
        if self.run_start_time is not None:
            end_t = self.run_end_time if self.run_end_time is not None else now
            elapsed_sec = max(0.0, end_t - self.run_start_time)

        return {
            "in_hideout": getattr(self, "in_hideout", False),
            "is_active": self.is_active,
            "is_paused": self.is_paused or is_paused,
            "is_completed": self.is_completed,
            "waiting_for_green_light": self.waiting_for_green_light,
            "waiting_for_user_key": getattr(self, "waiting_for_user_key", False),
            "waiting_user_key_name": getattr(self, "waiting_user_key_name", "f5"),
            "target": target,
            "target_index": target.get("index") if target else None,
            "target_name": target.get("name") if target else "None",
            "distance_to_target": round(dist, 1),
            "held_keys": keys,
            "held_keys_str": "+".join(k.upper() for k in sorted(keys)) if keys else "None",
            "status_message": status_msg,
            "recovery_event": self.latest_recovery_event,
            "is_orbiting": self.is_orbiting,
            "orbit_remaining_sec": round(rem_sec, 1),
            "orbit_zone": self.current_orbit_zone,
            "start_at_pink_dot": self.start_at_pink_dot,
            "total_pink_dots": len(self.movement_path.get_pink_waypoints()) if hasattr(self.movement_path, "get_pink_waypoints") else 0,
            "target_pink_wp_idx": getattr(self, "target_pink_wp_idx", None),
            "target_pink_name": getattr(self, "target_pink_name", None),
            "target_pink_pos": getattr(self, "target_pink_pos", None),
            "interacted_pink_dots": list(self.interacted_pink_dots),
            "persistent_combat": self.persistent_combat_active or self.persistent_right_click_active,
            "persistent_combat_action": self.persistent_combat_action,
            "persistent_combat_key": self.persistent_combat_key,
            "persistent_combat_interval": self.persistent_combat_interval,
            "has_executed_initial_hold": self.has_executed_initial_hold,
            "minimap_early_exit_enabled": getattr(self, "minimap_early_exit_enabled", True),
            "minimap_early_exit_min_seconds": getattr(self, "minimap_early_exit_min_seconds", 25.0),
            "minimap_early_exit_min_icons": getattr(self, "minimap_early_exit_min_icons", 1),
            "run_id": self.run_id,
            "run_active": self.is_active and not self.run_completed,
            "run_completed": self.run_completed,
            "run_elapsed_sec": round(elapsed_sec, 1),
            "run_elapsed_str": self._format_duration(elapsed_sec),
            "run_sims_count": len(self.run_sims_clicked),
            "run_loot_count": len(self.run_loot_picked),
            "run_last_room_cleared": self.run_last_room_cleared,
            "run_rooms_cleared_count": len(self.run_rooms_cleared),
        }

