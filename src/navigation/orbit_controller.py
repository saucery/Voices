"""
Yellow zone orbit navigation loop with counter-clockwise geometric sweeping.
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


class OrbitControllerMixin:
    """Yellow zone orbit navigation loop with counter-clockwise geometric sweeping."""

    def _run_orbit_loop(
        self,
        duration: float,
        best_zone: Dict[str, Any],
        right_click_interval: float = 0.75,
        zone_label: str = "ZONE",
        rolling_enabled: bool = False,
        rolling_interval: float = 5.0,
        portal_early_exit: bool = False,
        portal_early_exit_min_seconds: float = 30.0,
    ) -> bool:
        """
        Actively runs character orbit around the yellow zone perimeter for the specified duration.
        Drives WASD movement, right-click skill execution, and stuck recovery until duration expires.
        """
        zone_id = best_zone.get("id", "zone")
        if zone_id:
            self.interacted_zones.add(zone_id)

        self.is_orbiting = True
        self.orbit_start_time = time.time()
        self.last_orbit_right_click = time.time()
        self.orbit_duration = duration
        self.current_orbit_zone = best_zone
        self.orbit_perimeter_pts = best_zone.get("perimeter_points", [])

        c_pos = self.latest_pos or best_zone.get("center", [0.0, 0.0])
        if self.orbit_perimeter_pts:
            best_p_idx = 0
            best_p_dist = float("inf")
            for p_i, p_pt in enumerate(self.orbit_perimeter_pts):
                d = math.hypot(p_pt[0] - c_pos[0], p_pt[1] - c_pos[1])
                if d < best_p_dist:
                    best_p_dist = d
                    best_p_idx = p_i
            self.orbit_point_idx = best_p_idx

        orbit_grace_until = time.time() + 4.0
        stuck_counter = 0
        last_progress_pos = self.latest_pos or c_pos
        last_progress_time = time.time() + 4.0
        last_right_click = time.time()

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.move_mouse_inside_game()
        if rolling_enabled:
            _log(f"    [ACTION] Starting orbit inside yellow shape ({zone_id}) for {duration:.1f}s (ROLLING every {rolling_interval:.1f}s)...")
        else:
            _log(f"    [ACTION] Starting orbit inside yellow shape ({zone_id}) for {duration:.1f}s...")
        self.status_message = f"Orbiting Yellow Zone ({duration:.1f}s left)"

        last_roll_time = time.time()
        last_early_exit_check = 0.0
        early_exit_confirmations = 0
        early_exit_triggered = False
        orbit_start = time.time()
        try:
            while (time.time() - orbit_start) < duration:
                if stop_handler.is_stopped() or not self.is_active:
                    _log(f"    [ACTION] Orbit cancelled by stop handler / inactive state.")
                    return False

                if self.is_paused:
                    self.release_all_keys()
                    time.sleep(0.05)
                    orbit_start += 0.05
                    continue

                now = time.time()
                rem = max(0.0, duration - (now - orbit_start))

                # Periodic combat attack (key 't')
                if self.orbit_constant_right_click_enabled:
                    self._trigger_persistent_combat_if_due(now)

                # Check for Early Encounter Exit
                elapsed_orbit = now - orbit_start

                # 1. Main screen Portal Early Exit (Room 7 Delirium encounter - detects after 30s)
                if (
                    portal_early_exit
                    and elapsed_orbit >= portal_early_exit_min_seconds
                    and (now - last_early_exit_check) >= getattr(self, "minimap_early_exit_check_interval", 0.5)
                ):
                    last_early_exit_check = now
                    portal_pos = self.locate_portal()
                    if portal_pos is not None:
                        self._last_detected_portal_pos = portal_pos
                        self._last_detected_portal_time = now
                        _log(f"\n[ENCOUNTER EARLY EXIT] Confirmed exit portal visible on screen at {portal_pos} ({elapsed_orbit:.1f}s / {duration:.1f}s)! Ending combat early to interact with Delirium statue...")
                        self.status_message = f"[{zone_label}] Early Exit (Portal Detected @ {elapsed_orbit:.1f}s)"
                        self.disable_persistent_combat()
                        early_exit_triggered = True
                        break

                # 2. Minimap Loot Drop Early Exit (Rooms 1-6)
                elif (
                    getattr(self, "minimap_early_exit_enabled", True)
                    and elapsed_orbit >= getattr(self, "minimap_early_exit_min_seconds", 25.0)
                    and (now - last_early_exit_check) >= getattr(self, "minimap_early_exit_check_interval", 0.5)
                ):
                    last_early_exit_check = now
                    mm_loot_count, mm_icons = self.check_minimap_loot_drop()
                    min_req = getattr(self, "minimap_early_exit_min_icons", 1)
                    if mm_loot_count >= min_req:
                        early_exit_confirmations += 1
                        # Trigger if either >=2 loot icons detected immediately, or confirmed across 2 checks
                        if mm_loot_count >= 2 or early_exit_confirmations >= 2:
                            _log(f"\n[ENCOUNTER EARLY EXIT] Confirmed {mm_loot_count} minimap loot icon(s) (Monsters defeated at {elapsed_orbit:.1f}s / {duration:.1f}s)! Ending combat early to collect loot...")
                            self.status_message = f"[{zone_label}] Early Exit (Loot Detected @ {elapsed_orbit:.1f}s)"
                            # Save annotated debug screenshots
                            self.check_minimap_loot_drop(save_debug=True)
                            early_exit_triggered = True
                            break
                    else:
                        early_exit_confirmations = 0

                current_pos = self.latest_pos
                if current_pos is not None and best_zone is not None:
                    # Spatial boundary check during orbit: reject wild jumps outside zone
                    zc = best_zone.get("center")
                    zr = float(best_zone.get("radius", 60.0))
                    if zc and len(zc) >= 2:
                        dist_zc = math.hypot(current_pos[0] - zc[0], current_pos[1] - zc[1])
                        if dist_zc > max(95.0, zr * 1.6):
                            current_pos = None

                if current_pos is None:
                    # If tracking dropped momentarily during combat, keep heading towards target with last known pos
                    if self.last_known_pos is not None and (now - self.last_known_time) < 3.0:
                        current_pos = self.last_known_pos
                    else:
                        self.status_message = f"[{zone_label}] Orbiting - Tracking Lost ({rem:.1f}s left)"
                        time.sleep(0.04)
                        continue

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
                else:
                    zc = best_zone.get("center", current_pos)
                    target_pos = (zc[0], zc[1])

                # Stuck check: only count as physical stuck if position tracking was actively updating
                tracking_fresh = (now - self.last_known_time) < 1.2
                if not tracking_fresh:
                    # Tracking lost or stale - don't treat visual tracking loss as physical collision
                    stuck_counter = 0
                    last_progress_time = now
                elif now < orbit_grace_until:
                    stuck_counter = 0
                    last_progress_pos = current_pos
                    last_progress_time = now
                elif last_progress_pos is not None:
                    dist_moved = math.hypot(current_pos[0] - last_progress_pos[0], current_pos[1] - last_progress_pos[1])
                    if dist_moved < 4.5:
                        stuck_counter += 1
                        if stuck_counter >= self.orbit_stuck_step_limit or (now - last_progress_time) > self.orbit_stuck_timeout_sec:
                            # Verify current_pos is actually near the orbit zone before triggering physical stuck recovery
                            zc = best_zone.get("center") if best_zone else None
                            is_in_orbit_area = True
                            if zc and len(zc) >= 2:
                                zr = float(best_zone.get("radius", 60.0))
                                if math.hypot(current_pos[0] - zc[0], current_pos[1] - zc[1]) > (zr * 1.5):
                                    is_in_orbit_area = False

                            if is_in_orbit_area:
                                self._execute_stuck_recovery(reason=f"Stuck at ({current_pos[0]:.0f}, {current_pos[1]:.0f}) - no movement for {stuck_counter} steps during Yellow Orbit")
                            else:
                                self.orbit_point_idx = (self.orbit_point_idx + 1) % max(1, len(self.orbit_perimeter_pts))

                            orbit_grace_until = time.time() + 4.0
                            stuck_counter = 0
                            last_progress_pos = current_pos
                            last_progress_time = time.time() + 4.0
                            continue
                    else:
                        stuck_counter = 0
                        last_progress_pos = current_pos
                        last_progress_time = now
                else:
                    last_progress_pos = current_pos
                    last_progress_time = now

                needed_keys = self.compute_wasd_keys(current_pos, target_pos)
                if not needed_keys:
                    time.sleep(0.04)
                    continue

                self.last_held_keys = list(needed_keys)
                window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
                self.held_keys = set(needed_keys)
                key_str = "+".join(k.upper() for k in sorted(needed_keys))
                self.status_message = f"[{zone_label}] Orbiting [{key_str}] ({rem:.1f}s left)"

                # Determine if this step should be a roll
                should_roll = False
                if rolling_enabled and pydirectinput and (now - last_roll_time) >= rolling_interval:
                    should_roll = True

                self.is_simulating_key = True
                try:
                    if pydirectinput:
                        for k in needed_keys:
                            try:
                                pydirectinput.keyDown(k)
                            except Exception:
                                pass

                        if should_roll:
                            # Rolling: press SPACE while holding a movement key
                            time.sleep(0.05)  # brief hold before roll
                            try:
                                pydirectinput.keyDown('space')
                                time.sleep(0.12)  # hold space briefly for roll registration
                                pydirectinput.keyUp('space')
                            except Exception:
                                pass
                            last_roll_time = now
                            roll_key_str = "+".join(k.upper() for k in sorted(needed_keys))
                            _log(f"    [ROLL] Dodge roll [{roll_key_str}+SPACE] at ({current_pos[0]:.0f}, {current_pos[1]:.0f})")
                            time.sleep(self.step_duration * 0.5)  # shorter hold after roll
                        else:
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

                time.sleep(0.02)
        finally:
            self.release_all_keys()
            self.is_orbiting = False
            self.current_orbit_zone = None
            self.orbit_perimeter_pts = []
            if portal_early_exit and getattr(self, "_last_detected_portal_pos", None) is not None:
                self.disable_persistent_combat()

        if early_exit_triggered:
            _log(f"    [ACTION] Early encounter exit triggered for yellow shape ({zone_id}) ({time.time() - orbit_start:.1f}s elapsed)!")
        else:
            _log(f"    [ACTION] Finished orbiting yellow shape ({zone_id}) ({duration:.1f}s elapsed)!")
        return True

