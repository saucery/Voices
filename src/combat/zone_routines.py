"""
Zone routine configuration loading, early exit toggling, routine lookup, and coordinate walking.
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


class ZoneRoutinesMixin:
    """Zone routine configuration loading, early exit toggling, routine lookup, and coordinate walking."""

    def load_zone_routines(self, filepath: Optional[str] = None) -> bool:
        """Loads per-zone encounter and combat routine configuration."""
        target_path = filepath or self.zone_routines_file
        candidates = [target_path, "routines/zone_routines.json"]
        if self.config_path:
            cfg_dir = os.path.dirname(self.config_path)
            if cfg_dir:
                candidates.append(os.path.join(cfg_dir, "routines/zone_routines.json"))
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        self.zone_routines = json.load(f)
                    p_count = len(self.zone_routines.get("pink_zones", {}))
                    y_count = len(self.zone_routines.get("yellow_zones", {}))
                    _log(f"[ROUTINES] Loaded custom zone routines from '{candidate}': {p_count} pink zone(s), {y_count} yellow zone(s)")
                    return True
                except Exception as e:
                    _log(f"[ROUTINES] Warning: Error parsing '{candidate}': {e}")
        return False


    def toggle_minimap_early_exit(self, save_to_config: bool = True) -> bool:
        """Toggles the minimap early encounter exit on/off and optionally persists to config.json."""
        self.minimap_early_exit_enabled = not self.minimap_early_exit_enabled
        state_str = "ENABLED" if self.minimap_early_exit_enabled else "DISABLED"
        _log(f"[CONFIG] Minimap Early Encounter Exit is now {state_str} (Trigger >= {self.minimap_early_exit_min_seconds:.1f}s)")
        if save_to_config:
            self._save_minimap_early_exit_to_config(self.minimap_early_exit_enabled)
        return self.minimap_early_exit_enabled


    def _save_minimap_early_exit_to_config(self, enabled: bool):
        """Persists current minimap early exit setting to config.json."""
        config_path = getattr(self, "config_path", "config.json") or "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "autopilot" not in cfg:
                    cfg["autopilot"] = {}
                cfg["autopilot"]["minimap_early_exit_enabled"] = bool(enabled)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
            except Exception as e:
                _log(f"[CONFIG] Warning: Could not save minimap_early_exit_enabled to config.json: {e}")


    def _get_pink_zone_routine(self, target: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Resolves custom routine for the target pink encounter if configured."""
        if not self.zone_routines or "pink_zones" not in self.zone_routines:
            return None
        pink_dict = self.zone_routines["pink_zones"]
        if not isinstance(pink_dict, dict):
            return None

        pink_num: Optional[int] = None
        import re

        # 1. Check direct target name or metadata (e.g. "Pink Marker (pink_5)", "pink_5", etc.)
        if target and isinstance(target, dict):
            name = str(target.get("name", ""))
            m = re.search(r'pink[_\s]*(\d+)', name, re.IGNORECASE)
            if m:
                try:
                    pink_num = int(m.group(1))
                except Exception:
                    pass

            if pink_num is None:
                orbit_zone = target.get("orbit_zone")
                if isinstance(orbit_zone, dict):
                    assoc = str(orbit_zone.get("associated_pink", "")) or str(orbit_zone.get("id", ""))
                    m_oz = re.search(r'(?:pink|zone)[_\s]*(\d+)', assoc, re.IGNORECASE)
                    if m_oz:
                        try:
                            pink_num = int(m_oz.group(1))
                        except Exception:
                            pass

            if pink_num is None and target.get("pink_id"):
                m_pid = re.search(r'(\d+)', str(target.get("pink_id")))
                if m_pid:
                    try:
                        pink_num = int(m_pid.group(1))
                    except Exception:
                        pass

        # 2. Check sequential pink encounter waypoints
        pink_wps = []
        if hasattr(self.movement_path, "get_pink_waypoints") and callable(self.movement_path.get_pink_waypoints):
            try:
                pink_wps = self.movement_path.get_pink_waypoints()
            except Exception:
                pass

        if pink_num is None and target and isinstance(target, dict):
            t_idx = target.get("index")
            for p_i, item in enumerate(pink_wps):
                w_idx = item[0] if isinstance(item, (list, tuple)) else getattr(item, "index", None)
                w_data = item[1] if isinstance(item, (list, tuple)) else item
                if t_idx == w_idx or (isinstance(w_data, dict) and w_data.get("index") == t_idx):
                    pink_num = p_i + 1
                    break

        # 3. Check nearest pink_zone by coordinates
        if pink_num is None and hasattr(self, "movement_path") and getattr(self.movement_path, "pink_zones", None):
            ref_pos = None
            if target and isinstance(target, dict):
                if target.get("pink_pos"):
                    ref_pos = target["pink_pos"]
                elif "x" in target and "y" in target:
                    ref_pos = (target["x"], target["y"])
            if ref_pos is None and self.latest_pos is not None:
                ref_pos = self.latest_pos

            if ref_pos:
                best_pz_idx = None
                best_pz_dist = float("inf")
                for pz_i, pz in enumerate(self.movement_path.pink_zones):
                    pz_x, pz_y = pz.get("x", 0), pz.get("y", 0)
                    d = math.hypot(pz_x - ref_pos[0], pz_y - ref_pos[1])
                    if d < best_pz_dist:
                        best_pz_dist = d
                        best_pz_idx = pz_i + 1
                if best_pz_idx is not None and best_pz_dist <= 250.0:
                    pink_num = best_pz_idx

        # 4. Check current room ID from room classifier
        if pink_num is None and getattr(self, "current_room_id", None):
            try:
                pink_num = int(self.current_room_id)
            except Exception:
                pass

        # 5. Check start_at_pink_dot
        if pink_num is None and getattr(self, "start_at_pink_dot", 0) > 0:
            pink_num = self.start_at_pink_dot

        candidate_keys = []
        if pink_num is not None:
            candidate_keys.extend([f"pink_{pink_num}", str(pink_num), f"pink{pink_num}", f"zone_{pink_num}"])
        if target and isinstance(target, dict):
            if "index" in target:
                candidate_keys.extend([f"wp_{target['index']}", str(target['index'])])
            if "name" in target:
                candidate_keys.append(str(target["name"]).lower())

        for k in candidate_keys:
            if k in pink_dict:
                return pink_dict[k]
            for pk, pv in pink_dict.items():
                if pk.lower() == k.lower():
                    return pv

        return None


    def _get_yellow_zone_routine(self, zone: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Resolves custom routine for the active yellow zone if configured."""
        if not self.zone_routines or "yellow_zones" not in self.zone_routines:
            return None
        yellow_dict = self.zone_routines["yellow_zones"]
        if not isinstance(yellow_dict, dict):
            return None

        candidate_keys = []
        if zone and isinstance(zone, dict):
            if "id" in zone:
                candidate_keys.append(str(zone["id"]))
            if "index" in zone:
                candidate_keys.extend([f"zone_{zone['index']}", str(zone['index'])])
            if "name" in zone:
                candidate_keys.append(str(zone["name"]).lower())

        for k in candidate_keys:
            if k in yellow_dict:
                return yellow_dict[k]
            for yk, yv in yellow_dict.items():
                if yk.lower() == k.lower():
                    return yv

        return None


    def _execute_zone_routine(
        self,
        routine: Dict[str, Any],
        zone_label: str,
        target: Optional[Dict[str, Any]] = None,
        zone: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Executes a list of configured steps for a specific pink or yellow zone."""
        if getattr(self, "in_hideout", False):
            _log(f"  [ROUTINE] Aborted: Character is in Hideout (Safe Zone - no combat or routines allowed).")
            return False

        self._routine_did_orbit = False
        steps = routine.get("steps", [])
        routine_name = routine.get("name", zone_label)
        room_key = self._resolve_room_key(routine, zone_label=zone_label, target=target, zone=zone)
        prev_room_key = self.current_room_key
        self.current_room_key = room_key
        routine_start_time = time.time()

        _log(f"\n[ROUTINE] >>> Starting custom routine '{routine_name}' ({len(steps)} steps) for {zone_label} (room={room_key})...")
        self.status_message = f"[{zone_label}] Executing Routine..."
        pink_origin = None
        if target and isinstance(target, dict):
            if target.get("pink_pos"):
                pink_origin = tuple(target["pink_pos"])
            elif "x" in target and "y" in target and target.get("action") == "pink_encounter":
                pink_origin = (float(target["x"]), float(target["y"]))

        if pink_origin is None and self.latest_pos is not None:
            pink_origin = (float(self.latest_pos[0]), float(self.latest_pos[1]))

        context: Dict[str, Any] = {
            "target": target,
            "zone": zone,
            "pink_origin": pink_origin,
            "sims_clicked": False,
            "banner_clicked": False,
        }

        try:
            for idx, step in enumerate(steps, 1):
                if stop_handler.is_stopped() or not self.is_active:
                    _log(f"  [ROUTINE] Stopped during step {idx}/{len(steps)}.")
                    return False

                action = step.get("action", "").lower().strip()
                desc = step.get("description", action)
                _log(f"  [ROUTINE STEP {idx}/{len(steps)}] {desc} (action={action})")

                success = self._execute_zone_routine_step(step, context, zone_label=zone_label)
                if not success and (stop_handler.is_stopped() or not self.is_active):
                    return False

            _log(f"[ROUTINE] <<< Finished custom routine '{routine_name}' for {zone_label}!\n")
            routine_elapsed = time.time() - routine_start_time
            if room_key:
                if not any(r.get("room") == room_key for r in getattr(self, "run_rooms_cleared", [])):
                    self._record_room_cleared(room_key, routine_elapsed)
                if self._is_last_room(room_key) and not getattr(self, "run_completed", False):
                    self._finalize_run(last_room=room_key, reason="last_room_cleared")
            return True
        finally:
            self.current_room_key = prev_room_key


    def _resolve_target_loot_pos(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        zone_label: str = "",
    ) -> Optional[Tuple[float, float]]:
        """Resolves the (x, y) coordinates of the LOOT dot (white dot) for the active encounter."""
        if "target_x" in step and "target_y" in step:
            return (float(step["target_x"]), float(step["target_y"]))
        if "loot_pos" in step and isinstance(step["loot_pos"], (list, tuple)) and len(step["loot_pos"]) >= 2:
            return (float(step["loot_pos"][0]), float(step["loot_pos"][1]))

        target = context.get("target") or {}
        if isinstance(target, dict):
            if target.get("loot_pos"):
                return tuple(target["loot_pos"])
            orbit_z = target.get("orbit_zone")
            if isinstance(orbit_z, dict) and orbit_z.get("loot_pos"):
                return tuple(orbit_z["loot_pos"])

        zone = context.get("zone") or {}
        if isinstance(zone, dict) and zone.get("loot_pos"):
            return tuple(zone["loot_pos"])

        room = getattr(self, "current_room_key", None)
        if hasattr(self.movement_path, "get_orbit_zones"):
            try:
                for oz in self.movement_path.get_orbit_zones():
                    if isinstance(oz, dict) and oz.get("loot_pos"):
                        assoc = str(oz.get("associated_pink", "")).lower()
                        if room and assoc == str(room).lower():
                            return tuple(oz["loot_pos"])
                        elif not room and "pink_7" in zone_label.lower() and assoc == "pink_7":
                            return tuple(oz["loot_pos"])
            except Exception:
                pass

        if hasattr(self.movement_path, "waypoints") and isinstance(self.movement_path.waypoints, (list, tuple)):
            for wp in self.movement_path.waypoints:
                if isinstance(wp, dict) and wp.get("loot_pos"):
                    wp_orbit = wp.get("orbit_zone")
                    if isinstance(wp_orbit, dict):
                        assoc = str(wp_orbit.get("associated_pink", "")).lower()
                        if room and assoc == str(room).lower():
                            return tuple(wp["loot_pos"])
                        elif not room and "pink_7" in zone_label.lower() and assoc == "pink_7":
                            return tuple(wp["loot_pos"])

        return None


    def _walk_to_coordinate(
        self,
        target_pos: Tuple[float, float],
        label: str = "NAV",
        timeout: float = 8.0,
        arrival_threshold: float = 25.0,
    ) -> bool:
        """
        Walks the character to a target coordinate using WASD movement.
        Returns True if reached within arrival_threshold, False if timed out or stopped.
        """
        if target_pos is None:
            return True
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.move_mouse_inside_game()
        _log(f"    [{label}] Walking to ({target_pos[0]:.1f}, {target_pos[1]:.1f}) (timeout={timeout:.1f}s, thresh={arrival_threshold:.1f}px)...")
        start_time = time.time()
        last_progress_pos = None
        last_progress_time = start_time
        stuck_counter = 0

        prev_interacting = self.is_interacting
        self.is_interacting = True

        try:
            while (time.time() - start_time) < timeout:
                if stop_handler.is_stopped() or not self.is_active:
                    return False

                current_pos = self.latest_pos
                if current_pos is None:
                    if self.last_known_pos is not None and (time.time() - self.last_known_time) < 2.0:
                        current_pos = self.last_known_pos
                    else:
                        time.sleep(0.05)
                        continue

                dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
                if dist <= arrival_threshold:
                    _log(f"    [{label}] Arrived at ({current_pos[0]:.1f}, {current_pos[1]:.1f}) (dist={dist:.1f}px <= {arrival_threshold:.1f}px)")
                    self.status_message = f"[{label}] Arrived ({dist:.0f}px)"
                    return True

                rem = max(0.0, timeout - (time.time() - start_time))
                now = time.time()

                # Anti-stuck check during coordinate walk
                if last_progress_pos is not None:
                    dist_moved = math.hypot(current_pos[0] - last_progress_pos[0], current_pos[1] - last_progress_pos[1])
                    if dist_moved < 4.0:
                        stuck_counter += 1
                        if stuck_counter > 12:
                            _log(f"    [{label}] Stuck detected ({stuck_counter} steps with <4px move). Executing unstick backtrack...")
                            self._execute_inverse_movement_backtrack(pulses=1, reason=f"{label} Stuck")
                            stuck_counter = 0
                            last_progress_pos = None
                            last_progress_time = time.time()
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
                self.held_keys = set(needed_keys)
                key_str = "+".join(k.upper() for k in sorted(needed_keys))
                self.status_message = f"[{label}] Walking [{key_str}] (dist={dist:.0f}px, {rem:.1f}s)"

                # Periodic combat attack if persistent combat is active (key 't')
                if self.persistent_right_click_active or self.persistent_combat_active:
                    self._trigger_persistent_combat_if_due(now)

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

                time.sleep(0.02)

            _log(f"    [{label}] Walk window finished ({timeout:.1f}s elapsed). Continuing routine...")
            return True
        finally:
            self.release_all_keys()
            self.is_interacting = prev_interacting

