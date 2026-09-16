"""
Route Navigator Module
Handles real-time closed-loop WASD character navigation along a defined MovementPath.
Calculates 8-directional vector headings and holds/releases DirectInput WASD keys
towards active waypoints with arrival threshold detection and emergency stop safety.
"""

import json
import math
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.01
    pydirectinput.FAILSAFE = False
except ImportError:
    pydirectinput = None

try:
    import keyboard
except ImportError:
    keyboard = None

from .movement_path import MovementPath
from .screen_capturer import ScreenCapturer
from .stop_handler import stop_handler
from .window_focus import window_focuser



def _log(msg: str = "") -> None:
    """Prints a message prefixed with the current timestamp [HH:MM:SS]."""
    if not msg:
        print()
        return
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = ""
    if msg.startswith("\n"):
        prefix = "\n"
        msg = msg[1:]
    print(f"{prefix}[{ts}] {msg}")

class RouteNavigator:
    """
    Closed-loop autonomous character navigator using WASD controls in Path of Exile 2.
    """

    def __init__(
        self,
        movement_path: Optional[MovementPath] = None,
        arrival_threshold: float = 18.0,
        monitor_idx: int = 2,
        step_duration: float = 0.22,
        capturer: Optional[Any] = None,
        config_path: str = "config.json",
    ):
        """
        Initialize RouteNavigator.

        :param movement_path: MovementPath instance containing waypoints.
        :param arrival_threshold: Pixel distance to waypoint to trigger advancement (default: 18 px).
        :param monitor_idx: Target monitor index for game window focus.
        :param step_duration: Duration to hold WASD keys per step (default: 0.22s).
        :param capturer: ScreenCapturer instance for banner detection.
        :param config_path: Path to config.json.
        """
        self.movement_path = movement_path or MovementPath()
        path_arrival = getattr(self.movement_path, "arrival_distance", None)
        if isinstance(path_arrival, (int, float)):
            self.arrival_threshold = max(float(arrival_threshold), float(path_arrival))
        else:
            self.arrival_threshold = float(arrival_threshold)
        self.monitor_idx = monitor_idx
        self.step_duration = step_duration
        self.capturer = capturer
        self.config_path = config_path

        self.is_active: bool = False
        self.is_paused: bool = False
        self._last_f4_time: float = 0.0
        self.is_completed: bool = False
        self.is_simulating_key: bool = False
        self.held_keys: Set[str] = set()
        self.latest_pos: Optional[Tuple[float, float]] = None
        self._worker_thread: Optional[threading.Thread] = None

        self.last_target: Optional[Dict[str, Any]] = None
        self.last_distance: float = 0.0
        self.status_message: str = "Autopilot Ready (Press 'A' to start)"

        # Stuck & Recovery Detection State
        self.last_known_pos: Optional[Tuple[float, float]] = None
        self.last_known_time: float = time.time()
        self.last_progress_pos: Optional[Tuple[float, float]] = None
        self.last_progress_time: float = time.time()
        self.stuck_counter: int = 0
        self.stuck_step_limit: int = 6           # ~1.8s of no progress
        self.tracking_lost_timeout: float = 1.8   # seconds before declaring lost
        self.last_held_keys: List[str] = []
        self.latest_recovery_event: Optional[str] = None
        self.is_interacting: bool = False
        self._routine_did_orbit: bool = False
        self._log = _log

        # Yellow Shape Orbit Navigation State
        self.is_orbiting: bool = False
        self.orbit_start_time: float = 0.0
        self.orbit_duration: float = 10.0
        self.current_orbit_zone: Optional[Dict[str, Any]] = None
        self.orbit_perimeter_pts: List[List[float]] = []
        self.orbit_point_idx: int = 0
        self.orbit_stuck_step_limit: int = 16       # ~4.0s of continuous zero progress while orbiting
        self.orbit_stuck_timeout_sec: float = 6.0   # Allow attack animations & turns while orbiting
        self.orbit_grace_until: float = 0.0         # Grace period timestamp on entering orbit

        # Yellow Zone Encounter Banner & Interaction Configuration
        self.orbit_yellow_zone_enabled: bool = True
        self.orbit_constant_right_click_enabled: bool = True
        self.orbit_right_click_interval_seconds: float = 0.75
        self.click_banner_enabled: bool = True
        self.right_click_after_banner_enabled: bool = True
        self.middle_click_hold_enabled: bool = True
        self.middle_click_hold_seconds: float = 3.0
        self.encounter_banner_file: str = "ui/encounter_banner.png"
        self.encounter_match_threshold: float = 0.45
        self.banner_search_attempts: int = 5
        self.banner_approach_wait_seconds: float = 2.0
        self.sim_approach_wait_seconds: float = 2.0
        self.reclick_after_approach: bool = False
        self.interacted_zones: Set[str] = set()
        self.last_orbit_right_click: float = 0.0

        # Pink Dot Sim & Banner Encounter State
        self.pink_dot_stop_seconds: float = 2.5
        self.sim1_template_file: str = "ui/sim1.png"
        self.sim2_template_file: str = "ui/sim2.png"
        self.sim3_template_file: str = "ui/sim3.png"
        self.sim_match_threshold: float = 0.50
        self.sim_click_y_offset_px: int = 35
        self.sim_click_x_offset_px: int = 0
        self.start_at_pink_dot: int = 0
        self.target_pink_wp_idx: Optional[int] = None
        self.target_pink_name: Optional[str] = None
        self.target_pink_pos: Optional[List[float]] = None
        self.interacted_pink_dots: Set[int] = set()

        # Loot Pickup State & Configuration
        self.loot1_template_file: str = "ui/loot1.png"
        self.loot_match_threshold: float = 0.50
        self.loot_pickup_wait_seconds: float = 0.4
        self.loot_approach_wait_seconds: float = 1.5
        self.max_loot_pickups: int = 15
        self.wait_for_loot_confirmation: bool = True
        self.waiting_for_green_light: bool = False
        self.loot1_img: Optional[np.ndarray] = None

        # Zone Routines (Per-zone step sequences and timings)
        self.zone_routines_file: str = "routines/zone_routines.json"
        self.zone_routines: Dict[str, Any] = {}

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                ap_cfg = cfg.get("autopilot", {})
                self.orbit_yellow_zone_enabled = bool(ap_cfg.get("orbit_yellow_zone_enabled", self.orbit_yellow_zone_enabled))
                self.orbit_constant_right_click_enabled = bool(ap_cfg.get("orbit_constant_right_click_enabled", self.orbit_constant_right_click_enabled))
                self.orbit_right_click_interval_seconds = float(ap_cfg.get("orbit_right_click_interval_seconds", self.orbit_right_click_interval_seconds))
                self.click_banner_enabled = bool(ap_cfg.get("click_banner_enabled", self.click_banner_enabled))
                self.right_click_after_banner_enabled = bool(ap_cfg.get("right_click_after_banner_enabled", self.right_click_after_banner_enabled))
                self.middle_click_hold_enabled = bool(ap_cfg.get("middle_click_hold_enabled", self.middle_click_hold_enabled))
                self.middle_click_hold_seconds = float(ap_cfg.get("middle_click_hold_seconds", self.middle_click_hold_seconds))
                self.encounter_banner_file = ap_cfg.get("encounter_banner_file", self.encounter_banner_file)
                self.encounter_match_threshold = float(ap_cfg.get("encounter_match_threshold", self.encounter_match_threshold))
                self.banner_approach_wait_seconds = float(ap_cfg.get("banner_approach_wait_seconds", self.banner_approach_wait_seconds))
                self.sim_approach_wait_seconds = float(ap_cfg.get("sim_approach_wait_seconds", self.sim_approach_wait_seconds))
                self.reclick_after_approach = bool(ap_cfg.get("reclick_after_approach", self.reclick_after_approach))
                self.pink_dot_stop_seconds = float(ap_cfg.get("pink_dot_stop_seconds", self.pink_dot_stop_seconds))
                self.sim1_template_file = ap_cfg.get("sim1_template_file", self.sim1_template_file)
                self.sim2_template_file = ap_cfg.get("sim2_template_file", self.sim2_template_file)
                self.sim3_template_file = ap_cfg.get("sim3_template_file", self.sim3_template_file)
                self.sim_match_threshold = float(ap_cfg.get("sim_match_threshold", self.sim_match_threshold))
                self.sim_click_y_offset_px = int(ap_cfg.get("sim_click_y_offset_px", self.sim_click_y_offset_px))
                self.sim_click_x_offset_px = int(ap_cfg.get("sim_click_x_offset_px", self.sim_click_x_offset_px))
                self.start_at_pink_dot = int(ap_cfg.get("start_at_pink_dot", self.start_at_pink_dot))
                self.loot1_template_file = ap_cfg.get("loot1_template_file", self.loot1_template_file)
                self.loot_match_threshold = float(ap_cfg.get("loot_match_threshold", self.loot_match_threshold))
                self.loot_pickup_wait_seconds = float(ap_cfg.get("loot_pickup_wait_seconds", self.loot_pickup_wait_seconds))
                self.loot_approach_wait_seconds = float(ap_cfg.get("loot_approach_wait_seconds", self.loot_approach_wait_seconds))
                self.max_loot_pickups = int(ap_cfg.get("max_loot_pickups", self.max_loot_pickups))
                self.wait_for_loot_confirmation = bool(ap_cfg.get("wait_for_loot_confirmation", self.wait_for_loot_confirmation))
                self.zone_routines_file = ap_cfg.get("zone_routines_file", self.zone_routines_file)
            except Exception:
                pass

        self.encounter_banner_img: Optional[np.ndarray] = None
        for candidate_path in [self.encounter_banner_file, "ui/encounter_banner.png", "templates/ui/encounter_banner.png"]:
            if os.path.exists(candidate_path):
                self.encounter_banner_img = cv2.imread(candidate_path)
                if self.encounter_banner_img is not None:
                    break

        self.sim_templates: Dict[str, Optional[np.ndarray]] = {"sim1": None, "sim2": None, "sim3": None}
        self._load_sim_templates()
        self._load_loot_template()
        self.load_zone_routines()

        if self.start_at_pink_dot > 0 and self.movement_path.is_configured:
            self.set_start_pink_dot(self.start_at_pink_dot)

        if keyboard:
            try:
                keyboard.add_hotkey("f4", self.toggle_pause, suppress=False)
                _log("[NAVIGATOR] Global F4 Pause/Resume listener active.")
            except Exception as e:
                _log(f"[NAVIGATOR] Warning: Could not register global F4 hotkey: {e}")

    def _load_sim_templates(self):
        """Loads sim1, sim2, and sim3 template images if present on disk."""
        sim_files = {
            "sim1": self.sim1_template_file,
            "sim2": self.sim2_template_file,
            "sim3": self.sim3_template_file,
        }
        for key, filepath in sim_files.items():
            loaded_img = None
            for candidate in [filepath, f"ui/{key}.png", f"templates/ui/{key}.png"]:
                if candidate and os.path.exists(candidate):
                    loaded_img = cv2.imread(candidate)
                    if loaded_img is not None:
                        break
            self.sim_templates[key] = loaded_img

    def _load_loot_template(self):
        """Loads loot1 template image if present on disk."""
        self.loot1_img = None
        for candidate in [self.loot1_template_file, "ui/loot1.png", "templates/ui/loot1.png"]:
            if candidate and os.path.exists(candidate):
                self.loot1_img = cv2.imread(candidate)
                if self.loot1_img is not None:
                    break

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

    def _get_pink_zone_routine(self, target: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Resolves custom routine for the target pink encounter if configured."""
        if not self.zone_routines or "pink_zones" not in self.zone_routines:
            return None
        pink_dict = self.zone_routines["pink_zones"]
        if not isinstance(pink_dict, dict):
            return None

        pink_num: Optional[int] = None
        pink_wps = []
        if hasattr(self.movement_path, "get_pink_waypoints") and callable(self.movement_path.get_pink_waypoints):
            try:
                pink_wps = self.movement_path.get_pink_waypoints()
            except Exception:
                pass

        if target and isinstance(target, dict):
            t_idx = target.get("index")
            for p_i, item in enumerate(pink_wps):
                w_idx = item[0] if isinstance(item, (list, tuple)) else getattr(item, "index", None)
                w_data = item[1] if isinstance(item, (list, tuple)) else item
                if t_idx == w_idx or (isinstance(w_data, dict) and w_data.get("index") == t_idx):
                    pink_num = p_i + 1
                    break

        if pink_num is None and getattr(self, "start_at_pink_dot", 0) > 0:
            pink_num = self.start_at_pink_dot

        candidate_keys = []
        if pink_num is not None:
            candidate_keys.extend([f"pink_{pink_num}", str(pink_num), f"pink{pink_num}"])
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

    def _run_orbit_loop(
        self,
        duration: float,
        best_zone: Dict[str, Any],
        right_click_interval: float = 0.75,
        zone_label: str = "ZONE",
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

        prev_interacting = self.is_interacting
        self.is_interacting = False

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.move_mouse_inside_game()
        _log(f"    [ACTION] Starting orbit inside yellow shape ({zone_id}) for {duration:.1f}s...")
        self.status_message = f"Orbiting Yellow Zone ({duration:.1f}s left)"

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

                # Periodic right-click
                if self.orbit_constant_right_click_enabled and right_click_interval > 0:
                    if (now - last_right_click) >= right_click_interval:
                        last_right_click = now
                        self.move_mouse_inside_game()
                        if pydirectinput:
                            try:
                                pydirectinput.rightClick()
                                time.sleep(0.02)
                                pydirectinput.mouseUp(button="right")
                            except Exception:
                                pass

                current_pos = self.latest_pos
                if current_pos is None:
                    # If tracking dropped momentarily during combat, keep heading towards target with last known pos
                    if self.last_known_pos is not None and (now - self.last_known_time) < 2.0:
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
                            self._execute_stuck_recovery(reason=f"Stuck at ({current_pos[0]:.0f}, {current_pos[1]:.0f}) - no movement for {stuck_counter} steps during Yellow Orbit")
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
        finally:
            self.release_all_keys()
            self.is_orbiting = False
            self.current_orbit_zone = None
            self.orbit_perimeter_pts = []
            self.is_interacting = prev_interacting

        _log(f"    [ACTION] Finished orbiting yellow shape ({zone_id}) ({duration:.1f}s elapsed)!")
        return True

    def _execute_zone_routine(
        self,
        routine: Dict[str, Any],
        zone_label: str,
        target: Optional[Dict[str, Any]] = None,
        zone: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Executes a list of configured steps for a specific pink or yellow zone."""
        self._routine_did_orbit = False
        steps = routine.get("steps", [])
        routine_name = routine.get("name", zone_label)
        _log(f"\n[ROUTINE] >>> Starting custom routine '{routine_name}' ({len(steps)} steps) for {zone_label}...")
        self.status_message = f"[{zone_label}] Executing Routine..."
        context: Dict[str, Any] = {
            "target": target,
            "zone": zone,
            "sims_clicked": False,
            "banner_clicked": False,
        }

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
        return True

    def _execute_zone_routine_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        zone_label: str = "ZONE",
    ) -> bool:
        """Dispatches and executes an individual routine step."""
        action = step.get("action", "").lower().strip()

        if action == "stop":
            duration = float(step.get("duration", self.pink_dot_stop_seconds))
            self.release_all_keys()
            self.move_mouse_inside_game()
            stop_start = time.time()
            while (time.time() - stop_start) < duration:
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                rem = max(0.0, duration - (time.time() - stop_start))
                self.status_message = f"[{zone_label}] Stopped ({rem:.1f}s)..."
                time.sleep(0.05)
            return True

        elif action == "hold_mouse":
            button = str(step.get("button", "middle")).lower().strip()
            if button == "middle" and not getattr(self, "middle_click_hold_enabled", True):
                _log(f"    [CONFIG] Middle click hold disabled (middle_click_hold_enabled=false). Skipping...")
                return True
            duration = float(step.get("duration", self.middle_click_hold_seconds))
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            self.move_mouse_inside_game()
            _log(f"    [ACTION] Holding mouse '{button}' button for {duration:.1f}s inside game...")
            try:
                if pydirectinput:
                    pydirectinput.mouseDown(button=button)
                h_start = time.time()
                while (time.time() - h_start) < duration:
                    if stop_handler.is_stopped() or not self.is_active:
                        break
                    rem_h = max(0.0, duration - (time.time() - h_start))
                    self.status_message = f"[{zone_label}] Holding {button.title()} ({rem_h:.1f}s)..."
                    time.sleep(0.05)
            finally:
                if pydirectinput:
                    pydirectinput.mouseUp(button=button)
                try:
                    import ctypes
                    flag = 0x0040 if button == "middle" else (0x0010 if button == "right" else 0x0004)
                    ctypes.windll.user32.mouse_event(flag, 0, 0, 0, 0)
                except Exception:
                    pass
                _log(f"    [ACTION] Mouse '{button}' button released.")
            time.sleep(0.1)
            return True

        elif action == "click_mouse":
            button = str(step.get("button", "right")).lower().strip()
            if button == "right" and not getattr(self, "right_click_after_banner_enabled", True):
                _log(f"    [CONFIG] Right click after banner disabled (right_click_after_banner_enabled=false). Skipping...")
                return True
            clicks = int(step.get("clicks", 1))
            delay = float(step.get("delay", 0.15))
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            self.move_mouse_inside_game()
            self.status_message = f"[{zone_label}] Clicking {button.title()}..."
            for _ in range(clicks):
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                if pydirectinput:
                    if button == "right":
                        pydirectinput.rightClick()
                    elif button == "middle":
                        pydirectinput.middleClick()
                    else:
                        pydirectinput.click()
                time.sleep(delay)
            return True

        elif action == "detect_and_click_sims":
            priority = step.get("priority", ["sim1", "sim3", "sim2"])
            y_offset = int(step.get("click_y_offset", self.sim_click_y_offset_px))
            x_offset = int(step.get("click_x_offset", self.sim_click_x_offset_px))
            app_wait = float(step.get("approach_wait", self.sim_approach_wait_seconds))
            clicked = self._detect_and_click_sims(
                prefix=f"[{zone_label}]",
                sim_order=priority,
                y_offset_px=y_offset,
                x_offset_px=x_offset,
                approach_wait=app_wait,
            )
            context["sims_clicked"] = len(clicked) > 0
            return True

        elif action == "click_encounter_banner":
            if not getattr(self, "click_banner_enabled", True):
                _log(f"    [CONFIG] Banner clicking disabled (click_banner_enabled=false). Skipping...")
                return True

            if step.get("only_if_no_sims", False) and context.get("sims_clicked", False):
                _log(f"    [STEP] Sims were already selected and only_if_no_sims is set. Skipping banner click.")
                return True

            app_wait = float(step.get("approach_wait", self.banner_approach_wait_seconds))
            reclick = bool(step.get("reclick", self.reclick_after_approach))
            search_attempts = int(step.get("search_attempts", self.banner_search_attempts))

            self.status_message = f"[{zone_label}] Finding Encounter Banner..."
            banner_pos = None
            for attempt in range(1, search_attempts + 1):
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                banner_pos = self.locate_encounter_banner()
                if banner_pos is not None:
                    break
                time.sleep(0.25)

            if banner_pos is not None:
                bx, by = self.move_mouse_inside_game(banner_pos[0], banner_pos[1])
                _log(f"    [ACTION] Clicking encounter banner at ({bx}, {by})...")
                self.status_message = f"[{zone_label}] Clicking Banner ({bx}, {by})"
                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.08)
                    pydirectinput.mouseUp(button="left")
                time.sleep(0.15)

                if app_wait > 0:
                    self._wait_for_approach(app_wait, reason=f"{zone_label} BANNER")
                    if stop_handler.is_stopped() or not self.is_active:
                        return False
                    if reclick:
                        in_range_pos = self.locate_encounter_banner()
                        if in_range_pos is not None:
                            rx, ry = self.move_mouse_inside_game(in_range_pos[0], in_range_pos[1])
                            _log(f"    [ACTION] Re-clicking encounter banner in-range at ({rx}, {ry})...")
                            if pydirectinput:
                                pydirectinput.click()
                                time.sleep(0.08)
                                pydirectinput.mouseUp(button="left")
                            time.sleep(0.15)
                context["banner_clicked"] = True
            else:
                _log(f"    [WARNING] Encounter banner not detected on screen.")
                self.move_mouse_inside_game()
            return True

        elif action == "orbit_yellow_zone":
            orbit_duration = float(step.get("duration", self.orbit_duration))
            rc_interval = float(step.get("right_click_interval", self.orbit_right_click_interval_seconds))
            orbit_zones = self.movement_path.get_orbit_zones() if hasattr(self.movement_path, "get_orbit_zones") else []
            target = context.get("target")
            zone = context.get("zone")
            c_pos = self.latest_pos or ((target["x"], target["y"]) if isinstance(target, dict) and "x" in target else (0.0, 0.0))
            best_zone = zone
            if best_zone is None and orbit_zones:
                best_d = float("inf")
                for z in orbit_zones:
                    if isinstance(z, dict):
                        zc = z.get("center", [0, 0])
                        d = math.hypot(zc[0] - c_pos[0], zc[1] - c_pos[1])
                        if d < best_d:
                            best_d = d
                            best_zone = z

            if best_zone is not None:
                context["orbited"] = True
                self._routine_did_orbit = True
                return self._run_orbit_loop(
                    duration=orbit_duration,
                    best_zone=best_zone,
                    right_click_interval=rc_interval,
                    zone_label=zone_label,
                )
            else:
                _log(f"    [WARNING] No yellow orbit zone found for {zone_label}. Skipping orbit.")
                return True

        elif action == "pickup_loot":
            max_pickups = int(step.get("max_items", self.max_loot_pickups))
            wait_for_green = bool(step.get("wait_for_green_light", self.wait_for_loot_confirmation))
            pickup_delay = float(step.get("pickup_delay", self.loot_pickup_wait_seconds))
            app_wait = float(step.get("approach_wait", getattr(self, "loot_approach_wait_seconds", 1.5)))
            prev_delay = self.loot_pickup_wait_seconds
            self.loot_pickup_wait_seconds = pickup_delay
            try:
                self.collect_loot(max_pickups=max_pickups, approach_wait=app_wait)
            finally:
                self.loot_pickup_wait_seconds = prev_delay

            if wait_for_green:
                self.release_all_keys()
                self.waiting_for_green_light = True
                self.status_message = "WAITING FOR GREEN LIGHT (Verify Loot Pickup - Press 'G' to Resume)"
                _log("\n[AUTOPILOT] >>> LOOT PICKUP FINISHED! Waiting for GREEN LIGHT to continue...")
                while self.waiting_for_green_light:
                    if stop_handler.is_stopped() or not self.is_active:
                        return False
                    if keyboard:
                        try:
                            if keyboard.is_pressed('g') or keyboard.is_pressed('G') or keyboard.is_pressed('enter'):
                                self.give_green_light()
                                break
                        except Exception:
                            pass
                    time.sleep(0.05)
            return True

        elif action == "press_key":
            key = str(step.get("key", "q")).lower().strip()
            duration = float(step.get("duration", 0.1))
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            self.move_mouse_inside_game()
            _log(f"    [ACTION] Pressing key '{key}' for {duration:.2f}s...")
            self.status_message = f"[{zone_label}] Key '{key.upper()}' ({duration:.1f}s)..."
            if pydirectinput:
                pydirectinput.keyDown(key)
            k_start = time.time()
            while (time.time() - k_start) < duration:
                if stop_handler.is_stopped() or not self.is_active:
                    break
                time.sleep(0.02)
            if pydirectinput:
                pydirectinput.keyUp(key)
            time.sleep(0.05)
            return True

        elif action == "wait":
            duration = float(step.get("duration", 1.0))
            self.status_message = f"[{zone_label}] Waiting ({duration:.1f}s)..."
            w_start = time.time()
            while (time.time() - w_start) < duration:
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                time.sleep(0.05)
            return True

        else:
            _log(f"    [WARNING] Unknown routine action '{action}'. Skipping...")
            return True

    @staticmethod
    def compute_wasd_keys(current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> List[str]:
        """
        Calculates the 8-directional WASD key combination from current position to target.

        PoE2 2D map coordinate conventions:
        - X increases to the East (Right) -> 'd'
        - X decreases to the West (Left)  -> 'a'
        - Y increases to the South (Down) -> 's'
        - Y decreases to the North (Up)   -> 'w'
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        # In case we are right on top of the point
        if abs(dx) < 3.0 and abs(dy) < 3.0:
            return []

        angle = math.degrees(math.atan2(dy, dx))  # -180 to 180 degrees

        # 8-Directional Angular Sectors (each 45 degrees wide)
        if -22.5 <= angle < 22.5:
            return ["d"]                  # East
        elif 22.5 <= angle < 67.5:
            return ["s", "d"]             # South-East
        elif 67.5 <= angle < 112.5:
            return ["s"]                  # South
        elif 112.5 <= angle < 157.5:
            return ["s", "a"]             # South-West
        elif angle >= 157.5 or angle < -157.5:
            return ["a"]                  # West
        elif -157.5 <= angle < -112.5:
            return ["w", "a"]             # North-West
        elif -112.5 <= angle < -67.5:
            return ["w"]                  # North
        elif -67.5 <= angle < -22.5:
            return ["w", "d"]             # North-East
        return []

    def start(self, monitor_idx: Optional[int] = None):
        """Enables autopilot navigation."""
        if not self.movement_path.is_configured:
            self.status_message = "No route waypoints loaded"
            _log("[NAVIGATOR] Cannot start: No route waypoints loaded.")
            return

        if monitor_idx is not None:
            self.monitor_idx = monitor_idx

        # If previously completed or at the end, reset back to closest waypoint
        if self.is_completed or self.movement_path.current_idx >= len(self.movement_path.waypoints) - 1:
            self.is_completed = False
            if self.latest_pos is not None:
                self.movement_path.current_idx = self.movement_path.find_nearest_waypoint_index(self.latest_pos)
            else:
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
        self.status_message = "Autopilot Active (Press 'A' to stop | 'F4' to pause)"
        _log(f"[NAVIGATOR] Autopilot Navigation ACTIVATED. Press 'A' to stop | 'F4' to pause.")

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
        self.is_orbiting = False
        self.current_orbit_zone = None
        self.interacted_zones.clear()
        self.interacted_pink_dots.clear()
        self.release_all_keys()
        self.status_message = "Autopilot Paused (Press 'A' to resume)"
        _log("[NAVIGATOR] Autopilot Navigation STOPPED.")

    def pause(self):
        """Pauses navigation, releases all movement keys, and holds current waypoint position."""
        if not self.is_active:
            return
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
            self.is_completed = False
            self.is_paused = False
            self.waiting_for_green_light = False
            self.is_orbiting = False
            self.current_orbit_zone = None
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

        # If live player position is known, find closest waypoint on the segment towards target
        wps = getattr(self.movement_path, "waypoints", None)
        if self.latest_pos is not None and isinstance(wps, (list, tuple)):
            min_s = 0 if pink_idx == 1 else prev_wp_idx
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

    def locate_encounter_banner(self) -> Optional[Tuple[int, int]]:
        """
        Locates the encounter banner on screen using template correlation.
        Returns desktop absolute coordinates (X, Y) of the banner center, or None.
        """
        if self.encounter_banner_img is None:
            for candidate in [self.encounter_banner_file, "ui/encounter_banner.png", "templates/ui/encounter_banner.png"]:
                if os.path.exists(candidate):
                    self.encounter_banner_img = cv2.imread(candidate)
                    if self.encounter_banner_img is not None:
                        break

        if self.encounter_banner_img is None:
            return None

        capt = self._get_capturer()
        try:
            screen = capt.capture()
        except Exception as e:
            _log(f"  [WARNING] Screen capture failed during banner search: {e}")
            return None

        if screen is None or screen.size == 0:
            return None

        # Determine monitor offset if available
        mon_left = 0
        mon_top = 0
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        # Convert to grayscale
        g_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        g_tmpl = cv2.cvtColor(self.encounter_banner_img, cv2.COLOR_BGR2GRAY)
        th, tw = g_tmpl.shape[:2]
        sh, sw = g_screen.shape[:2]

        best_val = -1.0
        best_loc = None
        best_scale = 1.0

        # Try base 1.0 scale
        if sw >= tw and sh >= th:
            res = cv2.matchTemplate(g_screen, g_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            best_val = float(max_val)
            best_loc = max_loc

        # Multi-scale search if below threshold
        if best_val < self.encounter_match_threshold:
            for scale in [0.70, 0.80, 0.90, 1.10, 1.20, 1.30]:
                sc_w = int(tw * scale)
                sc_h = int(th * scale)
                if sc_w > sw or sc_h > sh or sc_w < 20 or sc_h < 20:
                    continue
                resized = cv2.resize(g_tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(g_screen, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = float(max_val)
                    best_loc = max_loc
                    best_scale = scale

        if best_val >= self.encounter_match_threshold and best_loc is not None:
            cx = int(best_loc[0] + (tw * best_scale) / 2)
            cy = int(best_loc[1] + (th * best_scale) / 2)
            desktop_x = mon_left + cx
            desktop_y = mon_top + cy
            _log(f"  [BANNER MATCH] Found encounter banner (conf={best_val:.2f}, scale={best_scale:.2f}) at screen ({desktop_x}, {desktop_y})")
            return desktop_x, desktop_y

        return None

    def locate_sim_template(self, sim_key: str) -> Optional[Tuple[int, int]]:
        """
        Locates a sim selection template (sim1, sim2, sim3) on screen using multi-scale template matching.
        Returns desktop absolute coordinates (X, Y) of the match center, or None if not found.
        """
        tmpl = self.sim_templates.get(sim_key)
        if tmpl is None:
            self._load_sim_templates()
            tmpl = self.sim_templates.get(sim_key)

        if tmpl is None:
            return None

        capt = self._get_capturer()
        try:
            screen = capt.capture()
        except Exception as e:
            _log(f"  [WARNING] Screen capture failed during {sim_key} search: {e}")
            return None

        if screen is None or screen.size == 0:
            return None

        mon_left = 0
        mon_top = 0
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        g_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        g_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        th, tw = g_tmpl.shape[:2]
        sh, sw = g_screen.shape[:2]

        best_val = -1.0
        best_loc = None
        best_scale = 1.0

        if sw >= tw and sh >= th:
            res = cv2.matchTemplate(g_screen, g_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            best_val = float(max_val)
            best_loc = max_loc

        if best_val < self.sim_match_threshold:
            for scale in [0.70, 0.80, 0.90, 1.10, 1.20, 1.30]:
                sc_w = int(tw * scale)
                sc_h = int(th * scale)
                if sc_w > sw or sc_h > sh or sc_w < 15 or sc_h < 15:
                    continue
                resized = cv2.resize(g_tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(g_screen, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = float(max_val)
                    best_loc = max_loc
                    best_scale = scale

        if best_val >= self.sim_match_threshold and best_loc is not None:
            cx = int(best_loc[0] + (tw * best_scale) / 2)
            cy = int(best_loc[1] + (th * best_scale) / 2)
            desktop_x = mon_left + cx
            desktop_y = mon_top + cy
            _log(f"  [SIM MATCH] Found {sim_key} (conf={best_val:.2f}, scale={best_scale:.2f}) at screen ({desktop_x}, {desktop_y})")
            return desktop_x, desktop_y

        return None

    def _detect_and_click_sims(
        self,
        prefix: str = "[SIM]",
        sim_order: Optional[List[str]] = None,
        y_offset_px: Optional[int] = None,
        x_offset_px: Optional[int] = None,
        approach_wait: Optional[float] = None,
    ) -> List[str]:
        """
        Detects and selects Sims according to priority:
        Default priority: sim1 -> wait 2.0s -> sim3 -> wait 2.0s -> sim2.
        Returns list of sim names that were clicked (e.g. ['sim1', 'sim3']).
        Clicks are offset by sim_click_y_offset_px (e.g. +35px) below the detected template center.
        """
        sims_clicked: List[str] = []
        eff_y_offset = y_offset_px if y_offset_px is not None else self.sim_click_y_offset_px
        eff_x_offset = x_offset_px if x_offset_px is not None else self.sim_click_x_offset_px
        eff_app_wait = approach_wait if approach_wait is not None else self.sim_approach_wait_seconds

        def _click_sim_at(sim_key: str, sim_label: str, detected_pos: Tuple[int, int]) -> Tuple[int, int]:
            target_x = detected_pos[0] + eff_x_offset
            target_y = detected_pos[1] + eff_y_offset
            cx, cy = self.move_mouse_inside_game(target_x, target_y)
            offset_info = f" [offset: +{eff_y_offset}px Y]" if eff_y_offset != 0 else ""
            _log(f"  [SIM DETECTED] Clicking {sim_label} at ({cx}, {cy}){offset_info} (detected at ({detected_pos[0]}, {detected_pos[1]}))...")
            self.status_message = f"{prefix} Clicked {sim_label} ({cx}, {cy})"
            if pydirectinput:
                pydirectinput.click()
                time.sleep(0.08)
                pydirectinput.mouseUp(button="left")
            sims_clicked.append(sim_key)

            # Approach wait: allow character to move closer to the sim object before proceeding
            if eff_app_wait > 0:
                self._wait_for_approach(eff_app_wait, reason=f"{prefix} {sim_label}")
                if not stop_handler.is_stopped() and self.is_active and self.reclick_after_approach:
                    in_range_pos = self.locate_sim_template(sim_key)
                    if in_range_pos is not None:
                        rx, ry = self.move_mouse_inside_game(
                            in_range_pos[0] + eff_x_offset,
                            in_range_pos[1] + eff_y_offset
                        )
                        _log(f"  [SIM IN-RANGE] Re-clicking {sim_label} in-range at ({rx}, {ry})...")
                        if pydirectinput:
                            pydirectinput.click()
                            time.sleep(0.08)
                            pydirectinput.mouseUp(button="left")
                        time.sleep(0.15)

            return cx, cy

        if sim_order is not None and len(sim_order) > 0:
            for sim_key in sim_order:
                if stop_handler.is_stopped() or not self.is_active:
                    return sims_clicked
                sim_label = f"Sim {sim_key.replace('sim', '')}"
                self.status_message = f"{prefix} Checking for {sim_label}..."
                sim_pos = self.locate_sim_template(sim_key)
                if sim_pos is not None:
                    _click_sim_at(sim_key, sim_label, sim_pos)
                    w_start = time.time()
                    while (time.time() - w_start) < 1.5:
                        if stop_handler.is_stopped() or not self.is_active:
                            return sims_clicked
                        time.sleep(0.05)
            return sims_clicked

        self.status_message = f"{prefix} Checking for Sim 1..."
        sim1_pos = self.locate_sim_template("sim1")
        if sim1_pos is not None:
            _click_sim_at("sim1", "Sim 1", sim1_pos)

            # Wait 2 seconds
            wait_start = time.time()
            while (time.time() - wait_start) < 2.0:
                if stop_handler.is_stopped() or not self.is_active:
                    return sims_clicked
                time.sleep(0.05)

            # Check Sim 3
            self.status_message = f"{prefix} Checking for Sim 3..."
            sim3_pos = self.locate_sim_template("sim3")
            if sim3_pos is not None:
                _click_sim_at("sim3", "Sim 3", sim3_pos)

                # Wait another 2 seconds
                wait_start = time.time()
                while (time.time() - wait_start) < 2.0:
                    if stop_handler.is_stopped() or not self.is_active:
                        return sims_clicked
                    time.sleep(0.05)

                # Check Sim 2
                self.status_message = f"{prefix} Checking for Sim 2..."
                sim2_pos = self.locate_sim_template("sim2")
                if sim2_pos is not None:
                    _click_sim_at("sim2", "Sim 2", sim2_pos)
                    time.sleep(0.5)
            else:
                sim2_pos = self.locate_sim_template("sim2")
                if sim2_pos is not None:
                    _click_sim_at("sim2", "Sim 2", sim2_pos)
                    time.sleep(0.5)
        else:
            # Sim 1 not found: check if Sim 3 or Sim 2 are available
            sim3_pos = self.locate_sim_template("sim3")
            if sim3_pos is not None:
                _click_sim_at("sim3", "Sim 3", sim3_pos)
                wait_start = time.time()
                while (time.time() - wait_start) < 2.0:
                    if stop_handler.is_stopped() or not self.is_active:
                        return sims_clicked
                    time.sleep(0.05)
                sim2_pos = self.locate_sim_template("sim2")
                if sim2_pos is not None:
                    _click_sim_at("sim2", "Sim 2", sim2_pos)
                    time.sleep(0.5)
            else:
                sim2_pos = self.locate_sim_template("sim2")
                if sim2_pos is not None:
                    _click_sim_at("sim2", "Sim 2", sim2_pos)
                    time.sleep(0.5)

        return sims_clicked

    def _wait_for_approach(self, max_wait_seconds: float, reason: str = "Approach") -> None:
        """
        Waits for the character to move closer towards an interactable target (banner or sim).
        Monitors character position to detect when character movement begins and settles (stops moving for >= 0.4s).
        Keeps self.is_interacting = True throughout so anti-stuck is not triggered while approaching.
        """
        if max_wait_seconds <= 0:
            return

        start_time = time.time()
        start_pos = self.latest_pos
        has_moved = False
        last_move_time = start_time
        last_check_pos = start_pos

        _log(f"  [{reason}] Character approaching target (allowing up to {max_wait_seconds:.1f}s)...")

        max_ticks = max(10, int(max_wait_seconds / 0.05) + 5)
        tick = 0

        while (time.time() - start_time) < max_wait_seconds and tick < max_ticks:
            tick += 1
            if stop_handler.is_stopped() or not self.is_active:
                return

            curr_pos = self.latest_pos
            now = time.time()

            if curr_pos is not None and last_check_pos is not None:
                dist_from_last = math.hypot(curr_pos[0] - last_check_pos[0], curr_pos[1] - last_check_pos[1])
                if dist_from_last > 1.5:
                    has_moved = True
                    last_move_time = now
                    last_check_pos = curr_pos
                elif has_moved and (now - last_move_time) >= 0.4:
                    _log(f"  [{reason}] Character reached target and settled ({now - start_time:.2f}s).")
                    break

            rem = max(0.0, max_wait_seconds - (now - start_time))
            self.status_message = f"[{reason}] Approaching ({rem:.1f}s)..."
            time.sleep(0.05)

        elapsed = time.time() - start_time
        _log(f"  [{reason}] Approach window finished ({elapsed:.2f}s).")

    def locate_loot(self) -> Optional[Tuple[int, int]]:
        """
        Locates a loot label on screen matching ui/loot1.png using multi-scale template matching.
        Returns desktop absolute coordinates (X, Y) of the loot label center, or None if not found.
        """
        if self.loot1_img is None:
            self._load_loot_template()

        if self.loot1_img is None:
            return None

        capt = self._get_capturer()
        try:
            screen = capt.capture()
        except Exception as e:
            _log(f"  [WARNING] Screen capture failed during loot search: {e}")
            return None

        if screen is None or screen.size == 0:
            return None

        mon_left = 0
        mon_top = 0
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        g_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        g_tmpl = cv2.cvtColor(self.loot1_img, cv2.COLOR_BGR2GRAY)
        th, tw = g_tmpl.shape[:2]
        sh, sw = g_screen.shape[:2]

        best_val = -1.0
        best_loc = None
        best_scale = 1.0

        if sw >= tw and sh >= th:
            res = cv2.matchTemplate(g_screen, g_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            best_val = float(max_val)
            best_loc = max_loc

        if best_val < self.loot_match_threshold:
            for scale in [0.70, 0.80, 0.90, 1.10, 1.20, 1.30]:
                sc_w = int(tw * scale)
                sc_h = int(th * scale)
                if sc_w > sw or sc_h > sh or sc_w < 15 or sc_h < 15:
                    continue
                resized = cv2.resize(g_tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(g_screen, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = float(max_val)
                    best_loc = max_loc
                    best_scale = scale

        if best_val >= self.loot_match_threshold and best_loc is not None:
            cx = int(best_loc[0] + (tw * best_scale) / 2)
            cy = int(best_loc[1] + (th * best_scale) / 2)
            desktop_x = mon_left + cx
            desktop_y = mon_top + cy
            _log(f"  [LOOT MATCH] Found loot1 (conf={best_val:.2f}, scale={best_scale:.2f}) at screen ({desktop_x}, {desktop_y})")
            return desktop_x, desktop_y

        return None

    def collect_loot(self, max_pickups: Optional[int] = None, approach_wait: Optional[float] = None) -> int:
        """
        Scans screen for loot labels matching ui/loot1.png.
        Clicks left mouse button on each detected loot item and scans again.
        Stops when no more loot is detected or max_pickups reached.
        Returns total number of loots clicked.
        """
        prior_interacting = self.is_interacting
        self.is_interacting = True
        self.stuck_counter = 0
        self.release_all_keys()
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        try:
            limit = max_pickups if max_pickups is not None else self.max_loot_pickups
            _log("\n[AUTOPILOT] >>> Scanning screen for LOOT (ui/loot1.png)...")
            self.status_message = "[LOOT] Scanning screen for loot..."

            picked_count = 0
            while picked_count < limit:
                if stop_handler.is_stopped() or not self.is_active:
                    break

                loot_pos = self.locate_loot()
                if loot_pos is None:
                    if picked_count == 0:
                        _log("  [LOOT] No loot detected on screen.")
                    else:
                        _log(f"  [LOOT] Finished picking up {picked_count} loot item(s). None remaining.")
                    break

                lx, ly = self.move_mouse_inside_game(loot_pos[0], loot_pos[1])
                picked_count += 1
                _log(f"  [LOOT #{picked_count}] Found loot at ({lx}, {ly}). Clicking left mouse button...")
                self.status_message = f"[LOOT] Picking #{picked_count} at ({lx}, {ly})"

                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.08)
                    pydirectinput.mouseUp(button="left")

                eff_app_wait = approach_wait if approach_wait is not None else self.loot_approach_wait_seconds
                if eff_app_wait > 0:
                    self._wait_for_approach(eff_app_wait, reason=f"LOOT #{picked_count}")

                time.sleep(self.loot_pickup_wait_seconds)

            self.status_message = f"Loot Check Complete ({picked_count} picked). Resuming route..."
            return picked_count
        finally:
            self.is_interacting = prior_interacting
            now = time.time()
            self.last_progress_pos = self.latest_pos
            self.last_progress_time = now
            self.last_known_time = now
            self.stuck_counter = 0

    def get_game_center_coords(self) -> Tuple[int, int]:
        """Returns safe screen coordinates inside the game window."""
        bounds = window_focuser.get_game_window_bounds()
        if bounds:
            left, top, right, bottom = bounds
            return (left + right) // 2, (top + bottom) // 2
        # Fallback to monitor center
        capt = self._get_capturer()
        mon_left, mon_top, mon_w, mon_h = 0, 0, 1920, 1080
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon = monitors[self.monitor_idx]
                mon_left = mon.get("left", 0)
                mon_top = mon.get("top", 0)
                mon_w = mon.get("width", 1920)
                mon_h = mon.get("height", 1080)
        return mon_left + mon_w // 2, mon_top + mon_h // 2

    def clamp_coords_to_game_window(self, x: int, y: int) -> Tuple[int, int]:
        """Clamps desktop coordinates (x, y) to guarantee they are strictly inside the game window."""
        bounds = window_focuser.get_game_window_bounds()
        if bounds:
            left, top, right, bottom = bounds
            margin = 60
            clamped_x = max(left + margin, min(right - margin, x))
            clamped_y = max(top + margin, min(bottom - margin, y))
            return clamped_x, clamped_y
        return x, y

    def move_mouse_inside_game(self, x: Optional[int] = None, y: Optional[int] = None) -> Tuple[int, int]:
        """
        Moves the mouse cursor to (x, y) or game center, strictly clamped to the game window.
        Uses Win32 SetCursorPos and MOUSEEVENTF_VIRTUALDESK SendInput for multi-monitor accuracy.
        """
        if x is None or y is None:
            x, y = self.get_game_center_coords()
        else:
            x, y = self.clamp_coords_to_game_window(x, y)

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        # 1. Direct Win32 SetCursorPos with input desktop attachment
        window_focuser._attach_input_desktop()
        try:
            import ctypes
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
        except Exception:
            pass

        # 2. Multi-monitor absolute virtual desktop mouse event
        try:
            import ctypes
            u32 = ctypes.windll.user32
            v_left = u32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
            v_top = u32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
            v_width = u32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            v_height = u32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
            if v_width > 0 and v_height > 0:
                norm_x = int(((x - v_left) * 65535) / v_width)
                norm_y = int(((y - v_top) * 65535) / v_height)
                u32.mouse_event(0x8000 | 0x4000 | 0x0001, norm_x, norm_y, 0, 0)
        except Exception:
            pass

        if pydirectinput:
            try:
                pydirectinput.moveTo(int(x), int(y))
            except Exception:
                pass

        time.sleep(0.04)
        return x, y

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

            # Ensure cursor starts inside the game window
            self.move_mouse_inside_game()

            # Check if all interactions are disabled
            if not self.click_banner_enabled and not self.right_click_after_banner_enabled and not self.middle_click_hold_enabled:
                return True

            self.status_message = "[YELLOW ZONE] Interacting..."
            _log("\n[AUTOPILOT] >>> REACHED YELLOW ZONE! Executing encounter activation sequence...")

            # 0. Check for Sims first! (Fallback safety in case dynamic resync landed on yellow zone directly)
            sims = self._detect_and_click_sims(prefix="[YELLOW ZONE]")
            if sims:
                _log(f"  [YELLOW ZONE] Sims selected ({', '.join(sims)}). Encounter activated via Sim!")
                return True

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

            if stop_handler.is_stopped() or not self.is_active:
                return False

            # 2. Click right mouse button once if enabled
            if self.right_click_after_banner_enabled:
                self.move_mouse_inside_game()
                _log("  [ACTION 2/3] Clicking right mouse button once inside game...")
                self.status_message = "[YELLOW ZONE] Right-Clicking..."
                if pydirectinput:
                    pydirectinput.rightClick()
                    time.sleep(0.08)
                    pydirectinput.mouseUp(button="right")
                time.sleep(0.15)
            else:
                _log("  [CONFIG] Right click after banner disabled. Skipping...")

            if stop_handler.is_stopped() or not self.is_active:
                return False

            # 3. Press and hold middle mouse button if enabled
            if self.middle_click_hold_enabled:
                self.move_mouse_inside_game()
                hold_sec = self.middle_click_hold_seconds
                _log(f"  [ACTION 3/3] Pressing and holding middle mouse button for {hold_sec:.1f}s inside game...")
                try:
                    if pydirectinput:
                        pydirectinput.mouseDown(button="middle")
                    hold_start = time.time()
                    while (time.time() - hold_start) < hold_sec:
                        if stop_handler.is_stopped() or not self.is_active:
                            break
                        rem_h = max(0.0, hold_sec - (time.time() - hold_start))
                        self.status_message = f"[YELLOW ZONE] Holding Middle Mouse ({rem_h:.1f}s)..."
                        time.sleep(0.05)
                finally:
                    if pydirectinput:
                        pydirectinput.mouseUp(button="middle")
                    try:
                        import ctypes
                        ctypes.windll.user32.mouse_event(0x0040, 0, 0, 0, 0)
                    except Exception:
                        pass
                    _log("  [ACTION 3/3] Middle mouse button released.")
                time.sleep(0.1)
            else:
                _log("  [CONFIG] Middle click hold disabled (middle_click_hold_enabled=false). Skipping...")

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

        self.release_all_keys()
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.move_mouse_inside_game()

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

        if stop_handler.is_stopped() or not self.is_active:
            return False

        # 3. Fallback: If no sims are available, click encounter_banner, right-click, middle-click for 4s
        if not sims_clicked:
            _log("  [PINK DOT] No sims available. Falling back to Encounter Banner sequence...")
            self.status_message = "[PINK DOT] Finding Encounter Banner..."
            banner_pos = None
            for attempt in range(1, self.banner_search_attempts + 1):
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                banner_pos = self.locate_encounter_banner()
                if banner_pos is not None:
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
                _log(f"  [WARNING] Encounter banner not detected on screen. Positioning cursor inside game.")
                self.move_mouse_inside_game()

            if stop_handler.is_stopped() or not self.is_active:
                return False

            # Click right button of mouse
            self.move_mouse_inside_game()
            _log("  [ACTION 2/3] Clicking right mouse button once inside game...")
            self.status_message = "[PINK DOT] Right-Clicking..."
            if pydirectinput:
                pydirectinput.rightClick()
                time.sleep(0.08)
                pydirectinput.mouseUp(button="right")
            time.sleep(0.15)

            if stop_handler.is_stopped() or not self.is_active:
                return False

            # Press and hold middle button of mouse for configured seconds
            self.move_mouse_inside_game()
            hold_sec = self.middle_click_hold_seconds
            _log(f"  [ACTION 3/3] Pressing and holding middle mouse button for {hold_sec:.1f}s inside game...")
            try:
                if pydirectinput:
                    pydirectinput.mouseDown(button="middle")
                hold_start = time.time()
                while (time.time() - hold_start) < hold_sec:
                    if stop_handler.is_stopped() or not self.is_active:
                        break
                    rem_h = max(0.0, hold_sec - (time.time() - hold_start))
                    self.status_message = f"[PINK DOT] Holding Middle Mouse ({rem_h:.1f}s)..."
                    time.sleep(0.05)
            finally:
                if pydirectinput:
                    pydirectinput.mouseUp(button="middle")
                try:
                    import ctypes
                    ctypes.windll.user32.mouse_event(0x0040, 0, 0, 0, 0)
                except Exception:
                    pass
                _log("  [ACTION 3/3] Middle mouse button released.")
            time.sleep(0.1)

            if stop_handler.is_stopped() or not self.is_active:
                return False

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
        return True

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

    def _execute_stuck_recovery(self, reason: str = "Stuck"):
        """
        Executes stuck / lost-location recovery maneuver:
        1. Releases current movement keys.
        2. Backtracks towards previously known location.
        3. If orbiting yellow zone: switches to next perimeter point in yellow zone.
           If traveling on green route: skips the waypoint where the character got stuck or lost tracking.
        4. Resumes navigation towards the next waypoint / perimeter point ahead.
        """
        if self.is_orbiting:
            _log(f"\n[AUTOPILOT RECOVERY] {reason} during Yellow Zone Orbit.")
            _log(f"[AUTOPILOT RECOVERY] Unsticking from geometry inside yellow area...")
            self.release_all_keys()

            backtrack_keys = []
            if self.last_held_keys:
                opp = {"w": "s", "s": "w", "a": "d", "d": "a"}
                backtrack_keys = [opp[k] for k in self.last_held_keys if k in opp]
            if not backtrack_keys:
                backtrack_keys = ["s"]

            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            self.is_simulating_key = True
            self.held_keys = set(backtrack_keys)
            try:
                for _ in range(1):
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
                        time.sleep(0.05)
                    else:
                        time.sleep(self.step_duration)
            finally:
                self.is_simulating_key = False
                self.held_keys.clear()

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
        _log(f"[AUTOPILOT RECOVERY] Moving back to previously known location...")

        # 1. Release current movement keys
        self.release_all_keys()

        # 2. Determine reverse/backtrack keys
        backtrack_keys = []
        if self.latest_pos is not None and self.last_known_pos is not None:
            dist_to_last = math.hypot(self.last_known_pos[0] - self.latest_pos[0], self.last_known_pos[1] - self.latest_pos[1])
            if dist_to_last > 4.0:
                backtrack_keys = self.compute_wasd_keys(self.latest_pos, self.last_known_pos)

        if not backtrack_keys and self.last_held_keys:
            opp = {"w": "s", "s": "w", "a": "d", "d": "a"}
            backtrack_keys = [opp[k] for k in self.last_held_keys if k in opp]

        if not backtrack_keys and curr_wp and self.latest_pos:
            target_pos = (curr_wp["x"], curr_wp["y"])
            dx = -(target_pos[0] - self.latest_pos[0])
            dy = -(target_pos[1] - self.latest_pos[1])
            virtual_back = (self.latest_pos[0] + dx, self.latest_pos[1] + dy)
            backtrack_keys = self.compute_wasd_keys(self.latest_pos, virtual_back)

        if not backtrack_keys:
            backtrack_keys = ["s"]

        # Ensure window focused
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        # Send 2 backstep pulses to unstick from geometry
        self.is_simulating_key = True
        self.held_keys = set(backtrack_keys)
        try:
            for _ in range(2):
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
                    time.sleep(0.05)
                else:
                    time.sleep(self.step_duration)
        finally:
            self.is_simulating_key = False
            self.held_keys.clear()

        # 3. Skip the waypoint where we got stuck / lost
        if curr_idx < len(self.movement_path.waypoints) - 1:
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

    def release_all_keys(self):
        """Releases all currently held WASD movement keys."""
        self.is_simulating_key = False
        for key in list(self.held_keys):
            if pydirectinput:
                try:
                    pydirectinput.keyUp(key)
                except Exception:
                    pass
        self.held_keys.clear()

        # Extra safety Win32 release for common keys
        if pydirectinput:
            for k in ["w", "a", "s", "d"]:
                try:
                    pydirectinput.keyUp(k)
                except Exception:
                    pass

    def _run_navigation_loop(self):
        """Background thread executing sustained WASD pulses into the game."""
        step_count = 0
        while self.is_active and not stop_handler.is_stopped():
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
                    # Periodic right-clicking while orbiting if enabled
                    if self.orbit_constant_right_click_enabled:
                        if (now - self.last_orbit_right_click) >= self.orbit_right_click_interval_seconds:
                            self.last_orbit_right_click = now
                            self.move_mouse_inside_game()
                            if pydirectinput:
                                try:
                                    pydirectinput.rightClick()
                                    time.sleep(0.02)
                                    pydirectinput.mouseUp(button="right")
                                except Exception:
                                    pass

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
                    self.release_all_keys()
                    self.status_message = "Route Completed!"
                    _log("\n[AUTOPILOT] >>> ALL WAYPOINTS COMPLETED! Reached destination.")
                    break

                target_pos = (target["x"], target["y"])
                dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
                self.last_distance = dist
                self.last_target = target

                # 3. Waypoint arrival check
                is_reached = dist <= self.arrival_threshold
                if not is_reached and target.get("action") == "pink_encounter" and target.get("pink_pos"):
                    pink_p = target["pink_pos"]
                    dist_to_pink = math.hypot(pink_p[0] - current_pos[0], pink_p[1] - current_pos[1])
                    if dist_to_pink <= max(self.arrival_threshold, 30.0):
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
                        self.release_all_keys()
                        self.status_message = "Route Finished!"
                        _log("\n[AUTOPILOT] >>> DESTINATION REACHED!")
                        break
                    target = next_target
                    target_pos = (target["x"], target["y"])
                    dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
                    self.last_distance = dist
                    self.last_target = target
                    # Reset stuck counters on normal waypoint arrival
                    self.stuck_counter = 0
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
                    if (now - self.last_orbit_right_click) >= self.orbit_right_click_interval_seconds:
                        self.last_orbit_right_click = now
                        self.move_mouse_inside_game()
                        if pydirectinput:
                            try:
                                pydirectinput.rightClick()
                                time.sleep(0.02)
                                pydirectinput.mouseUp(button="right")
                            except Exception:
                                pass

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
        if self.waiting_for_green_light:
            status_msg = "WAITING FOR GREEN LIGHT (Verify Loot - Press 'G' to Resume)"
        elif tracking_lost:
            status_msg = "Tracking Lost (Holding Position)"
        elif is_paused or self.is_paused:
            status_msg = "Autopilot PAUSED (Press 'F4' to resume)"

        return {
            "is_active": self.is_active,
            "is_paused": self.is_paused or is_paused,
            "is_completed": self.is_completed,
            "waiting_for_green_light": self.waiting_for_green_light,
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
        }

