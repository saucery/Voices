"""
Targeted mouse clicking near character, Hold Q execution, and encounter banner detection.
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


class ZoneActionsMixin:
    """Targeted mouse clicking near character, Hold Q execution, and encounter banner detection."""

    def _get_screen_char_center(self, screen: np.ndarray, capt: Any) -> Tuple[float, float]:
        """Calculates (x, y) coordinates of the character relative to the captured screen."""
        sh, sw = screen.shape[:2]
        char_center = (float(sw // 2), float(sh // 2))
        try:
            bounds = window_focuser.get_game_window_bounds()
            if bounds and isinstance(bounds, (list, tuple)) and len(bounds) == 4:
                gw_l, gw_t, gw_r, gw_b = bounds
                mon_left = 0
                mon_top = 0
                if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
                    monitors = capt._sct.monitors
                    if 0 <= self.monitor_idx < len(monitors):
                        mon_left = monitors[self.monitor_idx].get("left", 0)
                        mon_top = monitors[self.monitor_idx].get("top", 0)
                rcx = ((gw_l + gw_r) // 2) - mon_left
                rcy = ((gw_t + gw_b) // 2) - mon_top
                if 0 <= rcx < sw and 0 <= rcy < sh:
                    char_center = (float(rcx), float(rcy))
        except Exception:
            pass
        return char_center


    def click_mouse_near_character(
        self,
        button: str = "left",
        offset_x: int = 60,
        offset_y: int = 40,
        follow_up_key: Optional[str] = None,
        follow_up_delay: float = 2.0,
        label: str = "Frost Bomb",
    ) -> Tuple[int, int]:
        """
        Moves mouse next to character location, clicks mouse button, and optionally
        waits follow_up_delay to press a follow-up keyboard key (e.g. 'R').
        """
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        cx, cy = self.get_game_center_coords()
        tx = cx + int(offset_x)
        ty = cy + int(offset_y)
        mx, my = self.move_mouse_inside_game(tx, ty)
        _log(f"    [ACTION] Clicking {button.upper()} mouse next to character at ({mx}, {my}) [{label}]...")
        self.status_message = f"Clicking {button.upper()} ({label})"

        if pydirectinput:
            if button.lower() == "left":
                pydirectinput.click()
                time.sleep(0.05)
                pydirectinput.mouseUp(button="left")
            elif button.lower() == "right":
                pydirectinput.rightClick()
            elif button.lower() == "middle":
                pydirectinput.middleClick()
        time.sleep(0.08)

        if follow_up_key and not stop_handler.is_stopped():
            delay_start = time.time()
            while (time.time() - delay_start) < follow_up_delay:
                if stop_handler.is_stopped():
                    return mx, my
                time.sleep(0.02)

            _log(f"    [ACTION] Pressing follow-up key '{follow_up_key.upper()}' ({follow_up_delay:.1f}s after {button.upper()} click)...")
            self.status_message = f"Pressing [{follow_up_key.upper()}]..."
            if pydirectinput:
                try:
                    pydirectinput.keyDown(follow_up_key.lower())
                    time.sleep(0.03)
                    pydirectinput.keyUp(follow_up_key.lower())
                except Exception:
                    pass
            time.sleep(0.05)

        return mx, my


    def execute_hold_q(
        self,
        zone_label: str = "ENCOUNTER",
        duration: float = 3.5,
        step: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Executes holding and releasing of 'Q' key with either:
        1. Configurable fixed-duration hold (legacy mode).
        2. Enemy-reactive mode: presses 'Q' ONLY when enemies appear on screen,
           and holds 'Q' until an enemy is near the character or safety timeout expires.
        """
        is_reactive = getattr(self, "hold_q_enemy_reactive_enabled", True)
        if step and "enemy_reactive" in step:
            is_reactive = bool(step.get("enemy_reactive"))

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        # Legacy fixed-duration mode
        if not is_reactive:
            _log(f"    [ACTION] Holding 'Q' key for {duration:.1f}s (fixed duration mode)...")
            self.is_holding_mouse = True
            try:
                if pydirectinput:
                    pydirectinput.keyDown("q")
                h_start = time.time()
                while (time.time() - h_start) < duration:
                    if stop_handler.is_stopped() or (getattr(self, "is_navigating", False) and not self.is_active):
                        break
                    rem_h = max(0.0, duration - (time.time() - h_start))
                    self.status_message = f"[{zone_label}] Holding [Q] ({rem_h:.1f}s)..."
                    time.sleep(0.05)
            finally:
                self.is_holding_mouse = False
                if pydirectinput:
                    pydirectinput.keyUp("q")
                _log("    [ACTION] Key 'Q' released.")
            time.sleep(0.1)
            return True

        # Enemy-reactive mode
        wait_timeout = float(step.get("detect_timeout", self.enemy_detect_wait_timeout)) if step else self.enemy_detect_wait_timeout
        near_thresh = float(step.get("near_distance", self.enemy_near_distance_px)) if step else self.enemy_near_distance_px
        min_hold = float(step.get("min_hold_seconds", self.enemy_hold_min_seconds)) if step else self.enemy_hold_min_seconds
        max_hold = float(step.get("max_hold_seconds", step.get("duration", self.enemy_hold_max_seconds))) if step else self.enemy_hold_max_seconds

        detector = self._get_enemy_detector()
        capt = self._get_capturer()

        pre_cast_button = None
        if step:
            if "pre_cast_button" in step:
                pre_cast_button = str(step["pre_cast_button"]).lower().strip()
            elif step.get("frost_bomb", False):
                pre_cast_button = "left"

        pre_cast_interval = float(step.get("pre_cast_interval", 4.0)) if step else 4.0
        follow_up_key = step.get("follow_up_key") if step else None
        if pre_cast_button and follow_up_key is None and (not step or "follow_up_key" not in step):
            follow_up_key = "r"
        follow_up_delay = float(step.get("follow_up_delay", 2.0)) if step else 2.0

        offset_xy = step.get("click_offset") if step else None
        if not offset_xy and step:
            offset_xy = [step.get("click_offset_x", 60), step.get("click_offset_y", 40)]
        offset_x = int(offset_xy[0]) if offset_xy else 60
        offset_y = int(offset_xy[1]) if offset_xy else 40

        press_attack_key = step.get("combat_key", getattr(self, "persistent_combat_key", "t")) if step else getattr(self, "persistent_combat_key", "t")
        attack_interval = float(step.get("combat_interval", getattr(self, "persistent_combat_interval", 0.65))) if step else getattr(self, "persistent_combat_interval", 0.65)
        last_attack_time = 0.0

        if pre_cast_button:
            self.disable_persistent_combat()
            _log(f"    [FROST BOMB] Executing initial {pre_cast_button.upper()} click near character + follow-up '{follow_up_key.upper() if follow_up_key else 'None'}'...")
            self.click_mouse_near_character(
                button=pre_cast_button,
                offset_x=offset_x,
                offset_y=offset_y,
                follow_up_key=follow_up_key,
                follow_up_delay=follow_up_delay,
                label="Frost Bomb",
            )
            last_pre_cast_time = time.time()
            _log(f"    [ENEMY DETECT] Monitoring screen for enemies (repeat {pre_cast_button.upper()} click every {pre_cast_interval:.1f}s, timeout={wait_timeout:.1f}s, hold={min_hold:.1f}s)...")
            self.status_message = f"[{zone_label}] Frost Bomb active | Monitoring for enemies..."
        else:
            last_pre_cast_time = 0.0
            _log(f"    [ENEMY DETECT] Monitoring screen for enemies while pulsing skill '{press_attack_key.upper()}' before holding 'Q' (timeout={wait_timeout:.1f}s, hold={min_hold:.1f}s)...")
            self.status_message = f"[{zone_label}] Pulsing [{press_attack_key.upper()}] & waiting for enemies..."

        detect_start = time.time()
        enemies_detected = False
        first_det_result: Optional[Dict[str, Any]] = None

        while (time.time() - detect_start) < wait_timeout:
            if stop_handler.is_stopped() or (getattr(self, "is_navigating", False) and not self.is_active):
                return False

            now = time.time()
            if pre_cast_button and (now - last_pre_cast_time) >= pre_cast_interval:
                _log(f"    [FROST BOMB] No enemies detected after {pre_cast_interval:.1f}s. Repeating {pre_cast_button.upper()} click near character + follow-up '{follow_up_key.upper() if follow_up_key else ''}'...")
                self.click_mouse_near_character(
                    button=pre_cast_button,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    follow_up_key=follow_up_key,
                    follow_up_delay=follow_up_delay,
                    label="Frost Bomb",
                )
                last_pre_cast_time = time.time()

            if not pre_cast_button and press_attack_key and (now - last_attack_time) >= attack_interval:
                last_attack_time = now
                if pydirectinput:
                    try:
                        pydirectinput.keyDown(press_attack_key)
                        time.sleep(0.02)
                        pydirectinput.keyUp(press_attack_key)
                    except Exception:
                        pass

            try:
                screen = capt.capture()
            except Exception as e:
                _log(f"    [WARNING] Screen capture failed during enemy detection: {e}")
                screen = None

            if screen is not None and screen.size > 0:
                char_center = self._get_screen_char_center(screen, capt)
                det = detector.detect(screen, character_center=char_center, near_threshold_px=near_thresh)
                if det["detected"]:
                    enemies_detected = True
                    first_det_result = det
                    break

            time.sleep(0.05)

        if not enemies_detected:
            _log(f"    [ENEMY DETECT] No enemies detected within {wait_timeout:.1f}s. Skipping 'Q' hold.")
            return True

        # Enemies detected! Press and hold Q
        e_count = first_det_result["count"] if first_det_result else 1
        min_d = first_det_result["nearest_distance"] if first_det_result else 999.0
        _log(f"    [ACTION] Enemies detected ({e_count} health bar(s), nearest={min_d:.1f}px). Pressing and holding 'Q' key for {min_hold:.1f}s (max={max_hold:.1f}s)...")
        self.is_holding_mouse = True

        try:
            if pydirectinput:
                pydirectinput.keyDown("q")

            hold_start = time.time()
            while (time.time() - hold_start) < max_hold:
                if stop_handler.is_stopped() or (getattr(self, "is_navigating", False) and not self.is_active):
                    break

                elapsed_hold = time.time() - hold_start
                try:
                    screen = capt.capture()
                except Exception as e:
                    screen = None

                det = None
                if screen is not None and screen.size > 0:
                    char_center = self._get_screen_char_center(screen, capt)
                    det = detector.detect(screen, character_center=char_center, near_threshold_px=near_thresh)
                    curr_min_d = det["nearest_distance"]
                    if det["detected"] and det["has_enemy_near"]:
                        if elapsed_hold >= min_hold:
                            _log(f"    [ENEMY REACTIVE] Enemy near character ({curr_min_d:.1f}px <= {near_thresh:.1f}px) & min hold satisfied ({elapsed_hold:.2f}s >= {min_hold:.1f}s). Releasing 'Q'!")
                            break
                        rem_min = max(0.0, min_hold - elapsed_hold)
                        self.status_message = f"[{zone_label}] Holding [Q] (Enemy near! Min hold: {rem_min:.1f}s)..."
                    else:
                        self.status_message = f"[{zone_label}] Holding [Q] (Nearest enemy: {curr_min_d:.0f}px, target: <={near_thresh:.0f}px)..."
                else:
                    self.status_message = f"[{zone_label}] Holding [Q]..."

                if elapsed_hold >= min_hold and min_hold >= max_hold:
                    _log(f"    [ENEMY REACTIVE] Held 'Q' for {elapsed_hold:.2f}s (hold duration={min_hold:.1f}s reached). Releasing 'Q'!")
                    break

                time.sleep(0.05)
            else:
                _log(f"    [ENEMY REACTIVE] Max hold duration ({max_hold:.1f}s) reached. Releasing 'Q'.")
        finally:
            self.is_holding_mouse = False
            if pydirectinput:
                pydirectinput.keyUp("q")
            _log("    [ACTION] Key 'Q' released.")

        time.sleep(0.1)
        return True


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

