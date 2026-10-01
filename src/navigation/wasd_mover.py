"""
WASD key calculation, game window coordinate clamping, mouse movement inside game, and key releases.
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

from src.navigation.human_mouse import human_move_to

_log = log_msg
pydirectinput = DynamicModuleProxy("pydirectinput")
window_focuser = DynamicModuleProxy("window_focuser")
stop_handler = DynamicModuleProxy("stop_handler")
keyboard = DynamicModuleProxy("keyboard")


class WasdMoverMixin:
    """WASD key calculation, game window coordinate clamping, mouse movement inside game, and key releases."""

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

    # =========================================================================
    # RUN SESSION TIMING & PERSISTENCE
    # =========================================================================


    def get_game_center_coords(self) -> Tuple[int, int]:
        """Returns safe screen coordinates inside the game window."""
        bounds = window_focuser.get_game_window_bounds()
        if bounds and isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            left, top, right, bottom = bounds
            return (left + right) // 2, (top + bottom) // 2
        # Fallback to monitor center
        capt = self._get_capturer()
        mon_left, mon_top, mon_w, mon_h = 0, 0, 1920, 1080
        if hasattr(capt, "get_monitors") and callable(getattr(capt, "get_monitors")):
            monitors = capt.get_monitors()
            if 0 <= self.monitor_idx < len(monitors):
                mon = monitors[self.monitor_idx]
                mon_left = mon.get("left", 0)
                mon_top = mon.get("top", 0)
                mon_w = mon.get("width", 1920)
                mon_h = mon.get("height", 1080)
        elif getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
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
        if bounds and isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            left, top, right, bottom = bounds
            margin = 60
            clamped_x = max(left + margin, min(right - margin, x))
            clamped_y = max(top + margin, min(bottom - margin, y))
            return clamped_x, clamped_y
        return x, y


    def move_mouse_inside_game(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        human_like: Optional[bool] = None,
    ) -> Tuple[int, int]:
        """
        Moves the mouse cursor to (x, y) or game center, strictly clamped to the game window.
        Uses human-like curved movement with random speed (+/- 20%) to avoid robotic cursor jumps.
        """
        if x is None or y is None:
            target_x, target_y = self.get_game_center_coords()
        else:
            target_x, target_y = self.clamp_coords_to_game_window(x, y)

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        # Check if human_like is enabled (defaulting to True or config value)
        use_human = getattr(self, "human_mouse_enabled", True) if human_like is None else human_like
        speed_var = getattr(self, "mouse_speed_variation_pct", 20.0)

        if use_human:
            return human_move_to(
                target_x,
                target_y,
                clamp_fn=self.clamp_coords_to_game_window,
                speed_variance_pct=speed_var,
                monitor_idx=self.monitor_idx,
            )

        # Direct instant snap fallback (if human movement is explicitly disabled)
        window_focuser._attach_input_desktop()
        try:
            import ctypes
            ctypes.windll.user32.SetCursorPos(int(target_x), int(target_y))
        except Exception:
            pass

        # Multi-monitor absolute virtual desktop mouse event
        try:
            import ctypes
            u32 = ctypes.windll.user32
            v_left = u32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
            v_top = u32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
            v_width = u32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            v_height = u32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
            if v_width > 0 and v_height > 0:
                norm_x = int(((target_x - v_left) * 65535) / v_width)
                norm_y = int(((target_y - v_top) * 65535) / v_height)
                u32.mouse_event(0x8000 | 0x4000 | 0x0001, norm_x, norm_y, 0, 0)
        except Exception:
            pass

        if pydirectinput:
            try:
                pydirectinput.moveTo(int(target_x), int(target_y), _pause=False)
            except Exception:
                pass

        time.sleep(0.04)
        return target_x, target_y


    def release_all_keys(self, force: bool = False):
        """Releases all currently held WASD movement keys."""
        if not self.held_keys and not force and not self.is_simulating_key:
            return
        self.is_simulating_key = False
        for key in list(self.held_keys):
            if pydirectinput:
                try:
                    pydirectinput.keyUp(key)
                except Exception:
                    pass
        self.held_keys.clear()

        # Extra safety Win32 release for common keys only on explicit force
        if force and pydirectinput:
            for k in ["w", "a", "s", "d"]:
                try:
                    pydirectinput.keyUp(k)
                except Exception:
                    pass

