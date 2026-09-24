"""
Background combat attack pulsing thread and combat toggle controls.
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


class PersistentCombatMixin:
    """Background combat attack pulsing thread and combat toggle controls."""

    def enable_persistent_combat(
        self,
        interval: Optional[float] = None,
        action: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """Enables persistent combat attacking and launches the dedicated background combat engine."""
        self.persistent_combat_active = True
        self.persistent_right_click_active = True
        if interval is not None:
            self.persistent_combat_interval = float(interval)
            self.persistent_right_click_interval = float(interval)
        if action is not None:
            self.persistent_combat_action = str(action).lower().strip()
        if key is not None:
            self.persistent_combat_key = str(key).lower().strip()
        self._start_persistent_combat_thread()


    def enable_persistent_right_click(self, interval: Optional[float] = None):
        """Enables persistent combat attacking (backward-compatible method name)."""
        self.enable_persistent_combat(interval=interval)


    def disable_persistent_combat(self):
        """Disables persistent combat attacking."""
        self.persistent_combat_active = False
        self.persistent_right_click_active = False
        if self._combat_thread is not None and self._combat_thread.is_alive() and threading.current_thread() != self._combat_thread:
            self._combat_thread.join(timeout=0.5)
            self._combat_thread = None


    def disable_persistent_right_click(self):
        """Disables persistent combat right-clicking (backward-compatible method name)."""
        self.disable_persistent_combat()


    def toggle_persistent_combat(self) -> bool:
        """Toggles persistent combat attacking ON and OFF."""
        if self.persistent_combat_active or self.persistent_right_click_active:
            self.disable_persistent_combat()
            desc = f"Key '{self.persistent_combat_key.upper()}'" if self.persistent_combat_action == "key" else "Mouse Click"
            self.latest_recovery_event = f"Combat Attack PAUSED [F3] ({desc})"
            _log(f"\n[COMBAT] >>> Persistent combat attack ({desc}) PAUSED by hotkey [F3].")
            return False
        else:
            self.enable_persistent_combat()
            desc = f"Key '{self.persistent_combat_key.upper()}'" if self.persistent_combat_action == "key" else "Mouse Click"
            self.latest_recovery_event = f"Combat Attack RESUMED [F3] ({desc} @ {self.persistent_combat_interval:.2f}s)"
            _log(f"\n[COMBAT] >>> Persistent combat attack ({desc}) RESUMED by hotkey [F3] ({self.persistent_combat_interval:.2f}s).")
            return True


    def toggle_persistent_right_click(self) -> bool:
        """Toggles persistent combat right-clicking (backward-compatible method name)."""
        return self.toggle_persistent_combat()


    def _start_persistent_combat_thread(self):
        """Spawns background combat thread to continuously pulse combat attack every interval."""
        if self._combat_thread is not None and self._combat_thread.is_alive():
            return
        self._combat_thread = threading.Thread(target=self._run_persistent_combat_loop, daemon=True)
        self._combat_thread.start()


    def _trigger_persistent_combat_if_due(self, now: Optional[float] = None) -> bool:
        """
        Executes a periodic combat attack (key press 't', right-click, etc.) inside the game if persistent combat mode is active.
        Guarantees that attack execution continues during ALL events until the final destination is reached.
        If suppress_combat_during_approach is active and character is walking up to an interactable, pulses are suppressed.
        """
        if getattr(self, "in_hideout", False):
            return False

        if (not self.persistent_combat_active and not self.persistent_right_click_active and not (self.orbit_constant_right_click_enabled and self.is_orbiting)) or stop_handler.is_stopped():
            return False

        if getattr(self, "is_holding_mouse", False):
            return False

        if getattr(self, "is_approaching_interactable", False) and getattr(self, "suppress_combat_during_approach", False):
            return False

        if now is None:
            now = time.time()

        combat_int = getattr(self, "persistent_combat_interval", getattr(self, "persistent_right_click_interval", 0.65))
        if (now - self.last_orbit_right_click) >= combat_int:
            self.last_orbit_right_click = now
            action_type = getattr(self, "persistent_combat_action", "key")

            if action_type in ("key", "right_click"):
                attack_key = getattr(self, "persistent_combat_key", "t")
                if pydirectinput and attack_key:
                    try:
                        pydirectinput.keyDown(attack_key)
                        time.sleep(0.02)
                        pydirectinput.keyUp(attack_key)
                    except Exception:
                        pass
            elif action_type == "middle_click":
                attack_key = "q"
                if pydirectinput:
                    try:
                        pydirectinput.keyDown(attack_key)
                        time.sleep(0.02)
                        pydirectinput.keyUp(attack_key)
                    except Exception:
                        pass
            elif action_type == "left_click":
                self.move_mouse_inside_game()
                if pydirectinput:
                    try:
                        pydirectinput.click()
                        time.sleep(0.02)
                        pydirectinput.mouseUp(button="left")
                    except Exception:
                        pass
            return True
        return False


    def _trigger_persistent_right_click_if_due(self, now: Optional[float] = None) -> bool:
        """Backward-compatible alias for _trigger_persistent_combat_if_due."""
        return self._trigger_persistent_combat_if_due(now)


    def _run_persistent_combat_loop(self):
        """
        Dedicated background thread guaranteeing that combat attacking NEVER stops during ANY event
        (e.g., waiting for green light, loot approach/pickup, stop, wait, transit, orbit, etc.)
        until the destination Red Dot is reached or navigation is halted.
        """
        combat_desc = f"Key '{self.persistent_combat_key.upper()}'" if self.persistent_combat_action == "key" else "Mouse Click"
        _log(f"[COMBAT THREAD] Persistent combat engine started ({combat_desc} @ interval={self.persistent_combat_interval:.2f}s).")
        while not stop_handler.is_stopped() and self.is_active and (self.persistent_combat_active or self.persistent_right_click_active):
            self._trigger_persistent_combat_if_due()
            time.sleep(0.04)
        _log("[COMBAT THREAD] Persistent combat engine stopped.")

