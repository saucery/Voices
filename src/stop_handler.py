"""
Emergency Stop & Hotkey Handler Module
Listens for global emergency stop hotkey (F1) to immediately halt bot execution,
release held keys, and restore full manual keyboard/mouse control to the user.
"""

import threading
import time
from typing import Callable, Optional, List

try:
    import keyboard
except ImportError:
    keyboard = None

try:
    import pydirectinput
except ImportError:
    pydirectinput = None


class EmergencyStopHandler:
    """Manages global emergency stop hotkey (F1) listener."""

    def __init__(self, stop_key: str = "f1", on_stop_callback: Optional[Callable[[], None]] = None):
        """
        Initialize EmergencyStopHandler.

        :param stop_key: Key name for emergency stop (default: 'f1').
        :param on_stop_callback: Optional callback function invoked when stop triggered.
        """
        self.stop_key = stop_key.lower()
        self.on_stop_callback = on_stop_callback
        self.stop_requested = False
        self._listener_running = False
        self.tracked_keys: List[str] = ["w", "a", "s", "d", "q", "e", "space", "shift", "ctrl", "alt"]

        self.start_listener()

    def start_listener(self):
        """Starts global hotkey listener for emergency stop."""
        if keyboard and not self._listener_running:
            try:
                keyboard.add_hotkey(self.stop_key, self.trigger_stop, suppress=False)
                self._listener_running = True
                print(f"[EMERGENCY STOP ENGINE] F1 Stop Switch Active. Press '{self.stop_key.upper()}' anytime to halt bot.")
            except Exception as e:
                print(f"[EMERGENCY STOP ENGINE] Warning: Could not register global hotkey ({e})")

    def trigger_stop(self):
        """Triggers emergency stop, releases all keys, and invokes callback."""
        if not self.stop_requested:
            self.stop_requested = True
            print(f"\n\n{'!' * 75}")
            print(f" !!! EMERGENCY STOP ACTIVATED ({self.stop_key.upper()}) !!!")
            print(" Bot automation halted immediately. Control returned to user.")
            print(f"{'!' * 75}\n")

            self.release_all_keys()

            if self.on_stop_callback:
                try:
                    self.on_stop_callback()
                except Exception as e:
                    print(f"Error in stop callback: {e}")

    def is_stopped(self) -> bool:
        """
        Checks if emergency stop has been requested.
        Also polls fallback keyboard state if global hook wasn't active.
        """
        if self.stop_requested:
            return True

        if keyboard:
            try:
                if keyboard.is_pressed(self.stop_key):
                    self.trigger_stop()
                    return True
            except Exception:
                pass

        return self.stop_requested

    def release_all_keys(self):
        """Releases any potentially held movement, action keys, or mouse buttons."""
        print("  [SAFETY] Releasing all held keys and mouse buttons...")
        for key in self.tracked_keys:
            try:
                if pydirectinput:
                    pydirectinput.keyUp(key)
                if keyboard:
                    keyboard.release(key)
            except Exception:
                pass

        try:
            if pydirectinput:
                pydirectinput.mouseUp(button="right")
                pydirectinput.mouseUp(button="left")
                pydirectinput.mouseUp(button="middle")
            import ctypes
            user32 = ctypes.windll.user32
            # 0x0004 = LEFTUP, 0x0010 = RIGHTUP, 0x0040 = MIDDLEUP
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            user32.mouse_event(0x0010, 0, 0, 0, 0)
            user32.mouse_event(0x0040, 0, 0, 0, 0)
        except Exception:
            pass

    def reset(self):
        """Resets the emergency stop state for a new session."""
        self.stop_requested = False


# Global singleton instance
stop_handler = EmergencyStopHandler()
