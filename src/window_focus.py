"""
Game Window Focus Handler Module
Finds and brings the target game window ('Path of Exile 2' / 'Path Of Exile 2')
into focus / foreground when the bot starts execution or activates navigation.
"""

import json
import os
import sys
import time
from typing import Optional, List, Tuple

try:
    import pygetwindow as gw
except ImportError:
    gw = None

try:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
except Exception:
    user32 = None
    kernel32 = None


class WindowFocusHandler:
    """Finds and focuses the target game window."""

    def __init__(self, target_title: Optional[str] = None, config_path: str = "config.json"):
        """
        Initialize WindowFocusHandler.

        :param target_title: Target window title substring (e.g. 'Path of Exile 2').
        :param config_path: Path to config.json.
        """
        if not target_title and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                target_title = cfg.get("game_window_title")
            except Exception:
                pass

        # Target window titles for Path of Exile 2 / PoE
        self.target_title = target_title or "Path of Exile 2"
        self.target_titles = [
            "Path of Exile 2",
            "Path Of Exile 2",
            "Path of Exile",
            "Path Of Exile",
        ]
        self.target_class = "POEWindowClass"
        self.ignored_substrings = [
            "voices",
            "visualizer",
            "tracker",
            "chrome",
            "exiled exchange",
            "code",
            "firefox",
            "edge",
            "discord",
        ]

    def _attach_input_desktop(self):
        """Attaches calling thread to active input desktop if possible."""
        if not user32:
            return None
        try:
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
                return hdesk
        except Exception:
            pass
        return None

    def find_game_hwnd(self) -> Tuple[Optional[int], Optional[str]]:
        """
        Locates the exact HWND and title for the target game window.
        Returns: (hwnd, title) or (None, None)
        """
        if not user32:
            return None, None

        hdesk = self._attach_input_desktop()
        found: List[Tuple[int, int, str]] = []

        def enum_callback(hwnd, lparam):
            if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
                return True

            # Check window class name
            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls_buf, 256)
            cls_name = cls_buf.value.strip()

            # Check window title
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            title = buf.value.strip()

            # Priority 1: Window class is POEWindowClass (the official PoE/PoE2 engine class)
            if cls_name == self.target_class:
                found.append((1, hwnd, title or "Path of Exile 2"))
                return False

            # Priority 2: Exact title match
            if title in self.target_titles:
                found.append((2, hwnd, title))
                return False

            # Priority 3: Title substring match without blacklisted substrings
            low_t = title.lower()
            if any(t.lower() in low_t for t in self.target_titles):
                if not any(b in low_t for b in self.ignored_substrings):
                    found.append((3, hwnd, title))

            return True

        CB_TYPE = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        cb_func = CB_TYPE(enum_callback)

        if hdesk:
            user32.EnumDesktopWindows(hdesk, cb_func, 0)
            try:
                user32.CloseDesktop(hdesk)
            except Exception:
                pass

        if not found:
            user32.EnumWindows(cb_func, 0)

        if found:
            found.sort(key=lambda x: x[0])
            return found[0][1], found[0][2]

        return None, None

    def find_game_window(self):
        """Finds the pygetwindow object matching the game title."""
        if not gw:
            return None

        all_wins = gw.getAllWindows()

        # Priority 1: Exact matches
        for title in self.target_titles:
            for win in all_wins:
                if win.title and win.title.strip().lower() == title.lower():
                    return win

        # Priority 2: Filtered substring matches
        for win in all_wins:
            if not win.title:
                continue
            low_t = win.title.lower()
            if any(t.lower() in low_t for t in self.target_titles):
                if not any(b in low_t for b in self.ignored_substrings):
                    return win

        return None

    def find_window_by_process_name(self, process_names: Optional[List[str]] = None) -> Optional[int]:
        """Finds top-level HWND belonging to target process (e.g. PathOfExileSteam.exe)."""
        hwnd, _ = self.find_game_hwnd()
        if hwnd:
            return hwnd

        try:
            import psutil
        except ImportError:
            return None

        target_names = [n.lower() for n in (process_names or ["pathofexilesteam.exe", "pathofexile.exe", "pathofexile"])]
        target_pids = set()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info.get('name')
                if name and any(t in name.lower() for t in target_names):
                    target_pids.add(proc.info['pid'])
            except Exception:
                pass

        if not target_pids or not user32:
            return None

        found_hwnd = None

        def enum_cb(hwnd, lparam):
            nonlocal found_hwnd
            if user32.IsWindowVisible(hwnd):
                lp_pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(lp_pid))
                if lp_pid.value in target_pids:
                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w > 400 and h > 300:
                        found_hwnd = hwnd
                        return False
            return True

        CB_TYPE = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(CB_TYPE(enum_cb), 0)
        return found_hwnd

    def get_game_window_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns (left, top, right, bottom) screen desktop coordinates of the game window,
        or None if not located.
        """
        if not user32:
            return None
        hwnd, _ = self.find_game_hwnd()
        if not hwnd:
            hwnd = self.find_window_by_process_name()
        if not hwnd:
            return None

        rect = wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w > 100 and h > 100:
                return (rect.left, rect.top, rect.right, rect.bottom)
        return None

    def bring_hwnd_to_front(self, hwnd: int) -> bool:
        """Brings the given HWND to the foreground using advanced Win32 API sequence."""
        if not user32 or not hwnd or not kernel32:
            return False
        try:
            self._attach_input_desktop()

            # 1. Unlock Windows foreground lock timeout
            user32.SystemParametersInfoW(0x2001, 0, 0, 0x0003)  # SPI_SETFOREGROUNDLOCKTIMEOUT
            user32.AllowSetForegroundWindow(0xFFFFFFFF)  # ASFW_ANY

            # 2. Attach thread inputs to allow SetForegroundWindow
            fg_hwnd = user32.GetForegroundWindow()
            cur_tid = kernel32.GetCurrentThreadId()
            fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
            target_tid = user32.GetWindowThreadProcessId(hwnd, None)

            if fg_tid != 0 and fg_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, fg_tid, True)
            if target_tid != 0 and target_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, target_tid, True)

            # 3. Restore and bring to top
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.BringWindowToTop(hwnd)

            # 4. Alt-key foreground unlock trick
            user32.keybd_event(0x12, 0, 0, 0)
            res = user32.SetForegroundWindow(hwnd)
            user32.keybd_event(0x12, 0, 2, 0)
            user32.SetFocus(hwnd)

            # 5. Clean up thread inputs
            if fg_tid != 0 and fg_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, fg_tid, False)
            if target_tid != 0 and target_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, target_tid, False)

            time.sleep(0.05)
            curr_fg = user32.GetForegroundWindow()
            success = (curr_fg == hwnd) or bool(res)

            if success:
                print(f"[WINDOW FOCUS ENGINE] Successfully activated game window (HWND: {hwnd})!")
            return success
        except Exception as e:
            print(f"[WINDOW FOCUS ENGINE] Win32 focus warning: {e}")
            return False

    def focus_game_window(self, monitor_idx: Optional[int] = None) -> bool:
        """
        Brings the target game window to the foreground and focuses it.

        :param monitor_idx: Optional monitor index to target if window title match fails.
        :return: True if focused successfully, False otherwise.
        """
        # 1. Primary attempt: Locate HWND by POEWindowClass / exact title
        game_hwnd, game_title = self.find_game_hwnd()
        if game_hwnd and self.bring_hwnd_to_front(game_hwnd):
            return True

        # 2. Secondary attempt: Locate by process name (PathOfExileSteam.exe)
        poe_hwnd = self.find_window_by_process_name()
        if poe_hwnd and self.bring_hwnd_to_front(poe_hwnd):
            return True

        # 3. Third attempt: Find by pygetwindow
        win = self.find_game_window()
        if win:
            print(f"[WINDOW FOCUS ENGINE] Found game window: '{win.title}'")
            try:
                if hasattr(win, "_hWnd") and win._hWnd:
                    if self.bring_hwnd_to_front(win._hWnd):
                        return True
                if hasattr(win, "activate"):
                    win.activate()
                    time.sleep(0.1)
                return True
            except Exception as e:
                print(f"[WINDOW FOCUS ENGINE] Focus warning: {e}")

        # 4. Fourth attempt: Focus by monitor center click if monitor_idx specified
        if monitor_idx is not None:
            return self.focus_by_monitor(monitor_idx)

        print(f"[WINDOW FOCUS ENGINE] Notice: Please click into Path Of Exile 2 window on your game monitor to ensure WASD inputs are received.")
        return False

    def focus_by_monitor(self, monitor_idx: int) -> bool:
        """Activates focus on the target monitor by clicking the game window area."""
        try:
            from .screen_capturer import ScreenCapturer
            monitors = ScreenCapturer.list_monitors()
            target_mon = next((m for m in monitors if m["index"] == monitor_idx), None)
            if target_mon:
                cx = target_mon["left"] + target_mon["width"] // 2
                cy = target_mon["top"] + target_mon["height"] // 2
                import pyautogui
                # Click center of game screen to ensure game gets focus
                pyautogui.click(cx, cy)
                print(f"[WINDOW FOCUS ENGINE] Activated game window via center click on Monitor {monitor_idx} ({cx}, {cy}).")
                return True
        except Exception as e:
            print(f"[WINDOW FOCUS ENGINE] Focus by monitor error: {e}")
        return False

    def is_game_focused(self) -> bool:
        """Checks if the target game window currently has input focus."""
        if not user32:
            return True
        try:
            self._attach_input_desktop()
            fg_hwnd = user32.GetForegroundWindow()
            if not fg_hwnd:
                return False

            # Check class name of current foreground window
            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(fg_hwnd, cls_buf, 256)
            cls_name = cls_buf.value.strip()
            if cls_name == self.target_class:
                return True

            # Check title of current foreground window
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(fg_hwnd, buf, 512)
            fg_title = buf.value.strip()
            if fg_title in self.target_titles:
                return True

            # Check matching HWND
            game_hwnd, _ = self.find_game_hwnd()
            if game_hwnd and fg_hwnd == game_hwnd:
                return True

            return False
        except Exception:
            return False

    def ensure_focused(self, monitor_idx: Optional[int] = None):
        """Ensures game window is focused; if focus was lost, immediately re-focuses it."""
        if not self.is_game_focused():
            self.focus_game_window(monitor_idx=monitor_idx)

    def get_game_center(self) -> Optional[Tuple[int, int]]:
        """Returns (x, y) center pixel coordinates of the target game window."""
        hwnd, _ = self.find_game_hwnd()
        if hwnd and user32:
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w > 0 and h > 0:
                cx = rect.left + w // 2
                cy = rect.top + h // 2
                return (cx, cy)

        win = self.find_game_window()
        if win and hasattr(win, "left") and hasattr(win, "width"):
            cx = win.left + win.width // 2
            cy = win.top + win.height // 2
            return (cx, cy)
        return None


# Global singleton instance
window_focuser = WindowFocusHandler()
