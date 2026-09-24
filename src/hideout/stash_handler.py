"""
Stash detection, clicking, inventory depositing, and window closing in hideout.
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


class HideoutStashMixin:
    """Stash detection, clicking, inventory depositing, and window closing in hideout."""

    def locate_stash(
        self,
        threshold: Optional[float] = None,
        screen: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the Stash in the hideout using multi-scale template matching.
        Prioritizes the 'STASH' label banner, falling back to the chest template.
        Returns desktop absolute (x, y) coordinates of the click target, or None if not found.
        """
        if self.stash_label_tpl is None and self.stash_full_tpl is None:
            self._load_stash_and_inventory_templates()
        if self.stash_label_tpl is None and self.stash_full_tpl is None:
            return None

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return None

        eff_thresh = threshold if threshold is not None else self.stash_match_threshold

        mon_left = 0
        mon_top = 0
        capt = self._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        sh, sw = screen.shape[:2]
        scales = [1.0, 0.95, 1.05, 0.90, 1.10]

        # 1. Search for 'STASH' text label banner
        if self.stash_label_tpl is not None:
            lh, lw = self.stash_label_tpl.shape[:2]
            best_lbl_val = -1.0
            best_lbl_loc = None
            best_lbl_scale = 1.0

            for s in scales:
                tw = int(lw * s)
                th = int(lh * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.stash_label_tpl if s == 1.0 else cv2.resize(self.stash_label_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_lbl_val:
                    best_lbl_val = max_val
                    best_lbl_loc = max_loc
                    best_lbl_scale = s

            if best_lbl_val >= eff_thresh and best_lbl_loc is not None:
                cur_w = int(lw * best_lbl_scale)
                cur_h = int(lh * best_lbl_scale)
                cx = int(best_lbl_loc[0] + cur_w // 2)
                cy = int(best_lbl_loc[1] + cur_h // 2)
                _log(f"  [STASH MATCH] Found STASH label (conf={best_lbl_val:.3f} >= {eff_thresh:.2f}, scale={best_lbl_scale:.2f}) at screen ({cx}, {cy})")
                return (cx + mon_left, cy + mon_top)

        # 2. Search for Stash full chest
        if self.stash_full_tpl is not None:
            ch, cw = self.stash_full_tpl.shape[:2]
            best_chest_val = -1.0
            best_chest_loc = None
            best_chest_scale = 1.0
            chest_thresh = max(0.50, eff_thresh - 0.05)

            for s in scales:
                tw = int(cw * s)
                th = int(ch * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.stash_full_tpl if s == 1.0 else cv2.resize(self.stash_full_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_chest_val:
                    best_chest_val = max_val
                    best_chest_loc = max_loc
                    best_chest_scale = s

            if best_chest_val >= chest_thresh and best_chest_loc is not None:
                cur_w = int(cw * best_chest_scale)
                cur_h = int(ch * best_chest_scale)
                cx = int(best_chest_loc[0] + cur_w // 2)
                cy = int(best_chest_loc[1] + int(cur_h * 0.65))
                _log(f"  [STASH MATCH] Found Stash chest (conf={best_chest_val:.3f} >= {chest_thresh:.2f}, scale={best_chest_scale:.2f}) at screen ({cx}, {cy})")
                return (cx + mon_left, cy + mon_top)

        return None


    def click_stash(
        self,
        search_attempts: int = 10,
        timeout: float = 15.0,
        verify_inventory: bool = True,
    ) -> bool:
        """
        Waits for hideout to load (confirmed via hideout_layout minimap), locates the Stash,
        clicks it, confirms that both Stash and Inventory windows are open, and saves an inventory screenshot.
        """
        self.release_all_keys()

        # 1. Wait to confirm arrival in hideout
        self.status_message = "Waiting for Hideout Arrival..."
        _log(f"\n[HIDEOUT] Confirming character is in hideout (allowing up to {timeout:.1f}s for loading screen)...")
        self.in_hideout = True
        self.disable_persistent_combat()
        self.release_all_keys()
        hideout_start = time.time()
        in_hideout = False
        while (time.time() - hideout_start) < timeout:
            if stop_handler.is_stopped():
                return False
            if self.is_in_hideout():
                in_hideout = True
                break
            time.sleep(0.5)

        if in_hideout:
            _log("  [HIDEOUT] Arrival confirmed! Searching for STASH...")
        else:
            _log("  [HIDEOUT WARNING] Hideout minimap match timed out. Proceeding to search for STASH anyway...")

        # Ensure loot / object labels are unhidden in Hideout so STASH banner is visible
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.ensure_loot_labels_visible()

        # Check if Inventory is ALREADY open before searching or clicking
        if self.is_inventory_open():
            _log("  [STASH] Stash and Inventory windows are ALREADY OPEN. Skipping click.")
            self.status_message = "Stash & Inventory Already Open!"
            self.save_inventory_screenshot()
            return True

        self.status_message = "Locating Stash..."
        stash_start = time.time()
        stash_pos = None
        attempted_z_fallback = False

        # 2. Search loop for STASH
        while (time.time() - stash_start) < 8.0:
            if stop_handler.is_stopped():
                return False
            stash_pos = self.locate_stash()
            if stash_pos is not None:
                break
            # If STASH is not found after 1.5 seconds, force-press 'Z' once in case labels were hidden out-of-sync
            if not attempted_z_fallback and (time.time() - stash_start) >= 1.5:
                attempted_z_fallback = True
                _log("  [STASH] Stash not immediately detected. Toggling [Z] to ensure object labels are visible...")
                self.ensure_loot_labels_visible(force=True)
            time.sleep(0.4)

        if stash_pos is None:
            _log(f"  [WARNING] STASH not detected in hideout.")
            return False

        # 3. Click the Stash cleanly
        _log(f"  [ACTION] Left-clicking STASH at screen ({stash_pos[0]}, {stash_pos[1]})...")
        self.status_message = "Clicking STASH..."
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        rx, ry = self.move_mouse_inside_game(stash_pos[0], stash_pos[1])
        time.sleep(0.10)
        if pydirectinput:
            pydirectinput.mouseDown(button="left")
            time.sleep(0.08)
            pydirectinput.mouseUp(button="left")
        time.sleep(0.3)

        # 4. Wait for character to walk to stash and open inventory window
        if verify_inventory:
            self.status_message = "Verifying Stash & Inventory Open..."
            open_confirmed = False
            last_screen = None
            inv_start = time.time()
            # Allow up to 10.0s for character to walk across hideout and open stash
            while (time.time() - inv_start) < 10.0:
                if stop_handler.is_stopped():
                    return False
                capt = self._get_capturer()
                try:
                    last_screen = capt.capture()
                except Exception:
                    last_screen = None
                if self.is_inventory_open(screen=last_screen):
                    open_confirmed = True
                    break
                time.sleep(0.15)

            # Retry click once if not opened: check fresh screen FIRST before re-clicking!
            if not open_confirmed and not stop_handler.is_stopped():
                capt = self._get_capturer()
                fresh_screen = None
                try:
                    fresh_screen = capt.capture() if capt else None
                except Exception:
                    pass
                if fresh_screen is not None and self.is_inventory_open(screen=fresh_screen):
                    open_confirmed = True
                    last_screen = fresh_screen
                else:
                    _log("  [STASH RETRY] Inventory not open yet. Re-checking and re-clicking STASH...")
                    recheck_pos = self.locate_stash() or stash_pos
                    window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
                    rx, ry = self.move_mouse_inside_game(recheck_pos[0], recheck_pos[1])
                    time.sleep(0.10)
                    if pydirectinput:
                        pydirectinput.mouseDown(button="left")
                        time.sleep(0.08)
                        pydirectinput.mouseUp(button="left")
                    time.sleep(0.5)
                    inv_retry_start = time.time()
                    while (time.time() - inv_retry_start) < 6.0:
                        if stop_handler.is_stopped():
                            return False
                        capt = self._get_capturer()
                        try:
                            last_screen = capt.capture()
                        except Exception:
                            last_screen = None
                        if self.is_inventory_open(screen=last_screen):
                            open_confirmed = True
                            break
                        time.sleep(0.15)

            if open_confirmed:
                _log("  [STASH SUCCESS] Stash and Inventory windows confirmed OPEN in hideout!")
                self.status_message = "Stash & Inventory Open!"
                self.save_inventory_screenshot(screen=last_screen)
                return True
            else:
                _log("  [STASH WARNING] Clicked STASH, but Inventory window open confirmation timed out.")
                return False

        return True


    def stash_inventory_items(
        self,
        screen: Optional[np.ndarray] = None,
        exclude_last_columns: int = 3,
        dry_run: bool = False,
        close_after: bool = False,
    ) -> int:
        """
        Scans the open player inventory grid (5 rows x 12 columns).
        For every non-empty cell in columns 0..(12 - exclude_last_columns - 1),
        presses CTRL + Left Click to quick-deposit items into the active Stash tab.
        Uses dynamic row center detection and resolution scaling to support both Screen 2 (1080p)
        and Screen 1 (1800p OLED) seamlessly.
        If dry_run=True, scans and reports detected items without sending clicks or keystrokes.
        If close_after=True, presses Escape after depositing to close all open windows.
        """
        self.release_all_keys()
        self.disable_persistent_combat()
        self.status_message = "Stashing Inventory Items (Ctrl+Click)..."
        _log(f"\n[STASH DEPOSIT] Scanning inventory grid for non-empty items (excluding last {exclude_last_columns} columns)...")

        # Auto-detect if game window is currently on Monitor 1 or Monitor 2
        try:
            bounds = window_focuser.get_game_window_bounds()
            if bounds:
                center_x = (bounds[0] + bounds[2]) // 2
                capt = self._get_capturer()
                if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
                    for idx in [1, 2]:
                        if idx < len(capt._sct.monitors):
                            m = capt._sct.monitors[idx]
                            if m["left"] <= center_x < m["left"] + m["width"]:
                                if self.monitor_idx != idx:
                                    _log(f"  [MONITOR DETECT] Game window detected on Monitor {idx} ({m['width']}x{m['height']}). Adapting monitor_idx.")
                                    self.monitor_idx = idx
                                    if capt.monitor_idx != idx:
                                        capt.monitor_idx = idx
                                break
        except Exception:
            pass

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            _log("  [WARNING] Unable to capture screen for inventory stashing.")
            return 0

        mon_left = 0
        mon_top = 0
        capt = self._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        inv_info = self.locate_inventory_window(screen=screen)
        if inv_info is None:
            _log("  [WARNING] Inventory window not detected on screen.")
            return 0

        row_centers = inv_info["row_centers"]
        col_centers = inv_info["col_centers"]
        scale = inv_info["scale"]
        row_method = inv_info["row_method"]
        _log(f"  [STASH DEPOSIT] Inventory window located (conf={inv_info['confidence']:.3f}, scale={scale:.2f}, rows={row_method}).")

        sh, sw = screen.shape[:2]
        max_col = max(0, min(12, 12 - exclude_last_columns))
        half_box = max(8, int(14 * scale))
        candidates = []

        for r, cy in enumerate(row_centers):
            for c in range(max_col):
                cx = col_centers[c]
                cx_i = int(round(cx))
                cy_i = int(round(cy))

                # Bounds check inside screen
                if cy_i - half_box < 0 or cy_i + half_box > sh or cx_i - half_box < 0 or cx_i + half_box > sw:
                    continue

                patch = screen[cy_i - half_box : cy_i + half_box, cx_i - half_box : cx_i + half_box]
                is_non_empty = bool(patch.mean() > 16.0 or patch.max() > 40)
                if is_non_empty:
                    desktop_x = cx_i + mon_left
                    desktop_y = cy_i + mon_top
                    candidates.append((r, c, desktop_x, desktop_y))

        _log(f"  [STASH DEPOSIT] Found {len(candidates)} non-empty item cell(s) in columns 1..{max_col} (rows detected via {row_method}).")
        if not candidates:
            _log("  [STASH DEPOSIT] No items found to deposit. Stashing complete.")
            return 0

        if dry_run:
            _log(f"  [STASH DRY-RUN] Dry run enabled. Found {len(candidates)} items (no clicks performed):")
            for r, c, dx, dy in candidates:
                _log(f"    - Cell (Row {r+1}, Col {c+1}) at screen ({dx}, {dy})")
            return len(candidates)

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        stashed_count = 0

        try:
            if pydirectinput:
                pydirectinput.keyDown("ctrl")
                time.sleep(0.06)

            for r, c, dx, dy in candidates:
                if stop_handler.is_stopped():
                    break
                self.move_mouse_inside_game(dx, dy)
                time.sleep(0.04)
                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.06)
                stashed_count += 1
                self.status_message = f"Stashed item {stashed_count}/{len(candidates)} (R{r+1}C{c+1})"
        finally:
            if pydirectinput:
                pydirectinput.keyUp("ctrl")
                time.sleep(0.08)

        _log(f"  [STASH DEPOSIT] Successfully deposited {stashed_count} item(s) into Stash!")
        self.status_message = f"Deposited {stashed_count} items to Stash!"

        if close_after and not dry_run:
            self.close_all_hideout_windows()

        return stashed_count


    def close_all_hideout_windows(
        self,
        wait_seconds: float = 0.35,
        verify_close: bool = True,
        max_attempts: int = 2,
    ) -> bool:
        """
        Presses the Escape key to close all open game windows (Stash, Inventory, Atlas, etc.).
        Verifies closure via is_inventory_open() and retries if necessary.
        """
        self.status_message = "Closing windows (Escape)..."
        _log("\n[ESCAPE] Closing all open windows via Escape key...")
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        for attempt in range(1, max_attempts + 1):
            if pydirectinput:
                pydirectinput.keyDown("escape")
                time.sleep(0.08)
                pydirectinput.keyUp("escape")
                time.sleep(wait_seconds)

            if not verify_close:
                return True

            if not self.is_inventory_open():
                _log("  [ESCAPE] Confirmed all windows closed successfully.")
                self.status_message = "Windows Closed (Escape)"
                return True

            _log(f"  [ESCAPE] Windows still detected open after attempt {attempt}/{max_attempts}. Retrying...")
            time.sleep(0.15)

        _log("  [ESCAPE WARNING] Windows remained open after Escape attempts.")
        return False

