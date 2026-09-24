"""
Minimap loot detection, ground loot label location, Z-key label visibility, and loot pickup routines.
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


class LootCollectorMixin:
    """Minimap loot detection, ground loot label location, Z-key label visibility, and loot pickup routines."""

    def check_minimap_loot_drop(self, save_debug: bool = False) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Captures the current screen, extracts the minimap ROI, and detects
        if any loot filter drop icons (stars, diamonds, circles) are present.
        Returns (count, list_of_icons).
        If save_debug is True and icons are found, saves annotated screenshot.
        """
        try:
            capt = self._get_capturer()
            screen = capt.capture()
            if screen is None or screen.size == 0:
                return 0, []
            if not hasattr(self, "minimap_extractor") or self.minimap_extractor is None:
                self.minimap_extractor = MinimapExtractor()
            mm_crop = self.minimap_extractor.extract_roi(screen)
            if mm_crop is None or mm_crop.size == 0:
                return 0, []
            if hasattr(self, "loot_detector") and self.loot_detector:
                count, icons = self.loot_detector.detect_minimap_loot_icons(mm_crop)
                if count > 0 and save_debug:
                    mm_path, raw_path = self.loot_detector.save_minimap_early_exit_debug(
                        full_screen=screen,
                        minimap_crop=mm_crop,
                        icons=icons,
                        output_dir=self.loot_debug_dir or "loot_debug",
                    )
                    _log(f"  [EARLY EXIT SCREENSHOT] Saved detection debug image: '{mm_path}'")
                return count, icons
        except Exception as e:
            _log(f"[EARLY EXIT] Warning: Minimap loot check failed: {e}")
        return 0, []


    def locate_loot(
        self,
        exclude_positions: Optional[List[Tuple[int, int]]] = None,
        screen: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates high-value loot on screen using LootDetector (white box / red text, purple uniques, etc.)
        with fallback to template matching.
        Returns desktop absolute coordinates (X, Y) of the loot item center, or None if not found.
        """
        capt = self._get_capturer()
        self._last_clicked_loot_item = None
        if screen is None:
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

        # 1. First attempt: Use LootDetector (rules from loot_filter.json)
        if hasattr(self, "loot_detector") and self.loot_detector:
            detected_items = self.loot_detector.detect_loot(screen)
            self._last_detected_loot_items = detected_items
            self._last_loot_screen = screen
            for item in detected_items:
                desktop_x = mon_left + item.center_x
                desktop_y = mon_top + item.center_y

                # Check if this item was already clicked recently in this pickup cycle
                if exclude_positions:
                    too_close = False
                    for ex_x, ex_y in exclude_positions:
                        if math.hypot(desktop_x - ex_x, desktop_y - ex_y) < 28.0:
                            too_close = True
                            break
                    if too_close:
                        continue

                # Save debug screenshot and zoomed crop if enabled
                if self.save_loot_debug_screenshots or getattr(self.loot_detector, "save_debug_screenshots", False):
                    try:
                        self.loot_detector.save_debug_screenshot(
                            screen,
                            item,
                            all_items=detected_items,
                            output_dir=self.loot_debug_dir or getattr(self.loot_detector, "debug_dir", "loot_debug"),
                        )
                    except Exception as e:
                        _log(f"  [LOOT DEBUG] Failed to save loot debug screenshot: {e}")

                self._last_clicked_loot_item = item
                _log(f"  [LOOT MATCH] Found [P{item.priority}] {item.rule_name} (conf={item.confidence:.2f}, {item.w}x{item.h}) at screen ({desktop_x}, {desktop_y})")
                return desktop_x, desktop_y

        # 2. Fallback attempt: Template match with ui/loot1.png
        if self.loot1_img is None:
            self._load_loot_template()

        if self.loot1_img is not None:
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
                if exclude_positions:
                    for ex_x, ex_y in exclude_positions:
                        if math.hypot(desktop_x - ex_x, desktop_y - ex_y) < 28.0:
                            return None

                tmpl_item = LootItem(
                    x=int(best_loc[0]),
                    y=int(best_loc[1]),
                    w=int(tw * best_scale),
                    h=int(th * best_scale),
                    center_x=cx,
                    center_y=cy,
                    rule_id="custom_template_loot1",
                    rule_name="Template Matcher (ui/loot1.png)",
                    priority=99,
                    confidence=best_val,
                )
                self._last_clicked_loot_item = tmpl_item

                # Save debug screenshot for fallback template loot if enabled
                if (self.save_loot_debug_screenshots or getattr(self.loot_detector, "save_debug_screenshots", False)) and hasattr(self, "loot_detector") and self.loot_detector:
                    try:
                        self.loot_detector.save_debug_screenshot(
                            screen,
                            tmpl_item,
                            output_dir=self.loot_debug_dir or getattr(self.loot_detector, "debug_dir", "loot_debug"),
                        )
                    except Exception as e:
                        _log(f"  [LOOT DEBUG] Failed to save fallback template debug screenshot: {e}")

                _log(f"  [LOOT MATCH] Found fallback template loot1 (conf={best_val:.2f}, scale={best_scale:.2f}) at screen ({desktop_x}, {desktop_y})")
                return desktop_x, desktop_y

        return None


    def _press_z_key(self):
        """Sends a clean Z keypress with 40ms duration to toggle ground item visibility."""
        if pydirectinput:
            try:
                pydirectinput.keyDown("z")
                time.sleep(0.04)
                pydirectinput.keyUp("z")
            except Exception:
                try:
                    pydirectinput.press("z")
                except Exception:
                    pass
        elif pyautogui:
            try:
                pyautogui.press("z")
            except Exception:
                pass


    def ensure_loot_labels_visible(self, force: bool = False):
        """
        Ensures ground item/entity labels are visible (unhidden) so SIM templates,
        encounter banners, and Hideout STASH labels can be detected on screen.
        If loot labels were previously hidden (or force=True), presses 'Z' to unhide them.
        """
        if not getattr(self, "loot_z_toggle_enabled", True):
            return
        if force or getattr(self, "_loot_labels_hidden", False):
            _log("  [LABELS] Unhiding labels / item highlights (Pressing [Z])...")
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            self._press_z_key()
            self._loot_labels_hidden = False
            time.sleep(getattr(self, "loot_z_toggle_delay_seconds", 0.3))


    def hide_loot_labels(self):
        """
        Hides ground item/entity labels after SIMs and encounter banners have been clicked
        so the screen remains clean during combat, movement, and orbiting.
        If loot labels are currently visible, presses 'Z' to hide them.
        """
        if not getattr(self, "loot_z_toggle_enabled", True):
            return
        if not getattr(self, "_loot_labels_hidden", False):
            _log("  [LOOT] Hiding ground labels after Sims / Encounter banner interaction (Pressing [Z])...")
            self._press_z_key()
            self._loot_labels_hidden = True
            time.sleep(0.1)


    def collect_loot(self, max_pickups: Optional[int] = None, approach_wait: Optional[float] = None) -> int:
        """
        Scans screen for high-value loot matching active loot filter rules.
        Clicks left mouse button on each detected item and scans again.
        Toggles 'Z' key before looting (2x on first cycle, 3x on subsequent cycles) to collapse/unhide item labels.
        Toggles 'Z' key after looting to hide item labels.
        Provides detailed start-to-end timing and telemetry.
        Returns total number of loots clicked.
        """
        if getattr(self, "_is_collecting_loot", False):
            _log("  [LOOT] Concurrency guard: Loot collection already active. Skipping duplicate call.")
            return 0
        self._is_collecting_loot = True
        prior_interacting = self.is_interacting
        self.is_interacting = True
        self.stuck_counter = 0
        self.release_all_keys()
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        loot_start_time = time.time()
        start_timestr = time.strftime("%H:%M:%S", time.localtime(loot_start_time))
        z_enabled = getattr(self, "loot_z_toggle_enabled", True)
        z_delay = getattr(self, "loot_z_toggle_delay_seconds", 0.3)
        initial_hidden_state = getattr(self, "_loot_labels_hidden", False)
        clicked_positions: List[Tuple[int, int]] = []
        picked_count = 0
        pre_loot_duration = 0.0
        post_loot_duration = 0.0

        eff_app_wait = approach_wait if approach_wait is not None else getattr(self.loot_detector, "approach_wait_seconds", self.loot_approach_wait_seconds)

        _log("\n" + "=" * 80)
        _log(f"[LOOT SESSION START] Initiated at {start_timestr}")
        _log(f"  • Item labels state : {'Hidden (Subsequent Session)' if initial_hidden_state else 'Visible (Initial Session)'}")
        _log(f"  • Approach wait     : {eff_app_wait:.2f}s (default: {self.loot_approach_wait_seconds:.1f}s)")
        _log(f"  • Z-key realignment : {'Enabled' if z_enabled else 'Disabled'} (Delay: {z_delay*1000:.0f}ms)")
        _log("-" * 80)

        try:
            # Pre-loot Z-key sequence to collapse item labels closer
            if z_enabled and not stop_handler.is_stopped() and self.is_active:
                pre_z_start = time.time()
                if not initial_hidden_state:
                    # First looting cycle: Press Z (hide) -> wait 300ms -> Press Z (unhide & collapse closer)
                    _log("  [LOOT] Pre-loot sequence (Initial): Pressing [Z] -> wait 300ms -> [Z] to collapse item labels closer...")
                    self.status_message = "[LOOT] Re-aligning items (Z -> Z)..."
                    self._press_z_key()
                    time.sleep(z_delay)
                    self._press_z_key()
                    time.sleep(z_delay)
                else:
                    # Subsequent looting cycle: Press Z (unhide) -> wait 300ms -> Press Z (hide) -> wait 300ms -> Press Z (unhide & collapse)
                    _log("  [LOOT] Pre-loot sequence (Subsequent): Pressing [Z] (unhide) -> [Z] (hide) -> [Z] (unhide) to collapse labels closer...")
                    self.status_message = "[LOOT] Unhiding & re-aligning items (Z -> Z -> Z)..."
                    self._press_z_key()
                    time.sleep(z_delay)
                    self._press_z_key()
                    time.sleep(z_delay)
                    self._press_z_key()
                    time.sleep(z_delay)
                pre_loot_duration = time.time() - pre_z_start

            limit = max_pickups if max_pickups is not None else getattr(self.loot_detector, "max_pickups", self.max_loot_pickups)
            _log("[AUTOPILOT] >>> Scanning screen for HIGH-VALUE LOOT (White Box/Red Text, Purple Uniques)...")
            self.status_message = "[LOOT] Scanning screen for loot..."

            while picked_count < limit:
                if stop_handler.is_stopped() or not self.is_active:
                    break
                try:
                    loot_pos = self.locate_loot(exclude_positions=clicked_positions)
                except TypeError:
                    loot_pos = self.locate_loot()
                if loot_pos is None:
                    if picked_count == 0:
                        _log("  [LOOT] No high-value loot detected on screen.")
                    else:
                        _log(f"  [LOOT] Finished picking up {picked_count} loot item(s). None remaining.")
                    break

                # Save clean full-screen screenshot before picking up any loot item if enabled (reusing first scan capture)
                if picked_count == 0 and (self.save_pre_loot_screenshot or getattr(self.loot_detector, "save_pre_loot_screenshot", False)):
                    try:
                        cached_screen = getattr(self, "_last_loot_screen", None)
                        cached_items = getattr(self, "_last_detected_loot_items", None)
                        if cached_screen is None:
                            capt = self._get_capturer()
                            cached_screen = capt.capture()
                            if cached_screen is not None and cached_screen.size > 0 and hasattr(self, "loot_detector") and self.loot_detector:
                                cached_items = self.loot_detector.detect_loot(cached_screen)
                        if cached_screen is not None and cached_screen.size > 0 and hasattr(self, "loot_detector") and self.loot_detector:
                            out_p = self.loot_detector.save_pre_pickup_screenshot(
                                cached_screen,
                                all_items=cached_items,
                                output_dir=self.loot_debug_dir or getattr(self.loot_detector, "debug_dir", "loot_debug"),
                            )
                            _log(f"  [LOOT SCREENSHOT] Captured full-screen image before pickup: {out_p}")
                    except Exception as e:
                        _log(f"  [LOOT SCREENSHOT] Warning: Failed to save pre-pickup screenshot: {e}")

                lx, ly = self.move_mouse_inside_game(loot_pos[0], loot_pos[1])
                picked_count += 1
                clicked_positions.append((lx, ly))
                _log(f"  [LOOT #{picked_count}] Targeting loot item at ({lx}, {ly}). Clicking left mouse button...")
                self.status_message = f"[LOOT] Picking #{picked_count} at ({lx}, {ly})"

                # Record picked loot item in run session
                self._record_loot_picked(
                    getattr(self, "_last_clicked_loot_item", None),
                    pos=(lx, ly),
                    room_key=getattr(self, "current_room_key", None),
                )

                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.08)
                    pydirectinput.mouseUp(button="left")

                eff_app_wait = approach_wait if approach_wait is not None else getattr(self.loot_detector, "approach_wait_seconds", self.loot_approach_wait_seconds)
                if eff_app_wait > 0:
                    self._wait_for_approach(eff_app_wait, reason=f"LOOT #{picked_count}")

                pick_delay = getattr(self.loot_detector, "pickup_delay_seconds", self.loot_pickup_wait_seconds)
                self._trigger_persistent_combat_if_due()
                time.sleep(pick_delay)
                self._trigger_persistent_combat_if_due()

            self.status_message = f"Loot Check Complete ({picked_count} picked). Resuming route..."
            return picked_count
        finally:
            # Post-loot Z-key sequence to hide item labels on ground
            if z_enabled and not (stop_handler.is_stopped() or not self.is_active):
                post_z_start = time.time()
                _log("  [LOOT] Post-loot cleanup: Pressing [Z] to hide item labels on ground...")
                self._press_z_key()
                self._loot_labels_hidden = True
                time.sleep(0.1)
                post_loot_duration = time.time() - post_z_start

            total_duration = time.time() - loot_start_time
            end_timestr = time.strftime("%H:%M:%S", time.localtime())
            _log("-" * 80)
            _log(f"[LOOT SESSION FINISHED] Completed at {end_timestr}")
            _log(f"  • Items Collected    : {picked_count}")
            _log(f"  • Total Duration     : {total_duration:.2f}s (Start to End)")
            if z_enabled:
                scan_pickup_duration = max(0.0, total_duration - pre_loot_duration - post_loot_duration)
                _log(f"  • Timing Breakdown   : Pre-loot Z alignment: {pre_loot_duration:.2f}s | Scan & Pickup: {scan_pickup_duration:.2f}s | Post-loot Z hide: {post_loot_duration:.2f}s")
            _log("=" * 80 + "\n")

            self._is_collecting_loot = False
            self.is_interacting = prior_interacting
            now = time.time()
            self.last_progress_pos = self.latest_pos
            self.last_progress_time = now
            self.last_known_time = now
            self.stuck_counter = 0

