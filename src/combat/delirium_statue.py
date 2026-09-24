"""
Delirium mirror/statue detection, interaction, debug screenshot saving, exit portal clicking, and user key confirmation.
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


class DeliriumStatueMixin:
    """Delirium mirror/statue detection, interaction, debug screenshot saving, exit portal clicking, and user key confirmation."""

    def locate_portal(
        self,
        threshold: Optional[float] = None,
        screen: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the exit portal on the main game screen using masked template matching with portal.png.
        Returns desktop absolute coordinates (X, Y) of the portal center, or None if not found / below threshold.
        """
        if self.portal_template_img is None:
            self._load_delirium_and_portal_templates()

        if self.portal_template_img is None or self.portal_mask is None:
            return None

        eff_threshold = threshold if threshold is not None else getattr(self, "portal_match_threshold", 0.70)

        capt = self._get_capturer()
        if screen is None:
            try:
                screen = capt.capture()
            except Exception as e:
                _log(f"  [WARNING] Screen capture failed during portal search: {e}")
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

        try:
            res = cv2.matchTemplate(screen, self.portal_template_img, cv2.TM_SQDIFF_NORMED, mask=self.portal_mask)
            min_v, _, min_l, _ = cv2.minMaxLoc(res)
            conf = 1.0 - float(min_v)
            if conf >= eff_threshold and min_l is not None:
                th, tw = self.portal_template_img.shape[:2]
                cx = min_l[0] + tw // 2
                cy = min_l[1] + th // 2
                desktop_x = mon_left + cx
                desktop_y = mon_top + cy
                _log(f"  [PORTAL MATCH] Found exit portal (conf={conf:.3f} >= {eff_threshold:.2f}) at screen ({desktop_x}, {desktop_y})")
                return desktop_x, desktop_y
        except Exception as e:
            _log(f"  [PORTAL] Template matching error: {e}")

        return None


    def save_delirium_debug_screenshot(
        self,
        screen: np.ndarray,
        detected: bool,
        conf: float = 0.0,
        box: Optional[Tuple[int, int, int, int]] = None,
        click_pos: Optional[Tuple[int, int]] = None,
        template_name: str = "",
    ) -> Optional[str]:
        """
        Saves an annotated debug screenshot showing what was detected as the Statue of Delirium.
        Includes bounding box, click target crosshair, confidence score, and template name.
        """
        try:
            os.makedirs("debug_logs", exist_ok=True)
            vis = screen.copy()
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if detected and box is not None and click_pos is not None:
                bx, by, bw, bh = box
                cx, cy = click_pos
                # Draw bounding box around detected statue (neon green, 3px)
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)

                # Draw crosshair at click target (neon yellow/cyan)
                cv2.circle(vis, (cx, cy), 14, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(vis, (cx - 24, cy), (cx + 24, cy), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(vis, (cx, cy - 24), (cx, cy + 24), (0, 255, 255), 2, cv2.LINE_AA)

                # Overlay label banner
                label_text = f"STATUE OF DELIRIUM | conf={conf:.3f} | click=({cx},{cy}) | tmpl={template_name}"
                (tw_text, th_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tag_y = max(th_text + 10, by - 10)
                cv2.rectangle(vis, (bx, tag_y - th_text - 6), (bx + tw_text + 10, tag_y + 6), (0, 0, 0), -1)
                cv2.putText(vis, label_text, (bx + 5, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                filename = f"delirium_detected_{timestamp}_conf{int(conf * 100)}.png"
            else:
                label_text = f"STATUE OF DELIRIUM NOT DETECTED (best_conf={conf:.3f})"
                cv2.rectangle(vis, (20, 20), (620, 65), (0, 0, 0), -1)
                cv2.putText(vis, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                filename = f"delirium_not_found_{timestamp}.png"

            filepath = os.path.join("debug_logs", filename)
            cv2.imwrite(filepath, vis)
            _log(f"  [DELIRIUM SCREENSHOT] Saved detection preview to '{filepath}'")
            return filepath
        except Exception as e:
            _log(f"  [DELIRIUM SCREENSHOT] Warning: Failed to save debug screenshot: {e}")
            return None


    def locate_delirium_statue(
        self,
        threshold: Optional[float] = None,
        screen: Optional[np.ndarray] = None,
        save_debug: bool = True,
        portal_pos: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the Statue of Delirium on the game screen using dual-template masked matching
        (supporting both clean ground and heavy burning ground / ignite fire).
        Uses cv2.TM_SQDIFF_NORMED with inverse difference confidence (1.0 - min_sqdiff).
        Optimized with gameplay ROI, multi-scale early exit, and cached portal spatial constraint.
        Returns desktop coordinates (X, Y) to click the statue, or None if not detected.
        """
        if not self.delirium_templates:
            self._load_delirium_and_portal_templates()

        if not self.delirium_templates:
            return None

        eff_threshold = threshold if threshold is not None else getattr(self, "delirium_match_threshold", 0.60)

        capt = self._get_capturer()
        if screen is None:
            try:
                screen = capt.capture()
            except Exception as e:
                _log(f"  [WARNING] Screen capture failed during delirium statue search: {e}")
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

        best_conf = -1.0
        best_loc = None
        best_shape = None
        best_name = ""

        sh, sw = screen.shape[:2]

        # In Room 7, the Statue of Delirium is ALWAYS on the left side of the exit portal!
        # Always locate the portal fresh on the CURRENT screen so camera movement since orbit is accounted for!
        if portal_pos is None:
            portal_pos = self.locate_portal(screen=screen)

        if portal_pos is not None:
            portal_screen_x = portal_pos[0] - mon_left
            portal_screen_y = portal_pos[1] - mon_top

            # In Room 7, the Statue of Delirium is geometrically fixed in the arena relative to the exit portal:
            # - Horizontal click target (cx) is strictly between [portal_screen_x - 620, portal_screen_x - 410]
            # - Vertical click target (cy) is strictly between [portal_screen_y - 250, portal_screen_y + 80]
            # - Geometric fallback center: (portal_screen_x - 515, portal_screen_y - 85)
            min_click_cx = max(0, portal_screen_x - 620)
            max_click_cx = max(0, portal_screen_x - 410)
            min_click_cy = max(0, portal_screen_y - 250)
            max_click_cy = min(sh, portal_screen_y + 80)
            fallback_click_pos = (portal_screen_x - 515, portal_screen_y - 85)
            _log(f"  [DELIRIUM SPATIAL] Exit portal at screen ({portal_pos[0]}, {portal_pos[1]}). Constraining statue click to X: [{min_click_cx}, {max_click_cx}], Y: [{min_click_cy}, {max_click_cy}].")
        else:
            min_click_cx = 200
            max_click_cx = int(sw * 0.48)
            min_click_cy = 100
            max_click_cy = min(sh - 200, 600)
            fallback_click_pos = None
            _log(f"  [DELIRIUM SPATIAL] Portal not detected; constraining statue click to left area (X < {max_click_cx}).")

        # Gameplay Region of Interest (ROI):
        roi_y1 = max(0, min_click_cy - 400)
        roi_y2 = min(sh - 100, max_click_cy + 220)
        roi_x1 = max(0, min_click_cx - 180)
        roi_x2 = min(sw, max_click_cx + 180)

        if roi_x2 > roi_x1 + 100 and roi_y2 > roi_y1 + 100:
            search_roi = screen[roi_y1:roi_y2, roi_x1:roi_x2]
            offset_x, offset_y = roi_x1, roi_y1
        else:
            search_roi = screen
            offset_x, offset_y = 0, 0

        # Prioritize 0.90 (perspective scale under fire/distance), 0.95, 1.00 (native clean)
        scales = [0.90, 0.95, 1.00]

        for item in self.delirium_templates:
            if len(item) == 3:
                t_name, tmpl, mask = item
            else:
                tmpl, mask = item
                t_name = "delirium"
            th, tw = tmpl.shape[:2]

            for scale in scales:
                sc_w = int(tw * scale)
                sc_h = int(th * scale)
                if sc_w > search_roi.shape[1] or sc_h > search_roi.shape[0] or sc_w < 50 or sc_h < 50:
                    continue
                try:
                    if scale == 1.0:
                        r_tmpl = tmpl
                        r_mask = mask
                    else:
                        r_tmpl = cv2.resize(tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                        r_mask = cv2.resize(mask, (sc_w, sc_h), interpolation=cv2.INTER_NEAREST)

                    res = cv2.matchTemplate(search_roi, r_tmpl, cv2.TM_SQDIFF_NORMED, mask=r_mask)

                    # Spatial constraint: statue click target (cx, cy) MUST be strictly within valid bounds
                    for yy in range(res.shape[0]):
                        c_y = offset_y + yy + int(sc_h * 0.65)
                        if not (min_click_cy <= c_y <= max_click_cy):
                            res[yy, :] = 1.0
                    for xx in range(res.shape[1]):
                        c_x = offset_x + xx + sc_w // 2
                        if not (min_click_cx <= c_x <= max_click_cx):
                            res[:, xx] = 1.0

                    # Reject matches located inside bottom UI globes / HUD if full screen was used
                    if offset_x == 0 and offset_y == 0:
                        res[max(0, sh - 260):, :260] = 1.0
                        res[max(0, sh - 260):, max(0, sw - 260):] = 1.0

                    min_v, _, min_l, _ = cv2.minMaxLoc(res)
                    conf = 1.0 - float(min_v)

                    if min_l is not None and conf > best_conf:
                        best_conf = conf
                        best_loc = (offset_x + min_l[0], offset_y + min_l[1])
                        best_shape = (sc_h, sc_w)
                        best_name = f"{t_name} (s={scale:.2f})"

                    # Early break if confident match found
                    if best_conf >= max(eff_threshold, 0.75):
                        break
                except Exception as e:
                    _log(f"  [DELIRIUM] Match error with template {t_name} @ scale {scale:.2f}: {e}")

            if best_conf >= max(eff_threshold, 0.75):
                break

        if best_conf >= eff_threshold and best_loc is not None and best_shape is not None:
            th, tw = best_shape
            cx = best_loc[0] + tw // 2
            cy = best_loc[1] + int(th * 0.65)
            desktop_x = mon_left + cx
            desktop_y = mon_top + cy
            _log(f"  [DELIRIUM MATCH] Found Statue of Delirium via '{best_name}' (conf={best_conf:.3f} >= {eff_threshold:.2f}) at screen ({desktop_x}, {desktop_y})")

            if save_debug:
                self.save_delirium_debug_screenshot(
                    screen=screen,
                    detected=True,
                    conf=best_conf,
                    box=(best_loc[0], best_loc[1], tw, th),
                    click_pos=(cx, cy),
                    template_name=best_name,
                )

            return desktop_x, desktop_y
        elif fallback_click_pos is not None:
            cx, cy = fallback_click_pos
            desktop_x = mon_left + cx
            desktop_y = mon_top + cy
            _log(f"  [DELIRIUM GEOMETRIC] Using portal-anchored geometric statue target at screen ({desktop_x}, {desktop_y}) (template conf={best_conf:.3f} < {eff_threshold:.2f})")
            if save_debug:
                self.save_delirium_debug_screenshot(
                    screen=screen,
                    detected=True,
                    conf=max(best_conf, 0.85),
                    box=(cx - 135, cy - 345, 271, 531),
                    click_pos=(cx, cy),
                    template_name="portal_geometric_anchor",
                )
            return desktop_x, desktop_y

        return None


    def click_delirium_statue(
        self,
        search_attempts: int = 5,
        approach_wait: float = 1.5,
        loot_drop_delay: float = 2.0,
        label: str = "DELIRIUM STATUE",
        save_debug: bool = True,
        loot_pos: Optional[Tuple[float, float]] = None,
        require_loot_proximity: bool = True,
        max_loot_distance: float = 35.0,
        walk_to_loot_if_far: bool = True,
        portal_pos: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Locates the Statue of Delirium, moves mouse inside game window to click it,
        and waits loot_drop_delay (2.0s) for encounter loot to drop before looting begins.
        Automatically verifies that character is near/at the LOOT dot (white dot) before detecting
        the statue, walking to the LOOT dot first if character is far away.
        Caches portal position so exit portal is not redundantly re-detected across search attempts.
        Automatically saves an annotated debug screenshot showing the detected statue and click position.
        """
        # In Room 7, only stop combat if the exit portal is confirmed visible
        portal_seen = (getattr(self, "_last_detected_portal_pos", None) is not None) or (self.locate_portal() is not None)
        if portal_seen:
            self.disable_persistent_combat()

        # Proximity Check: Only detect/click Delirium statue if character is near / at the LOOT dot
        if require_loot_proximity and loot_pos is not None:
            curr_pos = self.latest_pos or (self.last_known_pos if (time.time() - getattr(self, "last_known_time", 0)) < 2.0 else None)
            if curr_pos is not None:
                dist_to_loot = math.hypot(loot_pos[0] - curr_pos[0], loot_pos[1] - curr_pos[1])
                if dist_to_loot > max_loot_distance:
                    if walk_to_loot_if_far:
                        _log(f"    [DELIRIUM PROXIMITY] Character at ({curr_pos[0]:.1f}, {curr_pos[1]:.1f}) is {dist_to_loot:.1f}px away from LOOT dot ({loot_pos[0]:.1f}, {loot_pos[1]:.1f}) > {max_loot_distance:.1f}px. Walking to LOOT dot first...")
                        self.status_message = f"[{label}] Walking to LOOT dot before Delirium..."
                        self._walk_to_coordinate(
                            (float(loot_pos[0]), float(loot_pos[1])),
                            label="NAV→LOOT (DELIRIUM)",
                            timeout=8.0,
                            arrival_threshold=min(15.0, max_loot_distance),
                        )
                        time.sleep(0.2)
                        curr_pos = self.latest_pos or self.last_known_pos
                        if curr_pos is not None:
                            dist_to_loot = math.hypot(loot_pos[0] - curr_pos[0], loot_pos[1] - curr_pos[1])

                    if dist_to_loot > max_loot_distance:
                        _log(f"    [DELIRIUM PROXIMITY] Character is {dist_to_loot:.1f}px away from LOOT dot ({loot_pos[0]:.1f}, {loot_pos[1]:.1f}) > {max_loot_distance:.1f}px! Skipping Delirium statue detection.")
                        return True
                    else:
                        _log(f"    [DELIRIUM PROXIMITY] Character arrived near LOOT dot ({curr_pos[0]:.1f}, {curr_pos[1]:.1f}, dist={dist_to_loot:.1f}px <= {max_loot_distance:.1f}px).")
                else:
                    _log(f"    [DELIRIUM PROXIMITY] Confirmed character is at/near LOOT dot ({curr_pos[0]:.1f}, {curr_pos[1]:.1f}, dist={dist_to_loot:.1f}px <= {max_loot_distance:.1f}px).")
            else:
                _log("    [DELIRIUM PROXIMITY] Live player position not available; proceeding with visual Delirium search.")

        # Keep loot labels hidden during Delirium statue interaction to avoid obscuring statue or minimap
        time.sleep(0.12)  # Settle pause after movement

        statue_pos = None
        last_screen = None

        for attempt in range(1, search_attempts + 1):
            if stop_handler.is_stopped() or not self.is_active:
                return False
            capt = self._get_capturer()
            try:
                last_screen = capt.capture()
            except Exception:
                last_screen = None

            # Always locate statue and portal fresh on current screen at the LOOT location
            statue_pos = self.locate_delirium_statue(screen=last_screen, save_debug=save_debug, portal_pos=portal_pos)
            if statue_pos is not None:
                break

            time.sleep(0.2)

        if statue_pos is None:
            _log(f"    [WARNING] Statue of Delirium not detected on screen after {search_attempts} attempts.")
            if save_debug and last_screen is not None:
                self.save_delirium_debug_screenshot(screen=last_screen, detected=False, conf=0.0)
            return True

        _log(f"    [ACTION] Clicking Statue of Delirium at screen ({statue_pos[0]}, {statue_pos[1]})...")
        self.status_message = f"[{label}] Clicking Delirium Statue..."
        if getattr(self, "_last_detected_portal_pos", None) is not None:
            self.disable_persistent_combat()
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        rx, ry = self.move_mouse_inside_game(statue_pos[0], statue_pos[1])
        time.sleep(0.06)
        if pydirectinput:
            pydirectinput.click()
            time.sleep(0.06)
            pydirectinput.mouseUp(button="left")
        time.sleep(0.12)

        if approach_wait > 0:
            self._wait_for_approach(approach_wait, reason=f"{label} DELIRIUM")
            if stop_handler.is_stopped() or not self.is_active:
                return False

        # Wait for loot to drop from the statue (2.0s)
        _log(f"    [ACTION] Waiting {loot_drop_delay:.1f}s for loot to drop from Delirium statue...")
        self.status_message = f"[{label}] Waiting for loot drop ({loot_drop_delay:.1f}s)..."
        drop_start = time.time()
        while (time.time() - drop_start) < loot_drop_delay:
            if stop_handler.is_stopped() or not self.is_active:
                return False
            time.sleep(0.05)

        _log(f"    [ACTION] Loot drop wait complete. Ready for loot pickup.")
        return True


    def wait_for_user_key(
        self,
        key: str = "f5",
        prompt: str = "Press [F5] to confirm",
        timeout: float = 0.0,
    ) -> bool:
        """
        Pauses bot execution and waits for user confirmation via key press (default: F5).
        Allows emergency stop (F2) and active checking.
        """
        target_k = str(key).lower().strip()
        self.release_all_keys()
        self.waiting_for_user_key = True
        self.waiting_user_key_name = target_k
        self.status_message = f"WAITING FOR USER: {prompt}"
        _log(f"\n[USER CONFIRMATION] >>> {prompt} (or 'F2' to halt bot)...")

        # Debounce: If key was held before entering, wait for release
        if keyboard:
            try:
                if keyboard.is_pressed(target_k):
                    while keyboard.is_pressed(target_k):
                        time.sleep(0.05)
            except Exception:
                pass

        start_time = time.time()
        confirmed = False
        try:
            while self.waiting_for_user_key:
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                if timeout > 0 and (time.time() - start_time) >= timeout:
                    _log(f"  [USER CONFIRMATION] Timeout ({timeout:.1f}s) elapsed. Proceeding...")
                    confirmed = True
                    break
                if keyboard:
                    try:
                        if keyboard.is_pressed(target_k):
                            _log(f"  [USER CONFIRMATION] '{target_k.upper()}' pressed! Proceeding with next action.")
                            confirmed = True
                            break
                    except Exception:
                        pass
                time.sleep(0.05)
        finally:
            self.waiting_for_user_key = False
            # Wait for key release
            if keyboard:
                try:
                    if keyboard.is_pressed(target_k):
                        while keyboard.is_pressed(target_k):
                            time.sleep(0.05)
                except Exception:
                    pass
        time.sleep(0.15)
        return confirmed


    def confirm_user_key(self) -> bool:
        """Allows visualizer or external triggers to satisfy waiting_for_user_key."""
        if self.waiting_for_user_key:
            _log(f"  [USER CONFIRMATION] Confirmed via external trigger/visualizer.")
            self.waiting_for_user_key = False
            return True
        return False


    def click_exit_portal(
        self,
        search_attempts: int = 5,
        approach_wait: float = 2.0,
        screen: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Locates and clicks the exit portal on screen to return to the hideout.
        If the portal is not found, walks back to the LOOT dot (where the portal is adjacent) and retries.
        """
        self.release_all_keys()
        self.status_message = "Locating Exit Portal..."
        portal_pos = None

        for attempt in range(1, search_attempts + 1):
            if stop_handler.is_stopped() or not self.is_active:
                return False
            portal_pos = self.locate_portal(screen=screen)
            if portal_pos is not None:
                break
            time.sleep(0.25)

        # If not found on screen, character may have drifted during looting: walk to LOOT dot and retry
        if portal_pos is None and not stop_handler.is_stopped() and self.is_active:
            loot_pos = self._resolve_target_loot_pos({}, context or {}, zone_label="PINK DOT #7")
            if loot_pos is not None:
                _log(f"  [EXIT PORTAL RETRY] Portal not visible on screen. Walking to LOOT dot at ({loot_pos[0]:.1f}, {loot_pos[1]:.1f}) and re-scanning...")
                self.status_message = "Walking to LOOT dot for Portal..."
                self._walk_to_coordinate(loot_pos, label="NAV→PORTAL-RETRY", timeout=8.0, arrival_threshold=15.0)
                time.sleep(0.3)
                for retry in range(1, search_attempts + 1):
                    if stop_handler.is_stopped() or not self.is_active:
                        return False
                    portal_pos = self.locate_portal()
                    if portal_pos is not None:
                        _log(f"  [EXIT PORTAL RETRY SUCCESS] Exit portal detected at screen ({portal_pos[0]}, {portal_pos[1]})!")
                        break
                    time.sleep(0.25)

        if portal_pos is None:
            _log(f"  [EXIT PORTAL] Exit portal not found after {search_attempts} attempts.")
            return False

        # Stop combat before exiting zone through portal
        self.disable_persistent_combat()

        _log(f"  [ACTION] Clicking Exit Portal at screen ({portal_pos[0]}, {portal_pos[1]})...")
        self.status_message = "Clicking Exit Portal..."
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        rx, ry = self.move_mouse_inside_game(portal_pos[0], portal_pos[1])
        time.sleep(0.06)
        if pydirectinput:
            pydirectinput.click()
            time.sleep(0.08)
            pydirectinput.mouseUp(button="left")
        time.sleep(0.2)

        if approach_wait > 0:
            self._wait_for_approach(approach_wait, reason="EXIT PORTAL")

        return True

