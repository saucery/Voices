"""
Map device location, clicking, completed green node detection, and node accessibility checking.
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


class HideoutMapDeviceMixin:
    """Map device location, clicking, completed green node detection, and node accessibility checking."""

    def locate_map_device(
        self,
        screen: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the Map Device in the hideout using multi-scale template matching.
        Prioritizes the 'MAP DEVICE' label banner, falling back to full map device template.
        Returns desktop absolute (x, y) coordinates of the click target, or None if not found.
        """
        if self.map_device_label_tpl is None and self.map_device_full_tpl is None:
            self._load_stash_and_inventory_templates()

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return None

        eff_thresh = threshold if threshold is not None else self.map_device_match_threshold

        mon_left = 0
        mon_top = 0
        capt = self._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        sh, sw = screen.shape[:2]
        scale_est = (sh / 1080.0) if sh >= 720 else 1.0
        scales = sorted(list(set([
            1.0,
            round(scale_est * 0.90, 3),
            round(scale_est * 0.95, 3),
            round(scale_est * 1.0, 3),
            round(scale_est * 1.05, 3),
            round(scale_est * 1.10, 3),
        ])))

        # 1. Search for 'MAP DEVICE' label banner
        if self.map_device_label_tpl is not None:
            lh, lw = self.map_device_label_tpl.shape[:2]
            best_lbl_val = -1.0
            best_lbl_loc = None
            best_lbl_scale = 1.0

            for s in scales:
                tw = int(lw * s)
                th = int(lh * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.map_device_label_tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                    self.map_device_label_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                )
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
                # When portals are open right beneath the Map Device, clicking near the bottom edge
                # can hit the portal label (e.g. 'BLUFF (COMPLETED)').
                # Click directly on the upper/center text of the label banner, or slightly above.
                if lh <= 26:
                    cy = int(best_lbl_loc[1] + int(cur_h * 0.35))
                else:
                    cy = int(best_lbl_loc[1] + int(cur_h * 0.22))
                cy += getattr(self, "map_device_click_y_offset_px", 0)
                _log(f"  [MAP DEVICE MATCH] Found Map Device label (conf={best_lbl_val:.3f} >= {eff_thresh:.2f}, scale={best_lbl_scale:.2f}) at screen ({cx}, {cy})")
                return (cx + mon_left, cy + mon_top)

        # 2. Search for Map Device full structure
        if self.map_device_full_tpl is not None:
            dh, dw = self.map_device_full_tpl.shape[:2]
            best_dev_val = -1.0
            best_dev_loc = None
            best_dev_scale = 1.0
            dev_thresh = max(0.50, eff_thresh - 0.05)

            for s in scales:
                tw = int(dw * s)
                th = int(dh * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.map_device_full_tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                    self.map_device_full_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                )
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_dev_val:
                    best_dev_val = max_val
                    best_dev_loc = max_loc
                    best_dev_scale = s

            if best_dev_val >= dev_thresh and best_dev_loc is not None:
                cur_w = int(dw * best_dev_scale)
                cur_h = int(dh * best_dev_scale)
                cx = int(best_dev_loc[0] + cur_w // 2)
                # Click top-middle (golden dome) to guarantee clicking above open portals
                cy = int(best_dev_loc[1] + int(cur_h * 0.25)) + getattr(self, "map_device_click_y_offset_px", 0)
                _log(f"  [MAP DEVICE MATCH] Found Map Device (conf={best_dev_val:.3f} >= {dev_thresh:.2f}, scale={best_dev_scale:.2f}) at screen ({cx}, {cy})")
                return (cx + mon_left, cy + mon_top)

        return None


    def click_map_device(
        self,
        search_attempts: int = 8,
        timeout: float = 12.0,
        approach_wait: float = 1.5,
    ) -> bool:
        """
        Locates the Map Device in the hideout, moves mouse to it, and left-clicks.
        Waits for character to approach and for the Atlas / Map Device screen to open.
        Includes robust fallbacks for opening Atlas via hotkey 'g' and hideout center click.
        """
        self.release_all_keys()
        self.status_message = "Locating Map Device..."
        _log("\n[MAP DEVICE] Locating and interacting with Map Device in hideout...")

        # 1. Locate Map Device via template matching (banner label or structure)
        start_t = time.time()
        map_device_pos = None

        while (time.time() - start_t) < min(timeout, 6.0):
            if stop_handler.is_stopped():
                return False
            map_device_pos = self.locate_map_device()
            if map_device_pos is not None:
                break
            time.sleep(0.3)

        if map_device_pos is not None:
            self.last_map_device_pos = map_device_pos
            _log(f"  [MAP DEVICE] Left-clicking Map Device at desktop ({map_device_pos[0]}, {map_device_pos[1]})...")
            self.status_message = "Clicking Map Device..."
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            rx, ry = self.move_mouse_inside_game(map_device_pos[0], map_device_pos[1])
            time.sleep(0.08)
            if pydirectinput:
                pydirectinput.click()
                time.sleep(0.08)
                pydirectinput.mouseUp(button="left")
            time.sleep(0.3)
            if approach_wait > 0:
                self._wait_for_approach(approach_wait, reason="MAP DEVICE")
        else:
            # 3. Fallback: Press 'G' (PoE Atlas Hotkey)
            _log("  [MAP DEVICE] Map Device template not found. Pressing 'G' hotkey to open Atlas...")
            self.status_message = "Opening Atlas ('G')..."
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            if pydirectinput:
                pydirectinput.press("g")
                time.sleep(0.6)

            # Check if Atlas opened via 'G'
            nodes = self.detect_simulacrum_map_nodes()
            if nodes:
                _log(f"  [MAP DEVICE SUCCESS] Atlas opened via 'G' hotkey ({len(nodes)} Simulacrum node(s) visible)!")
                return True

            # 4. Fallback: Click hideout center (where Map Device is standardly situated)
            _log("  [MAP DEVICE] 'G' key didn't reveal nodes yet. Clicking hideout center as fallback...")
            capt = self._get_capturer()
            screen = capt.capture() if capt else None
            if screen is not None and screen.size > 0:
                sh, sw = screen.shape[:2]
                mon_left = 0
                mon_top = 0
                if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
                    monitors = capt._sct.monitors
                    if 0 <= self.monitor_idx < len(monitors):
                        mon_left = monitors[self.monitor_idx].get("left", 0)
                        mon_top = monitors[self.monitor_idx].get("top", 0)
                cx = sw // 2 + mon_left
                cy = int(sh * 0.38) + mon_top
                self.move_mouse_inside_game(cx, cy)
                time.sleep(0.08)
                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.08)
                    pydirectinput.mouseUp(button="left")
                time.sleep(approach_wait)

        # Wait for Atlas screen to open (verify by detecting Simulacrum icon or open map)
        self.status_message = "Waiting for Map Screen..."
        atlas_open = False
        verify_start = time.time()
        while (time.time() - verify_start) < 4.0:
            if stop_handler.is_stopped():
                return False
            nodes = self.detect_simulacrum_map_nodes()
            if nodes:
                atlas_open = True
                _log(f"  [MAP DEVICE SUCCESS] Atlas map screen confirmed open ({len(nodes)} Simulacrum node(s) visible)!")
                break
            time.sleep(0.3)

        if not atlas_open:
            _log("  [MAP DEVICE INFO] Map Device interacted. Proceeding to Simulacrum search...")

        return True


    def detect_green_completed_nodes(
        self,
        screen: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detects all completed (green) map nodes across the Atlas map screen.
        Completed nodes appear as vibrant emerald orbs with a gold border and circular shape.
        All maps linked by a stripped/dashed line to ANY green location are accessible to the player.
        """
        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return []

        sh, sw = screen.shape[:2]
        scale = (sh / 1080.0) if sh >= 720 else 1.0
        b_ch, g_ch, r_ch = cv2.split(screen)
        hsv_img = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)

        # 1. Emerald green color dominance with neon brightness & morphological closing
        neon = (g_ch >= 150) & (g_ch.astype(np.int32) >= r_ch.astype(np.int32) * 1.20) & (g_ch.astype(np.int32) >= b_ch.astype(np.int32) * 1.02)
        neon = neon & (hsv_img[:, :, 0] >= 35) & (hsv_img[:, :, 0] <= 85) & (hsv_img[:, :, 1] >= 55) & (hsv_img[:, :, 2] >= 80)
        neon_u8 = neon.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(neon_u8, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = max(10, int(15 * (scale ** 2)))
        max_a = max(120, int(350 * (scale ** 2)))

        # Check inventory visibility to constrain search bounds appropriately
        inv_open = False
        try:
            inv_open = self.is_inventory_open(screen=screen)
        except Exception:
            pass
        max_g_x = int(0.66 * sw) if inv_open else int(0.97 * sw)

        raw_greens = []
        for c in cnts:
            area = cv2.contourArea(c)
            if min_a <= area <= max_a:
                peri = cv2.arcLength(c, True)
                if peri > 0:
                    circ = 4 * np.pi * area / (peri * peri)
                    if circ >= 0.20:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            if cx < int(180 * scale) and cy < int(240 * scale):
                                continue
                            if cx > max_g_x:
                                continue
                            if cy > int(0.92 * sh) or cy < int(35 * scale):
                                continue
                            raw_greens.append({"center": (cx, cy), "area": area, "circularity": circ})

        # Merge nearby duplicate green detections (within 16px)
        greens = []
        for g in raw_greens:
            cx, cy = g["center"]
            if not any(np.hypot(cx - mg["center"][0], cy - mg["center"][1]) < 16 for mg in greens):
                greens.append(g)
        return greens


    def check_node_accessibility(
        self,
        circle_pos: Tuple[int, int],
        green_nodes: List[Dict[str, Any]],
        screen: np.ndarray,
        max_dist: int = 180,
        medal_pos: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Determines whether a map node circle is ACCESSIBLE to the player:
        1. Checks whether the node is connected by a dashed/stripped line to any completed GREEN location.
        2. Checks whether the node's circle exhibits the 'MAP NODE (ACCESSIBLE)' bright white core and glowing blue ring.
        Assigns an accessibility score:
            3: Connected to green node AND has glowing blue portal core (highest accessibility)
            2: Connected to green node via dashed/stripped line
            1: Glowing blue portal core with white center
            0: Inaccessible (dark circle, not connected to any green node)
        """
        sh, sw = screen.shape[:2]
        scale = (sh / 1080.0) if sh >= 720 else 1.0
        cx, cy = circle_pos

        # Check circle appearance (white core + blue/cyan portal ring)
        r_w = max(12, int(15 * scale))
        r_h = max(10, int(12 * scale))
        patch = screen[max(0, cy - r_h) : min(sh, cy + r_h + 1), max(0, cx - r_w) : min(sw, cx + r_w + 1)]
        has_blue_glow = False
        has_white_core = False
        if patch.size > 0:
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            blue_mask = (hsv_patch[:, :, 0] >= 95) & (hsv_patch[:, :, 0] <= 135) & (hsv_patch[:, :, 1] >= 55) & (hsv_patch[:, :, 2] >= 75)
            blue_portal_px = int(np.count_nonzero(blue_mask))
            white_mask = (patch[:, :, 0] >= 195) & (patch[:, :, 1] >= 195) & (patch[:, :, 2] >= 195)
            white_portal_px = int(np.count_nonzero(white_mask))
            max_lum = int(np.max(patch))

            has_blue_glow = (blue_portal_px >= max(8, int(10 * scale * scale)))
            has_white_core = (white_portal_px >= 1) or (max_lum >= 235)

        # Check dashed line connectivity
        hsv_local = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        line_mask = (
            ((hsv_local[:, :, 0] >= 12) & (hsv_local[:, :, 0] <= 42) & (hsv_local[:, :, 1] >= 28) & (hsv_local[:, :, 2] >= 85))
            | (hsv_local[:, :, 2] >= 150)
        )
        line_mask_u8 = line_mask.astype(np.uint8) * 255

        scaled_max_dist = min(350, int(max_dist * (scale / 0.52 if scale < 0.7 else scale) * 1.6))
        margin = max(4, int(8 * scale))
        connected_greens = []

        for g in green_nodes:
            gx, gy = g["center"] if isinstance(g, dict) and "center" in g else g
            # Disregard green noise that overlaps the candidate medal itself
            if medal_pos is not None:
                if math.hypot(gx - medal_pos[0], gy - medal_pos[1]) < int(40 * scale):
                    continue
            dist = float(np.hypot(gx - cx, gy - cy))
            if dist < int(10 * scale) or dist > scaled_max_dist:
                continue
            num_pts = max(5, int(dist))
            t_vals = np.linspace(margin / dist, 1.0 - margin / dist, max(5, num_pts - 2 * margin))
            xs = (cx + t_vals * (gx - cx)).astype(int)
            ys = (cy + t_vals * (gy - cy)).astype(int)
            hits = 0
            box = max(1, int(1.5 * scale))
            for x, y in zip(xs, ys):
                if 0 <= y < sh and 0 <= x < sw:
                    if np.any(line_mask_u8[max(0, y - box) : y + box + 1, max(0, x - box) : x + box + 1] > 0):
                        hits += 1
            hit_ratio = hits / float(len(xs))
            if hit_ratio >= 0.18:
                connected_greens.append({"green_pos": (gx, gy), "dist": float(dist), "hit_ratio": float(hit_ratio)})

        has_green_link = bool(len(connected_greens) > 0)
        has_portal = bool(has_blue_glow and has_white_core)

        # In PoE2, accessible nodes have an active blue portal at their base
        if has_portal and has_green_link:
            acc_score = 3
        elif has_portal:
            acc_score = 2
        elif has_green_link:
            # Linked to green but no open blue portal
            acc_score = 0
        else:
            acc_score = 0

        is_accessible = bool(has_portal)
        return {
            "is_accessible": is_accessible,
            "acc_score": acc_score,
            "connected_greens": connected_greens,
            "has_blue_glow": has_blue_glow,
            "has_white_core": has_white_core,
        }

