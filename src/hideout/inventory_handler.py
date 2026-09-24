"""
Inventory window detection, open state verification, hideout layout verification, and inventory screenshots.
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


class HideoutInventoryMixin:
    """Inventory window detection, open state verification, hideout layout verification, and inventory screenshots."""

    def locate_inventory_window(
        self,
        screen: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Dynamically locates the player's open Inventory window and detects exact row and column centers.
        Uses multi-scale template matching across resolutions (1080p, 1440p, 1800p, 4K) and dynamic
        horizontal edge gradient profiling to find the 5 row centers, ensuring reliable operation
        across multiple screens (e.g. moving between Screen 2 1080p and Screen 1 1800p OLED).

        :param screen: Optional screen image array. If None, captures current screen.
        :param threshold: Match confidence threshold (defaults to self.inventory_match_threshold).
        :return: Dictionary containing origin, dimensions, scale, row_centers, col_centers, or None.
        """
        if self.inventory_title_tpl is None and self.inventory_close_tpl is None:
            self._load_stash_and_inventory_templates()
        if self.inventory_title_tpl is None and self.inventory_close_tpl is None:
            return None

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return None

        eff_thresh = threshold if threshold is not None else self.inventory_match_threshold
        sh, sw = screen.shape[:2]
        scale_est = sh / 1080.0

        # Build candidate scales tailored around current resolution scale
        scale_candidates = [
            round(scale_est * f, 3) for f in [0.90, 0.95, 1.0, 1.05, 1.10]
        ] + [1.0, 1.25, 1.333, 1.5, 1.667, 2.0]
        scales = sorted(set(scale_candidates))

        # In Path of Exile, the Inventory window is docked on the right side of the screen.
        # Check right side first for performance, and fallback to full screen if not found or on smaller canvases.
        roi_candidates = []
        if sw >= 1200:
            roi_candidates.append((int(sw * 0.45), screen[:, int(sw * 0.45) :]))
        roi_candidates.append((0, screen))

        best_conf = -1.0
        best_scale = scale_est
        best_loc = None

        # 1. Multi-scale match on golden 'INVENTORY' title banner
        if self.inventory_title_tpl is not None:
            th, tw = self.inventory_title_tpl.shape[:2]
            for right_x_offset, search_roi in roi_candidates:
                for s in scales:
                    scaled_w = int(tw * s)
                    scaled_h = int(th * s)
                    if search_roi.shape[0] < scaled_h or search_roi.shape[1] < scaled_w:
                        continue
                    scaled_tpl = (
                        self.inventory_title_tpl
                        if abs(s - 1.0) < 0.01
                        else cv2.resize(
                            self.inventory_title_tpl,
                            (scaled_w, scaled_h),
                            interpolation=cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA,
                        )
                    )
                    res = cv2.matchTemplate(search_roi, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, max_l = cv2.minMaxLoc(res)
                    if max_v > best_conf:
                        best_conf = max_v
                        best_scale = s
                        best_loc = (max_l[0] + right_x_offset, max_l[1])
                if best_conf >= eff_thresh:
                    break

        # 2. Fallback to red close button (X) if title was not detected
        if (best_conf < eff_thresh or best_loc is None) and self.inventory_close_tpl is not None:
            ch, cw = self.inventory_close_tpl.shape[:2]
            for right_x_offset, search_roi in roi_candidates:
                for s in scales:
                    scaled_w = int(cw * s)
                    scaled_h = int(ch * s)
                    if search_roi.shape[0] < scaled_h or search_roi.shape[1] < scaled_w:
                        continue
                    scaled_tpl = (
                        self.inventory_close_tpl
                        if abs(s - 1.0) < 0.01
                        else cv2.resize(
                            self.inventory_close_tpl,
                            (scaled_w, scaled_h),
                            interpolation=cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA,
                        )
                    )
                    res = cv2.matchTemplate(search_roi, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, max_l = cv2.minMaxLoc(res)
                    if max_v > best_conf and max_v >= eff_thresh:
                        best_conf = max_v
                        best_scale = s
                        best_loc = (max_l[0] + right_x_offset - int(393 * s), max_l[1])
                        break
                if best_conf >= eff_thresh:
                    break

        if best_conf < eff_thresh or best_loc is None:
            return None

        # Snap scale if very close to scale_est or 1.0 to prevent minor resampling drift
        if abs(best_scale - 1.0) <= 0.05:
            best_scale = 1.0
        elif abs(best_scale - scale_est) <= 0.05:
            best_scale = scale_est

        inv_origin_x = best_loc[0] - int(200 * best_scale)
        inv_origin_y = best_loc[1] - int(45 * best_scale)
        inv_w = int(629 * best_scale)
        inv_h = int(1024 * best_scale)

        # 3. Dynamic row center detection via horizontal edge gradient profiling
        y_band_min = max(0, inv_origin_y + int(530 * best_scale))
        y_band_max = min(sh, inv_origin_y + int(890 * best_scale))
        x_band_min = max(0, inv_origin_x)
        x_band_max = min(sw, inv_origin_x + inv_w)

        grid_roi = screen[y_band_min:y_band_max, x_band_min:x_band_max]
        row_centers = None
        row_lines = None
        row_method = "calibrated_scaled"

        if grid_roi.shape[0] > int(100 * best_scale) and grid_roi.shape[1] > int(200 * best_scale):
            try:
                import itertools
                gray = cv2.cvtColor(grid_roi, cv2.COLOR_BGR2GRAY)
                sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
                h_prof = sobel_y.sum(axis=1)
                k_size = max(3, int(5 * best_scale))
                if k_size % 2 == 0:
                    k_size += 1
                h_smooth = np.convolve(h_prof, np.ones(k_size) / float(k_size), mode="same")

                win = max(8, int(14 * best_scale))
                peaks = []
                for i in range(win, len(h_smooth) - win):
                    if h_smooth[i] == max(h_smooth[max(0, i - win) : min(len(h_smooth), i + win + 1)]):
                        peaks.append((y_band_min + i, h_smooth[i]))
                peaks.sort(key=lambda p: p[1], reverse=True)

                candidates = sorted([p[0] for p in peaks[:15]])
                min_step = 40.0 * best_scale
                max_step = 68.0 * best_scale

                peak_map = dict(peaks)
                best_score = -1.0
                best_comb = None

                for comb in itertools.combinations(candidates, 6):
                    diffs = [comb[j + 1] - comb[j] for j in range(5)]
                    if all(min_step <= d <= max_step for d in diffs):
                        score = sum(peak_map.get(y, 0) for y in comb)
                        if score > best_score:
                            best_score = score
                            best_comb = comb

                if best_comb is not None:
                    row_lines = list(best_comb)
                    row_centers = [(best_comb[j] + best_comb[j + 1]) / 2.0 for j in range(5)]
                    row_method = "dynamic_edges"
            except Exception as e:
                _log(f"  [INVENTORY WARNING] Dynamic edge row detection error: {e}")

        # Fallback to calibrated proportional row centers
        if row_centers is None:
            calibrated_rel_y = [609.5, 665.5, 718.5, 771.0, 825.5]
            row_centers = [inv_origin_y + c * best_scale for c in calibrated_rel_y]
            row_method = "calibrated_scaled"

        # Column centers for all 12 columns (scale-adapted)
        calibrated_rel_x = [26.0, 78.0, 130.5, 183.0, 235.5, 288.5, 341.0, 393.5, 446.5, 499.0, 551.5, 599.5]
        col_centers = [inv_origin_x + c * best_scale for c in calibrated_rel_x]

        return {
            "found": True,
            "confidence": best_conf,
            "scale": best_scale,
            "title_pos": best_loc,
            "inv_origin": (inv_origin_x, inv_origin_y),
            "inv_dim": (inv_w, inv_h),
            "row_centers": row_centers,
            "col_centers": col_centers,
            "row_lines": row_lines,
            "row_method": row_method,
        }


    def is_inventory_open(
        self,
        threshold: Optional[float] = None,
        screen: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Detects whether the player Inventory window is currently open on screen.
        Checks for the golden 'INVENTORY' header banner or the red close button (X)
        using multi-scale matching for multi-monitor compatibility.
        Includes a 20-60ms fast path on the right side of the screen.
        """
        if self.inventory_title_tpl is None and self.inventory_close_tpl is None:
            self._load_stash_and_inventory_templates()

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return False

        eff_thresh = threshold if threshold is not None else self.inventory_match_threshold
        sh, sw = screen.shape[:2]

        # Fast-path on right side of screen (sw * 0.40 : ) across standard scales [1.0, 1.05, 1.10, 0.95]
        if sw >= 800:
            roi_x_start = int(sw * 0.40)
            roi = screen[:, roi_x_start:]
            # 1. Fast check for red close button (X)
            if self.inventory_close_tpl is not None:
                for s in [1.0, 1.05, 1.10, 0.95]:
                    scaled_tpl = self.inventory_close_tpl if s == 1.0 else cv2.resize(
                        self.inventory_close_tpl,
                        (int(self.inventory_close_tpl.shape[1] * s), int(self.inventory_close_tpl.shape[0] * s)),
                        interpolation=cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA,
                    )
                    if roi.shape[0] >= scaled_tpl.shape[0] and roi.shape[1] >= scaled_tpl.shape[1]:
                        res = cv2.matchTemplate(roi, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                        _, mv, _, ml = cv2.minMaxLoc(res)
                        if mv >= 0.70:
                            _log(f"  [INVENTORY MATCH] Confirmed Inventory window open (close button conf={mv:.3f}, scale={s:.2f}) at ({ml[0] + roi_x_start}, {ml[1]}).")
                            return True
            # 2. Fast check for golden 'INVENTORY' title banner
            if self.inventory_title_tpl is not None:
                for s in [1.0, 1.05, 1.10, 0.95]:
                    scaled_tpl = self.inventory_title_tpl if s == 1.0 else cv2.resize(
                        self.inventory_title_tpl,
                        (int(self.inventory_title_tpl.shape[1] * s), int(self.inventory_title_tpl.shape[0] * s)),
                        interpolation=cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA,
                    )
                    if roi.shape[0] >= scaled_tpl.shape[0] and roi.shape[1] >= scaled_tpl.shape[1]:
                        res = cv2.matchTemplate(roi, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                        _, mv, _, ml = cv2.minMaxLoc(res)
                        if mv >= eff_thresh:
                            _log(f"  [INVENTORY MATCH] Confirmed Inventory window open (title conf={mv:.3f}, scale={s:.2f}) at ({ml[0] + roi_x_start}, {ml[1]}).")
                            return True

        # Fallback to full locate_inventory_window (for non-standard screens or off-scale layouts)
        info = self.locate_inventory_window(screen=screen, threshold=threshold)
        if info is not None:
            _log(f"  [INVENTORY MATCH] Confirmed Inventory window open (title conf={info['confidence']:.3f}, scale={info['scale']:.2f}, method={info['row_method']}) at ({info['title_pos'][0]}, {info['title_pos'][1]}).")
            return True
        return False


    def is_in_hideout(
        self,
        threshold: Optional[float] = None,
        screen: Optional[np.ndarray] = None,
        check_stash_fallback: bool = False,
        check_map_device_fallback: bool = True,
        require_map_device: Optional[bool] = None,
        verbose: bool = True,
        set_state: bool = True,
    ) -> bool:
        """
        Detects whether character is in the hideout by matching hideout_layout.png
        against the top-right minimap region using multi-scale matching.

        Detection Modes:
        - 'minimap': Strictly verifies Hideout minimap layout (threshold >= 0.50).
        - 'minimap_and_map_device': Dual confirmation requiring BOTH minimap match
          AND Map Device presence.
        - 'minimap_or_map_device' (default): Minimap is primary; if borderline,
          verified against Map Device.

        CRITICAL SAFETY RULE:
        STASH can appear in enemy areas (such as Simulacrum, Delve, etc.).
        Detecting STASH alone is NEVER sufficient to declare Hideout!
        """
        if self.hideout_layout_tpl is None:
            self._load_stash_and_inventory_templates()
        if self.hideout_layout_tpl is None:
            return False

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return False

        eff_thresh = threshold if threshold is not None else self.hideout_match_threshold
        req_md = require_map_device if require_map_device is not None else (getattr(self, "hideout_detection_mode", "minimap_or_map_device") == "minimap_and_map_device")

        sh, sw = screen.shape[:2]
        lh, lw = self.hideout_layout_tpl.shape[:2]

        scales = [1.0, 0.95, 1.05, 0.90, 1.10]
        best_conf = -1.0
        best_scale = 1.0

        # Check upper-right minimap quadrant (top 40%, right 35%)
        mm_region = screen[: int(sh * 0.40), int(sw * 0.65) :]
        for s in scales:
            tw = int(lw * s)
            th = int(lh * s)
            if mm_region.shape[0] < th or mm_region.shape[1] < tw:
                continue
            scaled_tpl = self.hideout_layout_tpl if s == 1.0 else cv2.resize(self.hideout_layout_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(mm_region, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_conf:
                best_conf = max_val
                best_scale = s

        # Fallback: check upper-right half (top 50%, right 50%)
        if best_conf < eff_thresh:
            top_half = screen[: int(sh * 0.50), int(sw * 0.50) :]
            for s in [1.0, 0.95, 1.05]:
                tw = int(lw * s)
                th = int(lh * s)
                if top_half.shape[0] < th or top_half.shape[1] < tw:
                    continue
                scaled_tpl = self.hideout_layout_tpl if s == 1.0 else cv2.resize(self.hideout_layout_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(top_half, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_conf:
                    best_conf = max_val
                    best_scale = s

        self.last_hideout_confidence = best_conf
        if verbose:
            _log(f"  [HIDEOUT CHECK] Minimap match confidence: {best_conf:.3f} (scale: {best_scale:.2f}, threshold: {eff_thresh:.2f})")

        is_confirmed_hideout = False

        # Mode A: Dual Confirmation required (Minimap AND Map Device)
        if req_md:
            if best_conf >= eff_thresh:
                md_pos = self.locate_map_device(screen=screen, threshold=0.68)
                if md_pos is not None:
                    if verbose or not self.in_hideout:
                        _log(f"  [HIDEOUT MATCH] Confirmed via Dual Check: Minimap (conf={best_conf:.3f}) AND Map Device at ({md_pos[0]}, {md_pos[1]})")
                    is_confirmed_hideout = True
                else:
                    if verbose:
                        _log(f"  [HIDEOUT CHECK] Minimap matched ({best_conf:.3f} >= {eff_thresh:.2f}) but Map Device NOT visible. Pending confirmation.")
            else:
                if verbose:
                    _log(f"  [HIDEOUT CHECK] Dual mode: Minimap below threshold ({best_conf:.3f} < {eff_thresh:.2f}). Not hideout.")

        # Mode B: Minimap Primary with Map Device Confirmation
        else:
            if best_conf >= eff_thresh:
                if verbose or not self.in_hideout:
                    _log(f"  [HIDEOUT MATCH] Confirmed in hideout via minimap (conf={best_conf:.3f} >= {eff_thresh:.2f})")
                is_confirmed_hideout = True
            elif check_map_device_fallback and best_conf >= 0.35:
                # Borderline minimap match: verify if Map Device is visible
                md_pos = self.locate_map_device(screen=screen, threshold=0.68)
                if md_pos is not None:
                    if verbose or not self.in_hideout:
                        _log(f"  [HIDEOUT MATCH] Minimap conf was {best_conf:.3f}, but Map Device detected on screen at ({md_pos[0]}, {md_pos[1]}) -> Confirmed in Hideout!")
                    is_confirmed_hideout = True

        # STASH fallback check: STASH ALONE CAN NEVER DECLARE HIDEOUT!
        if not is_confirmed_hideout and check_stash_fallback:
            stash_pos = self.locate_stash(screen=screen)
            if stash_pos is not None:
                # Stashes can appear in enemy areas (e.g. Simulacrum). Require Map Device to confirm hideout!
                md_pos = self.locate_map_device(screen=screen, threshold=0.68)
                if md_pos is not None:
                    if verbose or not self.in_hideout:
                        _log(f"  [HIDEOUT MATCH] STASH and Map Device both detected on screen -> Confirmed in Hideout!")
                    is_confirmed_hideout = True
                else:
                    if verbose:
                        _log(f"  [HIDEOUT CHECK] STASH detected at {stash_pos}, but Map Device and Hideout Minimap absent (conf={best_conf:.3f} < {eff_thresh:.2f}). Stash may be in enemy zone (Simulacrum). NOT Hideout!")

        if is_confirmed_hideout:
            if set_state:
                self.in_hideout = True
                # Keep self.is_active = True while autopilot is running so hideout routine steps
                # (stash, map device, delusion popup, portals) can execute without interruption.
                self.disable_persistent_combat()
                self.release_all_keys()
            return True

        return False


    def save_inventory_screenshot(self, screen: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Saves a screenshot showing the inventory window to a dedicated folder.
        """
        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return None

        os.makedirs(self.inventory_screenshots_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = os.path.join(self.inventory_screenshots_dir, f"inventory_{ts}.png")

        # Crop inventory window if located, or save full screen
        sh, sw = screen.shape[:2]
        crop_saved = False
        info = self.locate_inventory_window(screen=screen)
        if info is not None:
            inv_x, inv_y = info["inv_origin"]
            inv_w, inv_h = info["inv_dim"]
            inv_crop = screen[max(0, inv_y) : min(sh, inv_y + inv_h), max(0, inv_x) : min(sw, inv_x + inv_w)]
            if inv_crop.size > 0:
                cv2.imwrite(target_path, inv_crop)
                crop_saved = True

        if not crop_saved:
            cv2.imwrite(target_path, screen)

        _log(f"  [INVENTORY SCREENSHOT] Saved inventory window screenshot to '{target_path}'")
        return target_path

