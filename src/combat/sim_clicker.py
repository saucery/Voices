"""
Simulacrum item template matching, multi-pass detection, clicking, and approach waiting.
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


class SimClickerMixin:
    """Simulacrum item template matching, multi-pass detection, clicking, and approach waiting."""

    def locate_sim_template(self, sim_key: str, threshold: Optional[float] = None) -> Optional[Tuple[int, int]]:
        """
        Locates a sim selection template (sim1, sim2, sim3) on screen using multi-scale template matching
        with cross-template candidate discrimination to prevent false positives between sim banners.
        Returns desktop absolute coordinates (X, Y) of the match center, or None if not found.
        """
        tmpl = self.sim_templates.get(sim_key)
        if tmpl is None:
            self._load_sim_templates()
            tmpl = self.sim_templates.get(sim_key)

        if tmpl is None:
            return None

        eff_threshold = threshold if threshold is not None else self.sim_match_threshold
        self._last_sim_best_conf = 0.0
        self._last_sim_best_scale = 1.0

        capt = self._get_capturer()
        try:
            screen = capt.capture()
        except Exception as e:
            _log(f"  [WARNING] Screen capture failed during {sim_key} search: {e}")
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

        g_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        g_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        th, tw = g_tmpl.shape[:2]
        sh, sw = g_screen.shape[:2]

        primary_scales = [1.0, 0.95, 1.05]
        secondary_scales = [0.90, 1.10, 0.85, 1.15, 0.80, 1.20, 0.75, 1.25, 0.70, 1.30]

        temp_screen = g_screen.copy()
        overall_best_val = -1.0
        overall_best_scale = 1.0

        # Evaluate up to 3 candidate peaks to find genuine match even if another SIM has a higher peak on temp_screen
        for _ in range(3):
            best_val = -1.0
            best_loc = None
            best_scale = 1.0

            # Stage 1: Fast search on primary native scales
            for scale in primary_scales:
                sc_w = int(tw * scale)
                sc_h = int(th * scale)
                if sc_w > sw or sc_h > sh or sc_w < 15 or sc_h < 15:
                    continue
                resized = cv2.resize(g_tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(temp_screen, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = float(max_val)
                    best_loc = max_loc
                    best_scale = scale
                if best_val >= 0.82:
                    break

            # If confident match found, early-exit.
            # If best_val is very low (< 0.28), template is completely absent from screen -> skip remaining scales!
            if best_val < 0.82 and best_val >= 0.28:
                # Stage 2: Ambiguous candidate -> evaluate secondary scales
                for scale in secondary_scales:
                    sc_w = int(tw * scale)
                    sc_h = int(th * scale)
                    if sc_w > sw or sc_h > sh or sc_w < 15 or sc_h < 15:
                        continue
                    resized = cv2.resize(g_tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                    res = cv2.matchTemplate(temp_screen, resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_val:
                        best_val = float(max_val)
                        best_loc = max_loc
                        best_scale = scale

            if best_val > overall_best_val:
                overall_best_val = best_val
                overall_best_scale = best_scale

            if best_val < eff_threshold or best_loc is None:
                break

            # Cross-template discrimination: verify if candidate area actually matches target sim vs other loaded sims
            cx, cy = best_loc
            cw, ch = int(tw * best_scale), int(th * best_scale)
            pad_x, pad_y = 20, 15
            y1 = max(0, cy - pad_y)
            y2 = min(sh, cy + ch + pad_y)
            x1 = max(0, cx - pad_x)
            x2 = min(sw, cx + cw + pad_x)
            crop = g_screen[y1:y2, x1:x2]

            scores: Dict[str, float] = {}
            for k, other_tmpl in self.sim_templates.items():
                if other_tmpl is None:
                    continue
                gt = cv2.cvtColor(other_tmpl, cv2.COLOR_BGR2GRAY)
                t_best = -1.0
                for sc in [best_scale * 0.95, best_scale, best_scale * 1.05]:
                    stw, sth = int(gt.shape[1] * sc), int(gt.shape[0] * sc)
                    if stw > crop.shape[1] or sth > crop.shape[0] or stw < 15 or sth < 15:
                        continue
                    r = cv2.resize(gt, (stw, sth), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
                    mres = cv2.matchTemplate(crop, r, cv2.TM_CCOEFF_NORMED)
                    _, mv, _, _ = cv2.minMaxLoc(mres)
                    if mv > t_best:
                        t_best = float(mv)
                scores[k] = t_best

            winner = max(scores, key=scores.get) if scores else sim_key
            target_score = scores.get(sim_key, best_val)
            winner_score = scores.get(winner, target_score)

            if winner == sim_key or winner_score <= target_score + 0.05:
                # Validated genuine match for sim_key
                self._last_sim_best_conf = best_val
                self._last_sim_best_scale = best_scale
                center_x = int(cx + (tw * best_scale) / 2)
                center_y = int(cy + (th * best_scale) / 2)
                desktop_x = mon_left + center_x
                desktop_y = mon_top + center_y
                _log(f"  [SIM MATCH] Found {sim_key} (conf={best_val:.2f}, scale={best_scale:.2f}) at screen ({desktop_x}, {desktop_y})")
                return desktop_x, desktop_y
            else:
                # Candidate area is a competing SIM banner (e.g. sim2/sim3 while searching for sim1) -> mask region and search next peak
                mask_y1 = max(0, cy - 5)
                mask_y2 = min(sh, cy + ch + 5)
                mask_x1 = max(0, cx - 5)
                mask_x2 = min(sw, cx + cw + 5)
                temp_screen[mask_y1:mask_y2, mask_x1:mask_x2] = 0

        self._last_sim_best_conf = overall_best_val
        self._last_sim_best_scale = overall_best_scale
        return None


    def _detect_and_click_sims(
        self,
        prefix: str = "[SIM]",
        sim_order: Optional[List[str]] = None,
        y_offset_px: Optional[int] = None,
        x_offset_px: Optional[int] = None,
        approach_wait: Optional[float] = None,
        settle_wait: float = 0.0,
        search_attempts: int = 1,
        threshold: Optional[float] = None,
        verify_click: Optional[bool] = None,
        verify_delay: Optional[float] = None,
        max_click_attempts: Optional[int] = None,
        max_sim_clicks: Optional[int] = None,
        subsequent_search_attempts: int = 1,
    ) -> List[str]:
        """
        Detects and selects Sims according to strict priority order:
        Default priority: SIM1 -> SIM3 (if exists) -> SIM2 (if still exists).
        Returns list of sim names that were clicked (e.g. ['sim1', 'sim3']).
        Clicks are offset by sim_click_y_offset_px (e.g. +35px) below the detected template center.
        Includes double-check verification after approach/wait to ensure the SIM was registered.
        """
        sims_clicked: List[str] = []
        eff_y_offset = y_offset_px if y_offset_px is not None else self.sim_click_y_offset_px
        eff_x_offset = x_offset_px if x_offset_px is not None else self.sim_click_x_offset_px
        eff_app_wait = approach_wait if approach_wait is not None else self.sim_approach_wait_seconds
        eff_verify = verify_click if verify_click is not None else getattr(self, "sim_verify_click_enabled", True)
        eff_verify_delay = verify_delay if verify_delay is not None else getattr(self, "sim_verify_delay_seconds", 1.0)
        eff_max_attempts = max_click_attempts if max_click_attempts is not None else getattr(self, "sim_max_click_attempts", 2)
        eff_max_clicks = max_sim_clicks if max_sim_clicks is not None else getattr(self, "sim_max_clicks", None)
        effective_order = sim_order if (sim_order is not None and len(sim_order) > 0) else ["sim1", "sim3", "sim2"]

        self.ensure_loot_labels_visible()

        if settle_wait > 0:
            time.sleep(settle_wait)

        def _click_sim_at(sim_key: str, sim_label: str, detected_pos: Tuple[int, int]) -> Tuple[int, int]:
            target_x = detected_pos[0] + eff_x_offset
            target_y = detected_pos[1] + eff_y_offset
            cx, cy = self.move_mouse_inside_game(target_x, target_y)
            offset_info = f" [offset: +{eff_y_offset}px Y]" if eff_y_offset != 0 else ""
            _log(f"  [SIM DETECTED] Clicking {sim_label} at ({cx}, {cy}){offset_info} (detected at ({detected_pos[0]}, {detected_pos[1]}))...")
            self.status_message = f"{prefix} Clicked {sim_label} ({cx}, {cy})"
            if pydirectinput:
                pydirectinput.click()
                time.sleep(0.08)
                pydirectinput.mouseUp(button="left")
            sims_clicked.append(sim_key)
            self._record_sim_clicked(sim_key, room_key=getattr(self, "current_room_key", None))

            # Approach wait: allow character to move closer to the sim object before proceeding
            if eff_app_wait > 0:
                self._wait_for_approach(eff_app_wait, reason=f"{prefix} {sim_label}")

            # Double-check / Verification: Check after approach/wait if SIM is still visible on screen
            if eff_verify and not stop_handler.is_stopped() and self.is_active:
                if eff_app_wait <= 0 and eff_verify_delay > 0:
                    time.sleep(eff_verify_delay)

                for attempt_num in range(1, max(1, eff_max_attempts)):
                    if stop_handler.is_stopped() or not self.is_active:
                        break
                    recheck_pos = self.locate_sim_template(sim_key) if threshold is None else self.locate_sim_template(sim_key, threshold)
                    if recheck_pos is not None:
                        rx, ry = self.move_mouse_inside_game(
                            recheck_pos[0] + eff_x_offset,
                            recheck_pos[1] + eff_y_offset
                        )
                        _log(f"  [SIM DOUBLE-CHECK] {sim_label} is STILL visible on screen after approach. Re-clicking in-range at ({rx}, {ry}) (attempt {attempt_num + 1}/{eff_max_attempts})...")
                        self.status_message = f"{prefix} Re-clicking {sim_label} ({rx}, {ry})"
                        if pydirectinput:
                            pydirectinput.click()
                            time.sleep(0.08)
                            pydirectinput.mouseUp(button="left")
                        time.sleep(max(0.4, eff_verify_delay))
                    else:
                        _log(f"  [SIM VERIFIED] {sim_label} click confirmed (no longer visible on screen).")
                        break
            elif not stop_handler.is_stopped() and self.is_active and self.reclick_after_approach:
                in_range_pos = self.locate_sim_template(sim_key) if threshold is None else self.locate_sim_template(sim_key, threshold)
                if in_range_pos is not None:
                    rx, ry = self.move_mouse_inside_game(
                        in_range_pos[0] + eff_x_offset,
                        in_range_pos[1] + eff_y_offset
                    )
                    _log(f"  [SIM IN-RANGE] Re-clicking {sim_label} in-range at ({rx}, {ry})...")
                    if pydirectinput:
                        pydirectinput.click()
                        time.sleep(0.08)
                        pydirectinput.mouseUp(button="left")
                    time.sleep(0.15)

            return cx, cy

        for sim_key in effective_order:
            if stop_handler.is_stopped() or not self.is_active:
                return sims_clicked
            if eff_max_clicks is not None and len(sims_clicked) >= eff_max_clicks:
                break
            sim_label = f"Sim {sim_key.replace('sim', '')}"
            self.status_message = f"{prefix} Checking for {sim_label}..."
            sim_pos = None
            best_conf = 0.0
            curr_attempts = search_attempts if not sims_clicked else subsequent_search_attempts
            for attempt in range(1, max(1, curr_attempts) + 1):
                if stop_handler.is_stopped() or not self.is_active:
                    return sims_clicked
                sim_pos = self.locate_sim_template(sim_key) if threshold is None else self.locate_sim_template(sim_key, threshold)
                best_conf = max(best_conf, getattr(self, "_last_sim_best_conf", 0.0))
                if sim_pos is not None:
                    break
                if attempt < curr_attempts:
                    time.sleep(0.08)

            if sim_pos is not None:
                _click_sim_at(sim_key, sim_label, sim_pos)
                if eff_max_clicks is not None and len(sims_clicked) >= eff_max_clicks:
                    _log(f"  {prefix} Desired sim selection completed ({len(sims_clicked)}/{eff_max_clicks}). Proceeding without delay...")
                    break
                w_start = time.time()
                while (time.time() - w_start) < 0.25:
                    if stop_handler.is_stopped() or not self.is_active:
                        return sims_clicked
                    time.sleep(0.05)
            else:
                eff_thresh = threshold if threshold is not None else self.sim_match_threshold
                _log(f"  [SIM] {sim_label} not detected (best conf={best_conf:.2f}, threshold={eff_thresh:.2f})")

        if not sims_clicked:
            _log(f"  {prefix} No sim banners detected. Proceeding with routine...")
        self.hide_loot_labels()
        return sims_clicked


    def _wait_for_approach(self, max_wait_seconds: float, reason: str = "Approach") -> None:
        """
        Waits for the character to move closer towards an interactable target (banner or sim).
        Monitors character position to detect when character movement begins and settles (stops moving for >= 0.4s).
        Keeps self.is_interacting = True throughout so anti-stuck is not triggered while approaching.
        """
        if max_wait_seconds <= 0:
            return

        self.is_approaching_interactable = True
        try:
            start_time = time.time()
            start_pos = self.latest_pos
            has_moved = False
            last_move_time = start_time
            last_check_pos = start_pos

            _log(f"  [{reason}] Character approaching target (allowing up to {max_wait_seconds:.1f}s)...")

            max_ticks = max(10, int(max_wait_seconds / 0.05) + 5)
            tick = 0

            while (time.time() - start_time) < max_wait_seconds and tick < max_ticks:
                tick += 1
                if stop_handler.is_stopped() or not self.is_active:
                    return

                self._trigger_persistent_combat_if_due()
                curr_pos = self.latest_pos
                now = time.time()

                if curr_pos is not None and last_check_pos is not None:
                    dist_from_last = math.hypot(curr_pos[0] - last_check_pos[0], curr_pos[1] - last_check_pos[1])
                    if dist_from_last > 1.5:
                        has_moved = True
                        last_move_time = now
                        last_check_pos = curr_pos
                    elif has_moved and (now - last_move_time) >= 0.4:
                        _log(f"  [{reason}] Character reached target and settled ({now - start_time:.2f}s).")
                        break

                rem = max(0.0, max_wait_seconds - (now - start_time))
                self.status_message = f"[{reason}] Approaching ({rem:.1f}s)..."
                time.sleep(0.05)

            elapsed = time.time() - start_time
            _log(f"  [{reason}] Approach window finished ({elapsed:.2f}s).")
        finally:
            self.is_approaching_interactable = False

