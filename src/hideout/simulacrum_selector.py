"""
Simulacrum map node detection, popup detection, and accessible map selection.
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


class HideoutSimulacrumMixin:
    """Simulacrum map node detection, popup detection, and accessible map selection."""

    def detect_simulacrum_map_nodes(
        self,
        screen: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        nms_distance: float = 28.0,
    ) -> List[Dict[str, Any]]:
        """
        Scans the active Atlas map screen for Simulacrum nodes using multi-scale template matching.
        Uses simulacrum_node_v1.png, simulacrum_node_v2.png, simulacrum_node_v3.png,
        simulacrum_medal.png, simulacrum_icon.png, and simulacrum_node_full.png.
        For each detected medal/icon at (cx, cy), computes the exact map circle coordinate:
            circle_x = cx
            circle_y = cy + simulacrum_click_y_offset * scale (default +26px)
        Evaluates accessibility to verify if linked by stripped/dashed line to ANY green location.
        Returns a sorted list of candidate nodes with accessible nodes prioritized first!
        """
        if (
            self.simulacrum_icon_tpl is None
            and self.simulacrum_medal_tpl is None
            and self.simulacrum_node_tpl is None
            and getattr(self, "simulacrum_node_v1_tpl", None) is None
        ):
            self._load_stash_and_inventory_templates()

        templates_to_try = []
        for i in range(1, 8):
            tpl = getattr(self, f"simulacrum_node_v{i}_tpl", None)
            if tpl is not None:
                templates_to_try.append((f"node_v{i}", tpl, 0, 0))
        if self.simulacrum_medal_tpl is not None:
            templates_to_try.append(("medal", self.simulacrum_medal_tpl, 0, 0))
        if self.simulacrum_icon_tpl is not None:
            templates_to_try.append(("icon", self.simulacrum_icon_tpl, 0, 0))
        if self.simulacrum_node_tpl is not None:
            templates_to_try.append(("node_full", self.simulacrum_node_tpl, 0, -10))

        if not templates_to_try:
            return []

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return []

        eff_thresh = threshold if threshold is not None else max(0.75, self.simulacrum_match_threshold)

        mon_left = 0
        mon_top = 0
        capt = self._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        sh, sw = screen.shape[:2]
        scale_est = sh / 1080.0
        scales = sorted(set([
            round(scale_est * f, 3) for f in [0.92, 0.96, 1.0, 1.04, 1.08]
        ]))

        # Check if inventory panel is visible on right to constrain map search area
        inv_open = False
        try:
            inv_open = self.is_inventory_open(screen=screen)
        except Exception:
            inv_open = False
        max_map_x = int(0.655 * sw) if inv_open else int(0.68 * sw)

        raw_candidates = []

        for tname, tpl, x_off, y_off in templates_to_try:
            th, tw = tpl.shape[:2]
            for s in scales:
                scaled_w = int(tw * s)
                scaled_h = int(th * s)
                if sh < scaled_h or sw < scaled_w:
                    continue
                scaled_tpl = tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                    tpl, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                )
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= eff_thresh)
                for pt_y, pt_x in zip(locs[0], locs[1]):
                    conf = float(res[pt_y, pt_x])
                    cx = int(pt_x + scaled_w // 2 + x_off * s)
                    cy = int(pt_y + (scaled_h // 2 if tname != "node_full" else 17 * s) + y_off * s)

                    # Exclude HUD, inventory area, and borders
                    if cx > max_map_x or cx < int(0.12 * sw) or cy < int(0.08 * sh) or cy > int(0.85 * sh):
                        continue

                    raw_candidates.append({
                        "cx": cx,
                        "cy": cy,
                        "conf": conf,
                        "scale": s,
                        "template": tname,
                    })

        if not raw_candidates:
            return []

        # Non-Maximum Suppression (NMS) to merge overlapping detections
        raw_candidates.sort(key=lambda c: c["conf"], reverse=True)
        filtered = []

        for cand in raw_candidates:
            cx, cy = cand["cx"], cand["cy"]
            eff_dist = nms_distance * cand["scale"]
            is_dup = False
            for kept in filtered:
                dist = math.hypot(cx - kept["cx"], cy - kept["cy"])
                if dist < eff_dist:
                    is_dup = True
                    break
            if not is_dup:
                filtered.append(cand)

        # Detect green completed map nodes on the Atlas map
        green_nodes = self.detect_green_completed_nodes(screen=screen)

        results = []
        screen_center = (sw // 2, sh // 2)

        for f in filtered:
            s = f["scale"]
            medal_x = f["cx"]
            medal_y = f["cy"]
            # Circle is located below the medal at delta_y = +simulacrum_click_y_offset * scale (default +26px)
            circle_x = medal_x
            circle_y = int(round(medal_y + getattr(self, "simulacrum_click_y_offset", 26.0) * s))

            desktop_medal_x = medal_x + mon_left
            desktop_medal_y = medal_y + mon_top
            desktop_circle_x = circle_x + mon_left
            desktop_circle_y = circle_y + mon_top

            dist_from_center = math.hypot(circle_x - screen_center[0], circle_y - screen_center[1])

            # Check accessibility: connected to green location via stripped line and/or blue accessible glow
            acc_info = self.check_node_accessibility((circle_x, circle_y), green_nodes, screen=screen, medal_pos=(medal_x, medal_y))

            results.append({
                "screen_medal_pos": (medal_x, medal_y),
                "screen_circle_pos": (circle_x, circle_y),
                "medal_pos": (desktop_medal_x, desktop_medal_y),
                "circle_pos": (desktop_circle_x, desktop_circle_y),
                "confidence": f["conf"],
                "scale": s,
                "template": f["template"],
                "dist_from_center": dist_from_center,
                "is_accessible": acc_info["is_accessible"],
                "acc_score": acc_info.get("acc_score", 0),
                "connected_greens": acc_info["connected_greens"],
                "has_blue_glow": acc_info["has_blue_glow"],
                "has_white_core": acc_info["has_white_core"],
            })

        # Sort candidates:
        # 1. Highest accessibility score (connected to green & blue portal first)
        # 2. Highest template matching confidence
        # 3. Distance from screen center
        results.sort(key=lambda r: (-r["acc_score"], -r["confidence"], r["dist_from_center"]))

        _log(f"  [SIMULACRUM DETECT] Found {len(results)} distinct Simulacrum node(s) on screen (thresh={eff_thresh:.2f}, green_nodes={len(green_nodes)}):")
        for idx, r in enumerate(results):
            score = r.get("acc_score", 0)
            if score == 3:
                acc_str = f"ACCESSIBLE (Connected to {len(r['connected_greens'])} green node(s) & glowing blue portal)"
            elif score == 2:
                acc_str = f"ACCESSIBLE (Connected to {len(r['connected_greens'])} green node(s))"
            elif score == 1:
                acc_str = "ACCESSIBLE (Glowing blue portal core)"
            else:
                acc_str = "INACCESSIBLE (Not connected to any green node)"
            _log(f"    - Node #{idx+1}: medal={r['screen_medal_pos']}, circle={r['screen_circle_pos']}, conf={r['confidence']:.3f}, scale={r['scale']:.2f} [{acc_str}]")

        return results

        return results


    def is_simulacrum_popup_visible(
        self,
        screen: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        save_debug: bool = True,
    ) -> bool:
        """
        Detects whether the 'SIMULACRUM OF DELUSION' popup window is currently open and
        AVAILABLE to the player (i.e. contains the 2x2 square split in 4 empty spaces for maps
        and/or the TRAVERSE button).

        If ONLY the 'SIMULACRUM OF DELUSION' title banner is visible without the 4-square
        split below, it means this particular map node is NOT available to the player from here,
        so this function returns False, allowing the caller to iterate to the next map circle.
        """
        if self.delusion_popup_tpl is None and getattr(self, "delusion_4square_tpl", None) is None:
            self._load_stash_and_inventory_templates()

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return False

        eff_thresh = threshold if threshold is not None else self.delusion_popup_match_threshold
        sh, sw = screen.shape[:2]
        scale_est = (sh / 1080.0) if sh >= 720 else 1.0
        scales = [round(scale_est * f, 3) for f in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]]

        mon_left = 0
        mon_top = 0
        capt = self._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                mon_left = monitors[self.monitor_idx].get("left", 0)
                mon_top = monitors[self.monitor_idx].get("top", 0)

        best_device_val = -1.0
        best_device_loc = None
        best_device_scale = 1.0

        # Fast-path: Check scale 1.0 on delusion_4square_tpl or delusion_traverse_tpl
        if getattr(self, "delusion_4square_tpl", None) is not None:
            res_fast = cv2.matchTemplate(screen, self.delusion_4square_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_fast, _, max_loc_fast = cv2.minMaxLoc(res_fast)
            if max_fast >= max(0.65, eff_thresh):
                lx, ly = max_loc_fast
                self.delusion_detected_slots = [
                    (lx + int(63) + mon_left, ly + int(63) + mon_top),
                    (lx + int(131) + mon_left, ly + int(63) + mon_top),
                    (lx + int(63) + mon_left, ly + int(131) + mon_top),
                    (lx + int(131) + mon_left, ly + int(131) + mon_top),
                ]
                self.delusion_detected_traverse = (lx + int(95) + mon_left, ly + int(247) + mon_top)
                _log(f"  [DELUSION POPUP SUCCESS] Confirmed 'SIMULACRUM OF DELUSION' popup visible with 4-square slots (fast conf={max_fast:.3f})!")
                _log("  [DELUSION POPUP SUCCESS] Map IS AVAILABLE for player! 4 empty slots detected.")
                if save_debug:
                    os.makedirs(self.inventory_screenshots_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_path = os.path.join(self.inventory_screenshots_dir, f"delusion_popup_available_{ts}.png")
                    cv2.imwrite(debug_path, screen)
                return True

        if getattr(self, "delusion_traverse_tpl", None) is not None:
            res_trav = cv2.matchTemplate(screen, self.delusion_traverse_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_trav, _, max_loc_trav = cv2.minMaxLoc(res_trav)
            if max_trav >= 0.70:
                tx, ty = max_loc_trav
                th, tw = self.delusion_traverse_tpl.shape[:2]
                self.delusion_detected_traverse = (tx + tw // 2 + mon_left, ty + th // 2 + mon_top)
                lx = tx - 30
                ly = ty - 222
                self.delusion_detected_slots = [
                    (lx + int(63) + mon_left, ly + int(63) + mon_top),
                    (lx + int(131) + mon_left, ly + int(63) + mon_top),
                    (lx + int(63) + mon_left, ly + int(131) + mon_top),
                    (lx + int(131) + mon_left, ly + int(131) + mon_top),
                ]
                _log(f"  [DELUSION POPUP SUCCESS] Confirmed 'SIMULACRUM OF DELUSION' popup visible via TRAVERSE button (fast conf={max_trav:.3f})!")
                _log("  [DELUSION POPUP SUCCESS] Map IS AVAILABLE for player!")
                if save_debug:
                    os.makedirs(self.inventory_screenshots_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_path = os.path.join(self.inventory_screenshots_dir, f"delusion_popup_available_{ts}.png")
                    cv2.imwrite(debug_path, screen)
                return True

        # 1. Match against delusion_popup_tpl (full popup: banner + 4-square grid + traverse)
        if self.delusion_popup_tpl is not None:
            ph, pw = self.delusion_popup_tpl.shape[:2]
            for s in scales:
                tw = int(pw * s)
                th = int(ph * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.delusion_popup_tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                    self.delusion_popup_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                )
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_device_val:
                    best_device_val = max_val
                    best_device_loc = max_loc
                    best_device_scale = s

        # 2. Match against 4-square slots template (the 2x2 empty map spaces grid)
        best_4sq_val = -1.0
        best_4sq_loc = None
        best_4sq_scale = 1.0
        if getattr(self, "delusion_4square_tpl", None) is not None:
            qh, qw = self.delusion_4square_tpl.shape[:2]
            for s in scales:
                tw = int(qw * s)
                th = int(qh * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.delusion_4square_tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                    self.delusion_4square_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                )
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_4sq_val:
                    best_4sq_val = max_val
                    best_4sq_loc = max_loc
                    best_4sq_scale = s

        # 3. Match against TRAVERSE button template
        best_trav_val = -1.0
        best_trav_loc = None
        best_trav_scale = 1.0
        if getattr(self, "delusion_traverse_tpl", None) is not None:
            th_btn, tw_btn = self.delusion_traverse_tpl.shape[:2]
            for s in scales:
                tw = int(tw_btn * s)
                th = int(th_btn * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.delusion_traverse_tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                    self.delusion_traverse_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                )
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_trav_val:
                    best_trav_val = max_val
                    best_trav_loc = max_loc
                    best_trav_scale = s

        # Has the 4-square map device or traverse button been confirmed?
        has_device = (best_device_val >= eff_thresh) or (best_4sq_val >= eff_thresh) or (best_trav_val >= max(0.55, eff_thresh))

        if has_device:
            # Map is available! Compute exact slot pixel coordinates
            if best_device_val >= eff_thresh and best_device_loc is not None:
                s = best_device_scale
                lx, ly = best_device_loc
                self.delusion_detected_slots = [
                    (lx + int(173 * s) + mon_left, ly + int(153 * s) + mon_top),
                    (lx + int(241 * s) + mon_left, ly + int(153 * s) + mon_top),
                    (lx + int(173 * s) + mon_left, ly + int(221 * s) + mon_top),
                    (lx + int(241 * s) + mon_left, ly + int(221 * s) + mon_top),
                ]
                self.delusion_detected_traverse = (lx + int(207 * s) + mon_left, ly + int(337 * s) + mon_top)
            elif best_4sq_val >= eff_thresh and best_4sq_loc is not None:
                s = best_4sq_scale
                lx, ly = best_4sq_loc
                self.delusion_detected_slots = [
                    (lx + int(63 * s) + mon_left, ly + int(63 * s) + mon_top),
                    (lx + int(131 * s) + mon_left, ly + int(63 * s) + mon_top),
                    (lx + int(63 * s) + mon_left, ly + int(131 * s) + mon_top),
                    (lx + int(131 * s) + mon_left, ly + int(131 * s) + mon_top),
                ]
                self.delusion_detected_traverse = (lx + int(97 * s) + mon_left, ly + int(247 * s) + mon_top)

            # If TRAVERSE button was directly detected on screen, prioritize its exact coordinate
            if best_trav_val >= max(0.55, eff_thresh) and best_trav_loc is not None and getattr(self, "delusion_traverse_tpl", None) is not None:
                th_btn, tw_btn = self.delusion_traverse_tpl.shape[:2]
                s = best_trav_scale
                tw = int(tw_btn * s)
                th = int(th_btn * s)
                self.delusion_detected_traverse = (best_trav_loc[0] + tw // 2 + mon_left, best_trav_loc[1] + th // 2 + mon_top)

            conf_str = f"popup={best_device_val:.3f}, 4sq={best_4sq_val:.3f}, trav={best_trav_val:.3f}"
            _log(f"  [DELUSION POPUP SUCCESS] Confirmed 'SIMULACRUM OF DELUSION' popup visible with 4-square slots ({conf_str})!")
            _log("  [DELUSION POPUP SUCCESS] Map IS AVAILABLE for player! 4 empty slots detected.")
            if save_debug:
                os.makedirs(self.inventory_screenshots_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = os.path.join(self.inventory_screenshots_dir, f"delusion_popup_available_{ts}.png")
                cv2.imwrite(debug_path, screen)
            return True

        # 4. Check if ONLY the title banner is visible without the 4-square device below it
        best_title_val = -1.0
        if getattr(self, "delusion_title_tpl", None) is not None:
            bh, bw = self.delusion_title_tpl.shape[:2]
            for s in scales:
                tw = int(bw * s)
                th = int(bh * s)
                if sh < th or sw < tw:
                    continue
                scaled_tpl = self.delusion_title_tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                    self.delusion_title_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                )
                res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_title_val:
                    best_title_val = max_val

        if best_title_val >= eff_thresh:
            _log(f"  [DELUSION POPUP UNAVAILABLE] 'SIMULACRUM OF DELUSION' title banner is visible (conf={best_title_val:.3f} >= {eff_thresh:.2f}), BUT the square split in 4 below is NOT present!")
            _log("  [DELUSION POPUP UNAVAILABLE] This particular map is NOT accessible for the player from here.")
            return False

        # 5. Heuristic fallback when templates are missing (e.g., bare test environment)
        if self.delusion_popup_tpl is None and getattr(self, "delusion_4square_tpl", None) is None:
            center_roi = screen[int(sh * 0.20):int(sh * 0.75), int(sw * 0.25):int(sw * 0.75)]
            if center_roi.size > 0:
                gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = float(np.count_nonzero(edges)) / float(edges.size)
                has_dialog_structure = bool(0.035 <= edge_density <= 0.25)
                if has_dialog_structure:
                    hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
                    gold_mask = cv2.inRange(hsv, np.array([12, 70, 70]), np.array([35, 255, 255]))
                    gold_px = np.count_nonzero(gold_mask)
                    if gold_px > int(80 * scale_est):
                        _log(f"  [DELUSION POPUP HEURISTIC] Detected central dialog with golden banner (gold_px={gold_px}, edge_density={edge_density:.3f}).")
                        return True

        return False


    def select_accessible_simulacrum_map(
        self,
        max_attempts: int = 6,
        candidates: Optional[List[Dict[str, Any]]] = None,
        screen: Optional[np.ndarray] = None,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Iterates through detected Simulacrum map circles:
        1. Dynamically detects candidate nodes on the live Atlas screen.
        2. Prioritizes accessible candidates linked to green locations and glowing blue portals.
        3. Clicks the candidate node with a rock-solid, fixed mouse-down/up sequence to prevent map panning.
        4. Polls for up to 1.5s to verify if the 'SIMULACRUM OF DELUSION' popup is visible.
        5. If the popup does NOT open (e.g. map slightly moved or clicked slightly low), immediately captures
           a fresh screen, re-detects the live candidate coordinates, and retries with vertical target adjustment.
        6. Proceeds across candidate nodes until the Delusion popup is confirmed open.
        """
        passed_explicit_candidates = bool(candidates is not None and len(candidates) > 0)
        self.status_message = "Finding Simulacrum Map..."
        _log("\n[SIMULACRUM SELECTION] Searching for accessible Simulacrum map on Atlas...")

        capt = self._get_capturer()

        if candidates is None:
            poll_start = time.time()
            while (time.time() - poll_start) < 3.5:
                if stop_handler.is_stopped():
                    return None
                candidates = self.detect_simulacrum_map_nodes(screen=screen)
                if candidates:
                    break
                time.sleep(0.4)
                screen = None

        if not candidates:
            eff_thresh = getattr(self, "simulacrum_match_threshold", 0.78)
            relaxed_thresh = max(0.72, eff_thresh - 0.08)
            _log(f"  [SIMULACRUM RETRY] Retrying detection with safe relaxed threshold {relaxed_thresh:.2f}...")
            candidates = self.detect_simulacrum_map_nodes(screen=None, threshold=relaxed_thresh)

        if not candidates:
            _log("  [SIMULACRUM WARNING] No Simulacrum map nodes found on screen.")
            return None

        # Prioritize candidates confirmed accessible via connection to a GREEN location
        if getattr(self, "sim_require_green_connection", True):
            accessible_candidates = [c for c in candidates if c.get("is_accessible", False)]
            if accessible_candidates:
                _log(f"  [SIMULACRUM FILTER] Found {len(accessible_candidates)} accessible Simulacrum map(s) linked to GREEN location(s). Prioritizing strictly accessible nodes!")
                candidates = accessible_candidates
            else:
                _log("  [SIMULACRUM FILTER WARNING] None of the detected Simulacrum nodes are connected to a GREEN location. Testing detected nodes as fallback...")

        _log(f"  [SIMULACRUM SELECTION] Found {len(candidates)} candidate node(s). Testing accessibility with fresh-capture retry...")

        failed_nodes: List[Tuple[int, int]] = []

        for idx in range(max_attempts):
            if stop_handler.is_stopped():
                return None

            # On dynamic attempts (when candidates weren't manually passed by caller):
            # Take a fresh screen capture on EVERY attempt after idx > 0 so that if the map
            # slightly moved or panned during the click, all coordinates are freshly resynchronized!
            if idx > 0 and not passed_explicit_candidates:
                time.sleep(0.3)
                fresh_screen = None
                try:
                    fresh_screen = capt.capture() if capt else None
                except Exception:
                    pass
                if fresh_screen is not None and fresh_screen.size > 0:
                    fresh_candidates = self.detect_simulacrum_map_nodes(screen=fresh_screen)
                    if fresh_candidates:
                        if getattr(self, "sim_require_green_connection", True):
                            fresh_acc = [c for c in fresh_candidates if c.get("is_accessible", False)]
                            if fresh_acc:
                                fresh_candidates = fresh_acc
                        candidates = fresh_candidates
                        _log(f"  [SIMULACRUM RESYNC] Captured fresh screen: detected {len(candidates)} live node(s) on Atlas map.")

            if not candidates:
                _log("  [SIMULACRUM EXHAUSTED] No candidate circles detected on screen.")
                break

            # Pick which candidate to click:
            cand_idx = 0
            if passed_explicit_candidates:
                cand_idx = idx if idx < len(candidates) else len(candidates) - 1
            else:
                # Find the first candidate that hasn't exceeded 2 failed attempts
                chosen_cand_idx = 0
                for c_i, c in enumerate(candidates):
                    c_pt = c["screen_circle_pos"]
                    attempts_for_c = sum(1 for fp in failed_nodes if math.hypot(c_pt[0] - fp[0], c_pt[1] - fp[1]) < 45)
                    if attempts_for_c < 2:
                        chosen_cand_idx = c_i
                        break
                else:
                    chosen_cand_idx = idx % len(candidates)
                cand_idx = chosen_cand_idx

            node = candidates[cand_idx]
            circ_x, circ_y = node["circle_pos"]
            screen_x, screen_y = node["screen_circle_pos"]
            medal_x, medal_y = node["medal_pos"]
            acc_score = node.get("acc_score", 0)
            acc_tag = "[ACCESSIBLE - GREEN LINK]" if acc_score >= 2 or node.get("is_accessible") else "[UNCONFIRMED / INACCESSIBLE]"

            # Check if this node was already tried once: if so, retry by clicking slightly higher on the medal icon
            past_attempts_here = sum(1 for fp in failed_nodes if math.hypot(screen_x - fp[0], screen_y - fp[1]) < 45)
            target_x = circ_x
            target_y = circ_y
            if past_attempts_here >= 1:
                # Retry: click directly on the medal icon
                target_x = medal_x
                target_y = medal_y + int(5 * node.get("scale", 1.0))
                _log(f"\n  [SIMULACRUM ATTEMPT {idx+1}/{max_attempts}] Retrying {acc_tag} node #{cand_idx+1} at medal icon desktop ({target_x}, {target_y})...")
            else:
                _log(f"\n  [SIMULACRUM ATTEMPT {idx+1}/{max_attempts}] Clicking {acc_tag} node #{cand_idx+1} map circle at desktop ({target_x}, {target_y}) [screen ({screen_x}, {screen_y})]...")

            self.status_message = f"Testing Simulacrum Map #{idx+1}..."

            if dry_run:
                _log(f"    [DRY RUN] Would click circle #{idx+1} at ({target_x}, {target_y}).")
                return node

            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            rx, ry = self.move_mouse_inside_game(target_x, target_y)
            time.sleep(0.15)
            if pydirectinput:
                pydirectinput.mouseDown(button="left")
                time.sleep(0.08)
                pydirectinput.mouseUp(button="left")
            time.sleep(0.40)

            # Check if popup is visible
            verify_screen = None
            try:
                verify_screen = capt.capture() if capt else None
            except Exception:
                pass

            popup_visible = self.is_simulacrum_popup_visible(screen=verify_screen, save_debug=False)
            if not popup_visible and not passed_explicit_candidates:
                poll_popup_start = time.time()
                while (time.time() - poll_popup_start) < 1.4:
                    if stop_handler.is_stopped():
                        return None
                    time.sleep(0.2)
                    try:
                        verify_screen = capt.capture() if capt else None
                    except Exception:
                        pass
                    if self.is_simulacrum_popup_visible(screen=verify_screen, save_debug=False):
                        popup_visible = True
                        break

            # Save debug screenshot for this attempt
            if verify_screen is not None and getattr(verify_screen, "size", 0) > 0:
                os.makedirs(self.inventory_screenshots_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_p = os.path.join(self.inventory_screenshots_dir, f"sim_circle_attempt_{idx+1}_{'OPEN' if popup_visible else 'CLOSED'}_{ts}.png")
                cv2.imwrite(debug_p, verify_screen)

            if popup_visible:
                self._last_selected_sim_circle = (target_x, target_y)
                _log(f"  [SIMULACRUM SUCCESS] 'SIMULACRUM OF DELUSION' popup CONFIRMED OPEN on candidate #{cand_idx+1} (attempt {idx+1})!")
                self.status_message = "Simulacrum of Delusion Open!"
                return node
            else:
                _log(f"  [SIMULACRUM RETRY] Popup NOT visible after attempt #{idx+1}. Capturing fresh screen and retrying live coordinates...")
                failed_nodes.append((screen_x, screen_y))
                time.sleep(0.3)

        _log("  [SIMULACRUM EXHAUSTED] None of the tested Simulacrum circles opened the Delusion popup.")
        return None

