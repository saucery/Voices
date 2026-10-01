"""
Map insertion into Simulacrum popup, Traverse button clicking, and hideout portal interaction.
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


class HideoutTraverseMixin:
    """Map insertion into Simulacrum popup, Traverse button clicking, and hideout portal interaction."""

    def _get_monitor_offsets(self) -> Tuple[int, int]:
        capt = self._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= self.monitor_idx < len(monitors):
                return monitors[self.monitor_idx].get("left", 0), monitors[self.monitor_idx].get("top", 0)
        return 0, 0


    def locate_tier15_maps_in_inventory(
        self,
        screen: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scans columns 10, 11, and 12 (0-indexed 9, 10, 11) of the player's open inventory
        for Tier 15 maps (matching templates/ui/tier15_map.png or non-empty slot).
        Returns a list of detected map items with row, column, desktop (x, y) coordinates, and confidence.
        """
        if self.tier15_map_tpl is None:
            self._load_stash_and_inventory_templates()

        if screen is None:
            capt = self._get_capturer()
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return []

        inv_info = self.locate_inventory_window(screen=screen)
        if inv_info is None:
            _log("  [MAP SEARCH WARNING] Inventory window not detected on screen.")
            return []

        row_centers = inv_info["row_centers"]
        col_centers = inv_info["col_centers"]
        scale = inv_info["scale"]
        mon_left, mon_top = self._get_monitor_offsets()

        sh, sw = screen.shape[:2]
        eff_thresh = threshold if threshold is not None else self.tier15_map_match_threshold

        map_items = []
        target_cols = [9, 10, 11]  # Columns 10, 11, 12
        half_box = max(10, int(22 * scale))

        for r, cy in enumerate(row_centers):
            for c in target_cols:
                if c >= len(col_centers):
                    continue
                cx = col_centers[c]
                cx_i = int(round(cx))
                cy_i = int(round(cy))

                if cy_i - half_box < 0 or cy_i + half_box > sh or cx_i - half_box < 0 or cx_i + half_box > sw:
                    continue

                patch = screen[cy_i - half_box : cy_i + half_box, cx_i - half_box : cx_i + half_box]
                desktop_x = cx_i + mon_left
                desktop_y = cy_i + mon_top

                best_conf = 0.0
                if self.tier15_map_tpl is not None and patch.shape[0] >= 20 and patch.shape[1] >= 20:
                    mh, mw = self.tier15_map_tpl.shape[:2]
                    scaled_mw = max(10, int(mw * scale))
                    scaled_mh = max(10, int(mh * scale))
                    if patch.shape[0] >= scaled_mh and patch.shape[1] >= scaled_mw:
                        scaled_tpl = self.tier15_map_tpl if abs(scale - 1.0) < 0.01 else cv2.resize(
                            self.tier15_map_tpl, (scaled_mw, scaled_mh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                        )
                        res = cv2.matchTemplate(patch, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_v, _, _ = cv2.minMaxLoc(res)
                        best_conf = float(max_v)

                is_map = best_conf >= eff_thresh or (best_conf >= 0.40 and patch.mean() > 20.0) or (patch.mean() > 25.0 and patch.max() > 70)

                if is_map:
                    map_items.append({
                        "row": r,
                        "col": c,
                        "screen_pos": (cx_i, cy_i),
                        "desktop_pos": (desktop_x, desktop_y),
                        "confidence": best_conf,
                    })

        _log(f"  [MAP INVENTORY] Found {len(map_items)} Tier 15 map(s) in inventory columns 10..12.")
        for m in map_items:
            _log(f"    - Map at Row {m['row']+1}, Col {m['col']+1} (screen: {m['screen_pos']}, conf={m['confidence']:.3f})")

        return map_items


    def insert_map_into_simulacrum_popup(
        self,
        screen: Optional[np.ndarray] = None,
        target_slot_idx: int = 0,
        method: str = "drag",
        dry_run: bool = False,
    ) -> bool:
        """
        Transfers a Tier 15 map from one of the fields in the last 3 columns of inventory
        into one of the 4 available fields in the 'Simulacrum of Delusion' popup.
        Supports method='drag' (mouse down, drag, mouse up), method='click' (pick up & place),
        and method='ctrl_click' (quick transfer).
        """
        self.status_message = "Transferring Map to Delusion Popup..."
        _log(f"\n[MAP TRANSFER] Transferring Tier 15 map from inventory to Simulacrum of Delusion popup (method='{method}')...")

        capt = self._get_capturer()
        if screen is None:
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            _log("  [MAP TRANSFER WARNING] Unable to capture screen.")
            return False

        sh, sw = screen.shape[:2]
        scale_est = sh / 1080.0
        mon_left, mon_top = self._get_monitor_offsets()

        # 1. Ensure inventory is open
        # NOTE: In PoE, when the Atlas / Simulacrum popup is open, the inventory is already open
        # side-by-side on the right. Pressing 'I' while the Delusion popup is open would CLOSE it!
        maps = self.locate_tier15_maps_in_inventory(screen=screen)
        popup_already_open = self.is_simulacrum_popup_visible(screen=screen, save_debug=False)
        if not maps and not self.is_inventory_open(screen=screen) and not popup_already_open:
            _log("  [MAP TRANSFER] Inventory closed and Delusion popup not open. Pressing 'I' to open inventory...")
            if not dry_run and pydirectinput:
                window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
                pydirectinput.press("i")
                time.sleep(0.40)
            try:
                screen = capt.capture()
            except Exception:
                pass
            maps = self.locate_tier15_maps_in_inventory(screen=screen)

        # 2. Re-check for Tier 15 maps if not found on first pass (poll up to 2.0s for UI render)
        if not maps and not dry_run:
            poll_map_start = time.time()
            while (time.time() - poll_map_start) < 2.0:
                if stop_handler.is_stopped():
                    return False
                time.sleep(0.3)
                try:
                    screen = capt.capture()
                except Exception:
                    pass
                maps = self.locate_tier15_maps_in_inventory(screen=screen)
                if maps:
                    break

        if not maps:
            _log("  [MAP TRANSFER WARNING] No Tier 15 maps found in columns 10, 11, or 12!")
            return False

        src_map = maps[0]
        src_x, src_y = src_map["desktop_pos"]
        _log(f"  [MAP TRANSFER] Selected source map at Row {src_map['row']+1}, Col {src_map['col']+1} (desktop: {src_x}, {src_y}).")

        # 3. Ensure Delusion popup is open before transferring map; if closed, re-click the circle to reopen
        if not dry_run and not self.is_simulacrum_popup_visible(screen=screen, save_debug=False):
            last_circ = getattr(self, "_last_selected_sim_circle", None)
            if last_circ:
                lc_x, lc_y = last_circ
                _log(f"  [MAP TRANSFER RECOVER] Delusion popup closed. Re-clicking map circle at ({lc_x}, {lc_y}) to reopen...")
                self.move_mouse_inside_game(lc_x, lc_y)
                time.sleep(0.10)
                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.55)
                try:
                    screen = capt.capture()
                except Exception:
                    pass
                self.is_simulacrum_popup_visible(screen=screen, save_debug=False)

        # 4. Compute target slot position in the Simulacrum of Delusion popup
        if getattr(self, "delusion_detected_slots", None) and len(self.delusion_detected_slots) > (target_slot_idx % 4):
            dst_x, dst_y = self.delusion_detected_slots[target_slot_idx % 4]
            dst_screen_x = dst_x - mon_left
            dst_screen_y = dst_y - mon_top
            _log(f"  [MAP TRANSFER] Using detected 4-square popup Slot #{target_slot_idx+1} at desktop ({dst_x}, {dst_y}) [screen: ({dst_screen_x}, {dst_screen_y})].")
        else:
            sd = int(32 * scale_est)
            ox, oy = [(-sd, -sd), (sd, -sd), (-sd, sd), (sd, sd)][target_slot_idx % 4]
            dst_screen_x, dst_screen_y = sw // 2 + ox, int(sh * 0.48) + oy
            dst_x, dst_y = dst_screen_x + mon_left, dst_screen_y + mon_top
            _log(f"  [MAP TRANSFER] Target Delusion popup Slot #{target_slot_idx+1} at desktop ({dst_x}, {dst_y}) [screen: ({dst_screen_x}, {dst_screen_y})].")

        if dry_run:
            _log(f"  [DRY RUN] Would transfer map from ({src_x}, {src_y}) to popup slot ({dst_x}, {dst_y}) using method='{method}'.")
            return True

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

        # 4. Perform the transfer
        if method == "ctrl_click":
            _log(f"  [MAP TRANSFER] Quick-transferring via Ctrl+Click at ({src_x}, {src_y})...")
            self.move_mouse_inside_game(src_x, src_y)
            time.sleep(0.06)
            if pydirectinput:
                pydirectinput.keyDown("ctrl")
                time.sleep(0.06)
                pydirectinput.click()
                time.sleep(0.06)
                pydirectinput.keyUp("ctrl")
                time.sleep(0.15)
        elif method == "click":
            _log(f"  [MAP TRANSFER] Picking up map at ({src_x}, {src_y}) and placing into ({dst_x}, {dst_y})...")
            self.move_mouse_inside_game(src_x, src_y)
            time.sleep(0.08)
            if pydirectinput:
                pydirectinput.click()
                time.sleep(0.12)
            self.move_mouse_inside_game(dst_x, dst_y)
            time.sleep(0.08)
            if pydirectinput:
                pydirectinput.click()
                time.sleep(0.15)
        else:
            _log(f"  [MAP TRANSFER] Dragging map from ({src_x}, {src_y}) to ({dst_x}, {dst_y})...")
            self.move_mouse_inside_game(src_x, src_y)
            time.sleep(0.08)
            if pydirectinput:
                pydirectinput.mouseDown(button="left")
                time.sleep(0.10)
                self.move_mouse_inside_game((src_x + dst_x) // 2, (src_y + dst_y) // 2)
                time.sleep(0.05)
                self.move_mouse_inside_game(dst_x, dst_y)
                time.sleep(0.10)
                pydirectinput.mouseUp(button="left")
                time.sleep(0.20)

        _log("  [MAP TRANSFER SUCCESS] Map successfully transferred to Simulacrum of Delusion popup!")
        self.status_message = "Map Inserted into Delusion Popup!"
        return True


    def locate_traverse_button(
        self,
        screen: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the 'TRAVERSE' button at the bottom of the Simulacrum of Delusion popup.
        Returns desktop absolute (x, y) coordinates of the button center, or None if not found.
        """
        # If already detected and no custom screen was passed, return cached position
        if screen is None and getattr(self, "delusion_detected_traverse", None) is not None:
            return self.delusion_detected_traverse

        if getattr(self, "delusion_traverse_tpl", None) is None:
            self._load_stash_and_inventory_templates()
        if getattr(self, "delusion_traverse_tpl", None) is None:
            return None

        capt = self._get_capturer()
        if screen is None:
            try:
                screen = capt.capture()
            except Exception:
                screen = None
        if screen is None or screen.size == 0:
            return None

        sh, sw = screen.shape[:2]
        mon_left, mon_top = self._get_monitor_offsets()

        eff_thresh = threshold if threshold is not None else getattr(self, "traverse_match_threshold", 0.60)
        btn_h, btn_w = self.delusion_traverse_tpl.shape[:2]
        scales = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10]
        best_val = -1.0
        best_loc = None
        best_scale = 1.0

        for s in scales:
            tw = int(btn_w * s)
            th = int(btn_h * s)
            if sh < th or sw < tw:
                continue
            scaled_tpl = self.delusion_traverse_tpl if abs(s - 1.0) < 0.01 else cv2.resize(
                self.delusion_traverse_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            )
            res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = s

        if best_val >= eff_thresh and best_loc is not None:
            tw = int(btn_w * best_scale)
            th = int(btn_h * best_scale)
            cx = best_loc[0] + tw // 2
            cy = best_loc[1] + th // 2
            desktop_x = cx + mon_left
            desktop_y = cy + mon_top
            self.delusion_detected_traverse = (desktop_x, desktop_y)
            _log(f"  [TRAVERSE MATCH] Located TRAVERSE button (conf={best_val:.3f} >= {eff_thresh:.2f}, scale={best_scale:.2f}) at screen ({desktop_x}, {desktop_y})")
            return desktop_x, desktop_y

        # Fallback: check if is_simulacrum_popup_visible can locate it
        if self.is_simulacrum_popup_visible(screen=screen, save_debug=False):
            if getattr(self, "delusion_detected_traverse", None) is not None:
                return self.delusion_detected_traverse

        return None


    def click_traverse_button(
        self,
        timeout: float = 5.0,
        verify_close: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """
        Locates and left-clicks the 'TRAVERSE' button in the Simulacrum of Delusion popup.
        Once clicked, the popup and Atlas windows close automatically in Path of Exile.
        No Escape key is needed (Escape would open the game settings menu).
        Clears popup tracking state and prepares character for portal detection.
        """
        self.status_message = "Locating TRAVERSE Button..."
        _log("\n[TRAVERSE] Locating and clicking TRAVERSE button...")

        poll_start = time.time()
        trav_pos = None
        while (time.time() - poll_start) < timeout:
            if stop_handler.is_stopped():
                return False
            trav_pos = self.locate_traverse_button()
            if trav_pos is not None:
                break
            time.sleep(0.25)

        if trav_pos is None:
            _log(f"  [TRAVERSE WARNING] TRAVERSE button not detected on screen after {timeout:.1f}s.")
            return False

        tx, ty = trav_pos
        _log(f"  [TRAVERSE] Left-clicking TRAVERSE button at desktop ({tx}, {ty})...")
        self.status_message = "Clicking TRAVERSE..."

        if dry_run:
            _log(f"    [DRY RUN] Would click TRAVERSE button at ({tx}, {ty}).")
            self.delusion_detected_traverse, self.delusion_detected_slots = None, []
            return True

        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.move_mouse_inside_game(tx, ty)
        time.sleep(0.10)
        if pydirectinput:
            pydirectinput.click()
        time.sleep(0.55)

        # Clear popup state after clicking TRAVERSE
        self.delusion_detected_traverse, self.delusion_detected_slots = None, []

        # In PoE, clicking TRAVERSE automatically activates the map and closes all Atlas / Delusion popup dialogs.
        # No Escape key is needed (pressing Escape would open the PoE options menu).
        _log("  [TRAVERSE SUCCESS] TRAVERSE button clicked successfully. No Escape needed.")
        self.status_message = "TRAVERSE Clicked - Spawning Portals..."
        return True


    def locate_hideout_portal(
        self,
        screen: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        preferred_pos: Optional[Tuple[int, int]] = None,
        allow_completed: bool = False,
        click_target: Optional[str] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the Simulacrum Map Device portal in hideout.
        Distinguishes active 'SIMULACRUM OF DELUSION' from old 'SIMULACRUM OF DELUSION (COMPLETED)'.
        Skips completed portals unless allow_completed=True.
        Supports targeting either 'sign' (letters banner) or 'body' (blue portal rift).
        Falls back to legacy portal template if needed for backward compatibility.
        """
        if getattr(self, "hideout_portal_label_tpl", None) is None:
            self._load_delirium_and_portal_templates()

        eff_thresh = threshold if threshold is not None else getattr(self, "hideout_portal_match_threshold", 0.70)
        target = (click_target or getattr(self, "hideout_portal_click_target", "sign")).lower().strip()

        capt = self._get_capturer()
        if screen is None:
            try:
                screen = capt.capture()
            except Exception as e:
                _log(f"  [WARNING] Screen capture failed during hideout portal search: {e}")
                return None
        if screen is None or screen.size == 0:
            return None

        sh, sw = screen.shape[:2]
        mon_left, mon_top = self._get_monitor_offsets()
        candidates = []
        found_completed = False
        scale_est = sh / 1080.0
        scales = [1.0] if abs(scale_est - 1.0) < 0.05 else [round(scale_est, 2), 1.0]

        # 1. Primary: 'SIMULACRUM OF DELUSION' text banner
        lbl = getattr(self, "hideout_portal_label_tpl", None)
        comp_badge = getattr(self, "hideout_portal_completed_tpl", None)
        if lbl is not None:
            lw, lh = lbl.shape[1], lbl.shape[0]
            bw, bh = (comp_badge.shape[1], comp_badge.shape[0]) if comp_badge is not None else (0, 0)
            for s in scales:
                r_lbl = lbl if abs(s - 1.0) < 0.01 else cv2.resize(lbl, (max(20, int(lw * s)), max(10, int(lh * s))))
                rw, rh = r_lbl.shape[1], r_lbl.shape[0]
                if rw > sw or rh > sh:
                    continue
                try:
                    res_copy = cv2.matchTemplate(screen, r_lbl, cv2.TM_CCOEFF_NORMED)
                    while True:
                        _, max_v, _, (px, py) = cv2.minMaxLoc(res_copy)
                        if max_v < eff_thresh:
                            break
                        is_comp = False
                        if comp_badge is not None:
                            roi = screen[max(0, py - int(6 * s)):min(sh, py + rh + int(6 * s)), max(0, px + rw - int(20 * s)):min(sw, px + rw + int(bw * s) + int(25 * s))]
                            r_comp = comp_badge if abs(s - 1.0) < 0.01 else cv2.resize(comp_badge, (max(10, int(bw * s)), max(5, int(bh * s))))
                            if roi.shape[0] >= r_comp.shape[0] and roi.shape[1] >= r_comp.shape[1]:
                                if float(np.max(cv2.matchTemplate(roi, r_comp, cv2.TM_CCOEFF_NORMED))) >= 0.65:
                                    is_comp = True
                        if is_comp:
                            found_completed = True
                        sign_cx, sign_cy = px + rw // 2, py + rh // 2
                        body_cx, body_cy = sign_cx + (int(55 * s) if is_comp else 0), sign_cy + int(120 * s)
                        if not is_comp or allow_completed:
                            candidates.append({
                                "desktop_sign": (mon_left + sign_cx, mon_top + sign_cy),
                                "desktop_body": (mon_left + body_cx, mon_top + body_cy),
                                "confidence": float(max_v),
                                "is_completed": is_comp,
                                "scale": s,
                            })
                        else:
                            _log("  [HIDEOUT PORTAL] Detected portal is COMPLETED ('SIMULACRUM OF DELUSION (COMPLETED)'). Skipping...")
                        cv2.rectangle(res_copy, (max(0, px - int(50 * s)), max(0, py - int(20 * s))), (min(sw, px + rw + int(120 * s)), min(sh, py + rh + int(20 * s))), 0, -1)
                except Exception as e:
                    _log(f"  [HIDEOUT PORTAL] Label matching error: {e}")

        # 2. Secondary: Blue portal body if banner was hidden/unmatched
        ho_img, ho_mask = getattr(self, "hideout_portal_template_img", None), getattr(self, "hideout_portal_mask", None)
        if not candidates and ho_img is not None and ho_mask is not None:
            try:
                res_b = cv2.matchTemplate(screen, ho_img, cv2.TM_SQDIFF_NORMED, mask=ho_mask)
                res_b = np.where(np.isnan(res_b) | (res_b < 0.0) | (res_b > 1.0), 1.0, res_b)
                min_v, _, (bx, by), _ = cv2.minMaxLoc(res_b)
                b_conf = 1.0 - float(min_v)
                if b_conf >= eff_thresh:
                    bcx, bcy = bx + ho_img.shape[1] // 2, by + ho_img.shape[0] // 2
                    is_comp = False
                    if comp_badge is not None:
                        roi = screen[max(0, bcy - 160):max(0, bcy - 70), max(0, bcx - 180):min(sw, bcx + 180)]
                        if roi.shape[0] >= comp_badge.shape[0] and roi.shape[1] >= comp_badge.shape[1]:
                            if float(np.max(cv2.matchTemplate(roi, comp_badge, cv2.TM_CCOEFF_NORMED))) >= 0.65:
                                is_comp = True
                    if is_comp:
                        found_completed = True
                    if not is_comp or allow_completed:
                        candidates.append({
                            "desktop_sign": (mon_left + bcx, mon_top + max(0, bcy - 120)),
                            "desktop_body": (mon_left + bcx, mon_top + bcy),
                            "confidence": b_conf,
                            "is_completed": is_comp,
                            "scale": 1.0,
                        })
            except Exception as e:
                _log(f"  [HIDEOUT PORTAL] Body matching error: {e}")

        if candidates:
            anchor = preferred_pos or getattr(self, "last_map_device_pos", None)
            if anchor:
                ax, ay = anchor
                candidates.sort(key=lambda c: (math.hypot(c["desktop_body"][0] - ax, c["desktop_body"][1] - ay), -c["confidence"]))
            else:
                candidates.sort(key=lambda c: -c["confidence"])
            best = candidates[0]
            chosen = best["desktop_body"] if target == "body" else best["desktop_sign"]
            status_desc = "COMPLETED" if best["is_completed"] else "ACTIVE"
            _log(f"  [HIDEOUT PORTAL MATCH] Found {status_desc} portal (target={target}, conf={best['confidence']:.3f}, scale={best['scale']:.2f}) at screen {chosen}")
            return chosen

        if found_completed and not allow_completed:
            _log("  [HIDEOUT PORTAL] Only completed portal(s) detected on screen. Waiting for active portal...")
            return None

        # 3. Fallback: multi-scale legacy portal matching for backward compatibility
        if self.portal_template_img is not None and self.portal_mask is not None:
            pth, ptw = self.portal_template_img.shape[:2]
            anchor_pos = preferred_pos or getattr(self, "last_map_device_pos", None)
            legacy_cands = []
            for s in [0.75, 0.85, 0.95, 1.0, 1.10]:
                sc_w, sc_h = int(ptw * s), int(pth * s)
                if sc_w > sw or sc_h > sh or sc_w < 40 or sc_h < 40:
                    continue
                r_tpl = self.portal_template_img if abs(s - 1.0) < 0.01 else cv2.resize(self.portal_template_img, (sc_w, sc_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
                r_mask = self.portal_mask if abs(s - 1.0) < 0.01 else cv2.resize(self.portal_mask, (sc_w, sc_h), interpolation=cv2.INTER_NEAREST)
                try:
                    res = cv2.matchTemplate(screen, r_tpl, cv2.TM_SQDIFF_NORMED, mask=r_mask)
                    res = np.where(np.isnan(res) | (res < 0.0) | (res > 1.0), 1.0, res)
                    min_v, _, (lx, ly), _ = cv2.minMaxLoc(res)
                    conf = 1.0 - float(min_v)
                    if conf >= eff_thresh:
                        legacy_cands.append({"desktop_pos": (mon_left + lx + sc_w // 2, mon_top + ly + sc_h // 2), "confidence": conf, "scale": s})
                except Exception:
                    pass
            if legacy_cands:
                legacy_cands.sort(key=lambda c: (math.hypot(c["desktop_pos"][0] - anchor_pos[0], c["desktop_pos"][1] - anchor_pos[1]) if anchor_pos else 0, -c["confidence"]))
                return legacy_cands[0]["desktop_pos"]

        std_portal = self.locate_portal(threshold=eff_thresh, screen=screen)
        if std_portal is not None:
            return std_portal

        return None


    def click_hideout_portal(
        self,
        search_attempts: int = 12,
        timeout: float = 12.0,
        approach_wait: float = 3.5,
        verify_transition: bool = True,
        auto_start_route: bool = True,
        start_pink_dot: int = 1,
        hold_w_seconds: Optional[float] = None,
        settle_wait: Optional[float] = None,
        dry_run: bool = False,
        click_target: Optional[str] = None,
        allow_completed: bool = False,
    ) -> bool:
        """
        Polls for spawned Map Device portal(s) in the hideout, left-clicks the active portal
        (either on the SIMULACRUM OF DELUSION letters sign or the blue portal body) to enter
        the danger zone, waits for area transition, holds 'W' (default 1.3s), and starts route.
        """
        self.release_all_keys()
        self.status_message = "Looking for Portals..."
        eff_target = (click_target or getattr(self, "hideout_portal_click_target", "sign")).lower().strip()
        _log(f"\n[HIDEOUT PORTAL] Looking for spawned active portal ({eff_target}) in hideout...")

        poll_start = time.time()
        portal_pos = None
        while (time.time() - poll_start) < timeout:
            if stop_handler.is_stopped():
                return False
            portal_pos = self.locate_hideout_portal(
                allow_completed=allow_completed,
                click_target=click_target,
            )
            if portal_pos is not None:
                break
            time.sleep(0.4)

        if portal_pos is None:
            _log(f"  [HIDEOUT PORTAL WARNING] No spawned portals detected after {timeout:.1f}s.")
            return False

        if dry_run:
            _log(f"  [DRY RUN] Would left-click hideout portal at {portal_pos}.")
            if auto_start_route:
                _log(f"  [DRY RUN] Would initialize route targeting Pink Dot #{start_pink_dot}.")
                self.set_start_pink_dot(start_pink_dot)
            return True

        _log(f"  [HIDEOUT PORTAL] Left-clicking portal at desktop ({portal_pos[0]}, {portal_pos[1]})...")
        self.status_message = "Entering Simulacrum Portal..."
        window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
        self.move_mouse_inside_game(portal_pos[0], portal_pos[1])
        time.sleep(0.08)
        if pydirectinput:
            pydirectinput.mouseDown(button="left")
            time.sleep(0.08)
            pydirectinput.mouseUp(button="left")
        time.sleep(0.3)

        if approach_wait > 0:
            time.sleep(approach_wait)

        if verify_transition:
            _log("  [ZONE TRANSITION] Waiting for area transition into danger zone (Simulacrum)...")
            self.status_message = "Transitioning to Simulacrum..."
            trans_start = time.time()
            transition_confirmed = False
            while (time.time() - trans_start) < 10.0:
                if stop_handler.is_stopped():
                    return False
                in_ho = self.is_in_hideout()
                if not in_ho:
                    transition_confirmed = True
                    break
                time.sleep(0.5)

            if transition_confirmed:
                _log("  [DANGER ZONE ARRIVAL] Confirmed arrival in Simulacrum danger zone! (in_hideout = False)")
            else:
                _log("  [ZONE TRANSITION] Area transition timeout elapsed. Assuming character entered danger zone.")

        self.in_hideout = False

        # Start run timer immediately the moment bot enters portal to ENEMY area
        self._start_new_run()

        # Clean reset: entering new enemy area must ALWAYS start from first pink dot (WP #0)
        self.interacted_pink_dots.clear()
        self.interacted_zones.clear()
        self.latest_pos, self.last_known_pos = None, None
        self.is_completed = False
        self.start_at_pink_dot = 0
        self._route_reset_to_start = True
        if hasattr(self, "movement_path") and self.movement_path:
            self.movement_path.current_idx = 0

        if stop_handler.is_stopped():
            return False

        # Post-transition settle: allow loading screen to complete and game world to be active
        eff_settle = settle_wait if settle_wait is not None else getattr(self, "portal_entry_settle_seconds", 1.5)
        if eff_settle > 0:
            _log(f"  [ZONE TRANSITION] Allowing {eff_settle:.1f}s for loading screen to complete...")
            time.sleep(eff_settle)

        if stop_handler.is_stopped():
            return False

        # Keep 'W' key held for 1.3 seconds upon entering enemy territory to move away from portal spawn
        eff_hold_w = hold_w_seconds if hold_w_seconds is not None else getattr(self, "portal_entry_hold_w_seconds", 1.3)
        if eff_hold_w > 0:
            _log(f"  [PORTAL ENTRY] Holding 'W' for {eff_hold_w:.1f}s to move character away from portal spawn for reliable localization...")
            self.status_message = f"Entering Zone: Moving Forward ({eff_hold_w:.1f}s)..."
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            if pydirectinput:
                try:
                    pydirectinput.keyDown("w")
                    time.sleep(eff_hold_w)
                    pydirectinput.keyUp("w")
                except Exception as e:
                    _log(f"  [PORTAL ENTRY WARNING] Error while holding 'W' key: {e}")
            else:
                time.sleep(eff_hold_w)
            self.release_all_keys()
            time.sleep(0.2)
            _log(f"  [PORTAL ENTRY] 'W' key released after {eff_hold_w:.1f}s. Character clear of portal spawn.")

        if auto_start_route:
            _log(f"  [ROUTINE START] Starting standard navigation routine targeting Pink Dot #1 from beginning...")
            self.set_start_pink_dot(1)
            if hasattr(self, "movement_path") and self.movement_path:
                self.movement_path.current_idx = 0
            self._route_reset_to_start = True
            self.start()
            self.status_message = "Navigating to Pink Zone #1..."

        return True

