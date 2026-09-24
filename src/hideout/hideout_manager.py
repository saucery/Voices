"""
High-level hideout test sequences and full autonomous town routine orchestrator.
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


class HideoutManagerMixin:
    """High-level hideout test sequences and full autonomous town routine orchestrator."""

    def test_hideout_sequence(
        self,
        dry_run: bool = False,
        deposit_only: bool = False,
        screen: Optional[np.ndarray] = None,
        full_cycle: bool = False,
        traverse_and_enter: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a targeted test of the complete hideout flow:
        1. Checks hideout arrival confirmation via minimap (hideout_layout.png).
        2. Locates and clicks STASH in the hideout (if not deposit_only).
        3. Verifies Stash and Inventory windows are open.
        4. Saves an inventory screenshot to inventory_screenshots/.
        5. Scans 5x12 inventory grid and Ctrl+Clicks non-empty items (excluding last 3 columns).
        6. If full_cycle=True: closes windows (Escape), clicks Map Device, selects Simulacrum map circle,
           verifies Delusion popup, transfers Tier 15 map into popup, and optionally clicks TRAVERSE & portal.

        :param dry_run: If True, detects and reports all elements without sending clicks or keystrokes.
        :param deposit_only: If True, assumes stash and inventory are already open, only runs screenshot & stashing.
        :param screen: Optional screen frame to test on.
        :param full_cycle: If True, continues sequence through Escape, Map Device, and Simulacrum map insertion.
        :param traverse_and_enter: If True (with full_cycle), continues through TRAVERSE click and Portal click.
        :return: Diagnostic results dictionary.
        """
        self.in_hideout = True
        self.disable_persistent_combat()
        self.release_all_keys()
        self.ensure_loot_labels_visible()
        self._load_stash_and_inventory_templates(force_reload=True)
        report: Dict[str, Any] = {
            "dry_run": dry_run,
            "deposit_only": deposit_only,
            "in_hideout": False,
            "stash_pos": None,
            "stash_clicked": False,
            "inventory_open": False,
            "screenshot_path": None,
            "items_detected": 0,
            "items_stashed": 0,
            "traverse_clicked": False,
            "portal_clicked": False,
            "success": False,
            "messages": [],
        }

        def _step_log(msg: str):
            _log(f"  [TEST HIDEOUT] {msg}")
            report["messages"].append(msg)

        _log("\n========================================================")
        _log(" [TEST HIDEOUT] Starting Hideout Functionality Test")
        _log(f" Mode: {'DRY RUN (No Clicks)' if dry_run else 'LIVE EXECUTION'} | Stash: {'SKIP (Deposit Only)' if deposit_only else 'FIND & CLICK'}")
        _log("========================================================")

        capt = self._get_capturer()
        if screen is None:
            try:
                screen = capt.capture()
            except Exception as e:
                _step_log(f"Screen capture failed: {e}")
                return report

        if screen is None or screen.size == 0:
            _step_log("Screen capture returned empty image.")
            return report

        # Step 1: Check Hideout Minimap & Stash Presence
        if not deposit_only:
            _step_log("Step 1/5: Checking Hideout Arrival (Minimap Layout & Stash)...")
            in_ho = self.is_in_hideout(screen=screen)
            report["in_hideout"] = in_ho
            if not in_ho:
                _step_log("Minimap does not match hideout_layout.png and STASH not visible! Character may not be in hideout.")
                if not dry_run:
                    _step_log("Aborting live test because hideout layout was not verified.")
                    return report
            else:
                _step_log("Hideout arrival confirmed (minimap / stash verified).")
        else:
            report["in_hideout"] = True
            _step_log("Step 1/5: Skipping hideout minimap check (deposit_only=True).")

        # Step 2: Locate and Click Stash
        if not deposit_only:
            _step_log("Step 2/5: Locating STASH on screen...")
            stash_pos = self.locate_stash(screen=screen)
            report["stash_pos"] = stash_pos
            if not stash_pos:
                _step_log("Could not locate STASH label or chest on screen!")
                if not dry_run:
                    return report
            else:
                _step_log(f"STASH located at screen ({stash_pos[0]}, {stash_pos[1]}).")
                if not dry_run:
                    _step_log("Clicking STASH chest/label...")
                    window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
                    self.move_mouse_inside_game(stash_pos[0], stash_pos[1])
                    time.sleep(0.10)
                    if pydirectinput:
                        pydirectinput.mouseDown(button="left")
                        time.sleep(0.08)
                        pydirectinput.mouseUp(button="left")
                    report["stash_clicked"] = True
                    time.sleep(1.0)
                    # Re-capture screen after stash click
                    try:
                        screen = capt.capture()
                    except Exception:
                        pass
                else:
                    _step_log(f"[DRY RUN] Would click STASH at ({stash_pos[0]}, {stash_pos[1]}).")
        else:
            _step_log("Step 2/5: Skipping STASH click (deposit_only=True).")

        # Step 3: Verify Inventory Open
        _step_log("Step 3/5: Verifying Inventory & Stash window open...")
        inv_open = self.is_inventory_open(screen=screen)
        if not inv_open and not dry_run:
            # Poll for up to 8 seconds in case character is walking across hideout to stash
            poll_start = time.time()
            while (time.time() - poll_start) < 8.0:
                time.sleep(0.2)
                try:
                    screen = capt.capture()
                except Exception:
                    pass
                if self.is_inventory_open(screen=screen):
                    inv_open = True
                    break

        report["inventory_open"] = inv_open
        if not inv_open:
            if dry_run:
                _step_log("[DRY RUN] Inventory window is currently closed.")
            else:
                _step_log("Inventory window is not open after clicking stash.")
                return report
        else:
            _step_log("Confirmed Inventory window is OPEN.")

        # Step 4: Save Inventory Screenshot
        _step_log("Step 4/5: Saving Inventory Screenshot to dedicated folder...")
        saved_path = self.save_inventory_screenshot(screen=screen)
        report["screenshot_path"] = saved_path
        if saved_path:
            _step_log(f"Saved inventory screenshot to '{saved_path}'.")
        else:
            _step_log("Unable to save inventory screenshot.")

        # Step 5: Scan & Stash Items
        _step_log("Step 5/5: Scanning 5x12 inventory grid (excluding last 3 columns)...")
        stashed_items = self.stash_inventory_items(
            screen=screen,
            exclude_last_columns=3,
            dry_run=dry_run,
        )
        report["items_detected"] = stashed_items
        report["items_stashed"] = 0 if dry_run else stashed_items
        if dry_run:
            _step_log(f"[DRY RUN] Found {stashed_items} non-empty item(s) to deposit.")
        else:
            _step_log(f"Successfully deposited {stashed_items} item(s) into Stash!")
        # Steps 6..9: Full Cycle Continuation (Escape -> Map Device -> Simulacrum Map -> Insert Map)
        if full_cycle and not deposit_only:
            # Step 6: Press Escape to close all open windows
            _step_log("Step 6/9: Pressing Escape to close all windows...")
            if not dry_run:
                closed = self.close_all_hideout_windows()
                report["windows_closed"] = closed
            else:
                report["windows_closed"] = True
                _step_log("[DRY RUN] Would press Escape to close windows.")

            # Step 7: Locate and Click Map Device
            _step_log("Step 7/9: Locating & Clicking Map Device...")
            if not dry_run:
                md_ok = self.click_map_device()
                report["map_device_clicked"] = md_ok
                if not md_ok:
                    _step_log("Failed to locate or click Map Device.")
                    report["success"] = False
                    return report
            else:
                report["map_device_clicked"] = True
                _step_log("[DRY RUN] Would click Map Device.")

            # Step 8: Select Accessible Simulacrum Map Circle
            _step_log("Step 8/9: Finding and clicking accessible Simulacrum map circle...")
            sim_node = self.select_accessible_simulacrum_map(max_attempts=5, dry_run=dry_run)
            if sim_node is None and not dry_run:
                _step_log("No accessible Simulacrum map found.")
                report["success"] = False
                return report
            report["simulacrum_selected"] = True

            # Step 9: Transfer Tier 15 Map from last 3 columns into Delusion Popup
            _step_log("Step 9/9: Transferring Tier 15 map into Simulacrum of Delusion popup...")
            map_ok = self.insert_map_into_simulacrum_popup(target_slot_idx=0, method="drag", dry_run=dry_run)
            report["map_transferred"] = map_ok
            if not map_ok and not dry_run:
                _step_log("Failed to transfer Tier 15 map into Delusion popup.")
                report["success"] = False
                return report

            if traverse_and_enter:
                # Step 10: Find and Click TRAVERSE Button
                _step_log("Step 10/11: Locating and clicking TRAVERSE button...")
                trav_ok = self.click_traverse_button(dry_run=dry_run)
                report["traverse_clicked"] = trav_ok
                if not trav_ok and not dry_run:
                    _step_log("Failed to click TRAVERSE button.")
                    report["success"] = False
                    return report

                # Step 11: Locate and Click Hideout Portal to enter Simulacrum danger zone
                _step_log("Step 11/11: Locating and clicking hideout portal into Simulacrum danger zone...")
                portal_ok = self.click_hideout_portal(dry_run=dry_run, auto_start_route=False)
                report["portal_clicked"] = portal_ok
                if not portal_ok and not dry_run:
                    _step_log("Failed to enter hideout portal.")
                    report["success"] = False
                    return report

        if full_cycle and not deposit_only:
            if traverse_and_enter:
                report["success"] = bool(report.get("simulacrum_selected") and report.get("map_transferred") and report.get("traverse_clicked") and report.get("portal_clicked")) if not dry_run else True
            else:
                report["success"] = bool(report.get("simulacrum_selected") and report.get("map_transferred")) if not dry_run else True
        else:
            report["success"] = True
        _log("========================================================")
        _log(f" [TEST HIDEOUT] Hideout Functionality Test COMPLETE ({'SUCCESS' if report['success'] else 'FAILED'})")
        _log("========================================================\n")
        return report


    def run_hideout_full_cycle(
        self,
        dry_run: bool = False,
        screen: Optional[np.ndarray] = None,
        traverse_and_enter: bool = False,
        auto_start_route: bool = True,
        start_pink_dot: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes the complete end-to-end hideout sequence:
        1. Confirm Hideout Arrival via minimap / stash.
        2. Click Stash & Verify Inventory Open.
        3. Stash Inventory Items (excluding last 3 columns).
        4. Press Escape to close all open windows.
        5. Click Map Device & Wait for Atlas Map screen.
        6. Locate Simulacrum Node & Click Map Circle until Delusion Popup is visible.
        7. Transfer Tier 15 Map from last 3 columns into the Delusion Popup.
        8. (traverse_and_enter=True): Click TRAVERSE button.
        9. (traverse_and_enter=True): Click Hideout Portal to enter Simulacrum danger zone & auto-start route!
        """
        self.in_hideout = True
        self.disable_persistent_combat()
        self.release_all_keys()
        self.ensure_loot_labels_visible()
        self._load_stash_and_inventory_templates(force_reload=True)

        report: Dict[str, Any] = {
            "dry_run": dry_run,
            "in_hideout": False,
            "stash_clicked": False,
            "inventory_stashed": False,
            "windows_closed": False,
            "map_device_clicked": False,
            "simulacrum_selected": False,
            "map_transferred": False,
            "traverse_clicked": False,
            "portal_clicked": False,
            "success": False,
            "messages": [],
        }

        def _step_log(msg: str):
            _log(f"  [FULL CYCLE] {msg}")
            report["messages"].append(msg)

        _log("\n========================================================")
        _log(" [HIDEOUT FULL CYCLE] Starting Complete End-to-End Sequence")
        _log(f" Mode: {'DRY RUN (No Clicks)' if dry_run else 'LIVE EXECUTION'}")
        _log(f" Traverse & Enter: {traverse_and_enter} | Auto-Start Route: {auto_start_route} (Pink #{start_pink_dot})")
        _log("========================================================")

        capt = self._get_capturer()
        if screen is None:
            try:
                screen = capt.capture()
            except Exception as e:
                _step_log(f"Screen capture failed: {e}")
                return report

        # Step 1: Check Hideout (poll up to 4.0s for confirmation to avoid capture lag)
        _step_log("Step 1/7: Checking Hideout Arrival...")
        in_ho = False
        check_start = time.time()
        while (time.time() - check_start) < 4.0:
            if stop_handler.is_stopped():
                return report
            cur_scr = screen if (time.time() - check_start < 0.25) else None
            in_ho = self.is_in_hideout(screen=cur_scr)
            if in_ho:
                break
            time.sleep(0.35)

        report["in_hideout"] = in_ho
        if not in_ho and not dry_run:
            _step_log("Character not confirmed in hideout. Aborting full cycle.")
            return report

        # Step 2: Click Stash
        _step_log("Step 2/7: Locating & Clicking Stash...")
        if not dry_run:
            stash_ok = self.click_stash(search_attempts=8, timeout=10.0, verify_inventory=True)
            report["stash_clicked"] = stash_ok
            if not stash_ok:
                _step_log("Failed to open Stash & Inventory.")
                return report
        else:
            report["stash_clicked"] = True
            _step_log("[DRY RUN] Would click Stash.")

        # Step 3: Stash Items (cols 0..8)
        _step_log("Step 3/7: Stashing inventory items (excluding last 3 columns)...")
        stashed = self.stash_inventory_items(exclude_last_columns=3, dry_run=dry_run, close_after=False)
        report["inventory_stashed"] = True
        _step_log(f"Deposited {stashed} item(s) into Stash.")

        # Step 4: Press Escape to close all windows
        _step_log("Step 4/7: Pressing Escape to close all windows...")
        if not dry_run:
            closed = self.close_all_hideout_windows()
            report["windows_closed"] = closed
        else:
            report["windows_closed"] = True
            _step_log("[DRY RUN] Would press Escape to close windows.")

        # Step 5: Click Map Device
        _step_log("Step 5/7: Locating & Clicking Map Device...")
        if not dry_run:
            md_ok = self.click_map_device()
            report["map_device_clicked"] = md_ok
            if not md_ok:
                _step_log("Failed to locate or click Map Device.")
                return report
        else:
            report["map_device_clicked"] = True
            _step_log("[DRY RUN] Would click Map Device.")

        # Step 6: Select Accessible Simulacrum Map Circle
        _step_log("Step 6/7: Finding and clicking accessible Simulacrum map circle...")
        sim_node = self.select_accessible_simulacrum_map(max_attempts=5, dry_run=dry_run)
        if sim_node is None and not dry_run:
            _step_log("No accessible Simulacrum map found.")
            return report
        report["simulacrum_selected"] = True

        # Step 7: Insert Tier 15 Map into Delusion Popup
        _step_log("Step 7/7: Transferring Tier 15 map into Simulacrum of Delusion popup...")
        map_ok = self.insert_map_into_simulacrum_popup(target_slot_idx=0, method="drag", dry_run=dry_run)
        report["map_transferred"] = map_ok

        if traverse_and_enter:
            # Step 8: Click TRAVERSE button
            _step_log("Step 8/9: Locating and clicking TRAVERSE button...")
            trav_ok = self.click_traverse_button(dry_run=dry_run)
            report["traverse_clicked"] = trav_ok
            if not trav_ok and not dry_run:
                _step_log("Failed to click TRAVERSE button.")
                return report

            # Step 9: Click Hideout Portal to enter Simulacrum danger zone
            _step_log(f"Step 9/9: Locating & clicking hideout portal to enter Simulacrum danger zone (auto_start={auto_start_route})...")
            portal_ok = self.click_hideout_portal(
                dry_run=dry_run,
                auto_start_route=auto_start_route,
                start_pink_dot=start_pink_dot,
            )
            report["portal_clicked"] = portal_ok
            if not portal_ok and not dry_run:
                _step_log("Failed to enter hideout portal.")
                return report

            report["success"] = bool(report["simulacrum_selected"] and report["map_transferred"] and report.get("traverse_clicked") and report.get("portal_clicked")) if not dry_run else True
        else:
            report["success"] = bool(report["simulacrum_selected"] and report["map_transferred"]) if not dry_run else True

        _log("========================================================")
        _log(f" [HIDEOUT FULL CYCLE] Sequence Complete: {'SUCCESS' if report['success'] else 'FAILED'}")
        _log("========================================================\n")
        return report

