"""
Execution engine for individual zone routine steps (clicks, key holds, waits, attacks, loot, anti-stuck).
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


class ZoneStepExecutorMixin:
    """Execution engine for individual zone routine steps (clicks, key holds, waits, attacks, loot, anti-stuck)."""

    def _execute_zone_routine_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        zone_label: str = "ZONE",
    ) -> bool:
        """Dispatches and executes an individual routine step."""
        action = step.get("action", "").lower().strip()

        if action == "stop":
            duration = float(step.get("duration", self.pink_dot_stop_seconds))
            self.release_all_keys()
            self.move_mouse_inside_game()
            stop_start = time.time()
            while (time.time() - stop_start) < duration:
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                self._trigger_persistent_right_click_if_due()
                rem = max(0.0, duration - (time.time() - stop_start))
                self.status_message = f"[{zone_label}] Stopped ({rem:.1f}s)..."
                time.sleep(0.05)
            return True

        elif action in ("hold_mouse", "hold_key"):
            if self.has_executed_initial_hold:
                _log(f"    [STEP] Hold action already executed once in this session. Skipping hold for {zone_label}.")
                # Ensure persistent combat remains active throughout all rooms until Room 7 portal
                if not self.persistent_combat_active and not getattr(self, "in_hideout", False):
                    c_int = step.get("combat_interval") or step.get("right_click_interval") or step.get("persistent_right_click_interval")
                    c_act = step.get("combat_action", "key")
                    c_key = step.get("combat_key", "t")
                    self.enable_persistent_combat(interval=c_int, action=c_act, key=c_key)
                return True

            hold_k = step.get("key")
            if not hold_k:
                button = str(step.get("button", "middle")).lower().strip()
                if button == "middle":
                    hold_k = "q"
                elif button == "right":
                    hold_k = "t"
                else:
                    hold_k = "q"

            if hold_k == "q" and not getattr(self, "middle_click_hold_enabled", True):
                _log(f"    [CONFIG] Hold Q key disabled (middle_click_hold_enabled=false). Skipping...")
                self.has_executed_initial_hold = True
                self.enable_persistent_combat()
                return True

            duration = float(step.get("duration", self.middle_click_hold_seconds))
            if hold_k == "q":
                self.execute_hold_q(zone_label, duration, step=step)
            else:
                window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
                _log(f"    [ACTION] Holding '{hold_k.upper()}' key for {duration:.1f}s (ONCE on first encounter)...")
                self.is_holding_mouse = True
                try:
                    if pydirectinput:
                        pydirectinput.keyDown(hold_k)
                    h_start = time.time()
                    while (time.time() - h_start) < duration:
                        if stop_handler.is_stopped() or not self.is_active:
                            break
                        rem_h = max(0.0, duration - (time.time() - h_start))
                        self.status_message = f"[{zone_label}] Holding [{hold_k.upper()}] ({rem_h:.1f}s)..."
                        time.sleep(0.05)
                finally:
                    self.is_holding_mouse = False
                    if pydirectinput:
                        pydirectinput.keyUp(hold_k)
                    _log(f"    [ACTION] Key '{hold_k.upper()}' released.")

            # Mark initial hold as completed so subsequent pink dots never hold again in this run
            self.has_executed_initial_hold = True
            c_int = step.get("combat_interval") or step.get("right_click_interval") or step.get("persistent_right_click_interval")
            c_act = step.get("combat_action", "key")
            c_key = step.get("combat_key", "t")
            self.enable_persistent_combat(interval=c_int, action=c_act, key=c_key)
            combat_desc = f"Key '{self.persistent_combat_key.upper()}'" if self.persistent_combat_action == "key" else "Combat Action"
            _log(f"    [COMBAT] Persistent combat attack ACTIVATED ({combat_desc} @ interval={self.persistent_combat_interval:.2f}s) until destination reached.")
            time.sleep(0.1)
            return True

        elif action in ("click_mouse", "press_key"):
            if self.has_executed_initial_hold:
                _log(f"    [STEP] Initial skill already executed in this session. Skipping for {zone_label}.")
                return True

            press_k = step.get("key")
            button = str(step.get("button", "right")).lower().strip()
            if not press_k:
                if button == "right":
                    press_k = "t"
                elif button == "middle":
                    press_k = "q"
                elif button == "left":
                    press_k = None
                else:
                    press_k = "t"

            if press_k == "t" and not getattr(self, "right_click_after_banner_enabled", True):
                _log(f"    [CONFIG] Skill press after banner disabled (right_click_after_banner_enabled=false). Skipping...")
                return True

            clicks = int(step.get("clicks", 1))
            delay = float(step.get("delay", 0.15))
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)

            if press_k:
                self.status_message = f"[{zone_label}] Pressing [{press_k.upper()}]..."
                for _ in range(clicks):
                    if stop_handler.is_stopped() or not self.is_active:
                        return False
                    if pydirectinput:
                        pydirectinput.keyDown(press_k)
                        time.sleep(0.02)
                        pydirectinput.keyUp(press_k)
                    time.sleep(delay)
            else:
                self.move_mouse_inside_game()
                self.status_message = f"[{zone_label}] Clicking Left Mouse..."
                for _ in range(clicks):
                    if stop_handler.is_stopped() or not self.is_active:
                        return False
                    if pydirectinput:
                        pydirectinput.click()
                    time.sleep(delay)
            return True

        elif action == "navigate_to_sim_location":
            target = context.get("target") or {}
            sim_pos = None
            if "target_x" in step and "target_y" in step:
                sim_pos = (float(step["target_x"]), float(step["target_y"]))
            elif isinstance(target, dict) and target.get("sim_pos"):
                sim_pos = tuple(target["sim_pos"])
            elif isinstance(context.get("zone"), dict) and context["zone"].get("sim_pos"):
                sim_pos = tuple(context["zone"]["sim_pos"])

            if sim_pos is None:
                _log(f"    [NAV→SIM] No SIM location (cyan dot) configured for {zone_label}. Skipping walk.")
                return True

            timeout = float(step.get("timeout", 8.0))
            thresh = float(step.get("arrival_threshold", self.arrival_threshold))
            return self._walk_to_coordinate(sim_pos, label="NAV→SIM", timeout=timeout, arrival_threshold=thresh)

        elif action == "detect_and_click_sims":
            self.ensure_loot_labels_visible()
            priority = step.get("priority", ["sim1", "sim3", "sim2"])
            y_offset = int(step.get("click_y_offset", self.sim_click_y_offset_px))
            x_offset = int(step.get("click_x_offset", self.sim_click_x_offset_px))
            app_wait = float(step.get("approach_wait", self.sim_approach_wait_seconds))
            settle_wait = float(step.get("settle_wait", 0.15))
            attempts = int(step.get("search_attempts", 2))
            subsequent_attempts = int(step.get("subsequent_search_attempts", 1))
            threshold = float(step["confidence"]) if "confidence" in step else (float(step["threshold"]) if "threshold" in step else None)
            verify_delay = float(step.get("verify_delay", self.sim_verify_delay_seconds))
            max_attempts = int(step.get("max_click_attempts", step.get("max_attempts", self.sim_max_click_attempts)))
            verify_enabled = bool(step.get("verify_click", step.get("verify", self.sim_verify_click_enabled)))
            max_sim_clicks = int(step.get("max_sim_clicks", 3))
            clicked = self._detect_and_click_sims(
                prefix=f"[{zone_label}]",
                sim_order=priority,
                y_offset_px=y_offset,
                x_offset_px=x_offset,
                approach_wait=app_wait,
                settle_wait=settle_wait,
                search_attempts=attempts,
                subsequent_search_attempts=subsequent_attempts,
                threshold=threshold,
                verify_click=verify_enabled,
                verify_delay=verify_delay,
                max_click_attempts=max_attempts,
                max_sim_clicks=max_sim_clicks,
            )
            context["sims_clicked"] = len(clicked) > 0
            self.hide_loot_labels()
            if step.get("activate_combat", False):
                c_act = step.get("combat_action", getattr(self, "persistent_combat_action", "key"))
                c_key = step.get("combat_key", getattr(self, "persistent_combat_key", "t"))
                c_int = float(step.get("combat_interval", getattr(self, "persistent_combat_interval", 0.65)))
                self.enable_persistent_combat(interval=c_int, action=c_act, key=c_key)
                _log(f"    [COMBAT] Activated continuous '{c_key.upper()}' attack after sim selection.")
            return True

        elif action == "navigate_to_pink_location":
            target = context.get("target") or {}
            pink_pos = None
            if "target_x" in step and "target_y" in step:
                pink_pos = (float(step["target_x"]), float(step["target_y"]))
            elif isinstance(target, dict) and target.get("pink_pos"):
                pink_pos = tuple(target["pink_pos"])
            elif context.get("pink_origin"):
                pink_pos = tuple(context["pink_origin"])
            elif isinstance(target, dict) and "x" in target and "y" in target:
                pink_pos = (float(target["x"]), float(target["y"]))

            if pink_pos is None:
                _log(f"    [NAV→BANNER] No Pink encounter location configured for {zone_label}. Skipping walk.")
                return True

            timeout = float(step.get("timeout", 8.0))
            thresh = float(step.get("arrival_threshold", self.arrival_threshold))
            return self._walk_to_coordinate(pink_pos, label="NAV→BANNER", timeout=timeout, arrival_threshold=thresh)

        elif action == "click_encounter_banner":
            if not getattr(self, "click_banner_enabled", True):
                _log(f"    [CONFIG] Banner clicking disabled (click_banner_enabled=false). Skipping...")
                return True

            if step.get("only_if_no_sims", False) and context.get("sims_clicked", False):
                _log(f"    [STEP] Sims were already selected and only_if_no_sims is set. Skipping banner click.")
                self.hide_loot_labels()
                return True

            self.hide_loot_labels()
            app_wait = float(step.get("approach_wait", self.banner_approach_wait_seconds))
            search_attempts = int(step.get("search_attempts", self.banner_search_attempts))
            verify_delay = float(step.get("verify_delay", self.banner_verify_delay_seconds))
            max_attempts = int(step.get("max_click_attempts", step.get("max_attempts", self.banner_max_click_attempts)))
            verify_enabled = bool(step.get("verify_click", step.get("verify", self.banner_verify_click_enabled)))
            reclick = bool(step.get("reclick", self.reclick_after_approach))

            banner_pos = None
            for attempt in range(1, search_attempts + 1):
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                banner_pos = self.locate_encounter_banner()
                if banner_pos is not None:
                    break
                time.sleep(0.25)

            # Retry: If banner not found, walk closer to pink dot again and re-check
            if banner_pos is None and not stop_handler.is_stopped() and self.is_active:
                target = context.get("target")
                pink_pos = None
                if isinstance(target, dict):
                    pink_pos = target.get("pink_pos")
                if not pink_pos and "zone" in context and isinstance(context["zone"], dict):
                    pink_pos = context["zone"].get("center")

                if pink_pos:
                    _log(f"    [BANNER RETRY] Banner not detected on screen. Walking closer to pink dot at ({pink_pos[0]:.1f}, {pink_pos[1]:.1f}) and re-searching...")
                    self.status_message = f"[{zone_label}] Walking to Pink Dot Retry..."
                    self._walk_to_coordinate(
                        (float(pink_pos[0]), float(pink_pos[1])),
                        label="NAV→BANNER-RETRY",
                        timeout=5.0,
                        arrival_threshold=8.0,
                    )
                    time.sleep(0.4)
                    for retry_attempt in range(1, search_attempts + 1):
                        if stop_handler.is_stopped() or not self.is_active:
                            return False
                        banner_pos = self.locate_encounter_banner()
                        if banner_pos is not None:
                            _log(f"    [BANNER RETRY SUCCESS] Encounter banner detected on retry at ({banner_pos[0]}, {banner_pos[1]})!")
                            break
                        time.sleep(0.25)

            if banner_pos is not None:
                bx, by = self.move_mouse_inside_game(banner_pos[0], banner_pos[1])
                _log(f"    [ACTION] Clicking encounter banner at ({bx}, {by})...")
                self.status_message = f"[{zone_label}] Clicking Banner ({bx}, {by})"
                if pydirectinput:
                    pydirectinput.click()
                    time.sleep(0.08)
                    pydirectinput.mouseUp(button="left")
                time.sleep(0.15)

                if app_wait > 0:
                    self._wait_for_approach(app_wait, reason=f"{zone_label} BANNER")
                    if stop_handler.is_stopped() or not self.is_active:
                        return False

                # Double-check / Verification: Check after approach/wait if encounter banner is still visible on screen
                if verify_enabled and not stop_handler.is_stopped() and self.is_active:
                    if app_wait <= 0 and verify_delay > 0:
                        time.sleep(verify_delay)

                    for attempt_num in range(1, max(1, max_attempts)):
                        if stop_handler.is_stopped() or not self.is_active:
                            break
                        recheck_pos = self.locate_encounter_banner()
                        if recheck_pos is not None:
                            rx, ry = self.move_mouse_inside_game(recheck_pos[0], recheck_pos[1])
                            _log(f"    [BANNER DOUBLE-CHECK] Encounter banner is STILL visible on screen after approach. Re-clicking in-range at ({rx}, {ry}) (attempt {attempt_num + 1}/{max_attempts})...")
                            self.status_message = f"[{zone_label}] Re-clicking Banner ({rx}, {ry})"
                            if pydirectinput:
                                pydirectinput.click()
                                time.sleep(0.08)
                                pydirectinput.mouseUp(button="left")
                            time.sleep(max(0.4, verify_delay))
                        else:
                            _log(f"    [BANNER VERIFIED] Encounter banner click confirmed (no longer visible on screen).")
                            break
                elif reclick:
                    in_range_pos = self.locate_encounter_banner()
                    if in_range_pos is not None:
                        rx, ry = self.move_mouse_inside_game(in_range_pos[0], in_range_pos[1])
                        _log(f"    [ACTION] Re-clicking encounter banner in-range at ({rx}, {ry})...")
                        if pydirectinput:
                            pydirectinput.click()
                            time.sleep(0.08)
                            pydirectinput.mouseUp(button="left")
                        time.sleep(0.15)
                context["banner_clicked"] = True
                if step.get("activate_combat", False):
                    c_act = step.get("combat_action", getattr(self, "persistent_combat_action", "key"))
                    c_key = step.get("combat_key", getattr(self, "persistent_combat_key", "t"))
                    c_int = float(step.get("combat_interval", getattr(self, "persistent_combat_interval", 0.65)))
                    self.enable_persistent_combat(interval=c_int, action=c_act, key=c_key)
                    _log(f"    [COMBAT] Activated continuous '{c_key.upper()}' attack after encounter banner.")
            else:
                _log(f"    [WARNING] Encounter banner not detected on screen.")
                self.move_mouse_inside_game()
                if step.get("activate_combat", False):
                    c_act = step.get("combat_action", getattr(self, "persistent_combat_action", "key"))
                    c_key = step.get("combat_key", getattr(self, "persistent_combat_key", "t"))
                    c_int = float(step.get("combat_interval", getattr(self, "persistent_combat_interval", 0.65)))
                    self.enable_persistent_combat(interval=c_int, action=c_act, key=c_key)
                    _log(f"    [COMBAT] Activated continuous '{c_key.upper()}' attack after banner attempt.")

            # Hide loot labels after sims/banner interaction before combat/orbit
            self.hide_loot_labels()
            return True

        elif action == "orbit_yellow_zone":
            self.hide_loot_labels()
            orbit_duration = float(step.get("duration", self.orbit_duration))
            rc_interval = float(step.get("right_click_interval", self.orbit_right_click_interval_seconds))
            roll_enabled = bool(step.get("rolling_enabled", False))
            roll_interval = float(step.get("rolling_interval", 5.0))
            orbit_zones = self.movement_path.get_orbit_zones() if hasattr(self.movement_path, "get_orbit_zones") else []
            target = context.get("target")
            zone = context.get("zone")
            c_pos = self.latest_pos or ((target["x"], target["y"]) if isinstance(target, dict) and "x" in target else (0.0, 0.0))
            best_zone = zone
            if best_zone is None and orbit_zones:
                best_d = float("inf")
                for z in orbit_zones:
                    if isinstance(z, dict):
                        zc = z.get("center", [0, 0])
                        d = math.hypot(zc[0] - c_pos[0], zc[1] - c_pos[1])
                        if d < best_d:
                            best_d = d
                            best_zone = z

            if best_zone is not None:
                context["orbited"] = True
                self._routine_did_orbit = True
                z_id = best_zone.get("id")
                if z_id:
                    self.interacted_zones.add(z_id)
                portal_exit = bool(
                    step.get("portal_early_exit", False)
                    or "pink_7" in str(zone_label).lower()
                    or (best_zone and best_zone.get("associated_pink") == "pink_7")
                )
                portal_min_sec = float(step.get("portal_early_exit_min_seconds", 30.0))
                orbit_ok = self._run_orbit_loop(
                    duration=orbit_duration,
                    best_zone=best_zone,
                    right_click_interval=rc_interval,
                    zone_label=zone_label,
                    rolling_enabled=roll_enabled,
                    rolling_interval=roll_interval,
                    portal_early_exit=portal_exit,
                    portal_early_exit_min_seconds=portal_min_sec,
                )
                return orbit_ok
            else:
                _log(f"    [WARNING] No yellow orbit zone found for {zone_label}. Skipping orbit.")
                return True

        elif action == "click_delirium_statue":
            search_attempts = int(step.get("search_attempts", 5))
            app_wait = float(step.get("approach_wait", 1.5))
            loot_drop_delay = float(step.get("loot_drop_delay", 2.0))
            require_proximity = bool(step.get("require_loot_proximity", True))
            max_loot_dist = float(step.get("max_loot_distance", 35.0))
            walk_if_far = bool(step.get("walk_to_loot_if_far", True))
            loot_pos = self._resolve_target_loot_pos(step, context, zone_label)
            # Portal position should always be detected fresh on screen after arriving at LOOT location
            portal_pos = step.get("portal_pos")

            return self.click_delirium_statue(
                search_attempts=search_attempts,
                approach_wait=app_wait,
                loot_drop_delay=loot_drop_delay,
                label=zone_label,
                loot_pos=loot_pos,
                require_loot_proximity=require_proximity,
                max_loot_distance=max_loot_dist,
                walk_to_loot_if_far=walk_if_far,
                portal_pos=portal_pos,
            )

        elif action == "navigate_to_loot_location":
            loot_pos = self._resolve_target_loot_pos(step, context, zone_label)

            if loot_pos is None:
                _log(f"    [NAV→LOOT] No LOOT location (white dot) configured for {zone_label}. Skipping walk.")
                return True

            timeout = float(step.get("timeout", 10.0))
            thresh = float(step.get("arrival_threshold", self.arrival_threshold))
            return self._walk_to_coordinate(loot_pos, label="NAV→LOOT", timeout=timeout, arrival_threshold=thresh)

        elif action == "pickup_loot":
            max_pickups = int(step.get("max_items", self.max_loot_pickups))
            # Master toggle in config.json (wait_for_loot_confirmation) disables waiting for green light when False
            if not self.wait_for_loot_confirmation:
                wait_for_green = False
            else:
                wait_for_green = bool(step.get("wait_for_green_light", True))
            pickup_delay = float(step.get("pickup_delay", self.loot_pickup_wait_seconds))
            app_wait = float(step.get("approach_wait", getattr(self, "loot_approach_wait_seconds", 1.1)))
            prev_delay = self.loot_pickup_wait_seconds
            self.loot_pickup_wait_seconds = pickup_delay
            try:
                self.collect_loot(max_pickups=max_pickups, approach_wait=app_wait)
            finally:
                self.loot_pickup_wait_seconds = prev_delay

            if wait_for_green:
                self.release_all_keys()
                self.waiting_for_green_light = True
                self.status_message = "WAITING FOR GREEN LIGHT (Verify Loot Pickup - Press 'G' to Resume)"
                _log("\n[AUTOPILOT] >>> LOOT PICKUP FINISHED! Waiting for GREEN LIGHT to continue...")
                while self.waiting_for_green_light:
                    if stop_handler.is_stopped() or not self.is_active:
                        return False
                    self._trigger_persistent_right_click_if_due()
                    if keyboard:
                        try:
                            if keyboard.is_pressed('g') or keyboard.is_pressed('G') or keyboard.is_pressed('enter'):
                                self.give_green_light()
                                break
                        except Exception:
                            pass
                    time.sleep(0.05)

            # If looting is completed in Room 7 (last room), stop run timer and finalize run immediately
            current_room = getattr(self, "current_room_key", None)
            is_last = self._is_last_room(current_room) if current_room else ("pink_7" in str(zone_label).lower())
            if is_last and not getattr(self, "run_completed", False):
                rk = current_room or "pink_7"
                _log(f"\n[RUN TIMER] Room 7 looting completed! Stopping run timer and finalizing run...")
                if not any(r.get("room") == rk for r in getattr(self, "run_rooms_cleared", [])):
                    r_el = (time.time() - self.run_start_time) if self.run_start_time else 0.0
                    self._record_room_cleared(rk, r_el)
                self._finalize_run(last_room=rk, reason="room_7_looting_completed")

            return True

        elif action == "press_key":
            key = str(step.get("key", "q")).lower().strip()
            duration = float(step.get("duration", 0.1))
            window_focuser.ensure_focused(monitor_idx=self.monitor_idx)
            self.move_mouse_inside_game()
            _log(f"    [ACTION] Pressing key '{key}' for {duration:.2f}s...")
            self.status_message = f"[{zone_label}] Key '{key.upper()}' ({duration:.1f}s)..."
            if pydirectinput:
                pydirectinput.keyDown(key)
            k_start = time.time()
            while (time.time() - k_start) < duration:
                if stop_handler.is_stopped() or not self.is_active:
                    break
                time.sleep(0.02)
            if pydirectinput:
                pydirectinput.keyUp(key)
            time.sleep(0.05)
            return True

        elif action in ("wait_for_user_key", "wait_for_key", "wait_for_user_confirmation"):
            target_key = str(step.get("key", "f5")).lower().strip()
            prompt = str(step.get("prompt", f"Press [{target_key.upper()}] to continue"))
            timeout = float(step.get("timeout", 0.0))
            return self.wait_for_user_key(key=target_key, prompt=prompt, timeout=timeout)

        elif action in ("click_exit_portal", "click_portal"):
            search_attempts = int(step.get("search_attempts", 5))
            app_wait = float(step.get("approach_wait", 2.0))
            return self.click_exit_portal(search_attempts=search_attempts, approach_wait=app_wait, context=context)

        elif action == "click_stash":
            search_attempts = int(step.get("search_attempts", 10))
            timeout = float(step.get("timeout", 15.0))
            verify_inv = bool(step.get("verify_inventory_open", True))
            return self.click_stash(search_attempts=search_attempts, timeout=timeout, verify_inventory=verify_inv)

        elif action in ("stash_inventory_items", "stash_items"):
            exclude_cols = int(step.get("exclude_last_columns", 3))
            close_after = bool(step.get("close_after", False))
            return self.stash_inventory_items(exclude_last_columns=exclude_cols, close_after=close_after) >= 0

        elif action in ("close_all_hideout_windows", "press_escape", "close_windows"):
            wait_s = float(step.get("wait_seconds", 0.35))
            return self.close_all_hideout_windows(wait_seconds=wait_s)

        elif action in ("click_map_device", "interact_map_device"):
            search_attempts = int(step.get("search_attempts", 8))
            timeout = float(step.get("timeout", 12.0))
            return self.click_map_device(search_attempts=search_attempts, timeout=timeout)

        elif action in ("select_simulacrum_map", "open_simulacrum_map", "click_simulacrum_map"):
            max_attempts = int(step.get("max_attempts", 5))
            node = self.select_accessible_simulacrum_map(max_attempts=max_attempts)
            return node is not None

        elif action in ("insert_simulacrum_map", "insert_map", "insert_map_into_popup"):
            target_slot = int(step.get("target_slot", 0))
            method = str(step.get("method", "drag"))
            return self.insert_map_into_simulacrum_popup(target_slot_idx=target_slot, method=method)

        elif action in ("click_traverse", "click_traverse_button", "traverse"):
            timeout = float(step.get("timeout", 5.0))
            verify_close = bool(step.get("verify_close", True))
            return self.click_traverse_button(timeout=timeout, verify_close=verify_close)

        elif action in ("click_hideout_portal", "click_portal_to_danger_zone", "enter_portal", "enter_simulacrum_portal"):
            search_attempts = int(step.get("search_attempts", 12))
            timeout = float(step.get("timeout", 12.0))
            app_wait = float(step.get("approach_wait", 3.5))
            verify_trans = bool(step.get("verify_transition", True))
            auto_start = bool(step.get("auto_start_route", True))
            start_pink = int(step.get("start_pink_dot", 1))
            p_kwargs: Dict[str, Any] = {
                "search_attempts": search_attempts,
                "timeout": timeout,
                "approach_wait": app_wait,
                "verify_transition": verify_trans,
                "auto_start_route": auto_start,
                "start_pink_dot": start_pink,
            }
            if "hold_w_seconds" in step:
                p_kwargs["hold_w_seconds"] = step["hold_w_seconds"]
            if "settle_wait" in step:
                p_kwargs["settle_wait"] = step["settle_wait"]
            return self.click_hideout_portal(**p_kwargs)

        elif action in ("hideout_full_cycle", "run_hideout_full_cycle"):
            traverse_and_enter = bool(step.get("traverse_and_enter", False))
            auto_start = bool(step.get("auto_start_route", True))
            start_pink = int(step.get("start_pink_dot", 1))
            res = self.run_hideout_full_cycle(
                traverse_and_enter=traverse_and_enter,
                auto_start_route=auto_start,
                start_pink_dot=start_pink,
            )
            return bool(res.get("success", False))

        elif action in ("stop_combat", "disable_combat"):
            self.disable_persistent_combat()
            _log(f"    [COMBAT] Stopped persistent combat attacks.")
            return True

        elif action == "wait":
            duration = float(step.get("duration", 1.0))
            self.status_message = f"[{zone_label}] Waiting ({duration:.1f}s)..."
            w_start = time.time()
            while (time.time() - w_start) < duration:
                if stop_handler.is_stopped() or not self.is_active:
                    return False
                self._trigger_persistent_right_click_if_due()
                time.sleep(0.05)
            return True

        else:
            _log(f"    [WARNING] Unknown routine action '{action}'. Skipping...")
            return True

