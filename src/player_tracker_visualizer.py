"""
Player Position Tracker Visualizer Module
Provides an interactive real-time dual-view window comparing:
1. Game Minimap (live capture with exact orange 'X' player icon detection)
2. Active Room Template, Stitched World Map, or Reference Map with player localization
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
import cv2
import numpy as np

from .screen_capturer import ScreenCapturer
from .room_classifier import RoomClassifier
from .world_map import WorldMapTracker
from .map_localizer import MapLocalizer
from .minimap_extractor import MinimapExtractor
from .movement_path import MovementPath
from .route_navigator import RouteNavigator
from .stop_handler import stop_handler


class PlayerTrackerVisualizer:
    """
    Renders an interactive real-time dashboard comparing what the game shows
    on the minimap against what the bot detects in the room/world map.
    """

    VIEW_ROOM_TEMPLATE = 0
    VIEW_WORLD_MAP = 1
    VIEW_REFERENCE_MAP = 2

    def __init__(
        self,
        classifier: Optional[RoomClassifier] = None,
        world_map: Optional[WorldMapTracker] = None,
        localizer: Optional[MapLocalizer] = None,
        movement_path: Optional[MovementPath] = None,
        capturer: Optional[ScreenCapturer] = None,
        extractor: Optional[MinimapExtractor] = None,
        monitor_idx: int = 2,
        window_title: str = "Voices Visualizer - Player Position Tracker",
        history_len: int = 300,
        start_pink_dot: Optional[int] = None,
        navigator: Optional[RouteNavigator] = None,
    ):
        self.classifier = classifier or RoomClassifier()
        self.world_map = world_map or WorldMapTracker()
        self.localizer = localizer or MapLocalizer()
        self.movement_path = movement_path or MovementPath()
        self.monitor_idx = monitor_idx
        self.capturer = capturer
        self.navigator = navigator or RouteNavigator(movement_path=self.movement_path, monitor_idx=self.monitor_idx, capturer=self.capturer)
        if start_pink_dot is not None:
            self.navigator.set_start_pink_dot(start_pink_dot)
        self.extractor = extractor or MinimapExtractor()
        self.window_title = window_title
        self.history_len = history_len

        # Display mode: 0 = Active Room Template, 1 = Stitched World Map, 2 = Reference Map
        self.view_mode: int = self.VIEW_ROOM_TEMPLATE

        # Tracking state
        self.trajectory_history: List[Dict[str, Any]] = []
        self.is_paused: bool = False
        self.show_overlay: bool = False
        self.show_edges: bool = False
        self.show_trajectory: bool = True
        self.show_bounding_boxes: bool = True
        self.last_result: Optional[Dict[str, Any]] = None
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.snapshot_count: int = 0

        # UI Interactive Button & Notification State
        self.btn_refresh_rect: Tuple[int, int, int, int] = (580, 12, 175, 30)
        self.btn_refresh_hover: bool = False
        self.btn_refresh_clicked: float = 0.0
        self.btn_pink_dot_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.btn_pink_dot_hover: bool = False
        self.btn_early_exit_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.btn_early_exit_hover: bool = False
        self.btn_hideout_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.btn_hideout_hover: bool = False
        self.hideout_test_active: bool = False
        self.hideout_routine_active: bool = False
        self.btn_green_light_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.btn_green_light_hover: bool = False
        self.notification_msg: str = ""
        self.notification_expiry: float = 0.0

        # Room Bounding Box Editor State
        self.edit_boxes_mode: bool = False
        self.selected_room_for_edit: int = 1
        self.is_dragging_box: bool = False
        self.drag_mode: Optional[str] = None
        self.drag_start_map_pos: Optional[Tuple[int, int]] = None
        self.drag_initial_box: Optional[Tuple[int, int, int, int]] = None
        self.map_view_bounds: Tuple[int, int, int, int] = (0, 0, 1, 1)  # (offset_x, offset_y, disp_w, disp_h)
        self.map_img_size: Tuple[int, int] = (794, 581)
        self.cursor_map_pos: Optional[Tuple[int, int]] = None
        self.boxes_modified: bool = False

        self.btn_edit_boxes_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.btn_edit_boxes_hover: bool = False
        self.btn_save_boxes_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.btn_save_boxes_hover: bool = False
        self.btn_room_select_rects: Dict[int, Tuple[int, int, int, int]] = {}

        # Optimization & Pipeline Caching State
        self.frame_idx: int = 0
        self.cached_room_res: Optional[Dict[str, Any]] = None
        self.cached_map_res: Optional[Dict[str, Any]] = None

        # Current Area Detection State (ENEMY AREA vs HIDEOUT)
        self.area_type: str = "UNKNOWN"  # "ENEMY_AREA" | "HIDEOUT" | "UNKNOWN"
        self.area_label: str = "DETECTING..."
        self.area_sublabel: str = "Scanning minimap..."
        self.area_confidence: float = 0.0
        self.last_hideout_check_time: float = 0.0
        self.cached_hideout_match: bool = False

    def reload_route(self) -> bool:
        """Reloads route waypoints on demand (re-extracting from templates/route.png if present)."""
        self.btn_refresh_clicked = time.time()
        success = self.navigator.reload_route()
        if success:
            wps_count = len(self.movement_path.waypoints)
            zones_count = len(self.movement_path.get_orbit_zones())
            pinks_count = len(getattr(self.movement_path, "get_pink_zones", lambda: [])())
            zone_txt = f" + {zones_count} yellow zone(s)" if zones_count else ""
            pink_txt = f" + {pinks_count} pink dot(s)" if pinks_count else ""
            self.notification_msg = f"ROUTE RELOADED! ({wps_count} waypoints{zone_txt}{pink_txt})"
            self.notification_expiry = time.time() + 3.5
            print(f"\n[VISUALIZER] Successfully reloaded route: {wps_count} waypoints{zone_txt}{pink_txt} active!")
        else:
            self.notification_msg = "FAILED TO RELOAD ROUTE (Check templates/route.png)"
            self.notification_expiry = time.time() + 3.5
            print("\n[VISUALIZER] Failed to reload route. Ensure templates/route.png has Blue start, Green line, and Red finish.")
        return success

    def save_room_boxes(self) -> bool:
        """Saves custom room bounding boxes to disk."""
        success = self.movement_path.save_room_bounding_boxes("paths/room_bounding_boxes.json")
        if success:
            self.boxes_modified = False
            self.notification_msg = "SUCCESSFULLY SAVED 7 ROOM BOUNDING BOXES!"
            self.notification_expiry = time.time() + 3.5
            print("\n[VISUALIZER] Successfully saved room bounding boxes to paths/room_bounding_boxes.json")
        else:
            self.notification_msg = "FAILED TO SAVE ROOM BOXES"
            self.notification_expiry = time.time() + 3.5
        return success

    def start_hideout_full_routine(
        self,
        dry_run: bool = False,
        start_pink_dot: int = 1,
    ):
        """
        Launches the complete autonomous routine from Hideout in a background daemon thread:
        1. Confirm Hideout & unhide labels (Z key).
        2. Open Stash & deposit inventory loot (exclude last 3 columns).
        3. Close windows with Escape.
        4. Open Map Device / Atlas.
        5. Select accessible Simulacrum map circle.
        6. Transfer Tier 15 map into Delusion popup.
        7. Click TRAVERSE button.
        8. Click spawned Map Device portal to enter Simulacrum.
        9. Automatically launch autopilot targeting Pink Dot #1 and run all rooms as usual!
        """
        if getattr(self, "hideout_routine_active", False) or getattr(self, "hideout_test_active", False):
            self.notification_msg = "HIDEOUT ROUTINE ALREADY IN PROGRESS..."
            self.notification_expiry = time.time() + 2.5
            return

        def _worker():
            self.hideout_routine_active = True
            self.hideout_test_active = True
            self.navigator.in_hideout = True
            self.navigator.is_active = False
            self.navigator.disable_persistent_combat()
            self.navigator.release_all_keys()
            self.notification_msg = "STARTING BOT FROM HIDEOUT: UNLOAD -> MAP DEVICE -> SIM -> PORTAL..."
            self.notification_expiry = time.time() + 60.0
            print("\n[VISUALIZER] >>> Starting Bot Full Routine from Hideout...")
            try:
                rep = self.navigator.run_hideout_full_cycle(
                    dry_run=dry_run,
                    traverse_and_enter=True,
                    auto_start_route=True,
                    start_pink_dot=start_pink_dot,
                )
                if rep.get("success"):
                    self.notification_msg = f"BOT ROUTINE ACTIVE: ENTERED SIMULACRUM -> TARGETING PINK #{start_pink_dot}"
                    print(f"\n[VISUALIZER] >>> Hideout Routine Success! Bot entered Simulacrum and activated navigation targeting Pink #{start_pink_dot}.")
                else:
                    msgs = rep.get("messages", [])
                    last_err = msgs[-1] if msgs else "Routine failed"
                    self.notification_msg = f"HIDEOUT ROUTINE FAILED: {last_err}"
                    print(f"\n[VISUALIZER] >>> Hideout Routine Failed: {last_err}")
            except Exception as e:
                self.notification_msg = f"HIDEOUT ROUTINE ERROR: {e}"
                print(f"\n[VISUALIZER] >>> Hideout Routine Error: {e}")
            finally:
                self.hideout_routine_active = False
                self.hideout_test_active = False
                self.notification_expiry = time.time() + 6.0

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def trigger_hideout_test(self, dry_run: bool = False, full_cycle: bool = True):
        """Launches hideout functionality test in a background thread."""
        if getattr(self, "hideout_test_active", False):
            self.notification_msg = "HIDEOUT TEST ALREADY IN PROGRESS..."
            self.notification_expiry = time.time() + 2.5
            return

        def _worker():
            self.hideout_test_active = True
            self.navigator.in_hideout = True
            self.navigator.is_active = False
            self.navigator.disable_persistent_combat()
            self.navigator.release_all_keys()
            self.notification_msg = (
                "RUNNING HIDEOUT FULL CYCLE... STASH -> ESCAPE -> MAP DEVICE -> SIMULACRUM"
                if full_cycle
                else "RUNNING HIDEOUT TEST... STASH DEPOSIT"
            )
            self.notification_expiry = time.time() + (45.0 if full_cycle else 20.0)
            try:
                rep = self.navigator.test_hideout_sequence(dry_run=dry_run, full_cycle=full_cycle)
                if rep.get("success"):
                    stashed = rep.get("items_stashed", 0)
                    if rep.get("map_transferred"):
                        self.notification_msg = f"HIDEOUT CYCLE COMPLETE: STASHED {stashed} & MAP INSERTED!"
                    else:
                        self.notification_msg = f"HIDEOUT TEST SUCCESS: STASHED {stashed} ITEMS!"
                else:
                    msgs = rep.get("messages", [])
                    last_err = msgs[-1] if msgs else "Test failed"
                    self.notification_msg = f"HIDEOUT TEST FAILED: {last_err}"
            except Exception as e:
                self.notification_msg = f"HIDEOUT TEST ERROR: {e}"
            finally:
                self.navigator.in_hideout = True
                self.navigator.is_active = False
                self.navigator.disable_persistent_combat()
                self.navigator.release_all_keys()
                self.hideout_test_active = False
                self.notification_expiry = time.time() + 6.0

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def nudge_active_room_box(self, dx: int, dy: int, dw: int, dh: int):
        """Nudges or resizes the selected room bounding box."""
        cur_box = self.movement_path.get_room_bounding_box(self.selected_room_for_edit, margin=35.0)
        rw, rh = self.map_img_size
        if not cur_box:
            cur_box = (100, 100, 300, 300)
        x1, y1, x2, y2 = cur_box
        nx1 = max(0, min(rw - 20, x1 + dx - dw))
        ny1 = max(0, min(rh - 20, y1 + dy - dh))
        nx2 = max(nx1 + 20, min(rw, x2 + dx + dw))
        ny2 = max(ny1 + 20, min(rh, y2 + dy + dh))
        self.movement_path.set_room_bounding_box(self.selected_room_for_edit, (nx1, ny1, nx2, ny2))
        self.boxes_modified = True
        self.notification_msg = f"ROOM {self.selected_room_for_edit} BOX: [{nx1}, {ny1} -> {nx2}, {ny2}]"
        self.notification_expiry = time.time() + 2.5

    def _on_mouse(self, event, x, y, flags, param):
        """Handles interactive mouse hover, button clicks, and box editing/dragging on map canvas."""
        bx, by, bw, bh = self.btn_refresh_rect
        is_refresh_inside = (bx <= x <= bx + bw and by <= y <= by + bh)
        self.btn_refresh_hover = is_refresh_inside

        px, py, pw, ph = self.btn_pink_dot_rect
        is_pink_inside = (px <= x <= px + pw and py <= y <= py + ph)
        self.btn_pink_dot_hover = is_pink_inside

        ee_x, ee_y, ee_w, ee_h = self.btn_early_exit_rect
        is_ee_inside = (ee_x <= x <= ee_x + ee_w and ee_y <= y <= ee_y + ee_h)
        self.btn_early_exit_hover = is_ee_inside

        ho_x, ho_y, ho_w, ho_h = self.btn_hideout_rect
        is_ho_inside = (ho_x <= x <= ho_x + ho_w and ho_y <= y <= ho_y + ho_h)
        self.btn_hideout_hover = is_ho_inside

        gx, gy, gw, gh = self.btn_green_light_rect
        is_green_inside = (gx <= x <= gx + gw and gy <= y <= gy + gh)
        self.btn_green_light_hover = is_green_inside

        eb_x, eb_y, eb_w, eb_h = self.btn_edit_boxes_rect
        is_edit_inside = (eb_x <= x <= eb_x + eb_w and eb_y <= y <= eb_y + eb_h)
        self.btn_edit_boxes_hover = is_edit_inside

        sb_x, sb_y, sb_w, sb_h = self.btn_save_boxes_rect
        is_save_inside = (sb_x <= x <= sb_x + sb_w and sb_y <= y <= sb_y + sb_h)
        self.btn_save_boxes_hover = is_save_inside

        # Check map area coordinates
        mo_x, mo_y, mw, mh = self.map_view_bounds
        rw, rh = self.map_img_size
        is_on_map = (mo_x <= x < mo_x + mw and mo_y <= y < mo_y + mh)

        if is_on_map and mw > 0 and mh > 0:
            map_x = int(np.clip((x - mo_x) / float(mw) * rw, 0, rw - 1))
            map_y = int(np.clip((y - mo_y) / float(mh) * rh, 0, rh - 1))
            self.cursor_map_pos = (map_x, map_y)
        else:
            self.cursor_map_pos = None

        if event == cv2.EVENT_LBUTTONDOWN:
            if is_edit_inside:
                self.edit_boxes_mode = not self.edit_boxes_mode
                self.notification_msg = f"ROOM BOX EDITOR: {'ENABLED (Click & Drag to resize boxes)' if self.edit_boxes_mode else 'DISABLED'}"
                self.notification_expiry = time.time() + 3.0
                return

            if is_save_inside:
                self.save_room_boxes()
                return

            # Check room selection buttons
            for r_i, (rx_b, ry_b, rw_b, rh_b) in self.btn_room_select_rects.items():
                if rx_b <= x <= rx_b + rw_b and ry_b <= y <= ry_b + rh_b:
                    self.selected_room_for_edit = r_i
                    self.edit_boxes_mode = True
                    self.notification_msg = f"SELECTED ROOM {r_i} FOR EDITING"
                    self.notification_expiry = time.time() + 2.5
                    return

            if is_green_inside and getattr(self.navigator, "waiting_for_green_light", False):
                self.navigator.give_green_light()
                self.notification_msg = "GREEN LIGHT GIVEN -> RESUMING ROUTE!"
                self.notification_expiry = time.time() + 3.0
                return

            if is_pink_inside:
                new_pink = self.navigator.cycle_start_pink_dot(save_to_config=True)
                if new_pink == 0:
                    self.notification_msg = "ROUTE TARGET: START (ALL WAYPOINTS ACTIVE)"
                else:
                    t_wp = getattr(self.navigator, "target_pink_wp_idx", None)
                    wp_str = f" (WP #{t_wp})" if t_wp is not None else ""
                    self.notification_msg = f"TARGETING PINK DOT #{new_pink}{wp_str} (Next Pink #{new_pink})"
                self.notification_expiry = time.time() + 3.5
                return

            if is_ee_inside:
                new_state = self.navigator.toggle_minimap_early_exit(save_to_config=True)
                min_sec = getattr(self.navigator, "minimap_early_exit_min_seconds", 25.0)
                self.notification_msg = f"MINIMAP EARLY EXIT: {'ENABLED (Trigger >= ' + str(int(min_sec)) + 's on Loot Drop)' if new_state else 'DISABLED (Full 50s Orbit)'}"
                self.notification_expiry = time.time() + 3.0
                return

            if is_ho_inside:
                self.start_hideout_full_routine()
                return

            if is_refresh_inside:
                self.reload_route()
                return

            # Map Canvas Click Interaction in Edit Mode
            if is_on_map and self.edit_boxes_mode and self.cursor_map_pos:
                mx, my = self.cursor_map_pos
                cur_box = self.movement_path.get_room_bounding_box(self.selected_room_for_edit, margin=35.0)
                handle_radius = 16  # map pixels

                if cur_box:
                    bx1, by1, bx2, by2 = cur_box
                    # Check corner handles
                    if abs(mx - bx1) <= handle_radius and abs(my - by1) <= handle_radius:
                        self.drag_mode = "corner_tl"
                    elif abs(mx - bx2) <= handle_radius and abs(my - by1) <= handle_radius:
                        self.drag_mode = "corner_tr"
                    elif abs(mx - bx1) <= handle_radius and abs(my - by2) <= handle_radius:
                        self.drag_mode = "corner_bl"
                    elif abs(mx - bx2) <= handle_radius and abs(my - by2) <= handle_radius:
                        self.drag_mode = "corner_br"
                    # Check edge handles
                    elif abs(mx - bx1) <= handle_radius and by1 <= my <= by2:
                        self.drag_mode = "edge_l"
                    elif abs(mx - bx2) <= handle_radius and by1 <= my <= by2:
                        self.drag_mode = "edge_r"
                    elif abs(my - by1) <= handle_radius and bx1 <= mx <= bx2:
                        self.drag_mode = "edge_t"
                    elif abs(my - by2) <= handle_radius and bx1 <= mx <= bx2:
                        self.drag_mode = "edge_b"
                    elif bx1 <= mx <= bx2 and by1 <= my <= by2:
                        self.drag_mode = "move"
                    else:
                        self.drag_mode = "create"
                else:
                    self.drag_mode = "create"

                self.is_dragging_box = True
                self.drag_start_map_pos = (mx, my)
                self.drag_initial_box = cur_box or (mx, my, mx, my)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging_box and self.edit_boxes_mode and self.cursor_map_pos and self.drag_initial_box:
                mx, my = self.cursor_map_pos
                ix1, iy1, ix2, iy2 = self.drag_initial_box
                sx, sy = self.drag_start_map_pos or (mx, my)
                dx, dy = mx - sx, my - sy

                if self.drag_mode == "create":
                    nx1 = min(sx, mx)
                    ny1 = min(sy, my)
                    nx2 = max(sx, mx)
                    ny2 = max(sy, my)
                elif self.drag_mode == "move":
                    nx1 = max(0, min(rw - 1, ix1 + dx))
                    ny1 = max(0, min(rh - 1, iy1 + dy))
                    nx2 = max(0, min(rw - 1, ix2 + dx))
                    ny2 = max(0, min(rh - 1, iy2 + dy))
                elif self.drag_mode == "corner_tl":
                    nx1, ny1, nx2, ny2 = min(mx, ix2 - 10), min(my, iy2 - 10), ix2, iy2
                elif self.drag_mode == "corner_tr":
                    nx1, ny1, nx2, ny2 = ix1, min(my, iy2 - 10), max(mx, ix1 + 10), iy2
                elif self.drag_mode == "corner_bl":
                    nx1, ny1, nx2, ny2 = min(mx, ix2 - 10), iy1, ix2, max(my, iy1 + 10)
                elif self.drag_mode == "corner_br":
                    nx1, ny1, nx2, ny2 = ix1, iy1, max(mx, ix1 + 10), max(my, iy1 + 10)
                elif self.drag_mode == "edge_l":
                    nx1, ny1, nx2, ny2 = min(mx, ix2 - 10), iy1, ix2, iy2
                elif self.drag_mode == "edge_r":
                    nx1, ny1, nx2, ny2 = ix1, iy1, max(mx, ix1 + 10), max(my, iy1 + 10)
                elif self.drag_mode == "edge_t":
                    nx1, ny1, nx2, ny2 = ix1, min(my, iy2 - 10), ix2, iy2
                elif self.drag_mode == "edge_b":
                    nx1, ny1, nx2, ny2 = ix1, iy1, ix2, max(my, iy1 + 10)
                else:
                    nx1, ny1, nx2, ny2 = ix1, iy1, ix2, iy2

                self.movement_path.set_room_bounding_box(self.selected_room_for_edit, (nx1, ny1, nx2, ny2))
                self.boxes_modified = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_dragging_box:
                self.is_dragging_box = False
                self.drag_mode = None
                self.drag_start_map_pos = None
                self.drag_initial_box = None
                r_box = self.movement_path.get_room_bounding_box(self.selected_room_for_edit)
                if r_box:
                    self.notification_msg = f"ROOM {self.selected_room_for_edit} BOX: [{r_box[0]}, {r_box[1]} -> {r_box[2]}, {r_box[3]}] (Click [SAVE BOXES] or press 'S')"
                    self.notification_expiry = time.time() + 4.0

    def get_or_create_capturer(self) -> ScreenCapturer:
        """Ensures an active ScreenCapturer on the current monitor index."""
        if self.capturer is None or self.capturer.monitor_idx != self.monitor_idx:
            if self.capturer is not None:
                self.capturer.close()
            self.capturer = ScreenCapturer(monitor_idx=self.monitor_idx)
        return self.capturer

    def switch_monitor(self, target_idx: Optional[int] = None):
        """Switches between connected display monitors."""
        monitors = ScreenCapturer.list_monitors()
        num_mons = max(1, len(monitors) - 1)
        if target_idx is not None:
            self.monitor_idx = target_idx
        else:
            self.monitor_idx = 1 if self.monitor_idx == 2 else 2

        self.navigator.monitor_idx = self.monitor_idx

        if self.capturer is not None:
            self.capturer.close()
            self.capturer = ScreenCapturer(monitor_idx=self.monitor_idx)
        print(f"[TRACKER VISUALIZER] Switched to Monitor {self.monitor_idx}")

    def detect_minimap_player_icon(self, minimap_crop: np.ndarray) -> Tuple[bool, int, int, Optional[Tuple[int, int, int, int]]]:
        """
        Detects the orange 'X' player/party icon on the PoE2 minimap.
        Returns: (found, center_x, center_y, bounding_box)
        """
        hsv = cv2.cvtColor(minimap_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([8, 110, 110]), np.array([26, 255, 255]))
        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_h, c_w = minimap_crop.shape[:2]
        center = (c_w // 2, c_h // 2)

        best_candidate = None
        best_dist = 999.0

        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if 5 <= area <= 250 and 3 <= w <= 24 and 3 <= h <= 24:
                cx = x + w // 2
                cy = y + h // 2
                dist = ((cx - center[0]) ** 2 + (cy - center[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_candidate = (cx, cy, (x, y, w, h))

        if best_candidate:
            return True, best_candidate[0], best_candidate[1], best_candidate[2]
        return False, center[0], center[1], None

    def update_frame(self, frame_or_screenshot: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single game frame, identifies room, detects player icon on minimap,
        updates world map, and renders composite comparison dashboard.
        """
        start_time = time.time()

        if frame_or_screenshot.shape[0] > 400 and frame_or_screenshot.shape[1] > 400:
            minimap_crop = self.extractor.extract_roi(frame_or_screenshot)
        else:
            minimap_crop = frame_or_screenshot

        self.frame_idx += 1

        # 1. Detect Player Icon on Minimap (~0.5ms)
        icon_found, icon_x, icon_y, icon_box = self.detect_minimap_player_icon(minimap_crop)

        # Get route search ROI and expected position prior based on current waypoint progress
        search_roi = None
        expected_pos = None
        if hasattr(self.movement_path, "get_active_room_bounds"):
            search_roi = self.movement_path.get_active_room_bounds(margin=60.0)
        elif hasattr(self.movement_path, "get_search_roi_for_progress"):
            search_roi = self.movement_path.get_search_roi_for_progress(margin=80.0)
        curr_target = self.movement_path.get_current_target() if hasattr(self.movement_path, "get_current_target") else None
        if curr_target and "x" in curr_target and "y" in curr_target:
            expected_pos = (float(curr_target["x"]), float(curr_target["y"]))

        # 2. Room Classification & Template Matching (Authoritative Ground-Truth Tracking: ~15-20ms)
        # RoomClassifier matches live minimap features directly to the master map layout via RANSAC affine transform.
        room_res = self.classifier.classify(minimap_crop, is_crop=True, search_roi=search_roi, expected_pos=expected_pos)
        self.cached_room_res = room_res

        # 3. Reference Map Localization (Only run when in Reference Map view or as secondary fallback)
        if self.view_mode == self.VIEW_REFERENCE_MAP or not room_res.get("character_position"):
            loc_res = self.localizer.localize_player(minimap_crop, fast_track=True, search_roi=search_roi, expected_pos=expected_pos)
        else:
            loc_res = {"player_position": None, "confidence": 0.0, "matched": False}

        if room_res.get("jump_rejected") and time.time() > self.notification_expiry:
            self.notification_msg = "JUMP ANOMALY SUPPRESSED (Holding Position)"
            self.notification_expiry = time.time() + 2.0

        # 4. Dynamic World Map Stitching (Only run when viewing World Map or throttled)
        if self.view_mode == self.VIEW_WORLD_MAP or self.cached_map_res is None or (self.frame_idx % 10 == 0):
            map_res = self.world_map.update(minimap_crop)
            self.cached_map_res = map_res
        else:
            map_res = self.cached_map_res

        # 5. Closed-loop Autonomous Route Navigation (WASD)
        # RoomClassifier is authoritative ground-truth; reference map serves as fallback
        curr_player_coords = room_res.get("character_position") or loc_res.get("player_position")
        nav_res = self.navigator.update(curr_player_coords)
        if nav_res.get("recovery_event"):
            self.notification_msg = nav_res["recovery_event"]
            self.notification_expiry = time.time() + 4.0
            self.navigator.latest_recovery_event = None

        # 6. Current Area Detection (ENEMY AREA vs HIDEOUT)
        area_info = self._update_current_area(frame_or_screenshot, minimap_crop, room_res)

        # Composite Result Dictionary
        result: Dict[str, Any] = {
            "minimap_player": {
                "found": icon_found,
                "x": icon_x,
                "y": icon_y,
                "box": icon_box,
            },
            "room": room_res,
            "world_map": map_res,
            "reference_map": loc_res,
            "navigation": nav_res,
            "area": area_info,
        }
        self.last_result = result

        # Trajectory History
        curr_pos = None
        if self.view_mode == self.VIEW_ROOM_TEMPLATE:
            curr_pos = room_res.get("character_position") or loc_res.get("player_position")
        elif self.view_mode == self.VIEW_WORLD_MAP:
            curr_pos = map_res.get("global_position")
        elif self.view_mode == self.VIEW_REFERENCE_MAP:
            curr_pos = loc_res.get("player_position") or room_res.get("character_position")

        if curr_pos is not None:
            self.trajectory_history.append({"pos": curr_pos, "time": time.time()})
            if len(self.trajectory_history) > self.history_len:
                self.trajectory_history.pop(0)

        elapsed = time.time() - start_time
        self.latency_ms = elapsed * 1000.0

        # Build Dashboard Image
        dashboard = self.render_dashboard(minimap_crop, result)
        return dashboard, result

    def _update_current_area(
        self,
        frame_or_screenshot: np.ndarray,
        minimap_crop: np.ndarray,
        room_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dynamically detects whether character is currently in an ENEMY AREA (Room 1-7 combat zone)
        or in the HIDEOUT (safe zone).
        """
        room_rec = room_res.get("recognized", False)
        room_conf = room_res.get("confidence", 0.0)
        room_name = room_res.get("room_name", "Combat Zone")
        now = time.time()

        # 1. Strong Room Match (Room 1..7) -> Unambiguously ENEMY AREA
        if room_rec and room_conf >= 0.38:
            self.area_type = "ENEMY_AREA"
            self.area_label = f"ENEMY AREA ({room_name.upper()})"
            self.area_sublabel = f"Hostile Monsters Active | Room Conf: {room_conf:.0%}"
            self.area_confidence = room_conf
            self.cached_hideout_match = False
            if getattr(self.navigator, "in_hideout", False):
                self.navigator.in_hideout = False

        # 2. Navigator already confirmed in_hideout (e.g. from stash interaction or hideout test)
        elif getattr(self.navigator, "in_hideout", False) and not room_rec:
            self.area_type = "HIDEOUT"
            self.area_label = "HIDEOUT (SAFE ZONE)"
            self.area_sublabel = "Stash & Rest Hub | No Hostiles"
            self.area_confidence = max(0.85, getattr(self.navigator, "last_hideout_confidence", 0.85))
            self.cached_hideout_match = True

        # 3. Room not recognized: check Hideout minimap layout (throttled every 0.35s)
        else:
            if (now - self.last_hideout_check_time) >= 0.35:
                self.last_hideout_check_time = now
                is_ho = self.navigator.is_in_hideout(screen=frame_or_screenshot, verbose=False, set_state=False)
                self.cached_hideout_match = is_ho

            if self.cached_hideout_match:
                self.area_type = "HIDEOUT"
                self.area_label = "HIDEOUT (SAFE ZONE)"
                self.area_sublabel = "Stash & Rest Hub | No Hostiles"
                self.area_confidence = getattr(self.navigator, "last_hideout_confidence", 0.85)
                self.navigator.in_hideout = True
                self.navigator.disable_persistent_combat()
                self.navigator.release_all_keys()
            elif (
                getattr(self.navigator, "is_active", False)
                or room_res.get("character_position") is not None
                or (now - getattr(self.navigator, "last_known_time", 0.0)) < 4.0
                or room_conf >= 0.22
            ):
                self.area_type = "ENEMY_AREA"
                self.area_label = f"ENEMY AREA ({room_name.upper()})"
                self.area_sublabel = f"Combat Zone | Room Conf: {room_conf:.0%}"
                self.area_confidence = room_conf
                if getattr(self.navigator, "in_hideout", False):
                    self.navigator.in_hideout = False
            else:
                self.area_type = "UNKNOWN"
                self.area_label = "DETECTING AREA..."
                self.area_sublabel = "Analyzing Minimap & Surroundings"
                self.area_confidence = 0.0

        return {
            "type": self.area_type,
            "label": self.area_label,
            "sublabel": self.area_sublabel,
            "confidence": self.area_confidence,
            "is_hideout": (self.area_type == "HIDEOUT"),
            "is_enemy_area": (self.area_type == "ENEMY_AREA"),
        }

    def get_active_room_template_image(self, room_res: Dict[str, Any]) -> Optional[np.ndarray]:
        """Retrieves the active room template image matched by RoomClassifier."""
        matched_variant = room_res.get("matched_variant")
        room_id = room_res.get("room_id")

        for room in self.classifier.rooms:
            if room["id"] == room_id:
                for tmpl in room.get("templates", []):
                    if tmpl.get("variant") == matched_variant:
                        return tmpl.get("img")
                if room.get("templates"):
                    return room["templates"][0].get("img")

        # Fallback to localizer ref image
        return self.localizer.ref_img

    def render_dashboard(self, minimap_crop: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Renders the enlarged visualizer comparison dashboard with interactive Room Bounding Box Editor."""
        canvas_w, canvas_h = 1440, 860
        dashboard = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        dashboard[:] = (20, 22, 26)  # Dark sleek background

        # =========================================================================
        # 1. HEADER BAR
        # =========================================================================
        cv2.rectangle(dashboard, (0, 0), (canvas_w, 54), (30, 34, 42), -1)
        cv2.line(dashboard, (0, 54), (canvas_w, 54), (55, 62, 74), 1)

        # Title on left
        cv2.putText(
            dashboard,
            "POE2 TRACKER",
            (18, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.56,
            (235, 240, 250),
            1,
            cv2.LINE_AA,
        )

        room_res = result.get("room", {})
        loc_res = result.get("reference_map", {})
        map_res = result.get("world_map", {})
        nav_res = result.get("navigation", {})
        area_info = result.get("area", {})
        is_ho = area_info.get("is_hideout", False)
        is_enemy = area_info.get("is_enemy_area", False)
        room_recognized = room_res.get("recognized", False)
        room_name = room_res.get("room_name", "Searching...")
        room_conf = room_res.get("confidence", 0.0)

        # =========================================================================
        # PROMINENT CURRENT AREA BADGE (ENEMY AREA vs HIDEOUT)
        # =========================================================================
        area_badge_x = 172
        area_badge_y = 12
        area_badge_w = 215
        area_badge_h = 30

        if is_ho:
            # HIDEOUT: Sleek Emerald Safe Zone Badge
            ab_bg = (24, 75, 34)
            ab_border = (60, 225, 115)
            ab_txt = "[SAFE] HIDEOUT (SAFE ZONE)"
            ab_txt_col = (255, 255, 255)
            ab_dot_col = (50, 255, 120)
        elif is_enemy:
            # ENEMY AREA: Vibrant Warning Crimson Badge
            ab_bg = (18, 22, 135)
            ab_border = (45, 60, 245)
            ab_label = area_info.get("label", "ENEMY AREA")
            ab_txt = f"[!] {ab_label}"
            ab_txt_col = (255, 255, 255)
            ab_dot_col = (0, 60, 255)
        else:
            # DETECTING: Dark Slate/Amber Badge
            ab_bg = (38, 44, 52)
            ab_border = (75, 150, 205)
            ab_txt = "[?] AREA: DETECTING..."
            ab_txt_col = (210, 225, 235)
            ab_dot_col = (0, 215, 255)

        cv2.rectangle(dashboard, (area_badge_x, area_badge_y), (area_badge_x + area_badge_w, area_badge_y + area_badge_h), ab_bg, -1)
        cv2.rectangle(dashboard, (area_badge_x, area_badge_y), (area_badge_x + area_badge_w, area_badge_y + area_badge_h), ab_border, 1)
        # Glowing status dot
        cv2.circle(dashboard, (area_badge_x + 14, area_badge_y + 15), 5, ab_dot_col, -1, cv2.LINE_AA)
        cv2.circle(dashboard, (area_badge_x + 14, area_badge_y + 15), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            dashboard,
            ab_txt,
            (area_badge_x + 25, area_badge_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            ab_txt_col,
            1,
            cv2.LINE_AA,
        )

        # Autopilot / System Status badge on far right of header
        is_navigating = nav_res.get("is_active", False)
        is_nav_paused = nav_res.get("is_paused", False)
        held_keys_str = nav_res.get("held_keys_str", "None")

        if is_ho:
            badge_txt = "HIDEOUT | SAFE ZONE"
            badge_bg = (140, 80, 20)
            badge_w = 210
        elif is_nav_paused:
            badge_txt = f"PAUSED: WP {nav_res.get('target_index', 0)} [F4 to Resume]"
            badge_bg = (0, 140, 230)
            badge_w = 280
        elif is_navigating:
            badge_txt = f"AUTOPILOT: [{held_keys_str}] -> WP {nav_res.get('target_index', 0)}"
            badge_bg = (0, 160, 40)
            badge_w = 260
        elif self.is_paused:
            badge_txt, badge_bg = "FEED PAUSED", (140, 80, 20)
            badge_w = 210
        elif room_recognized and room_conf >= 0.40:
            badge_txt, badge_bg = f"AUTOPILOT: READY ({room_name.upper()})", (24, 130, 48)
            badge_w = 240
        else:
            badge_txt, badge_bg = "AUTOPILOT: STANDBY", (30, 90, 140)
            badge_w = 210

        badge_x = canvas_w - badge_w - 18
        cv2.rectangle(dashboard, (badge_x, 12), (badge_x + badge_w, 42), badge_bg, -1)
        cv2.putText(dashboard, badge_txt, (badge_x + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Interactive Button: [R] REFRESH ROUTE
        btn_w, btn_h = 165, 30
        btn_x = badge_x - btn_w - 12
        btn_y = 12
        self.btn_refresh_rect = (btn_x, btn_y, btn_w, btn_h)

        now = time.time()
        is_clicked = (now - self.btn_refresh_clicked) < 0.45
        if is_clicked:
            btn_bg = (0, 180, 80)
            btn_border = (120, 255, 160)
        elif self.btn_refresh_hover:
            btn_bg = (65, 115, 105)
            btn_border = (0, 255, 255)
        else:
            btn_bg = (38, 52, 60)
            btn_border = (0, 180, 180)

        cv2.rectangle(dashboard, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), btn_bg, -1)
        cv2.rectangle(dashboard, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), btn_border, 1)
        cv2.putText(
            dashboard,
            "[R] REFRESH ROUTE",
            (btn_x + 12, btn_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255) if (self.btn_refresh_hover or is_clicked) else (200, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # Interactive Button: [P] PINK DOT TARGET
        current_pink = nav_res.get("start_at_pink_dot", 0)
        target_wp_i = nav_res.get("target_pink_wp_idx")
        pink_btn_w, pink_btn_h = 175, 30
        pink_btn_x = btn_x - pink_btn_w - 12
        pink_btn_y = 12
        self.btn_pink_dot_rect = (pink_btn_x, pink_btn_y, pink_btn_w, pink_btn_h)

        if current_pink > 0:
            wp_str = f" (WP #{target_wp_i})" if target_wp_i is not None else ""
            p_label = f"[P] NEXT: PINK #{current_pink}{wp_str}"
            p_bg = (80, 20, 75) if not self.btn_pink_dot_hover else (115, 30, 105)
            p_border = (255, 80, 230)
            p_text_col = (255, 235, 255)
        else:
            p_label = "[P] TARGET: ALL (WP #0)"
            p_bg = (45, 25, 42) if not self.btn_pink_dot_hover else (75, 40, 70)
            p_border = (160, 70, 140) if not self.btn_pink_dot_hover else (220, 100, 190)
            p_text_col = (220, 190, 215)

        cv2.rectangle(dashboard, (pink_btn_x, pink_btn_y), (pink_btn_x + pink_btn_w, pink_btn_y + pink_btn_h), p_bg, -1)
        cv2.rectangle(dashboard, (pink_btn_x, pink_btn_y), (pink_btn_x + pink_btn_w, pink_btn_y + pink_btn_h), p_border, 1)
        # Glowing pink indicator dot
        cv2.circle(dashboard, (pink_btn_x + 14, pink_btn_y + 15), 5, (255, 60, 220) if current_pink > 0 else (160, 70, 140), -1, cv2.LINE_AA)
        cv2.circle(dashboard, (pink_btn_x + 14, pink_btn_y + 15), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            dashboard,
            p_label,
            (pink_btn_x + 24, pink_btn_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            p_text_col,
            1,
            cv2.LINE_AA,
        )

        # Interactive Button: [E] EARLY EXIT TOGGLE
        early_exit_enabled = nav_res.get("minimap_early_exit_enabled", getattr(self.navigator, "minimap_early_exit_enabled", True))
        early_min_sec = nav_res.get("minimap_early_exit_min_seconds", getattr(self.navigator, "minimap_early_exit_min_seconds", 25.0))
        ee_btn_w, ee_btn_h = 160, 30
        ee_btn_x = pink_btn_x - ee_btn_w - 12
        ee_btn_y = 12
        self.btn_early_exit_rect = (ee_btn_x, ee_btn_y, ee_btn_w, ee_btn_h)

        if early_exit_enabled:
            ee_label = f"[E] EARLY EXIT: {int(early_min_sec)}s"
            ee_bg = (20, 65, 95) if not self.btn_early_exit_hover else (30, 95, 135)
            ee_border = (0, 200, 255) if not self.btn_early_exit_hover else (50, 230, 255)
            ee_text_col = (230, 245, 255)
            ee_dot_col = (0, 215, 255)
        else:
            ee_label = "[E] EARLY EXIT: OFF"
            ee_bg = (40, 42, 48) if not self.btn_early_exit_hover else (60, 64, 72)
            ee_border = (90, 95, 105) if not self.btn_early_exit_hover else (140, 145, 160)
            ee_text_col = (170, 175, 185)
            ee_dot_col = (110, 115, 125)

        cv2.rectangle(dashboard, (ee_btn_x, ee_btn_y), (ee_btn_x + ee_btn_w, ee_btn_y + ee_btn_h), ee_bg, -1)
        cv2.rectangle(dashboard, (ee_btn_x, ee_btn_y), (ee_btn_x + ee_btn_w, ee_btn_y + ee_btn_h), ee_border, 1)
        # Glowing indicator dot
        cv2.circle(dashboard, (ee_btn_x + 14, ee_btn_y + 15), 5, ee_dot_col, -1, cv2.LINE_AA)
        if early_exit_enabled:
            cv2.circle(dashboard, (ee_btn_x + 14, ee_btn_y + 15), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            dashboard,
            ee_label,
            (ee_btn_x + 24, ee_btn_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            ee_text_col,
            1,
            cv2.LINE_AA,
        )

        # Interactive Button: [H] START BOT ROUTINE (From Hideout)
        ho_btn_w, ho_btn_h = 185, 30
        if nav_res.get("waiting_for_green_light"):
            gl_w = 180
            ho_btn_x = ee_btn_x - gl_w - 12 - ho_btn_w - 12
        else:
            ho_btn_x = ee_btn_x - ho_btn_w - 12
        ho_btn_y = 12
        self.btn_hideout_rect = (ho_btn_x, ho_btn_y, ho_btn_w, ho_btn_h)

        is_running = getattr(self, "hideout_routine_active", False) or getattr(self, "hideout_test_active", False)
        if is_running:
            pulse = int(50 * np.sin(now * 8.0))
            c_val = min(255, max(140, 200 + pulse))
            ho_label = "[H] RUNNING BOT..."
            ho_bg = (18, 65, 110)
            ho_border = (0, c_val, 255)
            ho_text_col = (255, 255, 255)
            ho_dot_col = (0, 255, 255)
        elif self.btn_hideout_hover:
            ho_label = "[H] START BOT ROUTINE"
            ho_bg = (30, 85, 50)
            ho_border = (80, 255, 160)
            ho_text_col = (255, 255, 255)
            ho_dot_col = (100, 255, 180)
        elif is_ho:
            ho_label = "[H] START BOT ROUTINE"
            ho_bg = (20, 68, 38)
            ho_border = (60, 215, 120)
            ho_text_col = (255, 255, 255)
            ho_dot_col = (50, 255, 130)
        else:
            ho_label = "[H] START BOT ROUTINE"
            ho_bg = (32, 44, 38)
            ho_border = (55, 140, 85)
            ho_text_col = (200, 235, 215)
            ho_dot_col = (60, 200, 110)

        cv2.rectangle(dashboard, (ho_btn_x, ho_btn_y), (ho_btn_x + ho_btn_w, ho_btn_y + ho_btn_h), ho_bg, -1)
        cv2.rectangle(dashboard, (ho_btn_x, ho_btn_y), (ho_btn_x + ho_btn_w, ho_btn_y + ho_btn_h), ho_border, 1)
        cv2.circle(dashboard, (ho_btn_x + 14, ho_btn_y + 15), 5, ho_dot_col, -1, cv2.LINE_AA)
        cv2.circle(dashboard, (ho_btn_x + 14, ho_btn_y + 15), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            dashboard,
            ho_label,
            (ho_btn_x + 24, ho_btn_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            ho_text_col,
            1,
            cv2.LINE_AA,
        )

        # Interactive Button: [G] GREEN LIGHT (GO) when waiting for loot verification
        if nav_res.get("waiting_for_green_light"):
            gl_w, gl_h = 190, 30
            gl_x = ee_btn_x - gl_w - 12
            gl_y = 12
            self.btn_green_light_rect = (gl_x, gl_y, gl_w, gl_h)

            pulse = int(45 * np.sin(now * 6.0))
            g_val = min(255, max(160, 210 + pulse))
            if self.btn_green_light_hover:
                gl_bg = (20, 175, 75)
                gl_border = (140, 255, 190)
            else:
                gl_bg = (12, 110, 45)
                gl_border = (0, g_val, 120)

            cv2.rectangle(dashboard, (gl_x, gl_y), (gl_x + gl_w, gl_y + gl_h), gl_bg, -1)
            cv2.rectangle(dashboard, (gl_x, gl_y), (gl_x + gl_w, gl_y + gl_h), gl_border, 2)
            cv2.circle(dashboard, (gl_x + 16, gl_y + 15), 6, (0, 255, 140), -1, cv2.LINE_AA)
            cv2.circle(dashboard, (gl_x + 16, gl_y + 15), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                dashboard,
                "GREEN LIGHT (GO)",
                (gl_x + 28, gl_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            self.btn_green_light_rect = (0, 0, 0, 0)

        # High-Visibility Banner / Notification Bar
        if nav_res.get("waiting_for_green_light"):
            notif_w = 640
            notif_h = 36
            notif_x = (canvas_w - notif_w) // 2
            notif_y = 54
            pulse = int(50 * np.sin(now * 5.5))
            g_b = min(255, max(150, 200 + pulse))
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (12, 45, 25), -1)
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (0, g_b, 100), 2)

            lamp_cx = notif_x + 22
            lamp_cy = notif_y + notif_h // 2
            cv2.circle(dashboard, (lamp_cx, lamp_cy), 11, (0, 100, 40), -1, cv2.LINE_AA)
            cv2.circle(dashboard, (lamp_cx, lamp_cy), 9, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(dashboard, (lamp_cx, lamp_cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

            cv2.putText(
                dashboard,
                ">>> WAITING FOR GREEN LIGHT: VERIFY ALL LOOT PICKED UP <<<",
                (notif_x + 42, notif_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 160),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                dashboard,
                "Click [GREEN LIGHT (GO)] button above or press 'G' / 'Enter' to continue route",
                (notif_x + 42, notif_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (200, 245, 220),
                1,
                cv2.LINE_AA,
            )
        elif nav_res.get("waiting_for_user_key"):
            notif_w = 660
            notif_h = 36
            notif_x = (canvas_w - notif_w) // 2
            notif_y = 54
            key_name = str(nav_res.get("waiting_user_key_name", "F5")).upper()
            pulse = int(50 * np.sin(now * 5.5))
            b_val = min(255, max(150, 200 + pulse))
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (50, 30, 10), -1)
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (0, b_val, 255), 2)
            cv2.putText(
                dashboard,
                f">>> USER CONFIRMATION REQUIRED: PRESS [{key_name}] TO PROCEED <<<",
                (notif_x + 20, notif_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                dashboard,
                f"Press '{key_name}' or 'Enter' in game/visualizer to exit to hideout & click Stash",
                (notif_x + 20, notif_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (200, 230, 255),
                1,
                cv2.LINE_AA,
            )
        elif now < self.notification_expiry and self.notification_msg:
            notif_w = 560
            notif_h = 30
            notif_x = (canvas_w - notif_w) // 2
            notif_y = 58
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (0, 120, 50), -1)
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (0, 255, 140), 1)
            cv2.putText(
                dashboard,
                self.notification_msg,
                (notif_x + 14, notif_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        elif nav_res.get("is_orbiting"):
            rem_s = nav_res.get("orbit_remaining_sec", 0.0)
            notif_w = 540
            notif_h = 30
            notif_x = (canvas_w - notif_w) // 2
            notif_y = 58
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (20, 50, 75), -1)
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (0, 220, 255), 1)
            cv2.putText(
                dashboard,
                f">>> [ORBIT MODE] RUNNING AROUND YELLOW SHAPE: {rem_s:.1f}s LEFT <<<",
                (notif_x + 14, notif_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 240, 255),
                1,
                cv2.LINE_AA,
            )

        # =========================================================================
        # 2. LEFT PANEL: WHAT GAME SHOWS (Minimap with Detected Orange X)
        # =========================================================================
        left_x, left_y = 18, 66
        left_w, left_h = 340, 670

        cv2.rectangle(dashboard, (left_x, left_y), (left_x + left_w, left_y + left_h), (28, 32, 38), -1)
        cv2.rectangle(dashboard, (left_x, left_y), (left_x + left_w, left_y + left_h), (48, 54, 64), 1)

        cv2.rectangle(dashboard, (left_x, left_y), (left_x + left_w, left_y + 34), (38, 43, 52), -1)
        view_lbl = "EDGES" if self.show_edges else "RAW"
        if is_ho:
            mm_title = f"1. MINIMAP: HIDEOUT (SAFE) [{view_lbl}]"
            mm_title_col = (110, 255, 160)
        elif is_enemy:
            mm_title = f"1. MINIMAP: ENEMY AREA [{view_lbl}]"
            mm_title_col = (110, 150, 255)
        else:
            mm_title = f"1. MINIMAP [{view_lbl}]"
            mm_title_col = (210, 220, 240)

        cv2.putText(
            dashboard,
            mm_title,
            (left_x + 10, left_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            mm_title_col,
            1,
            cv2.LINE_AA,
        )

        # Prepare view image
        if self.show_edges:
            view_crop = self.extractor.preprocess(minimap_crop, clean_noise=True)
            view_crop_bgr = cv2.cvtColor(view_crop, cv2.COLOR_GRAY2BGR) if len(view_crop.shape) == 2 else view_crop
        else:
            view_crop_bgr = minimap_crop.copy()

        # Render detected orange 'X' player marker
        mm_player = result.get("minimap_player", {})
        px, py = mm_player.get("x", minimap_crop.shape[1] // 2), mm_player.get("y", minimap_crop.shape[0] // 2)
        icon_found = mm_player.get("found", False)
        icon_box = mm_player.get("box")

        reticle_color = (0, 255, 100) if icon_found else (0, 180, 255)
        if icon_box:
            ibx, iby, ibw, ibh = icon_box
            cv2.rectangle(view_crop_bgr, (ibx - 2, iby - 2), (ibx + ibw + 2, iby + ibh + 2), (0, 255, 255), 1)

        cv2.drawMarker(view_crop_bgr, (px, py), reticle_color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.circle(view_crop_bgr, (px, py), 12, reticle_color, 2, cv2.LINE_AA)
        cv2.circle(view_crop_bgr, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)

        # Scale to fit left panel image area
        cw, ch = view_crop_bgr.shape[1], view_crop_bgr.shape[0]
        l_img_box_w, l_img_box_h = left_w - 24, left_h - 75
        aspect = float(cw) / float(ch) if ch > 0 else 1.0
        if l_img_box_w / float(l_img_box_h) > aspect:
            disp_h = l_img_box_h
            disp_w = int(disp_h * aspect)
        else:
            disp_w = l_img_box_w
            disp_h = int(disp_w / aspect)

        resized_left = cv2.resize(view_crop_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        offset_x = left_x + 12 + (l_img_box_w - disp_w) // 2
        offset_y = left_y + 42 + (l_img_box_h - disp_h) // 2
        dashboard[offset_y : offset_y + disp_h, offset_x : offset_x + disp_w] = resized_left
        cv2.rectangle(dashboard, (offset_x, offset_y), (offset_x + disp_w, offset_y + disp_h), (75, 85, 100), 1)

        # Subtext on left
        icon_status = f"Orange X: ({px},{py})" if icon_found else f"Center: ({px},{py})"
        cv2.putText(
            dashboard,
            f"{icon_status} | {cw}x{ch}px",
            (left_x + 12, left_y + left_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (160, 220, 160) if icon_found else (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        # =========================================================================
        # 3. RIGHT PANEL: WHAT BOT DETECTS (Enlarged Master Map & Box Editor)
        # =========================================================================
        right_x, right_y = 372, 66
        right_w, right_h = 1050, 670

        cv2.rectangle(dashboard, (right_x, right_y), (right_x + right_w, right_y + right_h), (28, 32, 38), -1)
        cv2.rectangle(dashboard, (right_x, right_y), (right_x + right_w, right_y + right_h), (48, 54, 64), 1)

        # Right Panel Header Toolbar (y: right_y .. right_y + 36)
        cv2.rectangle(dashboard, (right_x, right_y), (right_x + right_w, right_y + 36), (36, 42, 52), -1)

        if is_ho and self.view_mode == self.VIEW_ROOM_TEMPLATE:
            mode_str = "HIDEOUT (SAFE ZONE)"
        else:
            mode_names = {
                self.VIEW_ROOM_TEMPLATE: "ACTIVE ROOM TEMPLATE",
                self.VIEW_WORLD_MAP: "STITCHED WORLD MAP",
                self.VIEW_REFERENCE_MAP: "FULL REFERENCE MAP",
            }
            mode_str = mode_names.get(self.view_mode, "ACTIVE ROOM TEMPLATE")
        cv2.putText(
            dashboard,
            f"2. MASTER MAP [{mode_str}]",
            (right_x + 12, right_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (210, 220, 240),
            1,
            cv2.LINE_AA,
        )

        # --- Interactive Toolbar in Right Panel Header ---
        # 1. [EDIT BOXES] button
        eb_w, eb_h = 110, 26
        eb_x = right_x + 245
        eb_y = right_y + 5
        self.btn_edit_boxes_rect = (eb_x, eb_y, eb_w, eb_h)
        if self.edit_boxes_mode:
            eb_bg = (0, 140, 200)
            eb_border = (0, 255, 255)
            eb_txt = "[B] EDITING"
            eb_col = (255, 255, 255)
        elif self.btn_edit_boxes_hover:
            eb_bg = (55, 75, 95)
            eb_border = (0, 200, 255)
            eb_txt = "[B] EDIT BOXES"
            eb_col = (220, 240, 255)
        else:
            eb_bg = (40, 48, 58)
            eb_border = (90, 110, 130)
            eb_txt = "[B] EDIT BOXES"
            eb_col = (180, 195, 210)
        cv2.rectangle(dashboard, (eb_x, eb_y), (eb_x + eb_w, eb_y + eb_h), eb_bg, -1)
        cv2.rectangle(dashboard, (eb_x, eb_y), (eb_x + eb_w, eb_y + eb_h), eb_border, 1)
        cv2.putText(dashboard, eb_txt, (eb_x + 8, eb_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, eb_col, 1, cv2.LINE_AA)

        # 2. Room Select Tabs [R1] .. [R7]
        self.btn_room_select_rects = {}
        tab_start_x = eb_x + eb_w + 10
        tab_w, tab_h = 38, 26
        for r_i in range(1, 8):
            t_x = tab_start_x + (r_i - 1) * (tab_w + 4)
            self.btn_room_select_rects[r_i] = (t_x, eb_y, tab_w, tab_h)
            is_selected = (r_i == self.selected_room_for_edit and self.edit_boxes_mode)
            if is_selected:
                t_bg = (0, 180, 230)
                t_border = (180, 255, 255)
                t_col = (10, 20, 30)
            else:
                t_bg = (42, 50, 62)
                t_border = (80, 95, 115)
                t_col = (190, 205, 220)
            cv2.rectangle(dashboard, (t_x, eb_y), (t_x + tab_w, eb_y + tab_h), t_bg, -1)
            cv2.rectangle(dashboard, (t_x, eb_y), (t_x + tab_w, eb_y + tab_h), t_border, 1)
            cv2.putText(dashboard, f"R{r_i}", (t_x + 8, eb_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, t_col, 1, cv2.LINE_AA)

        # 3. [SAVE BOXES] button
        sb_w, sb_h = 110, 26
        sb_x = tab_start_x + 7 * (tab_w + 4) + 10
        self.btn_save_boxes_rect = (sb_x, eb_y, sb_w, sb_h)
        if self.boxes_modified:
            sb_bg = (0, 160, 60)
            sb_border = (0, 255, 120)
            sb_col = (255, 255, 255)
            sb_txt = "[S] SAVE*"
        elif self.btn_save_boxes_hover:
            sb_bg = (45, 90, 65)
            sb_border = (80, 230, 140)
            sb_col = (220, 255, 230)
            sb_txt = "[S] SAVE BOXES"
        else:
            sb_bg = (35, 55, 45)
            sb_border = (60, 120, 80)
            sb_col = (160, 200, 175)
            sb_txt = "[S] SAVE BOXES"
        cv2.rectangle(dashboard, (sb_x, eb_y), (sb_x + sb_w, eb_y + sb_h), sb_bg, -1)
        cv2.rectangle(dashboard, (sb_x, eb_y), (sb_x + sb_w, eb_y + sb_h), sb_border, 1)
        cv2.putText(dashboard, sb_txt, (sb_x + 8, eb_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, sb_col, 1, cv2.LINE_AA)

        # 4. Box Dimension & Cursor Readout
        cur_sel_box = self.movement_path.get_room_bounding_box(self.selected_room_for_edit, margin=35.0)
        if cur_sel_box:
            bx1, by1, bx2, by2 = cur_sel_box
            b_w_px = bx2 - bx1
            b_h_px = by2 - by1
            info_str = f"R{self.selected_room_for_edit}: [{bx1},{by1}->{bx2},{by2}] ({b_w_px}x{b_h_px}px)"
        else:
            info_str = f"R{self.selected_room_for_edit}: (No Box)"
        if self.cursor_map_pos:
            info_str += f" | Cursor: ({self.cursor_map_pos[0]},{self.cursor_map_pos[1]})"
        cv2.putText(dashboard, info_str, (sb_x + sb_w + 14, eb_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 220, 240), 1, cv2.LINE_AA)

        # Select right view image
        right_view_img: Optional[np.ndarray] = None
        player_map_pos: Optional[Tuple[float, float]] = None
        matched_info_txt = ""

        if self.view_mode == self.VIEW_ROOM_TEMPLATE:
            if is_ho and getattr(self.navigator, "hideout_layout_tpl", None) is not None:
                right_view_img = self.navigator.hideout_layout_tpl.copy()
                matched_info_txt = f"Hideout Layout (Safe Zone) | Conf: {area_info.get('confidence', 0.85):.0%}"
            else:
                tmpl_img = self.get_active_room_template_image(room_res)
                if tmpl_img is not None:
                    right_view_img = tmpl_img.copy()
                else:
                    right_view_img = np.zeros((581, 794, 3), dtype=np.uint8)
                player_map_pos = room_res.get("character_position") or loc_res.get("player_position")
                variant = room_res.get("matched_variant", "N/A")
                matched_info_txt = f"Room: {room_res.get('room_name')} ({variant}) | Conf: {room_res.get('confidence', 0):.0%}"

        elif self.view_mode == self.VIEW_WORLD_MAP:
            if hasattr(self.world_map, "render_map_view"):
                canvas = self.world_map.render_map_view()
            elif hasattr(self.world_map, "global_canvas"):
                canvas = self.world_map.global_canvas.copy()
            else:
                canvas = np.zeros((581, 794, 3), dtype=np.uint8)
            right_view_img = canvas
            player_map_pos = map_res.get("global_position")
            matched_info_txt = f"Global Map | Stitched Tiles: {getattr(self.world_map, 'tile_count', 0)}"

        else:  # Reference Map
            loc_res = result.get("reference_map", {})
            if self.localizer.ref_img is not None:
                right_view_img = self.localizer.ref_img.copy()
                if self.localizer.red_zone_bounds:
                    rx1, rx2, ry1, ry2 = self.localizer.red_zone_bounds
                    cv2.rectangle(right_view_img, (rx1, ry1), (rx2, ry2), (0, 220, 0), 1)
            else:
                right_view_img = np.zeros((581, 794, 3), dtype=np.uint8)
            player_map_pos = loc_res.get("player_position")
            matched_info_txt = f"Ref Map: {self.localizer.ref_w}x{self.localizer.ref_h}px | Conf: {loc_res.get('confidence', 0):.0%}"

        # Draw trajectory, waypoints & bounding boxes on right view image
        if right_view_img is not None:
            rw, rh = right_view_img.shape[1], right_view_img.shape[0]
            self.map_img_size = (rw, rh)

            # 1. Planned Navigation Route & Waypoints
            if self.movement_path and self.movement_path.is_configured:
                wps = self.movement_path.get_waypoints()
                for i in range(len(wps) - 1):
                    p1 = (int(wps[i]["x"]), int(wps[i]["y"]))
                    p2 = (int(wps[i + 1]["x"]), int(wps[i + 1]["y"]))
                    if 0 <= p1[0] < rw and 0 <= p1[1] < rh and 0 <= p2[0] < rw and 0 <= p2[1] < rh:
                        cv2.line(right_view_img, p1, p2, (0, 200, 60), 2, cv2.LINE_AA)

                for wp in wps:
                    wx, wy = int(wp["x"]), int(wp["y"])
                    if 0 <= wx < rw and 0 <= wy < rh:
                        cv2.circle(right_view_img, (wx, wy), 2, (0, 255, 120), -1)

                if wps:
                    s_wp = wps[0]
                    sx, sy = int(s_wp["x"]), int(s_wp["y"])
                    if 0 <= sx < rw and 0 <= sy < rh:
                        cv2.circle(right_view_img, (sx, sy), 6, (255, 140, 0), -1, cv2.LINE_AA)
                        cv2.putText(right_view_img, "START", (max(0, sx - 16), max(12, sy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 80), 1, cv2.LINE_AA)

                    f_wp = wps[-1]
                    fx, fy = int(f_wp["x"]), int(f_wp["y"])
                    if 0 <= fx < rw and 0 <= fy < rh:
                        cv2.circle(right_view_img, (fx, fy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.putText(right_view_img, "FINISH", (max(0, fx - 16), max(12, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 100, 255), 1, cv2.LINE_AA)

                curr_target = self.movement_path.get_current_target()
                if curr_target:
                    tx, ty = int(curr_target["x"]), int(curr_target["y"])
                    if 0 <= tx < rw and 0 <= ty < rh:
                        cv2.circle(right_view_img, (tx, ty), 9, (0, 255, 255), 2, cv2.LINE_AA)

                orbit_zones = self.movement_path.get_orbit_zones()
                for zone in orbit_zones:
                    zc = zone.get("center", [0, 0])
                    z_rad = zone.get("radius", 15.0)
                    z_x, z_y = int(zc[0]), int(zc[1])
                    if 0 <= z_x < rw and 0 <= z_y < rh:
                        cv2.circle(right_view_img, (z_x, z_y), int(z_rad), (0, 220, 255), 2, cv2.LINE_AA)
                        dur = zone.get("duration", 10.0)
                        cv2.putText(
                            right_view_img,
                            f"ORBIT ({dur:.0f}s)",
                            (max(0, z_x - 28), max(12, z_y - int(z_rad) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.32,
                            (0, 240, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    pts = zone.get("perimeter_points", [])
                    if pts and len(pts) > 2:
                        np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(right_view_img, [np_pts], True, (60, 210, 255), 1, cv2.LINE_AA)

                pink_wps = self.movement_path.get_pink_waypoints() if hasattr(self.movement_path, "get_pink_waypoints") else []
                interacted_pinks = set(nav_res.get("interacted_pink_dots", []))
                active_target_pink = nav_res.get("start_at_pink_dot", 0)

                for p_idx, (wp_i, p_wp) in enumerate(pink_wps, start=1):
                    pz_pos = p_wp.get("pink_pos") or [p_wp["x"], p_wp["y"]]
                    px_m, py_m = int(pz_pos[0]), int(pz_pos[1])
                    if not (0 <= px_m < rw and 0 <= py_m < rh):
                        continue

                    is_completed = (wp_i in interacted_pinks)
                    is_target = (p_idx == active_target_pink) or (active_target_pink == 0 and p_idx == 1 and not is_completed)

                    if is_target:
                        pulse_r = int(12 + 4 * np.sin(now * 8.0))
                        cv2.circle(right_view_img, (px_m, py_m), pulse_r, (255, 60, 230), 2, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px_m, py_m), 7, (255, 60, 230), -1, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px_m, py_m), 2, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.drawMarker(right_view_img, (px_m, py_m), (255, 200, 255), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

                        badge_txt = f"NEXT: PINK #{p_idx}"
                        cv2.putText(right_view_img, badge_txt, (max(4, px_m - 35), max(14, py_m - pulse_r - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(right_view_img, badge_txt, (max(4, px_m - 35), max(14, py_m - pulse_r - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 60, 230), 1, cv2.LINE_AA)
                    elif is_completed:
                        cv2.circle(right_view_img, (px_m, py_m), 5, (90, 45, 90), -1, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px_m, py_m), 7, (130, 60, 130), 1, cv2.LINE_AA)
                        cv2.putText(right_view_img, f"PINK #{p_idx} (DONE)", (max(0, px_m - 34), max(12, py_m - 9)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 110, 150), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(right_view_img, (px_m, py_m), 6, (203, 100, 255), -1, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px_m, py_m), 9, (255, 180, 255), 1, cv2.LINE_AA)
                        cv2.putText(right_view_img, f"PINK #{p_idx}", (max(0, px_m - 20), max(12, py_m - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 180, 255), 1, cv2.LINE_AA)

                    sim_p = p_wp.get("sim_pos")
                    if sim_p:
                        sx_m, sy_m = int(sim_p[0]), int(sim_p[1])
                        if 0 <= sx_m < rw and 0 <= sy_m < rh:
                            cv2.line(right_view_img, (px_m, py_m), (sx_m, sy_m), (200, 200, 0), 1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (sx_m, sy_m), 5, (255, 255, 0), -1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (sx_m, sy_m), 7, (200, 200, 0), 1, cv2.LINE_AA)
                            cv2.putText(right_view_img, f"SIM #{p_idx}", (max(0, sx_m - 16), max(10, sy_m - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 0), 1, cv2.LINE_AA)

                    loot_p = p_wp.get("loot_pos")
                    if loot_p:
                        lx_m, ly_m = int(loot_p[0]), int(loot_p[1])
                        if 0 <= lx_m < rw and 0 <= ly_m < rh:
                            cv2.line(right_view_img, (px_m, py_m), (lx_m, ly_m), (180, 180, 180), 1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (lx_m, ly_m), 5, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (lx_m, ly_m), 7, (200, 200, 200), 1, cv2.LINE_AA)
                            cv2.putText(right_view_img, f"LOOT #{p_idx}", (max(0, lx_m - 18), max(10, ly_m - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

            # 2. Room Bounding Boxes & Interactive Handles
            if (self.show_bounding_boxes or self.edit_boxes_mode) and self.movement_path and self.movement_path.is_configured:
                cur_room_idx = self.movement_path.get_current_room_index()
                for r_i in range(1, 8):
                    r_box = self.movement_path.get_room_bounding_box(r_i, margin=35.0)
                    if r_box:
                        bx1, by1, bx2, by2 = r_box
                        bx1 = max(0, min(rw - 1, bx1))
                        by1 = max(0, min(rh - 1, by1))
                        bx2 = max(0, min(rw - 1, bx2))
                        by2 = max(0, min(rh - 1, by2))

                        is_editing_this_room = (self.edit_boxes_mode and r_i == self.selected_room_for_edit)
                        is_active_room = (r_i == cur_room_idx)

                        if is_editing_this_room:
                            # Prominent glowing yellow/cyan box with handles
                            cv2.rectangle(right_view_img, (bx1, by1), (bx2, by2), (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.rectangle(right_view_img, (bx1, by1), (bx1 + 120, by1 + 18), (0, 180, 200), -1)
                            cv2.putText(
                                right_view_img,
                                f"EDITING: ROOM {r_i}",
                                (bx1 + 4, by1 + 13),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.32,
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                            # 4 Corner Handles (Squares)
                            h_sz = 6
                            for hx, hy in [(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)]:
                                cv2.rectangle(right_view_img, (hx - h_sz, hy - h_sz), (hx + h_sz, hy + h_sz), (0, 255, 255), -1)
                                cv2.rectangle(right_view_img, (hx - h_sz, hy - h_sz), (hx + h_sz, hy + h_sz), (0, 0, 0), 1)

                            # 4 Edge Midpoint Handles
                            mid_x, mid_y = (bx1 + bx2) // 2, (by1 + by2) // 2
                            for hx, hy in [(mid_x, by1), (mid_x, by2), (bx1, mid_y), (bx2, mid_y)]:
                                cv2.rectangle(right_view_img, (hx - h_sz + 1, hy - h_sz + 1), (hx + h_sz - 1, hy + h_sz - 1), (255, 200, 0), -1)
                                cv2.rectangle(right_view_img, (hx - h_sz + 1, hy - h_sz + 1), (hx + h_sz - 1, hy + h_sz - 1), (0, 0, 0), 1)

                        elif is_active_room and not self.edit_boxes_mode:
                            cv2.rectangle(right_view_img, (bx1, by1), (bx2, by2), (0, 230, 255), 2, cv2.LINE_AA)
                            cv2.rectangle(right_view_img, (bx1, by1), (bx1 + 95, by1 + 16), (0, 140, 160), -1)
                            cv2.putText(
                                right_view_img,
                                f"ROOM {r_i} (ACTIVE)",
                                (bx1 + 4, by1 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.28,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                        else:
                            # Subtle outline for other rooms
                            col = (0, 180, 220) if is_active_room else (90, 80, 70)
                            cv2.rectangle(right_view_img, (bx1, by1), (bx2, by2), col, 1, cv2.LINE_AA)
                            cv2.putText(
                                right_view_img,
                                f"ROOM {r_i}",
                                (bx1 + 4, by1 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.26,
                                (160, 150, 140),
                                1,
                                cv2.LINE_AA,
                            )

                # Active Search ROI Gated Window (Green / Lime border)
                active_roi = None
                if hasattr(self.movement_path, "get_active_room_bounds"):
                    active_roi = self.movement_path.get_active_room_bounds(margin=60.0)
                elif hasattr(self.movement_path, "get_search_roi_for_progress"):
                    active_roi = self.movement_path.get_search_roi_for_progress(margin=80.0)

                if active_roi and not self.edit_boxes_mode:
                    ax1, ay1, ax2, ay2 = active_roi
                    ax1, by_ay1 = max(0, min(rw - 1, ax1)), max(0, min(rh - 1, ay1))
                    ax2, by_ay2 = max(0, min(rw - 1, ax2)), max(0, min(rh - 1, ay2))
                    cv2.rectangle(right_view_img, (ax1, by_ay1), (ax2, by_ay2), (50, 255, 120), 1, cv2.LINE_AA)
                    cv2.putText(
                        right_view_img,
                        f"[SEARCH ROI: ROOM {cur_room_idx}]",
                        (ax1 + 4, max(12, by_ay2 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        (50, 255, 120),
                        1,
                        cv2.LINE_AA,
                    )

            # 3. Live Character Trajectory
            if self.show_trajectory and len(self.trajectory_history) > 1:
                pts = [h["pos"] for h in self.trajectory_history if h.get("pos")]
                for i in range(len(pts) - 1):
                    p1 = (int(pts[i][0]), int(pts[i][1]))
                    p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                    if 0 <= p1[0] < rw and 0 <= p1[1] < rh and 0 <= p2[0] < rw and 0 <= p2[1] < rh:
                        cv2.line(right_view_img, p1, p2, (255, 230, 40), 2, cv2.LINE_AA)

            # Player marker
            if player_map_pos:
                mx, my = int(player_map_pos[0]), int(player_map_pos[1])
                cv2.circle(right_view_img, (mx, my), 12, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(right_view_img, (mx, my), 4, (0, 0, 255), -1, cv2.LINE_AA)
                pos_label = f"({mx},{my})"
                cv2.putText(
                    right_view_img,
                    pos_label,
                    (max(4, mx - 28), max(16, my - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Scale to fit right panel
            r_img_box_w, r_img_box_h = right_w - 24, right_h - 75
            ref_aspect = float(rw) / float(rh) if rh > 0 else 1.0
            if r_img_box_w / float(r_img_box_h) > ref_aspect:
                disp_rh = r_img_box_h
                disp_rw = int(disp_rh * ref_aspect)
            else:
                disp_rw = r_img_box_w
                disp_rh = int(disp_rw / ref_aspect)

            resized_right = cv2.resize(right_view_img, (disp_rw, disp_rh), interpolation=cv2.INTER_LINEAR)
            r_offset_x = right_x + 12 + (r_img_box_w - disp_rw) // 2
            r_offset_y = right_y + 44 + (r_img_box_h - disp_rh) // 2
            dashboard[r_offset_y : r_offset_y + disp_rh, r_offset_x : r_offset_x + disp_rw] = resized_right
            cv2.rectangle(dashboard, (r_offset_x, r_offset_y), (r_offset_x + disp_rw, r_offset_y + disp_rh), (75, 85, 100), 1)

            # Store bounds for interactive mouse dragging & coordinate mapping
            self.map_view_bounds = (r_offset_x, r_offset_y, disp_rw, disp_rh)

        # Footer info on right panel
        cv2.putText(
            dashboard,
            f"{matched_info_txt} | [V] Switch View  |  [B/Click] Edit Boxes  |  [1-7] Pick Room",
            (right_x + 12, right_y + right_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        # =========================================================================
        # 4. TELEMETRY & DIAGNOSTICS HUD
        # =========================================================================
        telemetry_y = left_y + left_h + 10
        telemetry_h = 66
        cv2.rectangle(dashboard, (left_x, telemetry_y), (canvas_w - 18, telemetry_y + telemetry_h), (30, 34, 42), -1)
        cv2.rectangle(dashboard, (left_x, telemetry_y), (canvas_w - 18, telemetry_y + telemetry_h), (52, 58, 70), 1)

        # Col 1: Minimap Player Icon Coordinates
        cv2.putText(dashboard, "MINIMAP ICON (ORANGE X)", (left_x + 14, telemetry_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        icon_str = f"X: {px}  Y: {py}"
        icon_col = (0, 255, 120) if icon_found else (0, 200, 255)
        cv2.putText(dashboard, icon_str, (left_x + 14, telemetry_y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.65, icon_col, 1, cv2.LINE_AA)

        # Col 2: Room Position
        rpos_x = left_x + 290
        cv2.putText(dashboard, "ROOM POSITION", (rpos_x, telemetry_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        if player_map_pos:
            rpos_str = f"X: {player_map_pos[0]:.0f}  Y: {player_map_pos[1]:.0f}"
        else:
            rpos_str = "SEARCHING..."
        cv2.putText(dashboard, rpos_str, (rpos_x, telemetry_y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

        # Col 3: Current Area & Location Card
        room_col_x = left_x + 560
        cv2.putText(dashboard, "CURRENT AREA & LOCATION", (room_col_x, telemetry_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        if is_ho:
            primary_loc = "[HIDEOUT] Safe Zone"
            primary_col = (60, 245, 130)
            sub_loc = "Stash & Rest Hub | No Hostiles"
            sub_col = (160, 235, 190)
        elif is_enemy:
            primary_loc = f"[ENEMY AREA] {room_name}"
            primary_col = (50, 110, 255)
            sub_loc = f"Combat Zone | Room Conf: {room_conf:.0%}" if room_recognized else "Combat Zone | Hostiles Active"
            sub_col = (160, 195, 250)
        else:
            primary_loc = "SCANNING LOCATION..."
            primary_col = (0, 215, 255)
            sub_loc = "Analyzing Minimap & Surroundings"
            sub_col = (160, 175, 190)

        cv2.putText(dashboard, primary_loc, (room_col_x, telemetry_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, primary_col, 1, cv2.LINE_AA)
        cv2.putText(dashboard, sub_loc, (room_col_x, telemetry_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.33, sub_col, 1, cv2.LINE_AA)

        # Col 4: Autopilot / Navigation Status
        nav_col_x = left_x + 890
        cv2.putText(dashboard, "AUTOPILOT / ROUTE PROGRESS", (nav_col_x, telemetry_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        wp_target = nav_res.get("target_index", 0)
        total_wps = len(self.movement_path.waypoints) if self.movement_path else 0
        if getattr(self.navigator, "in_hideout", False) or nav_res.get("in_hideout", False):
            nav_status_str = "HIDEOUT (SAFE ZONE) | Inactive"
        else:
            nav_status_str = f"WP {wp_target}/{total_wps} | Keys: [{held_keys_str}]"
        cv2.putText(dashboard, nav_status_str, (nav_col_x, telemetry_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 220, 255), 1, cv2.LINE_AA)
        run_timer_str = nav_res.get("run_elapsed_str", "0.0s")
        sims_c = nav_res.get("run_sims_count", 0)
        loot_c = nav_res.get("run_loot_count", 0)
        is_run_done = nav_res.get("run_completed", False)
        last_room = nav_res.get("run_last_room_cleared")
        if is_run_done:
            timer_txt = f"Done: {run_timer_str} | SIMs: {sims_c} | Loot: {loot_c}"
            timer_col = (0, 255, 120)
        elif nav_res.get("is_active"):
            room_txt = f" ({last_room})" if last_room else ""
            timer_txt = f"Run: {run_timer_str} | SIMs: {sims_c} | Loot: {loot_c}{room_txt}"
            timer_col = (0, 255, 255)
        else:
            timer_txt = f"Run: Ready | SIMs: {sims_c} | Loot: {loot_c}"
            timer_col = (150, 160, 175)
        cv2.putText(dashboard, timer_txt, (nav_col_x, telemetry_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.36, timer_col, 1, cv2.LINE_AA)

        # Col 5: Monitor, Performance & Combat Attack State
        perf_x = left_x + 1170
        cv2.putText(dashboard, "SYSTEM & COMBAT HUD", (perf_x, telemetry_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        combat_on = nav_res.get("persistent_combat", False)
        c_action = nav_res.get("persistent_combat_action", "key")
        c_key = nav_res.get("persistent_combat_key", "t").upper()
        mode_label = f"Key '{c_key}'" if c_action == "key" else "R-Click"
        combat_txt = f"[F3] Attack ({mode_label}): {'ON' if combat_on else 'OFF'}"
        combat_col = (0, 255, 120) if combat_on else (130, 140, 160)
        perf_txt = f"Mon: {self.monitor_idx} | {self.fps:.1f} FPS"
        cv2.putText(dashboard, perf_txt, (perf_x, telemetry_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 210, 240), 1, cv2.LINE_AA)
        cv2.putText(dashboard, combat_txt, (perf_x, telemetry_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.36, combat_col, 1, cv2.LINE_AA)

        # =========================================================================
        # 5. FOOTER / KEY SHORTCUTS BAR
        # =========================================================================
        footer_y = canvas_h - 14
        if self.edit_boxes_mode:
            shortcuts = "[1-7] Select Room  |  [Drag / Arrows] Move Box  |  [+/-] Expand/Shrink  |  [S] Save Boxes  |  [B/E] Exit Editor  |  [Q] Exit"
            shortcut_color = (0, 255, 255)
        elif nav_res.get("waiting_for_green_light"):
            shortcuts = "[G / CLICK] GREEN LIGHT (RESUME)  |  [H] Hideout Test  |  [F3 / X] Attack  |  [P] Pink Dot  |  [F4] Pause  |  [Q] Exit"
            shortcut_color = (0, 255, 160)
        else:
            shortcuts = "[A / G] Autopilot  |  [H] Test Hideout  |  [E] Early Exit  |  [P] Pink Dot  |  [B] Edit Boxes  |  [F3 / X] Attack  |  [Q] Exit"
            shortcut_color = (140, 150, 165)
        cv2.putText(
            dashboard,
            shortcuts,
            (left_x + 10, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            shortcut_color,
            1,
            cv2.LINE_AA,
        )

        return dashboard

    def save_snapshot(self, dashboard_img: np.ndarray) -> str:
        """Saves current visualizer dashboard frame to debug_output folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_count += 1
        out_dir = "debug_output"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"player_tracking_snapshot_{timestamp}.png")
        cv2.imwrite(out_path, dashboard_img)
        print(f"\n[TRACKER VISUALIZER] Saved comparison snapshot: '{out_path}'")
        return out_path

    def run(self, max_frames: Optional[int] = None, interval: float = 0.04):
        """
        Runs the live interactive player tracking visualizer window.

        :param max_frames: Optional frame limit.
        :param interval: Refresh interval in seconds (default 0.04s = ~25 FPS).
        """
        capturer = self.get_or_create_capturer()
        stop_handler.reset()

        cv2.namedWindow(self.window_title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_title, self._on_mouse)

        print("=" * 70)
        print(" PLAYER POSITION TRACKER - LIVE COMPARISON WINDOW")
        print(f" Target Display:  Monitor {self.monitor_idx} (Path of Exile 2)")
        print(" Controls:")
        print("   [H]     Start Bot from Hideout (Stash -> Map Device -> SIM -> Portal -> Route)")
        print("   [A / G] Toggle Autopilot Navigation (WASD along route)")
        print("   [B / E] Toggle Room Bounding Box Editor (interactive drag/resize)")
        print("   [1 .. 7] Select Room 1..7 for Bounding Box editing")
        print("   [S]     Save Custom Room Bounding Boxes to disk")
        print("   [P]     Cycle Pink Dot Target (Pink #1, Pink #2, Pink #3, All)")
        print("   [F4]    Pause / Resume Autopilot (maintains route position)")
        print("   [N]     Skip current Waypoint (advance to next)")
        print("   [R]     Refresh / Reload Route from route.png (or click UI button)")
        print("   [V]     Cycle Map View: Active Room Template <-> World Map <-> Ref Map")
        print("   [T]     Toggle Trajectory path trail")
        print("   [C]     Clear Trajectory history")
        print("   [Space] Pause / Resume live feed")
        print("   [M]     Switch target monitor (Monitor 1 <-> Monitor 2)")
        print("   [Q/Esc] Quit visualizer")
        print("=" * 70)

        frame_count = 0
        last_tick_time = time.time()
        last_captured_frame: Optional[np.ndarray] = None
        last_a_press_time: float = 0.0

        try:
            while not stop_handler.is_stopped():
                if not self.is_paused or last_captured_frame is None:
                    captured = capturer.capture()
                    if captured is not None:
                        last_captured_frame = captured

                if last_captured_frame is not None:
                    dashboard, _ = self.update_frame(last_captured_frame)
                    cv2.imshow(self.window_title, dashboard)

                frame_count += 1
                now = time.time()
                dt = now - last_tick_time
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                last_tick_time = now

                if max_frames and frame_count >= max_frames:
                    break

                wait_ms = max(1, int(interval * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF

                if key in [ord("q"), ord("Q"), 27]:
                    print("\n[TRACKER VISUALIZER] Exit requested.")
                    break
                elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7")]:
                    r_selected = key - ord("0")
                    self.selected_room_for_edit = r_selected
                    self.edit_boxes_mode = True
                    self.notification_msg = f"SELECTED ROOM {r_selected} FOR EDITING"
                    self.notification_expiry = time.time() + 2.5
                elif key in [ord("b"), ord("B"), ord("e"), ord("E")]:
                    self.edit_boxes_mode = not self.edit_boxes_mode
                    self.notification_msg = f"ROOM BOX EDITOR: {'ENABLED (Click & Drag to resize boxes)' if self.edit_boxes_mode else 'DISABLED'}"
                    self.notification_expiry = time.time() + 3.0
                    print(f"\n[TRACKER VISUALIZER] Room Bounding Box Editor: {'ENABLED' if self.edit_boxes_mode else 'DISABLED'}")
                elif key in [ord("s"), ord("S")]:
                    if self.edit_boxes_mode or self.boxes_modified:
                        self.save_room_boxes()
                    else:
                        if last_captured_frame is not None:
                            self.save_snapshot(dashboard)
                elif key in [ord("+"), ord("=")]:
                    self.nudge_active_room_box(0, 0, 5, 5)
                elif key in [ord("-"), ord("_")]:
                    self.nudge_active_room_box(0, 0, -5, -5)
                elif key in [ord("i"), ord("I")]:
                    self.nudge_active_room_box(0, -5, 0, 0)
                elif key in [ord("k"), ord("K")]:
                    self.nudge_active_room_box(0, 5, 0, 0)
                elif key in [ord("j"), ord("J")]:
                    self.nudge_active_room_box(-5, 0, 0, 0)
                elif key in [ord("l"), ord("L")]:
                    self.nudge_active_room_box(5, 0, 0, 0)
                elif key in [ord("g"), ord("G")]:
                    if getattr(self.navigator, "waiting_for_green_light", False):
                        self.navigator.give_green_light()
                        self.notification_msg = "GREEN LIGHT GIVEN -> RESUMING ROUTE!"
                        self.notification_expiry = time.time() + 3.0
                    else:
                        if (now - last_a_press_time) < 0.60:
                            pass  # Debounce
                        else:
                            last_a_press_time = now
                            active = self.navigator.toggle(monitor_idx=self.monitor_idx)
                            print(f"\n[AUTOPILOT] Navigation {'ACTIVATED (WASD)' if active else 'STOPPED'}")
                elif key in [13, 10]:  # Enter key
                    if getattr(self.navigator, "waiting_for_green_light", False):
                        self.navigator.give_green_light()
                        self.notification_msg = "GREEN LIGHT GIVEN -> RESUMING ROUTE!"
                        self.notification_expiry = time.time() + 3.0
                    elif getattr(self.navigator, "waiting_for_user_key", False):
                        self.navigator.confirm_user_key()
                        self.notification_msg = "CONFIRMATION RECEIVED -> PROCEEDING TO HIDEOUT!"
                        self.notification_expiry = time.time() + 3.0
                elif key in [ord("5"), 116]:  # F5 or '5' key
                    if getattr(self.navigator, "waiting_for_user_key", False):
                        self.navigator.confirm_user_key()
                        self.notification_msg = "F5 RECEIVED -> PROCEEDING TO HIDEOUT!"
                        self.notification_expiry = time.time() + 3.0
                elif key in [ord("h"), ord("H")]:
                    self.start_hideout_full_routine()
                elif key in [ord("a"), ord("A")]:
                    if self.navigator.is_simulating_key or "a" in self.navigator.held_keys or "A" in self.navigator.held_keys:
                        pass
                    elif (now - last_a_press_time) < 0.60:
                        pass  # Debounce
                    else:
                        last_a_press_time = now
                        active = self.navigator.toggle(monitor_idx=self.monitor_idx)
                        print(f"\n[AUTOPILOT] Navigation {'ACTIVATED (WASD)' if active else 'STOPPED'}")
                elif key in [ord("p"), ord("P")]:
                    new_pink = self.navigator.cycle_start_pink_dot(save_to_config=True)
                    if new_pink == 0:
                        self.notification_msg = "ROUTE TARGET: START (ALL WAYPOINTS ACTIVE)"
                    else:
                        t_wp = getattr(self.navigator, "target_pink_wp_idx", None)
                        wp_str = f" (WP #{t_wp})" if t_wp is not None else ""
                        self.notification_msg = f"TARGETING PINK DOT #{new_pink}{wp_str} (Next Pink #{new_pink})"
                    self.notification_expiry = time.time() + 3.5
                elif key == 32:  # Space
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        self.navigator.stop()
                    print(f"\n[TRACKER VISUALIZER] Live Feed {'PAUSED' if self.is_paused else 'RESUMED'}")
                elif key in [ord("v"), ord("V")]:
                    self.view_mode = (self.view_mode + 1) % 3
                    mode_names = ["ACTIVE ROOM TEMPLATE", "STITCHED WORLD MAP", "FULL REFERENCE MAP"]
                    print(f"\n[TRACKER VISUALIZER] Switched View Mode: {mode_names[self.view_mode]}")
                elif key in [ord("t"), ord("T")]:
                    self.show_trajectory = not self.show_trajectory
                    print(f"\n[TRACKER VISUALIZER] Trajectory Trail: {'ON' if self.show_trajectory else 'OFF'}")
                elif key in [ord("c"), ord("C")]:
                    self.trajectory_history.clear()
                    print("\n[TRACKER VISUALIZER] Trajectory history cleared.")
                elif key in [ord("m"), ord("M")]:
                    self.switch_monitor()
                    capturer = self.get_or_create_capturer()
                    last_captured_frame = None
                elif key in [ord("x"), ord("X")]:
                    active = self.navigator.toggle_persistent_right_click()
                    self.notification_msg = "COMBAT ATTACK: ACTIVE" if active else "COMBAT ATTACK: PAUSED"
                    self.notification_expiry = time.time() + 3.0
                elif key in [ord("e"), ord("E")]:
                    if not self.edit_boxes_mode:
                        new_state = self.navigator.toggle_minimap_early_exit(save_to_config=True)
                        min_sec = getattr(self.navigator, "minimap_early_exit_min_seconds", 25.0)
                        self.notification_msg = f"MINIMAP EARLY EXIT: {'ENABLED (Exit >= ' + str(int(min_sec)) + 's on Loot Drop)' if new_state else 'DISABLED (Full 50s Orbit)'}"
                        self.notification_expiry = time.time() + 3.0
                elif key in [ord("r"), ord("R")]:
                    self.reload_route()
                elif key in [ord("n"), ord("N")]:
                    next_wp = self.navigator.skip_current_waypoint()
                    if next_wp:
                        self.notification_msg = f"SKIPPED WP -> Target: {next_wp.get('name')}"
                        self.notification_expiry = time.time() + 3.5

        finally:
            self.navigator.release_all_keys()
            cv2.destroyAllWindows()
            if self.capturer:
                self.capturer.close()
            print("[TRACKER VISUALIZER] Window closed.")
