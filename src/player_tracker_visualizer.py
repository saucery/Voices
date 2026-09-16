"""
Player Position Tracker Visualizer Module
Provides an interactive real-time dual-view window comparing:
1. Game Minimap (live capture with exact orange 'X' player icon detection)
2. Active Room Template, Stitched World Map, or Reference Map with player localization
"""

import os
import sys
import time
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
    ):
        self.classifier = classifier or RoomClassifier()
        self.world_map = world_map or WorldMapTracker()
        self.localizer = localizer or MapLocalizer()
        self.movement_path = movement_path or MovementPath()
        self.monitor_idx = monitor_idx
        self.capturer = capturer
        self.navigator = RouteNavigator(movement_path=self.movement_path, monitor_idx=self.monitor_idx, capturer=self.capturer)
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
        self.btn_green_light_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.btn_green_light_hover: bool = False
        self.notification_msg: str = ""
        self.notification_expiry: float = 0.0

        # Optimization & Pipeline Caching State
        self.frame_idx: int = 0
        self.cached_room_res: Optional[Dict[str, Any]] = None
        self.cached_map_res: Optional[Dict[str, Any]] = None

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

    def _on_mouse(self, event, x, y, flags, param):
        """Handles interactive mouse hover and clicks on dashboard buttons."""
        bx, by, bw, bh = self.btn_refresh_rect
        is_refresh_inside = (bx <= x <= bx + bw and by <= y <= by + bh)
        self.btn_refresh_hover = is_refresh_inside

        px, py, pw, ph = self.btn_pink_dot_rect
        is_pink_inside = (px <= x <= px + pw and py <= y <= py + ph)
        self.btn_pink_dot_hover = is_pink_inside

        gx, gy, gw, gh = self.btn_green_light_rect
        is_green_inside = (gx <= x <= gx + gw and gy <= y <= gy + gh)
        self.btn_green_light_hover = is_green_inside

        if event == cv2.EVENT_LBUTTONDOWN:
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
            if is_refresh_inside:
                self.reload_route()

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

        # 2. Room Classification & Template Matching (Authoritative Ground-Truth Tracking: ~15-20ms)
        # RoomClassifier matches live minimap features directly to the master map layout via RANSAC affine transform.
        room_res = self.classifier.classify(minimap_crop, is_crop=True)
        self.cached_room_res = room_res

        # 3. Reference Map Localization (Only run when in Reference Map view or as secondary fallback)
        if self.view_mode == self.VIEW_REFERENCE_MAP or not room_res.get("character_position"):
            loc_res = self.localizer.localize_player(minimap_crop, fast_track=True)
        else:
            loc_res = {"player_position": None, "confidence": 0.0, "matched": False}

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
        """Renders the comprehensive visualizer comparison dashboard."""
        canvas_w, canvas_h = 1080, 700
        dashboard = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        dashboard[:] = (20, 22, 26)  # Dark sleek background

        # =========================================================================
        # 1. HEADER BAR
        # =========================================================================
        cv2.rectangle(dashboard, (0, 0), (canvas_w, 54), (30, 34, 42), -1)
        cv2.line(dashboard, (0, 54), (canvas_w, 54), (55, 62, 74), 1)

        cv2.putText(
            dashboard,
            "PLAYER POSITION TRACKER - PATH OF EXILE 2",
            (18, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (240, 242, 248),
            1,
            cv2.LINE_AA,
        )

        room_res = result.get("room", {})
        loc_res = result.get("reference_map", {})
        map_res = result.get("world_map", {})
        nav_res = result.get("navigation", {})
        room_recognized = room_res.get("recognized", False)
        room_name = room_res.get("room_name", "Searching...")
        room_conf = room_res.get("confidence", 0.0)

        # Status badge in header
        is_navigating = nav_res.get("is_active", False)
        is_nav_paused = nav_res.get("is_paused", False)
        held_keys_str = nav_res.get("held_keys_str", "None")

        if is_nav_paused:
            badge_txt = f"PAUSED: WP {nav_res.get('target_index', 0)} [F4 to Resume]"
            badge_bg = (0, 140, 230)
            badge_w = 280
        elif is_navigating:
            badge_txt = f"AUTOPILOT: [{held_keys_str}] -> WP {nav_res.get('target_index', 0)}"
            badge_bg = (0, 160, 40)
            badge_w = 260
        elif self.is_paused:
            badge_txt, badge_bg = "FEED PAUSED", (140, 80, 20)
            badge_w = 230
        elif room_recognized and room_conf >= 0.50:
            badge_txt, badge_bg = f"ROOM: {room_name.upper()}", (24, 130, 48)
            badge_w = 230
        else:
            badge_txt, badge_bg = "AUTO-DETECTING ROOM...", (30, 110, 180)
            badge_w = 230

        badge_x = canvas_w - badge_w - 18
        cv2.rectangle(dashboard, (badge_x, 12), (badge_x + badge_w, 42), badge_bg, -1)
        cv2.putText(dashboard, badge_txt, (badge_x + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Interactive Button: [R] REFRESH ROUTE
        btn_w, btn_h = 175, 30
        btn_x = badge_x - btn_w - 14
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
            (btn_x + 14, btn_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
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

        # Interactive Button: [G] GREEN LIGHT (GO) when waiting for loot verification
        if nav_res.get("waiting_for_green_light"):
            gl_w, gl_h = 190, 30
            gl_x = pink_btn_x - gl_w - 12
            gl_y = 12
            self.btn_green_light_rect = (gl_x, gl_y, gl_w, gl_h)

            # Pulsing neon green border
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
            # Glowing traffic light dot
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
            # Graphic Indicator: WAITING FOR GREEN LIGHT (LOOT VERIFICATION)
            notif_w = 640
            notif_h = 36
            notif_x = (canvas_w - notif_w) // 2
            notif_y = 54
            pulse = int(50 * np.sin(now * 5.5))
            g_b = min(255, max(150, 200 + pulse))
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (12, 45, 25), -1)
            cv2.rectangle(dashboard, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (0, g_b, 100), 2)

            # Circular traffic lamp icon
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
        elif now < self.notification_expiry and self.notification_msg:
            notif_w = 460
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
                0.40,
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
        panel_w, panel_h = 505, 490
        left_x, left_y = 20, 70

        cv2.rectangle(dashboard, (left_x, left_y), (left_x + panel_w, left_y + panel_h), (28, 32, 38), -1)
        cv2.rectangle(dashboard, (left_x, left_y), (left_x + panel_w, left_y + panel_h), (48, 54, 64), 1)

        cv2.rectangle(dashboard, (left_x, left_y), (left_x + panel_w, left_y + 34), (38, 43, 52), -1)
        view_lbl = "EDGES & CYAN WALLS" if self.show_edges else "RAW MINIMAP CROP"
        cv2.putText(
            dashboard,
            f"1. WHAT GAME SHOWS [{view_lbl}]",
            (left_x + 12, left_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 220, 240),
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
        # Draw bounding box around detected icon
        if icon_box:
            bx, by, bw, bh = icon_box
            cv2.rectangle(view_crop_bgr, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), (0, 255, 255), 1)

        # Target reticle
        cv2.drawMarker(view_crop_bgr, (px, py), reticle_color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.circle(view_crop_bgr, (px, py), 12, reticle_color, 2, cv2.LINE_AA)
        cv2.circle(view_crop_bgr, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)

        # Scale to fit left panel image area
        cw, ch = view_crop_bgr.shape[1], view_crop_bgr.shape[0]
        img_box_w, img_box_h = panel_w - 24, panel_h - 85
        aspect = float(cw) / float(ch) if ch > 0 else 1.0
        if img_box_w / float(img_box_h) > aspect:
            disp_h = img_box_h
            disp_w = int(disp_h * aspect)
        else:
            disp_w = img_box_w
            disp_h = int(disp_w / aspect)

        resized_left = cv2.resize(view_crop_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        offset_x = left_x + 12 + (img_box_w - disp_w) // 2
        offset_y = left_y + 42 + (img_box_h - disp_h) // 2
        dashboard[offset_y : offset_y + disp_h, offset_x : offset_x + disp_w] = resized_left
        cv2.rectangle(dashboard, (offset_x, offset_y), (offset_x + disp_w, offset_y + disp_h), (75, 85, 100), 1)

        # Subtext on left
        icon_status = f"Player Icon (Orange X): FOUND at ({px}, {py})" if icon_found else f"Player Center: ({px}, {py})"
        cv2.putText(
            dashboard,
            f"{icon_status} | Size: {cw}x{ch}px",
            (left_x + 12, left_y + panel_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (160, 220, 160) if icon_found else (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        # =========================================================================
        # 3. RIGHT PANEL: WHAT BOT DETECTS (Room Template / World Map)
        # =========================================================================
        right_x = left_x + panel_w + 14

        cv2.rectangle(dashboard, (right_x, left_y), (right_x + panel_w, left_y + panel_h), (28, 32, 38), -1)
        cv2.rectangle(dashboard, (right_x, left_y), (right_x + panel_w, left_y + panel_h), (48, 54, 64), 1)

        cv2.rectangle(dashboard, (right_x, left_y), (right_x + panel_w, left_y + 34), (38, 43, 52), -1)

        # Determine right panel view mode
        mode_names = {
            self.VIEW_ROOM_TEMPLATE: "ACTIVE ROOM TEMPLATE",
            self.VIEW_WORLD_MAP: "STITCHED WORLD MAP",
            self.VIEW_REFERENCE_MAP: "FULL REFERENCE MAP",
        }
        mode_str = mode_names.get(self.view_mode, "ACTIVE ROOM TEMPLATE")
        cv2.putText(
            dashboard,
            f"2. WHAT BOT DETECTS [{mode_str}]",
            (right_x + 12, left_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 220, 240),
            1,
            cv2.LINE_AA,
        )

        # Select right view image
        right_view_img: Optional[np.ndarray] = None
        player_map_pos: Optional[Tuple[float, float]] = None
        matched_info_txt = ""

        if self.view_mode == self.VIEW_ROOM_TEMPLATE:
            tmpl_img = self.get_active_room_template_image(room_res)
            if tmpl_img is not None:
                right_view_img = tmpl_img.copy()
            else:
                right_view_img = np.zeros((300, 300, 3), dtype=np.uint8)
            player_map_pos = room_res.get("character_position") or loc_res.get("player_position")
            variant = room_res.get("matched_variant", "N/A")
            matched_info_txt = f"Room: {room_res.get('room_name')} ({variant}) | Conf: {room_res.get('confidence', 0):.0%}"

        elif self.view_mode == self.VIEW_WORLD_MAP:
            # Stitched world map canvas
            if hasattr(self.world_map, "render_map_view"):
                canvas = self.world_map.render_map_view()
            elif hasattr(self.world_map, "global_canvas"):
                canvas = self.world_map.global_canvas.copy()
            else:
                canvas = np.zeros((300, 300, 3), dtype=np.uint8)
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
                right_view_img = np.zeros((300, 300, 3), dtype=np.uint8)
            player_map_pos = loc_res.get("player_position")
            matched_info_txt = f"Ref Map: {self.localizer.ref_w}x{self.localizer.ref_h}px | Conf: {loc_res.get('confidence', 0):.0%}"

        # Draw trajectory & player position on right view image
        if right_view_img is not None:
            rw, rh = right_view_img.shape[1], right_view_img.shape[0]

            # 1. Planned Navigation Route & Waypoints
            if self.movement_path and self.movement_path.is_configured:
                wps = self.movement_path.get_waypoints()
                # Connect waypoints with green route line
                for i in range(len(wps) - 1):
                    p1 = (int(wps[i]["x"]), int(wps[i]["y"]))
                    p2 = (int(wps[i + 1]["x"]), int(wps[i + 1]["y"]))
                    if 0 <= p1[0] < rw and 0 <= p1[1] < rh and 0 <= p2[0] < rw and 0 <= p2[1] < rh:
                        cv2.line(right_view_img, p1, p2, (0, 200, 60), 2, cv2.LINE_AA)

                # Draw waypoint nodes
                for wp in wps:
                    wx, wy = int(wp["x"]), int(wp["y"])
                    if 0 <= wx < rw and 0 <= wy < rh:
                        cv2.circle(right_view_img, (wx, wy), 2, (0, 255, 120), -1)

                # Start Node (Blue)
                s_wp = wps[0]
                sx, sy = int(s_wp["x"]), int(s_wp["y"])
                if 0 <= sx < rw and 0 <= sy < rh:
                    cv2.circle(right_view_img, (sx, sy), 6, (255, 140, 0), -1, cv2.LINE_AA)
                    cv2.putText(right_view_img, "START", (max(0, sx - 16), max(12, sy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 80), 1, cv2.LINE_AA)

                # Finish Node (Red)
                f_wp = wps[-1]
                fx, fy = int(f_wp["x"]), int(f_wp["y"])
                if 0 <= fx < rw and 0 <= fy < rh:
                    cv2.circle(right_view_img, (fx, fy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(right_view_img, "FINISH", (max(0, fx - 16), max(12, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 100, 255), 1, cv2.LINE_AA)

                # Highlight Current Target Waypoint (Yellow Ring)
                curr_target = self.movement_path.get_current_target()
                if curr_target:
                    tx, ty = int(curr_target["x"]), int(curr_target["y"])
                    if 0 <= tx < rw and 0 <= ty < rh:
                        cv2.circle(right_view_img, (tx, ty), 9, (0, 255, 255), 2, cv2.LINE_AA)

                # Yellow Shape Orbit Zones & Perimeter Trails
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
                    # Draw perimeter orbit path
                    pts = zone.get("perimeter_points", [])
                    if pts and len(pts) > 2:
                        np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(right_view_img, [np_pts], True, (60, 210, 255), 1, cv2.LINE_AA)

                # Pink Encounter Markers (Numbered & Target-Sequenced)
                pink_wps = self.movement_path.get_pink_waypoints() if hasattr(self.movement_path, "get_pink_waypoints") else []
                interacted_pinks = set(nav_res.get("interacted_pink_dots", []))
                active_target_pink = nav_res.get("start_at_pink_dot", 0)

                for p_idx, (wp_i, p_wp) in enumerate(pink_wps, start=1):
                    pz_pos = p_wp.get("pink_pos") or [p_wp["x"], p_wp["y"]]
                    px, py = int(pz_pos[0]), int(pz_pos[1])
                    if not (0 <= px < rw and 0 <= py < rh):
                        continue

                    is_completed = (wp_i in interacted_pinks)
                    is_target = (p_idx == active_target_pink) or (active_target_pink == 0 and p_idx == 1 and not is_completed)

                    if is_target:
                        # Prominent pulsing target beacon for the selected next pink dot
                        pulse_r = int(12 + 4 * np.sin(now * 8.0))
                        cv2.circle(right_view_img, (px, py), pulse_r, (255, 60, 230), 2, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px, py), 7, (255, 60, 230), -1, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.drawMarker(right_view_img, (px, py), (255, 200, 255), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

                        badge_txt = f"NEXT: PINK #{p_idx}"
                        cv2.putText(right_view_img, badge_txt, (max(4, px - 35), max(14, py - pulse_r - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(right_view_img, badge_txt, (max(4, px - 35), max(14, py - pulse_r - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 60, 230), 1, cv2.LINE_AA)
                    elif is_completed:
                        cv2.circle(right_view_img, (px, py), 5, (90, 45, 90), -1, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px, py), 7, (130, 60, 130), 1, cv2.LINE_AA)
                        cv2.putText(right_view_img, f"PINK #{p_idx} (DONE)", (max(0, px - 34), max(12, py - 9)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 110, 150), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(right_view_img, (px, py), 6, (203, 100, 255), -1, cv2.LINE_AA)
                        cv2.circle(right_view_img, (px, py), 9, (255, 180, 255), 1, cv2.LINE_AA)
                        cv2.putText(right_view_img, f"PINK #{p_idx}", (max(0, px - 20), max(12, py - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 180, 255), 1, cv2.LINE_AA)

                    # Associated Cyan SIM Marker
                    sim_p = p_wp.get("sim_pos")
                    if sim_p:
                        sx, sy = int(sim_p[0]), int(sim_p[1])
                        if 0 <= sx < rw and 0 <= sy < rh:
                            cv2.line(right_view_img, (px, py), (sx, sy), (200, 200, 0), 1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (sx, sy), 5, (255, 255, 0), -1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (sx, sy), 7, (200, 200, 0), 1, cv2.LINE_AA)
                            cv2.putText(right_view_img, f"SIM #{p_idx}", (max(0, sx - 16), max(10, sy - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 0), 1, cv2.LINE_AA)

                    # Associated White Loot Marker
                    loot_p = p_wp.get("loot_pos")
                    if loot_p:
                        lx, ly = int(loot_p[0]), int(loot_p[1])
                        if 0 <= lx < rw and 0 <= ly < rh:
                            cv2.line(right_view_img, (px, py), (lx, ly), (180, 180, 180), 1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (lx, ly), 5, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(right_view_img, (lx, ly), 7, (200, 200, 200), 1, cv2.LINE_AA)
                            cv2.putText(right_view_img, f"LOOT #{p_idx}", (max(0, lx - 18), max(10, ly - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

            # 2. Live Character Trajectory
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
            ref_aspect = float(rw) / float(rh) if rh > 0 else 1.0
            if img_box_w / float(img_box_h) > ref_aspect:
                disp_rh = img_box_h
                disp_rw = int(disp_rh * ref_aspect)
            else:
                disp_rw = img_box_w
                disp_rh = int(disp_rw / ref_aspect)

            resized_right = cv2.resize(right_view_img, (disp_rw, disp_rh), interpolation=cv2.INTER_LINEAR)
            r_offset_x = right_x + 12 + (img_box_w - disp_rw) // 2
            r_offset_y = left_y + 42 + (img_box_h - disp_rh) // 2
            dashboard[r_offset_y : r_offset_y + disp_rh, r_offset_x : r_offset_x + disp_rw] = resized_right
            cv2.rectangle(dashboard, (r_offset_x, r_offset_y), (r_offset_x + disp_rw, r_offset_y + disp_rh), (75, 85, 100), 1)

        cv2.putText(
            dashboard,
            f"{matched_info_txt} | Press [V] to switch view",
            (right_x + 12, left_y + panel_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        # =========================================================================
        # 4. TELEMETRY & DIAGNOSTICS HUD
        # =========================================================================
        telemetry_y = left_y + panel_h + 12
        telemetry_h = 60
        cv2.rectangle(dashboard, (left_x, telemetry_y), (canvas_w - 20, telemetry_y + telemetry_h), (30, 34, 42), -1)
        cv2.rectangle(dashboard, (left_x, telemetry_y), (canvas_w - 20, telemetry_y + telemetry_h), (52, 58, 70), 1)

        # Col 1: Minimap Player Icon Coordinates
        cv2.putText(dashboard, "MINIMAP ICON (ORANGE X)", (left_x + 14, telemetry_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        icon_str = f"X: {px}  Y: {py}"
        icon_col = (0, 255, 120) if icon_found else (0, 200, 255)
        cv2.putText(dashboard, icon_str, (left_x + 14, telemetry_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, icon_col, 1, cv2.LINE_AA)

        # Col 2: Room Position
        rpos_x = left_x + 260
        cv2.putText(dashboard, "ROOM POSITION", (rpos_x, telemetry_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        if player_map_pos:
            rpos_str = f"X: {player_map_pos[0]:.0f}  Y: {player_map_pos[1]:.0f}"
        else:
            rpos_str = "SEARCHING..."
        cv2.putText(dashboard, rpos_str, (rpos_x, telemetry_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

        # Col 3: Active Room & Confidence
        room_col_x = left_x + 510
        cv2.putText(dashboard, "ACTIVE ROOM IDENTIFICATION", (room_col_x, telemetry_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        room_disp = f"{room_name} ({room_conf:.0%})" if room_recognized else "Unknown Room"
        cv2.putText(dashboard, room_disp, (room_col_x, telemetry_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 235), 1, cv2.LINE_AA)

        # Col 4: Monitor, Performance & Combat Attack State
        perf_x = left_x + 800
        cv2.putText(dashboard, "SYSTEM & COMBAT HUD", (perf_x, telemetry_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 160, 175), 1, cv2.LINE_AA)
        combat_on = nav_res.get("persistent_combat", False)
        combat_txt = "[F3] Attack: ON" if combat_on else "[F3] Attack: OFF"
        combat_col = (0, 255, 120) if combat_on else (130, 140, 160)
        perf_txt = f"Mon: {self.monitor_idx} | {self.fps:.1f} FPS | "
        cv2.putText(dashboard, perf_txt, (perf_x, telemetry_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 210, 240), 1, cv2.LINE_AA)
        t_sz = cv2.getTextSize(perf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        cv2.putText(dashboard, combat_txt, (perf_x + t_sz[0], telemetry_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, combat_col, 1, cv2.LINE_AA)

        # =========================================================================
        # 5. FOOTER / KEY SHORTCUTS BAR
        # =========================================================================
        footer_y = canvas_h - 20
        if nav_res.get("waiting_for_green_light"):
            shortcuts = "[G / CLICK] GREEN LIGHT (RESUME)  |  [F3 / X] Pause/Resume Attack  |  [P] Pink Dot  |  [F4] Pause  |  [N] Skip WP  |  [R] Reload  |  [Q] Exit"
            shortcut_color = (0, 255, 160)
        else:
            shortcuts = "[A / G] Autopilot  |  [F3 / X] Attack ON/OFF  |  [P] Pink Dot  |  [F4] Pause/Resume  |  [N] Skip WP  |  [R] Reload  |  [V] View  |  [Q] Exit"
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
        print("   [A / G] Toggle Autopilot Navigation (WASD along route)")
        print("   [P]     Cycle Pink Dot Target (Pink #1, Pink #2, Pink #3, All)")
        print("   [F4]    Pause / Resume Autopilot (maintains route position)")
        print("   [N]     Skip current Waypoint (advance to next)")
        print("   [R]     Refresh / Reload Route from route.png (or click UI button)")
        print("   [V]     Cycle Map View: Active Room Template <-> World Map <-> Ref Map")
        print("   [E]     Toggle Preprocessed Edges & Cyan Wall features")
        print("   [T]     Toggle Trajectory path trail")
        print("   [C]     Clear Trajectory history")
        print("   [Space] Pause / Resume live feed")
        print("   [M]     Switch target monitor (Monitor 1 <-> Monitor 2)")
        print("   [S]     Save comparison snapshot image")
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
                elif key in [ord("a"), ord("A")]:
                    # Ignore simulated keypresses from the bot's own WASD walking
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
                elif key in [ord("e"), ord("E")]:
                    self.show_edges = not self.show_edges
                    print(f"\n[TRACKER VISUALIZER] Preprocessed Edge View: {'ON' if self.show_edges else 'OFF'}")
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
                elif key in [ord("s"), ord("S")]:
                    if last_captured_frame is not None:
                        self.save_snapshot(dashboard)
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
