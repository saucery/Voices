"""
Movement Path Definition & Navigation Route Module
Handles loading, tracking, and managing bot movement waypoints from a dedicated movement file.
This module decouples map layout representation from bot movement instructions.
"""

import os
import json
import math
from typing import Dict, Any, List, Optional, Tuple


class MovementPath:
    """
    Manages navigation routes and waypoints loaded from a movement definition file.
    Defines how the bot should move across the map layout.
    """

    def __init__(
        self,
        movement_file_path: Optional[str] = None,
        config_path: str = "config.json",
    ):
        """
        Initialize MovementPath.

        :param movement_file_path: Optional path to movement definition JSON file.
        :param config_path: Path to config.json (used to read default movement_file).
        """
        self.config_path = config_path
        self.file_path: Optional[str] = movement_file_path
        self.route_name: str = "Default Route"
        self.description: str = ""
        self.arrival_distance: float = 25.0  # px threshold on reference map
        self.loop: bool = False
        self.waypoints: List[Dict[str, Any]] = []
        self.orbit_zones: List[Dict[str, Any]] = []
        self.pink_zones: List[Dict[str, Any]] = []
        self.cyan_zones: List[Dict[str, Any]] = []
        self.loot_zones: List[Dict[str, Any]] = []
        self.orbit_duration_seconds: float = 10.0
        self.orbit_margin_px: float = 8.0
        self.orbit_mode: str = "inside"
        self.current_idx: int = 0
        self.is_loaded: bool = False

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if not self.file_path:
                        self.file_path = cfg.get("movement_file", "paths/movement_route.json")
                    ap_cfg = cfg.get("autopilot", {})
                    self.orbit_duration_seconds = float(ap_cfg.get("orbit_duration_seconds", cfg.get("orbit_duration_seconds", 10.0)))
                    self.orbit_margin_px = float(ap_cfg.get("orbit_margin_px", cfg.get("orbit_margin_px", 8.0)))
                    self.orbit_mode = str(ap_cfg.get("orbit_mode", cfg.get("orbit_mode", "inside"))).lower()
            except Exception:
                if not self.file_path:
                    self.file_path = "paths/movement_route.json"

        # If default json path is used and doesn't exist, check for templates/route.png
        if self.file_path in ("paths/movement_route.json", None) and not os.path.exists(self.file_path or ""):
            if os.path.exists("templates/route.png"):
                self.file_path = "templates/route.png"

        if self.file_path:
            self.load(self.file_path)

    def load(self, file_path: str) -> bool:
        """
        Loads movement definition from JSON file OR directly from a painted route image.

        :param file_path: Path to movement route JSON or painted PNG file.
        :return: True if loaded successfully with valid waypoints.
        """
        self.file_path = file_path

        # Direct Image Extraction Mode (.png / .jpg)
        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            return self.load_from_painted_image(file_path)

        # Automatic re-extraction if templates/route.png is newer than paths/movement_route.json
        if file_path.lower().endswith(".json") and os.path.exists("templates/route.png") and os.path.exists(file_path):
            try:
                img_mtime = os.path.getmtime("templates/route.png")
                json_mtime = os.path.getmtime(file_path)
                if img_mtime > json_mtime:
                    print(f"[MOVEMENT] Detected updated 'templates/route.png'. Auto re-extracting route into '{file_path}'...")
                    if self.load_from_painted_image("templates/route.png", save_json_path=file_path):
                        return True
            except Exception as e:
                print(f"[MOVEMENT] Timestamp check warning: {e}")

        if not os.path.exists(file_path):
            # Only auto-extract for default route file
            if file_path in ("paths/movement_route.json", "") and os.path.exists("templates/route.png"):
                print(f"[MOVEMENT] Parsing painted route image 'templates/route.png' into '{file_path}'...")
                return self.load_from_painted_image("templates/route.png", save_json_path=file_path)

            print(f"[MOVEMENT] Movement file '{file_path}' not found (pending implementation).")
            self.is_loaded = False
            self.waypoints = []
            self.orbit_zones = []
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_waypoints = data.get("waypoints", [])
            # If JSON has 0 waypoints, attempt to auto-extract from templates/route.png if present
            if len(raw_waypoints) == 0 and os.path.exists("templates/route.png"):
                print(f"[MOVEMENT] '{file_path}' has 0 waypoints. Auto-extracting from 'templates/route.png'...")
                return self.load_from_painted_image("templates/route.png", save_json_path=file_path)

            self.route_name = data.get("route_name", "Custom Route")
            self.description = data.get("description", "")
            settings = data.get("settings", {})
            self.arrival_distance = float(settings.get("arrival_distance", 25.0))
            self.loop = bool(settings.get("loop", False))
            self.orbit_duration_seconds = float(settings.get("orbit_duration_seconds", self.orbit_duration_seconds))
            self.orbit_mode = str(settings.get("orbit_mode", self.orbit_mode)).lower()
            self.orbit_zones = data.get("orbit_zones", [])
            self.pink_zones = data.get("pink_zones", [])
            self.cyan_zones = data.get("cyan_zones", [])
            self.loot_zones = data.get("loot_zones", [])

            self.waypoints = []
            for idx, wp in enumerate(raw_waypoints):
                if "x" in wp and "y" in wp:
                    self.waypoints.append({
                        "index": idx,
                        "name": wp.get("name", f"Waypoint {idx + 1}"),
                        "x": float(wp["x"]),
                        "y": float(wp["y"]),
                        "action": wp.get("action", "walk"),
                        "wait_after": float(wp.get("wait_after", 0.0)),
                        "orbit_zone": wp.get("orbit_zone", None),
                        "pink_pos": wp.get("pink_pos", None),
                        "sim_pos": wp.get("sim_pos", None),
                        "loot_pos": wp.get("loot_pos", None),
                    })

            self.current_idx = 0
            self.is_loaded = len(self.waypoints) > 0
            if self.is_loaded:
                zones_str = f" | {len(self.orbit_zones)} yellow orbit zone(s)" if self.orbit_zones else ""
                pink_str = f" | {len(self.pink_zones)} pink marker(s)" if self.pink_zones else ""
                print(f"[MOVEMENT] Successfully loaded '{self.route_name}' with {len(self.waypoints)} waypoints{zones_str}{pink_str} from '{file_path}'.")
            else:
                print(f"[MOVEMENT] Movement file '{file_path}' contains 0 waypoints (pending implementation).")
            return self.is_loaded
        except Exception as e:
            print(f"[MOVEMENT] Warning: Could not parse movement file '{file_path}': {e}")
            self.is_loaded = False
            self.waypoints = []
            self.orbit_zones = []
            return False

    def load_from_painted_image(
        self,
        image_path: str,
        spacing: float = 20.0,
        save_json_path: Optional[str] = None,
    ) -> bool:
        """
        Extracts navigation waypoints from a painted map layout image:
        - Blue dot = Start Position
        - Green line = Navigation Route
        - Red dot = Finish / Destination Position

        :param image_path: Path to painted route image.
        :param spacing: Pixel distance spacing between consecutive waypoints.
        :param save_json_path: Optional path to save extracted route as JSON.
        :return: True if waypoints were successfully extracted.
        """
        import cv2
        import numpy as np
        from collections import deque

        if not os.path.exists(image_path):
            print(f"[MOVEMENT] Painted route image '{image_path}' not found.")
            return False

        img = cv2.imread(image_path)
        if img is None:
            print(f"[MOVEMENT] Could not read image at '{image_path}'.")
            return False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        b_chan, g_chan, r_chan = cv2.split(img)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # 0a. Detect Cyan Dots (SIM Locations: B >= 180, G >= 180, R <= 100, H 80..98, S >= 100, V >= 150)
        cyan_mask_bool = (b_chan >= 180) & (g_chan >= 180) & (r_chan <= 100) & (h_chan >= 80) & (h_chan <= 98) & (s_chan >= 100) & (v_chan >= 150)
        cyan_mask = (cyan_mask_bool.astype(np.uint8) * 255)
        c_cnts, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cyan_zones = []
        for cc in c_cnts:
            area = cv2.contourArea(cc)
            if area >= 8.0:
                M = cv2.moments(cc)
                if M["m00"] > 0:
                    cx_c = int(M["m10"] / M["m00"])
                    cy_c = int(M["m01"] / M["m00"])
                    self.cyan_zones.append({
                        "id": f"cyan_{len(self.cyan_zones) + 1}",
                        "x": cx_c,
                        "y": cy_c,
                        "area": float(area),
                    })

        # 0b. Detect White Dots (Loot Locations: R >= 220, G >= 220, B >= 220, S <= 40, V >= 220)
        white_mask_bool = (r_chan >= 220) & (g_chan >= 220) & (b_chan >= 220) & (s_chan <= 40) & (v_chan >= 220)
        white_mask = (white_mask_bool.astype(np.uint8) * 255)
        w_cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.loot_zones = []
        for wc in w_cnts:
            area = cv2.contourArea(wc)
            if area >= 8.0:
                M = cv2.moments(wc)
                if M["m00"] > 0:
                    wx_c = int(M["m10"] / M["m00"])
                    wy_c = int(M["m01"] / M["m00"])
                    if wx_c <= 5 or wy_c <= 5 or wx_c >= (w - 5) or wy_c >= (h - 5):
                        continue
                    self.loot_zones.append({
                        "id": f"loot_{len(self.loot_zones) + 1}",
                        "x": wx_c,
                        "y": wy_c,
                        "area": float(area),
                    })

        # 1. Detect Blue Start Dot (H: 98..135, S: 80..255, V: 80..255, B > G + 30, B > 140, R < 120)
        blue_mask_raw = cv2.inRange(hsv, (98, 80, 80), (140, 255, 255))
        is_blue_color = (b_chan.astype(int) > g_chan.astype(int) + 30) & (b_chan > 140) & (r_chan < 120)
        blue_mask = cv2.bitwise_and(blue_mask_raw, (is_blue_color.astype(np.uint8) * 255))
        blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(cyan_mask))
        blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(white_mask))
        b_cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        start_pt = None
        best_b_area = 0.0
        for c in b_cnts:
            area = cv2.contourArea(c)
            if area > 8.0 and area > best_b_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    start_pt = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    best_b_area = area

        # 2. Detect Red Finish Dot (H: 0..12 or 168..180, S: 80..255, V: 80..255, B < 80, G < 80)
        red1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
        red2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
        red_mask_raw = cv2.bitwise_or(red1, red2)
        is_red_color = (r_chan >= 120) & (b_chan < 80) & (g_chan < 80)
        red_mask = cv2.bitwise_and(red_mask_raw, (is_red_color.astype(np.uint8) * 255))
        r_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        finish_pt = None
        best_r_area = 0.0
        for c in r_cnts:
            area = cv2.contourArea(c)
            if area > 8.0 and area > best_r_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    finish_pt = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    best_r_area = area

        # 3. Detect Green Navigation Path (H: 35..88, S: 50..255, V: 50..255)
        green_mask = cv2.inRange(hsv, (35, 50, 50), (88, 255, 255))
        # Bridge minor brush gaps with morphological closing
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        path_canvas = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = np.ones((3, 3), np.uint8)
        path_canvas = cv2.dilate(path_canvas, dilate_kernel, iterations=1)
        gy_route, gx_route = np.where(path_canvas > 0)
        route_pts = np.column_stack((gx_route, gy_route)) if len(gx_route) > 0 else np.empty((0, 2))

        # 4. Detect Painted Yellow Shapes / Orbit Zones
        # Differentiates bright painted yellow (R>140, G>140, B<120, Hue 18..38, S>=90, V>=130)
        # from murky/brown/olive background terrain texture
        b_chan, g_chan, r_chan = cv2.split(img)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        yellow_mask_bool = (r_chan > 140) & (g_chan > 140) & (b_chan < 120) & (h_chan >= 18) & (h_chan <= 38) & (v_chan >= 130) & (s_chan >= 90)
        yellow_mask = (yellow_mask_bool.astype(np.uint8) * 255)
        open_kernel = np.ones((3, 3), np.uint8)
        yellow_mask_clean = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, open_kernel)

        y_cnts, _ = cv2.findContours(yellow_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.orbit_zones = []
        for yc in y_cnts:
            area = cv2.contourArea(yc)
            if area >= 35.0:
                (cx, cy), radius = cv2.minEnclosingCircle(yc)
                if radius < 5.0:
                    continue

                # Must intersect or be within reach of the route line
                if len(route_pts) > 0:
                    d_to_route = np.min(np.hypot(route_pts[:, 0] - cx, route_pts[:, 1] - cy))
                    if d_to_route > (radius + 35.0):
                        continue  # Far away map texture artifact, ignore

                bx, by, bw, bh = cv2.boundingRect(yc)
                radius = max(float(radius), 12.0)

                if self.orbit_mode == "inside":
                    # Orbit strictly INSIDE the yellow field
                    inner_margin = max(4.0, self.orbit_margin_px)
                    orbit_radius = max(6.0, radius - inner_margin if (radius - inner_margin) >= 6.0 else radius * 0.65)
                else:
                    orbit_radius = radius + self.orbit_margin_px

                # 8-point perimeter orbit path around / inside shape
                perimeter_pts = []
                for deg in [0, 45, 90, 135, 180, 225, 270, 315]:
                    rad = math.radians(deg)
                    px = round(cx + orbit_radius * math.cos(rad), 1)
                    py = round(cy + orbit_radius * math.sin(rad), 1)

                    # Ensure points are strictly inside the contour if orbit_mode == "inside"
                    if self.orbit_mode == "inside":
                        scale = 0.90
                        while cv2.pointPolygonTest(yc, (float(px), float(py)), False) < 0 and scale > 0.2:
                            px = round(cx + (orbit_radius * scale) * math.cos(rad), 1)
                            py = round(cy + (orbit_radius * scale) * math.sin(rad), 1)
                            scale -= 0.1

                    perimeter_pts.append([px, py])

                self.orbit_zones.append({
                    "id": f"zone_{len(self.orbit_zones) + 1}",
                    "center": [round(float(cx), 1), round(float(cy), 1)],
                    "radius": round(float(radius), 1),
                    "orbit_radius": round(float(orbit_radius), 1),
                    "orbit_mode": self.orbit_mode,
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "duration": self.orbit_duration_seconds,
                    "perimeter_points": perimeter_pts,
                })

        # If yellow shapes exist, incorporate yellow_mask into path_canvas so BFS connects continuously
        if len(self.orbit_zones) > 0:
            path_canvas = cv2.bitwise_or(path_canvas, yellow_mask_clean)

        # 5. Detect Painted Pink Dots / Sim & Encounter Markers
        # Robust multi-shade pink / magenta / rose detector covering:
        # standard pink (RGB 255, 174, 201), hot pink, magenta, deep pink, light pink, pastel pink, neon pink.
        hsv_mask1 = cv2.inRange(hsv, (125, 25, 70), (180, 255, 255))
        hsv_mask2 = cv2.inRange(hsv, (0, 25, 70), (10, 255, 255))
        hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)

        is_pink = (hsv_mask > 0) & (r_chan.astype(int) > g_chan.astype(int) + 10) & (r_chan >= 100)
        has_blue = (b_chan.astype(int) > (g_chan.astype(int) * 0.65)) | (b_chan >= 60) | (h_chan <= 168)
        is_pink = is_pink & has_blue

        # Explicit exclusions of other map features (yellow zones, blue start, green line, pure red finish, cyan, white)
        not_yellow = ~((r_chan > 140) & (g_chan > 140) & (b_chan < 130))
        not_blue = ~((b_chan > 140) & (r_chan < 100))
        not_cyan = ~cyan_mask_bool
        not_white = ~white_mask_bool
        not_green = ~(g_chan.astype(int) > r_chan.astype(int) + 15)
        not_pure_red = ~((r_chan >= 120) & (b_chan < 55) & (g_chan < 55) & ((h_chan <= 10) | (h_chan >= 170)))

        pink_mask_bool = is_pink & not_yellow & not_blue & not_cyan & not_white & not_green & not_pure_red
        pink_mask = (pink_mask_bool.astype(np.uint8) * 255)
        pink_mask_clean = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, open_kernel)

        p_cnts, _ = cv2.findContours(pink_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.pink_zones = []
        for pc in p_cnts:
            area = cv2.contourArea(pc)
            if area >= 8.0:
                M = cv2.moments(pc)
                if M["m00"] > 0:
                    px_c = int(M["m10"] / M["m00"])
                    py_c = int(M["m01"] / M["m00"])
                    if len(route_pts) > 0:
                        d_to_route = np.min(np.hypot(route_pts[:, 0] - px_c, route_pts[:, 1] - py_c))
                        if d_to_route > 150.0:
                            continue
                    self.pink_zones.append({
                        "id": f"pink_{len(self.pink_zones) + 1}",
                        "x": px_c,
                        "y": py_c,
                        "area": float(area),
                    })

        # Incorporate pink, cyan, and white dots into path_canvas so route can bridge through cleanly
        if len(self.pink_zones) > 0:
            path_canvas = cv2.bitwise_or(path_canvas, pink_mask_clean)
        if len(self.cyan_zones) > 0:
            path_canvas = cv2.bitwise_or(path_canvas, cyan_mask)
        if len(self.loot_zones) > 0:
            path_canvas = cv2.bitwise_or(path_canvas, white_mask)

        # Fallback endpoints if dots were omitted (use line extremities)
        if start_pt is None or finish_pt is None:
            gy, gx = np.where(path_canvas > 0)
            if len(gx) < 10:
                print("[MOVEMENT] No green route line or endpoints detected in image.")
                return False
            if start_pt is None:
                start_pt = (int(gx[0]), int(gy[0]))
            if finish_pt is None:
                finish_pt = (int(gx[-1]), int(gy[-1]))

        # Bridge distance from start_pt and finish_pt directly to closest green route pixels
        gy, gx = np.where(path_canvas > 0)
        if len(gx) > 0:
            g_pts = np.column_stack((gx, gy))
            if start_pt is not None:
                d_start = np.hypot(g_pts[:, 0] - start_pt[0], g_pts[:, 1] - start_pt[1])
                near_start = tuple(map(int, g_pts[np.argmin(d_start)]))
                cv2.line(path_canvas, start_pt, near_start, 255, 5)
            if finish_pt is not None:
                d_fin = np.hypot(g_pts[:, 0] - finish_pt[0], g_pts[:, 1] - finish_pt[1])
                near_fin = tuple(map(int, g_pts[np.argmin(d_fin)]))
                cv2.line(path_canvas, finish_pt, near_fin, 255, 5)

        sx, sy = start_pt
        fx, fy = finish_pt
        path_canvas[max(0, sy - 5):min(h, sy + 6), max(0, sx - 5):min(w, sx + 6)] = 255
        path_canvas[max(0, fy - 5):min(h, fy + 6), max(0, fx - 5):min(w, fx + 6)] = 255

        # 5. BFS Breadth-First Path Tracer along Green Route
        visited = np.zeros((h, w), dtype=bool)
        parent = {}
        queue = deque([start_pt])
        visited[sy, sx] = True

        found = False
        finish_reached = None

        while queue:
            cx, cy = queue.popleft()
            if abs(cx - fx) <= 3 and abs(cy - fy) <= 3:
                finish_reached = (cx, cy)
                found = True
                break

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and path_canvas[ny, nx] > 0:
                    visited[ny, nx] = True
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

        if not found:
            print("[MOVEMENT] Could not find continuous green path connecting Blue Start and Red Finish.")
            return False

        # Reconstruct path from start to finish
        full_path = []
        curr = finish_reached
        while curr in parent:
            full_path.append(curr)
            curr = parent[curr]
        full_path.append(start_pt)
        full_path.reverse()

        # 6. Downsample path to evenly spaced waypoints
        sampled = [full_path[0]]
        accum = 0.0
        for i in range(1, len(full_path)):
            dx = full_path[i][0] - full_path[i - 1][0]
            dy = full_path[i][1] - full_path[i - 1][1]
            accum += math.hypot(dx, dy)
            if accum >= spacing:
                sampled.append(full_path[i])
                accum = 0.0

        if sampled[-1] != finish_pt:
            sampled.append(finish_pt)

        self.route_name = "Painted Route"
        self.description = f"Auto-extracted from {image_path} (Blue start, Green path, Red finish)"
        self.waypoints = [
            {
                "index": i,
                "name": "Start" if i == 0 else ("Finish" if i == len(sampled) - 1 else f"Waypoint {i}"),
                "x": float(pt[0]),
                "y": float(pt[1]),
                "action": "walk" if i < len(sampled) - 1 else "interact",
                "wait_after": 0.0,
                "orbit_zone": None,
                "pink_pos": None,
                "sim_pos": None,
                "loot_pos": None,
            }
            for i, pt in enumerate(sampled)
        ]

        # Pre-link ALL detected pink dots to their nearest Cyan SIM, White Loot, and Yellow Orbit zone
        for p_i, pz in enumerate(self.pink_zones, start=1):
            pz["id"] = f"pink_{p_i}"
            px_c, py_c = pz["x"], pz["y"]

            # Associate nearest cyan dot (SIM location) within 150px
            best_c = None
            best_c_dist = float("inf")
            for cz in self.cyan_zones:
                d = math.hypot(cz["x"] - px_c, cz["y"] - py_c)
                if d < best_c_dist:
                    best_c_dist = d
                    best_c = cz
            if best_c is not None and best_c_dist <= 150.0:
                pz["sim_pos"] = [best_c["x"], best_c["y"]]
                best_c["associated_pink"] = pz["id"]

            # Associate nearest white dot (LOOT location) within 180px
            best_w = None
            best_w_dist = float("inf")
            for wz in self.loot_zones:
                d = math.hypot(wz["x"] - px_c, wz["y"] - py_c)
                if d < best_w_dist:
                    best_w_dist = d
                    best_w = wz
            if best_w is not None and best_w_dist <= 180.0:
                pz["loot_pos"] = [best_w["x"], best_w["y"]]
                best_w["associated_pink"] = pz["id"]

            # Associate nearest yellow orbit zone within 80px
            best_y = None
            best_y_dist = float("inf")
            for yz in self.orbit_zones:
                zc = yz.get("center", [0, 0])
                d = math.hypot(zc[0] - px_c, zc[1] - py_c)
                if d < best_y_dist:
                    best_y_dist = d
                    best_y = yz
            if best_y is not None and best_y_dist <= 80.0:
                pz["orbit_zone"] = best_y
                best_y["associated_pink"] = pz["id"]
                if pz.get("loot_pos") and not best_y.get("loot_pos"):
                    best_y["loot_pos"] = pz["loot_pos"]

        # Associate detected pink encounter dots with closest sampled waypoint along route
        matched_pinks = []
        for pz in self.pink_zones:
            px_c, py_c = pz["x"], pz["y"]
            best_idx = None
            best_d = float("inf")
            for idx, wp in enumerate(self.waypoints):
                if 0 < idx < len(self.waypoints) - 1:
                    d = math.hypot(wp["x"] - px_c, wp["y"] - py_c)
                    if d < best_d:
                        best_d = d
                        best_idx = idx

            if best_idx is not None and best_d <= 75.0:
                matched_pinks.append((best_idx, pz))

        matched_pinks.sort(key=lambda item: item[0])
        for p_i, (best_idx, pz) in enumerate(matched_pinks, start=1):
            pz["route_order"] = p_i
            self.waypoints[best_idx]["action"] = "pink_encounter"
            self.waypoints[best_idx]["name"] = f"Pink Marker ({pz['id']})"
            self.waypoints[best_idx]["pink_pos"] = [pz["x"], pz["y"]]
            if pz.get("sim_pos"):
                self.waypoints[best_idx]["sim_pos"] = pz["sim_pos"]
            if pz.get("loot_pos"):
                self.waypoints[best_idx]["loot_pos"] = pz["loot_pos"]
            if pz.get("orbit_zone"):
                self.waypoints[best_idx]["orbit_zone"] = pz["orbit_zone"]
                pz["orbit_zone"]["entry_waypoint_index"] = best_idx

        # Associate standalone yellow orbit zones (not part of a pink encounter)
        for z_i, zone in enumerate(self.orbit_zones, start=1):
            zone["id"] = f"zone_{z_i}"
            zc = zone["center"]
            if zone.get("associated_pink"):
                continue  # Managed directly through pink encounter routine

            best_idx = None
            best_d = float("inf")
            for idx, wp in enumerate(self.waypoints):
                if 0 < idx < len(self.waypoints) - 1:
                    d = math.hypot(wp["x"] - zc[0], wp["y"] - zc[1])
                    if d < best_d:
                        best_d = d
                        best_idx = idx

            if best_idx is not None and best_d <= (zone["radius"] * 2.0 + 20.0):
                zone["entry_waypoint_index"] = best_idx
                self.waypoints[best_idx]["action"] = "orbit"
                self.waypoints[best_idx]["name"] = f"Orbit Zone ({zone['id']})"
                self.waypoints[best_idx]["orbit_zone"] = zone

        # Also associate white dots with yellow orbit zones if any orbit zone does not have loot_pos yet
        for zone in self.orbit_zones:
            zc = zone.get("center", [0, 0])
            best_w = None
            best_w_dist = float("inf")
            for wz in self.loot_zones:
                d = math.hypot(wz["x"] - zc[0], wz["y"] - zc[1])
                if d < best_w_dist:
                    best_w_dist = d
                    best_w = wz
            if best_w is not None and best_w_dist <= 150.0:
                zone["loot_pos"] = [best_w["x"], best_w["y"]]
                e_idx = zone.get("entry_waypoint_index")
                if e_idx is not None and 0 <= e_idx < len(self.waypoints):
                    if not self.waypoints[e_idx].get("loot_pos"):
                        self.waypoints[e_idx]["loot_pos"] = [best_w["x"], best_w["y"]]

        self.current_idx = 0
        self.is_loaded = True

        zones_log = f", {len(self.orbit_zones)} yellow shape(s)" if self.orbit_zones else ""
        pink_log = f", {len(self.pink_zones)} pink marker(s)" if self.pink_zones else ""
        cyan_log = f", {len(self.cyan_zones)} cyan sim(s)" if self.cyan_zones else ""
        loot_log = f", {len(self.loot_zones)} white loot(s)" if self.loot_zones else ""
        print(f"[MOVEMENT] Successfully extracted {len(self.waypoints)} waypoints{zones_log}{pink_log}{cyan_log}{loot_log} from '{image_path}':")
        print(f"  Start  (Blue): {start_pt}")
        print(f"  Finish (Red):  {finish_pt}")
        print(f"  Path Length:   {len(full_path)} px")
        for zone in self.orbit_zones:
            print(f"  Yellow Orbit Zone: Center={zone['center']}, Radius={zone['radius']}px, OrbitRadius={zone['orbit_radius']}px ({zone.get('orbit_mode', 'inside')}), Duration={zone['duration']}s")
        for pz in self.pink_zones:
            sim_info = f", SimPos={pz.get('sim_pos')}" if pz.get('sim_pos') else ""
            loot_info = f", LootPos={pz.get('loot_pos')}" if pz.get('loot_pos') else ""
            print(f"  Pink Encounter Marker: Center=({pz['x']}, {pz['y']}), Area={pz.get('area', 0.0)}px{sim_info}{loot_info}")

        # Save to JSON for caching & user inspection
        if save_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_json_path)), exist_ok=True)
            export_data = {
                "route_name": self.route_name,
                "version": "1.2",
                "description": self.description,
                "source_image": image_path,
                "settings": {
                    "arrival_distance": self.arrival_distance,
                    "loop": self.loop,
                    "orbit_duration_seconds": self.orbit_duration_seconds,
                    "orbit_margin_px": self.orbit_margin_px,
                    "orbit_mode": self.orbit_mode,
                },
                "orbit_zones": self.orbit_zones,
                "pink_zones": self.pink_zones,
                "cyan_zones": self.cyan_zones,
                "loot_zones": self.loot_zones,
                "waypoints": self.waypoints,
            }
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            print(f"[MOVEMENT] Saved route waypoints to '{save_json_path}'.")

        return True

    def reload(self, image_path: Optional[str] = None) -> bool:
        """
        Reloads or re-extracts the route on demand.
        If image_path or 'templates/route.png' is available, re-extracts directly from the image.
        """
        img_target = image_path or ("templates/route.png" if os.path.exists("templates/route.png") else None)
        out_json = self.file_path if (self.file_path and self.file_path.endswith(".json")) else "paths/movement_route.json"

        if img_target and os.path.exists(img_target):
            print(f"[MOVEMENT] On-demand reload: Extracting route from '{img_target}'...")
            return self.load_from_painted_image(img_target, save_json_path=out_json)

        if self.file_path and os.path.exists(self.file_path):
            print(f"[MOVEMENT] On-demand reload: Reading from '{self.file_path}'...")
            return self.load(self.file_path)

        return False

    @property
    def is_configured(self) -> bool:
        """Returns True if valid waypoints are loaded."""
        return self.is_loaded and len(self.waypoints) > 0

    def get_waypoints(self) -> List[Dict[str, Any]]:
        """Returns list of all configured waypoints."""
        return self.waypoints

    def get_orbit_zones(self) -> List[Dict[str, Any]]:
        """Returns all configured yellow shape orbit zones."""
        return self.orbit_zones

    def get_orbit_zone_for_waypoint(self, wp_idx: int) -> Optional[Dict[str, Any]]:
        """Returns the orbit zone associated with a given waypoint index, if any."""
        if 0 <= wp_idx < len(self.waypoints):
            wp = self.waypoints[wp_idx]
            if wp.get("action") == "orbit" and wp.get("orbit_zone"):
                return wp["orbit_zone"]
        for zone in self.orbit_zones:
            if zone.get("entry_waypoint_index") == wp_idx:
                return zone
        return None

    def get_current_target(self) -> Optional[Dict[str, Any]]:
        """Returns the current active target waypoint, or None if route finished."""
        if not self.waypoints or self.current_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.current_idx]

    def distance_to_target(self, current_pos: Tuple[float, float]) -> Optional[float]:
        """
        Calculates 2D Euclidean distance from player position to current target waypoint.

        :param current_pos: (X, Y) tuple of current player position.
        :return: Distance in pixels, or None if no active target.
        """
        target = self.get_current_target()
        if not target or current_pos is None:
            return None
        dx = target["x"] - current_pos[0]
        dy = target["y"] - current_pos[1]
        return math.hypot(dx, dy)

    def is_target_reached(
        self, current_pos: Tuple[float, float], threshold: Optional[float] = None
    ) -> bool:
        """
        Checks if player has arrived within threshold distance of current target.

        :param current_pos: (X, Y) coordinates.
        :param threshold: Optional arrival radius override.
        :return: True if reached.
        """
        dist = self.distance_to_target(current_pos)
        if dist is None:
            return False
        max_dist = threshold if threshold is not None else self.arrival_distance
        return dist <= max_dist

    def get_orbit_zones(self) -> List[Dict[str, Any]]:
        """Returns list of all detected yellow orbit zones."""
        return self.orbit_zones

    def get_orbit_zone_for_waypoint(self, waypoint_index: int) -> Optional[Dict[str, Any]]:
        """Returns the orbit zone associated with the given waypoint index if any."""
        if 0 <= waypoint_index < len(self.waypoints):
            wp = self.waypoints[waypoint_index]
            if wp.get("orbit_zone"):
                return wp["orbit_zone"]
        for zone in self.orbit_zones:
            if zone.get("entry_waypoint_index") == waypoint_index:
                return zone
        return None

    def get_pink_zones(self) -> List[Dict[str, Any]]:
        """Returns list of all detected pink encounter/sim markers."""
        return self.pink_zones

    def get_cyan_zones(self) -> List[Dict[str, Any]]:
        """Returns list of all detected cyan SIM markers."""
        return self.cyan_zones

    def get_loot_zones(self) -> List[Dict[str, Any]]:
        """Returns list of all detected white loot markers."""
        return self.loot_zones

    def get_pink_waypoints(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Returns list of (waypoint_index, waypoint_dict) for all waypoints configured
        as pink encounters in sequential route order.
        """
        return [(idx, wp) for idx, wp in enumerate(self.waypoints) if wp.get("action") == "pink_encounter"]

    def find_nearest_waypoint_index(self, current_pos: Tuple[float, float]) -> int:
        """
        Finds index of the closest waypoint to current position.

        :param current_pos: (X, Y) tuple of current player position.
        :return: Integer index of nearest waypoint in route.
        """
        if not self.waypoints:
            return 0
        best_idx = 0
        best_dist = float("inf")
        for idx, wp in enumerate(self.waypoints):
            d = math.hypot(wp["x"] - current_pos[0], wp["y"] - current_pos[1])
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def update_to_nearest(self, current_pos: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """
        Closed-loop verification: compares current character map position against all route waypoints.
        If character has progressed forward along the route (e.g. at 5th or 10th WP),
        dynamically resynchronizes current_idx to the appropriate target waypoint ahead.

        :param current_pos: (X, Y) tuple of current character position.
        :return: Current target waypoint after resynchronization.
        """
        if not self.waypoints or current_pos is None:
            return self.get_current_target()

        nearest_idx = self.find_nearest_waypoint_index(current_pos)
        nearest_wp = self.waypoints[nearest_idx]
        dist_to_nearest = math.hypot(nearest_wp["x"] - current_pos[0], nearest_wp["y"] - current_pos[1])

        # If character has progressed ahead of current_idx
        if nearest_idx > self.current_idx:
            # Crucial protection: NEVER jump over an unvisited action waypoint (pink_encounter, orbit)
            for check_idx in range(self.current_idx, nearest_idx):
                if self.waypoints[check_idx].get("action") in ("pink_encounter", "orbit"):
                    nearest_idx = check_idx
                    nearest_wp = self.waypoints[nearest_idx]
                    dist_to_nearest = math.hypot(nearest_wp["x"] - current_pos[0], nearest_wp["y"] - current_pos[1])
                    break

            prev_target = self.current_idx
            # If already within arrival distance of the nearest waypoint, target the one ahead of it
            # UNLESS it's a special action waypoint (orbit, pink_encounter), in which case we stop at it
            if dist_to_nearest <= self.arrival_distance and nearest_idx < len(self.waypoints) - 1 and nearest_wp.get("action") not in ("orbit", "pink_encounter"):
                self.current_idx = nearest_idx + 1
            else:
                self.current_idx = nearest_idx

            if self.current_idx != prev_target:
                print(f"\n[AUTOPILOT] Dynamic Resync: Character at ({current_pos[0]:.0f}, {current_pos[1]:.0f}) -> Jumped WP #{prev_target} -> WP #{self.current_idx} ({self.waypoints[self.current_idx].get('name')})")

        return self.get_current_target()

    def advance(self) -> Optional[Dict[str, Any]]:
        """
        Advances to the next waypoint. Loops back to start if loop=True.

        :return: Next target waypoint or None if complete.
        """
        if not self.waypoints:
            return None

        self.current_idx += 1
        if self.current_idx >= len(self.waypoints):
            if self.loop:
                self.current_idx = 0
            else:
                return None
        return self.get_current_target()

    def reset(self) -> None:
        """Resets route back to the first waypoint."""
        self.current_idx = 0

    @staticmethod
    def create_sample_file(file_path: str = "paths/movement_route.json") -> bool:
        """
        Creates a starter movement route JSON template file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        sample = {
            "route_name": "Sample Movement Route",
            "version": "1.0",
            "description": "Defines waypoints for the bot to navigate across the map layout (to be implemented).",
            "settings": {
                "arrival_distance": 25.0,
                "loop": False
            },
            "waypoints": [
                {
                    "name": "Map Entrance",
                    "x": 200.0,
                    "y": 300.0,
                    "action": "walk",
                    "wait_after": 0.5
                },
                {
                    "name": "Midpoint Checkpoint",
                    "x": 350.0,
                    "y": 280.0,
                    "action": "walk",
                    "wait_after": 0.0
                },
                {
                    "name": "Destination",
                    "x": 500.0,
                    "y": 250.0,
                    "action": "interact",
                    "wait_after": 1.0
                }
            ]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)
        return True
