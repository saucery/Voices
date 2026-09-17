"""
Full Reference Map Localizer Module
Pinpoints player's exact 2D coordinates (X, Y) on a full map layout reference image
by matching live minimap crops in real-time with high precision and sub-pixel accuracy.
"""

import json
import os
from typing import Dict, Any, Tuple, Optional, Union, List
import cv2
import numpy as np
from PIL import Image

from .minimap_extractor import MinimapExtractor


class MapLocalizer:
    """Tracks player's exact position on a full map layout reference image."""

    def __init__(
        self,
        reference_map_path: str = "templates/full_map_reference.png",
        extractor: Optional[MinimapExtractor] = None,
        config_path: str = "config.json",
        min_scale: float = 0.16,
        max_scale: float = 0.38,
    ):
        """
        Initialize MapLocalizer.

        :param reference_map_path: Path to the full map layout reference image.
        :param extractor: Optional MinimapExtractor instance.
        :param config_path: Path to config.json.
        :param min_scale: Minimum scale candidate for minimap matching.
        :param max_scale: Maximum scale candidate for minimap matching.
        """
        # Resolve reference map path from config.json if default was passed
        resolved_path = reference_map_path
        if (reference_map_path == "templates/full_map_reference.png" or not reference_map_path) and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    cfg_map = cfg.get("map_layout_file")
                    if cfg_map and os.path.exists(cfg_map):
                        resolved_path = cfg_map
            except Exception:
                pass

        self.reference_map_path = resolved_path
        self.extractor = extractor or MinimapExtractor()
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.ref_img: Optional[np.ndarray] = None
        self.ref_edges: Optional[np.ndarray] = None
        self.ref_w: int = 0
        self.ref_h: int = 0
        self.red_zone_bounds: Optional[Tuple[int, int, int, int]] = None  # (min_x, max_x, min_y, max_y)

        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.last_confidence: float = 0.0
        self.last_scale: Optional[float] = None
        self.locked_counter: int = 0
        self.pending_jump_pos: Optional[Tuple[int, int]] = None
        self.pending_jump_count: int = 0

        self.load_reference_map(self.reference_map_path)

    def load_reference_map(self, map_path: str) -> bool:
        """
        Loads the reference map layout image and computes combined edge/wall features.

        :param map_path: Path to reference image file.
        :return: True if loaded successfully.
        """
        if not os.path.exists(map_path):
            print(f"[MAP LOCALIZER] Warning: Reference map file not found at '{map_path}'")
            return False

        img = cv2.imread(map_path)
        if img is None:
            print(f"[MAP LOCALIZER] Error: Could not read image from '{map_path}'")
            return False

        self.ref_img = img
        self.ref_h, self.ref_w = img.shape[:2]
        self.red_zone_bounds = None

        # Detect Painted Red Square / Zone (HSV red color range)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Fallback BGR condition if HSV range is narrow
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        bgr_red_mask = ((r > 160) & (g < 80) & (b < 80)).astype(np.uint8) * 255
        combined_red = cv2.bitwise_or(red_mask, bgr_red_mask)

        red_y, red_x = np.where(combined_red > 0)
        if len(red_x) > 0:
            min_x, max_x = int(np.min(red_x)), int(np.max(red_x))
            min_y, max_y = int(np.min(red_y)), int(np.max(red_y))
            self.red_zone_bounds = (min_x, max_x, min_y, max_y)
            print(f"[MAP LOCALIZER] Detected Painted Red Zone: X [{min_x}..{max_x}], Y [{min_y}..{max_y}] ({max_x - min_x}x{max_y - min_y} px)")

            # Inpaint / clean red pixels for sharp template matching against raw game minimap
            clean_img = cv2.inpaint(img, combined_red, 3, cv2.INPAINT_TELEA)
        else:
            clean_img = img

        # Extract combined edge and wall features from clean reference map
        self.ref_edges = self.extract_features(clean_img, is_crop=False)

        print(f"[MAP LOCALIZER] Successfully loaded Full Reference Map: {self.ref_w}x{self.ref_h} px")
        return True

    def extract_features(self, img_bgr: np.ndarray, is_crop: bool = True) -> np.ndarray:
        """
        Extracts high-contrast structural features combining Canny edges and Cyan/Blue wall masks.
        Masks out center character marker and outer borders for minimap crops.
        """
        h, w = img_bgr.shape[:2]

        # 1. Structural Canny edges
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 140)

        # 2. Cyan / Blue wall color detection (Path of Exile 2 minimap walls)
        if len(img_bgr.shape) == 3:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mask_cyan = cv2.inRange(hsv, np.array([75, 30, 30]), np.array([135, 255, 255]))
            b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
            mask_blue = ((b > 90) & (g > 80) & (r < 140)).astype(np.uint8) * 255
            cyan_wall = cv2.bitwise_or(mask_cyan, mask_blue)
            combined = cv2.bitwise_or(edges, cyan_wall)
        else:
            combined = edges

        if is_crop:
            # Mask out center character icon (circle around center)
            icon_mask = np.ones((h, w), dtype=np.uint8) * 255
            cv2.circle(icon_mask, (w // 2, h // 2), 18, 0, -1)
            # Mask outer 8px crop frame border
            cv2.rectangle(icon_mask, (0, 0), (w - 1, h - 1), 0, 10)
            return cv2.bitwise_and(combined, combined, mask=icon_mask)

        return combined

    def extract_crop_features(self, minimap_crop: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for extract_features."""
        return self.extract_features(minimap_crop, is_crop=True)

    def localize_player(
        self,
        screenshot_or_crop: Union[str, np.ndarray, Image.Image],
        fast_track: bool = True,
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        expected_pos: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Localizes player's exact (X, Y) pixel coordinates on the full map reference image
        using hierarchical multi-scale matching and temporal tracking.

        :param screenshot_or_crop: Full game screenshot or minimap crop.
        :param fast_track: If True, searches around cached scale first when locked.
        :param search_roi: Optional (min_x, min_y, max_x, max_y) bounding box for guided route search.
        :param expected_pos: Optional (X, Y) expected player coordinate prior from navigation.
        :return: Localization result dictionary with player_x, player_y, confidence, bounding_box, etc.
        """
        if self.ref_edges is None or self.ref_img is None:
            return {"located": False, "confidence": 0.0, "player_position": None}

        if isinstance(screenshot_or_crop, str):
            img = cv2.imread(screenshot_or_crop)
        elif isinstance(screenshot_or_crop, Image.Image):
            img = cv2.cvtColor(np.array(screenshot_or_crop), cv2.COLOR_RGB2BGR)
        else:
            img = screenshot_or_crop

        if img is None:
            return {"located": False, "confidence": 0.0, "player_position": None}

        # Extract minimap ROI if full screenshot passed
        if img.shape[0] > 400 and img.shape[1] > 400:
            minimap_crop = self.extractor.extract_roi(img)
        else:
            minimap_crop = img

        c_h, c_w = minimap_crop.shape[:2]
        crop_features = self.extract_features(minimap_crop, is_crop=True)

        # Tier 1: Fast Local ROI search around last known position (~2ms)
        if fast_track and self.last_player_pos is not None and self.locked_counter >= 1:
            lx, ly = self.last_player_pos
            roi_margin = 100
            rx1 = max(0, int(lx - roi_margin))
            ry1 = max(0, int(ly - roi_margin))
            rx2 = min(self.ref_w, int(lx + roi_margin))
            ry2 = min(self.ref_h, int(ly + roi_margin))
            ref_roi = self.ref_edges[ry1:ry2, rx1:rx2]

            cur_scale = self.last_scale or 0.22
            roi_scales = [max(self.min_scale, cur_scale - 0.015), cur_scale, min(self.max_scale, cur_scale + 0.015)]
            roi_best_score = -1.0
            roi_best_loc = None
            roi_best_scale = cur_scale
            roi_best_size = (1, 1)

            for s in roi_scales:
                tw = max(10, int(c_w * s))
                th = max(10, int(c_h * s))
                if tw >= (rx2 - rx1) or th >= (ry2 - ry1):
                    continue
                resized = cv2.resize(crop_features, (tw, th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(ref_roi, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if float(max_val) > roi_best_score:
                    roi_best_score = float(max_val)
                    roi_best_loc = (max_loc[0] + rx1, max_loc[1] + ry1)
                    roi_best_scale = s
                    roi_best_size = (tw, th)

            if roi_best_score >= 0.15 and roi_best_loc is not None:
                tw, th = roi_best_size
                player_x = int(roi_best_loc[0] + tw // 2)
                player_y = int(roi_best_loc[1] + th // 2)

                step_dist = ((player_x - lx) ** 2 + (player_y - ly) ** 2) ** 0.5
                if step_dist < 20.0:
                    player_x = int(round(0.65 * player_x + 0.35 * lx))
                    player_y = int(round(0.65 * player_y + 0.35 * ly))

                self.pending_jump_pos = None
                self.pending_jump_count = 0
                self.last_player_pos = (player_x, player_y)
                self.last_confidence = roi_best_score
                self.last_scale = roi_best_scale
                self.locked_counter = min(10, self.locked_counter + 1)
                bounding_box = (
                    max(0, roi_best_loc[0]),
                    max(0, roi_best_loc[1]),
                    min(self.ref_w, roi_best_loc[0] + tw),
                    min(self.ref_h, roi_best_loc[1] + th),
                )
                return {
                    "located": True,
                    "confidence": round(roi_best_score, 4),
                    "player_position": (player_x, player_y),
                    "player_x": player_x,
                    "player_y": player_y,
                    "bounding_box": bounding_box,
                    "matched_scale": round(roi_best_scale, 4),
                    "target_size": (tw, th),
                    "map_width": self.ref_w,
                    "map_height": self.ref_h,
                    "red_zone_bounds": self.red_zone_bounds,
                    "minimap_crop": minimap_crop,
                }

        # Tier 2: Guided Route Search ROI (~3ms)
        # If Tier 1 failed or wasn't locked, search specifically in the expected room / progress zone
        effective_roi = search_roi
        if effective_roi is None and expected_pos is not None:
            ex, ey = int(expected_pos[0]), int(expected_pos[1])
            m = 120
            effective_roi = (max(0, ex - m), max(0, ey - m), min(self.ref_w, ex + m), min(self.ref_h, ey + m))

        if effective_roi is not None:
            gx1, gy1, gx2, gy2 = effective_roi
            if (gx2 - gx1) > 40 and (gy2 - gy1) > 40:
                guided_roi = self.ref_edges[gy1:gy2, gx1:gx2]
                g_cur_scale = self.last_scale or 0.22
                g_scales = [
                    max(self.min_scale, g_cur_scale - 0.03),
                    g_cur_scale,
                    min(self.max_scale, g_cur_scale + 0.03),
                ]
                g_best_score = -1.0
                g_best_loc = None
                g_best_scale = g_cur_scale
                g_best_size = (1, 1)

                for s in g_scales:
                    tw = max(10, int(c_w * s))
                    th = max(10, int(c_h * s))
                    if tw >= (gx2 - gx1) or th >= (gy2 - gy1):
                        continue
                    resized = cv2.resize(crop_features, (tw, th), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(guided_roi, resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if float(max_val) > g_best_score:
                        g_best_score = float(max_val)
                        g_best_loc = (max_loc[0] + gx1, max_loc[1] + gy1)
                        g_best_scale = s
                        g_best_size = (tw, th)

                if g_best_score >= 0.16 and g_best_loc is not None:
                    tw, th = g_best_size
                    player_x = int(g_best_loc[0] + tw // 2)
                    player_y = int(g_best_loc[1] + th // 2)

                    self.pending_jump_pos = None
                    self.pending_jump_count = 0
                    self.last_player_pos = (player_x, player_y)
                    self.last_confidence = g_best_score
                    self.last_scale = g_best_scale
                    self.locked_counter = min(10, self.locked_counter + 1)
                    bounding_box = (
                        max(0, g_best_loc[0]),
                        max(0, g_best_loc[1]),
                        min(self.ref_w, g_best_loc[0] + tw),
                        min(self.ref_h, g_best_loc[1] + th),
                    )
                    return {
                        "located": True,
                        "confidence": round(g_best_score, 4),
                        "player_position": (player_x, player_y),
                        "player_x": player_x,
                        "player_y": player_y,
                        "bounding_box": bounding_box,
                        "matched_scale": round(g_best_scale, 4),
                        "target_size": (tw, th),
                        "map_width": self.ref_w,
                        "map_height": self.ref_h,
                        "red_zone_bounds": self.red_zone_bounds,
                        "minimap_crop": minimap_crop,
                    }

        # Tier 3: Coarse Global Multi-scale search fallback
        candidate_scales: List[float] = []
        is_narrow_search = False

        if fast_track and self.last_scale is not None and self.locked_counter >= 2:
            s_min = max(self.min_scale, self.last_scale - 0.035)
            s_max = min(self.max_scale, self.last_scale + 0.035)
            candidate_scales = list(np.linspace(s_min, s_max, 11))
            is_narrow_search = True
        else:
            # Full coarse search
            candidate_scales = list(np.linspace(self.min_scale, self.max_scale, 20))

        best_score = -1.0
        best_scale = self.last_scale or 0.22
        best_top_left = (0, 0)
        best_target_size = (1, 1)

        min_x, max_x, min_y, max_y = self.red_zone_bounds or (0, self.ref_w, 0, self.ref_h)

        def evaluate_scale(s: float) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
            tw = max(10, int(c_w * s))
            th = max(10, int(c_h * s))
            if tw >= self.ref_w or th >= self.ref_h:
                return -1.0, (0, 0), (tw, th)

            resized = cv2.resize(crop_features, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(self.ref_edges, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            score = float(max_val)

            # Spatial continuity bonus if near last known position or expected progress
            if self.last_player_pos is not None and self.last_confidence > 0.20:
                dist = ((cx - self.last_player_pos[0]) ** 2 + (cy - self.last_player_pos[1]) ** 2) ** 0.5
                if dist < 25.0:
                    score += 0.04
                elif dist > 80.0:
                    score -= 0.04

            if effective_roi is not None:
                rx1, ry1, rx2, ry2 = effective_roi
                if (rx1 - 20 <= cx <= rx2 + 20) and (ry1 - 20 <= cy <= ry2 + 20):
                    score += 0.04
                else:
                    score -= 0.05

            return score, max_loc, (tw, th)

        for scale in candidate_scales:
            score, loc, t_size = evaluate_scale(scale)
            if score > best_score:
                best_score = score
                best_scale = scale
                best_top_left = loc
                best_target_size = t_size

        # If narrow search gave poor score, fall back to global coarse search
        if is_narrow_search and best_score < 0.18:
            candidate_scales = list(np.linspace(self.min_scale, self.max_scale, 22))
            for scale in candidate_scales:
                score, loc, t_size = evaluate_scale(scale)
                if score > best_score:
                    best_score = score
                    best_scale = scale
                    best_top_left = loc
                    best_target_size = t_size

        # Two-stage refinement: fine scale search around best_scale
        fine_scales = np.linspace(
            max(self.min_scale, best_scale - 0.015),
            min(self.max_scale, best_scale + 0.015),
            7,
        )
        for scale in fine_scales:
            score, loc, t_size = evaluate_scale(scale)
            if score > best_score:
                best_score = score
                best_scale = scale
                best_top_left = loc
                best_target_size = t_size

        tw, th = best_target_size
        player_x = int(best_top_left[0] + tw // 2)
        player_y = int(best_top_left[1] + th // 2)

        # Bounding box of the minimap footprint on the reference map
        box_x1 = max(0, best_top_left[0])
        box_y1 = max(0, best_top_left[1])
        box_x2 = min(self.ref_w, best_top_left[0] + tw)
        box_y2 = min(self.ref_h, best_top_left[1] + th)
        bounding_box = (box_x1, box_y1, box_x2, box_y2)

        confidence = float(max(0.0, best_score))

        # Accept match if confidence threshold met
        if confidence < 0.12:
            self.locked_counter = max(0, self.locked_counter - 1)
            return {
                "located": False,
                "confidence": round(confidence, 4),
                "player_position": None,
                "player_x": None,
                "player_y": None,
                "bounding_box": bounding_box,
                "matched_scale": round(best_scale, 4),
                "target_size": (tw, th),
                "map_width": self.ref_w,
                "map_height": self.ref_h,
                "red_zone_bounds": self.red_zone_bounds,
                "minimap_crop": minimap_crop,
            }

        # Temporal smoothing & jump anomaly gating
        if self.last_player_pos is not None and self.locked_counter >= 1:
            lx, ly = self.last_player_pos
            step_dist = ((player_x - lx) ** 2 + (player_y - ly) ** 2) ** 0.5
            if step_dist < 20.0:
                # Weighted smoothing for sub-pixel stability
                smoothed_x = int(round(0.65 * player_x + 0.35 * lx))
                smoothed_y = int(round(0.65 * player_y + 0.35 * ly))
                player_x, player_y = smoothed_x, smoothed_y
                self.pending_jump_pos = None
                self.pending_jump_count = 0
            elif step_dist > 70.0:
                # Far jump anomaly filter: require 3-frame consensus
                if self.pending_jump_pos is not None:
                    c_dist = math.hypot(player_x - self.pending_jump_pos[0], player_y - self.pending_jump_pos[1])
                    if c_dist <= 30.0:
                        self.pending_jump_count += 1
                        if self.pending_jump_count >= 3 or confidence > 0.45:
                            # Accepted after consensus
                            self.pending_jump_pos = None
                            self.pending_jump_count = 0
                        else:
                            player_x, player_y = lx, ly
                    else:
                        self.pending_jump_pos = (player_x, player_y)
                        self.pending_jump_count = 1
                        player_x, player_y = lx, ly
                else:
                    self.pending_jump_pos = (player_x, player_y)
                    self.pending_jump_count = 1
                    player_x, player_y = lx, ly
            else:
                self.pending_jump_pos = None
                self.pending_jump_count = 0

        self.last_player_pos = (player_x, player_y)
        self.last_confidence = confidence
        self.last_scale = best_scale
        self.locked_counter = min(10, self.locked_counter + 1)

        edge_dists = self.calculate_edge_distances(player_x, player_y)

        return {
            "located": True,
            "confidence": round(confidence, 4),
            "player_position": (player_x, player_y),
            "player_x": player_x,
            "player_y": player_y,
            "bounding_box": bounding_box,
            "matched_scale": round(best_scale, 4),
            "target_size": (tw, th),
            "map_width": self.ref_w,
            "map_height": self.ref_h,
            "edge_distances": edge_dists,
            "red_zone_bounds": self.red_zone_bounds,
            "minimap_crop": minimap_crop,
        }

    def calculate_edge_distances(self, player_x: int, player_y: int) -> Dict[str, Any]:
        """
        Calculates exact distances in pixels from player (player_x, player_y) to:
        - Map Canvas Edges (Top, Bottom, Left, Right)
        - Nearest Map Boundary
        - Nearest Wall Line Contour
        """
        w, h = self.ref_w, self.ref_h
        dist_top = max(0, player_y)
        dist_bottom = max(0, h - player_y)
        dist_left = max(0, player_x)
        dist_right = max(0, w - player_x)

        boundaries = {
            "Top": dist_top,
            "Bottom": dist_bottom,
            "Left": dist_left,
            "Right": dist_right,
        }

        nearest_boundary_name = min(boundaries, key=boundaries.get)
        min_boundary_dist = boundaries[nearest_boundary_name]

        dist_nearest_wall = 0.0
        if self.ref_edges is not None and 0 <= player_y < h and 0 <= player_x < w:
            inv_edges = cv2.bitwise_not(self.ref_edges)
            dist_transform = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
            dist_nearest_wall = float(dist_transform[player_y, player_x])

        return {
            "dist_top": dist_top,
            "dist_bottom": dist_bottom,
            "dist_left": dist_left,
            "dist_right": dist_right,
            "nearest_boundary_name": nearest_boundary_name,
            "nearest_boundary_dist": min_boundary_dist,
            "dist_nearest_wall": round(dist_nearest_wall, 1),
        }

    def render_player_location(
        self,
        result: Dict[str, Any],
        output_path: str = "debug_output/player_map_location.png",
    ) -> str:
        """
        Renders player's position as a glowing marker with bounding box on the full reference map.
        """
        if self.ref_img is None:
            return ""

        canvas = self.ref_img.copy()

        # Draw Red Zone Bounding Box in green
        if self.red_zone_bounds:
            rx1, rx2, ry1, ry2 = self.red_zone_bounds
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 220, 0), 1)

        # Draw Minimap Footprint Bounding Box
        box = result.get("bounding_box")
        if box:
            bx1, by1, bx2, by2 = box
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 200, 0), 1)

        pos = result.get("player_position")
        if pos:
            px, py = pos
            conf = result.get("confidence", 0.0)

            # Draw outer glowing ring (yellow) and center dot (red)
            cv2.circle(canvas, (px, py), 10, (0, 255, 255), 2)
            cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1)

            # Draw text label
            label = f"({px},{py}) [{conf:.0%}]"
            cv2.putText(
                canvas,
                label,
                (max(5, px - 40), max(16, py - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    def render_trajectory_map(
        self,
        history: List[Dict[str, Any]],
        output_path: str = "debug_output/movement_trajectory.png",
    ) -> str:
        """Renders the full movement trajectory path on the reference map."""
        if self.ref_img is None or not history:
            return ""

        canvas = self.ref_img.copy()

        if self.red_zone_bounds:
            rx1, rx2, ry1, ry2 = self.red_zone_bounds
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)

        points = [h["pos"] for h in history if h.get("pos")]
        if len(points) > 1:
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                cv2.line(canvas, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

        for h in history:
            if h.get("pos"):
                px, py = int(h["pos"][0]), int(h["pos"][1])
                cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1)

        if points:
            curr_x, curr_y = int(points[-1][0]), int(points[-1][1])
            last_key = history[-1].get("key", "").upper()
            last_edge = history[-1].get("edge", "")
            last_conf = history[-1].get("conf", 0.0)

            cv2.circle(canvas, (curr_x, curr_y), 10, (0, 255, 255), 2)
            cv2.circle(canvas, (curr_x, curr_y), 4, (0, 0, 255), -1)

            status_txt = f"POS:({curr_x},{curr_y}) KEY:{last_key} [{last_edge}] ({last_conf:.0%})"
            cv2.putText(
                canvas,
                status_txt,
                (5, canvas.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path
