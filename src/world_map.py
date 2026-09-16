"""
World Map Tracker Module
Stitches live minimap wall features into a clean, crisp 2D global world map canvas,
tracks character's global coordinates (G_X, G_Y), and marks interactive encounter landmarks.
"""

import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image

from .minimap_extractor import MinimapExtractor


class WorldMapTracker:
    """Reconstructs a clean global world map and marks interactive encounter landmarks."""

    def __init__(
        self,
        canvas_size: Tuple[int, int] = (2400, 2400),
        extractor: Optional[MinimapExtractor] = None,
        min_stitch_distance: float = 12.0,
    ):
        """
        Initialize WorldMapTracker.

        :param canvas_size: Dimensions of the global world map canvas (width, height).
        :param extractor: Optional MinimapExtractor instance.
        :param min_stitch_distance: Minimum character movement in pixels required before stitching a new tile.
        """
        self.canvas_w, self.canvas_h = canvas_size
        self.extractor = extractor or MinimapExtractor()
        self.min_stitch_distance = min_stitch_distance

        # Global 2D BGR Canvas and Edge Canvas
        self.global_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.global_edges = np.zeros((self.canvas_h, self.canvas_w), dtype=np.uint8)
        self.coverage_mask = np.zeros((self.canvas_h, self.canvas_w), dtype=np.uint8)

        # Character start position at center of canvas
        self.global_pos: Tuple[float, float] = (self.canvas_w / 2.0, self.canvas_h / 2.0)
        self.last_stitched_pos: Tuple[float, float] = self.global_pos
        self.tile_count: int = 0
        self.last_minimap_crop: Optional[np.ndarray] = None
        self.last_clean_edges: Optional[np.ndarray] = None

        # Interactive Encounter Landmarks List
        self.encounter_landmarks: List[Dict[str, Any]] = []

    def reset(self):
        """Resets the global map canvas, character position, and landmarks."""
        self.global_canvas.fill(0)
        self.global_edges.fill(0)
        self.coverage_mask.fill(0)
        self.global_pos = (self.canvas_w / 2.0, self.canvas_h / 2.0)
        self.last_stitched_pos = self.global_pos
        self.tile_count = 0
        self.last_minimap_crop = None
        self.last_clean_edges = None
        self.encounter_landmarks.clear()

    def add_encounter_landmark(
        self,
        title: str,
        pos: Optional[Tuple[float, float]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registers an interactive encounter landmark on the global world map.

        :param title: Encounter title (e.g. 'TRAIL OF SUFFERING').
        :param pos: Global coordinate (gx, gy). Defaults to current global character position.
        :param details: Additional encounter details (wave, warning, etc.).
        :return: Created landmark dict.
        """
        target_pos = pos or self.global_pos

        # Check if already registered nearby (within 30px)
        for lm in self.encounter_landmarks:
            lx, ly = lm["global_pos"]
            dist = ((target_pos[0] - lx) ** 2 + (target_pos[1] - ly) ** 2) ** 0.5
            if dist < 30.0 and lm["title"] == title:
                return lm

        landmark = {
            "id": f"enc_{len(self.encounter_landmarks) + 1}",
            "title": title,
            "global_pos": (round(target_pos[0], 1), round(target_pos[1], 1)),
            "status": "UNCLICKED",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "details": details or {},
        }
        self.encounter_landmarks.append(landmark)
        return landmark

    def _extract_clean_wall_features(self, minimap_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts clean wall edge features with center character icon and outer ROI border masked out."""
        c_h, c_w = minimap_crop.shape[:2]

        edges = self.extractor.preprocess(minimap_crop)

        # Create mask: mask center 18px circle (character arrow) and outer 8px ROI frame border
        icon_mask = np.ones((c_h, c_w), dtype=np.uint8) * 255
        center_x, center_y = c_w // 2, c_h // 2
        cv2.circle(icon_mask, (center_x, center_y), 18, 0, -1)
        cv2.rectangle(icon_mask, (0, 0), (c_w - 1, c_h - 1), 0, 16)

        clean_edges = cv2.bitwise_and(edges, edges, mask=icon_mask)

        # Create crisp BGR wall representation (black background with gold/cyan wall lines)
        clean_bgr = np.zeros_like(minimap_crop)
        wall_pts = clean_edges > 0
        clean_bgr[wall_pts] = (255, 220, 100)

        return clean_bgr, clean_edges

    def update(self, screenshot_or_crop: np.ndarray) -> Dict[str, Any]:
        """Stitches new live minimap wall features into global map."""
        if screenshot_or_crop.shape[0] > 500 and screenshot_or_crop.shape[1] > 500:
            minimap_crop = self.extractor.extract_roi(screenshot_or_crop)
        else:
            minimap_crop = screenshot_or_crop

        c_h, c_w = minimap_crop.shape[:2]
        clean_bgr, clean_edges = self._extract_clean_wall_features(minimap_crop)

        gx, gy = self.global_pos
        dx, dy = 0.0, 0.0

        # Stage 1: Frame-to-frame minimap translation tracking
        if self.last_minimap_crop is not None and self.last_minimap_crop.shape == minimap_crop.shape:
            prev_gray = cv2.cvtColor(self.last_minimap_crop, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(minimap_crop, cv2.COLOR_BGR2GRAY)

            icon_mask = np.ones((c_h, c_w), dtype=np.uint8) * 255
            cv2.circle(icon_mask, (c_w // 2, c_h // 2), 20, 0, -1)
            prev_gray_masked = cv2.bitwise_and(prev_gray, prev_gray, mask=icon_mask)
            curr_gray_masked = cv2.bitwise_and(curr_gray, curr_gray, mask=icon_mask)

            margin = 40
            if c_h > 2 * margin and c_w > 2 * margin:
                roi_template = curr_gray_masked[margin:c_h - margin, margin:c_w - margin]
                res = cv2.matchTemplate(prev_gray_masked, roi_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.35:
                    shift_x = max_loc[0] - margin
                    shift_y = max_loc[1] - margin
                    f2f_dx = -float(shift_x)
                    f2f_dy = -float(shift_y)

                    if abs(f2f_dx) > 1.0 or abs(f2f_dy) > 1.0:
                        dx, dy = f2f_dx, f2f_dy
                        gx += dx
                        gy += dy
                        self.global_pos = (round(gx, 1), round(gy, 1))

        # Stage 2: Fine-tune alignment against global_edges canvas if accumulated edges exist
        if self.tile_count > 0 and (dx == 0.0 and dy == 0.0):
            half_w, half_h = c_w // 2, c_h // 2
            search_pad = 40
            y1 = max(0, int(gy) - half_h - search_pad)
            y2 = min(self.canvas_h, int(gy) + half_h + search_pad)
            x1 = max(0, int(gx) - half_w - search_pad)
            x2 = min(self.canvas_w, int(gx) + half_w + search_pad)

            ref_edge_region = self.global_edges[y1:y2, x1:x2]

            if ref_edge_region.shape[0] >= c_h and ref_edge_region.shape[1] >= c_w and np.any(ref_edge_region > 0):
                res = cv2.matchTemplate(ref_edge_region, clean_edges, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.20:
                    matched_x = x1 + max_loc[0] + half_w
                    matched_y = y1 + max_loc[1] + half_h
                    g_dx = matched_x - gx
                    g_dy = matched_y - gy
                    if abs(g_dx) > 1.5 or abs(g_dy) > 1.5:
                        gx += 0.5 * g_dx
                        gy += 0.5 * g_dy
                        self.global_pos = (round(gx, 1), round(gy, 1))

        self.last_minimap_crop = minimap_crop.copy()

        # Stitch canvas when character has moved or on initial tile
        dist_since_last_stitch = (
            (self.global_pos[0] - self.last_stitched_pos[0]) ** 2
            + (self.global_pos[1] - self.last_stitched_pos[1]) ** 2
        ) ** 0.5

        if self.tile_count == 0 or dist_since_last_stitch >= self.min_stitch_distance:
            top_y = max(0, int(gy) - c_h // 2)
            bottom_y = min(self.canvas_h, top_y + c_h)
            left_x = max(0, int(gx) - c_w // 2)
            right_x = min(self.canvas_w, left_x + c_w)

            crop_h = bottom_y - top_y
            crop_w = right_x - left_x

            if crop_h > 0 and crop_w > 0:
                # Mark tile region as explored on coverage_mask
                self.coverage_mask[top_y:bottom_y, left_x:right_x] = 255

                # Accumulate ONLY non-zero wall pixels onto canvas to prevent square image box smearing
                canvas_slice = self.global_canvas[top_y:bottom_y, left_x:right_x]
                edges_slice = self.global_edges[top_y:bottom_y, left_x:right_x]

                tile_edges = clean_edges[:crop_h, :crop_w]
                wall_mask = tile_edges > 0

                canvas_slice[wall_mask] = (255, 220, 100)
                edges_slice[wall_mask] = 255

                self.tile_count += 1
                self.last_stitched_pos = self.global_pos

        explored_pixels = int(np.count_nonzero(self.coverage_mask))
        total_canvas_pixels = self.canvas_w * self.canvas_h
        explored_pct = round((explored_pixels / float(total_canvas_pixels)) * 100.0, 2)

        return {
            "global_position": (round(self.global_pos[0], 1), round(self.global_pos[1], 1)),
            "displacement_delta": (round(dx, 1), round(dy, 1)),
            "explored_pixels": explored_pixels,
            "explored_pct": explored_pct,
            "tile_count": self.tile_count,
            "landmarks_count": len(self.encounter_landmarks),
        }

    def save_map(self, output_path: str = "debug_output/world_map.png"):
        """
        Saves the stitched 2D global world map image to disk,
        rendering interactive encounter landmarks as distinct purple markers.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        map_copy = self.global_canvas.copy()

        # Render interactive encounter landmarks (Purple / Magenta icons)
        for lm in self.encounter_landmarks:
            lx, ly = int(lm["global_pos"][0]), int(lm["global_pos"][1])
            status = lm.get("status", "UNCLICKED")

            # Draw outer purple glow ring (B=255, G=0, R=255)
            color = (255, 0, 255) if status == "UNCLICKED" else (0, 255, 0)
            cv2.circle(map_copy, (lx, ly), 14, color, 3)
            cv2.circle(map_copy, (lx, ly), 6, (255, 255, 255), -1)

            # Render text label above landmark
            label = f"[{lm['id']}] {lm['title']}"
            cv2.putText(
                map_copy,
                label,
                (lx - 50, ly - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw current character marker (Red/Yellow circle)
        gx, gy = int(self.global_pos[0]), int(self.global_pos[1])
        cv2.circle(map_copy, (gx, gy), 6, (0, 0, 255), -1)
        cv2.circle(map_copy, (gx, gy), 10, (0, 255, 255), 2)

        cv2.imwrite(output_path, map_copy)
        return output_path

    def render_map_view(self, crop_around_player: bool = True, view_size: Tuple[int, int] = (600, 600)) -> np.ndarray:
        """Returns a 2D BGR view of the global world map, optionally centered on the character."""
        if not crop_around_player:
            return self.global_canvas.copy()

        vw, vh = view_size
        gx, gy = int(self.global_pos[0]), int(self.global_pos[1])
        x1 = max(0, gx - vw // 2)
        y1 = max(0, gy - vh // 2)
        x2 = min(self.canvas_w, x1 + vw)
        y2 = min(self.canvas_h, y1 + vh)

        crop = self.global_canvas[y1:y2, x1:x2].copy()
        if crop.shape[0] != vh or crop.shape[1] != vw:
            padded = np.zeros((vh, vw, 3), dtype=np.uint8)
            padded[: crop.shape[0], : crop.shape[1]] = crop
            return padded
        return crop
