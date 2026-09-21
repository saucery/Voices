"""
Enemy Detector Module
Detects on-screen enemy health bars (horizontal red bars) using color segmentation
and morphological analysis, and calculates proximity to the player character.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import math
import cv2
import numpy as np
from PIL import Image


class EnemyDetector:
    """Detects enemy health bars and calculates their distance from the player character."""

    def __init__(
        self,
        min_bar_width: int = 15,
        max_bar_height: int = 14,
        min_aspect_ratio: float = 2.5,
        min_saturation: int = 110,
        min_value: int = 70,
        min_red_bgr: int = 80,
        red_dominance_ratio: float = 1.35,
        exclude_minimap: bool = True,
        exclude_bottom_ui: bool = True,
    ):
        """
        Initialize EnemyDetector with detection parameters.

        :param min_bar_width: Minimum pixel width of a detected health bar.
        :param max_bar_height: Maximum pixel height of a detected health bar.
        :param min_aspect_ratio: Minimum width / height ratio for a horizontal health bar.
        :param min_saturation: Minimum HSV saturation value for red detection.
        :param min_value: Minimum HSV brightness value for red detection.
        :param min_red_bgr: Minimum red channel intensity in BGR.
        :param red_dominance_ratio: Ratio required for red over green and blue channels.
        :param exclude_minimap: Whether to ignore the top-right minimap area.
        :param exclude_bottom_ui: Whether to ignore the bottom skill/globe bar.
        """
        self.min_bar_width = min_bar_width
        self.max_bar_height = max_bar_height
        self.min_aspect_ratio = min_aspect_ratio
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_red_bgr = min_red_bgr
        self.red_dominance_ratio = red_dominance_ratio
        self.exclude_minimap = exclude_minimap
        self.exclude_bottom_ui = exclude_bottom_ui

    def detect(
        self,
        image: Union[np.ndarray, Image.Image, str],
        character_center: Optional[Tuple[float, float]] = None,
        near_threshold_px: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Detects enemy health bars on screen and computes their distance to the character.

        :param image: Screenshot image as numpy array (BGR), PIL Image, or file path.
        :param character_center: Optional (x, y) coordinates of character on screen.
                                Defaults to screen center if None.
        :param near_threshold_px: Pixel distance threshold considered 'near' the character.
        :return: Detection summary dict:
            {
                "detected": bool,
                "count": int,
                "enemies": List[Dict[str, Any]],
                "has_enemy_near": bool,
                "nearest_distance": float,
                "nearest_enemy": Optional[Dict[str, Any]],
                "character_center": Tuple[float, float]
            }
        """
        img: Optional[np.ndarray] = None
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image

        if img is None or img.size == 0:
            return {
                "detected": False,
                "count": 0,
                "enemies": [],
                "has_enemy_near": False,
                "nearest_distance": float("inf"),
                "nearest_enemy": None,
                "character_center": (0.0, 0.0),
            }

        h, w = img.shape[:2]
        char_cx = float(w // 2) if character_center is None else float(character_center[0])
        char_cy = float(h // 2) if character_center is None else float(character_center[1])

        # 1. Convert to HSV for red hue segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(
            hsv,
            np.array([0, self.min_saturation, self.min_value]),
            np.array([10, 255, 255]),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([170, self.min_saturation, self.min_value]),
            np.array([180, 255, 255]),
        )
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 2. BGR dominance check to reject light pink/grey/tinted highlights
        b, g, r = cv2.split(img)
        red_dom = (
            (r >= self.min_red_bgr)
            & (r > self.red_dominance_ratio * g.astype(float))
            & (r > self.red_dominance_ratio * b.astype(float))
        )
        red_mask = cv2.bitwise_and(red_mask, (red_dom.astype(np.uint8) * 255))

        # 3. Exclude UI regions on full game screens (minimap top-right, skill/globes bottom)
        if h >= 400 and w >= 500:
            if self.exclude_minimap:
                mm_left = int(w * 0.84)
                mm_bottom = int(h * 0.28)
                red_mask[0:mm_bottom, mm_left:w] = 0

            if self.exclude_bottom_ui:
                ui_top = int(h * 0.88)
                red_mask[ui_top:h, 0:w] = 0

        # 4. Morphological horizontal opening to extract rectangular bar shapes
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        filtered = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_h)

        # 5. Connected components extraction
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered)
        enemies: List[Dict[str, Any]] = []
        nearest_dist = float("inf")
        nearest_enemy: Optional[Dict[str, Any]] = None

        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            aspect = bw / max(1, bh)

            if bw >= self.min_bar_width and bh <= self.max_bar_height and aspect >= self.min_aspect_ratio:
                cx, cy = float(centroids[i][0]), float(centroids[i][1])
                dist = math.hypot(cx - char_cx, cy - char_cy)

                enemy_info = {
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "center": (cx, cy),
                    "width": int(bw),
                    "height": int(bh),
                    "aspect_ratio": float(aspect),
                    "area": int(area),
                    "distance_to_character": float(dist),
                    "is_near": dist <= near_threshold_px,
                }
                enemies.append(enemy_info)

                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_enemy = enemy_info

        detected = len(enemies) > 0
        has_enemy_near = any(e["is_near"] for e in enemies)

        return {
            "detected": detected,
            "count": len(enemies),
            "enemies": enemies,
            "has_enemy_near": has_enemy_near,
            "nearest_distance": nearest_dist if detected else float("inf"),
            "nearest_enemy": nearest_enemy,
            "character_center": (char_cx, char_cy),
        }
