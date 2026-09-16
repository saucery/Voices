import json
import os
from typing import Union, Dict, Tuple
import cv2
import numpy as np
from PIL import Image


class MinimapExtractor:
    """Extracts and preprocesses the minimap region from game screenshots."""

    def __init__(self, roi_config: Dict[str, float] = None):
        """
        Initialize the MinimapExtractor with ROI configuration.

        :param roi_config: Dictionary containing relative crop percentages.
        """
        if roi_config is None:
            config_path = "config.json"
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    roi_config = cfg.get("minimap_roi")
                except Exception:
                    pass

        self.roi_config = roi_config or {
            "top_pct": 0.0,
            "bottom_pct": 0.26,
            "left_pct": 0.85,
            "right_pct": 1.00,
        }

    def _to_cv2(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Convert input image (file path, PIL Image, or numpy array) to OpenCV BGR format."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
            return img
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def extract_roi(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Crops the minimap region based on ROI configuration percentages.

        :param image: Input full screenshot.
        :return: Cropped minimap image (BGR numpy array).
        """
        img_bgr = self._to_cv2(image)
        h, w = img_bgr.shape[:2]

        top = int(h * self.roi_config.get("top_pct", 0.0))
        bottom = int(h * self.roi_config.get("bottom_pct", 0.26))
        left = int(w * self.roi_config.get("left_pct", 0.85))
        right = int(w * self.roi_config.get("right_pct", 1.00))

        # Clamp boundaries
        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))

        return img_bgr[top:bottom, left:right]

    def preprocess(
        self,
        img: np.ndarray,
        canny_t1: int = 50,
        canny_t2: int = 150,
        blur_kernel: Tuple[int, int] = (3, 3),
        clean_noise: bool = True,
    ) -> np.ndarray:
        """
        Preprocesses minimap ROI to extract clean structural wall edges,
        filtering out background ground textures, shadows, and terrain noise.

        :param img: Cropped minimap BGR image.
        :param canny_t1: Lower threshold for Canny edge detector.
        :param canny_t2: Upper threshold for Canny edge detector.
        :param blur_kernel: Gaussian blur kernel size.
        :param clean_noise: If True, uses color isolation & morphological filtering.
        :return: Preprocessed binary edge map containing structural wall outlines.
        """
        h, w = img.shape[:2]

        if clean_noise and len(img.shape) == 3:
            # 1. Target Path of Exile 2 Cyan / Electric-Blue wall outlines
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_cyan = cv2.inRange(hsv, np.array([75, 30, 40]), np.array([135, 255, 255]))
            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            mask_blue = ((b > 85) & (g > 75) & (b.astype(int) - r.astype(int) > 10)).astype(np.uint8) * 255
            wall_mask = cv2.bitwise_or(mask_cyan, mask_blue)

            # 2. Morphological opening removes 1-2px high-frequency floor speckles
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)

            # 3. Suppress orange player marker so movement doesn't create shifting false edges
            mask_orange = cv2.inRange(hsv, np.array([8, 100, 100]), np.array([26, 255, 255]))
            dilated_orange = cv2.dilate(mask_orange, np.ones((7, 7), np.uint8))
            cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(dilated_orange))

            # 4. Suppress outer 6px frame border
            border_mask = np.zeros((h, w), dtype=np.uint8)
            border_mask[6:h - 6, 6:w - 6] = 255
            cleaned = cv2.bitwise_and(cleaned, border_mask)

            # 5. Connected component filtering: keep meaningful wall contours and reject tiny noise blobs
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            wall_canvas = np.zeros((h, w), dtype=np.uint8)
            for c in contours:
                if cv2.contourArea(c) >= 10 or cv2.arcLength(c, False) >= 14:
                    cv2.drawContours(wall_canvas, [c], -1, 255, -1)

            # If clean wall contours found, return them directly
            if np.count_nonzero(wall_canvas) > 80:
                return wall_canvas

        # Fallback to Canny edge detection for grayscale or non-color inputs
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
        edges = cv2.Canny(blurred, canny_t1, canny_t2)
        return edges
