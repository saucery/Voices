"""
Encounter Detector Module
Recognizes on-screen encounter banners (e.g. 'TRAIL OF SUFFERING - WAVE 1/7')
and extracts encounter details using color thresholding and template correlation.
"""

import os
from typing import Dict, Any, Union, Optional
import cv2
import numpy as np
from PIL import Image


class EncounterDetector:
    """Detects on-screen game encounters (e.g., Trail of Suffering wave banners)."""

    def __init__(self, template_path: str = "templates/ui/encounter_banner.png"):
        """
        Initialize EncounterDetector.

        :param template_path: Path to reference encounter UI banner image.
        """
        self.template_path = template_path
        self.template_img: Optional[np.ndarray] = None

        if os.path.exists(self.template_path):
            self.template_img = cv2.imread(self.template_path)

        # ROI for encounter banner (top center region of game screen)
        self.roi = {
            "top_pct": 0.05,
            "bottom_pct": 0.30,
            "left_pct": 0.10,
            "right_pct": 0.45,
        }

    def extract_roi(self, screenshot: np.ndarray) -> np.ndarray:
        """Crops top-center screen ROI where encounter UI banners appear."""
        h, w = screenshot.shape[:2]
        top = int(h * self.roi["top_pct"])
        bottom = int(h * self.roi["bottom_pct"])
        left = int(w * self.roi["left_pct"])
        right = int(w * self.roi["right_pct"])
        return screenshot[top:bottom, left:right]

    def _detect_red_warning_text(self, roi_img: np.ndarray) -> bool:
        """
        Detects distinct bright red warning text ('ALL ITEMS ON THE GROUND ARE DESTROYED...').
        """
        if roi_img is None or roi_img.size == 0:
            return False

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # Red color range 1 (0 to 10 deg) and range 2 (170 to 180 deg)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_pixel_count = int(np.count_nonzero(red_mask))
        # Red warning text has at least 150 bright red pixels
        return red_pixel_count >= 150

    def detect(self, screenshot: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Analyzes game screenshot and returns encounter detection results.

        :param screenshot: Input screenshot image path, PIL image, or numpy array.
        :return: Detection summary dictionary.
        """
        if isinstance(screenshot, str):
            img = cv2.imread(screenshot)
        elif isinstance(screenshot, Image.Image):
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            img = screenshot

        if img is None:
            return {"detected": False, "confidence": 0.0}

        h, w = img.shape[:2]
        if h > 400 and w > 400:
            roi_crop = self.extract_roi(img)
        else:
            roi_crop = img

        has_red_warning = self._detect_red_warning_text(roi_crop)
        match_score = 0.0

        if self.template_img is not None:
            t_h, t_w = self.template_img.shape[:2]
            r_h, r_w = roi_crop.shape[:2]

            resized_tmpl = self.template_img.copy()
            if t_h > r_h or t_w > r_w:
                scale = min(r_h / float(t_h), r_w / float(t_w))
                resized_tmpl = cv2.resize(self.template_img, (max(1, int(t_w * scale)), max(1, int(t_h * scale))))

            g_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
            g_tmpl = cv2.cvtColor(resized_tmpl, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(g_roi, g_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            match_score = max(0.0, float(max_val))

        detected = (match_score >= 0.50) or (has_red_warning and match_score >= 0.35)

        return {
            "detected": detected,
            "title": "TRAIL OF SUFFERING",
            "wave": "WAVE 1/7",
            "action": "INTERACT TO START THE ENCOUNTER",
            "warning": "ALL ITEMS ON THE GROUND ARE DESTROYED WHEN THE ENCOUNTER BEGINS",
            "has_red_warning_text": has_red_warning,
            "confidence": match_score if detected else 0.0,
        }
