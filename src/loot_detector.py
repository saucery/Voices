"""
Universal Loot Detector & Filter System.
Detects high-value loot items on screen using configurable color-segmentation
and template matching rules defined in routines/loot_filter.json.

Supported core loot styles:
1. Tier 1 High Value: White background rectangle with red text/border (Divine Orb, Annulment, Liquid Isolation, etc.)
2. Unique High Value: Purple/Magenta background rectangle with white text (Raven's Reflection, etc.)
3. Custom Template Fallback: Exact template matching (ui/loot1.png, etc.)
"""

import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


@dataclass
class LootItem:
    """Represents a detected loot item on screen."""
    x: int
    y: int
    w: int
    h: int
    center_x: int
    center_y: int
    rule_id: str
    rule_name: str
    priority: int
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        return self.center_x, self.center_y

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "priority": self.priority,
            "confidence": round(self.confidence, 3),
        }


class LootDetector:
    """High-speed vectorized loot detection engine."""

    def __init__(self, config_file: str = "routines/loot_filter.json"):
        self.config_file = config_file
        self.enabled: bool = True
        self.max_pickups: int = 20
        self.pickup_delay_seconds: float = 0.35
        self.approach_wait_seconds: float = 1.5
        self.save_debug_screenshots: bool = True
        self.debug_dir: str = "loot_debug"
        self.rules: List[Dict[str, Any]] = []
        self._template_cache: Dict[str, Optional[np.ndarray]] = {}
        self.load_config(self.config_file)

    def load_config(self, filepath: Optional[str] = None) -> bool:
        """Loads or reloads loot filter rules from JSON."""
        target_path = filepath or self.config_file
        candidates = [target_path, "routines/loot_filter.json", "config/loot_filter.json"]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.enabled = bool(data.get("enabled", True))
                    self.max_pickups = int(data.get("max_pickups", 20))
                    self.pickup_delay_seconds = float(data.get("pickup_delay_seconds", 0.35))
                    self.approach_wait_seconds = float(data.get("approach_wait_seconds", 1.5))
                    self.save_debug_screenshots = bool(data.get("save_debug_screenshots", self.save_debug_screenshots))
                    self.debug_dir = str(data.get("debug_dir", self.debug_dir))
                    self.rules = data.get("rules", [])
                    # Sort rules by priority ascending (1 highest)
                    self.rules.sort(key=lambda r: int(r.get("priority", 99)))
                    self._load_templates()
                    self.config_file = candidate
                    return True
                except Exception as e:
                    print(f"[LOOT DETECTOR] Warning: Failed to load '{candidate}': {e}")
        # Fallback default rules if file not found
        self._set_default_rules()
        return False

    def _set_default_rules(self):
        """Sets safe in-memory defaults if config file is missing."""
        self.rules = [
            {
                "id": "tier1_white_box_red_text",
                "name": "Tier 1 High Value (White Box / Red Text)",
                "enabled": True,
                "priority": 1,
                "type": "color_box",
                "bg_color": "white",
                "text_color": "red",
                "min_width": 55,
                "max_width": 500,
                "min_height": 16,
                "max_height": 75,
                "min_aspect_ratio": 1.6,
                "min_bg_fraction": 0.40,
                "min_text_pixels": 16,
            },
            {
                "id": "gems_uncut_cut_olive",
                "name": "Gems (Ruby / Sapphire / Diamond / Emerald / Topaz)",
                "enabled": True,
                "priority": 2,
                "type": "color_box",
                "bg_color": "olive",
                "text_color": "yellow",
                "min_width": 40,
                "max_width": 240,
                "min_height": 16,
                "max_height": 55,
                "min_aspect_ratio": 1.3,
                "min_bg_fraction": 0.35,
                "min_text_pixels": 15,
            },
            {
                "id": "ravens_reflection_purple",
                "name": "Raven's Reflection / T1 Purple Uniques",
                "enabled": True,
                "priority": 3,
                "type": "color_box",
                "bg_color": "purple",
                "text_color": "white",
                "min_width": 55,
                "max_width": 450,
                "min_height": 16,
                "max_height": 75,
                "min_aspect_ratio": 1.6,
                "min_bg_fraction": 0.35,
            },
            {
                "id": "custom_template_loot1",
                "name": "Template Matcher (ui/loot1.png)",
                "enabled": True,
                "priority": 4,
                "type": "template",
                "template_file": "ui/loot1.png",
                "threshold": 0.50,
            }
        ]
        self._load_templates()

    def _load_templates(self):
        """Pre-loads any template files referenced in template-type rules."""
        self._template_cache.clear()
        for rule in self.rules:
            if rule.get("type") == "template" and rule.get("template_file"):
                t_path = rule["template_file"]
                if os.path.exists(t_path):
                    tmpl = cv2.imread(t_path)
                    if tmpl is not None:
                        self._template_cache[t_path] = tmpl

    def detect_loot(self, screen: np.ndarray) -> List[LootItem]:
        """
        Scans a screenshot and returns all matching loot items sorted by priority.
        Applies Non-Maximum Suppression to prevent duplicate hits on the same item.
        """
        if screen is None or screen.size == 0 or not self.enabled:
            return []

        all_detected: List[LootItem] = []
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(screen)

        for rule in self.rules:
            if not rule.get("enabled", True):
                continue

            r_type = rule.get("type", "color_box")
            if r_type == "color_box":
                bg = str(rule.get("bg_color", "")).lower().strip()
                if bg == "white":
                    items = self._detect_white_box_red_text(screen, hsv, b, g, r, rule)
                    all_detected.extend(items)
                elif bg in ("purple", "magenta"):
                    items = self._detect_purple_box(screen, hsv, b, g, r, rule)
                    all_detected.extend(items)
                elif bg in ("olive", "green_dark", "gem"):
                    items = self._detect_gem_box(screen, hsv, b, g, r, rule)
                    all_detected.extend(items)
            elif r_type == "template":
                items = self._detect_template(screen, rule)
                all_detected.extend(items)

        # Sort all items by priority (1 is highest), then by vertical position or confidence
        all_detected.sort(key=lambda item: (item.priority, -item.confidence))

        # Apply Non-Maximum Suppression (deduplicate overlapping boxes)
        filtered = self._apply_nms(all_detected, iou_threshold=0.30)
        return filtered

    @staticmethod
    def is_in_ui_exclusion_zone(x: int, y: int, w: int, h: int, screen_w: int, screen_h: int) -> bool:
        """
        Filters out detections falling inside static bottom/corner game HUD elements
        (Chat window, Health globe, Mana globe, Skill action bar, Buff bar).
        Does NOT block the top/top-right gameplay area where ground loot labels frequently appear.
        """
        if screen_w < 1200 or screen_h < 720:
            return False

        cx = x + w // 2
        cy = y + h // 2

        # 1. Chat window area (Bottom-Left)
        if cx < int(screen_w * 0.22) and cy > int(screen_h * 0.65):
            return True

        # 2. Life / Flask globe (Bottom-Left)
        if cx < 230 and cy > (screen_h - 220):
            return True

        # 3. Mana globe (Bottom-Right)
        if cx > (screen_w - 230) and cy > (screen_h - 220):
            return True

        # 4. Bottom skill & flask action bar
        if cy > (screen_h - 95):
            return True

        # 5. Top-Left Buff bar (compact zone)
        if cx < 220 and cy < 60:
            return True

        return False

    def _detect_white_box_red_text(
        self,
        screen: np.ndarray,
        hsv: np.ndarray,
        b: np.ndarray,
        g: np.ndarray,
        r: np.ndarray,
        rule: Dict[str, Any],
    ) -> List[LootItem]:
        """
        Detects white background loot boxes with red text/borders.
        Uses dual-pass detection (horizontal white box contour segmentation +
        red text cluster projection) to handle tightly stacked loot boxes and vertical light beams.
        """
        sw = screen.shape[1]
        sh = screen.shape[0]
        min_w = int(rule.get("min_width", 45))
        max_w = int(rule.get("max_width", 500))
        min_h = int(rule.get("min_height", 14))
        max_h = int(rule.get("max_height", 75))
        min_ar = float(rule.get("min_aspect_ratio", 1.3))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.35))
        min_text_px = int(rule.get("min_text_pixels", 14))

        # White background mask: high brightness, low saturation
        white_mask = (
            (r > 170) & (g > 165) & (b > 165) &
            (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 165)
        ).astype(np.uint8) * 255

        # Red text / border mask: pure red (high R, distinct difference from G and B)
        red_text_mask = (
            (r > 155) &
            (r.astype(np.int16) - g.astype(np.int16) > 50) &
            (r.astype(np.int16) - b.astype(np.int16) > 50) &
            (g < 120) & (b < 120)
        ).astype(np.uint8) * 255

        items: List[LootItem] = []

        # Pass 1: Horizontal White Box Contours (strictly 1D horizontal closing)
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        white_closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_horiz)
        w_contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in w_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Avoid UI edge overlays on full screenshots (Chat, status bars)
            if self.is_in_ui_exclusion_zone(x, y, w, h, sw, sh):
                continue

            is_edge = (x <= 6 or x + w >= sw - 6 or y <= 6)
            effective_min_w = 30 if is_edge else min_w
            effective_min_ar = 0.9 if is_edge else min_ar
            effective_min_text = 10 if is_edge else min_text_px

            if w < effective_min_w or w > max_w or h < min_h or h > max_h:
                continue

            aspect_ratio = w / max(1.0, float(h))
            if aspect_ratio < effective_min_ar:
                continue

            box_white = white_mask[y:y+h, x:x+w]
            box_red = red_text_mask[y:y+h, x:x+w]

            bg_frac = np.mean(box_white > 0)
            red_frac = np.mean(box_red > 0)
            red_pixels = int(np.count_nonzero(box_red > 0))

            # Must have dominant white background over red text (prevents false positives on red boxes)
            if bg_frac >= min_bg_frac and red_pixels >= effective_min_text and bg_frac > red_frac * 1.3:
                confidence = min(1.0, 0.5 + (red_pixels / 80.0) + (bg_frac * 0.3))
                items.append(
                    LootItem(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        center_x=x + w // 2,
                        center_y=y + h // 2,
                        rule_id=rule.get("id", "tier1_white_box_red_text"),
                        rule_name=rule.get("name", "Tier 1 High Value (White Box / Red Text)"),
                        priority=int(rule.get("priority", 1)),
                        confidence=confidence,
                    )
                )

        # Pass 2: Strictly 1D Horizontal Red Text Cluster Analysis (unbreakable for stacked loot & beams)
        k_remove_beams = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        red_no_beams = cv2.morphologyEx(red_text_mask, cv2.MORPH_OPEN, k_remove_beams)

        kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        red_dilated = cv2.dilate(red_no_beams, kernel_red)
        r_contours, _ = cv2.findContours(red_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in r_contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            # Avoid UI overlays
            if self.is_in_ui_exclusion_zone(rx, ry, rw, rh, sw, sh):
                continue

            if rw < 25 or rh < 6:
                continue

            raw_red_pixels = int(np.count_nonzero(red_no_beams[ry:ry+rh, rx:rx+rw] > 0))
            if raw_red_pixels < 10:
                continue

            # Expand to cover the surrounding white background rectangle
            pad_x = 8
            pad_y = 6
            bx = max(0, rx - pad_x)
            by = max(0, ry - pad_y)
            bw = min(sw - bx, rw + 2 * pad_x)
            bh = min(sh - by, rh + 2 * pad_y)

            is_edge = (bx <= 6 or bx + bw >= sw - 6 or by <= 6)
            effective_min_w = 30 if is_edge else min_w

            if bw < effective_min_w or bh < min_h:
                continue

            box_white = white_mask[by:by+bh, bx:bx+bw]
            box_red = red_text_mask[by:by+bh, bx:bx+bw]
            bg_frac = np.mean(box_white > 0)
            red_frac = np.mean(box_red > 0)

            # Must have dominant white background
            if bg_frac >= min_bg_frac and bg_frac > red_frac * 1.3:
                confidence = min(1.0, 0.5 + (raw_red_pixels / 80.0) + (bg_frac * 0.3))
                items.append(
                    LootItem(
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                        center_x=bx + bw // 2,
                        center_y=by + bh // 2,
                        rule_id=rule.get("id", "tier1_white_box_red_text"),
                        rule_name=rule.get("name", "Tier 1 High Value (White Box / Red Text)"),
                        priority=int(rule.get("priority", 1)),
                        confidence=confidence,
                    )
                )

        return items

    def _detect_purple_box(
        self,
        screen: np.ndarray,
        hsv: np.ndarray,
        b: np.ndarray,
        g: np.ndarray,
        r: np.ndarray,
        rule: Dict[str, Any],
    ) -> List[LootItem]:
        """Detects purple/magenta background loot boxes (e.g., Raven's Reflection)."""
        sw = screen.shape[1]
        sh = screen.shape[0]
        min_w = int(rule.get("min_width", 55))
        max_w = int(rule.get("max_width", 450))
        min_h = int(rule.get("min_height", 16))
        max_h = int(rule.get("max_height", 65))
        min_ar = float(rule.get("min_aspect_ratio", 1.6))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.35))

        # Purple / Magenta Hue in OpenCV HSV is ~ 135 to 172 with high saturation/value
        purple_mask = (
            (
                (hsv[:, :, 0] >= 135) & (hsv[:, :, 0] <= 170) &
                (hsv[:, :, 1] >= 105) & (hsv[:, :, 2] >= 95) &
                (b.astype(int) > g.astype(int) + 20) &
                (r.astype(int) > g.astype(int) + 20)
            ) | (
                (r > 125) & (b > 135) & (g < 85) &
                (hsv[:, :, 1] >= 105)
            )
        ).astype(np.uint8) * 255

        # Text glyphs inside unique box (black text like Inscribed Ultimatum or white text)
        dark_text_mask = (r < 65) & (g < 65) & (b < 65)
        white_text_mask = (r > 175) & (g > 175) & (b > 175)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        purple_closed = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(purple_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[LootItem] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.is_in_ui_exclusion_zone(x, y, w, h, sw, sh):
                continue

            if w < 70 or w > max_w or h < 18 or h > max_h:
                continue

            aspect_ratio = w / max(1.0, float(h))
            if aspect_ratio < 2.0:
                continue

            box_purple = purple_mask[y:y+h, x:x+w]
            bg_frac = np.mean(box_purple > 0)
            text_count = np.count_nonzero(dark_text_mask[y:y+h, x:x+w]) + np.count_nonzero(white_text_mask[y:y+h, x:x+w])

            # Must have solid purple background (>= 50%) AND distinct text glyphs inside (>= 25px)
            # Rejects ground fire, smoke, crystals, and transparent HUD text
            if bg_frac >= 0.48 and text_count >= 25:
                confidence = min(1.0, 0.6 + bg_frac * 0.4)
                items.append(
                    LootItem(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        center_x=x + w // 2,
                        center_y=y + h // 2,
                        rule_id=rule.get("id", "ravens_reflection_purple"),
                        rule_name=rule.get("name", "Raven's Reflection / T1 Purple Uniques"),
                        priority=int(rule.get("priority", 3)),
                        confidence=confidence,
                    )
                )

        return items

    def _detect_gem_box(
        self,
        screen: np.ndarray,
        hsv: np.ndarray,
        b: np.ndarray,
        g: np.ndarray,
        r: np.ndarray,
        rule: Dict[str, Any],
    ) -> List[LootItem]:
        """
        Detects uncut and cut gems (Ruby, Sapphire, Diamond, Emerald, Topaz, etc.)
        which are styled with dark olive background and yellow/lime text & border.
        """
        sw = screen.shape[1]
        sh = screen.shape[0]
        min_w = int(rule.get("min_width", 40))
        max_w = int(rule.get("max_width", 240))
        min_h = int(rule.get("min_height", 16))
        max_h = int(rule.get("max_height", 55))
        min_ar = float(rule.get("min_aspect_ratio", 1.3))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.35))
        min_text_px = int(rule.get("min_text_pixels", 15))

        # 1. Dark Olive / Greenish-Brown Background Mask for PoE Gems
        olive_bg = (
            (g >= 35) & (g <= 125) &
            (r >= 40) & (r <= 135) &
            (b < 55) &
            (g.astype(np.int16) >= b.astype(np.int16) + 10) &
            (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 42) &
            (hsv[:, :, 1] >= 60) &
            (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] <= 135)
        ).astype(np.uint8) * 255

        # 2. Gem Text & Border Mask (Yellow / Lime)
        gem_text = (
            (r >= 140) & (g >= 125) &
            (b < 120) &
            (g.astype(np.int16) >= b.astype(np.int16) + 20) &
            (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 38) &
            (hsv[:, :, 1] >= 65) &
            (hsv[:, :, 2] >= 140)
        ).astype(np.uint8) * 255

        # Isolate text touching or inside the olive background to prevent merging with adjacent rare items
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        olive_dilated = cv2.dilate(olive_bg, k_dilate)
        gem_text_isolated = cv2.bitwise_and(gem_text, olive_dilated)

        # 1D Horizontal closing of olive background + isolated text
        combined_gem = cv2.bitwise_or(olive_bg, gem_text_isolated)
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        gem_closed = cv2.morphologyEx(combined_gem, cv2.MORPH_CLOSE, k_horiz)

        contours, _ = cv2.findContours(gem_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[LootItem] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.is_in_ui_exclusion_zone(x, y, w, h, sw, sh):
                continue

            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue
            if (w / max(1.0, float(h))) < min_ar:
                continue

            box_olive = olive_bg[y:y+h, x:x+w]
            box_text = gem_text_isolated[y:y+h, x:x+w]
            bg_frac = np.mean(box_olive > 0)
            text_pixels = int(np.count_nonzero(box_text > 0))

            if bg_frac >= min_bg_frac and text_pixels >= min_text_px:
                # Tighten bounding box around text/border
                text_pts = cv2.findNonZero(box_text)
                if text_pts is not None:
                    tx, ty, tw, th = cv2.boundingRect(text_pts)
                    pad_x = 8
                    pad_y = 4
                    bx = max(x, x + tx - pad_x)
                    by = max(y, y + ty - pad_y)
                    bw = min(w - (bx - x), tw + 2 * pad_x)
                    bh = min(h - (by - y), th + 2 * pad_y)
                else:
                    bx, by, bw, bh = x, y, w, h

                conf = min(1.0, 0.65 + (text_pixels / 120.0) + (bg_frac * 0.25))
                items.append(
                    LootItem(
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                        center_x=bx + bw // 2,
                        center_y=by + bh // 2,
                        rule_id=rule.get("id", "gems_uncut_cut_olive"),
                        rule_name=rule.get("name", "Gems (Ruby / Sapphire / Diamond / Emerald / Topaz)"),
                        priority=int(rule.get("priority", 2)),
                        confidence=conf,
                    )
                )

        return items

    def _detect_template(self, screen: np.ndarray, rule: Dict[str, Any]) -> List[LootItem]:
        """Runs multi-scale template matching for specific items."""
        t_file = rule.get("template_file")
        if not t_file or t_file not in self._template_cache:
            return []

        tmpl = self._template_cache[t_file]
        if tmpl is None:
            return []

        thresh = float(rule.get("threshold", 0.50))
        g_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        g_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

        th, tw = g_tmpl.shape[:2]
        sh, sw = g_screen.shape[:2]

        items: List[LootItem] = []
        scales = [1.0, 0.85, 0.90, 1.10, 1.20]

        for scale in scales:
            sc_w = int(tw * scale)
            sc_h = int(th * scale)
            if sc_w > sw or sc_h > sh or sc_w < 15 or sc_h < 15:
                continue

            resized = cv2.resize(g_tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(g_screen, resized, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= thresh)

            for pt in zip(*locs[::-1]):
                items.append(
                    LootItem(
                        x=int(pt[0]),
                        y=int(pt[1]),
                        w=sc_w,
                        h=sc_h,
                        center_x=int(pt[0] + sc_w / 2),
                        center_y=int(pt[1] + sc_h / 2),
                        rule_id=rule.get("id", "custom_template"),
                        rule_name=rule.get("name", "Template Match"),
                        priority=int(rule.get("priority", 3)),
                        confidence=float(res[pt[1], pt[0]]),
                    )
                )

        return items

    def _apply_nms(self, items: List[LootItem], iou_threshold: float = 0.30) -> List[LootItem]:
        """Filters out redundant overlapping bounding boxes."""
        if not items:
            return []

        kept: List[LootItem] = []
        for candidate in items:
            cx1, cy1, cw1, ch1 = candidate.rect
            overlap = False
            for existing in kept:
                ex1, ey1, ew1, eh1 = existing.rect
                # Compute Intersection over Union (IoU)
                ix1 = max(cx1, ex1)
                iy1 = max(cy1, ey1)
                ix2 = min(cx1 + cw1, ex1 + ew1)
                iy2 = min(cy1 + ch1, ey1 + eh1)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih

                union_area = (cw1 * ch1) + (ew1 * eh1) - inter_area
                iou = inter_area / max(1.0, float(union_area))

                if iou >= 0.20 or (inter_area / max(1.0, min(cw1 * ch1, ew1 * eh1))) > 0.50:
                    overlap = True
                    break

            if not overlap:
                kept.append(candidate)

        return kept

    def draw_overlay(self, image: np.ndarray, items: List[LootItem]) -> np.ndarray:
        """Draws visual debugging overlays and bounding boxes for detected loot items."""
        vis = image.copy()
        for idx, item in enumerate(items):
            x, y, w, h = item.rect
            # Choose color based on priority / rule
            if item.priority == 1 or "white" in item.rule_id:
                box_color = (0, 0, 255)      # Red for Tier 1 White/Red
                text_color = (255, 255, 255)
                badge_bg = (0, 0, 200)
            elif "gem" in item.rule_id or "olive" in item.rule_id:
                box_color = (0, 255, 0)      # Green for Gems (Ruby / Sapphire / Diamond)
                text_color = (0, 0, 0)
                badge_bg = (50, 205, 50)     # Lime Green badge
            elif item.priority == 3 or "purple" in item.rule_id:
                box_color = (255, 0, 255)    # Magenta for Purple Uniques
                text_color = (255, 255, 255)
                badge_bg = (200, 0, 200)
            else:
                box_color = (255, 200, 0)    # Cyan / Yellow for other rules
                text_color = (0, 0, 0)
                badge_bg = (255, 200, 0)

            # Draw bounding box and crosshair at pickup center
            cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 2)
            cv2.drawMarker(vis, (item.center_x, item.center_y), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

            # Draw tag label
            label = f"#{idx+1} [P{item.priority}] {item.rule_name} ({w}x{h})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ty = max(th + 4, y - 4)
            cv2.rectangle(vis, (x, ty - th - 3), (x + tw + 6, ty + 3), badge_bg, -1)
            cv2.putText(vis, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

        return vis

    def save_debug_screenshot(
        self,
        screen: np.ndarray,
        target_item: LootItem,
        all_items: Optional[List[LootItem]] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Saves an annotated full screenshot and a zoomed item crop to the debug folder.
        Returns tuple of file paths: (full_screenshot_path, crop_path).
        """
        out_dir = output_dir or self.debug_dir
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1.0) * 1000)
        safe_rule = str(target_item.rule_id).replace(" ", "_").replace("/", "_")
        prefix = f"loot_{timestamp}_{millis:03d}_P{target_item.priority}_{safe_rule}"

        # 1. Annotated full screen
        items_to_draw = all_items or [target_item]
        annotated_screen = self.draw_overlay(screen, items_to_draw)
        full_path = os.path.join(out_dir, f"{prefix}_full.png")
        cv2.imwrite(full_path, annotated_screen)

        # 2. Zoomed crop around target item with margin
        margin = 35
        sh, sw = screen.shape[:2]
        cx1 = max(0, target_item.x - margin)
        cy1 = max(0, target_item.y - margin)
        cx2 = min(sw, target_item.x + target_item.w + margin)
        cy2 = min(sh, target_item.y + target_item.h + margin)

        crop_img = screen[cy1:cy2, cx1:cx2].copy()
        rel_x = target_item.x - cx1
        rel_y = target_item.y - cy1
        cv2.rectangle(crop_img, (rel_x, rel_y), (rel_x + target_item.w, rel_y + target_item.h), (0, 255, 0), 2)
        cv2.drawMarker(crop_img, (target_item.center_x - cx1, target_item.center_y - cy1), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

        # Add small badge on crop
        badge_txt = f"[P{target_item.priority}] {target_item.rule_name}"
        cv2.putText(crop_img, badge_txt, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)

        crop_path = os.path.join(out_dir, f"{prefix}_crop.png")
        cv2.imwrite(crop_path, crop_img)

        return full_path, crop_path


if __name__ == "__main__":
    import sys
    detector = LootDetector()
    print("=== LOOT DETECTOR CONFIGURATION ===")
    print(f"Filter Enabled: {detector.enabled}")
    print(f"Max Pickups: {detector.max_pickups}")
    print(f"Active Rules ({len(detector.rules)}):")
    for r in detector.rules:
        status = "ENABLED" if r.get("enabled", True) else "DISABLED"
        print(f"  - [P{r.get('priority')}] {r.get('name')} ({r.get('id')}) [{status}]")

    test_imgs = sys.argv[1:] if len(sys.argv) > 1 else [
        "C:/Users/gregg/.gemini/antigravity-ide/brain/bc444d8e-d778-41d6-a9f9-4d86757c5719/.user_uploaded/media_1789633348657.jpg",
        "C:/Users/gregg/.gemini/antigravity-ide/brain/bc444d8e-d778-41d6-a9f9-4d86757c5719/.user_uploaded/media_1789633349614.jpg",
    ]

    for img_path in test_imgs:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                detected = detector.detect_loot(img)
                print(f"\n--- Testing on '{os.path.basename(img_path)}' ({img.shape[1]}x{img.shape[0]}) ---")
                print(f"Found {len(detected)} high-value loot item(s):")
                for i, item in enumerate(detected):
                    print(f"  #{i+1}: {item.rule_name} at Center=({item.center_x}, {item.center_y}), Rect=({item.x},{item.y},{item.w},{item.h}), Conf={item.confidence:.2f}")
                
                # Save annotated inspection image
                out_name = f"detected_{os.path.splitext(os.path.basename(img_path))[0]}.png"
                out_dir = "C:/Users/gregg/.gemini/antigravity-ide/brain/bc444d8e-d778-41d6-a9f9-4d86757c5719/scratch"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, out_name)
                vis = detector.draw_overlay(img, detected)
                cv2.imwrite(out_path, vis)
                print(f"  -> Saved overlay preview: {out_path}")
