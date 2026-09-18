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
    details: Optional[Dict[str, Any]] = None

    @property
    def center(self) -> Tuple[int, int]:
        return self.center_x, self.center_y

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def to_dict(self) -> Dict[str, Any]:
        d = {
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
        if self.details:
            d["details"] = self.details
        return d


class LootDetector:
    """High-speed vectorized loot detection engine."""

    def __init__(self, config_file: str = "routines/loot_filter.json"):
        self.config_file = config_file
        self.enabled: bool = True
        self.max_pickups: int = 20
        self.pickup_delay_seconds: float = 0.35
        self.approach_wait_seconds: float = 1.1
        self.save_debug_screenshots: bool = True
        self.save_pre_loot_screenshot: bool = True
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
                    self.approach_wait_seconds = float(data.get("approach_wait_seconds", 1.1))
                    self.save_debug_screenshots = bool(data.get("save_debug_screenshots", self.save_debug_screenshots))
                    self.save_pre_loot_screenshot = bool(data.get("save_pre_loot_screenshot", data.get("save_full_screen_before_pickup", self.save_pre_loot_screenshot)))
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
        """Pre-loads any template files referenced in template-type rules and pre-caches grayscale."""
        self._template_cache.clear()
        self._template_gray_cache = {}
        for rule in self.rules:
            if rule.get("type") == "template" and rule.get("template_file"):
                t_path = rule["template_file"]
                if os.path.exists(t_path):
                    tmpl = cv2.imread(t_path)
                    if tmpl is not None and tmpl.size > 0:
                        self._template_cache[t_path] = tmpl
                        self._template_gray_cache[t_path] = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

    def save_config(self, filepath: Optional[str] = None) -> bool:
        """Saves current loot filter rules and configuration to JSON."""
        target_path = filepath or self.config_file or "routines/loot_filter.json"
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        data = {
            "enabled": self.enabled,
            "max_pickups": self.max_pickups,
            "pickup_delay_seconds": self.pickup_delay_seconds,
            "approach_wait_seconds": self.approach_wait_seconds,
            "save_debug_screenshots": self.save_debug_screenshots,
            "save_pre_loot_screenshot": self.save_pre_loot_screenshot,
            "debug_dir": self.debug_dir,
            "rules": self.rules,
        }
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._load_templates()
            return True
        except Exception as e:
            print(f"[LOOT DETECTOR] Error saving config to '{target_path}': {e}")
            return False

    def add_or_update_rule(self, rule_dict: Dict[str, Any], save: bool = True) -> bool:
        """Adds a new rule or updates an existing rule by ID."""
        rule_id = rule_dict.get("id")
        if not rule_id:
            rule_id = f"rule_{int(time.time() * 1000)}"
            rule_dict["id"] = rule_id

        # Replace existing rule with same id if present
        updated = False
        for i, existing in enumerate(self.rules):
            if existing.get("id") == rule_id:
                self.rules[i] = rule_dict
                updated = True
                break
        if not updated:
            self.rules.append(rule_dict)

        # Re-sort rules by priority
        self.rules.sort(key=lambda r: int(r.get("priority", 99)))
        self._load_templates()
        if save:
            return self.save_config()
        return True

    def remove_rule(self, rule_id: str, save: bool = True) -> bool:
        """Removes a rule by its ID."""
        self.rules = [r for r in self.rules if r.get("id") != rule_id]
        self._load_templates()
        if save:
            return self.save_config()
        return True

    def add_template_item_rule(
        self,
        name: str,
        crop_img: np.ndarray,
        priority: int = 1,
        threshold: float = 0.50,
        rule_id: Optional[str] = None,
        templates_dir: str = "ui/loot_templates",
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Saves a cropped loot item image to templates_dir and creates a template rule.
        """
        os.makedirs(templates_dir, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.lower().strip())
        rid = rule_id or f"template_{safe_name}_{int(time.time() % 10000)}"
        tmpl_filename = f"{safe_name}.png"
        tmpl_path = os.path.join(templates_dir, tmpl_filename).replace("\\", "/")

        cv2.imwrite(tmpl_path, crop_img)

        rule_dict = {
            "id": rid,
            "name": name,
            "enabled": True,
            "priority": priority,
            "type": "template",
            "template_file": tmpl_path,
            "threshold": float(threshold),
            "description": f"Visual template matcher for {name}",
        }
        self.add_or_update_rule(rule_dict, save=save)
        return rule_dict

    def update_template_image(
        self,
        rule_id: str,
        new_crop: np.ndarray,
        templates_dir: str = "ui/loot_templates",
        save: bool = True,
    ) -> bool:
        """
        Overwrites the template image on disk for an existing rule with a new crop.
        """
        for rule in self.rules:
            if rule.get("id") == rule_id:
                t_path = rule.get("template_file")
                if not t_path:
                    os.makedirs(templates_dir, exist_ok=True)
                    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(rule.get("name", rule_id)).lower().strip())
                    t_path = os.path.join(templates_dir, f"{safe_name}.png").replace("\\", "/")
                    rule["template_file"] = t_path
                    rule["type"] = "template"

                os.makedirs(os.path.dirname(t_path) or ".", exist_ok=True)
                cv2.imwrite(t_path, new_crop)
                self._template_cache[t_path] = new_crop.copy()
                if save:
                    return self.save_config()
                return True
        return False

    def detect_loot(self, screen: np.ndarray) -> List[LootItem]:
        """
        Scans a screenshot and returns all matching loot items sorted by priority.
        Applies Non-Maximum Suppression to prevent duplicate hits on the same item.
        Uses ROI search and pre-cached grayscale representations with parallel processing for maximum speed.
        """
        if screen is None or screen.size == 0 or not self.enabled:
            return []

        all_detected: List[LootItem] = []
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(screen)

        sh, sw = screen.shape[:2]
        roi_y1, roi_y2 = max(0, int(sh * 0.04)), min(sh, int(sh * 0.92))
        roi_x1, roi_x2 = max(0, int(sw * 0.08)), min(sw, int(sw * 0.92))
        roi_offset = (roi_x1, roi_y1)

        template_rules = []

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
                elif bg in ("orange", "brown", "gold", "unique", "red"):
                    items = self._detect_orange_box(screen, hsv, b, g, r, rule)
                    all_detected.extend(items)
                else:
                    items = self._detect_general_color_box(screen, hsv, b, g, r, rule)
                    all_detected.extend(items)
            elif r_type == "template":
                template_rules.append(rule)

        # High-speed parallel template matching over active gameplay ROI
        if template_rules:
            g_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            g_roi = g_screen[roi_y1:roi_y2, roi_x1:roi_x2]

            if len(template_rules) >= 4:
                from concurrent.futures import ThreadPoolExecutor
                def _run_tmpl(r):
                    return self._detect_template(screen, r, g_screen=g_screen, g_roi=g_roi, roi_offset=roi_offset)
                with ThreadPoolExecutor(max_workers=min(8, len(template_rules))) as executor:
                    res_lists = executor.map(_run_tmpl, template_rules)
                    for r_items in res_lists:
                        all_detected.extend(r_items)
            else:
                for r in template_rules:
                    items = self._detect_template(screen, r, g_screen=g_screen, g_roi=g_roi, roi_offset=roi_offset)
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
                        details={
                            "rule_type": "color_box",
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(red_pixels),
                            "aspect_ratio": round(w / max(1.0, float(h)), 2),
                            "pass": "contour",
                        },
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
                        details={
                            "rule_type": "color_box",
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(raw_red_pixels),
                            "aspect_ratio": round(bw / max(1.0, float(bh)), 2),
                            "pass": "text_cluster",
                        },
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
        min_text_px = int(rule.get("min_text_pixels", 20))

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

            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue

            aspect_ratio = w / max(1.0, float(h))
            if aspect_ratio < min_ar:
                continue

            box_purple = purple_mask[y:y+h, x:x+w]
            bg_frac = np.mean(box_purple > 0)
            text_count = np.count_nonzero(dark_text_mask[y:y+h, x:x+w]) + np.count_nonzero(white_text_mask[y:y+h, x:x+w])

            # Must have solid purple background AND distinct text glyphs inside
            if bg_frac >= min_bg_frac and text_count >= min_text_px:
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
                        details={
                            "rule_type": "color_box",
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(text_count),
                            "aspect_ratio": round(w / max(1.0, float(h)), 2),
                        },
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
        max_w = int(rule.get("max_width", 260))
        min_h = int(rule.get("min_height", 16))
        max_h = int(rule.get("max_height", 55))
        min_ar = float(rule.get("min_aspect_ratio", 1.3))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.35))
        min_text_px = int(rule.get("min_text_pixels", 25))

        # 1. Dark Olive / Greenish-Brown Background Mask for PoE Gems
        olive_bg = (
            (g >= 40) & (g <= 125) &
            (r >= 40) & (r <= 135) &
            (b <= 50) &
            (g.astype(np.int16) >= b.astype(np.int16) + 12) &
            (hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 40) &
            (hsv[:, :, 1] >= 65) &
            (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 135)
        ).astype(np.uint8) * 255

        # 2. Gem Text & Border Mask (Yellow / Lime)
        gem_text = (
            (r >= 140) & (g >= 130) &
            (b < 110) &
            (g.astype(np.int16) >= b.astype(np.int16) + 25) &
            (hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 38) &
            (hsv[:, :, 1] >= 70) &
            (hsv[:, :, 2] >= 140)
        ).astype(np.uint8) * 255

        # Isolate text touching or inside the olive background to prevent merging with adjacent rare items
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        olive_dilated = cv2.dilate(olive_bg, k_dilate)
        gem_text_isolated = cv2.bitwise_and(gem_text, olive_dilated)

        # 1D Horizontal closing of olive background + isolated text
        combined_gem = cv2.bitwise_or(olive_bg, gem_text_isolated)
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
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
                    if th < 7 or tw < 18:
                        continue  # Discard thin horizontal line/noise slivers (e.g. 5px high false positives)

                    pad_x = 8
                    pad_y = 4
                    bx = max(x, x + tx - pad_x)
                    by = max(y, y + ty - pad_y)
                    bw = min(w - (bx - x), max(tw + 2 * pad_x, 40))
                    bh = min(h - (by - y), max(th + 2 * pad_y, 16))

                    if bh < min_h or bw < min_w:
                        continue
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
                        details={
                            "rule_type": "color_box",
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(text_pixels),
                            "aspect_ratio": round(bw / max(1.0, float(bh)), 2),
                        },
                    )
                )

        return items

    def _detect_template(
        self,
        screen: np.ndarray,
        rule: Dict[str, Any],
        g_screen: Optional[np.ndarray] = None,
        g_roi: Optional[np.ndarray] = None,
        roi_offset: Tuple[int, int] = (0, 0),
    ) -> List[LootItem]:
        """Runs fast template matching for specific ground items."""
        t_file = rule.get("template_file")
        if not t_file:
            return []

        # Use pre-cached grayscale template
        if not hasattr(self, "_template_gray_cache") or t_file not in self._template_gray_cache:
            if t_file not in self._template_cache or self._template_cache[t_file] is None:
                return []
            g_tmpl = cv2.cvtColor(self._template_cache[t_file], cv2.COLOR_BGR2GRAY)
        else:
            g_tmpl = self._template_gray_cache[t_file]

        if g_tmpl is None or g_tmpl.size == 0:
            return []

        thresh = float(rule.get("threshold", 0.50))
        sh, sw = screen.shape[:2]

        if g_roi is not None:
            target_img = g_roi
            ox, oy = roi_offset
        elif g_screen is not None:
            target_img = g_screen
            ox, oy = 0, 0
        else:
            target_img = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            ox, oy = 0, 0

        th, tw = g_tmpl.shape[:2]
        if target_img.shape[0] < th or target_img.shape[1] < tw:
            return []

        scales = rule.get("scales", [1.0] if not rule.get("multi_scale", False) else [1.0, 0.85, 0.90, 1.10, 1.20])
        items: List[LootItem] = []

        for scale in scales:
            if scale == 1.0:
                resized = g_tmpl
                sc_w, sc_h = tw, th
            else:
                sc_w = int(tw * scale)
                sc_h = int(th * scale)
                if sc_w > target_img.shape[1] or sc_h > target_img.shape[0] or sc_w < 15 or sc_h < 15:
                    continue
                resized = cv2.resize(g_tmpl, (sc_w, sc_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

            res = cv2.matchTemplate(target_img, resized, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= thresh)

            for py, px in zip(locs[0], locs[1]):
                gx = ox + int(px)
                gy = oy + int(py)
                if self.is_in_ui_exclusion_zone(gx, gy, sc_w, sc_h, sw, sh):
                    continue
                items.append(
                    LootItem(
                        x=gx,
                        y=gy,
                        w=sc_w,
                        h=sc_h,
                        center_x=int(gx + sc_w / 2),
                        center_y=int(gy + sc_h / 2),
                        rule_id=rule.get("id", "custom_template"),
                        rule_name=rule.get("name", "Template Match"),
                        priority=int(rule.get("priority", 1)),
                        confidence=float(res[py, px]),
                        details={"rule_type": "template", "scale": scale},
                    )
                )

        return items

    @staticmethod
    def get_color_mask(hsv: np.ndarray, b: np.ndarray, g: np.ndarray, r: np.ndarray, color_name: str) -> np.ndarray:
        """Generates a binary mask for standard loot label color names."""
        c = color_name.lower().strip()
        if c == "white":
            return ((r > 165) & (g > 160) & (b > 160) & (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 160)).astype(np.uint8) * 255
        elif c in ("black", "dark"):
            return ((r < 85) & (g < 85) & (b < 85)).astype(np.uint8) * 255
        elif c in ("purple", "magenta"):
            return (
                ((hsv[:, :, 0] >= 135) & (hsv[:, :, 0] <= 172) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 80)) |
                ((r > 120) & (b > 130) & (g < 90))
            ).astype(np.uint8) * 255
        elif c in ("olive", "green_dark", "gem"):
            return (
                (g >= 40) & (g <= 125) & (r >= 40) & (r <= 135) & (b <= 55) &
                (g.astype(np.int16) >= b.astype(np.int16) + 12) &
                (hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 40) &
                (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 135)
            ).astype(np.uint8) * 255
        elif c in ("orange", "brown", "gold", "unique", "red", "coral"):
            return (
                # Coral red / unique fill (e.g. Liquid Disgust, Liquid Envy, Liquid Paranoia)
                ((r >= 160) & (g >= 55) & (g <= 150) & (b >= 40) & (b <= 145) &
                 (r.astype(np.int16) > g.astype(np.int16) + 25) & (r.astype(np.int16) > b.astype(np.int16) + 25)) |
                # Orange / Unique Gold fill (e.g. Exalted Orb, Regal Orb, Chaos Orb)
                ((r >= 130) & (g >= 60) & (g <= 170) & (b <= 110) &
                 (r.astype(np.int16) > b.astype(np.int16) + 35) &
                 (hsv[:, :, 0] <= 32) & (hsv[:, :, 1] >= 55))
            ).astype(np.uint8) * 255
        elif c == "blue":
            return (
                (b >= 100) & (b.astype(np.int16) >= r.astype(np.int16) + 20) &
                (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130) &
                (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 75)
            ).astype(np.uint8) * 255
        elif c == "yellow":
            return (
                (r >= 135) & (g >= 125) & (b <= 110) &
                (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 40) &
                (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 120)
            ).astype(np.uint8) * 255
        else:
            # "any" or wildcard
            return np.ones(hsv.shape[:2], dtype=np.uint8) * 255

    def _detect_orange_box(
        self,
        screen: np.ndarray,
        hsv: np.ndarray,
        b: np.ndarray,
        g: np.ndarray,
        r: np.ndarray,
        rule: Dict[str, Any],
    ) -> List[LootItem]:
        """
        Detects orange/coral/gold/unique ground loot boxes (e.g. Unique items, Inscribed Ultimatums,
        Exalted/Regal Orbs, Liquid Disgust/Envy/Paranoia) with black, white, yellow, or contrasting text.
        Uses dual-pass detection (Pass 1: Opened solid color contour segmentation; Pass 2: Horizontal
        text-cluster projection) to prevent noisy ground terrain / fire effects from merging boxes.
        """
        sw = screen.shape[1]
        sh = screen.shape[0]
        min_w = int(rule.get("min_width", 35))
        max_w = int(rule.get("max_width", 500))
        min_h = int(rule.get("min_height", 14))
        max_h = int(rule.get("max_height", 75))
        min_ar = float(rule.get("min_aspect_ratio", 1.2))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.30))
        min_text_px = int(rule.get("min_text_pixels", 15))

        orange_bg = self.get_color_mask(hsv, b, g, r, "orange")
        text_col = str(rule.get("text_color", "black")).lower().strip()
        if text_col in ("black", "dark"):
            text_mask = ((r < 75) & (g < 75) & (b < 75) | ((r > 185) & (g > 185) & (b > 185))).astype(np.uint8) * 255
        elif text_col == "white":
            text_mask = ((r > 175) & (g > 175) & (b > 175) & (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 175)).astype(np.uint8) * 255
        else:
            text_mask = self.get_color_mask(hsv, b, g, r, text_col)

        items: List[LootItem] = []

        # Pass 1: Morphological Opening to disconnect thin ground fire strands, followed by horizontal closing
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened_bg = cv2.morphologyEx(orange_bg, cv2.MORPH_OPEN, k_open)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
        orange_closed = cv2.morphologyEx(opened_bg, cv2.MORPH_CLOSE, k_close)
        contours, _ = cv2.findContours(orange_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.is_in_ui_exclusion_zone(x, y, w, h, sw, sh):
                continue
            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue
            ar = w / max(1.0, float(h))
            if ar < min_ar:
                continue

            box_bg = orange_bg[y:y+h, x:x+w]
            box_text = text_mask[y:y+h, x:x+w]
            bg_frac = np.mean(box_bg > 0)
            text_px = int(np.count_nonzero(box_text > 0))

            if bg_frac >= min_bg_frac and text_px >= min_text_px:
                confidence = min(1.0, 0.60 + (text_px / 90.0) + (bg_frac * 0.25))
                items.append(
                    LootItem(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        center_x=x + w // 2,
                        center_y=y + h // 2,
                        rule_id=rule.get("id", "orange_box_rule"),
                        rule_name=rule.get("name", "Orange / Unique Box"),
                        priority=int(rule.get("priority", 1)),
                        confidence=confidence,
                        details={
                            "rule_type": "color_box",
                            "bg_color": "orange",
                            "text_color": text_col,
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(text_px),
                            "aspect_ratio": round(ar, 2),
                            "pass": "contour",
                        },
                    )
                )

        # Pass 2: Horizontal Text Cluster Analysis (finds text lines and expands into background box)
        # Isolate text that touches or is inside the orange background to avoid vertical beams & adjacent non-orange boxes
        k_bg_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        orange_dil = cv2.dilate(orange_bg, k_bg_dil)
        text_in_bg = cv2.bitwise_and(text_mask, orange_dil)

        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        txt_clean = cv2.morphologyEx(text_in_bg, cv2.MORPH_OPEN, k_clean)
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        txt_dil = cv2.dilate(txt_clean, k_dil)
        t_cnts, _ = cv2.findContours(txt_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in t_cnts:
            tx, ty, tw, th = cv2.boundingRect(cnt)
            if tw < 20 or th < 6 or th > max_h:
                continue

            pad_x = 8
            pad_y = 6
            bx = max(0, tx - pad_x)
            by = max(0, ty - pad_y)
            bw = min(sw - bx, tw + 2 * pad_x)
            bh = min(sh - by, th + 2 * pad_y)

            if self.is_in_ui_exclusion_zone(bx, by, bw, bh, sw, sh):
                continue
            if bw < min_w or bw > max_w or bh < min_h or bh > max_h:
                continue
            ar = bw / max(1.0, float(bh))
            if ar < min_ar:
                continue

            box_bg = orange_bg[by:by+bh, bx:bx+bw]
            box_txt = text_mask[by:by+bh, bx:bx+bw]
            bg_frac = np.mean(box_bg > 0)
            text_px = int(np.count_nonzero(box_txt > 0))

            if bg_frac >= min_bg_frac and text_px >= min_text_px:
                confidence = min(1.0, 0.65 + (text_px / 90.0) + (bg_frac * 0.25))
                items.append(
                    LootItem(
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                        center_x=bx + bw // 2,
                        center_y=by + bh // 2,
                        rule_id=rule.get("id", "orange_box_rule"),
                        rule_name=rule.get("name", "Orange / Unique Box"),
                        priority=int(rule.get("priority", 1)),
                        confidence=confidence,
                        details={
                            "rule_type": "color_box",
                            "bg_color": "orange",
                            "text_color": text_col,
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(text_px),
                            "aspect_ratio": round(ar, 2),
                            "pass": "text_cluster",
                        },
                    )
                )

        return items

    def _detect_general_color_box(
        self,
        screen: np.ndarray,
        hsv: np.ndarray,
        b: np.ndarray,
        g: np.ndarray,
        r: np.ndarray,
        rule: Dict[str, Any],
    ) -> List[LootItem]:
        """
        Generic color segmentation loot box detector for user-specified bg_color and text_color.
        Uses dual-pass opened contour segmentation + horizontal text-cluster candidate expansion.
        """
        sw = screen.shape[1]
        sh = screen.shape[0]
        bg_col = str(rule.get("bg_color", "white")).lower().strip()
        txt_col = str(rule.get("text_color", "black")).lower().strip()

        min_w = int(rule.get("min_width", 35))
        max_w = int(rule.get("max_width", 500))
        min_h = int(rule.get("min_height", 14))
        max_h = int(rule.get("max_height", 75))
        min_ar = float(rule.get("min_aspect_ratio", 1.2))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.30))
        min_text_px = int(rule.get("min_text_pixels", 12))

        bg_mask = self.get_color_mask(hsv, b, g, r, bg_col)
        if txt_col == "any":
            text_mask = ((r < 75) & (g < 75) & (b < 75) | ((r > 175) & (g > 175) & (b > 175))).astype(np.uint8) * 255
        elif txt_col in ("black", "dark"):
            text_mask = ((r < 75) & (g < 75) & (b < 75)).astype(np.uint8) * 255
        elif txt_col == "white":
            text_mask = ((r > 175) & (g > 175) & (b > 175) & (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 175)).astype(np.uint8) * 255
        else:
            text_mask = self.get_color_mask(hsv, b, g, r, txt_col)

        items: List[LootItem] = []

        # Pass 1: Morphological Opening of background mask to prevent wide scene terrain merging
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened_bg = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, k_open)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
        bg_closed = cv2.morphologyEx(opened_bg, cv2.MORPH_CLOSE, k_close)
        contours, _ = cv2.findContours(bg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.is_in_ui_exclusion_zone(x, y, w, h, sw, sh):
                continue
            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue
            ar = w / max(1.0, float(h))
            if ar < min_ar:
                continue

            box_bg = bg_mask[y:y+h, x:x+w]
            box_text = text_mask[y:y+h, x:x+w]
            bg_frac = np.mean(box_bg > 0)
            text_px = int(np.count_nonzero(box_text > 0)) if txt_col != "any" else int(w * h * 0.1)

            if bg_frac >= min_bg_frac and (txt_col == "any" or text_px >= min_text_px):
                confidence = min(1.0, 0.55 + (text_px / 100.0) + (bg_frac * 0.25))
                items.append(
                    LootItem(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        center_x=x + w // 2,
                        center_y=y + h // 2,
                        rule_id=rule.get("id", f"{bg_col}_{txt_col}_box"),
                        rule_name=rule.get("name", f"Color Box ({bg_col.title()} / {txt_col.title()})"),
                        priority=int(rule.get("priority", 1)),
                        confidence=confidence,
                        details={
                            "rule_type": "color_box",
                            "bg_color": bg_col,
                            "text_color": txt_col,
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(text_px),
                            "aspect_ratio": round(ar, 2),
                            "pass": "contour",
                        },
                    )
                )

        # Pass 2: Horizontal Text Cluster Analysis
        if bg_col != "any":
            k_bg_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bg_dil = cv2.dilate(bg_mask, k_bg_dil)
            text_in_bg = cv2.bitwise_and(text_mask, bg_dil)
        else:
            text_in_bg = text_mask

        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        txt_clean = cv2.morphologyEx(text_in_bg, cv2.MORPH_OPEN, k_clean)
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        txt_dil = cv2.dilate(txt_clean, k_dil)
        t_cnts, _ = cv2.findContours(txt_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in t_cnts:
            tx, ty, tw, th = cv2.boundingRect(cnt)
            if tw < 20 or th < 6 or th > max_h:
                continue

            pad_x = 8
            pad_y = 6
            bx = max(0, tx - pad_x)
            by = max(0, ty - pad_y)
            bw = min(sw - bx, tw + 2 * pad_x)
            bh = min(sh - by, th + 2 * pad_y)

            if self.is_in_ui_exclusion_zone(bx, by, bw, bh, sw, sh):
                continue
            if bw < min_w or bw > max_w or bh < min_h or bh > max_h:
                continue
            ar = bw / max(1.0, float(bh))
            if ar < min_ar:
                continue

            box_bg = bg_mask[by:by+bh, bx:bx+bw]
            box_txt = text_mask[by:by+bh, bx:bx+bw]
            bg_frac = np.mean(box_bg > 0)
            text_px = int(np.count_nonzero(box_txt > 0)) if txt_col != "any" else int(bw * bh * 0.1)

            if bg_frac >= min_bg_frac and (txt_col == "any" or text_px >= min_text_px):
                confidence = min(1.0, 0.60 + (text_px / 100.0) + (bg_frac * 0.25))
                items.append(
                    LootItem(
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                        center_x=bx + bw // 2,
                        center_y=by + bh // 2,
                        rule_id=rule.get("id", f"{bg_col}_{txt_col}_box"),
                        rule_name=rule.get("name", f"Color Box ({bg_col.title()} / {txt_col.title()})"),
                        priority=int(rule.get("priority", 1)),
                        confidence=confidence,
                        details={
                            "rule_type": "color_box",
                            "bg_color": bg_col,
                            "text_color": txt_col,
                            "bg_fraction": round(float(bg_frac), 3),
                            "text_pixels": int(text_px),
                            "aspect_ratio": round(ar, 2),
                            "pass": "text_cluster",
                        },
                    )
                )

        return items

    @staticmethod
    def analyze_crop_colors_and_geometry(crop: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes a cropped bounding box image from a screenshot to auto-detect:
        - Dominant background color (e.g. orange, white, purple, olive, black, blue, yellow, red)
        - Contrasting foreground text color (e.g. black, white, red, yellow, blue, orange)
        - Background fill density fraction
        - Foreground text pixel count
        - Actual width, height, and aspect ratio
        - Suggested min/max thresholds for loot filter rules
        """
        if crop is None or crop.size == 0:
            return {
                "bg_color": "white",
                "text_color": "red",
                "bg_fraction": 0.35,
                "min_bg_fraction": 0.35,
                "text_pixels": 12,
                "min_text_pixels": 12,
                "detected_bg_fraction": 0.50,
                "detected_text_pixels": 20,
                "width": 0,
                "height": 0,
                "aspect_ratio": 1.0,
                "min_width": 20,
                "max_width": 500,
                "min_height": 12,
                "max_height": 65,
                "min_aspect_ratio": 1.2,
            }

        h, w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(crop)

        # 1. Sample border pixels (outer 2px rim) to reliably determine background color
        if h >= 4 and w >= 4:
            border_mask = np.zeros((h, w), dtype=bool)
            border_mask[0:2, :] = True
            border_mask[-2:, :] = True
            border_mask[:, 0:2] = True
            border_mask[:, -2:] = True
            border_b = b[border_mask]
            border_g = g[border_mask]
            border_r = r[border_mask]
            border_h = hsv[:, :, 0][border_mask]
            border_s = hsv[:, :, 1][border_mask]
            border_v = hsv[:, :, 2][border_mask]
        else:
            border_b, border_g, border_r = b.flatten(), g.flatten(), r.flatten()
            border_h, border_s, border_v = hsv[:, :, 0].flatten(), hsv[:, :, 1].flatten(), hsv[:, :, 2].flatten()

        med_h = float(np.median(border_h))
        med_s = float(np.median(border_s))
        med_v = float(np.median(border_v))
        med_r = float(np.median(border_r))
        med_g = float(np.median(border_g))
        med_b = float(np.median(border_b))

        # 2. Classify Background Color
        if med_v < 60 and med_s < 70:
            bg_color = "black"
        elif med_s < 45 and med_v > 155:
            bg_color = "white"
        elif 130 <= med_h <= 172 and med_s >= 80 and med_r > med_g + 12 and med_b > med_g + 12:
            bg_color = "purple"
        elif (14 <= med_h <= 40) and (med_g >= 35 and med_g <= 130) and (med_r >= 35 and med_r <= 140) and med_b <= 60:
            bg_color = "olive"
        elif ((med_h <= 30) or (med_h >= 170)) and (med_s >= 50) and (med_r >= 105 and med_r >= med_b + 20 and med_g >= 35):
            bg_color = "orange"
        elif (95 <= med_h <= 130) and med_s >= 55 and med_b > med_r + 15:
            bg_color = "blue"
        elif (20 <= med_h <= 40) and med_s >= 90 and med_v >= 130:
            bg_color = "yellow"
        elif (med_h < 6 or med_h > 170) and med_s >= 85 and med_r > med_g + 30:
            bg_color = "red"
        else:
            if med_v > 150 and med_s < 60:
                bg_color = "white"
            elif med_r > med_b + 20 and med_r > med_g:
                bg_color = "orange"
            else:
                bg_color = "white"

        # 3. Sample inner foreground text pixels (center area distinct from border background)
        color_diff = np.sqrt(
            (b.astype(float) - med_b)**2 +
            (g.astype(float) - med_g)**2 +
            (r.astype(float) - med_r)**2
        )
        inner_mask = np.zeros((h, w), dtype=bool)
        if h > 4 and w > 4:
            inner_mask[2:-2, 2:-2] = True
        else:
            inner_mask[:, :] = True

        fg_mask = inner_mask & (color_diff > 45)
        fg_count = int(np.count_nonzero(fg_mask))

        if fg_count > 5:
            fg_r = r[fg_mask]
            fg_g = g[fg_mask]
            fg_b = b[fg_mask]
            fg_h = hsv[:, :, 0][fg_mask]
            fg_s = hsv[:, :, 1][fg_mask]
            fg_v = hsv[:, :, 2][fg_mask]

            fg_med_h = float(np.median(fg_h))
            fg_med_s = float(np.median(fg_s))
            fg_med_v = float(np.median(fg_v))
            fg_med_r = float(np.median(fg_r))
            fg_med_g = float(np.median(fg_g))
            fg_med_b = float(np.median(fg_b))

            if fg_med_v < med_v * 0.55 or fg_med_v < 90 or (fg_med_r < 90 and fg_med_g < 90 and fg_med_b < 90):
                text_color = "black"
            elif fg_med_v > 150 and (fg_med_s < 60 or (fg_med_r > 160 and fg_med_g > 160 and fg_med_b > 160)):
                text_color = "white"
            elif fg_med_r > fg_med_g + 35 and fg_med_r > fg_med_b + 35:
                text_color = "red"
            elif fg_med_r > 130 and fg_med_g > 120 and fg_med_b < 110:
                text_color = "yellow"
            elif (6 <= fg_med_h <= 30) and fg_med_s >= 60 and fg_med_r > fg_med_b + 20:
                text_color = "orange"
            elif (95 <= fg_med_h <= 130) and fg_med_s >= 50:
                text_color = "blue"
            else:
                text_color = "black" if med_v > 120 else "white"
        else:
            text_color = "black" if med_v > 120 else "white"

        bg_frac = round(float(np.mean(color_diff < 45)), 2)
        aspect = round(w / max(1.0, float(h)), 2)
        safe_min_bg = max(0.20, min(0.35, round(bg_frac * 0.40, 2)))
        safe_min_txt = max(8, min(20, int(fg_count * 0.05)))

        return {
            "bg_color": bg_color,
            "text_color": text_color,
            "bg_fraction": safe_min_bg,
            "min_bg_fraction": safe_min_bg,
            "text_pixels": safe_min_txt,
            "min_text_pixels": safe_min_txt,
            "detected_bg_fraction": bg_frac,
            "detected_text_pixels": fg_count,
            "width": w,
            "height": h,
            "aspect_ratio": aspect,
            "min_width": max(20, int(w * 0.50)),
            "max_width": min(800, max(500, int(w * 1.5))),
            "min_height": max(10, int(h * 0.60)),
            "max_height": min(150, max(65, int(h * 1.45))),
            "min_aspect_ratio": max(0.8, min(1.5, round(aspect * 0.50, 2))),
        }

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

    def save_pre_pickup_screenshot(
        self,
        screen: np.ndarray,
        all_items: Optional[List[LootItem]] = None,
        output_dir: Optional[str] = None,
        save_annotated_too: bool = True,
    ) -> str:
        """
        Saves a clean full-screen screenshot (and optional annotated version) before picking up any loot.
        Returns the path to the raw full-screen screenshot.
        """
        out_dir = output_dir or self.debug_dir
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1.0) * 1000)
        prefix = f"loot_pre_pickup_{timestamp}_{millis:03d}"

        # 1. Clean raw full screenshot
        raw_path = os.path.join(out_dir, f"{prefix}_raw.png")
        cv2.imwrite(raw_path, screen)

        # 2. Annotated overlay if items provided
        if save_annotated_too and all_items:
            annotated_screen = self.draw_overlay(screen, all_items)
            annotated_path = os.path.join(out_dir, f"{prefix}_annotated.png")
            cv2.imwrite(annotated_path, annotated_screen)

        return raw_path

    def _load_minimap_templates(self) -> Dict[str, np.ndarray]:
        """Loads canonical minimap loot icon templates from disk."""
        if not hasattr(self, "_minimap_templates") or not self._minimap_templates:
            self._minimap_templates = {}
            search_dirs = [
                "templates/minimap_icons",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "minimap_icons"),
            ]
            for sdir in search_dirs:
                if os.path.exists(sdir):
                    for fn in sorted(os.listdir(sdir)):
                        if fn.endswith(".png"):
                            p = os.path.join(sdir, fn)
                            img = cv2.imread(p)
                            if img is not None and img.size > 0:
                                self._minimap_templates[fn] = img
                    if self._minimap_templates:
                        break
        return self._minimap_templates

    def _verify_minimap_icon_color(self, crop: np.ndarray, tname: str) -> bool:
        """Verifies candidate crop contains genuine saturated icon colors."""
        if crop is None or crop.size == 0:
            return False
        
        color_rules = {
            "star_orange": {"h_min": 5, "h_max": 16, "s_min": 130, "v_min": 120, "min_px": 10},
            "star_gold": {"h_min": 17, "h_max": 34, "s_min": 130, "v_min": 120, "min_px": 10},
            "star_red": {"h_min": 0, "h_max": 9, "h2_min": 171, "h2_max": 180, "s_min": 130, "v_min": 120, "min_px": 10},
            "star_blue": {"h_min": 95, "h_max": 115, "s_min": 130, "v_min": 120, "min_px": 8},
            "star_silver": {"s_max": 65, "v_min": 150, "min_px": 10},
            "diamond_cyan": {"h_min": 85, "h_max": 115, "s_min": 130, "v_min": 120, "min_px": 8},
            "diamond_cyan_4pack": {"h_min": 85, "h_max": 115, "s_min": 130, "v_min": 120, "min_px": 8},
            "diamond_blue": {"h_min": 95, "h_max": 118, "s_min": 130, "v_min": 120, "min_px": 8},
            "diamond_white": {"s_max": 45, "v_min": 170, "min_px": 8},
            "circle_gold_ring": {"h_min": 16, "h_max": 36, "s_min": 110, "v_min": 120, "min_px": 10},
            "circle_yellow2": {"h_min": 16, "h_max": 36, "s_min": 110, "v_min": 120, "min_px": 10},
            "circle_white": {"s_max": 50, "v_min": 175, "min_px": 12},
        }
        
        rule = None
        for k, r in color_rules.items():
            if k in tname:
                rule = r
                break
        if not rule:
            return True
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if "h2_min" in rule:
            m1 = (hsv[:, :, 0] >= rule["h_min"]) & (hsv[:, :, 0] <= rule["h_max"])
            m2 = (hsv[:, :, 0] >= rule["h2_min"]) & (hsv[:, :, 0] <= rule["h2_max"])
            m = (m1 | m2) & (hsv[:, :, 1] >= rule["s_min"]) & (hsv[:, :, 2] >= rule["v_min"])
        elif "h_min" in rule:
            m = (hsv[:, :, 0] >= rule["h_min"]) & (hsv[:, :, 0] <= rule["h_max"]) & (hsv[:, :, 1] >= rule["s_min"]) & (hsv[:, :, 2] >= rule["v_min"])
        elif "s_max" in rule:
            m = (hsv[:, :, 1] <= rule["s_max"]) & (hsv[:, :, 2] >= rule["v_min"])
        else:
            return True
            
        return int(np.count_nonzero(m)) >= rule.get("min_px", 8)

    def detect_minimap_loot_icons(
        self,
        minimap_crop: np.ndarray,
        threshold: float = 0.85,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detects saturated loot filter icons on the minimap (Stars, Diamonds, Circles).
        Uses Normalized Cross-Correlation Template Matching + Color Consistency Verification.
        Guarantees zero false positives against fire/lava terrain, guide lines, and player arrow.
        Used for early encounter exit when monsters die and drop loot.
        Returns (icon_count, list_of_detected_icons).
        """
        if minimap_crop is None or not isinstance(minimap_crop, np.ndarray) or minimap_crop.size == 0:
            return 0, []

        mh, mw = minimap_crop.shape[:2]
        if mh < 10 or mw < 10:
            return 0, []

        templates = self._load_minimap_templates()
        cx, cy = mw // 2, mh // 2
        raw_detections: List[Dict[str, Any]] = []

        # Pass 1: Multi-Template NCC Matching with Color Verification
        if templates:
            for tname, tpl in templates.items():
                th, tw = tpl.shape[:2]
                if th > mh or tw > mw:
                    continue
                res = cv2.matchTemplate(minimap_crop, tpl, cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= threshold)
                for pt_y, pt_x in zip(locs[0], locs[1]):
                    score = float(res[pt_y, pt_x])
                    icon_cx = int(pt_x + tw // 2)
                    icon_cy = int(pt_y + th // 2)
                    
                    # Exclude central player position marker (radius 16px)
                    dist_center = float(np.hypot(icon_cx - cx, icon_cy - cy))
                    if dist_center < 16.0:
                        continue
                    
                    # Exclude edge artifacts (within 3px of outer border)
                    if pt_x < 3 or pt_y < 3 or (pt_x + tw) > (mw - 3) or (pt_y + th) > (mh - 3):
                        continue
                    
                    # Verify color inside candidate patch
                    candidate_crop = minimap_crop[pt_y:pt_y+th, pt_x:pt_x+tw]
                    if not self._verify_minimap_icon_color(candidate_crop, tname):
                        continue
                        
                    raw_detections.append({
                        "x": int(pt_x),
                        "y": int(pt_y),
                        "w": int(tw),
                        "h": int(th),
                        "center_x": icon_cx,
                        "center_y": icon_cy,
                        "score": score,
                        "type": os.path.splitext(tname)[0],
                        "template": tname,
                    })

        # Pass 2: Non-Maximum Suppression (collapse duplicate/overlapping matches)
        if templates:
            if not raw_detections:
                return 0, []
            raw_detections.sort(key=lambda d: d.get("score", 0.0), reverse=True)
            kept_icons: List[Dict[str, Any]] = []
            for d in raw_detections:
                overlap = False
                for k in kept_icons:
                    dist = np.hypot(d["center_x"] - k["center_x"], d["center_y"] - k["center_y"])
                    if dist < 8.0:
                        overlap = True
                        break
                if not overlap:
                    kept_icons.append(d)
            return len(kept_icons), kept_icons

        # Pass 3: Fallback Strict Geometric Contour Validation (ONLY when templates are not present on disk)
        hsv = cv2.cvtColor(minimap_crop, cv2.COLOR_BGR2HSV)
        m_gold = cv2.inRange(hsv, np.array([10, 140, 140]), np.array([38, 255, 255]))
        m_red = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([9, 255, 255])) | cv2.inRange(hsv, np.array([171, 150, 150]), np.array([180, 255, 255]))
        m_cyan = cv2.inRange(hsv, np.array([85, 140, 140]), np.array([120, 255, 255]))
        combined = m_gold | m_red | m_cyan
        cv2.circle(combined, (cx, cy), 16, 0, -1)

        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_icons: List[Dict[str, Any]] = []

        for c in cnts:
            area = cv2.contourArea(c)
            if 15 <= area <= 240:
                x, y, w, h = cv2.boundingRect(c)
                ar = float(w) / max(1, h)
                if 0.65 <= ar <= 1.55 and 6 <= w <= 18 and 6 <= h <= 18:
                    if x > 3 and y > 3 and (x + w) < (mw - 3) and (y + h) < (mh - 3):
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / max(1.0, hull_area)
                        if solidity >= 0.70:
                            itype = "star_gold" if np.any(m_gold[y:y+h, x:x+w]) else ("star_red" if np.any(m_red[y:y+h, x:x+w]) else "diamond_cyan")
                            valid_icons.append({
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "center_x": x + w // 2,
                                "center_y": y + h // 2,
                                "area": area,
                                "score": 0.85,
                                "type": itype,
                                "template": itype,
                            })

        return len(valid_icons), valid_icons

    def save_minimap_early_exit_debug(
        self,
        full_screen: Optional[np.ndarray],
        minimap_crop: Optional[np.ndarray],
        icons: List[Dict[str, Any]],
        output_dir: str = "loot_debug",
    ) -> Tuple[str, str]:
        """
        Saves annotated minimap crop and full-screen image when early encounter exit triggers.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 1. Annotated minimap crop
        mm_path = os.path.join(output_dir, f"minimap_early_exit_{timestamp}_annotated.png")
        if minimap_crop is not None and minimap_crop.size > 0:
            annotated_mm = minimap_crop.copy()
            for ic in icons:
                ix, iy, iw, ih = ic["x"], ic["y"], ic["w"], ic["h"]
                itype = ic.get("type", "loot_icon")
                score = ic.get("score", 1.0)
                if "gold" in itype or "yellow" in itype:
                    col = (0, 215, 255)
                elif "orange" in itype:
                    col = (0, 140, 255)
                elif "red" in itype:
                    col = (0, 60, 255)
                elif "cyan" in itype or "blue" in itype:
                    col = (255, 200, 0)
                else:
                    col = (230, 230, 230)
                cv2.rectangle(annotated_mm, (ix, iy), (ix + iw, iy + ih), col, 2)
                badge_lbl = f"{itype[:6]} {score:.2f}" if score < 1.0 else itype[:8]
                cv2.putText(annotated_mm, badge_lbl, (ix, max(10, iy - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1, cv2.LINE_AA)

            cv2.putText(annotated_mm, f"EARLY EXIT ({len(icons)} loot icons)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, cv2.LINE_AA)
            cv2.imwrite(mm_path, annotated_mm)

        # 2. Full screen capture
        raw_path = os.path.join(output_dir, f"minimap_early_exit_{timestamp}_fullscreen.png")
        if full_screen is not None and full_screen.size > 0:
            cv2.imwrite(raw_path, full_screen)

        return mm_path, raw_path


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

    test_imgs = sys.argv[1:] if len(sys.argv) > 1 else []

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
                out_dir = "debug_output"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, out_name)
                vis = detector.draw_overlay(img, detected)
                cv2.imwrite(out_path, vis)
                print(f"  -> Saved overlay preview: {out_path}")
