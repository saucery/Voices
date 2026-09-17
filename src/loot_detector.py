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
                "min_width": 25,
                "max_width": 450,
                "min_height": 10,
                "max_height": 55,
                "min_aspect_ratio": 1.5,
                "min_bg_fraction": 0.30,
                "min_text_pixels": 12,
            },
            {
                "id": "ravens_reflection_purple",
                "name": "Raven's Reflection / T1 Purple Uniques",
                "enabled": True,
                "priority": 2,
                "type": "color_box",
                "bg_color": "purple",
                "text_color": "white",
                "min_width": 30,
                "max_width": 400,
                "min_height": 10,
                "max_height": 55,
                "min_aspect_ratio": 1.5,
                "min_bg_fraction": 0.35,
            },
            {
                "id": "custom_template_loot1",
                "name": "Template Matcher (ui/loot1.png)",
                "enabled": True,
                "priority": 3,
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
            elif r_type == "template":
                items = self._detect_template(screen, rule)
                all_detected.extend(items)

        # Sort all items by priority (1 is highest), then by vertical position or confidence
        all_detected.sort(key=lambda item: (item.priority, -item.confidence))

        # Apply Non-Maximum Suppression (deduplicate overlapping boxes)
        filtered = self._apply_nms(all_detected, iou_threshold=0.30)
        return filtered

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
        min_w = int(rule.get("min_width", 25))
        max_w = int(rule.get("max_width", 500))
        min_h = int(rule.get("min_height", 10))
        max_h = int(rule.get("max_height", 65))
        min_ar = float(rule.get("min_aspect_ratio", 1.3))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.25))
        min_text_px = int(rule.get("min_text_pixels", 14))

        # White background mask: high brightness, low saturation
        white_mask = (
            (r > 175) & (g > 170) & (b > 170) &
            (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 170)
        ).astype(np.uint8) * 255

        # Red text / border mask: pure red (high R, distinct difference from G and B)
        red_text_mask = (
            (r > 155) &
            (r.astype(np.int16) - g.astype(np.int16) > 55) &
            (r.astype(np.int16) - b.astype(np.int16) > 55) &
            (g < 110) & (b < 110)
        ).astype(np.uint8) * 255

        items: List[LootItem] = []

        # Pass 1: Horizontal White Box Contours (strictly 1D horizontal closing)
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        white_closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_horiz)
        w_contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in w_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Avoid top/bottom UI edge overlays on full screenshots (FPS graphs, status bars)
            if screen.shape[0] > 300 and (y < 35 or y + h > screen.shape[0] - 35):
                continue

            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue

            aspect_ratio = w / max(1.0, float(h))
            if aspect_ratio < min_ar:
                continue

            box_white = white_mask[y:y+h, x:x+w]
            box_red = red_text_mask[y:y+h, x:x+w]

            bg_frac = np.mean(box_white > 0)
            red_pixels = int(np.count_nonzero(box_red > 0))

            if bg_frac >= min_bg_frac and red_pixels >= min_text_px:
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
        # Filter out thin vertical light beams (width <= 2px) using horizontal morphological opening (3, 1)
        k_remove_beams = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        red_no_beams = cv2.morphologyEx(red_text_mask, cv2.MORPH_OPEN, k_remove_beams)

        kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        red_dilated = cv2.dilate(red_no_beams, kernel_red)
        r_contours, _ = cv2.findContours(red_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in r_contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            # Avoid top/bottom UI edge overlays on full screenshots
            if screen.shape[0] > 300 and (ry < 35 or ry + rh > screen.shape[0] - 35):
                continue

            if rw < 18 or rh < 5:
                continue

            raw_red_pixels = int(np.count_nonzero(red_no_beams[ry:ry+rh, rx:rx+rw] > 0))
            if raw_red_pixels < min_text_px:
                continue

            # Expand to cover the surrounding white background rectangle
            pad_x = 8
            pad_y = 6
            bx = max(0, rx - pad_x)
            by = max(0, ry - pad_y)
            bw = min(screen.shape[1] - bx, rw + 2 * pad_x)
            bh = min(screen.shape[0] - by, rh + 2 * pad_y)

            box_white = white_mask[by:by+bh, bx:bx+bw]
            bg_frac = np.mean(box_white > 0)

            if bg_frac >= min_bg_frac:
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
        min_w = int(rule.get("min_width", 25))
        max_w = int(rule.get("max_width", 450))
        min_h = int(rule.get("min_height", 10))
        max_h = int(rule.get("max_height", 65))
        min_ar = float(rule.get("min_aspect_ratio", 1.3))
        min_bg_frac = float(rule.get("min_bg_fraction", 0.30))

        # Purple / Magenta Hue in OpenCV HSV is ~ 135 to 172
        purple_mask = (
            (
                (hsv[:, :, 0] >= 135) & (hsv[:, :, 0] <= 172) &
                (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 70)
            ) | (
                (r > 115) & (b > 125) & (g < 95)
            )
        ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        purple_closed = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(purple_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[LootItem] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue

            aspect_ratio = w / max(1.0, float(h))
            if aspect_ratio < min_ar:
                continue

            box_purple = purple_mask[y:y+h, x:x+w]
            bg_frac = np.mean(box_purple > 0)

            if bg_frac >= min_bg_frac:
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
                        rule_name=rule.get("name", "Raven's Reflection (Purple)"),
                        priority=int(rule.get("priority", 2)),
                        confidence=confidence,
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
            elif item.priority == 2 or "purple" in item.rule_id:
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
