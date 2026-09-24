"""
Template loading for Simulacrums, loot labels, Delirium statue, portals, stash, and inventory UI elements.
"""
from __future__ import annotations

import os
import time
import math
import json
import threading
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np

from src.movement_path import MovementPath
from src.screen_capturer import ScreenCapturer
from src.loot_detector import LootDetector, LootItem
from src.minimap_extractor import MinimapExtractor
from src.enemy_detector import EnemyDetector
from src.navigator_base import DynamicModuleProxy, log_msg

_log = log_msg
pydirectinput = DynamicModuleProxy("pydirectinput")
window_focuser = DynamicModuleProxy("window_focuser")
stop_handler = DynamicModuleProxy("stop_handler")
keyboard = DynamicModuleProxy("keyboard")


class TemplateLoaderMixin:
    """Template loading for Simulacrums, loot labels, Delirium statue, portals, stash, and inventory UI elements."""

    def _load_sim_templates(self):
        """Loads sim1, sim2, and sim3 template images if present on disk."""
        sim_files = {
            "sim1": self.sim1_template_file,
            "sim2": self.sim2_template_file,
            "sim3": self.sim3_template_file,
        }
        for key, filepath in sim_files.items():
            loaded_img = None
            for candidate in [filepath, f"ui/{key}.png", f"templates/ui/{key}.png"]:
                if candidate and os.path.exists(candidate):
                    loaded_img = cv2.imread(candidate)
                    if loaded_img is not None:
                        break
            self.sim_templates[key] = loaded_img


    def _load_loot_template(self):
        """Loads loot1 template image if present on disk."""
        self.loot1_img = None
        for candidate in [self.loot1_template_file, "ui/loot1.png", "templates/ui/loot1.png"]:
            if candidate and os.path.exists(candidate):
                self.loot1_img = cv2.imread(candidate)
                if self.loot1_img is not None:
                    break


    def _load_delirium_and_portal_templates(self):
        """Loads and precomputes masks for Delirium statue and exit Portal templates."""
        # 1. Portal template (white background cutout)
        self.portal_template_img = None
        self.portal_mask = None
        for candidate in [self.portal_template_file, "ui/portal.png", "templates/ui/portal.png"]:
            if candidate and os.path.exists(candidate):
                p_img = cv2.imread(candidate)
                if p_img is not None:
                    self.portal_template_img = p_img
                    # Mask out pure white background (B>=245, G>=245, R>=245)
                    self.portal_mask = np.uint8(
                        ~((p_img[:, :, 0] >= 245) & (p_img[:, :, 1] >= 245) & (p_img[:, :, 2] >= 245)) * 255
                    )
                    break

        # 2. Delirium statue templates (supporting both clean ground and heavy burning ground fire)
        self.delirium_templates = []
        loaded_candidates = set()
        for name, candidate_list in [
            ("delirium_clean", [self.delirium_template_file, "templates/ui/delirium.png", "ui/delirium.png"]),
            ("delirium_fire", [self.delirium_fire_template_file, "templates/ui/delirium_fire.png", "ui/delirium_fire.png"]),
        ]:
            for cand in candidate_list:
                if cand and os.path.exists(cand) and cand not in loaded_candidates:
                    d_img = cv2.imread(cand)
                    if d_img is not None:
                        # Mask out pure black background
                        d_mask = np.uint8(((d_img[:, :, 0] > 10) | (d_img[:, :, 1] > 10) | (d_img[:, :, 2] > 10)) * 255)
                        self.delirium_templates.append((name, d_img, d_mask))
                        loaded_candidates.add(cand)
                        break


    def _load_stash_and_inventory_templates(self, force_reload: bool = False):
        """Loads reference templates for Hideout Stash and Inventory window detection."""
        if force_reload:
            self.stash_label_tpl = None
            self.stash_full_tpl = None
            self.inventory_title_tpl = None
            self.inventory_close_tpl = None
            self.hideout_layout_tpl = None
            self.map_device_label_tpl = None
            self.map_device_full_tpl = None
            self.simulacrum_icon_tpl = None
            self.simulacrum_medal_tpl = None
            self.simulacrum_node_tpl = None
            self.simulacrum_circle_tpl = None
            self.simulacrum_node_v1_tpl = None
            self.simulacrum_node_v2_tpl = None
            self.simulacrum_node_v3_tpl = None
            self.delusion_popup_tpl = None
            self.delusion_4square_tpl = None
            self.delusion_traverse_tpl = None
            self.delusion_title_tpl = None
            self.tier15_map_tpl = None

        if self.stash_label_tpl is None:
            for cand in [self.stash_label_template_file, "templates/ui/stash_label.png", "ui/stash_label.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.stash_label_tpl = img
                        break

        if self.stash_full_tpl is None:
            for cand in [self.stash_full_template_file, "templates/ui/stash_full.png", "ui/stash_full.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.stash_full_tpl = img
                        break

        if self.inventory_title_tpl is None:
            for cand in [self.inventory_title_template_file, "templates/ui/inventory_title.png", "ui/inventory_title.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.inventory_title_tpl = img
                        break

        if self.inventory_close_tpl is None:
            for cand in [self.inventory_close_template_file, "templates/ui/inventory_close.png", "ui/inventory_close.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.inventory_close_tpl = img
                        break

        if self.hideout_layout_tpl is None:
            for cand in [self.hideout_layout_template_file, "templates/ui/hideout_layout.png", "ui/hideout_layout.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.hideout_layout_tpl = img
                        break

        if self.map_device_label_tpl is None:
            for cand in [self.map_device_label_template_file, "templates/ui/map_device_label.png", "ui/map_device_label.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.map_device_label_tpl = img
                        break

        if self.map_device_full_tpl is None:
            for cand in [self.map_device_full_template_file, "templates/ui/map_device_full.png", "ui/map_device_full.png", "templates/ui/map_device.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.map_device_full_tpl = img
                        break

        if self.simulacrum_icon_tpl is None:
            for cand in [self.simulacrum_icon_template_file, "templates/ui/simulacrum_icon.png", "ui/simulacrum_icon.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.simulacrum_icon_tpl = img
                        break

        if self.simulacrum_medal_tpl is None:
            for cand in [self.simulacrum_medal_template_file, "templates/ui/simulacrum_medal.png", "ui/simulacrum_medal.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.simulacrum_medal_tpl = img
                        break

        if self.simulacrum_node_tpl is None:
            for cand in [self.simulacrum_node_template_file, "templates/ui/simulacrum_node_full.png", "ui/simulacrum_node_full.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.simulacrum_node_tpl = img
                        break

        if self.simulacrum_circle_tpl is None:
            for cand in [self.simulacrum_circle_template_file, "templates/ui/simulacrum_circle.png", "ui/simulacrum_circle.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.simulacrum_circle_tpl = img
                        break

        for attr, fname, cands in [
            ("simulacrum_node_v1_tpl", self.simulacrum_node_v1_template_file, ["templates/ui/simulacrum_node_v1.png", "ui/simulacrum_node_v1.png"]),
            ("simulacrum_node_v2_tpl", self.simulacrum_node_v2_template_file, ["templates/ui/simulacrum_node_v2.png", "ui/simulacrum_node_v2.png"]),
            ("simulacrum_node_v3_tpl", self.simulacrum_node_v3_template_file, ["templates/ui/simulacrum_node_v3.png", "ui/simulacrum_node_v3.png"]),
        ]:
            if getattr(self, attr, None) is None:
                for cand in [fname] + cands:
                    if cand and os.path.exists(cand):
                        img = cv2.imread(cand)
                        if img is not None:
                            setattr(self, attr, img[:, :, :3] if img.shape[-1] == 4 else img)
                            break

        if self.delusion_popup_tpl is None:
            for cand in [self.delusion_popup_template_file, getattr(self, "delusion_popup_full_template_file", None), "templates/ui/delusion_popup_full.png", "templates/ui/delusion_popup.png", "ui/delusion_popup.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.delusion_popup_tpl = img
                        break

        if getattr(self, "delusion_4square_tpl", None) is None:
            for cand in [getattr(self, "delusion_4square_template_file", None), "templates/ui/delusion_4square_slots.png", "ui/delusion_4square_slots.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.delusion_4square_tpl = img
                        break

        if getattr(self, "delusion_traverse_tpl", None) is None:
            for cand in [getattr(self, "delusion_traverse_template_file", None), "templates/ui/traverse_button.png", "ui/traverse_button.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.delusion_traverse_tpl = img
                        break

        if getattr(self, "delusion_title_tpl", None) is None:
            for cand in [getattr(self, "delusion_title_template_file", None), "templates/ui/delusion_title_banner.png", "ui/delusion_title_banner.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.delusion_title_tpl = img
                        break

        if self.tier15_map_tpl is None:
            for cand in [self.tier15_map_template_file, "templates/ui/tier15_map.png", "ui/tier15_map.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.tier15_map_tpl = img
                        break

        if getattr(self, "map_node_completed_tpl", None) is None:
            for cand in [getattr(self, "map_node_completed_template_file", None), "templates/ui/map_node_completed.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.map_node_completed_tpl = img
                        break

        if getattr(self, "map_node_accessible_tpl", None) is None:
            for cand in [getattr(self, "map_node_accessible_template_file", None), "templates/ui/map_node_accessible.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.map_node_accessible_tpl = img
                        break

        if getattr(self, "map_node_inaccessible_tpl", None) is None:
            for cand in [getattr(self, "map_node_inaccessible_template_file", None), "templates/ui/map_node_inaccessible.png"]:
                if cand and os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        self.map_node_inaccessible_tpl = img
                        break

