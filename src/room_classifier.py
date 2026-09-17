"""
Room Classifier Module
Recognizes current room location by matching extracted minimap ROI against room template variants,
OR automatically auto-discovers continuous map zones (Zone 1, Zone 2, etc.) if no room templates are provided.
"""

import json
import os
import glob
import math
from typing import Dict, Any, Union, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from .minimap_extractor import MinimapExtractor


class RoomClassifier:
    """Classifies rooms via templates OR automatically discovers continuous map zones."""

    def __init__(
        self,
        config_path_or_dict: Union[str, Dict[str, Any]] = "config.json",
        map_layout_path: Optional[str] = None,
    ):
        """
        Initialize RoomClassifier.

        :param config_path_or_dict: Path to config JSON file or dictionary.
        :param map_layout_path: Optional path to single master map layout file (overrides config).
        """
        if isinstance(config_path_or_dict, str):
            if os.path.exists(config_path_or_dict):
                with open(config_path_or_dict, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                self.config_dir = os.path.dirname(os.path.abspath(config_path_or_dict))
            else:
                self.config = {}
                self.config_dir = os.getcwd()
        else:
            self.config = config_path_or_dict
            self.config_dir = os.getcwd()

        roi_cfg = self.config.get("minimap_roi", {})
        self.extractor = MinimapExtractor(roi_cfg)
        self.matching_cfg = self.config.get("matching", {})
        self.rooms: List[Dict[str, Any]] = []

        # Feature detector
        nfeatures = self.matching_cfg.get("orb_features", 1000)
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Position smoothing state & temporal persistence
        self.last_known_pos: Optional[Tuple[float, float]] = None
        self.last_room_id: Optional[str] = None
        self.last_room_name: Optional[str] = None
        self.last_variant: Optional[str] = None
        self.still_threshold: float = 3.5
        self.jump_threshold: float = 60.0
        self.jump_consensus_frames: int = 3
        self.pending_jump_pos: Optional[Tuple[float, float]] = None
        self.pending_jump_count: int = 0
        self.jump_rejections_total: int = 0
        self.is_locked: bool = False
        self.lost_frame_count: int = 0
        self.outlier_frame_count: int = 0
        self.max_lost_frames: int = 8  # ~0.3s persistence buffer against transient frame drops

        # Zone Auto-Discovery state (when no template images exist)
        self.auto_zones: Dict[str, Tuple[float, float]] = {}  # zone_id -> (center_x, center_y)

        # 1. Single Master Map Layout Mode (Loads EXACTLY ONE map file)
        single_map_file = map_layout_path or self.config.get("map_layout_file")
        if single_map_file:
            full_map_path = (
                single_map_file if os.path.isabs(single_map_file)
                else os.path.join(self.config_dir, single_map_file)
            )
            if os.path.exists(full_map_path):
                self.add_room(
                    room_id="map_layout",
                    name="Map Layout",
                    template_paths=[full_map_path],
                    threshold=self.matching_cfg.get("match_threshold", 0.45),
                )
                print(f"[MAP LOADER] Loaded single map layout: {single_map_file}")
                return
            else:
                print(f"[MAP LOADER] Warning: Configured map_layout_file '{single_map_file}' not found.")

        # 2. Legacy fallback: Load room configurations and templates
        rooms_list = self.config.get("rooms", [])
        for r_cfg in rooms_list:
            templates = r_cfg.get("templates") or []
            if "template_path" in r_cfg and r_cfg["template_path"]:
                templates.append(r_cfg["template_path"])

            self.add_room(
                room_id=r_cfg["id"],
                name=r_cfg.get("name", r_cfg["id"]),
                template_paths=templates,
                threshold=r_cfg.get("threshold", 0.45),
            )

    def add_room(
        self,
        room_id: str,
        name: str,
        template_paths: Optional[List[str]] = None,
        threshold: float = 0.45,
        template_img: Optional[np.ndarray] = None,
        variant_name: str = "default",
    ):
        """Add or update a room reference template collection."""
        loaded_templates = []

        def process_img(img, v_name, p_name):
            e_map = self.extractor.preprocess(img)
            kp, des = self.orb.detectAndCompute(e_map, None)
            if des is None or len(kp) < 2:
                g_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                kp, des = self.orb.detectAndCompute(g_map, None)

            return {
                "variant": v_name,
                "path": p_name,
                "img": img,
                "edge_map": e_map,
                "keypoints": kp,
                "descriptors": des,
            }

        if template_img is not None:
            loaded_templates.append(process_img(template_img, variant_name, None))

        if template_paths:
            for p in template_paths:
                full_path = (
                    p if os.path.isabs(p) else os.path.join(self.config_dir, p)
                )
                if os.path.exists(full_path):
                    img = cv2.imread(full_path)
                    if img is not None:
                        variant = os.path.splitext(os.path.basename(p))[0]
                        loaded_templates.append(process_img(img, variant, p))

        room_entry = {
            "id": room_id,
            "name": name,
            "threshold": threshold,
            "templates": loaded_templates,
        }

        for idx, r in enumerate(self.rooms):
            if r["id"] == room_id:
                existing_tmpl = self.rooms[idx]["templates"]
                combined = existing_tmpl + loaded_templates
                room_entry["templates"] = combined
                self.rooms[idx] = room_entry
                return

        self.rooms.append(room_entry)

    def _match_template_multiscale(
        self, target_img: np.ndarray, template_img: np.ndarray
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        """Fallback multi-scale correlation matching with location estimation."""
        t_h, t_w = template_img.shape[:2]
        m_h, m_w = target_img.shape[:2]

        if t_w <= 0 or t_h <= 0 or m_w <= 0 or m_h <= 0:
            return 0.0, None

        canny_t1 = self.matching_cfg.get("canny_threshold1", 50)
        canny_t2 = self.matching_cfg.get("canny_threshold2", 150)

        e_target = self.extractor.preprocess(target_img, canny_t1, canny_t2)
        e_temp = self.extractor.preprocess(template_img, canny_t1, canny_t2)

        g_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape) == 3 else target_img
        g_temp = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) if len(template_img.shape) == 3 else template_img

        scale_base = min(m_w / float(t_w), m_h / float(t_h)) if t_w > 0 and t_h > 0 else 1.0
        best_score = 0.0
        best_pos = None

        for rel_scale in [0.85, 0.95, 1.0, 1.05, 1.15]:
            s_w = int(t_w * scale_base * rel_scale)
            s_h = int(t_h * scale_base * rel_scale)
            if s_w <= 0 or s_h <= 0 or s_w > m_w or s_h > m_h:
                continue

            resized_e_temp = cv2.resize(e_temp, (s_w, s_h))
            resized_g_temp = cv2.resize(g_temp, (s_w, s_h))

            res_gray = cv2.matchTemplate(g_target, resized_g_temp, cv2.TM_CCOEFF_NORMED)
            _, max_val_gray, _, max_loc_g = cv2.minMaxLoc(res_gray)
            max_val_gray = max(0.0, float(max_val_gray))

            res_edge = cv2.matchTemplate(e_target, resized_e_temp, cv2.TM_CCOEFF_NORMED)
            _, max_val_edge, _, max_loc_e = cv2.minMaxLoc(res_edge)
            max_val_edge = max(0.0, float(max_val_edge))

            score = max(max_val_edge * 1.2, max_val_gray)
            if score > best_score:
                best_score = float(score)
                best_loc = max_loc_e if (max_val_edge * 1.2 >= max_val_gray) else max_loc_g
                best_pos = (round(best_loc[0] + s_w / 2.0, 1), round(best_loc[1] + s_h / 2.0, 1))

        return round(min(1.0, best_score), 4), best_pos

    def _match_orb_features(
        self,
        target_img: np.ndarray,
        target_edges: np.ndarray,
        target_kp: List,
        target_des: np.ndarray,
        template_entry: Dict[str, Any],
    ) -> Tuple[float, Optional[Tuple[float, float]], int]:
        """Matches ORB features using 2D Rigid Affine Transformation."""
        temp_des = template_entry["descriptors"]
        temp_kp = template_entry["keypoints"]
        temp_img = template_entry["img"]

        corr_score, corr_pos = self._match_template_multiscale(target_img, temp_img)
        m_h, m_w = target_edges.shape[:2]
        t_h, t_w = temp_img.shape[:2]
        default_pos = corr_pos or (round(t_w / 2.0, 1), round(t_h / 2.0, 1))

        if target_des is None or temp_des is None or len(target_des) < 2 or len(temp_des) < 2:
            return corr_score, default_pos, 0

        raw_matches = self.bf_matcher.match(temp_des, target_des)
        matches = sorted(raw_matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 65]

        if len(good_matches) < 3:
            combined_score = max(corr_score, float(len(good_matches)) / 10.0)
            return round(min(1.0, combined_score), 4), default_pos, len(good_matches)

        src_pts = np.float32([temp_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M_affine, inliers_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.5, maxIters=2000
        )

        if M_affine is None or inliers_mask is None:
            return corr_score, default_pos, len(good_matches)

        inliers = int(np.sum(inliers_mask))
        # Adaptive threshold: when already locked onto the player, allow 4 inliers to maintain lock smoothly
        min_inliers_required = 4 if self.is_locked else 5
        if inliers < min_inliers_required:
            return corr_score, default_pos, inliers

        # Sanity check rotation & scale: PoE2 minimap does NOT rotate or scale wildly
        theta = math.atan2(M_affine[1, 0], M_affine[0, 0])
        scale_est = math.hypot(M_affine[0, 0], M_affine[0, 1])
        if abs(theta) > math.radians(12) or abs(scale_est - 1.0) > 0.20:
            # False correlation caused by background floor noise/particles
            return corr_score, default_pos, inliers

        center_pt = np.array([[[m_w / 2.0, m_h / 2.0]]], dtype=np.float32)

        try:
            M_inv = cv2.invertAffineTransform(M_affine)
            transformed = cv2.transform(center_pt, M_inv)
            raw_char_x = float(transformed[0][0][0])
            raw_char_y = float(transformed[0][0][1])

            char_x_clamped = round(max(0.0, min(raw_char_x, float(t_w))), 1)
            char_y_clamped = round(max(0.0, min(raw_char_y, float(t_h))), 1)
            char_pos = (char_x_clamped, char_y_clamped)
        except Exception:
            char_pos = default_pos

        feature_score = min(1.0, float(inliers) / 7.0)
        final_score = round(max(feature_score, corr_score), 4)

        return final_score, char_pos, inliers

    def classify(
        self,
        screenshot: Union[str, np.ndarray, Image.Image],
        is_crop: bool = False,
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        expected_pos: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Classifies current screenshot using loaded room templates,
        OR auto-discovers continuous map zones if no templates are provided.

        :param screenshot: Full game screenshot or pre-cropped minimap ROI.
        :param is_crop: Set to True if screenshot is already a cropped minimap ROI.
        :param search_roi: Optional (min_x, min_y, max_x, max_y) bounding box to prioritize candidates.
        :param expected_pos: Optional (X, Y) prior position from route progress.
        """
        if is_crop:
            minimap_crop = self.extractor._to_cv2(screenshot) if not isinstance(screenshot, np.ndarray) else screenshot
        else:
            minimap_crop = self.extractor.extract_roi(screenshot)
        target_edges = self.extractor.preprocess(minimap_crop)

        target_kp, target_des = self.orb.detectAndCompute(target_edges, None)
        if target_des is None or len(target_kp) < 2:
            g_crop = cv2.cvtColor(minimap_crop, cv2.COLOR_BGR2GRAY)
            target_kp, target_des = self.orb.detectAndCompute(g_crop, None)

        all_scores = {}
        best_room_id = None
        best_room_name = "Unknown"
        best_variant = None
        best_char_pos = None
        highest_score = -1.0
        best_threshold = 0.45
        best_inliers = 0

        # Check if any room templates have valid image files
        has_active_templates = any(len(r["templates"]) > 0 for r in self.rooms)

        if has_active_templates:
            for r in self.rooms:
                r_id = r["id"]
                r_name = r["name"]
                threshold = r["threshold"]
                templates = r["templates"]

                if not templates:
                    all_scores[r_id] = {
                        "name": r_name,
                        "score": 0.0,
                        "best_variant": None,
                        "character_position": None,
                        "inliers": 0,
                        "status": "template_missing",
                    }
                    continue

                room_best_score = -1.0
                room_best_variant = None
                room_best_pos = None
                room_best_inliers = 0

                for tmpl in templates:
                    variant_name = tmpl["variant"]
                    score, char_pos, inliers = self._match_orb_features(
                        minimap_crop, target_edges, target_kp, target_des, tmpl
                    )

                    # Spatial prior bonus/penalty if search_roi or expected_pos is configured
                    effective_score = score
                    if char_pos is not None:
                        cx, cy = char_pos
                        if search_roi is not None:
                            rx1, ry1, rx2, ry2 = search_roi
                            in_roi = (rx1 - 30 <= cx <= rx2 + 30) and (ry1 - 30 <= cy <= ry2 + 30)
                            if in_roi:
                                effective_score += 0.05
                            elif inliers < 8:
                                # Candidate is outside active route room zone with weak inliers
                                effective_score -= 0.08

                        if expected_pos is not None:
                            d_exp = math.hypot(cx - expected_pos[0], cy - expected_pos[1])
                            if d_exp < 60.0:
                                effective_score += 0.03
                            elif d_exp > 200.0 and inliers < 8:
                                effective_score -= 0.05

                    if effective_score > room_best_score:
                        room_best_score = effective_score
                        room_best_variant = variant_name
                        room_best_pos = char_pos
                        room_best_inliers = inliers

                all_scores[r_id] = {
                    "name": r_name,
                    "score": round(min(1.0, max(0.0, room_best_score)), 4),
                    "best_variant": room_best_variant,
                    "character_position": room_best_pos,
                    "inliers": room_best_inliers,
                    "threshold": threshold,
                    "status": "matched" if room_best_score >= threshold else "below_threshold",
                }

                if room_best_score > highest_score:
                    highest_score = room_best_score
                    best_room_id = r_id
                    best_room_name = r_name
                    best_variant = room_best_variant
                    best_char_pos = room_best_pos
                    best_threshold = threshold
                    best_inliers = room_best_inliers

            recognized = highest_score >= best_threshold and best_room_id is not None
        else:
            # Automatic Zone Discovery mode (no template images required)
            recognized = True
            best_room_id = "AutoMap"
            best_room_name = "Global World Map"
            best_variant = "auto_stitched"
            highest_score = 1.00
            best_char_pos = (round(minimap_crop.shape[1] / 2.0, 1), round(minimap_crop.shape[0] / 2.0, 1))
            best_inliers = 999

        # Position smoothing with velocity jump gating & temporal consensus
        final_char_pos = None
        jump_rejected = False

        if recognized and best_char_pos is not None:
            self.lost_frame_count = 0
            self.is_locked = True
            self.last_room_name = best_room_name
            self.last_variant = best_variant

            if self.last_known_pos is not None and self.last_room_id == best_room_id:
                dx = best_char_pos[0] - self.last_known_pos[0]
                dy = best_char_pos[1] - self.last_known_pos[1]
                dist_moved = (dx * dx + dy * dy) ** 0.5

                if dist_moved < self.still_threshold:
                    final_char_pos = self.last_known_pos
                    self.pending_jump_pos = None
                    self.pending_jump_count = 0
                elif dist_moved > self.jump_threshold:
                    # Potential wild jump / teleport anomaly
                    if self.pending_jump_pos is not None:
                        cluster_dist = math.hypot(
                            best_char_pos[0] - self.pending_jump_pos[0],
                            best_char_pos[1] - self.pending_jump_pos[1],
                        )
                        if cluster_dist <= 30.0:
                            self.pending_jump_count += 1
                            if self.pending_jump_count >= self.jump_consensus_frames or best_inliers >= 10:
                                # Confirmed legitimate teleport/leap after consensus frames
                                self.pending_jump_pos = None
                                self.pending_jump_count = 0
                                self.last_known_pos = best_char_pos
                                final_char_pos = best_char_pos
                            else:
                                # Awaiting confirmation: hold last known position
                                final_char_pos = self.last_known_pos
                                self.jump_rejections_total += 1
                                jump_rejected = True
                        else:
                            # Inconsistent jump targets (spurious noise)
                            self.pending_jump_pos = best_char_pos
                            self.pending_jump_count = 1
                            final_char_pos = self.last_known_pos
                            self.jump_rejections_total += 1
                            jump_rejected = True
                    else:
                        # First anomalous jump frame: hold position and start consensus counter
                        self.pending_jump_pos = best_char_pos
                        self.pending_jump_count = 1
                        final_char_pos = self.last_known_pos
                        self.jump_rejections_total += 1
                        jump_rejected = True
                else:
                    # Genuine smooth movement (<= jump_threshold)
                    self.pending_jump_pos = None
                    self.pending_jump_count = 0
                    self.outlier_frame_count = 0
                    alpha = 0.65
                    sm_x = round(alpha * best_char_pos[0] + (1.0 - alpha) * self.last_known_pos[0], 1)
                    sm_y = round(alpha * best_char_pos[1] + (1.0 - alpha) * self.last_known_pos[1], 1)
                    final_char_pos = (sm_x, sm_y)
                    self.last_known_pos = final_char_pos
            else:
                self.pending_jump_pos = None
                self.pending_jump_count = 0
                self.outlier_frame_count = 0
                final_char_pos = best_char_pos
                self.last_known_pos = final_char_pos
                self.last_room_id = best_room_id
        else:
            # Hysteresis grace period: if previously locked, hold last position for up to max_lost_frames
            if self.is_locked and self.last_known_pos is not None and self.lost_frame_count < self.max_lost_frames:
                self.lost_frame_count += 1
                final_char_pos = self.last_known_pos
                recognized = True
                best_room_id = self.last_room_id
                best_room_name = self.last_room_name or "Map Layout"
                best_variant = self.last_variant
                highest_score = max(highest_score, best_threshold)
            else:
                self.is_locked = False
                self.last_known_pos = None
                self.last_room_id = None
                self.last_room_name = None
                self.last_variant = None
                self.lost_frame_count = 0
                self.pending_jump_pos = None
                self.pending_jump_count = 0

        return {
            "recognized": recognized,
            "room_id": best_room_id if recognized else None,
            "room_name": best_room_name if recognized else "Unknown",
            "matched_variant": best_variant if recognized else None,
            "character_position": final_char_pos if recognized else None,
            "confidence": round(min(1.0, max(0.0, highest_score)), 4) if highest_score >= 0 else 0.0,
            "all_scores": all_scores,
            "minimap_roi_shape": minimap_crop.shape,
            "jump_rejected": jump_rejected,
            "jump_rejections_total": self.jump_rejections_total,
        }
