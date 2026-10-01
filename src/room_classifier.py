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
        self.max_lost_frames: int = 18  # ~0.7-0.8s persistence buffer with dead reckoning

        # Diagnostics & Automated Loss Recorder
        self.debug_log_dir: str = "debug_logs/tracker_lost"
        self.last_debug_dump_time: float = 0.0
        self.debug_dump_count: int = 0
        self.max_debug_dumps: int = 60
        self.min_debug_dump_interval: float = 1.5

        # Zone Auto-Discovery state (when no template images exist)
        self.auto_zones: Dict[str, Tuple[float, float]] = {}  # zone_id -> (center_x, center_y)

        # Room-specific threshold lookup table from config
        self.room_thresholds: Dict[str, float] = {
            r.get("id"): float(r.get("threshold", 0.45))
            for r in self.config.get("rooms", []) if isinstance(r, dict)
        }
        self.r7_template_ready: bool = False
        self.orb_r7 = None
        self.r7_kp = None
        self.r7_des = None
        self.bf_r7 = None
        self.r7_threshold: float = self.room_thresholds.get("room_7", 0.25)

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
                self._init_room_7_specialized_template(full_map_path)
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
                "variant": v_name, "path": p_name, "img": img,
                "edge_map": e_map, "keypoints": kp, "descriptors": des,
            }

        if template_img is not None:
            loaded_templates.append(process_img(template_img, variant_name, None))

        if template_paths:
            for p in template_paths:
                full_path = p if os.path.isabs(p) else os.path.join(self.config_dir, p)
                if os.path.exists(full_path):
                    img = cv2.imread(full_path)
                    if img is not None:
                        variant = os.path.splitext(os.path.basename(p))[0]
                        loaded_templates.append(process_img(img, variant, p))

        room_entry = {"id": room_id, "name": name, "threshold": threshold, "templates": loaded_templates}
        for idx, r in enumerate(self.rooms):
            if r["id"] == room_id:
                room_entry["templates"] = self.rooms[idx]["templates"] + loaded_templates
                self.rooms[idx] = room_entry
                return
        self.rooms.append(room_entry)

    def _init_room_7_specialized_template(self, map_path: str):
        """Initializes dedicated high-resolution Canny structural edge template for Room 7."""
        try:
            full_map = cv2.imread(map_path)
            if full_map is None:
                return
            # Room 7 bounding region on master layout [0:220, 0:220]
            r7_crop = full_map[0:220, 0:220]
            edges_r7 = cv2.Canny(cv2.cvtColor(r7_crop, cv2.COLOR_BGR2GRAY), 50, 150)
            self.orb_r7 = cv2.ORB_create(nfeatures=1000)
            self.r7_kp, self.r7_des = self.orb_r7.detectAndCompute(edges_r7, None)
            self.bf_r7 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.r7_threshold = self.room_thresholds.get("room_7", 0.25)
            self.r7_template_ready = (self.r7_des is not None and len(self.r7_kp) >= 10)
            if self.r7_template_ready:
                print(f"[ROOM 7 ENGINE] Initialized specialized Room 7 Canny template ({len(self.r7_kp)} features, thresh={self.r7_threshold})")
        except Exception as e:
            print(f"[ROOM CLASSIFIER] Warning initializing Room 7 template: {e}")

    def _match_room_7_orb(self, minimap_crop: np.ndarray) -> Tuple[float, Optional[Tuple[float, float]], int]:
        """Specialized ORB matcher for Room 7 thin purple walls using Canny edges and consensus fallback."""
        if not self.r7_template_ready or self.r7_des is None:
            return 0.0, None, 0
        try:
            edges_mini = cv2.Canny(cv2.cvtColor(minimap_crop, cv2.COLOR_BGR2GRAY), 50, 150)
            kp_mini, des_mini = self.orb_r7.detectAndCompute(edges_mini, None)
            if des_mini is None or len(kp_mini) < 3:
                return 0.0, None, 0
            matches = self.bf_r7.match(self.r7_des, des_mini)
            good = [m for m in sorted(matches, key=lambda x: x.distance) if m.distance < 65]
            if len(good) < 3:
                return 0.0, None, len(good)
            m_h, m_w = minimap_crop.shape[:2]
            src_pts = np.float32([self.r7_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_mini[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.5)
            aff_valid, aff_pos, aff_inliers = False, None, 0
            if M is not None and mask is not None:
                aff_inliers = int(np.sum(mask))
                if aff_inliers >= (4 if self.is_locked else 5):
                    theta, scale = math.atan2(M[1, 0], M[0, 0]), math.hypot(M[0, 0], M[0, 1])
                    if abs(theta) <= math.radians(14) and abs(scale - 1.0) <= 0.22:
                        M_inv = cv2.invertAffineTransform(M)
                        t = cv2.transform(np.array([[[m_w / 2.0, m_h / 2.0]]], dtype=np.float32), M_inv)
                        aff_pos = (round(float(t[0][0][0]), 1), round(float(t[0][0][1]), 1))
                        aff_valid = True
            # Translation consensus fallback
            con_pos, con_inliers = None, 0
            if not aff_valid or aff_inliers < 6:
                cands = [
                    (self.r7_kp[m.queryIdx].pt[0] - (kp_mini[m.trainIdx].pt[0] - m_w / 2.0),
                     self.r7_kp[m.queryIdx].pt[1] - (kp_mini[m.trainIdx].pt[1] - m_h / 2.0))
                    for m in good
                ]
                best_cl = []
                for p in cands:
                    cl = [q for q in cands if math.hypot(p[0] - q[0], p[1] - q[1]) <= 4.5]
                    if len(cl) > len(best_cl):
                        best_cl = cl
                if len(best_cl) >= (3 if self.is_locked else 4):
                    con_pos = (round(max(0.0, min(sum(c[0] for c in best_cl) / len(best_cl), 220.0)), 1),
                               round(max(0.0, min(sum(c[1] for c in best_cl) / len(best_cl), 220.0)), 1))
                    con_inliers = len(best_cl)
            if aff_valid and (con_pos is None or aff_inliers >= con_inliers):
                return min(1.0, float(aff_inliers) / 6.0), aff_pos, aff_inliers
            elif con_pos is not None:
                return min(1.0, float(con_inliers) / 6.0), con_pos, con_inliers
            return 0.0, None, max(aff_inliers, con_inliers)
        except Exception:
            return 0.0, None, 0

    def _match_template_multiscale(
        self, target_img: np.ndarray, template_img: np.ndarray
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        """Fallback multi-scale correlation matching for small room templates."""
        t_h, t_w = template_img.shape[:2]
        m_h, m_w = target_img.shape[:2]
        if t_w <= 0 or t_h <= 0 or m_w <= 0 or m_h <= 0 or t_w > 350 or t_h > 350:
            return 0.0, None
        g_t = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) if len(template_img.shape) == 3 else template_img
        g_m = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape) == 3 else target_img
        best_score, best_pos = 0.0, None
        for s in [0.90, 0.95, 1.0, 1.05]:
            sw, sh = int(t_w * s), int(t_h * s)
            if 0 < sw <= m_w and 0 < sh <= m_h:
                resized = cv2.resize(g_t, (sw, sh))
                r = cv2.matchTemplate(g_m, resized, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(r)
                if mv > best_score:
                    best_score = float(mv)
                    best_pos = (round(ml[0] + sw / 2.0, 1), round(ml[1] + sh / 2.0, 1))
        return round(min(1.0, max(0.0, best_score)), 4), best_pos

    def _match_orb_features(
        self,
        target_img: np.ndarray,
        target_edges: np.ndarray,
        target_kp: List,
        target_des: np.ndarray,
        template_entry: Dict[str, Any],
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        expected_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, Optional[Tuple[float, float]], int]:
        """Matches ORB features using strict rigid affine estimation and translation consensus."""
        temp_des = template_entry["descriptors"]
        temp_kp = template_entry["keypoints"]
        temp_img = template_entry["img"]
        m_h, m_w = target_edges.shape[:2]
        t_h, t_w = temp_img.shape[:2]
        corr_score, corr_pos = self._match_template_multiscale(target_img, temp_img)
        default_pos = corr_pos or (round(t_w / 2.0, 1), round(t_h / 2.0, 1))

        if target_des is None or temp_des is None or len(target_des) < 2 or len(temp_des) < 2:
            return corr_score, (default_pos if corr_score >= 0.35 else None), 0

        # Pre-filter template keypoints by search_roi or expected_pos to eliminate cross-room false matches
        cur_temp_kp = temp_kp
        cur_temp_des = temp_des
        if (t_w > 400 or t_h > 400) and (search_roi is not None or expected_pos is not None):
            if search_roi is not None:
                rx1, ry1, rx2, ry2 = search_roi
                margin = 70.0
                roi_indices = [
                    i for i, k in enumerate(temp_kp)
                    if (rx1 - margin <= k.pt[0] <= rx2 + margin) and (ry1 - margin <= k.pt[1] <= ry2 + margin)
                ]
            else:
                ex, ey = expected_pos
                rad = 180.0
                roi_indices = [
                    i for i, k in enumerate(temp_kp)
                    if abs(k.pt[0] - ex) <= rad and abs(k.pt[1] - ey) <= rad
                ]
            if len(roi_indices) >= 12:
                cur_temp_kp = [temp_kp[i] for i in roi_indices]
                cur_temp_des = temp_des[roi_indices]

        raw_matches = self.bf_matcher.match(cur_temp_des, target_des)
        matches = sorted(raw_matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 65]

        if len(good_matches) < 3:
            combined = max(corr_score, float(len(good_matches)) / 10.0)
            c_pos = default_pos if corr_score >= 0.35 else None
            return round(min(1.0, combined), 4), c_pos, len(good_matches)

        src_pts = np.float32([cur_temp_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 1. Rigid partial affine estimation with RANSAC
        M_affine, inliers_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.5, maxIters=2000
        )

        affine_valid = False
        affine_pos = None
        affine_inliers = 0
        if M_affine is not None and inliers_mask is not None:
            affine_inliers = int(np.sum(inliers_mask))
            min_inliers_required = 4 if self.is_locked else 5
            if affine_inliers >= min_inliers_required:
                theta = math.atan2(M_affine[1, 0], M_affine[0, 0])
                scale_est = math.hypot(M_affine[0, 0], M_affine[0, 1])
                # Minimap orientation & scale sanity check: PoE2 minimap does not wildly rotate
                if abs(theta) <= math.radians(12) and abs(scale_est - 1.0) <= 0.20:
                    try:
                        center_pt = np.array([[[m_w / 2.0, m_h / 2.0]]], dtype=np.float32)
                        M_inv = cv2.invertAffineTransform(M_affine)
                        transformed = cv2.transform(center_pt, M_inv)
                        raw_char_x = float(transformed[0][0][0])
                        raw_char_y = float(transformed[0][0][1])
                        affine_pos = (
                            round(max(0.0, min(raw_char_x, float(t_w))), 1),
                            round(max(0.0, min(raw_char_y, float(t_h))), 1),
                        )
                        affine_valid = True
                    except Exception:
                        affine_valid = False

        # 2. Translation Consensus Fallback (recovers frames where unconstrained affine rotated/scaled spuriously)
        consensus_pos = None
        consensus_inliers = 0
        if not affine_valid or affine_inliers < 6:
            cands = []
            for m in good_matches:
                pt_m = cur_temp_kp[m.queryIdx].pt
                pt_t = target_kp[m.trainIdx].pt
                px = pt_m[0] - (pt_t[0] - m_w / 2.0)
                py = pt_m[1] - (pt_t[1] - m_h / 2.0)
                cands.append((px, py))
            best_cluster = []
            for p in cands:
                cluster = [q for q in cands if math.hypot(p[0] - q[0], p[1] - q[1]) <= 4.5]
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster
            min_consensus = 3 if self.is_locked else 4
            if len(best_cluster) >= min_consensus:
                cx = round(sum(c[0] for c in best_cluster) / len(best_cluster), 1)
                cy = round(sum(c[1] for c in best_cluster) / len(best_cluster), 1)
                consensus_pos = (
                    round(max(0.0, min(cx, float(t_w))), 1),
                    round(max(0.0, min(cy, float(t_h))), 1),
                )
                consensus_inliers = len(best_cluster)

        # Select best candidate
        if affine_valid and (consensus_pos is None or affine_inliers >= consensus_inliers):
            final_pos = affine_pos
            inliers = affine_inliers
        elif consensus_pos is not None:
            final_pos = consensus_pos
            inliers = consensus_inliers
        else:
            final_pos = default_pos if corr_score >= 0.35 else None
            inliers = max(affine_inliers, consensus_inliers)

        feature_score = min(1.0, float(inliers) / 6.0) if (final_pos is not None and inliers > 0) else 0.0
        score = round(max(feature_score, corr_score if final_pos is not None else 0.0), 4)
        return score, final_pos, inliers

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
        recognized = False

        # Check if active search context indicates Room 7
        is_room_7 = False
        if search_roi is not None:
            rx1, ry1, rx2, ry2 = search_roi
            if rx2 <= 240 and ry2 <= 240:
                is_room_7 = True
        elif expected_pos is not None:
            if expected_pos[0] <= 210 and expected_pos[1] <= 210:
                is_room_7 = True

        # Fast specialized match for Room 7 purple walls
        if is_room_7 and self.r7_template_ready:
            r7_score, r7_pos, r7_inliers = self._match_room_7_orb(minimap_crop)
            if r7_score >= self.r7_threshold and r7_pos is not None:
                best_room_id = "room_7"
                best_room_name = "Room 7"
                best_variant = "room_7_canny"
                best_char_pos = r7_pos
                highest_score = r7_score
                best_threshold = self.r7_threshold
                best_inliers = r7_inliers
                all_scores["room_7"] = {
                    "name": "Room 7", "score": round(r7_score, 4), "best_variant": best_variant,
                    "character_position": r7_pos, "inliers": r7_inliers, "threshold": self.r7_threshold,
                    "status": "matched",
                }
                recognized = True

        # Check if any room templates have valid image files
        has_active_templates = any(len(r["templates"]) > 0 for r in self.rooms)

        if not recognized and has_active_templates:
            for r in self.rooms:
                r_id, r_name, threshold, templates = r["id"], r["name"], r["threshold"], r["templates"]
                if not templates:
                    all_scores[r_id] = {
                        "name": r_name, "score": 0.0, "best_variant": None, "character_position": None,
                        "inliers": 0, "status": "template_missing",
                    }
                    continue

                room_best_score = -1.0
                room_best_variant = None
                room_best_pos = None
                room_best_inliers = 0

                for tmpl in templates:
                    variant_name = tmpl["variant"]
                    score, char_pos, inliers = self._match_orb_features(
                        minimap_crop, target_edges, target_kp, target_des, tmpl,
                        search_roi=search_roi, expected_pos=expected_pos,
                    )

                    # Spatial prior bonus/penalty if search_roi or expected_pos is configured
                    effective_score = score
                    if char_pos is not None:
                        cx, cy = char_pos
                        if search_roi is not None:
                            rx1, ry1, rx2, ry2 = search_roi
                            in_roi = (rx1 - 35 <= cx <= rx2 + 35) and (ry1 - 35 <= cy <= ry2 + 35)
                            effective_score = max(effective_score, score + 0.05) if in_roi else -1.0
                        elif expected_pos is not None:
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
                    "name": r_name, "score": round(min(1.0, max(0.0, room_best_score)), 4),
                    "best_variant": room_best_variant, "character_position": room_best_pos,
                    "inliers": room_best_inliers, "threshold": threshold,
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
        elif not has_active_templates and not recognized:
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
        jump_anomaly = False

        if recognized and best_char_pos is not None:
            self.lost_frame_count = 0
            self.is_locked = True
            self.last_room_name = best_room_name
            self.last_variant = best_variant

            if self.last_known_pos is not None and self.last_room_id == best_room_id:
                # If last_known_pos was outside the active search_roi/expected_pos while best_char_pos is inside, snap immediately
                last_pos_invalid = False
                if search_roi is not None:
                    sx1, sy1, sx2, sy2 = search_roi
                    if not (sx1 - 35 <= self.last_known_pos[0] <= sx2 + 35 and sy1 - 35 <= self.last_known_pos[1] <= sy2 + 35):
                        last_pos_invalid = True
                elif expected_pos is not None:
                    if math.hypot(self.last_known_pos[0] - expected_pos[0], self.last_known_pos[1] - expected_pos[1]) > 180.0:
                        last_pos_invalid = True

                if last_pos_invalid:
                    self.pending_jump_pos = None
                    self.pending_jump_count = 0
                    self.last_known_pos = best_char_pos
                    final_char_pos = best_char_pos
                else:
                    dx = best_char_pos[0] - self.last_known_pos[0]
                    dy = best_char_pos[1] - self.last_known_pos[1]
                    dist_moved = (dx * dx + dy * dy) ** 0.5

                    if dist_moved < self.still_threshold:
                        final_char_pos = self.last_known_pos
                        self.pending_jump_pos = None
                        self.pending_jump_count = 0
                    elif dist_moved > self.jump_threshold:
                        # Check if this is a legitimate forward movement or recovery along route
                        is_likely_recovery = False
                        if expected_pos is not None and math.hypot(best_char_pos[0] - expected_pos[0], best_char_pos[1] - expected_pos[1]) <= 65.0 and best_inliers >= 5:
                            is_likely_recovery = True
                        elif search_roi is not None:
                            rx1, ry1, rx2, ry2 = search_roi
                            if (rx1 <= best_char_pos[0] <= rx2 and ry1 <= best_char_pos[1] <= ry2) and best_inliers >= 6:
                                is_likely_recovery = True

                        if is_likely_recovery:
                            self.pending_jump_pos = None
                            self.pending_jump_count = 0
                            self.last_known_pos = best_char_pos
                            final_char_pos = best_char_pos
                            jump_anomaly = False
                        else:
                            # Potential wild jump / teleport anomaly
                            jump_anomaly = True
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
                                        jump_anomaly = False
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
                            
                            # Log jump anomaly diagnostics
                            self._dump_tracker_loss_diagnostics(
                                minimap_crop=minimap_crop,
                                raw_char_pos=best_char_pos,
                                score=highest_score,
                                inliers=best_inliers,
                                search_roi=search_roi,
                                expected_pos=expected_pos,
                                reason=f"jump_dist_{int(dist_moved)}px",
                                all_scores=all_scores,
                            )
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
            # Hysteresis grace period: if previously locked, hold/dead-reckon position for up to max_lost_frames
            if self.is_locked and self.last_known_pos is not None and self.lost_frame_count < self.max_lost_frames:
                self.lost_frame_count += 1
                # Dead reckoning: softly project last position towards expected route progress
                if expected_pos is not None and self.lost_frame_count > 1:
                    exp_dx = expected_pos[0] - self.last_known_pos[0]
                    exp_dy = expected_pos[1] - self.last_known_pos[1]
                    exp_dist = math.hypot(exp_dx, exp_dy)
                    if 0 < exp_dist:
                        step = min(2.5, exp_dist)
                        dr_x = round(self.last_known_pos[0] + (exp_dx / exp_dist) * step, 1)
                        dr_y = round(self.last_known_pos[1] + (exp_dy / exp_dist) * step, 1)
                        self.last_known_pos = (dr_x, dr_y)
                final_char_pos = self.last_known_pos
                recognized = True
                best_room_id = self.last_room_id
                best_room_name = self.last_room_name or "Map Layout"
                best_variant = self.last_variant
                highest_score = max(highest_score, best_threshold)
            else:
                was_locked = self.is_locked
                self.is_locked = False
                self.last_known_pos = None
                self.last_room_id = None
                self.last_room_name = None
                self.last_variant = None
                self.lost_frame_count = 0
                self.pending_jump_pos = None
                self.pending_jump_count = 0

                # Dump diagnostic capture only when tracker genuinely lost an active lock
                if was_locked:
                    self._dump_tracker_loss_diagnostics(
                        minimap_crop=minimap_crop,
                        raw_char_pos=best_char_pos,
                        score=highest_score,
                        inliers=best_inliers,
                        search_roi=search_roi,
                        expected_pos=expected_pos,
                        reason="lock_lost_low_confidence",
                        all_scores=all_scores,
                    )

        return {
            "recognized": recognized, "room_id": best_room_id if recognized else None,
            "room_name": best_room_name if recognized else "Unknown", "matched_variant": best_variant if recognized else None,
            "character_position": final_char_pos if recognized else None,
            "confidence": round(min(1.0, max(0.0, highest_score)), 4) if highest_score >= 0 else 0.0,
            "all_scores": all_scores, "minimap_roi_shape": minimap_crop.shape,
            "jump_rejected": jump_rejected, "jump_rejections_total": self.jump_rejections_total,
            "jump_anomaly": jump_anomaly,
        }

    def _dump_tracker_loss_diagnostics(
        self,
        minimap_crop: np.ndarray,
        raw_char_pos: Optional[Tuple[float, float]],
        score: float,
        inliers: int,
        search_roi: Optional[Tuple[int, int, int, int]],
        expected_pos: Optional[Tuple[float, float]],
        reason: str = "position_lost",
        all_scores: Optional[Dict[str, Any]] = None,
    ):
        """Saves timestamped minimap crop and JSON telemetry to debug_logs/tracker_lost/."""
        import time, datetime
        now = time.time()
        if (now - self.last_debug_dump_time) < self.min_debug_dump_interval or self.debug_dump_count >= self.max_debug_dumps:
            return
        self.last_debug_dump_time = now
        self.debug_dump_count += 1
        try:
            os.makedirs(self.debug_log_dir, exist_ok=True)
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            base_name = f"loss_{ts_str}_score{max(0.0, score):.2f}_{reason}"
            if minimap_crop is not None and isinstance(minimap_crop, np.ndarray) and minimap_crop.size > 0:
                cv2.imwrite(os.path.join(self.debug_log_dir, f"{base_name}.png"), minimap_crop)
            meta = {
                "timestamp": ts_str, "reason": reason, "confidence_score": round(score, 4),
                "inliers": inliers, "raw_char_pos": raw_char_pos, "last_known_pos": self.last_known_pos,
                "last_room_name": self.last_room_name, "search_roi": search_roi,
                "expected_pos": expected_pos, "all_scores": all_scores,
            }
            json_path = os.path.join(self.debug_log_dir, f"{base_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[TRACKER DIAGNOSTIC] Dumped loss telemetry to {json_path} (Reason: {reason}, Score: {score:.2f})")
        except Exception:
            pass
