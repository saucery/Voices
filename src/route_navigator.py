"""
Route Navigator Module
Handles real-time closed-loop WASD character navigation along a defined MovementPath.
Calculates 8-directional vector headings and holds/releases DirectInput WASD keys
towards active waypoints with arrival threshold detection and emergency stop safety.
"""

import json
import math
import os
import time
import threading
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.01
    pydirectinput.FAILSAFE = False
except ImportError:
    pydirectinput = None

try:
    import keyboard
except ImportError:
    keyboard = None

from .movement_path import MovementPath
from .screen_capturer import ScreenCapturer
from .stop_handler import stop_handler
from .window_focus import window_focuser
from .loot_detector import LootDetector, LootItem
from .minimap_extractor import MinimapExtractor
from .enemy_detector import EnemyDetector

# Mixin Functional Modules
from src.hideout.stash_handler import HideoutStashMixin
from src.hideout.inventory_handler import HideoutInventoryMixin
from src.hideout.map_device import HideoutMapDeviceMixin
from src.hideout.simulacrum_selector import HideoutSimulacrumMixin
from src.hideout.map_traverse import HideoutTraverseMixin
from src.hideout.hideout_manager import HideoutManagerMixin
from src.combat.loot_collector import LootCollectorMixin
from src.combat.delirium_statue import DeliriumStatueMixin
from src.combat.persistent_combat import PersistentCombatMixin
from src.combat.zone_routines import ZoneRoutinesMixin
from src.combat.zone_step_executor import ZoneStepExecutorMixin
from src.combat.zone_actions import ZoneActionsMixin
from src.combat.sim_clicker import SimClickerMixin
from src.combat.zone_interactions import ZoneInteractionsMixin
from src.navigation.wasd_mover import WasdMoverMixin
from src.navigation.orbit_controller import OrbitControllerMixin
from src.navigation.stuck_recovery import StuckRecoveryMixin
from src.navigation.navigator_controls import NavigatorControlsMixin
from src.navigation.template_loader import TemplateLoaderMixin
from src.navigation.route_follower import RouteFollowerMixin
from src.analytics.run_stats import RunStatsMixin


def _log(msg: str = "") -> None:
    """Prints a message prefixed with the current timestamp [HH:MM:SS]."""
    if not msg:
        print()
        return
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = ""
    if msg.startswith("\n"):
        prefix = "\n"
        msg = msg[1:]
    print(f"{prefix}[{ts}] {msg}")


class RouteNavigator(
    HideoutStashMixin,
    HideoutInventoryMixin,
    HideoutMapDeviceMixin,
    HideoutSimulacrumMixin,
    HideoutTraverseMixin,
    HideoutManagerMixin,
    LootCollectorMixin,
    DeliriumStatueMixin,
    PersistentCombatMixin,
    ZoneRoutinesMixin,
    ZoneStepExecutorMixin,
    ZoneActionsMixin,
    SimClickerMixin,
    ZoneInteractionsMixin,
    WasdMoverMixin,
    OrbitControllerMixin,
    StuckRecoveryMixin,
    NavigatorControlsMixin,
    TemplateLoaderMixin,
    RouteFollowerMixin,
    RunStatsMixin
):
    """
    Closed-loop autonomous character navigator using WASD controls in Path of Exile 2.
    Modularized into dedicated functional mixins for Hideout, Combat, Navigation, and Analytics.
    """

    def __init__(
        self,
        movement_path: Optional[MovementPath] = None,
        arrival_threshold: float = 18.0,
        monitor_idx: int = 2,
        step_duration: float = 0.22,
        capturer: Optional[Any] = None,
        config_path: str = "config.json",
    ):
        """
        Initialize RouteNavigator.

        :param movement_path: MovementPath instance containing waypoints.
        :param arrival_threshold: Pixel distance to waypoint to trigger advancement (default: 18 px).
        :param monitor_idx: Target monitor index for game window focus.
        :param step_duration: Duration to hold WASD keys per step (default: 0.22s).
        :param capturer: ScreenCapturer instance for banner detection.
        :param config_path: Path to config.json.
        """
        self.movement_path = movement_path or MovementPath()
        path_arrival = getattr(self.movement_path, "arrival_distance", None)
        if isinstance(path_arrival, (int, float)):
            self.arrival_threshold = max(float(arrival_threshold), float(path_arrival))
        else:
            self.arrival_threshold = float(arrival_threshold)
        self.monitor_idx = monitor_idx
        self.step_duration = step_duration
        self.capturer = capturer
        self.config_path = config_path

        self.is_active: bool = False
        self.is_paused: bool = False
        self._last_f4_time: float = 0.0
        self.is_completed: bool = False
        self.is_simulating_key: bool = False
        self.held_keys: Set[str] = set()
        self.latest_pos: Optional[Tuple[float, float]] = None
        self._worker_thread: Optional[threading.Thread] = None

        self.last_target: Optional[Dict[str, Any]] = None
        self.last_distance: float = 0.0
        self.status_message: str = "Autopilot Ready (Press 'A' to start)"

        # Stuck & Recovery Detection State
        self.last_known_pos: Optional[Tuple[float, float]] = None
        self.last_known_time: float = time.time()
        self.last_progress_pos: Optional[Tuple[float, float]] = None
        self.last_progress_time: float = time.time()
        self.stuck_counter: int = 0
        self.stuck_step_limit: int = 6           # ~1.8s of no progress
        self.tracking_lost_timeout: float = 1.8   # seconds before declaring lost
        self.last_held_keys: List[str] = []
        self.recent_movements: deque = deque(maxlen=25)
        self.backtrack_attempts_at_wp: int = 0
        self.max_backtrack_attempts_per_wp: int = 2
        self.latest_recovery_event: Optional[str] = None
        self.is_interacting: bool = False
        self._routine_did_orbit: bool = False
        self._log = _log

        # Run Session Metrics and SIM Tracking
        self.run_history_file: str = "debug_logs/run_history.json"
        self.run_summary_file: str = "debug_logs/run_history.txt"
        self.run_id: Optional[str] = None
        self.run_start_time: Optional[float] = None
        self.run_end_time: Optional[float] = None
        self.run_pause_time: Optional[float] = None
        self.total_paused_duration: float = 0.0
        self.run_completed: bool = False
        self.run_last_room_cleared: Optional[str] = None
        self.run_rooms_cleared: List[Dict[str, Any]] = []
        self.run_sims_clicked: List[Dict[str, Any]] = []
        self.run_loot_picked: List[Dict[str, Any]] = []
        self._last_clicked_loot_item: Optional[Any] = None
        self.current_room_key: Optional[str] = None
        self._last_completed_run_data: Dict[str, Any] = {}

        # Yellow Shape Orbit Navigation State
        self.is_orbiting: bool = False
        self.orbit_start_time: float = 0.0
        self.orbit_duration: float = 10.0
        self.current_orbit_zone: Optional[Dict[str, Any]] = None
        self.orbit_perimeter_pts: List[List[float]] = []
        self.orbit_point_idx: int = 0
        self.orbit_stuck_step_limit: int = 16       # ~4.0s of continuous zero progress while orbiting
        self.orbit_stuck_timeout_sec: float = 6.0   # Allow attack animations & turns while orbiting
        self.orbit_grace_until: float = 0.0         # Grace period timestamp on entering orbit

        # Yellow Zone Encounter Banner & Interaction Configuration
        self.orbit_yellow_zone_enabled: bool = True
        self.orbit_constant_right_click_enabled: bool = True
        self.orbit_right_click_interval_seconds: float = 0.75
        self.click_banner_enabled: bool = True
        self.right_click_after_banner_enabled: bool = True
        self.middle_click_hold_enabled: bool = True
        self.middle_click_hold_seconds: float = 3.0
        self.hold_q_enemy_reactive_enabled: bool = False
        self.enemy_detect_wait_timeout: float = 5.0
        self.enemy_near_distance_px: float = 200.0
        self.enemy_hold_min_seconds: float = 2.0
        self.enemy_hold_max_seconds: float = 6.0
        self.enemy_detector: Optional[EnemyDetector] = None
        self.encounter_banner_file: str = "ui/encounter_banner.png"
        self.encounter_match_threshold: float = 0.45
        self.banner_search_attempts: int = 5
        self.banner_approach_wait_seconds: float = 2.0
        self.banner_verify_delay_seconds: float = 1.0
        self.banner_max_click_attempts: int = 2
        self.banner_verify_click_enabled: bool = True
        self.sim_approach_wait_seconds: float = 2.0
        self.sim_verify_delay_seconds: float = 1.0
        self.sim_max_click_attempts: int = 2
        self.sim_verify_click_enabled: bool = True
        self.reclick_after_approach: bool = False
        self.interacted_zones: Set[str] = set()
        self.last_orbit_right_click: float = 0.0
        self.persistent_combat_active: bool = False
        self.persistent_combat_action: str = "key"
        self.persistent_combat_key: str = "t"
        self.persistent_combat_interval: float = 0.65
        self.persistent_right_click_active: bool = False
        self.persistent_right_click_interval: float = 0.65
        self.suppress_combat_during_approach: bool = False
        self.is_approaching_interactable: bool = False
        self._combat_thread: Optional[threading.Thread] = None
        self.is_holding_mouse: bool = False
        self.has_executed_initial_hold: bool = False

        # Pink Dot Sim & Banner Encounter State
        self.pink_dot_stop_seconds: float = 2.5
        self.sim1_template_file: str = "ui/sim1.png"
        self.sim2_template_file: str = "ui/sim2.png"
        self.sim3_template_file: str = "ui/sim3.png"
        self.sim_match_threshold: float = 0.50
        self.sim_click_y_offset_px: int = 35
        self.sim_click_x_offset_px: int = 0
        self.sim_max_clicks: Optional[int] = None
        self.start_at_pink_dot: int = 0
        self.target_pink_wp_idx: Optional[int] = None
        self.target_pink_name: Optional[str] = None
        self.target_pink_pos: Optional[List[float]] = None
        self.interacted_pink_dots: Set[int] = set()
        self._route_reset_to_start: bool = False

        # Delirium & Portal Encounter State (Room 7)
        self.portal_template_file: str = "templates/ui/portal.png"
        self.portal_match_threshold: float = 0.70
        self.portal_early_exit_min_seconds: float = 30.0
        self.portal_template_img: Optional[np.ndarray] = None
        self.portal_mask: Optional[np.ndarray] = None
        self._last_detected_portal_pos: Optional[Tuple[int, int]] = None
        self._last_detected_portal_time: float = 0.0

        self.delirium_template_file: str = "templates/ui/delirium.png"
        self.delirium_fire_template_file: str = "templates/ui/delirium_fire.png"
        self.delirium_match_threshold: float = 0.60
        self.delirium_templates: List[Tuple[np.ndarray, np.ndarray]] = []

        # Hideout & Stash Interaction State
        self.in_hideout: bool = False
        self.last_hideout_confidence: float = 0.0
        self.stash_label_template_file: str = "templates/ui/stash_label.png"
        self.stash_full_template_file: str = "templates/ui/stash_full.png"
        self.stash_match_threshold: float = 0.58
        self.stash_label_tpl: Optional[np.ndarray] = None
        self.stash_full_tpl: Optional[np.ndarray] = None

        self.inventory_title_template_file: str = "templates/ui/inventory_title.png"
        self.inventory_close_template_file: str = "templates/ui/inventory_close.png"
        self.inventory_match_threshold: float = 0.58
        self.inventory_title_tpl: Optional[np.ndarray] = None
        self.inventory_close_tpl: Optional[np.ndarray] = None

        self.hideout_layout_template_file: str = "templates/ui/hideout_layout.png"
        self.hideout_match_threshold: float = 0.50
        self.hideout_layout_tpl: Optional[np.ndarray] = None
        self.hideout_detection_mode: str = "minimap_or_map_device"
        self.inventory_screenshots_dir: str = "inventory_screenshots"

        # Atlas Map Device & Simulacrum Map Selection State
        self.map_device_label_template_file: str = "templates/ui/map_device_label.png"
        self.map_device_full_template_file: str = "templates/ui/map_device_full.png"
        self.map_device_match_threshold: float = 0.68
        self.map_device_click_y_offset_px: int = 0
        self.map_device_label_tpl: Optional[np.ndarray] = None
        self.map_device_full_tpl: Optional[np.ndarray] = None

        self.simulacrum_icon_template_file: str = "templates/ui/simulacrum_icon.png"
        self.simulacrum_medal_template_file: str = "templates/ui/simulacrum_medal.png"
        self.simulacrum_node_template_file: str = "templates/ui/simulacrum_node_full.png"
        self.simulacrum_circle_template_file: str = "templates/ui/simulacrum_circle.png"
        self.simulacrum_node_v1_template_file: str = "templates/ui/simulacrum_node_v1.png"
        self.simulacrum_node_v2_template_file: str = "templates/ui/simulacrum_node_v2.png"
        self.simulacrum_node_v3_template_file: str = "templates/ui/simulacrum_node_v3.png"
        self.simulacrum_match_threshold: float = 0.78
        self.simulacrum_click_y_offset: float = 26.0
        self.simulacrum_icon_tpl: Optional[np.ndarray] = None
        self.simulacrum_medal_tpl: Optional[np.ndarray] = None
        self.simulacrum_node_tpl: Optional[np.ndarray] = None
        self.simulacrum_circle_tpl: Optional[np.ndarray] = None
        self.simulacrum_node_v1_tpl: Optional[np.ndarray] = None
        self.simulacrum_node_v2_tpl: Optional[np.ndarray] = None
        self.simulacrum_node_v3_tpl: Optional[np.ndarray] = None

        self.delusion_popup_template_file: str = "templates/ui/delusion_popup.png"
        self.delusion_popup_full_template_file: str = "templates/ui/delusion_popup_full.png"
        self.delusion_4square_template_file: str = "templates/ui/delusion_4square_slots.png"
        self.delusion_traverse_template_file: str = "templates/ui/traverse_button.png"
        self.delusion_title_template_file: str = "templates/ui/delusion_title_banner.png"
        self.delusion_popup_tpl: Optional[np.ndarray] = None
        self.delusion_4square_tpl: Optional[np.ndarray] = None
        self.delusion_traverse_tpl: Optional[np.ndarray] = None
        self.delusion_title_tpl: Optional[np.ndarray] = None
        self.delusion_popup_match_threshold: float = 0.55
        self.delusion_detected_slots: List[Tuple[int, int]] = []
        self.delusion_detected_traverse: Optional[Tuple[int, int]] = None
        self.last_map_device_pos: Optional[Tuple[int, int]] = None
        self.traverse_match_threshold: float = 0.60
        self.hideout_portal_match_threshold: float = 0.65
        self.portal_entry_hold_w_seconds: float = 1.3
        self.portal_entry_settle_seconds: float = 1.5

        self.tier15_map_template_file: str = "templates/ui/tier15_map.png"
        self.tier15_map_tpl: Optional[np.ndarray] = None
        self.tier15_map_match_threshold: float = 0.60

        self.map_node_completed_template_file: str = "templates/ui/map_node_completed.png"
        self.map_node_accessible_template_file: str = "templates/ui/map_node_accessible.png"
        self.map_node_inaccessible_template_file: str = "templates/ui/map_node_inaccessible.png"
        self.map_node_completed_tpl: Optional[np.ndarray] = None
        self.map_node_accessible_tpl: Optional[np.ndarray] = None
        self.map_node_inaccessible_tpl: Optional[np.ndarray] = None
        self.sim_require_green_connection: bool = True

        self.waiting_for_user_key: bool = False
        self.waiting_user_key_name: str = "f5"

        # Loot Pickup State & Configuration
        self.loot1_template_file: str = "ui/loot1.png"
        self.loot_match_threshold: float = 0.50
        self.loot_pickup_wait_seconds: float = 0.4
        self.loot_approach_wait_seconds: float = 1.1
        self.loot_z_toggle_enabled: bool = True
        self.loot_z_toggle_delay_seconds: float = 0.3
        self._loot_labels_hidden: bool = False
        self.max_loot_pickups: int = 15
        self.wait_for_loot_confirmation: bool = True
        self.waiting_for_green_light: bool = False
        self.save_loot_debug_screenshots: bool = True
        self.save_pre_loot_screenshot: bool = True
        self.loot_debug_dir: str = "loot_debug"
        self.zone_routines_file: str = "routines/zone_routines.json"
        self.zone_routines: Optional[Dict[str, Any]] = None

        # Minimap Early Encounter Exit State & Configuration
        self.minimap_early_exit_enabled: bool = True
        self.minimap_early_exit_min_seconds: float = 25.0
        self.minimap_early_exit_check_interval: float = 0.5
        self.minimap_early_exit_min_icons: int = 1

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                ap_cfg = cfg.get("autopilot", {})
                self.orbit_yellow_zone_enabled = bool(ap_cfg.get("orbit_yellow_zone_enabled", self.orbit_yellow_zone_enabled))
                self.orbit_constant_right_click_enabled = bool(ap_cfg.get("orbit_constant_right_click_enabled", self.orbit_constant_right_click_enabled))
                self.orbit_right_click_interval_seconds = float(ap_cfg.get("orbit_right_click_interval_seconds", self.orbit_right_click_interval_seconds))
                self.click_banner_enabled = bool(ap_cfg.get("click_banner_enabled", self.click_banner_enabled))
                self.right_click_after_banner_enabled = bool(ap_cfg.get("right_click_after_banner_enabled", self.right_click_after_banner_enabled))
                self.middle_click_hold_enabled = bool(ap_cfg.get("middle_click_hold_enabled", self.middle_click_hold_enabled))
                self.middle_click_hold_seconds = float(ap_cfg.get("middle_click_hold_seconds", self.middle_click_hold_seconds))
                self.hold_q_enemy_reactive_enabled = bool(ap_cfg.get("hold_q_enemy_reactive_enabled", self.hold_q_enemy_reactive_enabled))
                self.enemy_detect_wait_timeout = float(ap_cfg.get("enemy_detect_wait_timeout", self.enemy_detect_wait_timeout))
                self.enemy_near_distance_px = float(ap_cfg.get("enemy_near_distance_px", self.enemy_near_distance_px))
                self.enemy_hold_min_seconds = float(ap_cfg.get("enemy_hold_min_seconds", self.enemy_hold_min_seconds))
                self.enemy_hold_max_seconds = float(ap_cfg.get("enemy_hold_max_seconds", self.enemy_hold_max_seconds))
                self.encounter_banner_file = ap_cfg.get("encounter_banner_file", self.encounter_banner_file)
                self.encounter_match_threshold = float(ap_cfg.get("encounter_match_threshold", self.encounter_match_threshold))
                self.banner_approach_wait_seconds = float(ap_cfg.get("banner_approach_wait_seconds", self.banner_approach_wait_seconds))
                self.banner_verify_delay_seconds = float(ap_cfg.get("banner_verify_delay_seconds", self.banner_verify_delay_seconds))
                self.banner_max_click_attempts = int(ap_cfg.get("banner_max_click_attempts", self.banner_max_click_attempts))
                self.banner_verify_click_enabled = bool(ap_cfg.get("banner_verify_click_enabled", self.banner_verify_click_enabled))
                self.sim_approach_wait_seconds = float(ap_cfg.get("sim_approach_wait_seconds", self.sim_approach_wait_seconds))
                self.sim_verify_delay_seconds = float(ap_cfg.get("sim_verify_delay_seconds", self.sim_verify_delay_seconds))
                self.sim_max_click_attempts = int(ap_cfg.get("sim_max_click_attempts", self.sim_max_click_attempts))
                self.sim_verify_click_enabled = bool(ap_cfg.get("sim_verify_click_enabled", self.sim_verify_click_enabled))
                self.reclick_after_approach = bool(ap_cfg.get("reclick_after_approach", self.reclick_after_approach))
                self.portal_entry_hold_w_seconds = float(ap_cfg.get("portal_entry_hold_w_seconds", self.portal_entry_hold_w_seconds))
                self.portal_entry_settle_seconds = float(ap_cfg.get("portal_entry_settle_seconds", self.portal_entry_settle_seconds))
                self.pink_dot_stop_seconds = float(ap_cfg.get("pink_dot_stop_seconds", self.pink_dot_stop_seconds))
                self.sim1_template_file = ap_cfg.get("sim1_template_file", self.sim1_template_file)
                self.sim2_template_file = ap_cfg.get("sim2_template_file", self.sim2_template_file)
                self.sim3_template_file = ap_cfg.get("sim3_template_file", self.sim3_template_file)
                self.sim_match_threshold = float(ap_cfg.get("sim_match_threshold", self.sim_match_threshold))
                self.sim_click_y_offset_px = int(ap_cfg.get("sim_click_y_offset_px", self.sim_click_y_offset_px))
                self.sim_click_x_offset_px = int(ap_cfg.get("sim_click_x_offset_px", self.sim_click_x_offset_px))
                self.sim_max_clicks = int(ap_cfg["sim_max_clicks"]) if "sim_max_clicks" in ap_cfg and ap_cfg["sim_max_clicks"] is not None else None
                self.start_at_pink_dot = int(ap_cfg.get("start_at_pink_dot", self.start_at_pink_dot))
                self.persistent_combat_action = str(ap_cfg.get("persistent_combat_action", self.persistent_combat_action)).lower().strip()
                self.persistent_combat_key = str(ap_cfg.get("persistent_combat_key", self.persistent_combat_key)).lower().strip()
                self.persistent_combat_interval = float(ap_cfg.get("persistent_combat_interval_seconds", ap_cfg.get("persistent_right_click_interval", self.persistent_combat_interval)))
                self.persistent_right_click_interval = self.persistent_combat_interval
                self.suppress_combat_during_approach = bool(ap_cfg.get("suppress_combat_during_approach", self.suppress_combat_during_approach))
                self.hideout_detection_mode = str(ap_cfg.get("hideout_detection_mode", self.hideout_detection_mode)).lower().strip()
                self.hideout_match_threshold = float(ap_cfg.get("hideout_match_threshold", self.hideout_match_threshold))
                self.map_device_match_threshold = float(ap_cfg.get("map_device_match_threshold", self.map_device_match_threshold))
                self.map_device_click_y_offset_px = int(ap_cfg.get("map_device_click_y_offset_px", self.map_device_click_y_offset_px))
                self.simulacrum_click_y_offset = float(ap_cfg.get("simulacrum_click_y_offset", self.simulacrum_click_y_offset))
                self.simulacrum_match_threshold = float(ap_cfg.get("simulacrum_match_threshold", self.simulacrum_match_threshold))
                self.simulacrum_node_v1_template_file = ap_cfg.get("simulacrum_node_v1_template_file", self.simulacrum_node_v1_template_file)
                self.simulacrum_node_v2_template_file = ap_cfg.get("simulacrum_node_v2_template_file", self.simulacrum_node_v2_template_file)
                self.simulacrum_node_v3_template_file = ap_cfg.get("simulacrum_node_v3_template_file", self.simulacrum_node_v3_template_file)
                self.loot1_template_file = ap_cfg.get("loot1_template_file", self.loot1_template_file)
                self.loot_match_threshold = float(ap_cfg.get("loot_match_threshold", self.loot_match_threshold))
                self.loot_pickup_wait_seconds = float(ap_cfg.get("loot_pickup_wait_seconds", self.loot_pickup_wait_seconds))
                self.loot_approach_wait_seconds = float(ap_cfg.get("loot_approach_wait_seconds", self.loot_approach_wait_seconds))
                self.loot_z_toggle_enabled = bool(ap_cfg.get("loot_z_toggle_enabled", self.loot_z_toggle_enabled))
                self.loot_z_toggle_delay_seconds = float(ap_cfg.get("loot_z_toggle_delay_seconds", self.loot_z_toggle_delay_seconds))
                self.max_loot_pickups = int(ap_cfg.get("max_loot_pickups", self.max_loot_pickups))
                self.wait_for_loot_confirmation = bool(ap_cfg.get("wait_for_loot_confirmation", self.wait_for_loot_confirmation))
                self.save_loot_debug_screenshots = bool(ap_cfg.get("save_loot_debug_screenshots", self.save_loot_debug_screenshots))
                self.save_pre_loot_screenshot = bool(ap_cfg.get("save_pre_loot_screenshot", ap_cfg.get("save_full_screen_before_pickup", self.save_pre_loot_screenshot)))
                self.loot_debug_dir = str(ap_cfg.get("loot_debug_dir", self.loot_debug_dir))
                self.zone_routines_file = ap_cfg.get("zone_routines_file", self.zone_routines_file)
                self.loot_filter_file = ap_cfg.get("loot_filter_file", "routines/loot_filter.json")
                self.minimap_early_exit_enabled = bool(ap_cfg.get("minimap_early_exit_enabled", self.minimap_early_exit_enabled))
                self.minimap_early_exit_min_seconds = float(ap_cfg.get("minimap_early_exit_min_seconds", self.minimap_early_exit_min_seconds))
                self.minimap_early_exit_check_interval = float(ap_cfg.get("minimap_early_exit_check_interval", self.minimap_early_exit_check_interval))
                self.minimap_early_exit_min_icons = int(ap_cfg.get("minimap_early_exit_min_icons", self.minimap_early_exit_min_icons))
                self.run_history_file = str(ap_cfg.get("run_history_file", self.run_history_file))
                self.run_summary_file = str(ap_cfg.get("run_summary_file", self.run_summary_file))
            except Exception:
                pass

        self.encounter_banner_img: Optional[np.ndarray] = None
        for candidate_path in [self.encounter_banner_file, "ui/encounter_banner.png", "templates/ui/encounter_banner.png"]:
            if os.path.exists(candidate_path):
                self.encounter_banner_img = cv2.imread(candidate_path)
                if self.encounter_banner_img is not None:
                    break

        self.sim_templates: Dict[str, Optional[np.ndarray]] = {"sim1": None, "sim2": None, "sim3": None}
        self._load_sim_templates()
        self._load_loot_template()
        self._load_delirium_and_portal_templates()
        self._load_stash_and_inventory_templates()
        self.loot_detector = LootDetector(getattr(self, "loot_filter_file", "routines/loot_filter.json"))
        if hasattr(self.loot_detector, "save_pre_loot_screenshot"):
            self.save_pre_loot_screenshot = bool(self.save_pre_loot_screenshot or self.loot_detector.save_pre_loot_screenshot)
        self.minimap_extractor = MinimapExtractor()
        self.load_zone_routines()

        if self.start_at_pink_dot > 0 and self.movement_path.is_configured:
            self.set_start_pink_dot(self.start_at_pink_dot)

        if keyboard:
            try:
                keyboard.add_hotkey("f4", self.toggle_pause, suppress=False)
                _log("[NAVIGATOR] Global F4 Pause/Resume listener active.")
            except Exception as e:
                _log(f"[NAVIGATOR] Warning: Could not register global F4 hotkey: {e}")
            try:
                keyboard.add_hotkey("f3", self.toggle_persistent_right_click, suppress=False)
                _log("[NAVIGATOR] Global F3 Right-Click Combat Attack Toggle active.")
            except Exception as e:
                _log(f"[NAVIGATOR] Warning: Could not register global F3 hotkey: {e}")

