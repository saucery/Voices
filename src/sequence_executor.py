"""
Sequence Executor Module
Executes instruction-based macro movement sequences, visual detection gates,
mouse clicks, and room exploration routines.
"""

import json
import math
import os
import time
from typing import Dict, Any, List, Optional, Union

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    import pydirectinput
except ImportError:
    pydirectinput = None

from .screen_capturer import ScreenCapturer
from .encounter_detector import EncounterDetector
from .stop_handler import stop_handler
from .window_focus import window_focuser


class SequenceExecutor:
    """Executes step-by-step navigation, key pressing, visual detection, and movement routines."""

    def __init__(
        self,
        capturer: Optional[ScreenCapturer] = None,
        detector: Optional[EncounterDetector] = None,
        localizer: Optional[Any] = None,
        monitor_idx: int = 1,
    ):
        """
        Initialize SequenceExecutor.

        :param capturer: ScreenCapturer instance.
        :param detector: EncounterDetector instance.
        :param localizer: MapLocalizer instance (optional).
        :param monitor_idx: Monitor index for screen coordinates.
        """
        self.capturer = capturer or ScreenCapturer(monitor_idx=monitor_idx)
        self.detector = detector or EncounterDetector()
        self.localizer = localizer
        self.monitor_idx = monitor_idx

        # Configure PyAutoGUI / PyDirectInput safety settings
        if pyautogui:
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.05
        if pydirectinput:
            pydirectinput.FAILSAFE = False
            pydirectinput.PAUSE = 0.05

    def press_key(self, key: str, duration: float = 1.0):
        """
        Executes a key press or holds key for specified duration.
        - Movement keys ('w', 'a', 's', 'd'): Holds key continuously down for duration.
        - Skill/action keys ('q', 'e', 'r', '1'-'5', etc.): Sends clean DirectInput key press cycles.
        Responds instantly to F1 emergency stop.

        :param key: Key name (e.g. 'w', 'q', 'e', 'space').
        :param duration: Time in seconds to hold/repeat key.
        """
        if stop_handler.is_stopped():
            print("  [ABORTED] Emergency stop active.")
            return

        # Ensure game window maintains input focus before sending key events
        window_focuser.ensure_focused()

        key_lower = key.lower()
        is_movement = key_lower in ["w", "a", "s", "d", "up", "down", "left", "right"]

        print(f"  [ACTION] Executing key '{key}' ({'movement hold' if is_movement else 'skill activation'}) for {duration:.1f}s (Press F1 to STOP)...")
        start_time = time.time()

        if pydirectinput:
            if is_movement:
                # Movement key: hold down for duration
                pydirectinput.keyDown(key_lower)
                try:
                    while (time.time() - start_time) < duration:
                        if stop_handler.is_stopped():
                            print("  [STOP] F1 key pressed - aborting movement key!")
                            break
                        time.sleep(0.05)
                finally:
                    pydirectinput.keyUp(key_lower)
            else:
                # Skill/action key: send clean DirectInput key presses continuously over duration
                while (time.time() - start_time) < duration:
                    if stop_handler.is_stopped():
                        print("  [STOP] F1 key pressed - aborting skill key!")
                        break
                    pydirectinput.press(key_lower)
                    time.sleep(0.25)
        elif pyautogui:
            if is_movement:
                pyautogui.keyDown(key_lower)
                try:
                    while (time.time() - start_time) < duration:
                        if stop_handler.is_stopped():
                            break
                        time.sleep(0.05)
                finally:
                    pyautogui.keyUp(key_lower)
            else:
                while (time.time() - start_time) < duration:
                    if stop_handler.is_stopped():
                        break
                    pyautogui.press(key_lower)
                    time.sleep(0.25)
        else:
            print(f"  [SIMULATED] Key '{key}' executed for {duration:.1f}s")
            time.sleep(duration)

        if stop_handler.is_stopped():
            stop_handler.release_all_keys()

    def click(self, button: str = "right", x: Optional[int] = None, y: Optional[int] = None):
        """
        Performs exactly ONE single mouse click (left/right) at target coordinates or current mouse location,
        guaranteeing the mouse button is immediately released.

        :param button: 'left' or 'right'.
        :param x: Target screen X coordinate (optional).
        :param y: Target screen Y coordinate (optional).
        """
        if stop_handler.is_stopped():
            print("  [ABORTED] Emergency stop active.")
            return

        # Ensure game window maintains input focus before mouse click
        window_focuser.ensure_focused()

        # If no coordinates specified, target center of game window
        if x is None or y is None:
            game_center = window_focuser.get_game_center()
            if game_center:
                x, y = game_center

        button_str = button.lower()
        if x is not None and y is not None:
            print(f"  [ACTION] Executing single {button_str.upper()}-click at game coordinate ({x}, {y})...")
        else:
            print(f"  [ACTION] Executing single {button_str.upper()}-click at cursor...")

        if pydirectinput:
            if x is not None and y is not None:
                pydirectinput.moveTo(x, y)
                time.sleep(0.05)

            if button_str == "right":
                pydirectinput.rightClick()
                time.sleep(0.05)
                pydirectinput.mouseUp(button="right")
            else:
                pydirectinput.click()
                time.sleep(0.05)
                pydirectinput.mouseUp(button="left")
        elif pyautogui:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            pyautogui.click(button=button_str, clicks=1)
        else:
            print(f"  [SIMULATED] Single {button_str.upper()}-click at ({x}, {y})")

        # Explicit Win32 mouse release safety check (0x0010 = RIGHTUP, 0x0004 = LEFTUP)
        try:
            import ctypes
            user32 = ctypes.windll.user32
            flag = 0x0010 if button_str == "right" else 0x0004
            user32.mouse_event(flag, 0, 0, 0, 0)
        except Exception:
            pass

        # Debounce sleep to guarantee only 1 single click occurs
        time.sleep(0.15)

    def wait_for_detection(self, target_name: str = "TRAIL OF SUFFERING", timeout: float = 10.0, poll_interval: float = 0.3) -> bool:
        """
        Polls game screen until target encounter banner is visually detected or timeout occurs.

        :param target_name: Name/Title of target encounter banner.
        :param timeout: Max time to wait in seconds.
        :param poll_interval: Polling frequency in seconds.
        :return: True if detected, False if timed out or stopped.
        """
        print(f"  [VISUAL GATE] Waiting for visual detection: '{target_name}' (Timeout: {timeout}s | Press F1 to STOP)...")
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if stop_handler.is_stopped():
                print("  [ABORTED] F1 pressed during visual gate.")
                return False

            screenshot = self.capturer.capture()
            det_res = self.detector.detect(screenshot)

            if det_res.get("detected") and target_name.upper() in det_res.get("title", "").upper():
                elapsed = time.time() - start_time
                print(f"  [VISUAL GATE PASSED] Found '{target_name}' in {elapsed:.2f}s!")
                return True

            time.sleep(poll_interval)

        print(f"  [VISUAL GATE TIMEOUT] '{target_name}' not detected within {timeout}s.")
        return False

    def run_around(self, duration: float = 5.0, radius: int = 180, num_points: int = 8):
        """
        Executes a smooth circular movement pattern around screen center to 'run around' the room.

        :param duration: Total duration of run-around routine in seconds.
        :param radius: Pixel distance from screen center for movement click points.
        :param num_points: Number of waypoint directions around the circle.
        """
        if stop_handler.is_stopped():
            return

        print(f"  [ACTION] Running around room for {duration:.1f}s (Radius: {radius}px | Press F1 to STOP)...")
        
        monitors = self.capturer.list_monitors()
        target_mon = next((m for m in monitors if m["index"] == self.monitor_idx), monitors[0])
        screen_cx = target_mon["left"] + target_mon["width"] // 2
        screen_cy = target_mon["top"] + target_mon["height"] // 2

        start_time = time.time()
        step_idx = 0

        while (time.time() - start_time) < duration:
            if stop_handler.is_stopped():
                print("  [STOP] F1 key pressed - aborting room movement!")
                break

            angle = (2.0 * math.pi / num_points) * (step_idx % num_points)
            click_x = int(screen_cx + radius * math.cos(angle))
            click_y = int(screen_cy + radius * math.sin(angle))

            self.click(button="right", x=click_x, y=click_y)
            time.sleep(0.4)
            step_idx += 1

        print(f"  [ACTION COMPLETED] Finished running around room ({step_idx} waypoints).")

    def key_loop(self, keys: Optional[List[str]] = None, key_duration: float = 2.0, total_duration: float = 50.0):
        """
        Loops through keyboard movement keys (e.g. S, D, W, A), holding each key for key_duration seconds
        for total_duration seconds.

        :param keys: List of keys to press sequentially (default: ['s', 'd', 'w', 'a']).
        :param key_duration: Duration in seconds to hold each key (default: 2.0s).
        :param total_duration: Total loop duration in seconds (default: 50.0s).
        """
        if stop_handler.is_stopped():
            return

        key_list = keys or ["s", "d", "w", "a"]
        print(f"  [ACTION] Running key loop {key_list} ({key_duration:.1f}s each) for total {total_duration:.1f}s (Press F1 to STOP)...")

        start_time = time.time()
        key_idx = 0

        while (time.time() - start_time) < total_duration:
            if stop_handler.is_stopped():
                print("  [STOP] F1 key pressed - aborting key loop!")
                break

            current_key = key_list[key_idx % len(key_list)]
            remaining = total_duration - (time.time() - start_time)
            press_dur = min(key_duration, remaining)

            if press_dur > 0.05:
                self.press_key(current_key, press_dur)

            key_idx += 1

        print(f"  [ACTION COMPLETED] Finished key loop sequence ({key_idx} key steps executed).")

    def follow_edge(self, target_distance: float = 20.0, duration: float = 40.0, direction: str = "clockwise"):
        """
        Navigates character around the full map perimeter, sweeping systematically from edge to edge
        (Top edge: Left -> Right, Right edge: Top -> Bottom, Bottom edge: Right -> Left, Left edge: Bottom -> Top),
        maintaining a safety distance of target_distance (20px) from map boundaries.
        Automatically detects if character is stuck (stationary for >= 1.0s) and advances to the next perimeter edge.

        :param target_distance: Target distance from map edge in pixels (default: 20.0px).
        :param duration: Total navigation duration in seconds (default: 40.0s).
        :param direction: Initial traversal direction ('clockwise' or 'counter_clockwise').
        """
        if stop_handler.is_stopped():
            return

        if self.localizer:
            localizer = self.localizer
        else:
            from .map_localizer import MapLocalizer
            localizer = MapLocalizer()

        current_direction = direction.lower()
        print(f"  [EDGE NAVIGATOR] Sweeping map perimeter keeping {target_distance:.0f}px edge distance ({current_direction}) for {duration:.1f}s (Press F1 to STOP)...")

        start_time = time.time()
        step_count = 0
        last_pos: Optional[Tuple[int, int]] = None
        last_pos_time: float = time.time()
        last_attempted_key: str = "d"

        # Perimeter state machine tracking
        clockwise_edges = ["TOP_EDGE", "RIGHT_EDGE", "BOTTOM_EDGE", "LEFT_EDGE"]
        ccw_edges = ["TOP_EDGE", "LEFT_EDGE", "BOTTOM_EDGE", "RIGHT_EDGE"]
        edge_sequence = clockwise_edges if current_direction == "clockwise" else ccw_edges
        current_edge_state: Optional[str] = None
        trajectory_history: List[Dict[str, Any]] = []

        while (time.time() - start_time) < duration:
            if stop_handler.is_stopped():
                print("  [STOP] F1 key pressed - aborting edge navigation!")
                break

            screenshot = self.capturer.capture()
            loc_res = localizer.localize_player(screenshot)

            if loc_res.get("located"):
                px, py = loc_res["player_position"]
                w, h = loc_res.get("map_width", 273), loc_res.get("map_height", 270)
                dists = loc_res.get("edge_distances", {})

                # Define valid map interior boundaries (Use painted red zone if detected)
                red_bounds = loc_res.get("red_zone_bounds")
                if red_bounds:
                    min_x, max_x, min_y, max_y = red_bounds
                else:
                    min_x = target_distance
                    max_x = max(min_x + 50.0, w - target_distance)
                    min_y = target_distance
                    max_y = max(min_y + 50.0, h - target_distance)

                # Initialize edge state on first located frame based on Red Zone boundaries
                if current_edge_state is None:
                    edges_dist = {
                        "TOP_EDGE": abs(py - min_y),
                        "RIGHT_EDGE": abs(px - max_x),
                        "BOTTOM_EDGE": abs(py - max_y),
                        "LEFT_EDGE": abs(px - min_x),
                    }
                    current_edge_state = min(edges_dist, key=edges_dist.get)

                # Stuck Detection: Check if player has moved >= 2px since last check
                if last_pos is not None:
                    dist_moved = math.hypot(px - last_pos[0], py - last_pos[1])
                    if dist_moved >= 2.0:
                        last_pos = (px, py)
                        last_pos_time = time.time()
                    else:
                        stuck_dur = time.time() - last_pos_time
                        if stuck_dur >= 1.0:
                            # Advance to next perimeter edge state
                            curr_idx = edge_sequence.index(current_edge_state) if current_edge_state in edge_sequence else 0
                            current_edge_state = edge_sequence[(curr_idx + 1) % len(edge_sequence)]

                            # Calculate opposite recovery key to dislodge from obstacle
                            opposites = {"w": "s", "s": "w", "a": "d", "d": "a"}
                            recovery_key = opposites.get(last_attempted_key.lower(), "a")

                            print(f"\n  [STUCK DETECTED] Character stationary for {stuck_dur:.1f}s at ({px:.0f}, {py:.0f})!")
                            print(f"  [RECOVERY] Advancing to next edge ({current_edge_state}) & Executing Key '{recovery_key.upper()}' for 0.6s...")

                            self.press_key(recovery_key, duration=0.6)
                            last_pos = (px, py)
                            last_pos_time = time.time()
                            last_attempted_key = recovery_key
                            step_count += 1
                            continue
                else:
                    last_pos = (px, py)
                    last_pos_time = time.time()

                # HARD CONTAINMENT: If physically outside painted red zone bounds, steer back into zone
                if px > max_x + 8:
                    move_key = "a"
                elif px < min_x - 8:
                    move_key = "d"
                elif py > max_y + 8:
                    move_key = "w"
                elif py < min_y - 8:
                    move_key = "s"
                else:
                    # INSIDE ZONE: Continuous corner-to-corner perimeter sweep along Red Zone edges
                    if current_edge_state == "TOP_EDGE":
                        if current_direction == "counter_clockwise":
                            move_key = "a"
                            if px <= min_x + 5:
                                current_edge_state = "LEFT_EDGE"
                                move_key = "s"
                        else:
                            move_key = "d"
                            if px >= max_x - 5:
                                current_edge_state = "RIGHT_EDGE"
                                move_key = "s"

                    elif current_edge_state == "LEFT_EDGE":
                        if current_direction == "counter_clockwise":
                            move_key = "s"
                            if py >= max_y - 5:
                                current_edge_state = "BOTTOM_EDGE"
                                move_key = "d"
                        else:
                            move_key = "w"
                            if py <= min_y + 5:
                                current_edge_state = "TOP_EDGE"
                                move_key = "d"

                    elif current_edge_state == "BOTTOM_EDGE":
                        if current_direction == "counter_clockwise":
                            move_key = "d"
                            if px >= max_x - 5:
                                current_edge_state = "RIGHT_EDGE"
                                move_key = "w"
                        else:
                            move_key = "a"
                            if px <= min_x + 5:
                                current_edge_state = "LEFT_EDGE"
                                move_key = "w"

                    else:  # RIGHT_EDGE
                        if current_direction == "counter_clockwise":
                            move_key = "w"
                            if py <= min_y + 5:
                                current_edge_state = "TOP_EDGE"
                                move_key = "a"
                        else:
                            move_key = "s"
                            if py >= max_y - 5:
                                current_edge_state = "BOTTOM_EDGE"
                                move_key = "a"

                print(f"    [PRE-MOVE POS: ({px:3.0f}, {py:3.0f}) | Edge: {current_edge_state:11s}] -> Key: {move_key.upper()}", end="\r", flush=True)

                self.press_key(move_key, duration=0.25)
                last_attempted_key = move_key

                # Empirical Post-Movement Position Verification
                screenshot_after = self.capturer.capture()
                loc_after = localizer.localize_player(screenshot_after)

                if loc_after.get("located"):
                    px_after, py_after = loc_after["player_position"]
                    conf_after = loc_after.get("confidence", 0.0)
                    trajectory_history.append({
                        "pos": (px_after, py_after),
                        "key": move_key,
                        "edge": current_edge_state,
                        "conf": conf_after
                    })
                    localizer.render_trajectory_map(trajectory_history, "debug_output/movement_trajectory.png")
                else:
                    trajectory_history.append({
                        "pos": (px, py),
                        "key": move_key,
                        "edge": current_edge_state,
                        "conf": loc_res.get("confidence", 0.0)
                    })
            else:
                fallback_key = ["d", "s", "a", "w"][step_count % 4]
                self.press_key(fallback_key, duration=0.25)
                last_attempted_key = fallback_key

            step_count += 1
            time.sleep(0.05)

        # Final render of complete trajectory map
        if trajectory_history:
            out_img = localizer.render_trajectory_map(trajectory_history, "debug_output/movement_trajectory.png")
            print(f"\n  [VISUALIZATION] Saved full movement trajectory map to: {out_img}")

        print(f"  [ACTION COMPLETED] Finished edge navigation ({step_count} adjustments made).")

    def follow_route(
        self,
        route_path: Optional[str] = None,
        map_path: Optional[str] = None,
        config_path: str = "config.json",
        arrival_threshold: float = 20.0,
        step_duration: float = 0.30,
        max_duration: float = 180.0,
    ) -> bool:
        """
        Navigates character automatically along waypoints from MovementPath (paths/movement_route.json or route.png)
        using closed-loop WASD movement, continuously tracking player position on the map layout.

        :param route_path: Path to route JSON or painted PNG file.
        :param map_path: Path to reference map layout PNG.
        :param config_path: Path to config.json.
        :param arrival_threshold: Distance in px to consider waypoint reached.
        :param step_duration: Time in seconds to hold WASD key(s) per step.
        :param max_duration: Max time in seconds before aborting.
        :return: True if destination reached.
        """
        if stop_handler.is_stopped():
            return False

        from .movement_path import MovementPath
        from .room_classifier import RoomClassifier
        from .route_navigator import RouteNavigator

        stop_handler.reset()
        window_focuser.focus_game_window()

        path_mgr = MovementPath(movement_file_path=route_path, config_path=config_path)
        if not path_mgr.is_configured:
            print("[NAVIGATOR] Error: No waypoints configured in route.")
            return False

        classifier = RoomClassifier(config_path_or_dict=config_path, map_layout_path=map_path)

        print("=" * 70)
        print(" AUTONOMOUS ROUTE NAVIGATION (WASD)")
        print(f" Route:           {path_mgr.route_name} ({len(path_mgr.waypoints)} waypoints)")
        print(f" Target Display:  Monitor {self.monitor_idx} (Path of Exile 2)")
        print(f" Arrival Radius:  {arrival_threshold:.0f} px")
        print(f" Step Duration:   {step_duration:.2f} s")
        print(" [SAFETY] Press 'F1' anytime to immediately STOP.")
        print("=" * 70)
        print(f"Starting in 2.0s... Click into Path of Exile 2 window on Monitor {self.monitor_idx} now!\n")
        time.sleep(2.0)

        start_time = time.time()
        step_count = 0

        while (time.time() - start_time) < max_duration:
            if stop_handler.is_stopped():
                print("\n[STOP] F1 key pressed - aborting route navigation!")
                stop_handler.release_all_keys()
                return False

            # Capture current frame and localize player
            screenshot = self.capturer.capture()
            res = classifier.classify(screenshot)
            player_pos = res.get("character_position")

            target = path_mgr.get_current_target()
            if target is None:
                print("\n\n[NAVIGATOR] >>> ALL WAYPOINTS COMPLETED! Reached destination.")
                stop_handler.release_all_keys()
                return True

            if player_pos is None:
                print("  [NAVIGATOR] Searching for player position...", end="\r", flush=True)
                time.sleep(0.08)
                continue

            dist = math.hypot(target["x"] - player_pos[0], target["y"] - player_pos[1])

            # Waypoint Arrival Check
            if dist <= arrival_threshold:
                print(f"\n  [WAYPOINT REACHED] Reached #{target['index']} ({target.get('name')}) at ({player_pos[0]:.0f}, {player_pos[1]:.0f})!")
                next_target = path_mgr.advance()
                if next_target is None:
                    print("\n[NAVIGATOR] >>> FINISH REACHED! Destination arrived.")
                    stop_handler.release_all_keys()
                    return True
                target = next_target
                dist = math.hypot(target["x"] - player_pos[0], target["y"] - player_pos[1])

            # Calculate 8-directional WASD keys
            keys = RouteNavigator.compute_wasd_keys(player_pos, (target["x"], target["y"]))
            if not keys:
                time.sleep(0.05)
                continue

            key_str = "+".join(k.upper() for k in keys)
            print(f"  [STEP #{step_count+1:3d}] Pos: ({player_pos[0]:3.0f}, {player_pos[1]:3.0f}) -> WP #{target['index']} ({target['x']:3.0f}, {target['y']:3.0f}) [Dist: {dist:3.0f}px] -> Holding [{key_str:3s}] ({step_duration:.2f}s)...", end="\r", flush=True)

            # Hold keys for step_duration
            if pydirectinput:
                for k in keys:
                    pydirectinput.keyDown(k)
                time.sleep(step_duration)
                for k in keys:
                    pydirectinput.keyUp(k)
            elif pyautogui:
                for k in keys:
                    pyautogui.keyDown(k)
                time.sleep(step_duration)
                for k in keys:
                    pyautogui.keyUp(k)

            step_count += 1
            time.sleep(0.03)

        stop_handler.release_all_keys()
        print(f"\n[NAVIGATOR] Max duration ({max_duration}s) reached.")
        return True

    def execute_step(self, step: Dict[str, Any]) -> bool:
        """
        Executes a single step dictionary.

        :param step: Step configuration dictionary.
        :return: Success boolean.
        """
        if stop_handler.is_stopped():
            return False

        action = step.get("action", "").lower()
        desc = step.get("description", action)

        if step.get("disabled", False):
            print(f"\n>>> Step [SKIPPED (DISABLED)]: {desc}")
            return True

        print(f"\n>>> Step: {desc}")

        if action == "press_key":
            key = step.get("key", "w")
            dur = float(step.get("duration", 1.0))
            self.press_key(key, dur)
        elif action == "click":
            btn = step.get("button", "right")
            x = step.get("x")
            y = step.get("y")
            self.click(btn, x, y)
        elif action == "wait_for_detection":
            target = step.get("target", "TRAIL OF SUFFERING")
            to = float(step.get("timeout", 10.0))
            return self.wait_for_detection(target, timeout=to)
        elif action == "run_around":
            dur = float(step.get("duration", 5.0))
            rad = int(step.get("radius", 180))
            self.run_around(dur, rad)
        elif action == "key_loop" or action == "wasd_loop":
            keys = step.get("keys", ["s", "d", "w", "a"])
            k_dur = float(step.get("key_duration", 2.0))
            t_dur = float(step.get("total_duration", 50.0))
            self.key_loop(keys, k_dur, t_dur)
        elif action == "follow_edge" or action == "wall_hug":
            t_dist = float(step.get("target_distance", 20.0))
            dur = float(step.get("duration", 40.0))
            dir_str = step.get("direction", "clockwise")
            self.follow_edge(target_distance=t_dist, duration=dur, direction=dir_str)
        elif action == "delay" or action == "sleep":
            dur = float(step.get("duration", 1.0))
            print(f"  [ACTION] Pausing for {dur:.1f}s...")
            start_pause = time.time()
            while (time.time() - start_pause) < dur:
                if stop_handler.is_stopped():
                    break
                time.sleep(0.05)
        else:
            print(f"  [WARNING] Unknown action '{action}'. Skipping.")
        
        return not stop_handler.is_stopped()

    def execute_routine(self, routine: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """
        Executes a full sequence routine from a JSON file path, dict, or list of step dicts.

        :param routine: Path to JSON routine file, routine dict, or list of steps.
        :return: True if all steps completed successfully.
        """
        stop_handler.reset()
        window_focuser.focus_game_window()

        if isinstance(routine, str):
            if not os.path.exists(routine):
                print(f"Error: Routine file not found: {routine}")
                return False
            with open(routine, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps = data.get("steps", [])
            title = data.get("name", os.path.basename(routine))
        elif isinstance(routine, dict):
            steps = routine.get("steps", [])
            title = routine.get("name", "Custom Routine")
        elif isinstance(routine, list):
            steps = routine
            title = "Custom Steps Routine"
        else:
            print(f"Error: Invalid routine format: {type(routine)}")
            return False

        print("=" * 70)
        print(f" EXECUTING ROUTINE: {title} ({len(steps)} steps)")
        print(" [SAFETY] Press 'F1' anytime to immediately STOP and regain full control.")
        print("=" * 70)

        for idx, step in enumerate(steps, 1):
            if stop_handler.is_stopped():
                print("\n[STOP] Routine halted by user (F1). Control returned.")
                return False

            print(f"\n[{idx}/{len(steps)}]", end=" ")
            success = self.execute_step(step)

            if stop_handler.is_stopped():
                print("\n[STOP] Routine halted by user (F1). Control returned.")
                return False

            if not success and step.get("fail_on_timeout", False):
                print(f"Routine aborted at step {idx} due to failure.")
                return False

        print("\n" + "=" * 70)
        print(f" ROUTINE COMPLETED SUCCESSFULLY: {title}")
        print("=" * 70)
        return True
