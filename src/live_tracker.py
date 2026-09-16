"""
Live Room Tracker & Event Engine Module
Monitors game screen periodically, tracking global map reconstruction, room locations,
and dispatching event tasks.
"""

import time
import threading
from datetime import datetime
from typing import Callable, Optional, Dict, Any

from .screen_capturer import ScreenCapturer
from .room_classifier import RoomClassifier
from .world_map import WorldMapTracker
from .map_localizer import MapLocalizer
from .event_engine import GameEventManager
from .task_runner import TaskExecutor
from .stop_handler import stop_handler


class LiveRoomTracker:
    """Continuously monitors game screen, updates global map, and executes event tasks."""

    def __init__(
        self,
        classifier: RoomClassifier,
        capturer: Optional[ScreenCapturer] = None,
        world_map: Optional[WorldMapTracker] = None,
        localizer: Optional[MapLocalizer] = None,
        event_manager: Optional[GameEventManager] = None,
        task_executor: Optional[TaskExecutor] = None,
        interval: float = 1.0,
        on_room_change: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        on_tick: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize LiveRoomTracker with Event-Driven Architecture and Reference Map Localization.
        """
        self.classifier = classifier
        self.capturer = capturer or ScreenCapturer()
        self.world_map = world_map or WorldMapTracker()
        self.localizer = localizer or MapLocalizer()
        self.event_manager = event_manager or GameEventManager()
        self.task_executor = task_executor or TaskExecutor(
            self.event_manager, self.world_map, getattr(self.classifier, "config", {})
        )

        self.interval = max(0.1, interval)
        self.on_room_change = on_room_change
        self.on_tick = on_tick

        self.current_room_id: Optional[str] = None
        self.current_result: Optional[Dict[str, Any]] = None
        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None

    def tick(self) -> Dict[str, Any]:
        """Performs a single screenshot capture, room classification, map update, player localization, and event task dispatch."""
        screenshot = self.capturer.capture()
        result = self.classifier.classify(screenshot)
        timestamp = datetime.now().strftime("%H:%M:%S")
        result["timestamp"] = timestamp

        # Update global world map canvas
        map_res = self.world_map.update(screenshot)
        result["world_map"] = map_res

        # Localize exact player position on Full Reference Map
        loc_res = self.localizer.localize_player(screenshot)
        result["localization"] = loc_res
        if loc_res.get("located"):
            self.localizer.render_player_location(loc_res, "debug_output/player_map_location.png")

        # Process frame state and dispatch game events to registered tasks
        self.event_manager.process_frame(screenshot, result, map_res)

        new_room_id = result["room_id"] if result["recognized"] else "UNKNOWN"

        # State change detection
        if new_room_id != self.current_room_id:
            prev_result = self.current_result
            self.current_room_id = new_room_id
            self.current_result = result

            if self.on_room_change:
                self.on_room_change(prev_result, result)

        if self.on_tick:
            self.on_tick(result)

        return result

    def start(self, max_ticks: Optional[int] = None):
        """Runs the monitoring loop synchronously."""
        self.is_running = True
        stop_handler.reset()
        tick_count = 0

        try:
            while self.is_running and not stop_handler.is_stopped():
                self.tick()
                tick_count += 1

                if max_ticks and tick_count >= max_ticks:
                    break

                start_sleep = time.time()
                while (time.time() - start_sleep) < self.interval:
                    if stop_handler.is_stopped():
                        break
                    time.sleep(0.05)
        finally:
            self.is_running = False

    def start_async(self):
        """Starts monitoring loop in a background daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self.start, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the monitoring loop."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
