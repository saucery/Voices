"""
Game Event Engine Module
Emits structured game events (room changes, position updates, encounter detection, map exploration)
to trigger automated bot tasks.
"""

from datetime import datetime
from typing import Callable, Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

from .encounter_detector import EncounterDetector


class GameEventManager:
    """Manages game event listeners, event detection, and task dispatching."""

    def __init__(self, encounter_detector: Optional[EncounterDetector] = None):
        """
        Initialize GameEventManager.

        :param encounter_detector: Optional EncounterDetector instance.
        """
        self._listeners: Dict[str, List[Callable[..., None]]] = {}
        self.encounter_detector = encounter_detector or EncounterDetector()

        self.last_room_id: Optional[str] = None
        self.last_global_pos: Optional[Tuple[float, float]] = None
        self.last_encounter_detected: bool = False

    def on(self, event_name: str, handler: Callable[..., None]):
        """Registers an event listener callback."""
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(handler)

    def emit(self, event_name: str, **payload):
        """Emits a game event to all registered listeners."""
        payload["timestamp"] = datetime.now().strftime("%H:%M:%S")
        payload["event_name"] = event_name

        if event_name in self._listeners:
            for handler in self._listeners[event_name]:
                try:
                    handler(payload)
                except Exception as e:
                    print(f"Error in event handler for {event_name}: {e}")

    def process_frame(
        self,
        screenshot: np.ndarray,
        classifier_result: Dict[str, Any],
        map_result: Dict[str, Any],
    ):
        """
        Analyzes live frame state and fires appropriate game events.

        :param screenshot: Input full screenshot.
        :param classifier_result: RoomClassifier result dict.
        :param map_result: WorldMapTracker update result dict.
        """
        current_room = classifier_result.get("room_id")
        current_pos = map_result.get("global_position")

        # 1. Room Change Event
        if current_room != self.last_room_id and current_room is not None:
            prev_room = self.last_room_id
            self.last_room_id = current_room
            self.emit(
                "ON_ROOM_ENTER",
                room_id=current_room,
                room_name=classifier_result.get("room_name"),
                prev_room=prev_room,
                global_position=current_pos,
                confidence=classifier_result.get("confidence"),
            )

        # 2. Position Update Event
        if current_pos != self.last_global_pos and current_pos is not None:
            self.last_global_pos = current_pos
            self.emit(
                "ON_POSITION_UPDATE",
                global_position=current_pos,
                room_id=current_room,
                displacement=map_result.get("displacement_delta"),
            )

        # 3. Map Exploration Milestone Event
        explored_pct = map_result.get("explored_pct", 0.0)
        if explored_pct > 0 and int(explored_pct) % 5 == 0:
            self.emit(
                "ON_MAP_DISCOVERED",
                explored_pct=explored_pct,
                explored_pixels=map_result.get("explored_pixels"),
            )

        # 4. Encounter UI Banner Detection Event
        enc_res = self.encounter_detector.detect(screenshot)
        if enc_res["detected"] and not self.last_encounter_detected:
            self.last_encounter_detected = True
            self.emit(
                "ON_ENCOUNTER_DETECTED",
                title=enc_res["title"],
                wave=enc_res["wave"],
                action=enc_res["action"],
                warning=enc_res["warning"],
                confidence=enc_res["confidence"],
                global_position=current_pos,
            )
        elif not enc_res["detected"]:
            self.last_encounter_detected = False
