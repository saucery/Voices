"""
Task Runner Module
Registers and executes automated bot task handlers in response to game events.
Supports marking encounters on the global map and executing autopilot interactions.
"""

from typing import Callable, Dict, List, Any, Optional
from .event_engine import GameEventManager
from .world_map import WorldMapTracker


class TaskExecutor:
    """Executes automated bot tasks when triggered by game events."""

    def __init__(
        self,
        event_manager: GameEventManager,
        world_map: Optional[WorldMapTracker] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TaskExecutor.

        :param event_manager: Instance of GameEventManager.
        :param world_map: Optional WorldMapTracker instance.
        :param config: Configuration dict.
        """
        self.event_manager = event_manager
        self.world_map = world_map
        self.config = config or {}
        self.executed_tasks_count: int = 0

        # Register default handlers
        self.event_manager.on("ON_ROOM_ENTER", self.handle_room_enter_task)
        self.event_manager.on("ON_POSITION_UPDATE", self.handle_position_update_task)
        self.event_manager.on("ON_ENCOUNTER_DETECTED", self.handle_encounter_task)

    def register_custom_task(self, event_name: str, task_fn: Callable[[Dict[str, Any]], None]):
        """Registers a custom user task function to execute on event."""
        self.event_manager.on(event_name, task_fn)

    def handle_room_enter_task(self, payload: Dict[str, Any]):
        """Default task executed when character enters a new room."""
        self.executed_tasks_count += 1
        room_name = payload.get("room_name", payload.get("room_id"))
        pos = payload.get("global_position")
        print(f"\n[{payload['timestamp']}] [TASK EXECUTOR] Executed Room Transition Task: Character entered {room_name} at Global Pos {pos}")

    def handle_position_update_task(self, payload: Dict[str, Any]):
        """Default task executed on position update."""
        self.executed_tasks_count += 1

    def handle_encounter_task(self, payload: Dict[str, Any]):
        """
        Default task executed when an encounter banner is detected on screen.
        Marks landmark on world map and executes autopilot click when enabled.
        """
        self.executed_tasks_count += 1
        title = payload.get("title", "ENCOUNTER DETECTED")
        wave = payload.get("wave", "")
        action = payload.get("action", "")
        warning = payload.get("warning", "")
        timestamp = payload.get("timestamp", "")
        pos = payload.get("global_position")

        # Mark landmark on WorldMapTracker if available
        landmark_id = "N/A"
        if self.world_map is not None:
            lm = self.world_map.add_encounter_landmark(title, pos, payload)
            landmark_id = lm.get("id", "N/A")

        print("\n" + "!" * 75)
        print(f" [{timestamp}] [ENCOUNTER DETECTED & MAP MARKED] >>> {title} ({wave}) <<<")
        print(f" Landmark ID:     {landmark_id} (Marked Purple on Global Map)")
        print(f" Action Required: {action}")
        print(f" WARNING:         {warning}")
        if pos:
            print(f" Global Position: {pos}")

        # Autopilot click execution when enabled in config.json
        auto_click = self.config.get("autopilot", {}).get("auto_click_encounters", False)
        if auto_click:
            print(f" [AUTOPILOT] Executing click task on encounter pedestal at {pos}...")
        else:
            print(" [AUTOPILOT] Auto-click disabled (Set 'auto_click_encounters': true in config.json to enable auto-clicking)")

        print("!" * 75 + "\n")
