"""
Unit Tests for Game Event Manager and Task Executor
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.event_engine import GameEventManager
from src.task_runner import TaskExecutor


def test_event_manager_emission():
    em = GameEventManager()
    events_fired = []

    def on_custom_event(payload):
        events_fired.append(payload)

    em.on("ON_TEST_EVENT", on_custom_event)
    em.emit("ON_TEST_EVENT", test_data="hello")

    assert len(events_fired) == 1
    assert events_fired[0]["test_data"] == "hello"
    assert events_fired[0]["event_name"] == "ON_TEST_EVENT"


def test_task_executor_integration():
    em = GameEventManager()
    executor = TaskExecutor(em)

    # Emit room enter event
    em.emit("ON_ROOM_ENTER", room_id="room_1", room_name="Room 1", global_position=(1000.0, 1000.0))

    assert executor.executed_tasks_count >= 1
