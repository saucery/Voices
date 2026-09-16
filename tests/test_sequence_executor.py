"""
Unit Tests for SequenceExecutor Module
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sequence_executor import SequenceExecutor


def test_sequence_executor_initialization():
    executor = SequenceExecutor(monitor_idx=1)
    assert executor.monitor_idx == 1
    assert executor.detector is not None


def test_sequence_executor_step_execution():
    executor = SequenceExecutor(monitor_idx=1)

    steps = [
        {"action": "delay", "duration": 0.05, "description": "Short pause test"},
        {"action": "press_key", "key": "w", "duration": 0.05, "description": "Key press test"},
        {"action": "click", "button": "right", "description": "Right click test"},
    ]

    res = executor.execute_routine(steps)
    assert res is True


def test_sequence_routine_json():
    executor = SequenceExecutor(monitor_idx=1)
    routine_file = "routines/trail_of_suffering.json"

    assert os.path.exists(routine_file)
    # Execute with visual gate timeout fallback
    steps = [
        {"action": "delay", "duration": 0.05, "description": "Pause"},
        {"action": "press_key", "key": "w", "duration": 0.05, "description": "Key W"},
        {"action": "run_around", "duration": 0.1, "radius": 50, "description": "Run around test"},
    ]
    assert executor.execute_routine(steps) is True


def test_key_loop_and_disabled_step():
    executor = SequenceExecutor(monitor_idx=1)
    steps = [
        {"action": "click", "button": "left", "disabled": True, "description": "Disabled step"},
        {"action": "key_loop", "keys": ["s", "d"], "key_duration": 0.05, "total_duration": 0.1, "description": "Key loop test"},
    ]
    assert executor.execute_routine(steps) is True


def test_follow_edge_step():
    executor = SequenceExecutor(monitor_idx=1)
    # Mock capturer to return synthetic screen
    executor.capturer.capture = lambda: np.zeros((300, 300, 3), dtype=np.uint8)
    steps = [
        {"action": "follow_edge", "target_distance": 20.0, "duration": 0.1, "direction": "clockwise", "description": "Edge navigation test"},
    ]
    assert executor.execute_routine(steps) is True
