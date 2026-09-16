"""
Unit Tests for Integrated CLI Navigation and Monitoring
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.player_tracker_visualizer import PlayerTrackerVisualizer
from src.room_classifier import RoomClassifier
from src.map_localizer import MapLocalizer
from src.movement_path import MovementPath


def test_visualizer_has_route_navigator():
    config_path = "config.json"
    classifier = RoomClassifier(config_path)
    localizer = MapLocalizer(config_path=config_path)
    movement_path = MovementPath(config_path=config_path)

    visualizer = PlayerTrackerVisualizer(
        classifier=classifier,
        localizer=localizer,
        movement_path=movement_path,
        monitor_idx=2,
    )

    assert visualizer.navigator is not None
    assert visualizer.navigator.monitor_idx == 2
    assert visualizer.navigator.movement_path.is_configured is True
    assert len(visualizer.navigator.movement_path.waypoints) >= 5
    assert visualizer.navigator.is_active is False


def test_visualizer_autonavigation_toggle():
    config_path = "config.json"
    movement_path = MovementPath(config_path=config_path)

    visualizer = PlayerTrackerVisualizer(
        movement_path=movement_path,
        monitor_idx=2,
    )

    # Toggle On
    active = visualizer.navigator.toggle(monitor_idx=2)
    assert active is True
    assert visualizer.navigator.is_active is True

    # Toggle Off
    active_off = visualizer.navigator.toggle(monitor_idx=2)
    assert active_off is False
    assert visualizer.navigator.is_active is False
