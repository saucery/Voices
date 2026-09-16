"""
Unit Tests for Emergency Stop & Hotkey Handler Module
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stop_handler import EmergencyStopHandler, stop_handler


def test_stop_handler_initialization():
    handler = EmergencyStopHandler(stop_key="f1")
    assert handler.stop_key == "f1"
    assert handler.is_stopped() is False


def test_stop_handler_trigger():
    handler = EmergencyStopHandler(stop_key="f1")
    assert handler.is_stopped() is False

    handler.trigger_stop()
    assert handler.is_stopped() is True

    handler.reset()
    assert handler.is_stopped() is False
