"""
Tests for Enemy-Reactive 'Q' Key Hold & Release in RouteNavigator
"""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.route_navigator import RouteNavigator
from src.movement_path import MovementPath
from src.enemy_detector import EnemyDetector


class MockCapturer:
    def __init__(self, frames=None):
        self.frames = list(frames) if frames else []
        self.call_count = 0
        self._sct = MagicMock()
        self._sct.monitors = [{"left": 0, "top": 0, "width": 800, "height": 600}]

    def capture(self, bbox=None):
        if not self.frames:
            # Blank dark canvas
            return np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)
        idx = min(self.call_count, len(self.frames) - 1)
        self.call_count += 1
        return self.frames[idx]


def test_hold_q_legacy_mode():
    """When hold_q_enemy_reactive_enabled is False, Q should be held for fixed duration."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.hold_q_enemy_reactive_enabled = False

    with patch("src.route_navigator.pydirectinput") as mock_input, \
         patch("src.route_navigator.window_focuser"):
        t0 = time.time()
        success = nav.execute_hold_q(zone_label="TEST", duration=0.08)
        elapsed = time.time() - t0

        assert success is True
        assert elapsed >= 0.08
        mock_input.keyDown.assert_any_call("q")
        mock_input.keyUp.assert_any_call("q")


def test_hold_q_reactive_no_enemies_skips_q():
    """In reactive mode, if no enemies appear within timeout, Q must NEVER be pressed."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.hold_q_enemy_reactive_enabled = True
    nav.enemy_detect_wait_timeout = 0.12  # short timeout for test

    # Capturer returns blank images with no enemies
    blank_frame = np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)
    nav.capturer = MockCapturer([blank_frame])

    with patch("src.route_navigator.pydirectinput") as mock_input, \
         patch("src.route_navigator.window_focuser"):
        success = nav.execute_hold_q(zone_label="TEST", duration=1.0)

        assert success is True
        # Q should never be pressed down because no enemies were detected
        assert not any(call[0] == ("q",) for call in mock_input.keyDown.call_args_list)
        assert not any(call[0] == ("q",) for call in mock_input.keyUp.call_args_list)


def test_hold_q_reactive_presses_and_releases_when_enemy_near():
    """In reactive mode, Q is pressed when enemies appear and released when an enemy gets near."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.hold_q_enemy_reactive_enabled = True
    nav.enemy_detect_wait_timeout = 1.0
    nav.enemy_near_distance_px = 150.0
    nav.enemy_hold_min_seconds = 0.05
    nav.enemy_hold_max_seconds = 2.0

    # Frame 1: Distant enemy at (100, 100), distance to center (400, 300) is ~360px
    frame_distant = np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)
    frame_distant[98:103, 80:120] = (25, 25, 160)  # red bar

    # Frame 2: Near enemy at (410, 310), distance to center (400, 300) is ~14px (<= 150px)
    frame_near = np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)
    frame_near[308:313, 390:430] = (25, 25, 160)  # red bar

    # Feed distant frame first, then near frame
    frames = [frame_distant, frame_distant, frame_near]
    nav.capturer = MockCapturer(frames)

    with patch("src.route_navigator.pydirectinput") as mock_input, \
         patch("src.route_navigator.window_focuser"):
        success = nav.execute_hold_q(zone_label="TEST", duration=5.0)

        assert success is True
        mock_input.keyDown.assert_any_call("q")
        mock_input.keyUp.assert_any_call("q")


def test_hold_q_step_override():
    """Step configuration with enemy_reactive=False overrides global reactive setting."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.hold_q_enemy_reactive_enabled = True  # Globally enabled

    with patch("src.route_navigator.pydirectinput") as mock_input, \
         patch("src.route_navigator.window_focuser"):
        # Step explicitly disables reactive mode
        step = {"enemy_reactive": False, "duration": 0.05}
        success = nav.execute_hold_q(zone_label="TEST", duration=0.05, step=step)

        assert success is True
        mock_input.keyDown.assert_any_call("q")
        mock_input.keyUp.assert_any_call("q")


def test_hold_q_reactive_safety_timeout():
    """If enemy stays far away, Q is safely released after max_hold_seconds."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.hold_q_enemy_reactive_enabled = True
    nav.enemy_detect_wait_timeout = 0.5
    nav.enemy_near_distance_px = 100.0
    nav.enemy_hold_max_seconds = 0.12  # short max hold for test

    # Frame with distant enemy only
    frame_distant = np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)
    frame_distant[98:103, 80:120] = (25, 25, 160)
    nav.capturer = MockCapturer([frame_distant])

    with patch("src.route_navigator.pydirectinput") as mock_input, \
         patch("src.route_navigator.window_focuser"):
        success = nav.execute_hold_q(zone_label="TEST", duration=5.0)

        assert success is True
        mock_input.keyDown.assert_any_call("q")
        mock_input.keyUp.assert_any_call("q")


def test_hold_q_reactive_respects_min_hold_duration():
    """Even if an enemy is immediately in melee range, Q must be held for at least min_hold_seconds."""
    nav = RouteNavigator(movement_path=MovementPath())
    nav.hold_q_enemy_reactive_enabled = True
    nav.enemy_detect_wait_timeout = 1.0
    nav.enemy_near_distance_px = 150.0
    nav.enemy_hold_min_seconds = 0.20  # hold for at least 0.2s
    nav.enemy_hold_max_seconds = 2.0

    # Enemy is immediately close at (410, 310), distance ~14px (already near!)
    frame_near = np.full((600, 800, 3), (30, 30, 30), dtype=np.uint8)
    frame_near[308:313, 390:430] = (25, 25, 160)
    nav.capturer = MockCapturer([frame_near])

    with patch("src.route_navigator.pydirectinput") as mock_input, \
         patch("src.route_navigator.window_focuser"):
        t0 = time.time()
        success = nav.execute_hold_q(zone_label="TEST", duration=5.0)
        elapsed = time.time() - t0

        assert success is True
        # Must have held for at least min_hold (0.2s)
        assert elapsed >= 0.20
        mock_input.keyDown.assert_any_call("q")
        mock_input.keyUp.assert_any_call("q")


