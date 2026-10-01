import os
import sys
import math
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath("."))

from src.navigation.human_mouse import (
    calculate_duration,
    generate_human_path,
    human_move_to,
)
from src.route_navigator import RouteNavigator


def test_calculate_duration_random_speed_variance():
    """Validates that duration calculation respects +/- 20% random speed variance."""
    assert calculate_duration(1.0) == 0.0

    dist = 600.0
    base_dur = 0.08 + ((dist / 1600.0) ** 0.75) * 0.35

    # Run 50 samples to test speed variance distribution
    durations = [calculate_duration(dist, speed_variance_pct=20.0) for _ in range(50)]

    min_expected = base_dur / 1.205  # +20% speed -> ~0.83x duration
    max_expected = base_dur / 0.795  # -20% speed -> ~1.25x duration

    for d in durations:
        assert min_expected <= d <= max_expected, f"Duration {d} outside expected [{min_expected}, {max_expected}]"

    # Confirm there is actual variation across samples (not static)
    assert max(durations) > min(durations), "Durations should have random variation"


def test_generate_human_path_endpoints_and_curvature():
    """Validates that generated Bezier path starts and ends accurately and forms an organic curve."""
    start = (100.0, 100.0)
    end = (800.0, 600.0)
    steps = 20

    path = generate_human_path(start, end, num_steps=steps)

    assert len(path) == steps
    assert path[-1] == (800, 600)

    # Check that path is not a rigid straight line (it forms a subtle arc)
    direct_slope = (end[1] - start[1]) / (end[0] - start[0])
    straight_line_pts = [(round(start[0] + (end[0] - start[0]) * (i / steps)),
                          round(start[1] + (end[1] - start[1]) * (i / steps)))
                         for i in range(1, steps + 1)]

    # At least one point should deviate from the exact linear chord
    has_arc_deviation = any(p != s for p, s in zip(path, straight_line_pts))
    assert has_arc_deviation, "Path should contain curved arc deviation"


def test_generate_human_path_clamping():
    """Validates that clamp_fn constraints are applied along the trajectory."""
    start = (100.0, 100.0)
    end = (400.0, 400.0)
    steps = 15

    # Restrict x to max 300
    def clamp_box(x, y):
        return min(300, x), y

    path = generate_human_path(start, end, num_steps=steps, clamp_fn=clamp_box)
    for px, py in path:
        assert px <= 300, f"Point {px} exceeded clamp boundary 300"


def test_human_move_to_respects_emergency_stop(monkeypatch):
    """Validates that human_move_to aborts immediately when stop_handler is triggered."""
    import src.navigation.human_mouse as hm

    # Simulate emergency stop active
    monkeypatch.setattr(hm, "_is_stopped", lambda: True)

    moves = []
    monkeypatch.setattr(hm, "send_cursor_pos", lambda x, y, sync_directinput=False: moves.append((x, y)))

    hm.human_move_to(500, 500, speed_variance_pct=20.0)

    # When stopped from step 0, loop aborts immediately
    assert len(moves) <= 1


def test_wasd_mover_delegates_to_human_mouse(monkeypatch):
    """Validates that RouteNavigator.move_mouse_inside_game routes to human_move_to when enabled."""
    nav = RouteNavigator()
    nav.human_mouse_enabled = True
    nav.mouse_speed_variation_pct = 20.0

    human_calls = []

    def mock_human_move(tx, ty, clamp_fn=None, speed_variance_pct=20.0, monitor_idx=0):
        human_calls.append((tx, ty, speed_variance_pct))
        return tx, ty

    # Unpatch the conftest default mock so we can test the real WasdMoverMixin method
    import src.navigation.wasd_mover as wm
    monkeypatch.setattr(wm, "human_move_to", mock_human_move)

    # Call real method directly from mixin
    res = wm.WasdMoverMixin.move_mouse_inside_game(nav, 650, 420)
    assert res == (650, 420)
    assert len(human_calls) == 1
    assert human_calls[0] == (650, 420, 20.0)
