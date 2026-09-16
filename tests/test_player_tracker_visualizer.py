"""
Unit Tests for PlayerTrackerVisualizer Module
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.player_tracker_visualizer import PlayerTrackerVisualizer
from src.map_localizer import MapLocalizer


def test_visualizer_initialization():
    ref_map = "templates/full_map_reference.png"
    localizer = MapLocalizer(reference_map_path=ref_map)
    visualizer = PlayerTrackerVisualizer(localizer=localizer, monitor_idx=1)

    assert visualizer.localizer is not None
    assert visualizer.monitor_idx == 1
    assert visualizer.is_paused is False
    assert visualizer.show_overlay is False
    assert visualizer.show_edges is False
    assert visualizer.show_trajectory is True


def test_visualizer_update_frame_synthetic():
    ref_map = "templates/full_map_reference.png"
    if not os.path.exists(ref_map):
        pytest.skip("Reference map missing")

    localizer = MapLocalizer(reference_map_path=ref_map)
    visualizer = PlayerTrackerVisualizer(localizer=localizer, monitor_idx=1)

    # Use a 280x288 synthetic test image
    test_crop = np.zeros((280, 288, 3), dtype=np.uint8)
    dashboard, loc_res = visualizer.update_frame(test_crop)

    assert dashboard is not None
    assert dashboard.shape == (700, 1080, 3)
    assert isinstance(loc_res, dict)
    assert "minimap_player" in loc_res
    assert "reference_map" in loc_res


def test_visualizer_toggles_and_snapshot(tmp_path):
    ref_map = "templates/full_map_reference.png"
    if not os.path.exists(ref_map):
        pytest.skip("Reference map missing")

    localizer = MapLocalizer(reference_map_path=ref_map)
    visualizer = PlayerTrackerVisualizer(localizer=localizer, monitor_idx=1)

    visualizer.show_overlay = True
    visualizer.show_edges = True
    visualizer.show_trajectory = False

    test_crop = np.zeros((280, 288, 3), dtype=np.uint8)
    dashboard, _ = visualizer.update_frame(test_crop)

    assert dashboard.shape == (700, 1080, 3)

    out_file = visualizer.save_snapshot(dashboard)
    assert os.path.exists(out_file)


def test_visualizer_all_view_modes():
    """Verify that update_frame and render_dashboard work across all 3 view modes without UnboundLocalError."""
    visualizer = PlayerTrackerVisualizer(monitor_idx=1)
    test_crop = np.zeros((280, 288, 3), dtype=np.uint8)

    for mode in [
        PlayerTrackerVisualizer.VIEW_ROOM_TEMPLATE,
        PlayerTrackerVisualizer.VIEW_WORLD_MAP,
        PlayerTrackerVisualizer.VIEW_REFERENCE_MAP,
    ]:
        visualizer.view_mode = mode
        dashboard, result = visualizer.update_frame(test_crop)
        assert dashboard is not None
        assert dashboard.shape == (700, 1080, 3)
        assert "room" in result
        assert "reference_map" in result
        assert "world_map" in result

