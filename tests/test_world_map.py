"""
Unit Tests for World Map Tracker and Global Canvas Stitching
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.world_map import WorldMapTracker


def test_world_map_initialization():
    tracker = WorldMapTracker(canvas_size=(1000, 1000))
    assert tracker.global_pos == (500.0, 500.0)
    assert tracker.tile_count == 0
    assert tracker.global_canvas.shape == (1000, 1000, 3)


def test_world_map_update():
    tracker = WorldMapTracker(canvas_size=(500, 500))

    # Synthetic live crop
    crop1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
    res1 = tracker.update(crop1)

    assert tracker.tile_count == 1
    assert res1["explored_pixels"] > 0
    assert res1["explored_pct"] > 0.0

    # Save map output
    saved_path = tracker.save_map("debug_output/test_world_map.png")
    assert os.path.exists(saved_path)
