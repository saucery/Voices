"""
Unit Tests for MapLocalizer Module
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.map_localizer import MapLocalizer


def test_map_localizer_initialization():
    ref_map = "templates/full_map_reference.png"
    localizer = MapLocalizer(reference_map_path=ref_map)

    if os.path.exists(ref_map):
        assert localizer.ref_img is not None
        assert localizer.ref_w > 0
        assert localizer.ref_h > 0


def test_map_localizer_localization():
    ref_map = "templates/full_map_reference.png"
    if not os.path.exists(ref_map):
        pytest.skip("Reference map file not present")

    localizer = MapLocalizer(reference_map_path=ref_map)

    # Use a crop from reference map
    ref_img = localizer.ref_img
    crop = ref_img[50:150, 50:150].copy()

    res = localizer.localize_player(crop)
    assert res["located"] is True
    assert res["confidence"] > 0.30
    assert res["player_position"] is not None

    saved_path = localizer.render_player_location(res, "debug_output/test_player_location.png")
    assert os.path.exists(saved_path)
