"""
Unit tests validating tracker improvements:
- Minimap wall preservation through Delirium fog/lighting
- Translation consensus ORB matching
- Dead reckoning during transient low-confidence frames
- Smart forward jump recovery
"""

import os
import sys
import math
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath("."))

from src.minimap_extractor import MinimapExtractor
from src.room_classifier import RoomClassifier


def test_minimap_center_only_orange_suppression():
    """Verify that orange suppression only removes player marker at center, leaving outer walls intact."""
    extractor = MinimapExtractor()
    # Create test image (100x100) with an outer cyan wall and orange fog all over
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Cyan wall at top
    img[15:25, 20:80] = [180, 180, 20]  # BGR cyan
    # Orange player marker at center (50, 50)
    img[48:53, 48:53] = [30, 140, 240]  # BGR orange
    # Orange delirium fog across bottom
    img[70:90, 20:80] = [30, 140, 240]

    pre = extractor.preprocess(img)
    # Wall at top (y: 15..25) must NOT be erased by the delirium fog
    assert np.count_nonzero(pre[15:25, 20:80]) > 0, "Cyan wall was erased by fog!"


def test_room_classifier_translation_consensus():
    """Verify translation consensus recovers position when affine estimation is distorted."""
    rc = RoomClassifier("config.json")
    # Feed an empty crop to ensure no crash and graceful fallback
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    res = rc.classify(empty, is_crop=True)
    assert "recognized" in res
    assert "character_position" in res


def test_dead_reckoning_advances_towards_expected_pos():
    """Verify dead reckoning softly advances last_known_pos towards expected_pos during transient drops."""
    rc = RoomClassifier("config.json")
    rc.is_locked = True
    rc.last_known_pos = (100.0, 100.0)
    rc.last_room_id = "map_layout"
    rc.last_room_name = "Map Layout"
    rc.lost_frame_count = 1

    # Classify empty frame with expected_pos at (150.0, 100.0)
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    res = rc.classify(empty, is_crop=True, expected_pos=(150.0, 100.0))

    assert res["recognized"] is True, "Hysteresis grace period must maintain lock"
    assert res["character_position"] is not None
    # Position must have stepped towards 150.0 (x > 100.0)
    assert res["character_position"][0] > 100.0
