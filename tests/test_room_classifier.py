"""
Unit Tests for Minimap Extractor and Room Classifier (ORB Feature Matching & Position Tracking)
"""

import os
import sys
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.minimap_extractor import MinimapExtractor
from src.room_classifier import RoomClassifier


def test_minimap_extractor_roi():
    synthetic_img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi_config = {"top_pct": 0.0, "bottom_pct": 0.3, "left_pct": 0.7, "right_pct": 1.0}

    extractor = MinimapExtractor(roi_config)
    roi = extractor.extract_roi(synthetic_img)
    assert roi.shape == (30, 30, 3)


def test_minimap_extractor_preprocess():
    extractor = MinimapExtractor()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), 2)

    edges = extractor.preprocess(img)
    assert edges.shape == (100, 100)
    assert np.max(edges) > 0


def test_orb_feature_matching_and_position_tracking():
    config_dict = {
        "minimap_roi": {"top_pct": 0.0, "bottom_pct": 0.5, "left_pct": 0.5, "right_pct": 1.0},
        "matching": {"orb_features": 1000, "match_threshold": 0.40},
        "rooms": []
    }
    classifier = RoomClassifier(config_dict)

    # Reference Template Image with distinct geometric map features (cross + circles)
    template_screen = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(template_screen, (130, 10), (195, 80), (255, 255, 255), 2)
    cv2.circle(template_screen, (145, 30), 5, (255, 255, 255), -1)
    cv2.circle(template_screen, (180, 60), 6, (255, 255, 255), -1)
    cv2.line(template_screen, (140, 45), (185, 45), (255, 255, 255), 2)

    crop_temp = classifier.extractor.extract_roi(template_screen)
    classifier.add_room("room_orb", "ORB Test Room", threshold=0.40, template_img=crop_temp)

    # Live screenshot with SAME map features, slightly translated / moved
    live_screen = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(live_screen, (128, 12), (193, 82), (255, 255, 255), 2)
    cv2.circle(live_screen, (143, 32), 5, (255, 255, 255), -1)
    cv2.circle(live_screen, (178, 62), 6, (255, 255, 255), -1)
    cv2.line(live_screen, (138, 47), (183, 47), (255, 255, 255), 2)

    result = classifier.classify(live_screen)

    assert result["recognized"] is True
    assert result["room_id"] == "room_orb"
    assert result["character_position"] is not None
    # Verify character position is valid tuple of floats (X, Y)
    char_x, char_y = result["character_position"]
    assert char_x >= 0.0 and char_y >= 0.0
