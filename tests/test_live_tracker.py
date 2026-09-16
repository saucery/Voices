"""
Unit Tests for Live Room Tracker and Screen Capturer
"""

import os
import sys
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.screen_capturer import ScreenCapturer
from src.live_tracker import LiveRoomTracker
from src.room_classifier import RoomClassifier


class MockCapturer:
    """Mock ScreenCapturer providing synthetic screenshots."""

    def __init__(self, screens):
        self.screens = screens
        self.idx = 0

    def capture(self, bbox=None):
        screen = self.screens[self.idx % len(self.screens)]
        self.idx += 1
        return screen


def test_live_tracker_room_change_event():
    # Synthetic screen 1 with Circle
    screen1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(screen1, (75, 25), 15, (255, 255, 255), -1)

    # Synthetic screen 2 with Square
    screen2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(screen2, (60, 10), (90, 40), (255, 255, 255), -1)

    classifier = RoomClassifier({
        "minimap_roi": {"top_pct": 0.0, "bottom_pct": 0.5, "left_pct": 0.5, "right_pct": 1.0},
        "rooms": []
    })

    # Register room 1 (Circle) and room 2 (Square)
    crop1 = classifier.extractor.extract_roi(screen1)
    crop2 = classifier.extractor.extract_roi(screen2)
    classifier.add_room("room_1", "Room 1", threshold=0.5, template_img=crop1)
    classifier.add_room("room_2", "Room 2", threshold=0.5, template_img=crop2)

    mock_cap = MockCapturer([screen1, screen1, screen2, screen2])
    room_changes = []

    def handle_change(prev, curr):
        room_changes.append((prev["room_id"] if prev else None, curr["room_id"]))

    tracker = LiveRoomTracker(
        classifier=classifier,
        capturer=mock_cap,
        interval=0.01,
        on_room_change=handle_change,
    )

    # Run 4 ticks
    tracker.start(max_ticks=4)

    # Room transition sequence expected: None -> room_1, then room_1 -> room_2
    assert len(room_changes) == 2
    assert room_changes[0] == (None, "room_1")
    assert room_changes[1] == ("room_1", "room_2")
