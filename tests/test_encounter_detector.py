"""
Unit Tests for Encounter Detector
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encounter_detector import EncounterDetector
from src.event_engine import GameEventManager
from src.task_runner import TaskExecutor


def test_encounter_detector_initialization():
    detector = EncounterDetector()
    assert detector.roi["top_pct"] == 0.05
    assert detector.roi["left_pct"] == 0.10


def test_encounter_detection_on_provided_image():
    user_image_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".gemini",
        "antigravity",
        "brain",
        "e589bb3c-cfca-4b1e-b5ad-346c8b054b70",
        "media__1789401937945.png",
    )

    if not os.path.exists(user_image_path):
        user_image_path = "templates/ui/encounter_banner.png"

    detector = EncounterDetector("templates/ui/encounter_banner.png")
    res = detector.detect(user_image_path)

    assert res["detected"] is True
    assert res["title"] == "TRAIL OF SUFFERING"
    assert res["wave"] == "WAVE 1/7"
    assert res["has_red_warning_text"] is True


def test_encounter_event_emission():
    em = GameEventManager()
    executor = TaskExecutor(em)

    events_fired = []

    def handle_encounter(payload):
        events_fired.append(payload)

    em.on("ON_ENCOUNTER_DETECTED", handle_encounter)

    user_image_path = "templates/ui/encounter_banner.png"
    if os.path.exists(user_image_path):
        cv2 = pytest.importorskip("cv2")
        tmpl = cv2.imread(user_image_path)
        if tmpl is not None:
            t_h, t_w = tmpl.shape[:2]
            # Create a 1080p full screenshot (1080x1920) and place tmpl in top center ROI
            dummy_screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
            dummy_screenshot[60 : 60 + t_h, 250 : 250 + t_w] = tmpl

            em.process_frame(dummy_screenshot, {"room_id": "room_1"}, {"global_position": (1000.0, 1000.0)})

            assert len(events_fired) >= 1
            assert events_fired[0]["title"] == "TRAIL OF SUFFERING"
