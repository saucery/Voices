"""
Unit tests validating Simulacrum feedback reporting, template cropping, and UI backend logic.
"""
import os
import sys
import json
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))

from src.hideout.simulacrum_report import (
    export_simulacrum_feedback_report,
    save_cropped_simulacrum_template,
)
from src.route_navigator import RouteNavigator


def test_simulacrum_report_export(tmp_path):
    """Validates that export_simulacrum_feedback_report generates JSON & image reports with correct stats."""
    # Create synthetic test image
    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)

    dummy_nodes = [
        {
            "screen_medal_pos": (200, 100),
            "screen_circle_pos": (200, 126),
            "confidence": 0.94,
            "scale": 1.0,
            "is_accessible": True,
            "acc_score": 3,
        },
        {
            "screen_medal_pos": (400, 150),
            "screen_circle_pos": (400, 176),
            "confidence": 0.88,
            "scale": 1.0,
            "is_accessible": False,
            "acc_score": 0,
        },
        {
            "screen_medal_pos": (100, 300),
            "screen_circle_pos": (100, 326),
            "confidence": 0.82,
            "scale": 1.0,
            "is_accessible": True,
            "acc_score": 2,
        },
    ]

    bot_selected = dummy_nodes[0]

    # User feedback: Node 0 correct, Node 1 wrong accessibility, Node 2 false positive
    feedback = {
        0: "correct_acc",
        1: "wrong_acc",
        2: "false_pos",
    }

    out_dir = str(tmp_path / "reports")
    json_path, img_path, summary = export_simulacrum_feedback_report(
        cv_image=dummy_img,
        detected_nodes=dummy_nodes,
        bot_selected_node=bot_selected,
        node_feedback=feedback,
        image_source="test_synthetic.png",
        threshold=0.78,
        reports_dir=out_dir,
    )

    assert os.path.exists(json_path), f"JSON report not created at {json_path}"
    assert os.path.exists(img_path), f"Annotated image not created at {img_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_detected"] == 3
    assert data["summary"]["verified_correct"] == 1
    assert data["summary"]["wrong_accessibility"] == 1
    assert data["summary"]["false_positives"] == 1
    assert data["summary"]["accuracy"] == pytest.approx(0.333, abs=0.01)
    assert data["nodes"][0]["is_bot_selection"] is True


def test_save_cropped_template(tmp_path):
    """Validates that save_cropped_simulacrum_template creates a new template file correctly."""
    dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    tpl_dir = str(tmp_path / "templates")

    # 1. Test cropping from detected node
    node = {
        "screen_medal_pos": (100, 100),
        "scale": 1.0,
    }
    path1, desc1 = save_cropped_simulacrum_template(
        cv_image=dummy_img,
        node=node,
        templates_dir=tpl_dir,
    )
    assert os.path.exists(path1)
    assert "simulacrum_node_v1.png" in path1

    # 2. Test cropping from manual user box
    manual_box = (20, 20, 60, 60)
    path2, desc2 = save_cropped_simulacrum_template(
        cv_image=dummy_img,
        manual_box=manual_box,
        templates_dir=tpl_dir,
    )
    assert os.path.exists(path2)
    assert "simulacrum_node_v2.png" in path2


def test_template_loader_reload():
    """Validates that RouteNavigator reloads templates cleanly."""
    nav = RouteNavigator()
    count = nav.reload_simulacrum_templates()
    assert count >= 7, f"Expected at least 7 simulacrum templates loaded, got {count}"


def test_select_next_available_node():
    """Validates that select_next_available_node cycles through accessible candidate nodes."""
    import tkinter as tk
    from tools.simulacrum_feedback_ui import SimulacrumFeedbackUI

    root = tk.Tk()
    root.withdraw()
    try:
        app = SimulacrumFeedbackUI(root)
        app.detected_nodes = [
            {"screen_medal_pos": (100, 100), "screen_circle_pos": (100, 126), "confidence": 0.95, "scale": 1.0, "is_accessible": True, "acc_score": 3, "connected_greens": []},
            {"screen_medal_pos": (200, 100), "screen_circle_pos": (200, 126), "confidence": 0.90, "scale": 1.0, "is_accessible": False, "acc_score": 0, "connected_greens": []},
            {"screen_medal_pos": (300, 100), "screen_circle_pos": (300, 126), "confidence": 0.85, "scale": 1.0, "is_accessible": True, "acc_score": 2, "connected_greens": []},
        ]
        app.available_candidate_indices = [0, 2]
        app.current_candidate_pointer = 0
        app.bot_selected_node = app.detected_nodes[0]

        # Call select_next_available_node -> should advance pointer to 1 (which maps to candidate index 2)
        app.select_next_available_node()
        assert app.current_candidate_pointer == 1
        assert app.bot_selected_node["screen_circle_pos"] == (300, 126)

        # Call again -> should wrap back to pointer 0 (candidate index 0)
        app.select_next_available_node()
        assert app.current_candidate_pointer == 0
        assert app.bot_selected_node["screen_circle_pos"] == (100, 126)
    finally:
        root.destroy()

