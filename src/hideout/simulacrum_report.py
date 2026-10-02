"""
Simulacrum Feedback Report Generator & Template Exporter.
Compiles diagnostic JSON reports, annotated visual comparison screenshots,
and saves newly cropped templates for active detection training.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np


def export_simulacrum_feedback_report(
    cv_image: np.ndarray,
    detected_nodes: List[Dict[str, Any]],
    bot_selected_node: Optional[Dict[str, Any]],
    node_feedback: Dict[int, str],
    image_source: Optional[str] = None,
    threshold: float = 0.78,
    reports_dir: str = "reports",
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Compiles a structured JSON diagnostic report and renders an annotated
    comparison image comparing bot detections against user feedback labels.

    Returns:
        (report_json_path, report_img_path, summary_dict)
    """
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = os.path.join(reports_dir, f"simulacrum_feedback_{ts}.json")
    report_img_path = os.path.join(reports_dir, f"simulacrum_feedback_{ts}.png")

    items = []
    correct_count = 0
    wrong_acc_count = 0
    false_pos_count = 0

    for i, n in enumerate(detected_nodes):
        fb = node_feedback.get(i, "unreviewed")
        if fb in ["correct_acc", "correct_inacc"]:
            correct_count += 1
        elif fb == "wrong_acc":
            wrong_acc_count += 1
        elif fb == "false_pos":
            false_pos_count += 1

        is_bot_pick = bool(
            bot_selected_node
            and n["screen_circle_pos"] == bot_selected_node.get("screen_circle_pos")
        )

        items.append({
            "node_index": i + 1,
            "screen_medal_pos": list(n["screen_medal_pos"]),
            "screen_circle_pos": list(n["screen_circle_pos"]),
            "confidence": round(float(n["confidence"]), 4),
            "scale": round(float(n["scale"]), 3),
            "bot_accessible": bool(n["is_accessible"]),
            "bot_acc_score": int(n.get("acc_score", 0)),
            "is_bot_selection": is_bot_pick,
            "user_feedback": fb,
        })

    total = len(detected_nodes)
    accuracy = round(correct_count / max(1, total), 3)

    summary = {
        "verified_correct": correct_count,
        "wrong_accessibility": wrong_acc_count,
        "false_positives": false_pos_count,
        "accuracy": accuracy,
    }

    report_data = {
        "timestamp": ts,
        "image_source": image_source or "live_capture",
        "resolution": [cv_image.shape[1], cv_image.shape[0]],
        "threshold": float(threshold),
        "total_detected": total,
        "summary": summary,
        "nodes": items,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Generate annotated diagnostic image
    ann = cv_image.copy()
    for i, n in enumerate(detected_nodes):
        mx, my = n["screen_medal_pos"]
        cx, cy = n["screen_circle_pos"]
        fb = node_feedback.get(i, "unreviewed")

        if fb in ["correct_acc", "correct_inacc"]:
            color = (0, 255, 120)  # Green
        elif fb == "wrong_acc":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red

        cv2.circle(ann, (cx, cy), 14, color, 2, cv2.LINE_AA)
        cv2.drawMarker(ann, (cx, cy), color, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.rectangle(ann, (mx - 16, my - 16), (mx + 16, my + 16), color, 2)
        cv2.putText(
            ann,
            f"#{i+1} [{fb}]",
            (mx - 25, my - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(report_img_path, ann)
    return report_json_path, report_img_path, summary


def save_cropped_simulacrum_template(
    cv_image: np.ndarray,
    manual_box: Optional[Tuple[int, int, int, int]] = None,
    node: Optional[Dict[str, Any]] = None,
    templates_dir: str = "templates/ui",
) -> Tuple[str, str]:
    """
    Crops a template patch from a user selection or detected node and saves
    it to templates/ui/simulacrum_node_v{N}.png.

    Returns:
        (saved_file_path, description)
    """
    os.makedirs(templates_dir, exist_ok=True)
    sh, sw = cv_image.shape[:2]

    if manual_box is not None:
        x1, y1, x2, y2 = manual_box
        crop = cv_image[y1:y2, x1:x2]
        desc = f"manual user box ({x2 - x1}x{y2 - y1}px)"
    elif node is not None:
        mx, my = node["screen_medal_pos"]
        s = node.get("scale", 1.0)
        hw, hh = int(16 * s), int(16 * s)
        crop = cv_image[max(0, my - hh) : min(sh, my + hh), max(0, mx - hw) : min(sw, mx + hw)]
        desc = f"node medal patch ({crop.shape[1]}x{crop.shape[0]}px)"
    else:
        raise ValueError("Neither manual_box nor node was provided for template cropping.")

    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        raise ValueError("Selected area is too small to serve as a valid template.")

    next_idx = 1
    while os.path.exists(os.path.join(templates_dir, f"simulacrum_node_v{next_idx}.png")):
        next_idx += 1

    out_path = os.path.join(templates_dir, f"simulacrum_node_v{next_idx}.png")
    cv2.imwrite(out_path, crop)
    return out_path, desc
