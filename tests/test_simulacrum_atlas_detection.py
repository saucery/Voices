"""
Unit tests validating Atlas map Simulacrum node detection and accessibility classification.
Tests with user-provided screenshot containing 3 accessible nodes (purple) and 4 inaccessible nodes (black).
"""
import os
import sys
import pytest
import cv2

sys.path.insert(0, os.path.abspath("."))
from src.route_navigator import RouteNavigator


def test_atlas_simulacrum_detection_and_accessibility():
    """
    Validates that:
    1. All 7 Simulacrum nodes are detected on Atlas map.
    2. Accessible nodes (with active glowing blue/cyan portal rift) are correctly flagged is_accessible=True.
    3. Inaccessible nodes (dark base with no blue portal) are flagged is_accessible=False.
    4. Accessible nodes are ranked first.
    """
    img_path = r"C:\Users\gregg\.gemini\antigravity-ide\brain\829d893f-727f-4e73-bd1c-db41c3ee9e7e\.user_uploaded\media_1790872842765.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found at {img_path}")

    screen = cv2.imread(img_path)
    assert screen is not None, "Failed to load Atlas test screenshot"

    nav = RouteNavigator()
    nodes = nav.detect_simulacrum_map_nodes(screen=screen, threshold=0.78)

    assert len(nodes) == 7, f"Expected 7 Simulacrum nodes, found {len(nodes)}"

    # The top 3 ranked nodes must all be accessible
    accessible_nodes = [n for n in nodes if n["is_accessible"]]
    inaccessible_nodes = [n for n in nodes if not n["is_accessible"]]

    assert len(accessible_nodes) == 3, f"Expected 3 accessible nodes (purple), found {len(accessible_nodes)}"
    assert len(inaccessible_nodes) == 4, f"Expected 4 inaccessible nodes (black), found {len(inaccessible_nodes)}"

    # Verify that the top 3 items in the sorted results are the accessible ones
    for i in range(3):
        assert nodes[i]["is_accessible"] is True, f"Node #{i+1} should be accessible"
        assert nodes[i]["acc_score"] >= 2, f"Node #{i+1} should have acc_score >= 2"

    for i in range(3, 7):
        assert nodes[i]["is_accessible"] is False, f"Node #{i+1} should be inaccessible"
        assert nodes[i]["acc_score"] == 0, f"Node #{i+1} should have acc_score 0"

    # Verify select_accessible_simulacrum_map picks an accessible node
    picked = nav.select_accessible_simulacrum_map(candidates=nodes, dry_run=True)
    assert picked is not None
    assert picked["is_accessible"] is True
