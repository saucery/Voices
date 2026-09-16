"""
Test Movement Trajectory Visualization
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.map_localizer import MapLocalizer

def test_visualization():
    localizer = MapLocalizer("templates/full_map_reference.png")

    # Simulate a counter-clockwise trajectory around the red box edges
    simulated_history = [
        {"pos": (180, 60), "key": "a", "edge": "TOP_EDGE", "conf": 0.88},
        {"pos": (150, 60), "key": "a", "edge": "TOP_EDGE", "conf": 0.90},
        {"pos": (120, 60), "key": "a", "edge": "TOP_EDGE", "conf": 0.85},
        {"pos": (95, 60),  "key": "a", "edge": "TOP_EDGE", "conf": 0.89},
        {"pos": (95, 90),  "key": "s", "edge": "LEFT_EDGE", "conf": 0.87},
        {"pos": (95, 120), "key": "s", "edge": "LEFT_EDGE", "conf": 0.91},
        {"pos": (95, 155), "key": "s", "edge": "LEFT_EDGE", "conf": 0.86},
        {"pos": (130, 155), "key": "d", "edge": "BOTTOM_EDGE", "conf": 0.92},
        {"pos": (170, 155), "key": "d", "edge": "BOTTOM_EDGE", "conf": 0.88},
        {"pos": (190, 155), "key": "d", "edge": "BOTTOM_EDGE", "conf": 0.89},
        {"pos": (190, 120), "key": "w", "edge": "RIGHT_EDGE", "conf": 0.90},
        {"pos": (190, 80),  "key": "w", "edge": "RIGHT_EDGE", "conf": 0.85},
    ]

    out_path = localizer.render_trajectory_map(simulated_history, "debug_output/movement_trajectory.png")
    print(f"Generated Movement Trajectory Map at: {out_path}")
    assert os.path.exists(out_path)

test_visualization()
