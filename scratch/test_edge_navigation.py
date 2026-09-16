import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.sequence_executor import SequenceExecutor

class MockCapturer:
    def capture(self):
        return None

class MockLocalizer:
    def __init__(self, positions):
        self.positions = positions
        self.idx = 0

    def localize_player(self, screenshot):
        if self.idx < len(self.positions):
            pos = self.positions[self.idx]
            self.idx += 1
            return {
                "located": True,
                "player_position": pos,
                "map_width": 273,
                "map_height": 270,
                "edge_distances": {
                    "dist_top": pos[1],
                    "dist_bottom": 270 - pos[1],
                    "dist_left": pos[0],
                    "dist_right": 273 - pos[0],
                }
            }
        return {"located": False}

def test_full_perimeter_traversal():
    positions = [
        (30, 20),
        (100, 20),
        (200, 20),
        (250, 20),  # Right corner threshold hit!
        (250, 80),
        (250, 160),
        (250, 246), # Bottom corner threshold hit!
    ]

    mock_loc = MockLocalizer(positions)
    executor = SequenceExecutor(localizer=mock_loc, monitor_idx=1)
    
    # Track pressed keys
    pressed_keys = []
    def mock_press_key(key, duration=1.0):
        pressed_keys.append(key)

    executor.press_key = mock_press_key

    # Run follow_edge for 7 steps
    executor.follow_edge(target_distance=20.0, duration=3.0, direction="clockwise")

    print(f"\nExecuted Keys Sequence: {pressed_keys}")

test_full_perimeter_traversal()
