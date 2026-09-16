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
        pos = self.positions[min(self.idx, len(self.positions)-1)]
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

def test_stuck_recovery():
    # Player stuck at (50, 20) for multiple frames
    positions = [(50, 20)] * 10

    mock_loc = MockLocalizer(positions)
    executor = SequenceExecutor(localizer=mock_loc, monitor_idx=1)
    
    pressed_keys = []
    def mock_press_key(key, duration=1.0):
        pressed_keys.append(f"{key}:{duration:.1f}s")

    executor.press_key = mock_press_key

    executor.follow_edge(target_distance=20.0, duration=2.5, direction="clockwise")

    print(f"\nStuck Recovery Sequence: {pressed_keys}")

test_stuck_recovery()
