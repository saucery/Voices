"""
Test Perimeter Sweep State Machine
"""
import math

def simulate_perimeter_sweep():
    map_w, map_h = 273, 270
    target_distance = 20.0

    min_x = target_distance
    max_x = map_w - target_distance
    min_y = target_distance
    max_y = map_h - target_distance

    print(f"Map Bounds: X [{min_x:.0f}..{max_x:.0f}], Y [{min_y:.0f}..{max_y:.0f}]")

    # Simulate player starting at top-left corner (30, 25)
    player_x, player_y = 30.0, 25.0
    direction = "clockwise"

    # Determine initial edge
    edges = {
        "TOP_EDGE": player_y,
        "RIGHT_EDGE": map_w - player_x,
        "BOTTOM_EDGE": map_h - player_y,
        "LEFT_EDGE": player_x
    }
    current_edge = min(edges, key=edges.get)
    print(f"Initial Edge: {current_edge} at ({player_x}, {player_y})")

    clockwise_order = ["TOP_EDGE", "RIGHT_EDGE", "BOTTOM_EDGE", "LEFT_EDGE"]

    for step in range(20):
        if current_edge == "TOP_EDGE":
            move_key = "d" if direction == "clockwise" else "a"
            # Simulate movement
            player_x += 15.0 if direction == "clockwise" else -15.0
            if player_x >= max_x - 5:
                next_idx = (clockwise_order.index(current_edge) + 1) % 4
                current_edge = clockwise_order[next_idx]
                print(f"Step {step}: Reached Right Corner! Transition to {current_edge}")
        elif current_edge == "RIGHT_EDGE":
            move_key = "s" if direction == "clockwise" else "w"
            player_y += 15.0 if direction == "clockwise" else -15.0
            if player_y >= max_y - 5:
                next_idx = (clockwise_order.index(current_edge) + 1) % 4
                current_edge = clockwise_order[next_idx]
                print(f"Step {step}: Reached Bottom Corner! Transition to {current_edge}")
        elif current_edge == "BOTTOM_EDGE":
            move_key = "a" if direction == "clockwise" else "d"
            player_x -= 15.0 if direction == "clockwise" else -15.0
            if player_x <= min_x + 5:
                next_idx = (clockwise_order.index(current_edge) + 1) % 4
                current_edge = clockwise_order[next_idx]
                print(f"Step {step}: Reached Left Corner! Transition to {current_edge}")
        elif current_edge == "LEFT_EDGE":
            move_key = "w" if direction == "clockwise" else "s"
            player_y -= 15.0 if direction == "clockwise" else +15.0
            if player_y <= min_y + 5:
                next_idx = (clockwise_order.index(current_edge) + 1) % 4
                current_edge = clockwise_order[next_idx]
                print(f"Step {step}: Reached Top Corner! Transition to {current_edge}")

        print(f"Step {step:2d} | Key: {move_key.upper()} | Edge: {current_edge:11s} | Pos: ({player_x:5.1f}, {player_y:5.1f})")

simulate_perimeter_sweep()
