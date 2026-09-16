"""
Simulate Counter-Clockwise Red Edge Sweeper
"""
def test_counter_clockwise_sweeper():
    min_x, max_x = 93, 194
    min_y, max_y = 57, 163

    print(f"Red Box Bounds: X [{min_x}..{max_x}], Y [{min_y}..{max_y}]")

    # Simulate player starting near top-right corner inside red box (185, 60)
    px, py = 185.0, 60.0

    # Determine initial closest edge
    edges_dist = {
        "TOP_EDGE": abs(py - min_y),
        "LEFT_EDGE": abs(px - min_x),
        "BOTTOM_EDGE": abs(py - max_y),
        "RIGHT_EDGE": abs(px - max_x),
    }
    state = min(edges_dist, key=edges_dist.get)
    print(f"Initial Closest Edge: {state} at ({px}, {py})")

    ccw_order = ["TOP_EDGE", "LEFT_EDGE", "BOTTOM_EDGE", "RIGHT_EDGE"]

    for step in range(25):
        if state == "TOP_EDGE":
            primary_key = "a"
            px -= 12.0
            if px <= min_x + 5:
                state = "LEFT_EDGE"
                print(f"Step {step:2d} | Reached Left Corner! Transition to {state}")

        elif state == "LEFT_EDGE":
            primary_key = "s"
            py += 12.0
            if py >= max_y - 5:
                state = "BOTTOM_EDGE"
                print(f"Step {step:2d} | Reached Bottom Corner! Transition to {state}")

        elif state == "BOTTOM_EDGE":
            primary_key = "d"
            px += 12.0
            if px >= max_x - 5:
                state = "RIGHT_EDGE"
                print(f"Step {step:2d} | Reached Right Corner! Transition to {state}")

        elif state == "RIGHT_EDGE":
            primary_key = "w"
            py -= 12.0
            if py <= min_y + 5:
                state = "TOP_EDGE"
                print(f"Step {step:2d} | Reached Top Corner! Transition to {state}")

        print(f"Step {step:2d} | Key: {primary_key.upper()} | State: {state:11s} | Pos: ({px:5.1f}, {py:5.1f})")

test_counter_clockwise_sweeper()
