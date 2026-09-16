"""
Simulate Red Edge Perimeter Hugger
"""
def test_red_perimeter_hugger():
    min_x, max_x = 93, 194
    min_y, max_y = 57, 163

    print(f"Red Edge Perimeter Box: X [{min_x}..{max_x}], Y [{min_y}..{max_y}]")

    # Start at top-left corner of red box (95, 60)
    px, py = 95.0, 60.0
    direction = "clockwise"
    state = "TOP_EDGE"

    for step in range(25):
        if state == "TOP_EDGE":
            primary_key = "d" if direction == "clockwise" else "a"
            if direction == "clockwise" and px >= max_x - 5:
                state = "RIGHT_EDGE"
                primary_key = "s"

            if py > min_y + 8:
                move_key = "w"
                py -= 5.0
            elif py < min_y - 3:
                move_key = "s"
                py += 5.0
            else:
                move_key = primary_key
                px += 10.0 if direction == "clockwise" else -10.0

        elif state == "RIGHT_EDGE":
            primary_key = "s" if direction == "clockwise" else "w"
            if direction == "clockwise" and py >= max_y - 5:
                state = "BOTTOM_EDGE"
                primary_key = "a"

            if px < max_x - 8:
                move_key = "d"
                px += 5.0
            elif px > max_x + 3:
                move_key = "a"
                px -= 5.0
            else:
                move_key = primary_key
                py += 10.0 if direction == "clockwise" else -10.0

        elif state == "BOTTOM_EDGE":
            primary_key = "a" if direction == "clockwise" else "d"
            if direction == "clockwise" and px <= min_x + 5:
                state = "LEFT_EDGE"
                primary_key = "w"

            if py < max_y - 8:
                move_key = "s"
                py += 5.0
            elif py > max_y + 3:
                move_key = "w"
                py -= 5.0
            else:
                move_key = primary_key
                px -= 10.0 if direction == "clockwise" else 10.0

        else: # LEFT_EDGE
            primary_key = "w" if direction == "clockwise" else "s"
            if direction == "clockwise" and py <= min_y + 5:
                state = "TOP_EDGE"
                primary_key = "d"

            if px > min_x + 8:
                move_key = "a"
                px -= 5.0
            elif px < min_x - 3:
                move_key = "d"
                px += 5.0
            else:
                move_key = primary_key
                py -= 10.0 if direction == "clockwise" else 10.0

        print(f"Step {step:2d} | State: {state:11s} | Key: {move_key.upper()} | Pos: ({px:5.1f}, {py:5.1f})")

test_red_perimeter_hugger()
