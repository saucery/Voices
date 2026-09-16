"""
Simulate Red Square Boundary Enforcement
"""
def simulate_red_zone_navigation():
    min_x, max_x = 93, 194
    min_y, max_y = 57, 163

    print(f"Red Zone Box: X [{min_x}..{max_x}], Y [{min_y}..{max_y}]")

    # Scenario 1: Player starts outside zone at (220, 100) (to the right of max_x)
    player_x, player_y = 220.0, 100.0

    for step in range(15):
        # 1. Zone Entry Enforcement
        if player_x > max_x + 5:
            move_key = "a"
            player_x -= 15.0
            print(f"Step {step:2d} | OUTSIDE ZONE (Right) | Pos: ({player_x:5.1f}, {player_y:5.1f}) -> Move KEY: {move_key.upper()} (Heading Left into Zone)")
        elif player_x < min_x - 5:
            move_key = "d"
            player_x += 15.0
            print(f"Step {step:2d} | OUTSIDE ZONE (Left)  | Pos: ({player_x:5.1f}, {player_y:5.1f}) -> Move KEY: {move_key.upper()} (Heading Right into Zone)")
        elif player_y > max_y + 5:
            move_key = "w"
            player_y -= 15.0
            print(f"Step {step:2d} | OUTSIDE ZONE (Bottom)| Pos: ({player_x:5.1f}, {player_y:5.1f}) -> Move KEY: {move_key.upper()} (Heading Up into Zone)")
        elif player_y < min_y - 5:
            move_key = "s"
            player_y += 15.0
            print(f"Step {step:2d} | OUTSIDE ZONE (Top)   | Pos: ({player_x:5.1f}, {player_y:5.1f}) -> Move KEY: {move_key.upper()} (Heading Down into Zone)")
        else:
            print(f"Step {step:2d} | INSIDE ZONE          | Pos: ({player_x:5.1f}, {player_y:5.1f}) -> Sweeping Perimeter!")
            break

simulate_red_zone_navigation()
