import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.map_localizer import MapLocalizer

def test_map_localizer_red_zone():
    localizer = MapLocalizer("templates/full_map_reference.png")
    print(f"Red Zone Bounds: {localizer.red_zone_bounds}")
    assert localizer.red_zone_bounds is not None
    min_x, max_x, min_y, max_y = localizer.red_zone_bounds
    print(f"Verified Red Zone: X [{min_x}..{max_x}], Y [{min_y}..{max_y}] ({max_x - min_x}x{max_y - min_y} px)")

test_map_localizer_red_zone()
