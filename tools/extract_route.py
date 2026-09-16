"""
Route Extraction CLI Tool
Extracts navigation waypoints from a painted route image:
- Blue dot = Start Position
- Green line = Navigation Route
- Red dot = Finish / Destination Position
Exports the waypoints to paths/movement_route.json.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.movement_path import MovementPath


def main():
    parser = argparse.ArgumentParser(
        description="Extract navigation waypoints from a painted route image (Blue start, Green line, Red finish)."
    )
    parser.add_argument(
        "--image",
        "-i",
        default="templates/route.png",
        help="Path to painted route image. Default: templates/route.png",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="paths/movement_route.json",
        help="Output JSON file path for waypoints. Default: paths/movement_route.json",
    )
    parser.add_argument(
        "--spacing",
        "-s",
        type=float,
        default=20.0,
        help="Pixel spacing between consecutive waypoints. Default: 20.0 px",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Route image not found at '{args.image}'")
        sys.exit(1)

    print("=" * 65)
    print(" EXTRACTING NAVIGATION ROUTE FROM PAINTED IMAGE")
    print(f" Source Image: {args.image}")
    print(f" Output JSON:  {args.output}")
    print(f" Node Spacing: {args.spacing} px")
    print("=" * 65)

    path_mgr = MovementPath()
    success = path_mgr.load_from_painted_image(
        image_path=args.image,
        spacing=args.spacing,
        save_json_path=args.output,
    )

    if success:
        print("\n[SUCCESS] Route extracted successfully!")
        print(f"Total Waypoints: {len(path_mgr.get_waypoints())}")
        for wp in path_mgr.get_waypoints():
            print(f" - [{wp['index']}] {wp['name']}: ({wp['x']:.0f}, {wp['y']:.0f})")
    else:
        print("\n[FAILURE] Could not extract route. Ensure the image has a Blue start dot, Green line, and Red finish dot.")


if __name__ == "__main__":
    main()
