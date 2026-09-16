"""
Player Position Tracker Window CLI Tool
Launches an interactive live GUI window to track player position and compare
what the bot detects vs what the game shows on the minimap.
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.player_tracker_visualizer import PlayerTrackerVisualizer
from src.map_localizer import MapLocalizer
from src.room_classifier import RoomClassifier
from src.screen_capturer import ScreenCapturer


def main():
    parser = argparse.ArgumentParser(
        description="Launch Interactive Player Position Tracker Window to compare bot detection vs game minimap."
    )
    parser.add_argument(
        "--monitor",
        "-m",
        type=int,
        default=2,
        help="Monitor display index to capture live (1 = Primary, 2 = Secondary). Default: 2 (Game Screen)",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Path to full reference map layout image. Default: from config.json (map_layout_file)",
    )
    parser.add_argument(
        "--image",
        "-i",
        help="Optional static screenshot image to inspect in the visualizer window instead of live capture.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.04,
        help="Refresh interval in seconds (e.g. 0.04 = ~25 FPS). Default: 0.04",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="List connected display monitors and exit.",
    )

    args = parser.parse_args()

    if args.list_monitors:
        print("=" * 60)
        print(" CONNECTED DISPLAY MONITORS")
        print("=" * 60)
        monitors = ScreenCapturer.list_monitors()
        for m in monitors:
            print(f" - Monitor {m['index']}: {m['description']}")
        print("=" * 60)
        return

    localizer = MapLocalizer(reference_map_path=args.map)
    classifier = RoomClassifier(map_layout_path=args.map)

    # Static image mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Specified image file not found: {args.image}")
            sys.exit(1)

        print(f"Loading static screenshot: {args.image}")
        img = cv2.imread(args.image)
        visualizer = PlayerTrackerVisualizer(classifier=classifier, localizer=localizer, monitor_idx=args.monitor)

        cv2.namedWindow(visualizer.window_title, cv2.WINDOW_AUTOSIZE)
        print("Press [V] for map view, [E] for edges, [T] for trail, [S] to save, [Q/Esc] to exit.")

        while True:
            dashboard, loc_res = visualizer.update_frame(img)
            cv2.imshow(visualizer.window_title, dashboard)
            key = cv2.waitKey(30) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key in [ord("v"), ord("V")]:
                visualizer.view_mode = (visualizer.view_mode + 1) % 3
            elif key in [ord("e"), ord("E")]:
                visualizer.show_edges = not visualizer.show_edges
            elif key in [ord("t"), ord("T")]:
                visualizer.show_trajectory = not visualizer.show_trajectory
            elif key in [ord("s"), ord("S")]:
                visualizer.save_snapshot(dashboard)

        cv2.destroyAllWindows()
        return

    # Live capture mode
    capturer = ScreenCapturer(monitor_idx=args.monitor)
    visualizer = PlayerTrackerVisualizer(
        classifier=classifier,
        localizer=localizer,
        capturer=capturer,
        monitor_idx=args.monitor,
    )

    try:
        visualizer.run(interval=args.interval)
    except KeyboardInterrupt:
        print("\nStopping Tracker Visualizer.")


if __name__ == "__main__":
    main()
