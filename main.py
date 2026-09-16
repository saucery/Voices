"""
Main CLI Entry Point
Executes map & room recognition, global world map stitching, and event-driven task execution.
"""

import argparse
import os
import sys
import time
import cv2

from src.room_classifier import RoomClassifier
from src.minimap_extractor import MinimapExtractor
from src.screen_capturer import ScreenCapturer
from src.live_tracker import LiveRoomTracker
from src.world_map import WorldMapTracker
from src.event_engine import GameEventManager
from src.task_runner import TaskExecutor
from src.sequence_executor import SequenceExecutor
from src.window_focus import window_focuser


def on_room_changed(prev_res, curr_res):
    prev_name = prev_res["room_name"] if prev_res else "None"
    curr_name = curr_res["room_name"]
    variant = curr_res.get("matched_variant", "")
    conf = curr_res["confidence"]
    pos = curr_res.get("character_position")
    timestamp = curr_res.get("timestamp", "")
    variant_str = f" [{variant}]" if variant else ""
    pos_str = f" | Pos: {pos}" if pos else ""
    print(f"\n[{timestamp}] >>> ROOM CHANGED: {prev_name} -> {curr_name}{variant_str}{pos_str} (Confidence: {conf:.2%})")


def on_tick(result):
    timestamp = result.get("timestamp", "")
    room = result["room_name"]
    variant = result.get("matched_variant", "")
    conf = result["confidence"]
    map_res = result.get("world_map", {})
    loc_res = result.get("localization", {})
    player_pos = loc_res.get("player_position", ("N/A", "N/A"))
    loc_conf = loc_res.get("confidence", 0.0)
    edge_dists = loc_res.get("edge_distances", {})

    if loc_res.get("located"):
        near_edge = edge_dists.get("nearest_boundary_name", "")
        near_dist = edge_dists.get("nearest_boundary_dist", 0)
        wall_dist = edge_dists.get("dist_nearest_wall", 0.0)
        rec = f"Pos: {player_pos} [{loc_conf:.0%}] | Edge: {near_edge} ({near_dist}px) | Wall: {wall_dist}px"
    else:
        rec = "SEARCHING MAP..."

    variant_str = f" ({variant})" if variant else ""
    print(f"[{timestamp}] Polling Game Screen... Current: {room}{variant_str} | {rec}", end="\r", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Game Bot Minimap, Global Map Stitching & Event-Driven Task Engine CLI"
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to full game screenshot image to analyze (single screenshot mode).",
    )
    parser.add_argument(
        "--live",
        "-l",
        action="store_true",
        help="Enable continuous live game screen monitoring.",
    )
    parser.add_argument(
        "--routine",
        "-r",
        help="Path to JSON sequence routine file to execute (e.g. routines/trail_of_suffering.json).",
    )
    parser.add_argument(
        "--monitor",
        "-m",
        nargs="?",
        const=2,
        type=int,
        default=2,
        help="Target monitor index for live capture (1 = Primary, 2 = Secondary). Default: 2 (Game Screen)",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="List all connected displays and exit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for live mode. Default: 1.0",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum number of screenshot polling ticks for live mode (optional).",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Path to config file. Default: config.json",
    )
    parser.add_argument(
        "--save-map",
        action="store_true",
        help="Save stitched global world map image to disk.",
    )
    parser.add_argument(
        "--track",
        "-t",
        action="store_true",
        help="Launch interactive Player Position Tracker window (compares bot detection vs game minimap).",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Path to full reference map layout image for tracking. Default: from config.json (map_layout_file)",
    )
    parser.add_argument(
        "--movement-file",
        "-p",
        default=None,
        help="Path to movement route JSON file. Default: from config.json (movement_file)",
    )
    parser.add_argument(
        "--navigate",
        "-n",
        action="store_true",
        help="Launch autonomous route navigation with live position monitoring (press [A] to toggle WASD autonavigation).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run route navigation in headless console mode without GUI monitor window.",
    )
    parser.add_argument(
        "--pink-dot",
        type=int,
        default=None,
        help="Start route navigation targeting a specific Pink Dot index (1-indexed: 1 = Pink #1, 2 = Pink #2, 3 = Pink #3).",
    )

    args = parser.parse_args()

    # 1. List Monitors Mode
    if args.list_monitors:
        print("=" * 60)
        print(" CONNECTED DISPLAY MONITORS")
        print("=" * 60)
        monitors = ScreenCapturer.list_monitors()
        for m in monitors:
            print(f" - Monitor {m['index']}: {m['description']}")
        print("=" * 60)
        print("Usage: Pass --monitor <index> to capture a specific screen.")
        return

    # 2. Headless Navigation Mode (explicit console-only request)
    if args.navigate and args.headless:
        capturer = ScreenCapturer(monitor_idx=args.monitor)
        executor = SequenceExecutor(capturer=capturer, monitor_idx=args.monitor)
        executor.follow_route(
            route_path=args.movement_file,
            map_path=args.map,
            config_path=args.config,
        )
        return

    # 3. Interactive Player Position Tracker & Integrated Autonomous Navigation
    # Triggered by --navigate, --track, or --monitor as primary flag
    monitor_invoked_as_cmd = ("--monitor" in sys.argv or "-m" in sys.argv) and not (
        args.live or args.routine or args.input or args.list_monitors or args.save_map
    )

    if args.track or args.navigate or monitor_invoked_as_cmd:
        from src.player_tracker_visualizer import PlayerTrackerVisualizer
        from src.map_localizer import MapLocalizer
        from src.room_classifier import RoomClassifier
        from src.movement_path import MovementPath

        window_focuser.focus_game_window(monitor_idx=args.monitor)
        classifier = RoomClassifier(args.config, map_layout_path=args.map)
        localizer = MapLocalizer(reference_map_path=args.map, config_path=args.config)
        movement_path = MovementPath(movement_file_path=args.movement_file, config_path=args.config)
        capturer = ScreenCapturer(monitor_idx=args.monitor)
        visualizer = PlayerTrackerVisualizer(
            classifier=classifier,
            localizer=localizer,
            movement_path=movement_path,
            capturer=capturer,
            monitor_idx=args.monitor,
            start_pink_dot=args.pink_dot,
        )
        refresh_rate = args.interval if args.interval < 0.5 else 0.04
        print("\n" + "=" * 70)
        print(" PLAYER POSITION TRACKER & CLOSED-LOOP ROUTE NAVIGATOR")
        print(f" Target Display:  Monitor {args.monitor} (Path of Exile 2)")
        print(" Dual Monitoring:")
        print("   - Left:  Where it is in REAL (Live game minimap + orange icon)")
        print("   - Right: Where BOT THINKS it is (Room layout + route waypoints + trajectory)")
        print(" Controls:")
        print("   [A / G] Toggle Autonomous Navigation (WASD along route)")
        print("   [R]     Refresh Route from route.png (or click UI button)")
        print("   [V]     Cycle Map View: Active Room Template <-> World Map <-> Ref Map")
        print("   [F1]    Emergency STOP (halts all movement immediately)")
        print("   [Q/Esc] Exit Visualizer")
        print("=" * 70 + "\n")
        try:
            visualizer.run(interval=refresh_rate)
        except KeyboardInterrupt:
            print("\nStopping Player Position Tracker.")
        return

    # 4. Sequence Routine Execution Mode
    if args.routine:
        capturer = ScreenCapturer(monitor_idx=args.monitor)
        executor = SequenceExecutor(capturer=capturer, monitor_idx=args.monitor)
        executor.execute_routine(args.routine)
        return

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    classifier = RoomClassifier(args.config)

    # 3. Continuous Live Monitoring Mode (Global Map & Event Engine)
    if args.live:
        window_focuser.focus_game_window()
        monitors = ScreenCapturer.list_monitors()
        target_desc = f"Monitor {args.monitor}"
        for m in monitors:
            if m["index"] == args.monitor:
                target_desc = m["description"]

        print("=" * 70)
        print(" STARTING LIVE GLOBAL MAP STITCHING & EVENT-DRIVEN TASK ENGINE")
        print(f" Target Display:   {target_desc}")
        print(f" Polling Interval: {args.interval}s")
        print(f" Configured Rooms: {len(classifier.rooms)}")
        print(" Press Ctrl+C to stop.")
        print("=" * 70)

        capturer = ScreenCapturer(monitor_idx=args.monitor)
        world_map = WorldMapTracker()
        event_manager = GameEventManager()
        task_executor = TaskExecutor(event_manager)

        tracker = LiveRoomTracker(
            classifier=classifier,
            capturer=capturer,
            world_map=world_map,
            event_manager=event_manager,
            task_executor=task_executor,
            interval=args.interval,
            on_room_change=on_room_changed,
            on_tick=on_tick,
        )

        try:
            tracker.start(max_ticks=args.max_ticks)
        except KeyboardInterrupt:
            print("\nStopping Live Monitor.")
        finally:
            map_path = world_map.save_map("debug_output/world_map.png")
            print(f"\nSaved stitched global world map to: '{map_path}'")
            print(f"Total tasks executed: {task_executor.executed_tasks_count}")
            capturer.close()
        return

    # 4. Single Screenshot Analysis Mode
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input screenshot file not found: {args.input}")
            sys.exit(1)

        print(f"Analyzing screenshot: {args.input}...")
        result = classifier.classify(args.input)
        world_map = WorldMapTracker()
        map_res = world_map.update(cv2.imread(args.input))

        print("\n" + "=" * 60)
        print(" RECOGNITION & GLOBAL MAP RESULT")
        print("=" * 60)
        print(f"Status:             {'SUCCESS' if result['recognized'] else 'UNRECOGNIZED / UNKNOWN'}")
        print(f"Current Room:       {result['room_name']} ({result['room_id'] or 'N/A'})")
        print(f"Matched Variant:    {result['matched_variant'] or 'N/A'}")
        print(f"Global Position:    {map_res['global_position']} (G_X, G_Y on World Canvas)")
        print(f"Confidence:         {result['confidence']:.2%}")
        print("=" * 60)

        map_path = world_map.save_map("debug_output/world_map.png")
        print(f"\nSaved stitched global world map to: '{map_path}'")
        return

    # Default Help
    print("Game Bot Engine: Global Map & Sequence Routine Engine")
    print(f"Configured Rooms: {len(classifier.rooms)}")
    print("\nUsage:")
    print("  Live Player Position Tracker: python main.py --track [--monitor 2]")
    print("  Execute Navigation Routine:   python main.py --routine routines/trail_of_suffering.json --monitor 2")
    print("  List monitors:                python main.py --list-monitors")
    print("  Live map mode:                python main.py --live --monitor 2 [--interval 0.5]")
    print("  Single screenshot analysis:   python main.py --input screenshot.png")


if __name__ == "__main__":
    main()
