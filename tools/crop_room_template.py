"""
Room Template Cropper Tool
Extracts minimap region from a game screenshot file or directly from a specified monitor,
supporting variant labels like 'entry' (fog-of-war) and 'full' (fully unrevealed).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from src.minimap_extractor import MinimapExtractor
from src.screen_capturer import ScreenCapturer


def main():
    parser = argparse.ArgumentParser(
        description="Crop minimap from a game screenshot or live monitor and save as room template variant."
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to full-screen game screenshot image.",
    )
    parser.add_argument(
        "--room-id",
        "-r",
        default="room_1",
        help="Room identifier (e.g. room_1, room_2). Default: room_1",
    )
    parser.add_argument(
        "--variant",
        "-v",
        default="",
        help="Optional variant label (e.g. 'entry' for entrance view, 'full' for revealed map).",
    )
    parser.add_argument(
        "--monitor",
        "-m",
        type=int,
        default=1,
        help="Monitor index to capture live if --input is not provided. Default: 1",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="templates",
        help="Directory to save room template. Default: templates",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Path to configuration JSON file. Default: config.json",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    extractor = MinimapExtractor()

    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input screenshot not found: {args.input}")
            sys.exit(1)
        image_source = args.input
        print(f"Cropping template from screenshot file: {args.input}")
        minimap_crop = extractor.extract_roi(image_source)
    else:
        print(f"Capturing live screenshot from Monitor {args.monitor}...")
        capturer = ScreenCapturer(monitor_idx=args.monitor)
        screenshot = capturer.capture()
        capturer.close()
        minimap_crop = extractor.extract_roi(screenshot)

    # Build filename: room_1_entry.png or room_1_full.png or room_1.png
    if args.variant:
        filename = f"{args.room_id}_{args.variant}.png"
    else:
        filename = f"{args.room_id}.png"

    output_path = os.path.join(args.output_dir, filename)
    cv2.imwrite(output_path, minimap_crop)

    print(f"Successfully saved room template variant: '{output_path}' ({minimap_crop.shape[1]}x{minimap_crop.shape[0]} px)")


if __name__ == "__main__":
    main()
