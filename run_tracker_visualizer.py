"""
Convenience launcher script for Voices Visualizer & Bounding Box Editor
Usage:
    python run_tracker_visualizer.py
    python run_tracker_visualizer.py --monitor 2
"""

import sys
from main import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--track")
    elif "--track" not in sys.argv and "-t" not in sys.argv and "-m" not in sys.argv and "--monitor" not in sys.argv:
        sys.argv.append("--track")
    main()
