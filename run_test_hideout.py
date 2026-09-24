"""
Convenience launcher script for Voices Hideout Functionality Tester
Usage:
    python run_test_hideout.py
    python run_test_hideout.py --live
    python run_test_hideout.py --dry-run
    python run_test_hideout.py --deposit-only
    python run_test_hideout.py --monitor 2
"""

import sys
from tools.test_hideout import main

if __name__ == "__main__":
    main()
