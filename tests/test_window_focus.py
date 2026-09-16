"""
Unit Tests for Game Window Focus Handler Module
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.window_focus import WindowFocusHandler, window_focuser


def test_window_focus_initialization():
    handler = WindowFocusHandler(target_title="Path of Exile 2")
    assert handler.target_title == "Path of Exile 2"


def test_window_focus_find():
    handler = WindowFocusHandler(target_title="Path of Exile 2")
    # Search game window or fallback
    win = handler.find_game_window()
    if win:
        assert "Path of Exile" in win.title
        for ignored in handler.ignored_substrings:
            assert ignored not in win.title.lower()


def test_window_focus_find_game_hwnd():
    handler = WindowFocusHandler(target_title="Path of Exile 2")
    hwnd, title = handler.find_game_hwnd()
    if hwnd:
        assert isinstance(hwnd, int)
        assert hwnd > 0
        if title:
            assert any(t.lower() in title.lower() for t in handler.target_titles)


def test_window_focus_blacklist():
    handler = WindowFocusHandler(target_title="Path of Exile 2")
    # Simulate a fake visualizer or browser window
    for ignored in ["Voices Visualizer", "Chrome - Path of Exile Wiki"]:
        low_t = ignored.lower()
        has_match = any(t.lower() in low_t for t in handler.target_titles)
        is_ignored = any(b in low_t for b in handler.ignored_substrings)
        # Should either not match target or be flagged as ignored
        assert not has_match or is_ignored

