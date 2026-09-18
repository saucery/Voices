"""
Pytest Global Configuration & Hardware Input Mocking.
Prevents automated test suites from taking over mouse, keyboard, or window focus.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def prevent_hardware_takeover_during_tests(monkeypatch):
    """
    Globally intercepts and neuters direct OS inputs and window focus stealing
    during test execution so tests never type keys, click buttons, or steal focus from the user.
    """
    # 1. Neuter pydirectinput
    try:
        import pydirectinput
        for fn in [
            "keyDown", "keyUp", "press", "click", "rightClick", "middleClick",
            "mouseDown", "mouseUp", "moveTo", "moveRel", "move"
        ]:
            if hasattr(pydirectinput, fn):
                monkeypatch.setattr(pydirectinput, fn, MagicMock(return_value=None))
    except ImportError:
        pass

    # 2. Neuter pyautogui
    try:
        import pyautogui
        for fn in [
            "keyDown", "keyUp", "press", "click", "rightClick", "middleClick",
            "mouseDown", "mouseUp", "moveTo", "moveRel", "move"
        ]:
            if hasattr(pyautogui, fn):
                monkeypatch.setattr(pyautogui, fn, MagicMock(return_value=None))
    except ImportError:
        pass

    # 3. Neuter keyboard hotkeys and key sending
    try:
        import keyboard
        for fn in [
            "add_hotkey", "remove_hotkey", "press", "release", "send", "hook", "unhook_all"
        ]:
            if hasattr(keyboard, fn):
                monkeypatch.setattr(keyboard, fn, MagicMock(return_value=None))
    except ImportError:
        pass

    # 4. Neuter window focus stealing
    try:
        from src.window_focus import window_focuser
        monkeypatch.setattr(window_focuser, "focus_game_window", MagicMock(return_value=True))
        monkeypatch.setattr(window_focuser, "ensure_focused", MagicMock(return_value=True))
        monkeypatch.setattr(window_focuser, "bring_to_front", MagicMock(return_value=True))
    except Exception:
        pass
