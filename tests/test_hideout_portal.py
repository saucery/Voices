"""
Tests for Hideout Simulacrum of Delusion portal detection and interaction.
Verifies active portal detection (sign and body targets), completed portal rejection,
and multi-portal priority resolution.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.movement_path import MovementPath
from src.route_navigator import RouteNavigator


@pytest.fixture
def nav():
    n = RouteNavigator(movement_path=MovementPath())
    yield n
    n.stop()


def test_locate_active_hideout_portal_sign_and_body(nav):
    """Verifies that an active 'SIMULACRUM OF DELUSION' portal is located and returns sign or body pos."""
    u1_path = r"C:\Users\gregg\.gemini\antigravity-ide\brain\a81aaf84-9e2b-44f9-b1f1-721072691c12\.user_uploaded\media_1790325693185.png"
    if os.path.exists(u1_path):
        screen = cv2.imread(u1_path)
    else:
        # Synthetic active canvas
        screen = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        lbl = nav.hideout_portal_label_tpl
        screen[300:300 + lbl.shape[0], 500:500 + lbl.shape[1]] = lbl

    # 1. Target: sign (letters)
    pos_sign = nav.locate_hideout_portal(screen=screen, click_target="sign")
    assert pos_sign is not None
    assert isinstance(pos_sign, tuple) and len(pos_sign) == 2

    # 2. Target: body (blue portal rift)
    pos_body = nav.locate_hideout_portal(screen=screen, click_target="body")
    assert pos_body is not None
    assert isinstance(pos_body, tuple) and len(pos_body) == 2

    # Body Y should be positioned below sign Y
    assert pos_body[1] > pos_sign[1]


def test_rejects_completed_portal(nav):
    """Verifies that an old 'SIMULACRUM OF DELUSION (COMPLETED)' portal is rejected when allow_completed=False."""
    u2_path = r"C:\Users\gregg\.gemini\antigravity-ide\brain\a81aaf84-9e2b-44f9-b1f1-721072691c12\.user_uploaded\media_1790325693217.png"
    if os.path.exists(u2_path):
        screen = cv2.imread(u2_path)
    else:
        # Synthetic completed canvas
        screen = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        lbl = nav.hideout_portal_label_tpl
        comp = nav.hideout_portal_completed_tpl
        screen[300:300 + lbl.shape[0], 500:500 + lbl.shape[1]] = lbl
        screen[300:300 + comp.shape[0], 500 + lbl.shape[1] - 5:500 + lbl.shape[1] - 5 + comp.shape[1]] = comp

    # Must be skipped when allow_completed=False
    pos_rejected = nav.locate_hideout_portal(screen=screen, allow_completed=False)
    assert pos_rejected is None

    # Can be returned if explicitly allowed
    pos_allowed = nav.locate_hideout_portal(screen=screen, allow_completed=True)
    assert pos_allowed is not None


def test_multi_portal_prefers_active_over_completed(nav):
    """Verifies that when both active and completed portals exist on screen, the active one is selected."""
    canvas = np.full((1080, 1920, 3), 35, dtype=np.uint8)
    lbl = nav.hideout_portal_label_tpl
    comp = nav.hideout_portal_completed_tpl
    assert lbl is not None and comp is not None

    # Completed portal at x=300
    canvas[300:300 + lbl.shape[0], 300:300 + lbl.shape[1]] = lbl
    canvas[300:300 + comp.shape[0], 300 + lbl.shape[1] - 5:300 + lbl.shape[1] - 5 + comp.shape[1]] = comp

    # Active portal at x=1100
    canvas[300:300 + lbl.shape[0], 1100:1100 + lbl.shape[1]] = lbl

    pos = nav.locate_hideout_portal(screen=canvas, click_target="sign", allow_completed=False)
    assert pos is not None
    # Position must correspond to active portal at x ~ 1100 + lbl.shape[1] // 2
    expected_x = 1100 + lbl.shape[1] // 2
    assert abs(pos[0] - expected_x) < 15


def test_click_hideout_portal_dispatches_with_target(nav):
    """Verifies click_hideout_portal left-clicks the located portal and handles dry run."""
    with patch.object(nav, "locate_hideout_portal", return_value=(960, 540)) as mock_loc:
        ok = nav.click_hideout_portal(
            dry_run=True,
            auto_start_route=False,
            click_target="body",
            allow_completed=False,
        )
        assert ok is True
        mock_loc.assert_called_with(allow_completed=False, click_target="body")
