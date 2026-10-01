"""
Human-like mouse movement engine for Path of Exile 2.
Generates smooth cubic Bezier trajectories with minimum-jerk easing,
natural arc deflection, human micro-tremor, and +/- 20% randomized speed variation.
"""
from __future__ import annotations

import math
import random
import time
from typing import Tuple, List, Optional, Callable, Any

from src.navigator_base import DynamicModuleProxy, log_msg

_log = log_msg
pydirectinput = DynamicModuleProxy("pydirectinput")
window_focuser = DynamicModuleProxy("window_focuser")
stop_handler = DynamicModuleProxy("stop_handler")


def _is_stopped() -> bool:
    """Checks if emergency stop or navigation abort has been requested."""
    try:
        if stop_handler and hasattr(stop_handler, "is_stopped"):
            return bool(stop_handler.is_stopped())
    except Exception:
        pass
    try:
        from src.stop_handler import stop_handler as sh
        return bool(sh.is_stopped())
    except Exception:
        return False


def get_current_cursor_pos() -> Optional[Tuple[int, int]]:
    """Retrieves current physical cursor position on desktop."""
    try:
        import ctypes
        from ctypes import wintypes
        pt = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except Exception:
        pass
    try:
        import pyautogui
        p = pyautogui.position()
        return int(p[0]), int(p[1])
    except Exception:
        pass
    return None


def send_cursor_pos(x: int, y: int, sync_directinput: bool = False) -> None:
    """
    Dispatches cursor position to Windows via SetCursorPos and virtual desktop mouse_event.
    Optionally synchronizes DirectInput state without blocking pause.
    """
    try:
        import ctypes
        u32 = ctypes.windll.user32
        u32.SetCursorPos(int(x), int(y))
        v_left = u32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
        v_top = u32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
        v_width = u32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
        v_height = u32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
        if v_width > 0 and v_height > 0:
            norm_x = int(((x - v_left) * 65535) / v_width)
            norm_y = int(((y - v_top) * 65535) / v_height)
            u32.mouse_event(0x8000 | 0x4000 | 0x0001, norm_x, norm_y, 0, 0)
    except Exception:
        pass

    if sync_directinput:
        try:
            if pydirectinput:
                pydirectinput.moveTo(int(x), int(y), _pause=False)
        except Exception:
            pass


def calculate_duration(
    distance: float,
    speed_variance_pct: float = 20.0,
    base_speed_factor: float = 1.0,
) -> float:
    """
    Computes movement duration in seconds based on distance, modulated by randomized
    speed variance (+/- 20% by default).

    Models human hand/arm motor control:
      - Short moves (50px): ~0.08s - 0.12s
      - Medium moves (300px): ~0.15s - 0.22s
      - Long moves (800px): ~0.24s - 0.35s
    """
    if distance <= 2.0:
        return 0.0

    # Base duration according to human hand/arm motor Fitts-like model
    base_dur = 0.08 + ((distance / 1600.0) ** 0.75) * 0.35
    base_dur = base_dur / max(0.1, base_speed_factor)

    # Random speed +/- variance percentage (e.g. 20% -> 0.80 to 1.20)
    var_ratio = max(0.0, min(0.50, float(speed_variance_pct) / 100.0))
    speed_factor = random.uniform(1.0 - var_ratio, 1.0 + var_ratio)
    duration = base_dur / max(0.01, speed_factor)

    # Sensible bounds
    min_dur = 0.035 if distance < 35.0 else 0.065
    max_dur = 0.55
    return max(min_dur, min(max_dur, duration))


def generate_human_path(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    num_steps: int,
    clamp_fn: Optional[Callable[[int, int], Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Generates a list of (x, y) coordinates along a cubic Bezier curve with minimum-jerk
    speed parameterization, natural arc deviation, and subtle human micro-tremor.
    """
    x0, y0 = float(start_pos[0]), float(start_pos[1])
    x1, y1 = float(end_pos[0]), float(end_pos[1])

    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)

    if dist < 4.0 or num_steps <= 1:
        fx, fy = int(round(x1)), int(round(y1))
        if clamp_fn:
            fx, fy = clamp_fn(fx, fy)
        return [(fx, fy)]

    # Unit normal vector (perpendicular to movement direction)
    nx = -dy / dist
    ny = dx / dist

    # Arc direction: randomly curve left or right
    arc_sign = random.choice([-1.0, 1.0])
    # Proportional arc deviation: 5% to 15% of distance, capped at 85px
    max_dev = min(85.0, dist * random.uniform(0.05, 0.15))

    # Control Point 1 (~30% along the path)
    t1 = random.uniform(0.25, 0.38)
    dev1 = max_dev * arc_sign * random.uniform(0.8, 1.2)
    p1_x = x0 + dx * t1 + nx * dev1 + random.uniform(-2.0, 2.0)
    p1_y = y0 + dy * t1 + ny * dev1 + random.uniform(-2.0, 2.0)

    # Control Point 2 (~70% along the path)
    t2 = random.uniform(0.62, 0.78)
    dev2 = max_dev * arc_sign * random.uniform(0.5, 0.9)
    p2_x = x0 + dx * t2 + nx * dev2 + random.uniform(-2.0, 2.0)
    p2_y = y0 + dy * t2 + ny * dev2 + random.uniform(-2.0, 2.0)

    # If clamp function provided, clamp control points so arc doesn't stray outside window
    if clamp_fn:
        p1_cx, p1_cy = clamp_fn(int(round(p1_x)), int(round(p1_y)))
        p1_x, p1_y = float(p1_cx), float(p1_cy)
        p2_cx, p2_cy = clamp_fn(int(round(p2_x)), int(round(p2_y)))
        p2_x, p2_y = float(p2_cx), float(p2_cy)

    path: List[Tuple[int, int]] = []
    for step_i in range(1, num_steps + 1):
        tau = step_i / float(num_steps)
        # Minimum-Jerk easing: 10*tau^3 - 15*tau^4 + 6*tau^5
        u = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)

        # Cubic Bezier position
        bx = (1.0 - u)**3 * x0 + 3.0 * (1.0 - u)**2 * u * p1_x + 3.0 * (1.0 - u) * (u**2) * p2_x + (u**3) * x1
        by = (1.0 - u)**3 * y0 + 3.0 * (1.0 - u)**2 * u * p1_y + 3.0 * (1.0 - u) * (u**2) * p2_y + (u**3) * y1

        if step_i < num_steps:
            # Human micro-tremor (amplitude highest at mid-movement, strictly 0 at the end)
            tremor_amp = 0.5 * (1.0 - abs(tau - 0.5) * 2.0)
            bx += random.uniform(-tremor_amp, tremor_amp)
            by += random.uniform(-tremor_amp, tremor_amp)
        else:
            bx, by = x1, y1

        px, py = int(round(bx)), int(round(by))
        if clamp_fn:
            px, py = clamp_fn(px, py)
        path.append((px, py))

    return path


def human_move_to(
    target_x: int,
    target_y: int,
    clamp_fn: Optional[Callable[[int, int], Tuple[int, int]]] = None,
    speed_variance_pct: float = 20.0,
    base_speed_factor: float = 1.0,
    monitor_idx: int = 0,
    settle_delay: bool = True,
) -> Tuple[int, int]:
    """
    Smoothly moves the cursor to (target_x, target_y) using human-like kinematics.
    Moves along a gentle curved trajectory with minimum-jerk acceleration/deceleration
    and a randomized speed variation within +/- speed_variance_pct (default: +/- 20%).
    """
    if clamp_fn:
        target_x, target_y = clamp_fn(target_x, target_y)

    # Attach to input desktop if window_focuser is available
    try:
        if window_focuser and hasattr(window_focuser, "_attach_input_desktop"):
            window_focuser._attach_input_desktop()
    except Exception:
        pass

    start_pos = get_current_cursor_pos()
    if start_pos is None or start_pos == (0, 0):
        # Fallback: if cursor is at (0, 0) or uninitialized, start near target
        start_x, start_y = target_x, target_y
    else:
        start_x, start_y = start_pos
        if clamp_fn:
            # If cursor is completely outside game bounds, clamp starting point
            start_x, start_y = clamp_fn(start_x, start_y)

    dist = math.hypot(target_x - start_x, target_y - start_y)
    if dist < 4.0:
        send_cursor_pos(target_x, target_y, sync_directinput=True)
        return target_x, target_y

    duration = calculate_duration(
        dist,
        speed_variance_pct=speed_variance_pct,
        base_speed_factor=base_speed_factor,
    )
    # Approx 120-140 updates per second (7-9ms per step)
    step_interval = 0.008
    num_steps = max(6, int(round(duration / step_interval)))
    dt = duration / float(num_steps)

    path = generate_human_path((start_x, start_y), (target_x, target_y), num_steps, clamp_fn=clamp_fn)

    # Request high precision timer on Windows
    winmm = None
    try:
        import ctypes
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)
    except Exception:
        pass

    try:
        for idx, (px, py) in enumerate(path):
            if _is_stopped():
                break

            is_last = (idx == len(path) - 1)
            send_cursor_pos(px, py, sync_directinput=is_last)
            if not is_last:
                time.sleep(dt)

        # Ensure exact final coordinate
        send_cursor_pos(target_x, target_y, sync_directinput=True)

        if settle_delay and not _is_stopped():
            # Subtle physiological pause before next action (15-35ms)
            time.sleep(random.uniform(0.015, 0.035))

    finally:
        if winmm:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass

    return target_x, target_y
