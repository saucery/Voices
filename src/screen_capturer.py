"""
Screen Capturer Module
Provides fast multi-monitor screen capture capabilities for live game monitoring.
"""

from typing import Optional, Dict, List, Any
import numpy as np
import cv2

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

from PIL import ImageGrab


class ScreenCapturer:
    """Captures real-time screen imagery from specified monitor displays."""

    def __init__(self, monitor_idx: int = 1):
        """
        Initialize ScreenCapturer.

        :param monitor_idx: Index of monitor to capture (1 = primary, 2 = secondary, etc.).
        """
        self.monitor_idx = monitor_idx
        self._sct = mss.mss() if HAS_MSS else None

    @staticmethod
    def list_monitors() -> List[Dict[str, Any]]:
        """Returns a list of all detected display monitors with their indices and bounds."""
        monitors_info = []
        if HAS_MSS:
            with mss.mss() as sct:
                for idx, mon in enumerate(sct.monitors):
                    if idx == 0:
                        desc = f"Display {idx} [All Displays Combined]"
                    else:
                        desc = f"Display {idx} [{mon['width']}x{mon['height']} at ({mon['left']},{mon['top']})]"

                    monitors_info.append({
                        "index": idx,
                        "description": desc,
                        "width": mon["width"],
                        "height": mon["height"],
                        "left": mon["left"],
                        "top": mon["top"],
                    })
        else:
            monitors_info.append({
                "index": 1,
                "description": "Primary Display (PIL Fallback)",
                "width": 0,
                "height": 0,
                "left": 0,
                "top": 0,
            })
        return monitors_info

    def capture(self, bbox: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Captures a screenshot of the specified monitor or sub-region.

        :param bbox: Optional pixel region dict {'top': 0, 'left': 0, 'width': 800, 'height': 600}
        :return: OpenCV BGR image numpy array.
        """
        if HAS_MSS and self._sct is not None:
            if bbox:
                sct_img = self._sct.grab(bbox)
            else:
                monitors = self._sct.monitors
                if 0 <= self.monitor_idx < len(monitors):
                    mon = monitors[self.monitor_idx]
                else:
                    # Fallback to monitor 1 if index out of range
                    mon = monitors[1] if len(monitors) > 1 else monitors[0]
                sct_img = self._sct.grab(mon)

            # mss returns BGRA array -> convert to BGR
            img_np = np.array(sct_img)
            return cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        else:
            # Fallback to PIL ImageGrab
            if bbox:
                box = (bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])
                pil_img = ImageGrab.grab(bbox=box)
            else:
                pil_img = ImageGrab.grab()
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def close(self):
        """Clean up mss resources."""
        if self._sct:
            self._sct.close()
            self._sct = None
