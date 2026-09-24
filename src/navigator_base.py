"""
Navigator Base Module
Shared dynamic proxies and logging utilities for RouteNavigator mixins.
Ensures unittest.mock patches on 'src.route_navigator.*' dynamically resolve across all submodules.
"""
import sys
from typing import Any

class DynamicModuleProxy:
    """Proxies attribute lookups to sys.modules['src.route_navigator'] dynamically."""
    def __init__(self, target_name: str):
        self._target_name = target_name

    def _get_target(self) -> Any:
        rn = sys.modules.get("src.route_navigator")
        if rn is not None and hasattr(rn, self._target_name):
            return getattr(rn, self._target_name)
        return None

    def __getattr__(self, item: str) -> Any:
        target = self._get_target()
        if target is None:
            raise AttributeError(f"'{self._target_name}' is None or not found in src.route_navigator, cannot access '{item}'")
        return getattr(target, item)

    def __bool__(self) -> bool:
        return bool(self._get_target())

    def __repr__(self) -> str:
        return repr(self._get_target())

def log_msg(msg: str = "") -> None:
    """Delegates logging to src.route_navigator._log."""
    rn = sys.modules.get("src.route_navigator")
    if rn is not None and hasattr(rn, "_log"):
        rn._log(msg)
    else:
        print(msg)
