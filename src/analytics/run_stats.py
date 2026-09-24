"""
Run statistics recording, room clearance tracking, run finalization, and history JSON persistence.
"""
from __future__ import annotations

import os
import time
import math
import json
import threading
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np

from src.movement_path import MovementPath
from src.screen_capturer import ScreenCapturer
from src.loot_detector import LootDetector, LootItem
from src.minimap_extractor import MinimapExtractor
from src.enemy_detector import EnemyDetector
from src.navigator_base import DynamicModuleProxy, log_msg

_log = log_msg
pydirectinput = DynamicModuleProxy("pydirectinput")
window_focuser = DynamicModuleProxy("window_focuser")
stop_handler = DynamicModuleProxy("stop_handler")
keyboard = DynamicModuleProxy("keyboard")


class RunStatsMixin:
    """Run statistics recording, room clearance tracking, run finalization, and history JSON persistence."""

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Formats seconds into readable 'Xm Y.Ys' or 'X.Xs' string."""
        if seconds < 0:
            seconds = 0.0
        m = int(seconds // 60)
        s = seconds % 60
        if m > 0:
            return f"{m}m {s:04.1f}s" if s < 10 else f"{m}m {s:.1f}s"
        return f"{s:.1f}s"


    def _start_new_run(self):
        """Initializes tracking metrics for a new bot run session."""
        now = time.time()
        run_ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{run_ts_str}"
        self.run_start_time = now
        self.run_end_time = None
        self.run_pause_time = None
        self.total_paused_duration = 0.0
        self.run_completed = False
        self.run_last_room_cleared = None
        self.run_rooms_cleared = []
        self.run_sims_clicked = []
        self.run_loot_picked = []
        self._last_clicked_loot_item = None
        self.current_room_key = None
        _log(f"[RUN TIMER] Run session '{self.run_id}' started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


    def _record_sim_clicked(self, sim_key: str, room_key: Optional[str] = None):
        """Records a pressed SIM into the active run session."""
        now = time.time()
        elapsed = round(now - self.run_start_time, 2) if self.run_start_time else 0.0
        room = room_key or getattr(self, "current_room_key", None) or "unknown"
        sim_entry = {
            "sim": sim_key,
            "room": room,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
        }
        self.run_sims_clicked.append(sim_entry)
        _log(f"  [RUN STATS] Recorded SIM pressed: {sim_key} in {room} (Run Elapsed: {self._format_duration(elapsed)})")


    def _record_loot_picked(self, item: Any, pos: Tuple[int, int], room_key: Optional[str] = None) -> Dict[str, Any]:
        """Records a picked up loot item into the active run session."""
        now = time.time()
        elapsed = round(now - self.run_start_time, 2) if self.run_start_time else 0.0
        room = room_key or getattr(self, "current_room_key", None) or "unknown"

        name = getattr(item, "rule_name", None) or "Unknown Loot"
        rule_id = getattr(item, "rule_id", None) or "unknown_rule"
        priority = getattr(item, "priority", 99) if item else 99
        conf = getattr(item, "confidence", 1.0) if item else 1.0
        box = getattr(item, "rect", (pos[0], pos[1], 0, 0)) if item else (pos[0], pos[1], 0, 0)

        loot_entry = {
            "name": name,
            "rule_id": rule_id,
            "room": room,
            "priority": priority,
            "confidence": round(float(conf), 3),
            "screen_pos": [int(pos[0]), int(pos[1])],
            "box": list(box) if isinstance(box, (list, tuple)) else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "elapsed_formatted": self._format_duration(elapsed),
        }
        self.run_loot_picked.append(loot_entry)
        _log(f"  [RUN STATS] Recorded Loot collected: '{name}' in {room} (Run Elapsed: {self._format_duration(elapsed)})")
        return loot_entry


    def _record_room_cleared(self, room_key: str, routine_duration: float):
        """Records a completed room into the active run session."""
        now = time.time()
        elapsed = round(now - self.run_start_time, 2) if self.run_start_time else 0.0
        sims_in_room = [s["sim"] for s in self.run_sims_clicked if s.get("room") == room_key]
        loot_in_room = [l for l in self.run_loot_picked if l.get("room") == room_key]
        loot_names = [l["name"] for l in loot_in_room]
        room_entry = {
            "room": room_key,
            "cleared_at_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cleared_at_elapsed_sec": elapsed,
            "cleared_at_elapsed_formatted": self._format_duration(elapsed),
            "routine_duration_sec": round(routine_duration, 2),
            "routine_duration_formatted": self._format_duration(routine_duration),
            "sims_clicked": list(sims_in_room),
            "loot_collected": list(loot_names),
            "loot_count": len(loot_names),
            "loot_detail": list(loot_in_room),
        }
        existing = [r for r in self.run_rooms_cleared if r.get("room") == room_key]
        if not existing:
            self.run_rooms_cleared.append(room_entry)
        self.run_last_room_cleared = room_key
        loot_note = f", Loot: {len(loot_names)} items" if loot_names else ""
        _log(f"[RUN STATS] Room '{room_key}' cleared! (Split: {self._format_duration(elapsed)}, Routine: {routine_duration:.1f}s, SIMs: {sims_in_room or 'None'}{loot_note})")


    def _get_last_pink_room_key(self) -> str:
        """Determines the configured final pink room key (e.g. 'pink_7')."""
        import re
        if self.zone_routines and "pink_zones" in self.zone_routines:
            pz = self.zone_routines["pink_zones"]
            max_num = -1
            max_key = None
            for k in pz.keys():
                m = re.search(r'(\d+)', k)
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
                        max_key = k
            if max_key:
                return max_key

        if hasattr(self.movement_path, "waypoints"):
            max_num = -1
            for wp in self.movement_path.waypoints:
                name = str(wp.get("name", ""))
                m = re.search(r'pink[_\s]*(\d+)', name, re.IGNORECASE)
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
            if max_num > 0:
                return f"pink_{max_num}"

        return "pink_7"


    def _is_last_room(self, room_key: str) -> bool:
        """Checks whether the given room_key represents the last room to be cleared."""
        if not room_key:
            return False
        last_key = self._get_last_pink_room_key()
        if room_key.lower() == last_key.lower():
            return True
        import re
        m1 = re.search(r'(\d+)', room_key)
        m2 = re.search(r'(\d+)', last_key)
        if m1 and m2 and m1.group(1) == m2.group(1):
            return True
        return False


    def _resolve_room_key(
        self,
        routine: Dict[str, Any],
        zone_label: str = "",
        target: Optional[Dict[str, Any]] = None,
        zone: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolves canonical room key (e.g. 'pink_1', ..., 'pink_7') for a routine."""
        import re
        # 1. Match against configured pink_zones by identity
        if self.zone_routines and "pink_zones" in self.zone_routines:
            pz = self.zone_routines["pink_zones"]
            for k, v in pz.items():
                if v is routine:
                    return k

        # 2. Check routine name (e.g. 'Pink Dot #7 (WP #74 - Room 7)')
        r_name = str(routine.get("name", ""))
        m_r = re.search(r'pink(?:[_\s]*dot)?[_\s]*#?(\d+)', r_name, re.IGNORECASE)
        if m_r:
            return f"pink_{m_r.group(1)}"
        m_room = re.search(r'room[_\s]*(\d+)', r_name, re.IGNORECASE)
        if m_room:
            return f"pink_{m_room.group(1)}"

        # 3. Check target name or pink_id
        if target and isinstance(target, dict):
            t_name = str(target.get("name", ""))
            m_t = re.search(r'pink[_\s]*(\d+)', t_name, re.IGNORECASE)
            if m_t:
                return f"pink_{m_t.group(1)}"
            pid = str(target.get("pink_id", ""))
            m_pid = re.search(r'(\d+)', pid)
            if m_pid:
                return f"pink_{m_pid.group(1)}"

        # 4. Check zone_label
        m_zl = re.search(r'(\d+)', zone_label)
        if m_zl and "pink" in zone_label.lower():
            return f"pink_{m_zl.group(1)}"

        # 5. Check orbit_zone
        if zone and isinstance(zone, dict):
            z_id = str(zone.get("id", ""))
            m_z = re.search(r'(\d+)', z_id)
            if m_z:
                return f"zone_{m_z.group(1)}"

        # 6. Fallback target waypoint index
        if target and isinstance(target, dict):
            idx = target.get("index")
            if idx is not None:
                return f"wp_{idx}"

        return zone_label or "unknown_room"


    def _finalize_run(self, last_room: Optional[str] = None, reason: str = "last_room_cleared") -> Dict[str, Any]:
        """Finalizes the current bot run, computes metrics, and writes summary files to disk."""
        if self.run_start_time is None:
            return {}
        if self.run_completed:
            return getattr(self, "_last_completed_run_data", {})

        now = time.time()
        self.run_end_time = now
        self.run_completed = True
        if last_room:
            self.run_last_room_cleared = last_room
        elif self.run_rooms_cleared:
            self.run_last_room_cleared = self.run_rooms_cleared[-1]["room"]

        wall_duration = max(0.0, self.run_end_time - self.run_start_time)
        active_duration = max(0.0, wall_duration - self.total_paused_duration)

        sims_list = [s["sim"] for s in self.run_sims_clicked]
        sims_by_room: Dict[str, List[str]] = {}
        for s in self.run_sims_clicked:
            r = s.get("room", "unknown")
            sims_by_room.setdefault(r, []).append(s["sim"])

        sim_counts: Dict[str, int] = {}
        for s in sims_list:
            sim_counts[s] = sim_counts.get(s, 0) + 1
        sim_counts["total"] = len(sims_list)

        loot_list = [l["name"] for l in self.run_loot_picked]
        loot_by_room: Dict[str, List[str]] = {}
        loot_by_room_detail: Dict[str, List[Dict[str, Any]]] = {}
        for l in self.run_loot_picked:
            r = l.get("room", "unknown")
            loot_by_room.setdefault(r, []).append(l["name"])
            loot_by_room_detail.setdefault(r, []).append(l)

        loot_counts: Dict[str, int] = {}
        for item_name in loot_list:
            loot_counts[item_name] = loot_counts.get(item_name, 0) + 1
        loot_counts["total"] = len(loot_list)

        rooms_list = [r["room"] for r in self.run_rooms_cleared]

        run_summary = {
            "run_id": self.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "completed": reason in ("last_room_cleared", "route_completed"),
            "completion_reason": reason,
            "start_time": datetime.fromtimestamp(self.run_start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.run_end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(wall_duration, 2),
            "duration_formatted": self._format_duration(wall_duration),
            "active_duration_seconds": round(active_duration, 2),
            "active_duration_formatted": self._format_duration(active_duration),
            "paused_seconds": round(self.total_paused_duration, 2),
            "last_room_cleared": self.run_last_room_cleared or "None",
            "total_rooms_cleared": len(self.run_rooms_cleared),
            "rooms_cleared": rooms_list,
            "rooms_detail": self.run_rooms_cleared,
            "sims_total_count": len(sims_list),
            "sims_counts": sim_counts,
            "sims_clicked_by_room": sims_by_room,
            "sims_log": self.run_sims_clicked,
            "total_loot_collected": len(loot_list),
            "loot_counts": loot_counts,
            "loot_by_room": loot_by_room,
            "loot_by_room_detail": loot_by_room_detail,
            "loot_log": self.run_loot_picked,
        }

        self._last_completed_run_data = run_summary

        self._save_run_to_history_json(run_summary)
        self._append_run_to_summary_txt(run_summary)
        self._save_last_run_json(run_summary)

        _log("\n" + "=" * 70)
        _log(f"[RUN COMPLETE] Duration: {run_summary['duration_formatted']} | Last Room: {run_summary['last_room_cleared']}")
        _log(f"  Rooms Cleared ({len(rooms_list)}): {', '.join(rooms_list)}")
        _log(f"  SIMs Clicked ({len(sims_list)} total): {sims_by_room}")
        _log(f"  Loot Collected ({len(loot_list)} total): {loot_by_room if loot_by_room else 'None'}")
        _log(f"  Saved run metrics to: {self.run_history_file} & {self.run_summary_file}")
        _log("=" * 70 + "\n")

        return run_summary


    def _save_run_to_history_json(self, run_summary: Dict[str, Any]):
        """Appends run summary to persistent JSON history file."""
        try:
            target_dir = os.path.dirname(self.run_history_file) or "."
            os.makedirs(target_dir, exist_ok=True)
            history = []
            if os.path.exists(self.run_history_file):
                try:
                    with open(self.run_history_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            history = data
                        elif isinstance(data, dict):
                            history = [data]
                except Exception as e:
                    _log(f"[RUN TIMER] Warning reading {self.run_history_file}: {e}")
            history.append(run_summary)
            with open(self.run_history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            _log(f"[RUN TIMER] Error saving history JSON: {e}")


    def _append_run_to_summary_txt(self, run_summary: Dict[str, Any]):
        """Appends a formatted human-readable summary block to text log file."""
        try:
            target_dir = os.path.dirname(self.run_summary_file) or "."
            os.makedirs(target_dir, exist_ok=True)
            entry_lines = [
                "=" * 78,
                f"RUN ID: {run_summary['run_id']} | {run_summary['start_time']} -> {run_summary['end_time']}",
                f"Status: {'COMPLETED' if run_summary['completed'] else 'PARTIAL'} ({run_summary['completion_reason']})",
                f"Total Duration: {run_summary['duration_formatted']} ({run_summary['duration_seconds']:.1f}s) | Active: {run_summary['active_duration_formatted']}",
                f"Last Room Cleared: {run_summary['last_room_cleared']}",
                f"Rooms Cleared ({run_summary['total_rooms_cleared']}): {', '.join(run_summary['rooms_cleared']) if run_summary['rooms_cleared'] else 'None'}",
                f"SIMs Pressed ({run_summary['sims_total_count']} total):",
            ]
            if run_summary['sims_clicked_by_room']:
                for room, sims in run_summary['sims_clicked_by_room'].items():
                    entry_lines.append(f"  - {room}: {', '.join(sims)}")
            else:
                entry_lines.append("  - (No SIMs pressed)")

            total_loot = run_summary.get('total_loot_collected', 0)
            entry_lines.append(f"Loot Collected ({total_loot} total):")
            if run_summary.get('loot_by_room'):
                for room, items in run_summary['loot_by_room'].items():
                    item_counts: Dict[str, int] = {}
                    for it in items:
                        item_counts[it] = item_counts.get(it, 0) + 1
                    item_strs = [f"{count}x {name}" if count > 1 else name for name, count in item_counts.items()]
                    entry_lines.append(f"  - {room} ({len(items)} items): {', '.join(item_strs)}")
            else:
                entry_lines.append("  - (No loot collected)")

            if run_summary.get('rooms_detail'):
                entry_lines.append("Room Splits:")
                for r in run_summary['rooms_detail']:
                    sim_note = f" [SIMs: {', '.join(r['sims_clicked'])}]" if r.get('sims_clicked') else ""
                    loot_count = r.get('loot_count', len(r.get('loot_collected', [])))
                    loot_note = f" [Loot: {loot_count} item{'s' if loot_count != 1 else ''}]" if loot_count > 0 else ""
                    entry_lines.append(f"  - {r['room']}: cleared at +{r['cleared_at_elapsed_formatted']} (routine: {r['routine_duration_sec']}s){sim_note}{loot_note}")
            entry_lines.append("=" * 78 + "\n")

            with open(self.run_summary_file, "a", encoding="utf-8") as f:
                f.write("\n".join(entry_lines) + "\n")
        except Exception as e:
            _log(f"[RUN TIMER] Error writing summary TXT: {e}")


    def _save_last_run_json(self, run_summary: Dict[str, Any]):
        """Saves latest run summary to debug_logs/last_run.json."""
        try:
            target_file = os.path.join(os.path.dirname(self.run_history_file) or "debug_logs", "last_run.json")
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(run_summary, f, indent=2)
        except Exception as e:
            _log(f"[RUN TIMER] Error saving last run JSON: {e}")

