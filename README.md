# Voices - Path of Exile 2 Route Navigator & Autopilot

Voices is a real-time computer vision and autopilot navigation system for Path of Exile 2. It tracks player position via minimap localization, navigates custom-drawn routes, handles combat encounter banner activations, priority sim selections, and yellow zone area orbits with anti-stuck recovery, inverse backtracking, and high-value loot filtering.

---

## 1. Quick Start & CLI Launchers

| Command | Description |
| :--- | :--- |
| `python main.py` | Launches the interactive **Visualizer & Autopilot Dashboard** (default). |
| `python run_tracker_visualizer.py` | Convenience shortcut launcher for the Visualizer. |
| `python main.py --pink-dot <N>` | Starts navigation directly targeting Pink Dot #`N` (e.g. `--pink-dot 4` or `--pink-dot 7`). |
| `python main.py --monitor 2` | Directs screen capture to Monitor 2 (or Monitor 1). Default: 2. |
| `python main.py --list-monitors` | Lists all detected display monitors and resolutions. |
| `python main.py --headless` | Runs navigation in console mode without rendering the OpenCV window. |
| `python main.py --loot-ui` | Launches the interactive **Loot Item Selector & Filter Builder UI**. |
| `python tools/loot_item_selector_ui.py` | Direct standalone launch for the Loot Item Selector UI tool. |
| `pytest tests/` | Runs the full automated test suite (131+ unit and integration tests). |

---

## 2. In-App Keybindings & Live Controls

| Key / UI Control | Action |
| :--- | :--- |
| **`F2`** | **Global Emergency Stop**: Immediately releases all pressed keys/mouse buttons and halts execution. |
| **`F4`** | **Pause / Resume Autopilot**: Pauses movement without losing current waypoint progress. |
| **`A`** (or UI Button) | **Toggle Autopilot**: Starts or stops autonomous WASD navigation. |
| **`P`** (or UI Button) | **Cycle Next Pink Dot**: Instantly shifts target to Pink Dot 1, 2, ..., 7. |
| **`R`** (or UI Button) | **Reload Route & Routines**: Re-reads `movement_route.json` and `zone_routines.json` live without restarting. |
| **Green Light Button** | Manually confirms completion of the loot phase when manual confirmation is enabled. |

---

## 3. Standalone Testing & Diagnostic Tools

### A. Interactive Loot Item Selector & Filter Builder UI
Launch the graphical loot selection tool to open any game screenshot, visually draw bounding boxes around multiple loot items, assign names & priority tiers, and save them directly into the bot's filter:

```bash
python main.py --loot-ui
# or
python tools/loot_item_selector_ui.py
```

- **Interactive Canvas**: Pan (right-click / middle-click drag), Zoom (mouse wheel), and draw boxes (left-click drag).
- **Multi-Item Selection**: Crop and name 5, 10, or more items on a single screenshot.
- **Rule Types**: Supports **Exact Visual Template Matching** (saved to `ui/loot_templates/`) or **Color Box Rules** with custom priority levels (P1 = Highest).
- **Correct Mismatches & False Positives**:
  - **Click directly on any green detected box** on the image to inspect which rule matched it.
  - **Tune Threshold Live**: Adjust the confidence slider to eliminate false positive background noise on the spot.
  - **Re-crop / Replace Template**: Draw a cleaner box around the actual item and click **"✏️ Replace Template with Current Box Crop"**.
  - **Delete or Disable**: Remove or turn off unwanted rules with 1 click.
- **Live In-App Testing**: Click **"Test Detection on Image"** to immediately verify that the filter matches your items with green highlighted bounding boxes.
- **One-Click Save**: Persists rules to `routines/loot_filter.json` without manually editing JSON files.

### B. High-Speed Loot Detector CLI Tester
Test your loot filter rules on saved game screenshots without launching the game client:

```bash
# Test against a specific game screenshot
python -m src.loot_detector "path/to/screenshot.png"
```

- **Output**: Prints detected items, priority tiers (`[P1] Tier 1`, `[P2] Gems`, `[P3] Uniques`), coordinates, and match confidence.
- **Visual Preview**: Saves an annotated bounding box preview to `scratch/` highlighting the exact click target.

### B. Route Extraction & Map Template Tools
- **Re-extract Route from Painted PNG**:
  ```bash
  python tools/extract_route.py
  ```
  *(Parses `templates/route.png` and regenerates `paths/movement_route.json`)*.
- **Crop Room Template**:
  ```bash
  python tools/crop_room_template.py
  ```

### C. Live Diagnostic Log Folders
- `debug_logs/tracker_lost/`: Captures minimap crops (`.png`) and match telemetry (`.json`) whenever tracking lock is degraded or lost in Rooms 6/7.
- `loot_debug/`: Captures raw & annotated full-screen screenshots before picking up loot (`loot_pre_pickup_*_raw.png`), plus zoomed-in item crops and click targets.

---

## 4. Configuration Reference

### A. Per-Zone & Encounter Customization (`routines/zone_routines.json`)
You can customize the step-by-step routine for each room/zone individually (`pink_1` through `pink_7` and `yellow_1` through `yellow_7`):

```json
"pink_1": {
  "name": "Pink Dot #1 (WP #7 - Room 1)",
  "steps": [
    { "action": "stop", "duration": 2.5 },
    { "action": "navigate_to_sim_location", "timeout": 8.0, "arrival_threshold": 15.0 },
    { "action": "detect_and_click_sims", "priority": ["sim1", "sim3", "sim2"], "approach_wait": 2.0 },
    { "action": "navigate_to_pink_location", "timeout": 8.0 },
    { "action": "click_encounter_banner", "approach_wait": 2.0, "search_attempts": 5 },
    { "action": "click_mouse", "button": "right", "clicks": 1 },
    { "action": "hold_mouse", "button": "middle", "duration": 5, "combat_key": "t", "combat_interval": 0.65 },
    { "action": "orbit_yellow_zone", "duration": 50.0, "mode": "inside" },
    { "action": "detect_and_pickup_loot", "max_pickups": 20 }
  ]
}
```

#### Supported Step Actions:
- **`stop`**: Duration (`duration` in seconds) to halt movement.
- **`navigate_to_sim_location` / `navigate_to_pink_location`**: Moves character to the cyan SIM dot or pink banner dot.
- **`detect_and_click_sims`**: `priority` (e.g. `["sim1", "sim3", "sim2"]`), `click_y_offset`, `approach_wait`.
- **`click_encounter_banner`**: `search_attempts`, `approach_wait`, `reclick`.
- **`hold_mouse` / `click_mouse`**: `button` (`"right"`, `"middle"`, `"left"`), `duration`, `combat_key` (`"t"`), `combat_interval`.
- **`orbit_yellow_zone`**: `duration`, `mode` (`"inside"`, `"outside"`, `"boundary"`), `margin_px`.
- **`detect_and_pickup_loot`**: `max_pickups`, `approach_wait`, `pickup_delay`.

---

### B. Loot Filter Rules (`routines/loot_filter.json`)
Defines the visual detection rules and priorities for high-value items:

- **Tier 1 High Value** (Priority 1): White background rectangle with red text/border (Divine Orbs, Annulment, Mirror, Liquid Isolation).
- **Gems** (Priority 2): Olive background rectangle with yellow/lime text (Uncut & Cut Ruby, Sapphire, Diamond, Emerald, Topaz).
- **Uniques** (Priority 3): Purple background rectangle with white text (Raven's Reflection, valuable unique jewels).
- **Custom Template Matcher** (Priority 4): Exact template fallback images from `ui/`.

---

### C. Global Bot Settings (`config.json`)
- **`minimap_roi`**: Normalized screen percentage boundaries `[top, bottom, left, right]` for minimap cropping.
- **`matching`**: ORB feature count (`2500`), Canny edge thresholds, and template match thresholds.
- **`autopilot`**: Default fallback timings (orbit duration, combat intervals, sim templates).
- **`rooms`**: List of room IDs and confidence thresholds.

---

### D. Room Bounding Boxes (`paths/room_bounding_boxes.json`)
Defines `[x1, y1, x2, y2]` coordinates of each room on the reference map. Used by the localizer to restrict searches and isolate chamber detection.

---

## 5. Installation & Setup

```bash
pip install -r requirements.txt
```
