# Voices - Path of Exile 2 Route Navigator & Autopilot

Voices is a real-time computer vision and autopilot navigation system for Path of Exile 2. It tracks player position via minimap localization, navigates custom-drawn routes, handles combat encounter banner activations, priority sim selections, and yellow zone area orbits with anti-stuck recovery and loot verification.

## Features

- **Real-Time Minimap Localization**: Fast template matching and edge-feature detection to locate the player's position on the full map in real time.
- **Painted Route Extraction**: Automatically parses painted navigation routes from `templates/route.png`:
  - **Blue Dot**: Route start
  - **Green Line**: Walking path with sampled waypoints
  - **Yellow Zones**: Orbit areas with perimeter trajectory generation and auto-attack
  - **Pink Dots**: Interactive encounter banners / sims
  - **Red Dot**: Route finish
- **Interactive UI Dashboard**:
  - Live minimap crop vs reference map localization
  - Real-time waypoint tracking and telemetry display
  - Interactive buttons for **[A] Autopilot**, **[P] Next Pink Point**, **[R] Reload Route**, and **Green Light Loot Confirmation**
- **Encounter Sequence Handling**:
  - Configurable 2.5s stop window on pink encounters
  - Priority sim selection (`sim1` -> `sim3` -> `sim2`) with click Y-offsets
  - Approach-wait logic to ensure the character reaches banners/sims before executing combat actions
  - Middle-mouse button hold and right-click combat rotations
  - Loot detection and manual/automatic green-light verification
- **Robust Anti-Stuck System**:
  - Active both on green travel paths and during yellow zone orbits
  - Multi-direction unstuck movements with key backtracking
- **Hotkeys**:
  - `F1`: Global emergency stop
  - `F4`: Pause / resume autopilot without losing waypoint progress
  - `A` / `G`: Start / stop autopilot from visualizer
  - `P`: Cycle Next Pink Point target
  - `R`: Reload route definition

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

To launch the real-time visualizer and navigator:

```bash
python main.py
```

## Testing

Run the test suite using pytest:

```bash
pytest tests/
```

## Configuration

Settings can be customized in `config.json`:
- `autopilot`: Timing thresholds, sim templates, encounter approach delays, loot confirmation.
- `rooms`: Room template configurations for map localization.
