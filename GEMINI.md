# Voices Project Rules & Architecture

## Core Development Rules
- **File Length Limit**: Every Python code file must remain strictly under 600–700 lines. Never allow monolithic growth.
- **RouteNavigator Architecture**: Assembled via multiple inheritance (mixins) in `src/route_navigator.py`. Shared state lives on `src/navigator_base.py`.
- **Testing**: Run `pytest tests/` after modifying modules to ensure backward compatibility and no syntax or regression issues.

## Modular Responsibility Map

### `src/hideout/` (Hideout & Preparation)
- `stash_handler.py`: Stash opening, affinity dumping, tab switching, deposit routines.
- `inventory_handler.py`: Inventory grid scanning, item transfers, slot interactions.
- `map_device.py`: Map device interaction, map/fragment insertion, activation button clicking.
- `simulacrum_selector.py`: Simulacrum wave/tier UI navigation and selection.
- `map_traverse.py`: Hideout pathfinding to map device and portals.
- `hideout_manager.py`: High-level hideout lifecycle coordination.

### `src/combat/` (Zone, Combat & Encounter Actions)
- `loot_collector.py`: Loot detection, drop filtering, loot clicking/pickup.
- `delirium_statue.py`: Delirium mirror/statue detection and interaction.
- `persistent_combat.py`: Continuous combat loop and enemy engagement logic.
- `zone_routines.py`: Zone lifecycle and sequence setup.
- `zone_step_executor.py`: Individual step execution within zone routines.
- `zone_actions.py`: Skill execution, buffs, flasks/potions.
- `sim_clicker.py`: Simulacrum wave starter clicking.
- `zone_interactions.py`: Zone transitions, portals, doors, interactive chest objects.

### `src/navigation/` (Movement & Wayfinding)
- `wasd_mover.py`: Direct WASD movement and directional vector execution.
- `orbit_controller.py`: Orbiting around waypoints and points of interest.
- `stuck_recovery.py`: Unsticking routines, movement skill jumps, jittering.
- `navigator_controls.py`: Movement states, mouse click pathing.
- `template_loader.py`: Minimap/screen template matching cache and loaders.
- `route_follower.py`: Pink dot waypoint navigation and path interpolation.

### `src/analytics/`
- `run_stats.py`: Run timing, loot metrics, kill counts, performance tracking.

### Core & Base
- `src/navigator_base.py`: Common base state, shared locks, and initialization attributes.
- `src/route_navigator.py`: Master assembly class inheriting all mixins (~470 lines).
