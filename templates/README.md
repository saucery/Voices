# Room Templates Directory

This folder contains template images used to identify game rooms based on top-right minimap screenshots.

## Configured Rooms (Total: 7)

1. `room_1.png` - Room 1 (Generated from initial sample screenshot)
2. `room_2.png` - Room 2
3. `room_3.png` - Room 3
4. `room_4.png` - Room 4
5. `room_5.png` - Room 5
6. `room_6.png` - Room 6
7. `room_7.png` - Room 7

## How to add layout templates for Rooms 2 through 7:

Whenever your character enters a new room:
1. Take a full-screen screenshot of the game.
2. Run the helper cropping tool:
   ```bash
   python tools/crop_room_template.py --input path/to/room2_screenshot.png --room-id room_2
   ```
3. The tool will crop the top-right minimap region and save it directly as `templates/room_2.png`.

Alternatively, you can manually place PNG reference images inside this directory (`templates/room_X.png`).
