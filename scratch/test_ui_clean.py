def is_ui_zone_clean(x, y, w, h, sw, sh):
    # If image is a crop (less than 1200 width or 720 height), do not apply full-screen UI heuristics
    if sw < 1200 or sh < 720:
        return False
    
    cx = x + w // 2
    cy = y + h // 2

    # 1. Top status bar / latency graph
    if y < 45 or cy < 45:
        return True

    # 2. Chat window area (Bottom-Left: x < 26% width and y > 58% height)
    if cx < int(sw * 0.26) and cy > int(sh * 0.58):
        return True

    # 3. Life / Flask globe (Bottom-Left: x < 240 and y > height - 240)
    if cx < 240 and cy > (sh - 240):
        return True

    # 4. Mana globe (Bottom-Right: x > width - 240 and y > height - 240)
    if cx > (sw - 240) and cy > (sh - 240):
        return True

    # 5. Bottom skill & flask action bar (y > height - 110)
    if cy > (sh - 110):
        return True

    # 6. Minimap & Quest tracker (Top-Right: x > width - 360 and y < 360)
    if cx > (sw - 360) and cy < 360:
        return True

    # 7. Top-Left Buff bar (x < 320 and y < 80)
    if cx < 320 and cy < 80:
        return True

    return False
