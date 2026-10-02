"""
Standalone Hideout Functionality Diagnostic & Test Runner

Tests the complete Hideout workflow:
1. Hideout Minimap layout detection (templates/ui/hideout_layout.png)
2. STASH label / chest detection & click (templates/ui/stash_label.png, stash_full.png)
3. Inventory & Stash window confirmation (templates/ui/inventory_title.png, inventory_close.png)
4. Inventory window cropping & screenshot archive (inventory_screenshots/)
5. 5x12 Inventory Grid scan & Ctrl+Click quick-deposit (excluding columns 10, 11, 12)
6. Post-stash Escape keypress to close all windows
7. Map Device location & click (templates/ui/map_device_label.png, map_device_full.png)
8. Simulacrum Icon & Map Circle detection (templates/ui/simulacrum_icon.png, simulacrum_medal.png, simulacrum_node_full.png)
9. Map Circle click & 'SIMULACRUM OF DELUSION' popup verification
10. Tier 15 map scan in columns 10..12 and transfer into Delusion popup

Usage:
    python tools/test_hideout.py                       # Interactive test menu
    python tools/test_hideout.py --live                # Full live test on Game Screen
    python tools/test_hideout.py --dry-run             # Test detection only (NO clicks/keys)
    python tools/test_hideout.py --deposit-only        # Skip stash click; deposit from open inventory
    python tools/test_hideout.py --map-device          # Locate and click Map Device
    python tools/test_hideout.py --simulacrum-map      # Detect Simulacrum icons & circle nodes on screen
    python tools/test_hideout.py --insert-map          # Locate Tier 15 map and transfer to popup
    python tools/test_hideout.py --full-cycle          # Run complete end-to-end hideout sequence
    python tools/test_hideout.py --input <image.png>   # Test on a saved screenshot
    python tools/test_hideout.py --monitor 2           # Specify target monitor
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure repository root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.route_navigator import RouteNavigator
from src.movement_path import MovementPath
from src.screen_capturer import ScreenCapturer
from src.window_focus import window_focuser


def annotate_diagnostic_screen(
    nav: RouteNavigator,
    screen: np.ndarray,
    report: dict,
    out_path: str = "debug_output/hideout_diagnostic.png",
) -> str:
    """
    Renders an annotated image showing detections:
    - Hideout minimap bounding box & match score
    - Stash detection reticle
    - Inventory window bounding box
    - 5x12 inventory grid with non-empty vs empty cells and excluded columns
    - Map Device click target
    - Simulacrum icons & map circle click targets (cyan circle reticle)
    - Tier 15 maps in columns 10..12
    - Delusion popup status
    """
    annotated = screen.copy()
    sh, sw = screen.shape[:2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. Minimap check
    if nav.hideout_layout_tpl is not None:
        lh, lw = nav.hideout_layout_tpl.shape[:2]
        mm_region = screen[: int(sh * 0.40), int(sw * 0.65) :]
        if mm_region.shape[0] >= lh and mm_region.shape[1] >= lw:
            res = cv2.matchTemplate(mm_region, nav.hideout_layout_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            rx = int(sw * 0.65) + max_loc[0]
            ry = max_loc[1]
            col = (0, 255, 120) if max_val >= nav.hideout_match_threshold else (0, 165, 255)
            cv2.rectangle(annotated, (rx, ry), (rx + lw, ry + lh), col, 2)
            cv2.putText(annotated, f"HIDEOUT MINIMAP ({max_val:.1%})", (rx, max(20, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 2. Stash detection
    stash_pos = report.get("stash_pos") or nav.locate_stash(screen=screen)
    if stash_pos:
        sx, sy = stash_pos[0], stash_pos[1]
        capt = nav._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= nav.monitor_idx < len(monitors):
                sx -= monitors[nav.monitor_idx].get("left", 0)
                sy -= monitors[nav.monitor_idx].get("top", 0)

        if 0 <= sx < sw and 0 <= sy < sh:
            cv2.circle(annotated, (sx, sy), 16, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(annotated, (sx, sy), (0, 255, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"STASH CLICK TARGET ({sx}, {sy})", (max(10, sx - 80), max(20, sy - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 3. Map Device detection
    map_dev_pos = nav.locate_map_device(screen=screen)
    if map_dev_pos:
        mdx, mdy = map_dev_pos[0], map_dev_pos[1]
        capt = nav._get_capturer()
        if getattr(capt, "_sct", None) and getattr(capt._sct, "monitors", None):
            monitors = capt._sct.monitors
            if 0 <= nav.monitor_idx < len(monitors):
                mdx -= monitors[nav.monitor_idx].get("left", 0)
                mdy -= monitors[nav.monitor_idx].get("top", 0)

        if 0 <= mdx < sw and 0 <= mdy < sh:
            cv2.circle(annotated, (mdx, mdy), 18, (255, 180, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(annotated, (mdx, mdy), (255, 180, 0), cv2.MARKER_DIAMOND, 30, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"MAP DEVICE TARGET ({mdx}, {mdy})", (max(10, mdx - 90), max(20, mdy - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)

    # 4. Inventory window & 5x12 grid overlay
    inv_info = nav.locate_inventory_window(screen=screen)
    if inv_info is not None:
        inv_x, inv_y = inv_info["inv_origin"]
        inv_w, inv_h = inv_info["inv_dim"]
        row_centers = inv_info["row_centers"]
        col_centers = inv_info["col_centers"]
        scale = inv_info["scale"]
        half_box = max(10, int(20 * scale))
        cv2.rectangle(annotated, (inv_x, inv_y), (min(sw, inv_x + inv_w), min(sh, inv_y + inv_h)), (0, 200, 255), 2)
        cv2.putText(annotated, f"INVENTORY ({inv_info['confidence']:.1%}, scale={scale:.2f}, {inv_info['row_method']})", (inv_x, max(20, inv_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        for r, cy in enumerate(row_centers):
            for c, cx in enumerate(col_centers):
                cx_i, cy_i = int(round(cx)), int(round(cy))
                x1, y1 = cx_i - half_box, cy_i - half_box
                x2, y2 = cx_i + half_box, cy_i + half_box

                if c >= 9:
                    # Excluded last 3 columns - check if Tier 15 map is present
                    patch_half = max(6, int(12 * scale))
                    patch = screen[cy_i - patch_half : cy_i + patch_half, cx_i - patch_half : cx_i + patch_half]
                    is_item = bool(patch.mean() > 16.0 or patch.max() > 40)
                    if is_item:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 220), 2)
                        cv2.putText(annotated, "MAP", (cx_i - 12, cy_i + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 220), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 60, 220), 1)
                        cv2.putText(annotated, "EX", (cx_i - 8, cy_i + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 255), 1, cv2.LINE_AA)
                else:
                    patch_half = max(6, int(12 * scale))
                    patch = screen[cy_i - patch_half : cy_i + patch_half, cx_i - patch_half : cx_i + patch_half]
                    is_item = bool(patch.mean() > 16.0 or patch.max() > 40)
                    if is_item:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 100), 2)
                        cv2.circle(annotated, (cx_i, cy_i), 3, (0, 255, 100), -1)
                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (100, 110, 120), 1)

    # 5. Green Completed Map Nodes & Simulacrum Map Node Detections
    green_nodes = nav.detect_green_completed_nodes(screen=screen)
    for g_idx, g in enumerate(green_nodes):
        gx, gy = g["center"]
        cv2.circle(annotated, (gx, gy), 8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(annotated, (gx, gy), 3, (0, 255, 0), -1)
        cv2.putText(annotated, "GREEN", (gx - 16, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA)

    sim_nodes = nav.detect_simulacrum_map_nodes(screen=screen)
    for idx, node in enumerate(sim_nodes):
        mx, my = node["screen_medal_pos"]
        cx, cy = node["screen_circle_pos"]
        s = node["scale"]
        conf = node["confidence"]
        is_acc = node.get("is_accessible", False)
        connected_greens = node.get("connected_greens", [])

        # Color: Green/Cyan for accessible, Orange/Red for inaccessible
        theme_color = (0, 255, 120) if is_acc else (0, 140, 255)

        # Draw lines to connected green nodes
        for cg in connected_greens:
            cgx, cgy = cg["green_pos"]
            cv2.line(annotated, (cx, cy), (cgx, cgy), (0, 255, 255), 2, cv2.LINE_AA)

        # Medal box
        mw, mh = int(32 * s), int(33 * s)
        cv2.rectangle(annotated, (mx - mw // 2, my - mh // 2), (mx + mw // 2, my + mh // 2), theme_color, 2)

        # Arrow down to map circle
        cv2.arrowedLine(annotated, (mx, my + mh // 2), (cx, cy - 8), theme_color, 1, tipLength=0.3)

        # Map circle target
        cv2.circle(annotated, (cx, cy), max(8, int(12 * s)), theme_color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 3, theme_color, -1)

        tag = "ACCESSIBLE (GREEN LINK)" if is_acc else "INACCESSIBLE (NO GREEN)"
        cv2.putText(
            annotated,
            f"SIM #{idx+1} ({conf:.1%}) [{tag}]",
            (cx - 30, cy + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            theme_color,
            1,
            cv2.LINE_AA,
        )

    # 6. Delusion Popup Check
    if nav.is_simulacrum_popup_visible(screen=screen):
        cv2.rectangle(annotated, (int(sw * 0.25), int(sh * 0.20)), (int(sw * 0.75), int(sh * 0.75)), (0, 255, 120), 3)
        cv2.putText(annotated, "SIMULACRUM OF DELUSION POPUP [OPEN]", (int(sw * 0.26), int(sh * 0.24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, annotated)
    return out_path


def print_report_card(report: dict, annotated_path: str = None):
    """Prints a styled terminal summary of hideout test results."""
    print("\n" + "=" * 68)
    print(" HIDEOUT FUNCTIONALITY TEST SUMMARY")
    print("=" * 68)
    dry = report.get("dry_run", False)
    mode_str = "DRY RUN (Scan Only)" if dry else "LIVE EXECUTION"
    print(f" Test Mode:              {mode_str}")

    # 1. Hideout Minimap
    ho_status = "[OK] Confirmed" if report.get("in_hideout") else "[X] Failed / Not Detected"
    print(f" 1. Hideout Minimap:     {ho_status}")

    # 2. Stash
    stash_pos = report.get("stash_pos")
    if stash_pos:
        clk_str = " (Clicked)" if report.get("stash_clicked") else (" (Dry Run)" if dry else "")
        stash_status = f"[OK] Located at {stash_pos}{clk_str}"
    else:
        stash_status = "[X] Not Found / Skipped"
    print(f" 2. Stash Chest/Label:   {stash_status}")

    # 3. Inventory Window
    inv_status = "[OK] Confirmed Open" if report.get("inventory_open") else "[X] Closed / Not Detected"
    print(f" 3. Inventory Window:    {inv_status}")

    # 4. Items Deposited
    det_c = report.get("items_detected", 0)
    stash_c = report.get("items_stashed", 0)
    if dry:
        dep_status = f"[OK] {det_c} non-empty item(s) detected (Dry run)"
    else:
        dep_status = f"[OK] {stash_c} item(s) deposited via Ctrl+Click"
    print(f" 4. Grid Deposit:        {dep_status}")

    # 5. Windows Closed (Escape)
    if "windows_closed" in report:
        esc_status = "[OK] Closed via Escape" if report.get("windows_closed") else "[X] Windows Still Open"
        print(f" 5. Windows Close (Esc): {esc_status}")

    # 6. Map Device
    if "map_device_clicked" in report:
        md_status = "[OK] Located & Clicked" if report.get("map_device_clicked") else "[X] Failed"
        print(f" 6. Map Device:          {md_status}")

    # 7. Simulacrum Map Selection
    if "simulacrum_selected" in report:
        sim_status = "[OK] Accessible Node Confirmed" if report.get("simulacrum_selected") else "[X] Not Found / Inaccessible"
        print(f" 7. Simulacrum Map Node: {sim_status}")

    # 8. Map Transferred to Popup
    if "map_transferred" in report:
        trans_status = "[OK] Tier 15 Map Inserted" if report.get("map_transferred") else "[X] Failed"
        print(f" 8. Map Transfer:        {trans_status}")

    # 9. TRAVERSE Button
    if "traverse_clicked" in report:
        trav_status = "[OK] TRAVERSE Clicked" if report.get("traverse_clicked") else "[X] Failed / Not Detected"
        print(f" 9. TRAVERSE Button:     {trav_status}")

    # 10. Hideout Portal
    if "portal_clicked" in report:
        portal_status = "[OK] Entered Simulacrum Danger Zone" if report.get("portal_clicked") else "[X] Failed / Not Detected"
        print(f" 10. Hideout Portal:     {portal_status}")

    overall = "PASSED" if report.get("success") else "FAILED"
    print("-" * 68)
    print(f" OVERALL RESULT:         >>> {overall} <<<")
    if annotated_path and os.path.exists(annotated_path):
        print(f" Diagnostic Image:       '{annotated_path}'")
    print("=" * 68 + "\n")


def run_interactive_menu(monitor: int):
    """Presents a clean interactive menu for testing hideout features."""
    print("=" * 68)
    print(" VOICES - HIDEOUT FUNCTIONALITY TESTER")
    print("=" * 68)
    print(f" Target Display: Monitor {monitor}")
    print(" Select test mode:")
    print("   [1] Full Live End-to-End Cycle (Stash -> Esc -> Map Device -> Sim Map -> Insert Map -> Traverse -> Portal)")
    print("   [2] Stash Deposit Only (Inventory open -> Quick-Deposit -> Escape)")
    print("   [3] Test Map Device Click (Locate & click Map Device in Hideout)")
    print("   [4] Test Simulacrum Map Node Detection (Scan Atlas for icon & circle)")
    print("   [5] Test Map Transfer (Columns 10..12 -> Simulacrum of Delusion popup)")
    print("   [6] Test TRAVERSE Button Click (Delusion popup -> Click TRAVERSE)")
    print("   [7] Test Hideout Portal Detection & Click (Spawned Map Device portal -> Enter danger zone)")
    print("   [8] Live Dry Run (Scan & diagnose elements on screen without clicks/keys)")
    print("   [9] Screenshot Diagnostic Mode (Analyze a saved screenshot file)")
    print("   [10] Interactive Simulacrum Node Finder & Feedback UI (Mark nodes, save templates, generate report)")
    print("   [Q] Quit")
    print("-" * 68)

    choice = input("Enter choice [1-10, Q]: ").strip().lower()
    if choice == "1":
        run_full_cycle_test(monitor=monitor, dry_run=False, traverse_and_enter=True)
    elif choice == "2":
        run_live_test(monitor=monitor, dry_run=False, deposit_only=True)
    elif choice == "3":
        run_map_device_test(monitor=monitor, dry_run=False)
    elif choice == "4":
        run_simulacrum_map_test(monitor=monitor, dry_run=False)
    elif choice == "5":
        run_map_insert_test(monitor=monitor, dry_run=False)
    elif choice == "6":
        run_traverse_button_test(monitor=monitor, dry_run=False)
    elif choice == "7":
        run_hideout_portal_test(monitor=monitor, dry_run=False)
    elif choice == "8":
        run_live_test(monitor=monitor, dry_run=True, deposit_only=False)
    elif choice == "9":
        img_path = input("Enter path to screenshot image: ").strip().strip('"').strip("'")
        if not os.path.exists(img_path):
            print(f"[ERROR] File not found: '{img_path}'")
            return
        run_file_test(image_path=img_path, monitor=monitor)
    elif choice == "10":
        import tkinter as tk
        from tools.simulacrum_feedback_ui import SimulacrumFeedbackUI
        root = tk.Tk()
        app = SimulacrumFeedbackUI(root, monitor_idx=monitor)
        root.mainloop()
    else:
        print("Exiting...")


def run_live_test(monitor: int, dry_run: bool = False, deposit_only: bool = False):
    """Executes hideout test live against game screen."""
    print(f"\n[TEST HIDEOUT] Initializing live test on Monitor {monitor}...")
    window_focuser.ensure_focused(monitor_idx=monitor)
    time.sleep(0.3)

    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    nav.in_hideout = True
    nav.is_active = False
    nav.disable_persistent_combat()
    capt = ScreenCapturer(monitor_idx=monitor)

    try:
        screen = capt.capture()
    except Exception as e:
        print(f"[ERROR] Failed to capture game screen: {e}")
        return

    report = nav.test_hideout_sequence(
        dry_run=dry_run,
        deposit_only=deposit_only,
        screen=screen,
        full_cycle=not deposit_only,
    )
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print_report_card(report, annotated_path=ann_path)
    nav.stop()


def run_full_cycle_test(monitor: int, dry_run: bool = False, traverse_and_enter: bool = True):
    """Runs the complete end-to-end hideout sequence."""
    print(f"\n[TEST HIDEOUT] Initializing Full End-to-End Cycle on Monitor {monitor} (traverse_and_enter={traverse_and_enter})...")
    window_focuser.ensure_focused(monitor_idx=monitor)
    time.sleep(0.3)

    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    nav.in_hideout = True
    nav.is_active = False
    nav.disable_persistent_combat()
    capt = ScreenCapturer(monitor_idx=monitor)

    try:
        screen = capt.capture()
    except Exception as e:
        print(f"[ERROR] Failed to capture game screen: {e}")
        return

    report = nav.run_hideout_full_cycle(
        dry_run=dry_run,
        screen=screen,
        traverse_and_enter=traverse_and_enter,
        auto_start_route=False if dry_run else True,
        start_pink_dot=1,
    )
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print_report_card(report, annotated_path=ann_path)
    nav.stop()


def run_traverse_button_test(monitor: int, dry_run: bool = False):
    """Tests locating and clicking the TRAVERSE button in the Delusion popup."""
    print(f"\n[TEST TRAVERSE] Testing TRAVERSE button interaction on Monitor {monitor}...")
    window_focuser.ensure_focused(monitor_idx=monitor)
    time.sleep(0.3)

    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    capt = ScreenCapturer(monitor_idx=monitor)
    screen = capt.capture()

    pos = nav.locate_traverse_button(screen=screen)
    if pos:
        print(f"[OK] Located TRAVERSE button at desktop ({pos[0]}, {pos[1]}).")
        ok = nav.click_traverse_button(dry_run=dry_run)
        print(f"[RESULT] TRAVERSE button click: {'SUCCESS' if ok else 'FAILED'}")
    else:
        print("[X] TRAVERSE button not detected on screen.")

    report = {"in_hideout": True, "traverse_clicked": pos is not None, "success": pos is not None}
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print(f"Diagnostic image saved to '{ann_path}'.")
    nav.stop()


def run_hideout_portal_test(monitor: int, dry_run: bool = False):
    """Tests locating and clicking spawned Map Device portal(s) in the hideout."""
    print(f"\n[TEST HIDEOUT PORTAL] Testing spawned portal interaction on Monitor {monitor}...")
    window_focuser.ensure_focused(monitor_idx=monitor)
    time.sleep(0.3)

    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    capt = ScreenCapturer(monitor_idx=monitor)
    screen = capt.capture()

    pos = nav.locate_hideout_portal(screen=screen)
    if pos:
        print(f"[OK] Located spawned portal at desktop ({pos[0]}, {pos[1]}).")
        ok = nav.click_hideout_portal(dry_run=dry_run, auto_start_route=False)
        print(f"[RESULT] Portal click: {'SUCCESS' if ok else 'FAILED'}")
    else:
        print("[X] Spawned portal not detected on screen.")

    report = {"in_hideout": True, "portal_clicked": pos is not None, "success": pos is not None}
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print(f"Diagnostic image saved to '{ann_path}'.")
    nav.stop()


def run_map_device_test(monitor: int, dry_run: bool = False):
    """Tests locating and clicking the Map Device."""
    print(f"\n[TEST MAP DEVICE] Testing Map Device interaction on Monitor {monitor}...")
    window_focuser.ensure_focused(monitor_idx=monitor)
    time.sleep(0.3)

    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    capt = ScreenCapturer(monitor_idx=monitor)
    screen = capt.capture()

    pos = nav.locate_map_device(screen=screen)
    if pos:
        print(f"[OK] Located Map Device at desktop ({pos[0]}, {pos[1]}).")
        if not dry_run:
            ok = nav.click_map_device()
            print(f"[RESULT] Map Device click: {'SUCCESS' if ok else 'FAILED'}")
    else:
        print("[X] Map Device not detected on screen.")

    report = {"in_hideout": True, "map_device_clicked": pos is not None, "success": pos is not None}
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print(f"Diagnostic image saved to '{ann_path}'.")
    nav.stop()


def run_simulacrum_map_test(monitor: int, dry_run: bool = False):
    """Tests detection of Simulacrum icons and circle click coordinates."""
    print(f"\n[TEST SIMULACRUM MAP] Scanning screen for Simulacrum nodes on Monitor {monitor}...")
    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    capt = ScreenCapturer(monitor_idx=monitor)
    screen = capt.capture()

    nodes = nav.detect_simulacrum_map_nodes(screen=screen)
    print(f"\nDetected {len(nodes)} Simulacrum map node(s):")
    for idx, n in enumerate(nodes):
        print(f"  #{idx+1}: Medal={n['screen_medal_pos']}, Circle Target={n['screen_circle_pos']}, Conf={n['confidence']:.3f}, Scale={n['scale']:.2f}")

    if nodes and not dry_run:
        acc = nav.select_accessible_simulacrum_map(candidates=nodes, screen=screen, dry_run=dry_run)
        print(f"Accessible map selection result: {'SUCCESS' if acc else 'FAILED'}")

    report = {"in_hideout": True, "simulacrum_selected": len(nodes) > 0, "success": len(nodes) > 0}
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print(f"Diagnostic image saved to '{ann_path}'.")
    nav.stop()


def run_map_insert_test(monitor: int, dry_run: bool = False):
    """Tests locating Tier 15 map in columns 10..12 and transferring to popup."""
    print(f"\n[TEST MAP INSERT] Testing Tier 15 map search and transfer on Monitor {monitor}...")
    window_focuser.ensure_focused(monitor_idx=monitor)
    time.sleep(0.3)
    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    capt = ScreenCapturer(monitor_idx=monitor)
    screen = capt.capture()

    maps = nav.locate_tier15_maps_in_inventory(screen=screen)
    print(f"\nFound {len(maps)} Tier 15 map(s) in columns 10..12:")
    for m in maps:
        print(f"  Row {m['row']+1}, Col {m['col']+1} at screen {m['screen_pos']} (conf={m['confidence']:.3f})")

    if maps:
        ok = nav.insert_map_into_simulacrum_popup(screen=screen, target_slot_idx=0, method="drag", dry_run=dry_run)
        print(f"Map transfer result: {'SUCCESS' if ok else 'FAILED'}")

    report = {"in_hideout": True, "map_transferred": len(maps) > 0, "success": len(maps) > 0}
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print(f"Diagnostic image saved to '{ann_path}'.")
    nav.stop()


def run_file_test(image_path: str, monitor: int):
    """Executes offline diagnostic analysis on a screenshot image."""
    print(f"\n[TEST HIDEOUT] Loading screenshot from '{image_path}'...")
    screen = cv2.imread(image_path)
    if screen is None:
        print(f"[ERROR] Could not decode image: '{image_path}'")
        return

    nav = RouteNavigator(movement_path=MovementPath(), monitor_idx=monitor)
    nav.in_hideout = True
    nav.is_active = False
    nav.disable_persistent_combat()
    report = nav.test_hideout_sequence(dry_run=True, deposit_only=False, screen=screen)
    ann_path = annotate_diagnostic_screen(nav, screen, report)
    print_report_card(report, annotated_path=ann_path)
    nav.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Hideout Functionality Diagnostic & Test Runner"
    )
    parser.add_argument(
        "--live",
        "-l",
        action="store_true",
        help="Run live test against the game screen.",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Run in dry-run mode (scans elements and reports items without sending clicks/keys).",
    )
    parser.add_argument(
        "--deposit-only",
        action="store_true",
        help="Skip stash click and test inventory screenshot + quick-deposit only.",
    )
    parser.add_argument(
        "--sim-ui",
        "--sim-inspector",
        action="store_true",
        help="Launch interactive Simulacrum Node Finder & Feedback Inspector UI.",
    )
    parser.add_argument(
        "--map-device",
        action="store_true",
        help="Test locating and clicking the Map Device.",
    )
    parser.add_argument(
        "--simulacrum-map",
        action="store_true",
        help="Test detection of Simulacrum icons and circle click coordinates.",
    )
    parser.add_argument(
        "--insert-map",
        action="store_true",
        help="Test locating Tier 15 map in columns 10..12 and transferring to popup.",
    )
    parser.add_argument(
        "--traverse",
        action="store_true",
        help="Test locating and clicking the TRAVERSE button in the Delusion popup.",
    )
    parser.add_argument(
        "--portal",
        action="store_true",
        help="Test locating and clicking spawned Map Device portal(s) in hideout.",
    )
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="Run complete end-to-end hideout sequence.",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to screenshot image for offline testing.",
    )
    parser.add_argument(
        "--monitor",
        "-m",
        type=int,
        default=2,
        help="Target monitor index for live capture (default: 2).",
    )

    args = parser.parse_args()

    if args.sim_ui:
        import tkinter as tk
        from tools.simulacrum_feedback_ui import SimulacrumFeedbackUI
        root = tk.Tk()
        app = SimulacrumFeedbackUI(root, initial_image=args.input, monitor_idx=args.monitor)
        root.mainloop()
    elif args.input:
        run_file_test(image_path=args.input, monitor=args.monitor)
    elif args.full_cycle:
        run_full_cycle_test(monitor=args.monitor, dry_run=args.dry_run, traverse_and_enter=True)
    elif args.map_device:
        run_map_device_test(monitor=args.monitor, dry_run=args.dry_run)
    elif args.simulacrum_map:
        run_simulacrum_map_test(monitor=args.monitor, dry_run=args.dry_run)
    elif args.insert_map:
        run_map_insert_test(monitor=args.monitor, dry_run=args.dry_run)
    elif args.traverse:
        run_traverse_button_test(monitor=args.monitor, dry_run=args.dry_run)
    elif args.portal:
        run_hideout_portal_test(monitor=args.monitor, dry_run=args.dry_run)
    elif args.live or args.dry_run or args.deposit_only:
        run_live_test(monitor=args.monitor, dry_run=args.dry_run, deposit_only=args.deposit_only)
    else:
        run_interactive_menu(monitor=args.monitor)


if __name__ == "__main__":
    main()
