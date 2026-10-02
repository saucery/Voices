"""
Interactive Simulacrum Map Node Finder & Feedback Inspector UI.

Allows users to:
1. Initiate finding Simulacrum nodes process ONLY on live game screen or saved screenshots.
2. Find next available Simulacrum node and cycle target priority.
3. Visually inspect all detected nodes, accessibility states, and see which node the bot would select.
4. Mark nodes as Correct, False Positive, or Incorrect Accessibility.
5. Draw boxes around missed Simulacrum nodes and crop them as new templates (simulacrum_node_vX.png).
6. Generate structured diagnostic JSON reports and annotated review screenshots.
"""
from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from src.route_navigator import RouteNavigator
from src.screen_capturer import ScreenCapturer
from src.hideout.simulacrum_report import (
    export_simulacrum_feedback_report,
    save_cropped_simulacrum_template,
)


class SimulacrumFeedbackUI:
    """Desktop GUI for inspecting, reviewing, and training Simulacrum map node detection."""

    def __init__(self, root: tk.Tk, initial_image: Optional[str] = None, monitor_idx: int = 2):
        self.root = root
        self.root.title("Voices - Simulacrum Node Finder & Feedback Inspector")
        self.root.geometry("1360x860")
        self.root.minsize(1080, 680)

        self.monitor_idx = monitor_idx
        self.nav = RouteNavigator(monitor_idx=self.monitor_idx)

        self.cv_image: Optional[np.ndarray] = None
        self.pil_image: Optional[Image.Image] = None
        self.current_image_path: Optional[str] = None

        # Viewport zoom & pan
        self.scale: float = 1.0
        self.offset_x, self.offset_y = 0.0, 0.0
        self.pan_start_x, self.pan_start_y = 0, 0
        self.is_panning: bool = False

        # Node detection results & user feedback
        self.detected_nodes: List[Dict[str, Any]] = []
        self.selected_node_idx: Optional[int] = None
        self.bot_selected_node: Optional[Dict[str, Any]] = None
        self.available_candidate_indices: List[int] = []
        self.current_candidate_pointer: int = 0
        self.node_feedback: Dict[int, str] = {}

        # Manual box selection for missed nodes / new templates
        self.is_drawing_box: bool = False
        self.box_start_x, self.box_start_y = 0, 0
        self.box_curr_x, self.box_curr_y = 0, 0
        self.manual_box: Optional[Tuple[int, int, int, int]] = None
        self.node_thumb_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

        if initial_image and os.path.exists(initial_image):
            self.load_image_from_file(initial_image)
        else:
            test_img = r"C:\Users\gregg\.gemini\antigravity-ide\brain\829d893f-727f-4e73-bd1c-db41c3ee9e7e\.user_uploaded\media_1790872842765.jpg"
            if os.path.exists(test_img):
                self.load_image_from_file(test_img)

    def _build_ui(self):
        # 1. Top Controls Bar
        top_bar = ttk.Frame(self.root, padding=6)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_bar, text="📸 Capture Screen", command=self.capture_live_screen).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="📁 Open Image...", command=self.browse_screenshot).pack(side=tk.LEFT, padx=2)

        ttk.Label(top_bar, text="Mon:").pack(side=tk.LEFT, padx=(6, 1))
        self.mon_var = tk.IntVar(value=self.monitor_idx)
        mon_combo = ttk.Combobox(top_bar, textvariable=self.mon_var, values=[1, 2], width=3, state="readonly")
        mon_combo.pack(side=tk.LEFT, padx=1)
        mon_combo.bind("<<ComboboxSelected>>", lambda e: setattr(self, "monitor_idx", self.mon_var.get()))

        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(top_bar, text="Thresh:").pack(side=tk.LEFT, padx=1)
        self.thresh_var = tk.DoubleVar(value=0.78)
        self.thresh_lbl = ttk.Label(top_bar, text="0.78", width=4)
        self.thresh_lbl.pack(side=tk.LEFT, padx=1)
        scale_slider = ttk.Scale(top_bar, from_=0.60, to_=0.95, variable=self.thresh_var, length=100, command=self._on_thresh_slide)
        scale_slider.pack(side=tk.LEFT, padx=2)

        ttk.Button(top_bar, text="🔍 FIND NODES ONLY", command=self.detect_nodes).pack(side=tk.LEFT, padx=3)
        ttk.Button(top_bar, text="⏭ FIND NEXT AVAILABLE", command=self.select_next_available_node).pack(side=tk.LEFT, padx=3)

        # Zoom controls
        zoom_frame = ttk.Frame(top_bar)
        zoom_frame.pack(side=tk.RIGHT, padx=4)
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Fit View", command=self.fit_to_view).pack(side=tk.LEFT, padx=2)

        # 2. Main Body
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: Interactive Canvas
        canvas_frame = ttk.Frame(paned)
        paned.add(canvas_frame, weight=3)
        self.canvas = tk.Canvas(canvas_frame, bg="#1a1c22", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_up)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_down)
        self.canvas.bind("<B3-Motion>", self._on_canvas_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_right_up)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Right: Sidebar Panel
        sidebar = ttk.Frame(paned, padding=8, width=380)
        paned.add(sidebar, weight=1)

        # 2.1 Bot Selection Target Banner
        self.bot_pick_frame = ttk.LabelFrame(sidebar, text="🤖 Bot Autonomous Selection", padding=8)
        self.bot_pick_frame.pack(fill=tk.X, pady=(0, 6))

        self.lbl_bot_choice = ttk.Label(self.bot_pick_frame, text="Run detection to view which node the bot would select.", font=("Segoe UI", 9, "bold"), foreground="#00d2ff", wraplength=340)
        self.lbl_bot_choice.pack(fill=tk.X)

        pick_btn_frame = ttk.Frame(self.bot_pick_frame)
        pick_btn_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(pick_btn_frame, text="⏭ Next Available Node", command=self.select_next_available_node).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(pick_btn_frame, text="🖱 Test Click Target", command=self.test_click_current_target).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # 2.2 Selected Node Inspector & Feedback Actions
        insp_frame = ttk.LabelFrame(sidebar, text="🔍 Selected Node Inspector", padding=8)
        insp_frame.pack(fill=tk.X, pady=(0, 6))

        info_grid = ttk.Frame(insp_frame)
        info_grid.pack(fill=tk.X)
        self.lbl_node_info = ttk.Label(info_grid, text="Select a node on canvas or list below.", font=("Segoe UI", 9), wraplength=230)
        self.lbl_node_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_node_thumb = ttk.Label(info_grid, text="[Patch]", background="#121316", width=12, anchor="center")
        self.lbl_node_thumb.pack(side=tk.RIGHT, padx=4)

        ttk.Label(insp_frame, text="Your Feedback / Correction:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(6, 2))
        fb_btn_frame = ttk.Frame(insp_frame)
        fb_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(fb_btn_frame, text="✅ Correct (Acc)", command=lambda: self.set_feedback("correct_acc")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(fb_btn_frame, text="✅ Correct (Inacc)", command=lambda: self.set_feedback("correct_inacc")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        fb_btn_frame2 = ttk.Frame(insp_frame)
        fb_btn_frame2.pack(fill=tk.X, pady=2)
        ttk.Button(fb_btn_frame2, text="⚠ Wrong Accessibility", command=self.toggle_accessibility_feedback).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(fb_btn_frame2, text="❌ False Positive", command=lambda: self.set_feedback("false_pos")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # 2.3 Detected Nodes List
        list_frame = ttk.LabelFrame(sidebar, text="📋 Detected Simulacrum Nodes", padding=6)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        columns = ("id", "acc", "conf", "feedback")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=7, selectmode="browse")
        for col, heading, w in [("id", "Node", 65), ("acc", "Bot Acc", 80), ("conf", "Conf", 65), ("feedback", "Feedback", 120)]:
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tree_sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.config(yscrollcommand=tree_sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # 2.4 Actions: Save Template & Export Report
        action_frame = ttk.LabelFrame(sidebar, text="💾 Learning & Diagnostic Actions", padding=8)
        action_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(action_frame, text="✂ Crop Selection as New Template", command=self.save_new_template).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="📄 Generate Diagnostic Report & Images", command=self.export_report).pack(fill=tk.X, pady=2)

        # 3. Bottom Status Bar
        self.status_var = tk.StringVar(value="Ready. Click 'FIND NODES ONLY' or 'FIND NEXT AVAILABLE'.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4).pack(side=tk.BOTTOM, fill=tk.X)

    def _on_thresh_slide(self, val):
        self.thresh_lbl.config(text=f"{float(val):.2f}")

    def capture_live_screen(self):
        screen = ScreenCapturer(monitor_idx=self.monitor_idx).capture()
        if screen is None or screen.size == 0:
            messagebox.showerror("Capture Error", f"Failed to capture screen from Monitor {self.monitor_idx}.")
            return
        self.current_image_path = None
        self.set_cv_image(screen)
        self.status_var.set(f"Live screen captured from Monitor {self.monitor_idx}. Finding nodes...")
        self.detect_nodes()

    def browse_screenshot(self):
        path = filedialog.askopenfilename(title="Open Screenshot", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All", "*.*")])
        if path:
            self.load_image_from_file(path)

    def load_image_from_file(self, path: str):
        if not os.path.exists(path):
            return
        img = cv2.imread(path)
        if img is None:
            return
        self.current_image_path = path
        self.set_cv_image(img)
        self.status_var.set(f"Loaded '{os.path.basename(path)}'. Running detection...")
        self.detect_nodes()

    def set_cv_image(self, img: np.ndarray):
        self.cv_image = img
        self.pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.reset_zoom()

    def detect_nodes(self):
        """Runs Simulacrum node detection ONLY and identifies the bot's selection."""
        if self.cv_image is None:
            messagebox.showinfo("No Image", "Please capture a live screen or open a screenshot first.")
            return

        thresh = float(self.thresh_var.get())
        self.status_var.set(f"Scanning Atlas map for Simulacrum nodes (threshold={thresh:.2f})...")
        self.root.update_idletasks()

        self.detected_nodes = self.nav.detect_simulacrum_map_nodes(screen=self.cv_image, threshold=thresh)
        self.node_feedback.clear()
        self.selected_node_idx = None
        self.manual_box = None

        self.available_candidate_indices = [
            i for i, n in enumerate(self.detected_nodes) if n.get("is_accessible")
        ] or list(range(len(self.detected_nodes)))
        self.current_candidate_pointer = 0

        self.bot_selected_node = self.nav.select_accessible_simulacrum_map(candidates=self.detected_nodes, dry_run=True)
        if self.bot_selected_node:
            for i, n in enumerate(self.detected_nodes):
                if n["screen_circle_pos"] == self.bot_selected_node["screen_circle_pos"]:
                    if i in self.available_candidate_indices:
                        self.current_candidate_pointer = self.available_candidate_indices.index(i)
                    break

        for i, n in enumerate(self.detected_nodes):
            self.node_feedback[i] = "correct_acc" if n["is_accessible"] else "correct_inacc"

        self._update_bot_choice_ui()
        self._refresh_node_tree()
        self.redraw()
        acc_count = sum(1 for n in self.detected_nodes if n.get("is_accessible"))
        self.status_var.set(f"Detected {len(self.detected_nodes)} Simulacrum nodes ({acc_count} accessible).")

    def _update_bot_choice_ui(self, target_idx: Optional[int] = None):
        if not self.bot_selected_node:
            self.lbl_bot_choice.config(text="❌ No accessible Simulacrum node found to select.", foreground="#ff5252")
            return

        b_mx, b_my = self.bot_selected_node["screen_medal_pos"]
        b_cx, b_cy = self.bot_selected_node["screen_circle_pos"]
        b_conf = self.bot_selected_node["confidence"]
        b_acc_score = self.bot_selected_node.get("acc_score", 0)
        is_acc = self.bot_selected_node.get("is_accessible")
        acc_label = "ACCESSIBLE" if is_acc else "INACCESSIBLE"

        if target_idx is None:
            for i, n in enumerate(self.detected_nodes):
                if n["screen_circle_pos"] == (b_cx, b_cy):
                    target_idx = i
                    break

        idx_str = f"Node #{target_idx+1}" if target_idx is not None else "Node"
        total_avail = len(self.available_candidate_indices)
        rank_str = f" [Choice {self.current_candidate_pointer+1} of {total_avail}]" if total_avail > 0 else ""

        self.lbl_bot_choice.config(
            text=f"⭐ TARGET{rank_str}: {idx_str} at circle ({b_cx}, {b_cy})\nMedal ({b_mx}, {b_my}) | Conf: {b_conf:.1%} | {acc_label} (Score: {b_acc_score})",
            foreground="#00e676" if is_acc else "#ff9100",
        )

    def select_next_available_node(self):
        """Advances bot target selection to the next available (accessible) Simulacrum node."""
        if not self.detected_nodes:
            self.detect_nodes()
            if not self.detected_nodes:
                return

        if not self.available_candidate_indices:
            self.available_candidate_indices = [
                i for i, n in enumerate(self.detected_nodes) if n.get("is_accessible")
            ] or list(range(len(self.detected_nodes)))

        if not self.available_candidate_indices:
            messagebox.showinfo("No Nodes", "No Simulacrum nodes detected on screen.")
            return

        self.current_candidate_pointer = (self.current_candidate_pointer + 1) % len(self.available_candidate_indices)
        target_idx = self.available_candidate_indices[self.current_candidate_pointer]
        self.bot_selected_node = self.detected_nodes[target_idx]
        self.select_node(target_idx)
        self._update_bot_choice_ui(target_idx=target_idx)
        self._refresh_node_tree()
        self.tree.selection_set(str(target_idx))
        self.tree.see(str(target_idx))
        self.redraw()
        self.status_var.set(
            f"Bot target shifted to Node #{target_idx+1} ({self.current_candidate_pointer+1}/{len(self.available_candidate_indices)})."
        )

    def test_click_current_target(self):
        """Performs a test click on the active target node in the game."""
        if not self.bot_selected_node:
            messagebox.showinfo("No Target", "No target node selected to click.")
            return

        cx, cy = self.bot_selected_node["screen_circle_pos"]
        ok = self.nav.select_accessible_simulacrum_map(
            candidates=[self.bot_selected_node],
            screen=self.cv_image,
            dry_run=False,
        )
        if ok:
            messagebox.showinfo("Target Clicked", f"Successfully clicked target circle ({cx}, {cy}) and verified Delusion popup!")
            self.status_var.set(f"Target circle ({cx}, {cy}) clicked successfully!")
        else:
            self.status_var.set(f"Attempted click on target circle ({cx}, {cy}).")

    def _refresh_node_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, n in enumerate(self.detected_nodes):
            fb = self.node_feedback.get(i, "Unreviewed")
            fb_text = {"correct_acc": "✓ Correct (Acc)", "correct_inacc": "✓ Correct (Inacc)", "wrong_acc": "⚠ Wrong Acc", "false_pos": "✗ False Pos"}.get(fb, fb)
            acc_str = "ACC" if n["is_accessible"] else "INACC"
            is_bot_pick = bool(self.bot_selected_node and n["screen_circle_pos"] == self.bot_selected_node["screen_circle_pos"])
            prefix = "⭐ #" if is_bot_pick else "#"
            self.tree.insert("", tk.END, iid=str(i), values=(f"{prefix}{i+1}", acc_str, f"{n['confidence']:.1%}", fb_text))

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if sel:
            self.select_node(int(sel[0]))

    def select_node(self, idx: int):
        if idx < 0 or idx >= len(self.detected_nodes):
            return
        self.selected_node_idx = idx
        node = self.detected_nodes[idx]
        mx, my = node["screen_medal_pos"]
        cx, cy = node["screen_circle_pos"]
        conf, s = node["confidence"], node["scale"]
        acc = "ACCESSIBLE" if node.get("is_accessible") else "INACCESSIBLE"
        score = node.get("acc_score", 0)
        greens = len(node.get("connected_greens", []))
        is_bot_pick = bool(self.bot_selected_node and cx == self.bot_selected_node["screen_circle_pos"][0] and cy == self.bot_selected_node["screen_circle_pos"][1])

        pick_str = "⭐ CURRENT BOT CHOICE\n" if is_bot_pick else ""
        self.lbl_node_info.config(
            text=f"{pick_str}Node #{idx+1} ({acc}, Score: {score})\nMedal: ({mx}, {my}) | Circle: ({cx}, {cy})\nConf: {conf:.1%} | Scale: {s:.2f} | Green links: {greens}"
        )

        if self.cv_image is not None:
            sh, sw = self.cv_image.shape[:2]
            r = max(24, int(28 * s))
            patch = self.cv_image[max(0, my - r) : min(sh, my + r), max(0, mx - r) : min(sw, mx + r)]
            if patch.size > 0:
                patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)).resize((80, 80), Image.Resampling.NEAREST)
                self.node_thumb_photo = ImageTk.PhotoImage(patch_pil)
                self.lbl_node_thumb.config(image=self.node_thumb_photo, text="")

        self.redraw()

    def set_feedback(self, status: str):
        if self.selected_node_idx is None:
            messagebox.showinfo("Select Node", "Please select a node to mark feedback.")
            return
        self.node_feedback[self.selected_node_idx] = status
        self._refresh_node_tree()
        self.redraw()
        self.status_var.set(f"Node #{self.selected_node_idx+1} marked as: {status}")

    def toggle_accessibility_feedback(self):
        if self.selected_node_idx is None:
            return
        curr = self.node_feedback.get(self.selected_node_idx, "")
        self.set_feedback("correct_acc" if curr == "wrong_acc" else "wrong_acc")

    def save_new_template(self):
        """Crops the selected node or manual user box and saves as templates/ui/simulacrum_node_vX.png."""
        if self.cv_image is None:
            return

        try:
            node = self.detected_nodes[self.selected_node_idx] if self.selected_node_idx is not None else None
            out_path, desc = save_cropped_simulacrum_template(cv_image=self.cv_image, manual_box=self.manual_box, node=node)
            loaded_count = self.nav.reload_simulacrum_templates()
            messagebox.showinfo("Template Saved", f"Successfully saved {desc} as:\n'{out_path}'\n\nReloaded {loaded_count} Simulacrum templates.")
            self.status_var.set(f"Saved template '{out_path}'. Re-running detection...")
            self.detect_nodes()
        except Exception as e:
            messagebox.showerror("Error Saving Template", str(e))

    def export_report(self):
        """Generates a structured diagnostic JSON report and annotated comparison image."""
        if self.cv_image is None or not self.detected_nodes:
            messagebox.showinfo("No Data", "Please run detection on an image first.")
            return

        try:
            json_path, img_path, summary = export_simulacrum_feedback_report(
                cv_image=self.cv_image,
                detected_nodes=self.detected_nodes,
                bot_selected_node=self.bot_selected_node,
                node_feedback=self.node_feedback,
                image_source=self.current_image_path,
                threshold=float(self.thresh_var.get()),
            )
            messagebox.showinfo("Report Generated", f"Simulacrum Feedback Report saved:\n- JSON: '{json_path}'\n- Image: '{img_path}'\n\nAccuracy: {summary['accuracy']:.1%}")
            self.status_var.set(f"Exported diagnostic report to '{json_path}'.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to generate report: {e}")

    # --- Canvas Zoom & Pan & Drawing ---
    def reset_zoom(self):
        self.scale, self.offset_x, self.offset_y = 1.0, 0.0, 0.0
        self.redraw()

    def fit_to_view(self):
        if self.pil_image is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.pil_image.size
        if cw > 10 and ch > 10:
            self.scale = min(cw / iw, ch / ih) * 0.95
            self.offset_x = (cw - iw * self.scale) / 2
            self.offset_y = (ch - ih * self.scale) / 2
            self.redraw()

    def _screen_to_img_coords(self, sx: float, sy: float) -> Tuple[int, int]:
        return int((sx - self.offset_x) / self.scale), int((sy - self.offset_y) / self.scale)

    def _on_canvas_left_down(self, event):
        ix, iy = self._screen_to_img_coords(event.x, event.y)
        self.box_start_x, self.box_start_y = ix, iy
        self.box_curr_x, self.box_curr_y = ix, iy
        self.is_drawing_box = True

        clicked_idx = None
        for i, n in enumerate(self.detected_nodes):
            cx, cy = n["screen_circle_pos"]
            mx, my = n["screen_medal_pos"]
            if (abs(ix - cx) < 20 and abs(iy - cy) < 20) or (abs(ix - mx) < 20 and abs(iy - my) < 20):
                clicked_idx = i
                break

        if clicked_idx is not None:
            self.select_node(clicked_idx)
            self.tree.selection_set(str(clicked_idx))
            self.tree.see(str(clicked_idx))

    def _on_canvas_left_drag(self, event):
        if not self.is_drawing_box or self.pil_image is None:
            return
        ix, iy = self._screen_to_img_coords(event.x, event.y)
        iw, ih = self.pil_image.size
        self.box_curr_x = max(0, min(iw, ix))
        self.box_curr_y = max(0, min(ih, iy))
        self.redraw()

    def _on_canvas_left_up(self, event):
        if self.is_drawing_box:
            self.is_drawing_box = False
            x1, y1 = min(self.box_start_x, self.box_curr_x), min(self.box_start_y, self.box_curr_y)
            x2, y2 = max(self.box_start_x, self.box_curr_x), max(self.box_start_y, self.box_curr_y)
            if (x2 - x1) > 12 and (y2 - y1) > 12:
                self.manual_box = (x1, y1, x2, y2)
                self.status_var.set(f"Manual box: ({x1}, {y1}) -> ({x2}, {y2}). Click 'Crop Selection as New Template' to save.")
            self.redraw()

    def _on_canvas_right_down(self, event):
        self.pan_start_x, self.pan_start_y = event.x, event.y
        self.is_panning = True

    def _on_canvas_right_drag(self, event):
        if self.is_panning:
            self.offset_x += (event.x - self.pan_start_x)
            self.offset_y += (event.y - self.pan_start_y)
            self.pan_start_x, self.pan_start_y = event.x, event.y
            self.redraw()

    def _on_canvas_right_up(self, event):
        self.is_panning = False

    def _on_canvas_mousewheel(self, event):
        factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        new_scale = max(0.2, min(5.0, self.scale * factor))
        mx, my = event.x, event.y
        self.offset_x = mx - (mx - self.offset_x) * (new_scale / self.scale)
        self.offset_y = my - (my - self.offset_y) * (new_scale / self.scale)
        self.scale = new_scale
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if self.pil_image is None:
            return

        iw, ih = self.pil_image.size
        dw, dh = int(iw * self.scale), int(ih * self.scale)
        if dw < 10 or dh < 10:
            return

        resample = Image.Resampling.BILINEAR if self.scale < 1.0 else Image.Resampling.NEAREST
        self.tk_image = ImageTk.PhotoImage(self.pil_image.resize((dw, dh), resample))
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        for i, n in enumerate(self.detected_nodes):
            mx, my = n["screen_medal_pos"]
            cx, cy = n["screen_circle_pos"]
            s, is_acc = n["scale"], n["is_accessible"]
            fb = self.node_feedback.get(i, "unreviewed")

            smx = self.offset_x + mx * self.scale
            smy = self.offset_y + my * self.scale
            scx = self.offset_x + cx * self.scale
            scy = self.offset_y + cy * self.scale
            sr = max(8, int(14 * s * self.scale))

            border_col = "#00e676" if is_acc else "#ff9100"
            if fb == "wrong_acc":
                border_col = "#ffd600"
            elif fb == "false_pos":
                border_col = "#ff5252"

            is_selected = (self.selected_node_idx == i)
            is_bot_pick = bool(self.bot_selected_node and n["screen_circle_pos"] == self.bot_selected_node["screen_circle_pos"])

            mw = int(16 * s * self.scale)
            self.canvas.create_rectangle(smx - mw, smy - mw, smx + mw, smy + mw, outline=border_col, width=3 if is_selected else 2)
            self.canvas.create_line(smx, smy + mw, scx, scy - sr, fill=border_col, width=1, arrow=tk.LAST)
            self.canvas.create_oval(scx - sr, scy - sr, scx + sr, scy + sr, outline=border_col, width=3 if is_selected else 2)
            self.canvas.create_oval(scx - 3, scy - 3, scx + 3, scy + 3, fill=border_col, outline="")

            if is_bot_pick:
                self.canvas.create_oval(scx - sr - 6, scy - sr - 6, scx + sr + 6, scy + sr + 6, outline="#00d2ff", width=2, dash=(4, 2))
                self.canvas.create_text(scx, scy - sr - 12, text=f"⭐ TARGET #{i+1}", fill="#00d2ff", font=("Segoe UI", 9, "bold"))

            self.canvas.create_text(smx, smy - mw - 8, text=f"#{i+1} ({n['confidence']:.0%})", fill="#ffffff", font=("Segoe UI", 8, "bold"))

        if self.manual_box or (self.is_drawing_box and self.box_start_x != self.box_curr_x):
            bx1 = min(self.box_start_x, self.box_curr_x) if self.is_drawing_box else self.manual_box[0]
            by1 = min(self.box_start_y, self.box_curr_y) if self.is_drawing_box else self.manual_box[1]
            bx2 = max(self.box_start_x, self.box_curr_x) if self.is_drawing_box else self.manual_box[2]
            by2 = max(self.box_start_y, self.box_curr_y) if self.is_drawing_box else self.manual_box[3]

            sbx1 = self.offset_x + bx1 * self.scale
            sby1 = self.offset_y + by1 * self.scale
            sbx2 = self.offset_x + bx2 * self.scale
            sby2 = self.offset_y + by2 * self.scale
            self.canvas.create_rectangle(sbx1, sby1, sbx2, sby2, outline="#ffff00", width=2, dash=(4, 2))
            self.canvas.create_text(sbx1 + 4, sby1 - 10, text=f"New Template ({bx2-bx1}x{by2-by1})", fill="#ffff00", font=("Segoe UI", 8), anchor=tk.W)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voices - Simulacrum Node Finder & Feedback Inspector UI")
    parser.add_argument("--input", "-i", default=None, help="Path to screenshot image to analyze.")
    parser.add_argument("--monitor", "-m", type=int, default=2, help="Target monitor index for live capture. Default: 2")
    args = parser.parse_args()

    root = tk.Tk()
    app = SimulacrumFeedbackUI(root, initial_image=args.input, monitor_idx=args.monitor)
    root.mainloop()


if __name__ == "__main__":
    main()
