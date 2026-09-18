"""
Interactive Loot Item Selector, Rule Modifier & Filter Builder UI.
Allows users to:
1. Open game screenshots and visually crop/select multiple loot items.
2. Inspect and correct false positives or mismatches by clicking directly on detected boxes.
3. Live-tune detection thresholds, enable/disable rules, and re-crop/replace template images.
4. Save updates directly into routines/loot_filter.json.
"""

import os
import sys
import glob
import json
import time
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from src.loot_detector import LootDetector, LootItem


class LootItemSelectorUI:
    """Desktop GUI for selecting, cropping, editing, and correcting loot filter rules."""

    def __init__(self, root: tk.Tk, initial_image: Optional[str] = None):
        self.root = root
        self.root.title("Voices - Loot Filter Builder & Template Editor")
        self.root.geometry("1340x860")
        self.root.minsize(1020, 680)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.detector = LootDetector(config_file="routines/loot_filter.json")
        self.current_image_path: Optional[str] = None
        self.cv_image: Optional[np.ndarray] = None
        self.pil_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        # Canvas navigation state
        self.scale: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.pan_start_x: int = 0
        self.pan_start_y: int = 0
        self.is_panning: bool = False

        # Selection rectangle state
        self.is_drawing: bool = False
        self.start_orig_x: int = 0
        self.start_orig_y: int = 0
        self.curr_orig_x: int = 0
        self.curr_orig_y: int = 0
        self.selected_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.current_crop: Optional[np.ndarray] = None

        # Staged items for saving (list of dicts)
        self.staged_items: List[Dict[str, Any]] = []

        # Detection test overlay & selected detected item
        self.detected_overlay_items: List[LootItem] = []
        self.highlighted_detected_item: Optional[LootItem] = None
        self.hovered_detected_item: Optional[LootItem] = None
        self.hover_screen_pos: Optional[Tuple[int, int]] = None

        # Currently selected active rule in editor tab
        self.selected_rule_id: Optional[str] = None

        self._build_ui()

        # Load initial image or browse most recent
        if initial_image and os.path.exists(initial_image):
            self.load_image(initial_image)
        else:
            self._auto_load_recent_screenshot()

    def _build_ui(self):
        # Top Bar
        top_bar = ttk.Frame(self.root, padding=6)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_bar, text="📁 Open Screenshot...", command=self.browse_image).pack(side=tk.LEFT, padx=4)

        # Quick Load Dropdown
        self.recent_combo_var = tk.StringVar()
        self.recent_combo = ttk.Combobox(top_bar, textvariable=self.recent_combo_var, state="readonly", width=45)
        self.recent_combo.pack(side=tk.LEFT, padx=6)
        self.recent_combo.bind("<<ComboboxSelected>>", self._on_recent_selected)
        self._refresh_recent_screenshots_list()

        ttk.Button(top_bar, text="🔄 Refresh", command=self._refresh_recent_screenshots_list).pack(side=tk.LEFT, padx=2)

        # Zoom Controls
        zoom_frame = ttk.Frame(top_bar)
        zoom_frame.pack(side=tk.RIGHT, padx=6)
        ttk.Button(zoom_frame, text="🔍 Reset (100%)", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Fit Window", command=self.fit_to_window).pack(side=tk.LEFT, padx=2)

        # Body Container
        body_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body_paned.pack(fill=tk.BOTH, expand=True)

        # Left/Center: Canvas Frame
        canvas_container = ttk.Frame(body_paned)
        body_paned.add(canvas_container, weight=3)

        self.canvas = tk.Canvas(canvas_container, bg="#181818", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas event bindings
        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        self.canvas.bind("<ButtonPress-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)

        self.canvas.bind("<ButtonPress-2>", self._on_right_down)
        self.canvas.bind("<B2-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_right_up)

        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # Right: Notebook Tabs Sidebar
        sidebar = ttk.Frame(body_paned, padding=6, width=420)
        body_paned.add(sidebar, weight=1)

        self.notebook = ttk.Notebook(sidebar)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: ➕ Select & Add New Item
        self.tab_add = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_add, text="➕ Add New Item")
        self._build_add_tab(self.tab_add)

        # Tab 2: ⚙️ Active Rules & Template Editor
        self.tab_edit = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_edit, text="⚙️ Active Rules & Corrections")
        self._build_edit_tab(self.tab_edit)

        # Bottom Global Test / Save Bar
        bottom_bar = ttk.Frame(sidebar, padding=4)
        bottom_bar.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(bottom_bar, text="🎯 Test Detection on Image", command=self.test_detection_on_current_image).pack(fill=tk.X, pady=2)
        ttk.Button(bottom_bar, text="💾 Save to routines/loot_filter.json", command=self.save_to_loot_filter).pack(fill=tk.X, pady=2)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready. Click and drag to crop loot, or click on green detected boxes to edit rules.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_add_tab(self, parent):
        """Builds Tab 1: New Item Selection."""
        edit_group = ttk.LabelFrame(parent, text="Current Box Selection", padding=8)
        edit_group.pack(fill=tk.X, pady=(0, 8))

        # Thumbnail
        self.thumb_label = ttk.Label(edit_group, text="[ Draw a box on screenshot ]", background="#2a2a2a", foreground="#cccccc", anchor="center")
        self.thumb_label.pack(fill=tk.X, pady=4, ipady=12)

        self.dim_label = ttk.Label(edit_group, text="Dimensions: 0 x 0 px", foreground="#888888")
        self.dim_label.pack(anchor=tk.W)

        # Item Name
        ttk.Label(edit_group, text="Item Name:").pack(anchor=tk.W, pady=(4, 0))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(edit_group, textvariable=self.name_var, width=30)
        self.name_entry.pack(fill=tk.X, pady=(2, 6))

        # Priority
        prio_frame = ttk.Frame(edit_group)
        prio_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prio_frame, text="Priority Tier:").pack(side=tk.LEFT)
        self.prio_var = tk.IntVar(value=1)
        prio_combo = ttk.Combobox(prio_frame, textvariable=self.prio_var, values=[1, 2, 3, 4, 5], state="readonly", width=5)
        prio_combo.pack(side=tk.RIGHT)

        # Match Type
        type_frame = ttk.Frame(edit_group)
        type_frame.pack(fill=tk.X, pady=4)
        ttk.Label(type_frame, text="Rule Type:").pack(side=tk.LEFT)
        self.rule_type_var = tk.StringVar(value="template")
        ttk.Radiobutton(type_frame, text="Template Match", variable=self.rule_type_var, value="template").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(type_frame, text="Color Box", variable=self.rule_type_var, value="color_box").pack(side=tk.LEFT, padx=4)

        # Threshold Slider
        thresh_frame = ttk.Frame(edit_group)
        thresh_frame.pack(fill=tk.X, pady=4)
        ttk.Label(thresh_frame, text="Match Threshold:").pack(side=tk.LEFT)
        self.thresh_var = tk.DoubleVar(value=0.50)
        self.thresh_label = ttk.Label(thresh_frame, text="0.50")
        self.thresh_label.pack(side=tk.RIGHT)
        thresh_scale = ttk.Scale(edit_group, from_=0.30, to_=0.90, variable=self.thresh_var, command=self._on_thresh_change)
        thresh_scale.pack(fill=tk.X, pady=(0, 6))

        # Add Button
        ttk.Button(edit_group, text="➕ Add / Stage Item", command=self.stage_current_selection).pack(fill=tk.X, pady=4)

        # Staged Items List
        list_group = ttk.LabelFrame(parent, text="Staged Items for Filter", padding=6)
        list_group.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.items_listbox = tk.Listbox(list_group, bg="#222222", fg="#ffffff", selectbackground="#007acc", font=("Arial", 9))
        self.items_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(list_group, orient=tk.VERTICAL, command=self.items_listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.items_listbox.config(yscrollcommand=sb.set)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="❌ Remove", command=self.remove_staged_item).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_row, text="🗑️ Clear", command=self.clear_staged_items).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

    def _build_edit_tab(self, parent):
        """Builds Tab 2: Existing Rules & Template Modifier."""
        list_group = ttk.LabelFrame(parent, text="Existing Loot Rules (Click to Edit)", padding=6)
        list_group.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.rules_listbox = tk.Listbox(list_group, bg="#222222", fg="#ffffff", selectbackground="#e67e22", font=("Arial", 9), height=7)
        self.rules_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.rules_listbox.bind("<<ListboxSelect>>", self._on_rule_listbox_select)
        sb_rules = ttk.Scrollbar(list_group, orient=tk.VERTICAL, command=self.rules_listbox.yview)
        sb_rules.pack(side=tk.RIGHT, fill=tk.Y)
        self.rules_listbox.config(yscrollcommand=sb_rules.set)

        # Active Rule Editor Panel
        rule_edit_box = ttk.LabelFrame(parent, text="Selected Rule Properties", padding=8)
        rule_edit_box.pack(fill=tk.X, pady=(0, 4))

        # Rule Name & Status
        self.edit_rule_name_var = tk.StringVar()
        ttk.Label(rule_edit_box, text="Rule Name:").pack(anchor=tk.W)
        self.edit_rule_name_entry = ttk.Entry(rule_edit_box, textvariable=self.edit_rule_name_var)
        self.edit_rule_name_entry.pack(fill=tk.X, pady=(2, 4))

        row1 = ttk.Frame(rule_edit_box)
        row1.pack(fill=tk.X, pady=2)
        self.edit_rule_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Rule Enabled", variable=self.edit_rule_enabled_var).pack(side=tk.LEFT)

        ttk.Label(row1, text="Priority:").pack(side=tk.LEFT, padx=(15, 2))
        self.edit_rule_prio_var = tk.IntVar(value=1)
        prio_cb = ttk.Combobox(row1, textvariable=self.edit_rule_prio_var, values=[1, 2, 3, 4, 5], state="readonly", width=4)
        prio_cb.pack(side=tk.LEFT)

        # Template Controls Frame (shown when template rule selected)
        self.template_controls_frame = ttk.Frame(rule_edit_box)
        self.template_controls_frame.pack(fill=tk.X, pady=2)

        self.edit_template_thumb = ttk.Label(self.template_controls_frame, text="[ No Template ]", background="#2a2a2a", foreground="#cccccc", anchor="center")
        self.edit_template_thumb.pack(fill=tk.X, pady=2, ipady=6)

        edit_thresh_frame = ttk.Frame(self.template_controls_frame)
        edit_thresh_frame.pack(fill=tk.X, pady=2)
        ttk.Label(edit_thresh_frame, text="Match Threshold (Confidence):").pack(side=tk.LEFT)
        self.edit_thresh_var = tk.DoubleVar(value=0.50)
        self.edit_thresh_lbl = ttk.Label(edit_thresh_frame, text="0.50")
        self.edit_thresh_lbl.pack(side=tk.RIGHT)
        self.edit_thresh_scale = ttk.Scale(self.template_controls_frame, from_=0.20, to_=0.95, variable=self.edit_thresh_var, command=self._on_edit_thresh_change)
        self.edit_thresh_scale.pack(fill=tk.X, pady=(0, 2))

        ttk.Button(self.template_controls_frame, text="✏️ Replace Template with Current Box Crop", command=self.replace_selected_rule_template).pack(fill=tk.X, pady=2)

        # Color Box Parameters Frame (shown when color_box rule selected)
        self.color_box_controls_frame = ttk.Frame(rule_edit_box)

        # Auto-detect from box crop button
        ttk.Button(
            self.color_box_controls_frame,
            text="⚡ Auto-Detect Colors & Sliders from Box Crop",
            command=self.auto_fill_colors_from_current_crop
        ).pack(fill=tk.X, pady=(2, 4))

        # Color selectors
        c_row = ttk.Frame(self.color_box_controls_frame)
        c_row.pack(fill=tk.X, pady=2)
        ttk.Label(c_row, text="BG Color:").pack(side=tk.LEFT)
        self.edit_bg_color_var = tk.StringVar(value="orange")
        bg_cb = ttk.Combobox(c_row, textvariable=self.edit_bg_color_var, values=["orange", "white", "purple", "olive", "black", "blue", "yellow", "red", "any"], state="readonly", width=8)
        bg_cb.pack(side=tk.LEFT, padx=(2, 8))
        bg_cb.bind("<<ComboboxSelected>>", lambda e: self._on_color_param_change("bg_color", self.edit_bg_color_var.get()))

        ttk.Label(c_row, text="Text Color:").pack(side=tk.LEFT)
        self.edit_text_color_var = tk.StringVar(value="black")
        txt_cb = ttk.Combobox(c_row, textvariable=self.edit_text_color_var, values=["black", "white", "red", "yellow", "orange", "blue", "any"], state="readonly", width=8)
        txt_cb.pack(side=tk.LEFT, padx=2)
        txt_cb.bind("<<ComboboxSelected>>", lambda e: self._on_color_param_change("text_color", self.edit_text_color_var.get()))

        # Helper to build a slider row
        def make_param_slider(parent, label_text, var, from_, to_, step_fn, key_name):
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=label_text, font=("Arial", 8)).pack(side=tk.LEFT)
            lbl = ttk.Label(f, text=f"{var.get()}", font=("Arial", 8, "bold"))
            lbl.pack(side=tk.RIGHT)
            def _on_cmd(v):
                val = step_fn(float(v))
                var.set(val)
                lbl.config(text=f"{val}")
                self._on_color_param_change(key_name, val)
            s = ttk.Scale(parent, from_=from_, to_=to_, variable=var, command=_on_cmd)
            s.pack(fill=tk.X, pady=(0, 2))
            return s, lbl

        # Dimension sliders
        self.edit_min_w_var = tk.IntVar(value=35)
        self.scale_min_w, self.lbl_min_w = make_param_slider(self.color_box_controls_frame, "Min Width (px):", self.edit_min_w_var, 10, 400, int, "min_width")

        self.edit_max_w_var = tk.IntVar(value=500)
        self.scale_max_w, self.lbl_max_w = make_param_slider(self.color_box_controls_frame, "Max Width (px):", self.edit_max_w_var, 40, 800, int, "max_width")

        self.edit_min_h_var = tk.IntVar(value=14)
        self.scale_min_h, self.lbl_min_h = make_param_slider(self.color_box_controls_frame, "Min Height (px):", self.edit_min_h_var, 6, 80, int, "min_height")

        self.edit_max_h_var = tk.IntVar(value=75)
        self.scale_max_h, self.lbl_max_h = make_param_slider(self.color_box_controls_frame, "Max Height (px):", self.edit_max_h_var, 15, 150, int, "max_height")

        self.edit_min_ar_var = tk.DoubleVar(value=1.20)
        self.scale_min_ar, self.lbl_min_ar = make_param_slider(self.color_box_controls_frame, "Min Aspect Ratio (W/H):", self.edit_min_ar_var, 0.50, 4.00, lambda v: round(v, 2), "min_aspect_ratio")

        self.edit_min_bg_var = tk.DoubleVar(value=0.35)
        self.scale_min_bg, self.lbl_min_bg = make_param_slider(self.color_box_controls_frame, "Min BG Density Fraction:", self.edit_min_bg_var, 0.05, 0.90, lambda v: round(v, 2), "min_bg_fraction")

        self.edit_min_text_var = tk.IntVar(value=12)
        self.scale_min_text, self.lbl_min_text = make_param_slider(self.color_box_controls_frame, "Min Text Pixels:", self.edit_min_text_var, 0, 80, int, "min_text_pixels")

        # Action buttons
        btn_grid = ttk.Frame(rule_edit_box)
        btn_grid.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_grid, text="💾 Apply & Save to File", command=self.apply_rule_updates).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_grid, text="🗑️ Delete Rule", command=self.delete_selected_rule).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self._refresh_rules_listbox()

    def auto_fill_colors_from_current_crop(self):
        """Automatically detects dominant background and text colors from the drawn crop and sets sliders with tolerant thresholds."""
        if self.current_crop is None:
            messagebox.showwarning("No Box Drawn", "Please draw a box around a loot item on the screenshot first.")
            return

        analysis = LootDetector.analyze_crop_colors_and_geometry(self.current_crop)
        self.last_crop_analysis = analysis

        bg_col = analysis.get("bg_color", "orange")
        txt_col = analysis.get("text_color", "black")
        min_w = int(analysis.get("min_width", 25))
        max_w = int(analysis.get("max_width", 350))
        min_h = int(analysis.get("min_height", 14))
        max_h = int(analysis.get("max_height", 60))
        min_ar = float(analysis.get("min_aspect_ratio", 1.20))
        min_bg = float(analysis.get("min_bg_fraction", analysis.get("bg_fraction", 0.35)))
        min_txt = int(analysis.get("min_text_pixels", analysis.get("text_pixels", 12)))

        self.edit_bg_color_var.set(bg_col)
        self.edit_text_color_var.set(txt_col)

        self.edit_min_w_var.set(min_w)
        self.lbl_min_w.config(text=str(min_w))

        self.edit_max_w_var.set(max_w)
        self.lbl_max_w.config(text=str(max_w))

        self.edit_min_h_var.set(min_h)
        self.lbl_min_h.config(text=str(min_h))

        self.edit_max_h_var.set(max_h)
        self.lbl_max_h.config(text=str(max_h))

        self.edit_min_ar_var.set(min_ar)
        self.lbl_min_ar.config(text=f"{min_ar:.2f}")

        self.edit_min_bg_var.set(min_bg)
        self.lbl_min_bg.config(text=f"{min_bg:.2f}")

        self.edit_min_text_var.set(min_txt)
        self.lbl_min_text.config(text=str(min_txt))

        if self.selected_rule_id:
            for r in self.detector.rules:
                if r.get("id") == self.selected_rule_id:
                    r["bg_color"] = bg_col
                    r["text_color"] = txt_col
                    r["min_width"] = min_w
                    r["max_width"] = max_w
                    r["min_height"] = min_h
                    r["max_height"] = max_h
                    r["min_aspect_ratio"] = min_ar
                    r["min_bg_fraction"] = min_bg
                    r["min_text_pixels"] = min_txt
                    break

        detected_bg_pct = int(analysis.get("detected_bg_fraction", analysis["bg_fraction"]) * 100)
        detected_txt_px = analysis.get("detected_text_pixels", analysis["text_pixels"])
        self.status_var.set(
            f"⚡ Auto-Detected from crop: {analysis['bg_color'].upper()} BG / {analysis['text_color'].upper()} Text "
            f"({analysis['width']}x{analysis['height']} px | {detected_bg_pct}% BG, {detected_txt_px} text px). Sliders set with safety margins!"
        )
        self._trigger_debounced_test()

    def _on_color_param_change(self, param_name: str, val: Any):
        if self.selected_rule_id:
            for r in self.detector.rules:
                if r.get("id") == self.selected_rule_id:
                    r[param_name] = val
                    break
        self._trigger_debounced_test()


    def _on_thresh_change(self, val):
        self.thresh_label.config(text=f"{float(val):.2f}")
        self._trigger_debounced_test()

    def _on_edit_thresh_change(self, val):
        thresh_val = round(float(val), 2)
        self.edit_thresh_lbl.config(text=f"{thresh_val:.2f}")
        # Sync in-memory rule directly without saving to disk yet
        if self.selected_rule_id:
            for r in self.detector.rules:
                if r.get("id") == self.selected_rule_id:
                    r["threshold"] = thresh_val
                    break
        self._trigger_debounced_test()

    def _trigger_debounced_test(self):
        """Triggers live detection test after a brief 120ms debounce."""
        if hasattr(self, "_retest_timer") and self._retest_timer:
            try:
                self.root.after_cancel(self._retest_timer)
            except Exception:
                pass
        self._retest_timer = self.root.after(120, self.test_detection_on_current_image)

    def _refresh_rules_listbox(self):
        """Loads and lists all active rules from loot detector."""
        self.rules_listbox.delete(0, tk.END)
        for r in self.detector.rules:
            rid = r.get("id", "rule")
            name = r.get("name", rid)
            rtype = r.get("type", "color_box")
            prio = r.get("priority", 1)
            thresh = r.get("threshold", "-")
            status = "ON" if r.get("enabled", True) else "OFF"
            thresh_str = f"th={thresh:.2f}" if isinstance(thresh, (int, float)) else ""
            self.rules_listbox.insert(tk.END, f"[{status}] [P{prio}] {name} ({rtype} {thresh_str})")

    def _on_rule_listbox_select(self, event=None):
        sel = self.rules_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.detector.rules):
            rule = self.detector.rules[idx]
            self._populate_rule_editor(rule)

    def _populate_rule_editor(self, rule: Dict[str, Any]):
        self.selected_rule_id = rule.get("id")
        self.edit_rule_name_var.set(rule.get("name", ""))
        self.edit_rule_enabled_var.set(bool(rule.get("enabled", True)))
        self.edit_rule_prio_var.set(int(rule.get("priority", 1)))

        rtype = rule.get("type", "color_box")
        if rtype == "template":
            self.color_box_controls_frame.pack_forget()
            self.template_controls_frame.pack(fill=tk.X, pady=2)
            thresh = float(rule.get("threshold", 0.50))
            self.edit_thresh_var.set(thresh)
            self.edit_thresh_lbl.config(text=f"{thresh:.2f}")

            t_path = rule.get("template_file")
            if t_path and os.path.exists(t_path):
                try:
                    img = cv2.imread(t_path)
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)
                        pil_img.thumbnail((240, 60), Image.Resampling.LANCZOS)
                        self.rule_thumb_tk = ImageTk.PhotoImage(pil_img)
                        self.edit_template_thumb.config(image=self.rule_thumb_tk, text="")
                        return
                except Exception:
                    pass
            self.edit_template_thumb.config(image="", text="[ Template Missing ]")
        else:
            self.template_controls_frame.pack_forget()
            self.color_box_controls_frame.pack(fill=tk.X, pady=2)

            self.edit_bg_color_var.set(str(rule.get("bg_color", "white")))
            self.edit_text_color_var.set(str(rule.get("text_color", "red")))

            min_w = int(rule.get("min_width", 35))
            self.edit_min_w_var.set(min_w)
            self.lbl_min_w.config(text=str(min_w))

            max_w = int(rule.get("max_width", 500))
            self.edit_max_w_var.set(max_w)
            self.lbl_max_w.config(text=str(max_w))

            min_h = int(rule.get("min_height", 14))
            self.edit_min_h_var.set(min_h)
            self.lbl_min_h.config(text=str(min_h))

            max_h = int(rule.get("max_height", 75))
            self.edit_max_h_var.set(max_h)
            self.lbl_max_h.config(text=str(max_h))

            min_ar = float(rule.get("min_aspect_ratio", 1.20))
            self.edit_min_ar_var.set(min_ar)
            self.lbl_min_ar.config(text=f"{min_ar:.2f}")

            min_bg = float(rule.get("min_bg_fraction", 0.35))
            self.edit_min_bg_var.set(min_bg)
            self.lbl_min_bg.config(text=f"{min_bg:.2f}")

            min_txt = int(rule.get("min_text_pixels", 12))
            self.edit_min_text_var.set(min_txt)
            self.lbl_min_text.config(text=str(min_txt))

    def select_rule_by_id(self, rule_id: str):
        """Switches to Edit Tab and selects the rule matching rule_id."""
        for i, r in enumerate(self.detector.rules):
            if r.get("id") == rule_id:
                self.notebook.select(self.tab_edit)
                self.rules_listbox.selection_clear(0, tk.END)
                self.rules_listbox.selection_set(i)
                self.rules_listbox.see(i)
                self._populate_rule_editor(r)
                break

    def replace_selected_rule_template(self):
        """Replaces the template image of the selected rule with the currently drawn box crop."""
        if not self.selected_rule_id:
            messagebox.showwarning("No Rule Selected", "Please select a rule from the list above first.")
            return
        if self.current_crop is None:
            messagebox.showwarning("No Box Drawn", "Please draw a new selection box around the item on the screenshot first.")
            return

        success = self.detector.update_template_image(self.selected_rule_id, self.current_crop, save=True)
        if success:
            for r in self.detector.rules:
                if r.get("id") == self.selected_rule_id:
                    self._populate_rule_editor(r)
                    break
            self._refresh_rules_listbox()
            self.status_var.set(f"Successfully updated template image for rule '{self.selected_rule_id}'.")
            self.test_detection_on_current_image()
        else:
            messagebox.showerror("Error", f"Failed to update template for rule '{self.selected_rule_id}'.")

    def apply_rule_updates(self):
        """Persists current rule edits to routines/loot_filter.json."""
        if not self.selected_rule_id:
            messagebox.showwarning("No Rule Selected", "Please select a rule from the list first.")
            return

        for r in self.detector.rules:
            if r.get("id") == self.selected_rule_id:
                r["name"] = self.edit_rule_name_var.get().strip()
                r["enabled"] = self.edit_rule_enabled_var.get()
                r["priority"] = self.edit_rule_prio_var.get()
                if r.get("type") == "template":
                    r["threshold"] = round(self.edit_thresh_var.get(), 3)
                elif r.get("type") == "color_box":
                    r["bg_color"] = self.edit_bg_color_var.get()
                    r["text_color"] = self.edit_text_color_var.get()
                    r["min_width"] = int(self.edit_min_w_var.get())
                    r["max_width"] = int(self.edit_max_w_var.get())
                    r["min_height"] = int(self.edit_min_h_var.get())
                    r["max_height"] = int(self.edit_max_h_var.get())
                    r["min_aspect_ratio"] = round(float(self.edit_min_ar_var.get()), 2)
                    r["min_bg_fraction"] = round(float(self.edit_min_bg_var.get()), 2)
                    r["min_text_pixels"] = int(self.edit_min_text_var.get())
                break

        self.detector.rules.sort(key=lambda x: int(x.get("priority", 99)))
        self.detector.save_config()
        self._refresh_rules_listbox()
        self.status_var.set(f"Saved changes for rule '{self.selected_rule_id}' to routines/loot_filter.json.")
        messagebox.showinfo("Saved", f"Rule '{self.selected_rule_id}' saved successfully to disk.")
        self.test_detection_on_current_image()

    def delete_selected_rule(self):
        """Removes the selected rule entirely from loot_filter.json."""
        if not self.selected_rule_id:
            messagebox.showwarning("No Rule Selected", "Please select a rule from the list first.")
            return
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete rule '{self.selected_rule_id}'?"):
            self.detector.remove_rule(self.selected_rule_id, save=True)
            self.selected_rule_id = None
            self._refresh_rules_listbox()
            self.edit_rule_name_var.set("")
            self.edit_template_thumb.config(image="", text="[ Deleted ]")
            self.status_var.set("Rule deleted.")
            self.test_detection_on_current_image()

    def _refresh_recent_screenshots_list(self):
        """Scans loot_debug, scratch, templates for recent png/jpg files."""
        patterns = [
            "loot_debug/*.png",
            "scratch/*.png",
            "scratch/*.jpg",
            "debug_logs/**/*.png",
            ".user_uploaded/*.png",
            ".user_uploaded/*.jpg",
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        files.sort(key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0, reverse=True)
        self.recent_files = files[:40]
        display_names = [os.path.basename(f) for f in self.recent_files]
        self.recent_combo["values"] = display_names
        if display_names and not self.current_image_path:
            self.recent_combo.current(0)

    def _auto_load_recent_screenshot(self):
        if getattr(self, "recent_files", None) and len(self.recent_files) > 0:
            self.load_image(self.recent_files[0])

    def _on_recent_selected(self, event=None):
        idx = self.recent_combo.current()
        if 0 <= idx < len(self.recent_files):
            self.load_image(self.recent_files[idx])

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Game Screenshot",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        if not os.path.exists(path):
            messagebox.showerror("File Error", f"Cannot find image: {path}")
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Image Error", f"Failed to decode image: {path}")
            return
        self.current_image_path = path
        self.cv_image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(rgb)
        self.selected_rect = None
        self.current_crop = None
        self.detected_overlay_items.clear()
        self.highlighted_detected_item = None
        self.fit_to_window()
        self.status_var.set(f"Loaded '{os.path.basename(path)}' ({img.shape[1]}x{img.shape[0]} px)")

    def fit_to_window(self):
        if self.pil_image is None:
            return
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        iw, ih = self.pil_image.size
        scale_w = cw / iw
        scale_h = ch / ih
        self.scale = min(scale_w, scale_h, 1.0)
        self.offset_x = max(0, (cw - iw * self.scale) / 2)
        self.offset_y = max(0, (ch - ih * self.scale) / 2)
        self._redraw()

    def reset_zoom(self):
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._redraw()

    def _redraw(self):
        if self.pil_image is None:
            self.canvas.delete("all")
            return

        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = self.pil_image.size

        disp_w = max(1, int(iw * self.scale))
        disp_h = max(1, int(ih * self.scale))

        resized = self.pil_image.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        # Draw detected items overlay
        for item in self.detected_overlay_items:
            sx1 = self.offset_x + item.x * self.scale
            sy1 = self.offset_y + item.y * self.scale
            sx2 = sx1 + item.w * self.scale
            sy2 = sy1 + item.h * self.scale

            is_hl = (self.highlighted_detected_item is not None and self.highlighted_detected_item.rule_id == item.rule_id and self.highlighted_detected_item.x == item.x)
            is_hover = (self.hovered_detected_item is not None and self.hovered_detected_item.rule_id == item.rule_id and self.hovered_detected_item.x == item.x)

            if is_hover:
                color = "#00e5ff"
                width = 3
            elif is_hl:
                color = "#ffff00"
                width = 3
            else:
                color = "#00ff00"
                width = 2

            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=width)
            badge_txt = f"[P{item.priority}] {item.rule_name} ({item.confidence:.2f})"
            self.canvas.create_text(sx1 + 4, sy1 - 8, text=badge_txt, fill=color, anchor=tk.W, font=("Arial", 9, "bold"))

        # Draw staged items
        for staged in self.staged_items:
            rx, ry, rw, rh = staged["rect"]
            sx1 = self.offset_x + rx * self.scale
            sy1 = self.offset_y + ry * self.scale
            sx2 = sx1 + rw * self.scale
            sy2 = sy1 + rh * self.scale
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#00e5ff", width=2, dash=(4, 2))
            self.canvas.create_text(sx1 + 4, sy1 + 10, text=staged["name"], fill="#00e5ff", anchor=tk.W, font=("Arial", 9))

        # Draw active drawing selection rectangle
        if self.selected_rect:
            rx, ry, rw, rh = self.selected_rect
            sx1 = self.offset_x + rx * self.scale
            sy1 = self.offset_y + ry * self.scale
            sx2 = sx1 + rw * self.scale
            sy2 = sy1 + rh * self.scale
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#ff0055", width=2)

        # Draw hover inspection HUD / Tooltip
        if self.hovered_detected_item is not None:
            item = self.hovered_detected_item
            sx1 = self.offset_x + item.x * self.scale
            sy1 = self.offset_y + item.y * self.scale
            sx2 = sx1 + item.w * self.scale
            sy2 = sy1 + item.h * self.scale

            # Highlight hovered box with glowing cyan outline
            self.canvas.create_rectangle(sx1 - 2, sy1 - 2, sx2 + 2, sy2 + 2, outline="#00e5ff", width=2)

            # Center target crosshair
            scx = (sx1 + sx2) / 2
            scy = (sy1 + sy2) / 2
            self.canvas.create_line(scx - 6, scy, scx + 6, scy, fill="#00e5ff", width=1)
            self.canvas.create_line(scx, scy - 6, scx, scy + 6, fill="#00e5ff", width=1)

            # Dimension indicators on edges
            # Width dimension along bottom
            dim_w_txt = f"↔ {item.w} px"
            self.canvas.create_text(scx, sy2 + 10, text=dim_w_txt, fill="#00e5ff", anchor=tk.N, font=("Consolas", 9, "bold"))
            # Height dimension along right
            dim_h_txt = f"↕ {item.h} px"
            self.canvas.create_text(sx2 + 6, scy, text=dim_h_txt, fill="#00e5ff", anchor=tk.W, font=("Consolas", 9, "bold"))

            # Build Tooltip lines
            ar = item.w / max(1.0, float(item.h))
            lines = [
                f"🏷️ {item.rule_name}  [Priority {item.priority}]",
                f"📐 Size: {item.w}W x {item.h}H px  (Aspect Ratio: {ar:.2f})",
                f"🎯 Confidence: {item.confidence * 100:.1f}%  ({item.confidence:.3f})",
                f"📍 Position: ({item.x}, {item.y})  |  Center: ({item.center_x}, {item.center_y})",
            ]
            if item.details:
                if "bg_fraction" in item.details and "text_pixels" in item.details:
                    lines.append(f"📊 BG Density: {item.details['bg_fraction']*100:.1f}%  |  Text Pixels: {item.details['text_pixels']} px")
                elif "template_file" in item.details:
                    tname = os.path.basename(item.details.get("template_file", ""))
                    scale = item.details.get("scale", 1.0)
                    lines.append(f"🖼️ Template: {tname}  (Scale: {scale:.2f}x)")
            lines.append("💡 Click box to select & tune rule in Tab 2")

            # Calculate tooltip bounding box
            tt_w = 340
            line_h = 18
            tt_h = len(lines) * line_h + 16

            # Determine best placement (above box or below box, keeping inside canvas)
            cw = max(self.canvas.winfo_width(), 400)
            ch = max(self.canvas.winfo_height(), 300)

            # Try placing above first
            tt_x1 = max(10, min(cw - tt_w - 10, sx1))
            tt_y1 = sy1 - tt_h - 16
            if tt_y1 < 10:  # If too high, place below
                tt_y1 = sy2 + 28
            if tt_y1 + tt_h > ch - 10:  # If too low, clamp
                tt_y1 = max(10, ch - tt_h - 10)

            tt_x2 = tt_x1 + tt_w
            tt_y2 = tt_y1 + tt_h

            # Draw tooltip background with border
            self.canvas.create_rectangle(tt_x1, tt_y1, tt_x2, tt_y2, fill="#12161f", outline="#00e5ff", width=2)
            # Header accent line
            self.canvas.create_line(tt_x1 + 4, tt_y1 + line_h + 8, tt_x2 - 4, tt_y1 + line_h + 8, fill="#2a3b50", width=1)

            # Draw lines
            curr_y = tt_y1 + 10
            for i, line in enumerate(lines):
                if i == 0:
                    self.canvas.create_text(tt_x1 + 10, curr_y, text=line, fill="#ffdd59", anchor=tk.NW, font=("Arial", 10, "bold"))
                elif i == len(lines) - 1:
                    self.canvas.create_text(tt_x1 + 10, curr_y, text=line, fill="#a4b0be", anchor=tk.NW, font=("Arial", 8, "italic"))
                elif "Size" in line:
                    self.canvas.create_text(tt_x1 + 10, curr_y, text=line, fill="#00e5ff", anchor=tk.NW, font=("Consolas", 9, "bold"))
                elif "Confidence" in line:
                    self.canvas.create_text(tt_x1 + 10, curr_y, text=line, fill="#2ed573", anchor=tk.NW, font=("Arial", 9, "bold"))
                else:
                    self.canvas.create_text(tt_x1 + 10, curr_y, text=line, fill="#f1f2f6", anchor=tk.NW, font=("Arial", 9))
                curr_y += line_h

    def _screen_to_orig(self, sx: int, sy: int) -> Tuple[int, int]:
        if self.pil_image is None or self.scale <= 0:
            return 0, 0
        ox = int((sx - self.offset_x) / self.scale)
        oy = int((sy - self.offset_y) / self.scale)
        iw, ih = self.pil_image.size
        return max(0, min(iw - 1, ox)), max(0, min(ih - 1, oy))

    def _find_detected_item_at(self, ox: int, oy: int) -> Optional[LootItem]:
        """Finds any detected item that contains the point (ox, oy)."""
        for item in reversed(self.detected_overlay_items):
            if item.x <= ox <= (item.x + item.w) and item.y <= oy <= (item.y + item.h):
                return item
        return None

    def _on_mouse_leave(self, event=None):
        if self.hovered_detected_item is not None:
            self.hovered_detected_item = None
            self.hover_screen_pos = None
            self._redraw()
            self.status_var.set("Ready. Click and drag to crop loot, or hover over detected boxes to inspect parameters.")

    def _on_mouse_move(self, event):
        if getattr(self, "is_drawing", False) or getattr(self, "is_panning", False) or self.pil_image is None:
            return
        if not self.detected_overlay_items:
            if self.hovered_detected_item is not None:
                self.hovered_detected_item = None
                self.hover_screen_pos = None
                self._redraw()
            return

        ox, oy = self._screen_to_orig(event.x, event.y)
        item = self._find_detected_item_at(ox, oy)
        self.hover_screen_pos = (event.x, event.y)

        if item != self.hovered_detected_item:
            self.hovered_detected_item = item
            self._redraw()
            if item:
                ar = item.w / max(1.0, float(item.h))
                det_str = ""
                if item.details:
                    if "bg_fraction" in item.details:
                        det_str += f" | BG: {item.details['bg_fraction']*100:.0f}%"
                    if "text_pixels" in item.details:
                        det_str += f" | TextPx: {item.details['text_pixels']}"
                    if "template_file" in item.details:
                        det_str += f" | Tmpl: {os.path.basename(item.details['template_file'])}"
                self.status_var.set(
                    f"🎯 [{item.rule_name}] | Size: {item.w}W x {item.h}H px (AR: {ar:.2f}) | Conf: {item.confidence*100:.1f}% | P{item.priority} | Rect: ({item.x}, {item.y}){det_str}"
                )
            else:
                self.status_var.set("Ready. Hover over green detected boxes to inspect details, or click to edit rule.")

    def _on_left_down(self, event):
        if self.pil_image is None:
            return
        self.is_drawing = True
        ox, oy = self._screen_to_orig(event.x, event.y)
        self.start_orig_x = ox
        self.start_orig_y = oy
        self.curr_orig_x = ox
        self.curr_orig_y = oy

    def _on_left_drag(self, event):
        if not self.is_drawing or self.pil_image is None:
            return
        ox, oy = self._screen_to_orig(event.x, event.y)
        self.curr_orig_x = ox
        self.curr_orig_y = oy
        x1 = min(self.start_orig_x, self.curr_orig_x)
        y1 = min(self.start_orig_y, self.curr_orig_y)
        w = max(1, abs(self.curr_orig_x - self.start_orig_x))
        h = max(1, abs(self.curr_orig_y - self.start_orig_y))
        self.selected_rect = (x1, y1, w, h)
        self._redraw()

    def _on_left_up(self, event):
        if not self.is_drawing:
            return
        self.is_drawing = False
        drag_dist = max(abs(self.curr_orig_x - self.start_orig_x), abs(self.curr_orig_y - self.start_orig_y))

        # If user just clicked without dragging, check if they clicked a detected box
        if drag_dist < 6 and self.detected_overlay_items:
            clicked_item = self._find_detected_item_at(self.start_orig_x, self.start_orig_y)
            if clicked_item:
                self.highlighted_detected_item = clicked_item
                self._redraw()
                self.select_rule_by_id(clicked_item.rule_id)
                self.status_var.set(f"Selected match '{clicked_item.rule_name}' (rule_id: {clicked_item.rule_id}, conf: {clicked_item.confidence:.2f}). Adjust threshold or replace template.")
                return

        if self.selected_rect and self.selected_rect[2] > 4 and self.selected_rect[3] > 4:
            self._update_crop_preview(self.selected_rect)

    def _on_right_down(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def _on_right_drag(self, event):
        if not self.is_panning:
            return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.offset_x += dx
        self.offset_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self._redraw()

    def _on_right_up(self, event):
        self.is_panning = False

    def _on_mousewheel(self, event):
        if self.pil_image is None:
            return
        factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        new_scale = max(0.1, min(10.0, self.scale * factor))

        mx = event.x
        my = event.y
        self.offset_x = mx - (mx - self.offset_x) * (new_scale / self.scale)
        self.offset_y = my - (my - self.offset_y) * (new_scale / self.scale)
        self.scale = new_scale
        self._redraw()

    def _update_crop_preview(self, rect: Tuple[int, int, int, int]):
        if self.cv_image is None:
            return
        x, y, w, h = rect
        crop = self.cv_image[y:y+h, x:x+w].copy()
        self.current_crop = crop

        analysis = LootDetector.analyze_crop_colors_and_geometry(crop)
        self.last_crop_analysis = analysis

        dim_txt = (
            f"Dimensions: {w} x {h} px at ({x}, {y}) (AR: {analysis['aspect_ratio']:.2f})\n"
            f"🎨 Auto-Detected: {analysis['bg_color'].upper()} BG / {analysis['text_color'].upper()} Text "
            f"({int(analysis.get('detected_bg_fraction', analysis['bg_fraction'])*100)}% BG, {analysis.get('detected_text_pixels', analysis['text_pixels'])} text px)"
        )
        self.dim_label.config(text=dim_txt)

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(rgb_crop)
        pil_crop.thumbnail((240, 80), Image.Resampling.LANCZOS)
        self.thumb_tk = ImageTk.PhotoImage(pil_crop)
        self.thumb_label.config(image=self.thumb_tk, text="")

        # If user is in Tab 2 with a color_box rule selected, automatically notify status bar
        if self.notebook.index(self.notebook.select()) == 1 and self.selected_rule_id:
            self.status_var.set(
                f"💡 Selected {analysis['bg_color'].upper()} BG / {analysis['text_color'].upper()} Text ({w}x{h}px). "
                f"Click '⚡ Auto-Detect Colors & Sliders from Box Crop' to apply."
            )

        if not self.name_var.get().strip():
            self.name_var.set(f"Item #{len(self.staged_items) + 1}")
        self.name_entry.focus_set()
        self.name_entry.select_range(0, tk.END)

    def stage_current_selection(self):
        if self.current_crop is None or self.selected_rect is None:
            messagebox.showwarning("No Selection", "Please draw a box around a loot item first.")
            return

        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a descriptive name for this item.")
            return

        analysis = getattr(self, "last_crop_analysis", None)
        if not analysis:
            analysis = LootDetector.analyze_crop_colors_and_geometry(self.current_crop)

        staged_item = {
            "name": name,
            "priority": self.prio_var.get(),
            "type": self.rule_type_var.get(),
            "threshold": self.thresh_var.get(),
            "rect": self.selected_rect,
            "crop": self.current_crop.copy(),
            "bg_color": analysis.get("bg_color", "orange"),
            "text_color": analysis.get("text_color", "black"),
            "min_width": analysis.get("min_width", 20),
            "max_width": analysis.get("max_width", 400),
            "min_height": analysis.get("min_height", 12),
            "max_height": analysis.get("max_height", 60),
            "min_aspect_ratio": analysis.get("min_aspect_ratio", 1.2),
            "min_bg_fraction": analysis.get("min_bg_fraction", 0.35),
            "min_text_pixels": analysis.get("min_text_pixels", 12),
        }
        self.staged_items.append(staged_item)
        self._refresh_staged_listbox()
        self.status_var.set(f"Added '{name}' (Priority {self.prio_var.get()}, {analysis['bg_color']}/{analysis['text_color']}) to staging list.")

        self.selected_rect = None
        self.current_crop = None
        self.name_var.set("")
        self.thumb_label.config(image="", text="[ Selection Staged. Draw next box ]")
        self._redraw()

    def _refresh_staged_listbox(self):
        self.items_listbox.delete(0, tk.END)
        for i, it in enumerate(self.staged_items):
            info_tag = f"{it['bg_color']}/{it['text_color']}" if it.get("type") == "color_box" else "template"
            self.items_listbox.insert(tk.END, f"#{i+1}: [P{it['priority']}] {it['name']} ({info_tag})")

    def remove_staged_item(self):
        sel = self.items_listbox.curselection()
        if sel:
            idx = sel[0]
            del self.staged_items[idx]
            self._refresh_staged_listbox()
            self._redraw()

    def clear_staged_items(self):
        self.staged_items.clear()
        self._refresh_staged_listbox()
        self._redraw()

    def test_detection_on_current_image(self):
        if self.cv_image is None:
            return

        # 1. Sync any active form values into self.detector in-memory rules
        if self.selected_rule_id:
            for r in self.detector.rules:
                if r.get("id") == self.selected_rule_id:
                    name_val = self.edit_rule_name_var.get().strip()
                    if name_val:
                        r["name"] = name_val
                    r["enabled"] = bool(self.edit_rule_enabled_var.get())
                    r["priority"] = int(self.edit_rule_prio_var.get())
                    if r.get("type") == "template":
                        r["threshold"] = round(float(self.edit_thresh_var.get()), 3)
                    elif r.get("type") == "color_box":
                        r["bg_color"] = self.edit_bg_color_var.get()
                        r["text_color"] = self.edit_text_color_var.get()
                        r["min_width"] = int(self.edit_min_w_var.get())
                        r["max_width"] = int(self.edit_max_w_var.get())
                        r["min_height"] = int(self.edit_min_h_var.get())
                        r["max_height"] = int(self.edit_max_h_var.get())
                        r["min_aspect_ratio"] = round(float(self.edit_min_ar_var.get()), 2)
                        r["min_bg_fraction"] = round(float(self.edit_min_bg_var.get()), 2)
                        r["min_text_pixels"] = int(self.edit_min_text_var.get())
                    break

        # 2. Build in-memory test detector with current rules (without re-reading from disk)
        test_detector = LootDetector(config_file=None)
        test_detector.enabled = self.detector.enabled
        test_detector.max_pickups = self.detector.max_pickups
        test_detector.rules = [dict(r) for r in self.detector.rules]
        test_detector._template_cache = dict(self.detector._template_cache)

        # 3. Add any staged items from Tab 1
        for staged in self.staged_items:
            if staged["type"] == "template":
                test_detector.add_template_item_rule(
                    name=staged["name"],
                    crop_img=staged["crop"],
                    priority=staged["priority"],
                    threshold=staged["threshold"],
                    save=False
                )
            else:
                rx, ry, rw, rh = staged["rect"]
                rule_dict = {
                    "id": f"color_{staged['name'].lower().replace(' ', '_')}_{int(time.time()%10000)}",
                    "name": staged["name"],
                    "enabled": True,
                    "priority": staged["priority"],
                    "type": "color_box",
                    "bg_color": staged.get("bg_color", "orange"),
                    "text_color": staged.get("text_color", "black"),
                    "min_width": staged.get("min_width", max(15, int(rw * 0.70))),
                    "max_width": staged.get("max_width", int(rw * 1.35)),
                    "min_height": staged.get("min_height", max(10, int(rh * 0.70))),
                    "max_height": staged.get("max_height", int(rh * 1.35)),
                    "min_aspect_ratio": staged.get("min_aspect_ratio", round(rw / max(1.0, float(rh)) * 0.75, 2)),
                    "min_bg_fraction": staged.get("min_bg_fraction", 0.30),
                    "min_text_pixels": staged.get("min_text_pixels", 12),
                    "description": f"Custom color rule ({staged.get('bg_color')}/{staged.get('text_color')}) for {staged['name']}",
                }
                test_detector.add_or_update_rule(rule_dict, save=False)

        # 4. Run detection
        detected = test_detector.detect_loot(self.cv_image)
        self.detected_overlay_items = detected
        self._redraw()
        msg = f"Detection test: Found {len(detected)} item match(es)."
        self.status_var.set(msg)

    def save_to_loot_filter(self):
        if not self.staged_items:
            messagebox.showwarning("Empty", "No new items staged. Please select and add items first.")
            return

        saved_count = 0
        for item in self.staged_items:
            if item["type"] == "template":
                self.detector.add_template_item_rule(
                    name=item["name"],
                    crop_img=item["crop"],
                    priority=item["priority"],
                    threshold=item["threshold"],
                    templates_dir="ui/loot_templates",
                    save=True
                )
                saved_count += 1
            else:
                rx, ry, rw, rh = item["rect"]
                rule_dict = {
                    "id": f"color_{item['name'].lower().replace(' ', '_')}_{int(time.time()%10000)}",
                    "name": item["name"],
                    "enabled": True,
                    "priority": item["priority"],
                    "type": "color_box",
                    "bg_color": item.get("bg_color", "orange"),
                    "text_color": item.get("text_color", "black"),
                    "min_width": item.get("min_width", max(15, int(rw * 0.70))),
                    "max_width": item.get("max_width", int(rw * 1.35)),
                    "min_height": item.get("min_height", max(10, int(rh * 0.70))),
                    "max_height": item.get("max_height", int(rh * 1.35)),
                    "min_aspect_ratio": item.get("min_aspect_ratio", round(rw / max(1.0, float(rh)) * 0.75, 2)),
                    "min_bg_fraction": item.get("min_bg_fraction", 0.30),
                    "min_text_pixels": item.get("min_text_pixels", 12),
                    "description": f"Custom color rule ({item.get('bg_color')}/{item.get('text_color')}) for {item['name']}",
                }
                self.detector.add_or_update_rule(rule_dict, save=True)
                saved_count += 1

        self.staged_items.clear()
        self._refresh_staged_listbox()
        self._refresh_rules_listbox()
        self._redraw()
        messagebox.showinfo("Success", f"Successfully saved {saved_count} item rule(s) to 'routines/loot_filter.json'!")
        self.status_var.set(f"Saved {saved_count} rules to routines/loot_filter.json.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Loot Item Selector & Filter Builder UI")
    parser.add_argument("--input", "-i", help="Initial screenshot image path to load.")
    args = parser.parse_args()

    root = tk.Tk()
    app = LootItemSelectorUI(root, initial_image=args.input)
    root.mainloop()


if __name__ == "__main__":
    main()
