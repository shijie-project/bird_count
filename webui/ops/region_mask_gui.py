import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from PIL import Image, ImageDraw, ImageTk


HANDLE_SIZE = 6
MIN_BOX = 5
MIN_POLY_VERTICES = 3
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_SUBDIR = "masked"

# (cursor, x-edge, y-edge) — edges are -1 (min), 0 (none), +1 (max)
RECT_HANDLES = {
    "nw": ("size_nw_se", -1, -1),
    "n": ("sb_v_double_arrow", 0, -1),
    "ne": ("size_ne_sw", 1, -1),
    "e": ("sb_h_double_arrow", 1, 0),
    "se": ("size_nw_se", 1, 1),
    "s": ("sb_v_double_arrow", 0, 1),
    "sw": ("size_ne_sw", -1, 1),
    "w": ("sb_h_double_arrow", -1, 0),
}


class RegionMaskGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Region Mask Tool")

        self.folder: Optional[Path] = None
        self.image_paths: list[Path] = []
        self.current_idx: int = -1

        self.image: Optional[Image.Image] = None
        self.image_path: Optional[Path] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0

        # Shape state — only one shape exists at a time.
        self.mode = tk.StringVar(value="rect")  # "rect" | "poly" | "bitmap"
        # Rectangle, image coords: [x0, y0, x1, y1] with x0<x1, y0<y1
        self.box: Optional[list[int]] = None
        # Polygon, image coords: [[x, y], ...]
        self.polygon: list[list[int]] = []
        self.polygon_closed = False
        # Bitmap mask in current image coords; 'L' mode, 0/255.
        self.bitmap_mask: Optional[Image.Image] = None

        # Drag state
        self.drag_mode: Optional[str] = None  # rect: "draw" | "move" | handle key
        self.drag_anchor: Optional[tuple[int, int]] = None
        self.drag_box_start: Optional[list[int]] = None
        self.drag_vertex_idx: Optional[int] = None
        self.drag_poly_start: Optional[list[list[int]]] = None

        self._build_ui()

        self.root.bind("<Left>", lambda _e: self.prev_image())
        self.root.bind("<Right>", lambda _e: self.next_image())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Row 1: folder + navigation
        row_nav = tk.Frame(toolbar)
        row_nav.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 0))

        tk.Button(row_nav, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT)
        tk.Button(row_nav, text="◀", command=self.prev_image, width=3).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(row_nav, text="▶", command=self.next_image, width=3).pack(side=tk.LEFT)
        self.status = tk.Label(row_nav, text="No folder loaded", anchor="w")
        self.status.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # Row 2: mode + mask actions
        row_tools = tk.Frame(toolbar)
        row_tools.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 4))

        tk.Label(row_tools, text="Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(
            row_tools, text="Rectangle", variable=self.mode, value="rect", command=self._on_mode_change
        ).pack(side=tk.LEFT)
        tk.Radiobutton(row_tools, text="Polygon", variable=self.mode, value="poly", command=self._on_mode_change).pack(
            side=tk.LEFT
        )
        tk.Radiobutton(
            row_tools, text="Bitmap", variable=self.mode, value="bitmap", command=self._on_mode_change
        ).pack(side=tk.LEFT)

        tk.Frame(row_tools, width=2, bg="gray60").pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        tk.Button(row_tools, text="Reset", command=self.reset_selection).pack(side=tk.LEFT, padx=2)
        tk.Button(row_tools, text="Save Mask", command=self.save_masked).pack(side=tk.LEFT, padx=2)
        tk.Button(row_tools, text="Load Mask", command=self.load_mask).pack(side=tk.LEFT, padx=2)
        tk.Button(row_tools, text="Apply to All", command=self.apply_to_all).pack(side=tk.LEFT, padx=2)

        # Bottom hint bar
        self.hint = tk.Label(self.root, text="", anchor="w", bd=1, relief=tk.SUNKEN, padx=6, pady=2)
        self.hint.pack(side=tk.BOTTOM, fill=tk.X)
        self._update_hint()

        self.canvas = tk.Canvas(self.root, bg="gray20", cursor="cross")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Configure>", lambda _e: self._render_image())

    def _update_hint(self) -> None:
        mode = self.mode.get()
        if mode == "rect":
            self.hint.config(text="Click-drag empty area to draw. Drag inside to move. Drag handles to resize.")
        elif mode == "poly":
            self.hint.config(
                text="Left-click to add vertices. Right-click to close. "
                "After closing: drag vertices/body to edit, right-click vertex to delete."
            )
        else:  # bitmap
            self.hint.config(
                text="Bitmap mode: click Load Mask to load a saved mask image. "
                "Switch to Rectangle/Polygon to draw a new one."
            )

    def _on_mode_change(self) -> None:
        self.reset_selection()
        self._update_hint()

    # ------------------------------------------------------------------
    # Image I/O + rendering
    # ------------------------------------------------------------------

    def open_folder(self) -> None:
        path = filedialog.askdirectory(title="Select folder of images")
        if not path:
            return
        folder = Path(path)
        files = sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.parent.name != OUTPUT_SUBDIR
        )
        if not files:
            messagebox.showwarning("Empty", f"No image files found in:\n{folder}")
            return
        self.folder = folder
        self.image_paths = files
        self.image = None
        self.image_path = None
        self.current_idx = -1
        self.reset_selection()
        self._load_image(0)

    def _load_image(self, idx: int) -> None:
        if not self.image_paths:
            return
        idx = max(0, min(idx, len(self.image_paths) - 1))
        new_path = self.image_paths[idx]
        try:
            new_image = Image.open(new_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not open {new_path.name}:\n{e}")
            return

        # Rescale existing mask if the new image has different dimensions, so the
        # mask stays anchored to roughly the same region across the folder.
        if self.image is not None and self.image.size != new_image.size:
            old_w, old_h = self.image.size
            new_w, new_h = new_image.size
            sx, sy = new_w / old_w, new_h / old_h
            if self.box is not None:
                x0, y0, x1, y1 = self.box
                self.box = [
                    int(round(x0 * sx)),
                    int(round(y0 * sy)),
                    int(round(x1 * sx)),
                    int(round(y1 * sy)),
                ]
            if self.polygon:
                self.polygon = [[int(round(x * sx)), int(round(y * sy))] for x, y in self.polygon]
            if self.bitmap_mask is not None:
                self.bitmap_mask = self.bitmap_mask.resize((new_w, new_h), Image.NEAREST)

        self.image = new_image
        self.image_path = new_path
        self.current_idx = idx
        self._update_status()
        self._render_image()

    def _update_status(self) -> None:
        if self.image_path is None or not self.image_paths:
            self.status.config(text="No folder loaded")
        else:
            self.status.config(text=f"[{self.current_idx + 1}/{len(self.image_paths)}] {self.image_path.name}")

    def prev_image(self) -> None:
        if self.image_paths and self.current_idx > 0:
            self._load_image(self.current_idx - 1)

    def next_image(self) -> None:
        if self.image_paths and self.current_idx < len(self.image_paths) - 1:
            self._load_image(self.current_idx + 1)

    def _render_image(self) -> None:
        if self.image is None:
            return
        self.root.update_idletasks()
        cw = max(self.canvas.winfo_width(), 800)
        ch = max(self.canvas.winfo_height(), 600)
        iw, ih = self.image.size
        self.scale = min(cw / iw, ch / ih, 1.0)
        dw, dh = max(int(iw * self.scale), 1), max(int(ih * self.scale), 1)
        resized = self.image.resize((dw, dh), Image.LANCZOS)
        if self.mode.get() == "bitmap" and self.bitmap_mask is not None:
            mask_resized = self.bitmap_mask.resize((dw, dh), Image.NEAREST)
            # Darken pixels outside the mask region so the user sees the effect directly.
            overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
            overlay.putalpha(mask_resized.point(lambda v: 0 if v > 127 else 140))
            resized = Image.alpha_composite(resized.convert("RGBA"), overlay).convert("RGB")
        self.display_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image, tags="img")
        self._redraw_selection()

    def reset_selection(self) -> None:
        self.box = None
        self.polygon = []
        self.polygon_closed = False
        self.bitmap_mask = None
        self.drag_mode = None
        self.drag_vertex_idx = None
        self.canvas.delete("sel")
        if self.image is not None:
            self._render_image()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_image_coords(self, x: int, y: int) -> tuple[int, int]:
        assert self.image is not None
        iw, ih = self.image.size
        ix = int(round(min(max(x / self.scale, 0), iw)))
        iy = int(round(min(max(y / self.scale, 0), ih)))
        return ix, iy

    def _to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.scale, iy * self.scale

    @staticmethod
    def _normalize_box(box: list[int]) -> list[int]:
        x0, y0, x1, y1 = box
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    # ------------------------------------------------------------------
    # Hit tests (canvas coords in, result interpreted in image space)
    # ------------------------------------------------------------------

    def _hit_rect_handle(self, cx: int, cy: int) -> Optional[str]:
        if self.box is None:
            return None
        x0, y0, x1, y1 = self.box
        cx0, cy0 = self._to_canvas(x0, y0)
        cx1, cy1 = self._to_canvas(x1, y1)
        points = {
            "nw": (cx0, cy0),
            "ne": (cx1, cy0),
            "sw": (cx0, cy1),
            "se": (cx1, cy1),
            "n": ((cx0 + cx1) / 2, cy0),
            "s": ((cx0 + cx1) / 2, cy1),
            "w": (cx0, (cy0 + cy1) / 2),
            "e": (cx1, (cy0 + cy1) / 2),
        }
        for key, (hx, hy) in points.items():
            if abs(cx - hx) <= HANDLE_SIZE and abs(cy - hy) <= HANDLE_SIZE:
                return key
        return None

    def _inside_box(self, cx: int, cy: int) -> bool:
        if self.box is None:
            return False
        x0, y0, x1, y1 = self.box
        cx0, cy0 = self._to_canvas(x0, y0)
        cx1, cy1 = self._to_canvas(x1, y1)
        return cx0 < cx < cx1 and cy0 < cy < cy1

    def _hit_polygon_vertex(self, cx: int, cy: int) -> Optional[int]:
        for i, (ix, iy) in enumerate(self.polygon):
            vx, vy = self._to_canvas(ix, iy)
            if abs(cx - vx) <= HANDLE_SIZE and abs(cy - vy) <= HANDLE_SIZE:
                return i
        return None

    def _inside_polygon(self, cx: int, cy: int) -> bool:
        if not self.polygon_closed or len(self.polygon) < 3:
            return False
        verts = [self._to_canvas(ix, iy) for ix, iy in self.polygon]
        inside = False
        n = len(verts)
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / ((yj - yi) or 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def on_hover(self, event: tk.Event) -> None:
        if self.image is None:
            self.canvas.config(cursor="arrow")
            return
        mode = self.mode.get()
        if mode == "rect":
            handle = self._hit_rect_handle(event.x, event.y)
            if handle:
                self.canvas.config(cursor=RECT_HANDLES[handle][0])
            elif self._inside_box(event.x, event.y):
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="cross")
        elif mode == "poly":
            if self._hit_polygon_vertex(event.x, event.y) is not None:
                self.canvas.config(cursor="hand2")
            elif self._inside_polygon(event.x, event.y):
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="cross")
        else:  # bitmap — no interactive editing
            self.canvas.config(cursor="arrow")

    def on_press(self, event: tk.Event) -> None:
        if self.image is None:
            return
        mode = self.mode.get()
        if mode == "rect":
            self._press_rect(event)
        elif mode == "poly":
            self._press_poly(event)
        # bitmap: no-op

    def on_drag(self, event: tk.Event) -> None:
        if self.image is None or self.drag_mode is None:
            return
        mode = self.mode.get()
        if mode == "rect":
            self._drag_rect(event)
        elif mode == "poly":
            self._drag_poly(event)

    def on_release(self, _event: tk.Event) -> None:
        if self.mode.get() == "rect" and self.box is not None:
            self.box = self._normalize_box(self.box)
            x0, y0, x1, y1 = self.box
            if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
                self.box = None
            self._redraw_selection()
        self.drag_mode = None
        self.drag_anchor = None
        self.drag_box_start = None
        self.drag_vertex_idx = None
        self.drag_poly_start = None

    def on_right_press(self, event: tk.Event) -> None:
        if self.image is None or self.mode.get() != "poly":
            return
        if not self.polygon_closed:
            # Close the polygon if we have enough vertices.
            if len(self.polygon) >= MIN_POLY_VERTICES:
                self.polygon_closed = True
                self._redraw_selection()
            return
        # Closed: right-click on a vertex deletes it (keep at least MIN).
        idx = self._hit_polygon_vertex(event.x, event.y)
        if idx is None or len(self.polygon) <= MIN_POLY_VERTICES:
            return
        del self.polygon[idx]
        self._redraw_selection()

    # --- rectangle press/drag ---

    def _press_rect(self, event: tk.Event) -> None:
        handle = self._hit_rect_handle(event.x, event.y)
        if handle:
            self.drag_mode = handle
            self.drag_box_start = list(self.box) if self.box else None
            return
        if self._inside_box(event.x, event.y):
            self.drag_mode = "move"
            self.drag_anchor = self._to_image_coords(event.x, event.y)
            self.drag_box_start = list(self.box) if self.box else None
            return
        ix, iy = self._to_image_coords(event.x, event.y)
        self.drag_mode = "draw"
        self.box = [ix, iy, ix, iy]
        self._redraw_selection()

    def _drag_rect(self, event: tk.Event) -> None:
        assert self.image is not None
        if self.box is None:
            return
        ix, iy = self._to_image_coords(event.x, event.y)
        iw, ih = self.image.size

        if self.drag_mode == "draw":
            self.box[2] = ix
            self.box[3] = iy
        elif self.drag_mode == "move":
            assert self.drag_anchor is not None and self.drag_box_start is not None
            dx = ix - self.drag_anchor[0]
            dy = iy - self.drag_anchor[1]
            x0, y0, x1, y1 = self.drag_box_start
            w, h = x1 - x0, y1 - y0
            nx0 = min(max(x0 + dx, 0), iw - w)
            ny0 = min(max(y0 + dy, 0), ih - h)
            self.box = [nx0, ny0, nx0 + w, ny0 + h]
        else:
            _, ex, ey = RECT_HANDLES[self.drag_mode]
            x0, y0, x1, y1 = self.box
            if ex == -1:
                x0 = ix
            elif ex == 1:
                x1 = ix
            if ey == -1:
                y0 = iy
            elif ey == 1:
                y1 = iy
            self.box = [x0, y0, x1, y1]

        self._redraw_selection()

    # --- polygon press/drag ---

    def _press_poly(self, event: tk.Event) -> None:
        if self.polygon_closed:
            idx = self._hit_polygon_vertex(event.x, event.y)
            if idx is not None:
                self.drag_mode = "vertex"
                self.drag_vertex_idx = idx
                return
            if self._inside_polygon(event.x, event.y):
                self.drag_mode = "move"
                self.drag_anchor = self._to_image_coords(event.x, event.y)
                self.drag_poly_start = [list(v) for v in self.polygon]
                return
            # Click outside a closed polygon — ignore (use Reset to start over).
            return
        # Open construction phase: add a vertex.
        self.polygon.append(list(self._to_image_coords(event.x, event.y)))
        self._redraw_selection()

    def _drag_poly(self, event: tk.Event) -> None:
        assert self.image is not None
        ix, iy = self._to_image_coords(event.x, event.y)
        if self.drag_mode == "vertex":
            assert self.drag_vertex_idx is not None
            self.polygon[self.drag_vertex_idx] = [ix, iy]
        elif self.drag_mode == "move":
            assert self.drag_anchor is not None and self.drag_poly_start is not None
            dx = ix - self.drag_anchor[0]
            dy = iy - self.drag_anchor[1]
            iw, ih = self.image.size
            new_poly = []
            for sx, sy in self.drag_poly_start:
                nx = min(max(sx + dx, 0), iw)
                ny = min(max(sy + dy, 0), ih)
                new_poly.append([nx, ny])
            self.polygon = new_poly
        self._redraw_selection()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw_selection(self) -> None:
        self.canvas.delete("sel")
        mode = self.mode.get()
        if mode == "rect":
            self._draw_rect()
        elif mode == "poly":
            self._draw_poly()
        # bitmap: overlay is composited in _render_image, no canvas overlay needed

    def _draw_rect(self) -> None:
        if self.box is None:
            return
        x0, y0, x1, y1 = self.box
        cx0, cy0 = self._to_canvas(x0, y0)
        cx1, cy1 = self._to_canvas(x1, y1)
        self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="red", width=2, tags="sel")
        points = [
            (cx0, cy0),
            (cx1, cy0),
            (cx0, cy1),
            (cx1, cy1),
            ((cx0 + cx1) / 2, cy0),
            ((cx0 + cx1) / 2, cy1),
            (cx0, (cy0 + cy1) / 2),
            (cx1, (cy0 + cy1) / 2),
        ]
        for hx, hy in points:
            self.canvas.create_rectangle(
                hx - HANDLE_SIZE,
                hy - HANDLE_SIZE,
                hx + HANDLE_SIZE,
                hy + HANDLE_SIZE,
                fill="white",
                outline="red",
                tags="sel",
            )

    def _draw_poly(self) -> None:
        if not self.polygon:
            return
        canvas_pts = [self._to_canvas(ix, iy) for ix, iy in self.polygon]

        # Edges
        if self.polygon_closed and len(canvas_pts) >= 3:
            flat = [c for pt in canvas_pts for c in pt]
            self.canvas.create_polygon(flat, outline="red", fill="", width=2, tags="sel")
        elif len(canvas_pts) >= 2:
            for (x0, y0), (x1, y1) in zip(canvas_pts, canvas_pts[1:]):
                self.canvas.create_line(x0, y0, x1, y1, fill="red", width=2, tags="sel")

        # Vertices
        for i, (vx, vy) in enumerate(canvas_pts):
            fill = "#fffacd" if (not self.polygon_closed and i == 0) else "white"
            self.canvas.create_rectangle(
                vx - HANDLE_SIZE,
                vy - HANDLE_SIZE,
                vx + HANDLE_SIZE,
                vy + HANDLE_SIZE,
                fill=fill,
                outline="red",
                tags="sel",
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _has_mask(self) -> bool:
        mode = self.mode.get()
        if mode == "rect":
            return self.box is not None
        if mode == "poly":
            return self.polygon_closed and len(self.polygon) >= MIN_POLY_VERTICES
        return self.bitmap_mask is not None

    def _build_mask_for_size(self, size: tuple[int, int]) -> Optional[Image.Image]:
        """Build a binary mask ('L' mode, 0/255) sized to `size`, scaling from current image coords."""
        if self.image is None or not self._has_mask():
            return None
        mode = self.mode.get()
        if mode == "bitmap":
            assert self.bitmap_mask is not None
            if self.bitmap_mask.size == size:
                return self.bitmap_mask.copy()
            return self.bitmap_mask.resize(size, Image.NEAREST)

        ref_w, ref_h = self.image.size
        tw, th = size
        sx, sy = tw / ref_w, th / ref_h
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        if mode == "rect":
            assert self.box is not None
            x0, y0, x1, y1 = self.box
            draw.rectangle(
                [int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))],
                fill=255,
            )
        else:
            pts = [(int(round(x * sx)), int(round(y * sy))) for x, y in self.polygon]
            draw.polygon(pts, fill=255)
        return mask

    def _missing_region_warning(self) -> None:
        mode = self.mode.get()
        if mode == "rect":
            messagebox.showwarning("No region", "Draw a rectangle first.")
        elif mode == "poly":
            messagebox.showwarning(
                "No region",
                f"Draw at least {MIN_POLY_VERTICES} vertices and right-click to close the polygon.",
            )
        else:
            messagebox.showwarning("No mask", "Click Load Mask to load a saved mask image.")

    def load_mask(self) -> None:
        if self.image is None:
            messagebox.showwarning("No image", "Open a folder first.")
            return
        path = filedialog.askopenfilename(
            title="Load mask image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            mask = Image.open(path).convert("L")
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not open mask:\n{e}")
            return
        # Threshold to binary, then conform to current image size.
        mask = mask.point(lambda v: 255 if v > 127 else 0)
        if mask.size != self.image.size:
            mask = mask.resize(self.image.size, Image.NEAREST)
        # Switch to bitmap mode and clear any drawn shapes.
        self.box = None
        self.polygon = []
        self.polygon_closed = False
        self.bitmap_mask = mask
        self.mode.set("bitmap")
        self._update_hint()
        self._render_image()

    def save_masked(self) -> None:
        if self.image is None:
            messagebox.showwarning("No image", "Open a folder first.")
            return
        if not self._has_mask():
            self._missing_region_warning()
            return

        mask = self._build_mask_for_size(self.image.size)
        assert mask is not None
        result = Image.merge("RGB", (mask, mask, mask))

        assert self.image_path is not None
        default_name = f"{self.image_path.stem}_mask.png"
        out_path = filedialog.asksaveasfilename(
            initialdir=str(self.image_path.parent),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not out_path:
            return
        result.save(out_path)
        messagebox.showinfo("Saved", f"Saved binary mask to:\n{out_path}")

    def apply_to_all(self) -> None:
        if self.folder is None or not self.image_paths:
            messagebox.showwarning("No folder", "Open a folder first.")
            return
        if not self._has_mask():
            self._missing_region_warning()
            return

        out_dir = self.folder / OUTPUT_SUBDIR
        if not messagebox.askyesno(
            "Apply to All",
            f"Apply mask to {len(self.image_paths)} image(s)?\n\nOutput folder:\n{out_dir}",
        ):
            return
        out_dir.mkdir(exist_ok=True)

        ok = 0
        failed: list[tuple[str, str]] = []
        for p in self.image_paths:
            try:
                img = Image.open(p).convert("RGB")
                mask = self._build_mask_for_size(img.size)
                if mask is None:
                    failed.append((p.name, "mask build failed"))
                    continue
                black = Image.new("RGB", img.size, (0, 0, 0))
                masked = Image.composite(img, black, mask)
                masked.save(out_dir / p.name)
                ok += 1
            except Exception as e:
                failed.append((p.name, str(e)))

        msg = f"Saved {ok} masked image(s) to:\n{out_dir}"
        if failed:
            preview = "\n".join(f"- {n}: {e}" for n, e in failed[:5])
            more = f"\n... and {len(failed) - 5} more" if len(failed) > 5 else ""
            msg += f"\n\nFailed: {len(failed)}\n{preview}{more}"
        messagebox.showinfo("Apply to All", msg)


def main() -> None:
    root = tk.Tk()
    root.geometry("1000x720")
    RegionMaskGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
