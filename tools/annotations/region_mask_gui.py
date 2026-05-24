import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk


HANDLE_SIZE = 6
MIN_BOX = 5
MIN_POLY_VERTICES = 3

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

        self.image: Image.Image | None = None
        self.image_path: Path | None = None
        self.display_image: ImageTk.PhotoImage | None = None
        self.scale = 1.0

        # Shape state — only one shape exists at a time.
        self.mode = tk.StringVar(value="rect")
        # Rectangle, image coords: [x0, y0, x1, y1] with x0<x1, y0<y1
        self.box: list[int] | None = None
        # Polygon, image coords: [[x, y], ...]
        self.polygon: list[list[int]] = []
        self.polygon_closed = False

        # Drag state
        self.drag_mode: str | None = None  # rect: "draw" | "move" | handle key
        self.drag_anchor: tuple[int, int] | None = None
        self.drag_box_start: list[int] | None = None
        self.drag_vertex_idx: int | None = None
        self.drag_poly_start: list[list[int]] | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Radiobutton(toolbar, text="Rectangle", variable=self.mode, value="rect", command=self._on_mode_change).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        tk.Radiobutton(toolbar, text="Polygon", variable=self.mode, value="poly", command=self._on_mode_change).pack(
            side=tk.LEFT
        )

        tk.Button(toolbar, text="Reset", command=self.reset_selection).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Save Masked", command=self.save_masked).pack(side=tk.LEFT, padx=2)

        self.hint = tk.Label(toolbar, text="")
        self.hint.pack(side=tk.LEFT, padx=8)
        self._update_hint()

        self.canvas = tk.Canvas(self.root, bg="gray20", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Configure>", lambda _e: self._render_image())

    def _update_hint(self) -> None:
        if self.mode.get() == "rect":
            self.hint.config(text="  Click-drag empty area to draw. Drag inside to move. Drag handles to resize.")
        else:
            self.hint.config(
                text="  Left-click to add vertices. Right-click to close. "
                "After closing: drag vertices/body to edit, right-click vertex to delete."
            )

    def _on_mode_change(self) -> None:
        self.reset_selection()
        self._update_hint()

    # ------------------------------------------------------------------
    # Image I/O + rendering
    # ------------------------------------------------------------------

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        self.image_path = Path(path)
        self.image = Image.open(path).convert("RGB")
        self.reset_selection()
        self._render_image()

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
        self.display_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image, tags="img")
        self._redraw_selection()

    def reset_selection(self) -> None:
        self.box = None
        self.polygon = []
        self.polygon_closed = False
        self.drag_mode = None
        self.drag_vertex_idx = None
        self.canvas.delete("sel")

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

    def _hit_rect_handle(self, cx: int, cy: int) -> str | None:
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

    def _hit_polygon_vertex(self, cx: int, cy: int) -> int | None:
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
        if self.mode.get() == "rect":
            handle = self._hit_rect_handle(event.x, event.y)
            if handle:
                self.canvas.config(cursor=RECT_HANDLES[handle][0])
            elif self._inside_box(event.x, event.y):
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="cross")
        else:  # poly
            if self._hit_polygon_vertex(event.x, event.y) is not None:
                self.canvas.config(cursor="hand2")
            elif self._inside_polygon(event.x, event.y):
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="cross")

    def on_press(self, event: tk.Event) -> None:
        if self.image is None:
            return
        if self.mode.get() == "rect":
            self._press_rect(event)
        else:
            self._press_poly(event)

    def on_drag(self, event: tk.Event) -> None:
        if self.image is None or self.drag_mode is None:
            return
        if self.mode.get() == "rect":
            self._drag_rect(event)
        else:
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
        if self.mode.get() == "rect":
            self._draw_rect()
        else:
            self._draw_poly()

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

    def save_masked(self) -> None:
        if self.image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return

        if self.mode.get() == "rect":
            if self.box is None:
                messagebox.showwarning("No region", "Draw a rectangle first.")
                return
            result = Image.new("RGB", self.image.size, (0, 0, 0))
            x0, y0, x1, y1 = self.box
            ImageDraw.Draw(result).rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
        else:
            if not self.polygon_closed or len(self.polygon) < MIN_POLY_VERTICES:
                messagebox.showwarning(
                    "No region",
                    f"Draw at least {MIN_POLY_VERTICES} vertices and right-click to close the polygon.",
                )
                return
            result = Image.new("RGB", self.image.size, (0, 0, 0))
            ImageDraw.Draw(result).polygon([(x, y) for x, y in self.polygon], fill=(255, 255, 255))

        assert self.image_path is not None
        default_name = f"{self.image_path.stem}_masked{self.image_path.suffix}"
        out_path = filedialog.asksaveasfilename(
            initialdir=str(self.image_path.parent),
            initialfile=default_name,
            defaultextension=self.image_path.suffix,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not out_path:
            return
        result.save(out_path)
        messagebox.showinfo("Saved", f"Saved to:\n{out_path}")


def main() -> None:
    root = tk.Tk()
    root.geometry("1000x720")
    RegionMaskGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
