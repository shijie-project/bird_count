"""Draw each camera's region mask on one of its own frames.

    python -m birdcount.deploy.tools.mask_editor                  # every configured stream
    python -m birdcount.deploy.tools.mask_editor --image f.jpg --mac B8A44FD51C3C

Steps through the streams the runtime would open (SOURCE_TYPE: the
topology's cameras live, or the clips in VIDEO_PATHS), grabbing one frame
each, with the camera's current mask from MASK_DIR shown over it. Draw the
area to count as a rectangle or a polygon; Save writes it where the runtime
reads it — MASK_DIR/<MAC>.png at frame geometry, white = count — replacing
that camera's existing mask file if it has one.

The preview tints the ignored area exactly as the live monitor's mask
overlay does, so what you see here is what the monitor will show.

    rectangle  drag on empty space to draw; drag inside to move; drag handles to resize
    polygon    click to add points; right-click to close; then drag points or
               the body, right-click a point to delete it
    keys       Left / Right: previous / next camera; Ctrl+S: save; Esc: discard the drawing
"""

from __future__ import annotations

import argparse
import logging
import sys
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from birdcount.core.camera_identity import index_by_mac, normalize_mac
from birdcount.core.frames import fit_frame
from birdcount.core.log import setup_logging
from birdcount.core.masks import read_keep_mask
from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.core.visualize import mask_layers, overlay_mask


logger = logging.getLogger(__name__)

HANDLE = 6  # px, half-size of a drag handle on screen
MIN_BOX = 5  # px at frame size
MIN_POINTS = 3
SEEK_SECONDS = 5.0  # where in a clip the frame is taken

# (cursor, x edge, y edge): -1 = min edge, 1 = max edge, 0 = neither
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


# ----------------------------------------------------------------------------
# The region, in frame pixels: no Tk
# ----------------------------------------------------------------------------


@dataclass
class Region:
    """The area to count: a rectangle, a polygon, or a mask loaded from file."""

    mode: str = "rect"  # rect | poly | file
    box: list[int] | None = None  # x0, y0, x1, y1
    points: list[list[int]] | None = None
    closed: bool = False
    keep: np.ndarray | None = None  # mode "file": (H, W) uint8, 1 = count

    @property
    def complete(self) -> bool:
        if self.mode == "rect":
            return self.box is not None
        if self.mode == "poly":
            return self.closed and len(self.points or []) >= MIN_POINTS
        return self.keep is not None

    def render(self, size: tuple[int, int]) -> np.ndarray:
        """(H, W) uint8 keep-mask at `size` = (width, height); all zeros if incomplete."""
        width, height = size
        if self.mode == "file" and self.keep is not None:
            if self.keep.shape == (height, width):
                return self.keep.copy()
            return cv2.resize(self.keep, size, interpolation=cv2.INTER_NEAREST)
        keep = np.zeros((height, width), np.uint8)
        if self.mode == "rect" and self.box is not None:
            x0, y0, x1, y1 = normalized(self.box)
            keep[y0:y1, x0:x1] = 1
        elif self.mode == "poly" and self.complete:
            cv2.fillPoly(keep, [np.array(self.points, np.int32)], 1)
        return keep


def normalized(box: list[int]) -> list[int]:
    x0, y0, x1, y1 = box
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def inside_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    inside, j = False, len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def rect_handles(box: list[int]) -> dict[str, tuple[float, float]]:
    x0, y0, x1, y1 = box
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return {
        "nw": (x0, y0),
        "n": (xm, y0),
        "ne": (x1, y0),
        "e": (x1, ym),
        "se": (x1, y1),
        "s": (xm, y1),
        "sw": (x0, y1),
        "w": (x0, ym),
    }


def ppm(frame_bgr: np.ndarray) -> bytes:
    """A Tk-loadable image of a BGR frame (PPM: no encoding, unlike PNG)."""
    h, w = frame_bgr.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).tobytes()


# ----------------------------------------------------------------------------
# What to draw on
# ----------------------------------------------------------------------------


@dataclass
class Target:
    label: str
    mac: str | None
    load: Callable[[], np.ndarray | None]  # a BGR frame, or None if unavailable


def frame_from_source(uri: str, is_file: bool) -> np.ndarray | None:
    cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    try:
        if is_file:
            cap.set(cv2.CAP_PROP_POS_MSEC, SEEK_SECONDS * 1000)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def stream_targets() -> tuple[list[Target], Path | None, tuple[int, int]]:
    from birdcount.deploy.settings import DeploySettings
    from birdcount.deploy.streams import resolve_streams
    from birdcount.deploy.topology import load_topology

    settings = DeploySettings.load()
    streams = resolve_streams(settings, load_topology(settings.topology_path))
    targets = [
        Target(f"#{s.sid} {s.label}", s.mac, lambda s=s: frame_from_source(s.source.uri, s.source.is_file))
        for s in streams
    ]
    return targets, settings.mask_dir, settings.frame_size


# ----------------------------------------------------------------------------
# The window
# ----------------------------------------------------------------------------


class Editor:
    def __init__(self, root: tk.Tk, targets: list[Target], mask_dir: Path | None, frame_size: tuple[int, int]):
        self.root, self.targets, self.mask_dir, self.frame_size = root, targets, mask_dir, frame_size
        self.index = -1
        self.frame: np.ndarray | None = None  # BGR at frame_size
        self.saved: Path | None = None  # the current camera's mask file
        self.region = Region()
        self.scale = 1.0
        self._photo = None
        self._drag: tuple | None = None
        self._dirty = False
        self.mode = tk.StringVar(value="rect")
        self._build()
        self.go(0)

    # --- layout -----------------------------------------------------------

    def _build(self) -> None:
        self.root.title("Region Mask Editor")
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        tk.Button(bar, text="◀", width=3, command=lambda: self.go(self.index - 1)).pack(side=tk.LEFT)
        tk.Button(bar, text="▶", width=3, command=lambda: self.go(self.index + 1)).pack(side=tk.LEFT)
        for text, value in (("Rectangle", "rect"), ("Polygon", "poly")):
            tk.Radiobutton(bar, text=text, variable=self.mode, value=value, command=self.new_drawing).pack(
                side=tk.LEFT, padx=(8 if value == "rect" else 0, 0)
            )
        tk.Button(bar, text="Save", command=self.save).pack(side=tk.LEFT, padx=(12, 2))
        tk.Button(bar, text="Revert", command=self.revert).pack(side=tk.LEFT, padx=2)
        self.status = tk.Label(bar, anchor="w")
        self.status.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        self.hint = tk.Label(self.root, anchor="w", bd=1, relief=tk.SUNKEN, padx=6)
        self.hint.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas = tk.Canvas(self.root, bg="gray20", cursor="cross", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Configure>", lambda _e: self.render())
        self.root.bind("<Left>", lambda _e: self.go(self.index - 1))
        self.root.bind("<Right>", lambda _e: self.go(self.index + 1))
        self.root.bind("<Control-s>", lambda _e: self.save())
        self.root.bind("<Escape>", lambda _e: self.revert())

    # --- cameras ----------------------------------------------------------

    def go(self, index: int) -> None:
        if not 0 <= index < len(self.targets) or index == self.index:
            return
        if self._dirty and not messagebox.askyesno("Unsaved mask", "Discard the unsaved mask?"):
            return
        target = self.targets[index]
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        frame = target.load()
        self.root.config(cursor="")
        self.index = index
        self.frame = None if frame is None else fit_frame(frame, self.frame_size)
        self.saved = self._mask_file(target.mac)
        self.revert()

    def _mask_file(self, mac: str | None) -> Path | None:
        if not mac or self.mask_dir is None or not self.mask_dir.is_dir():
            return None
        unique, _ = index_by_mac(p for p in self.mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        return unique.get(mac)

    def revert(self) -> None:
        """Back to the camera's saved mask (or nothing)."""
        self.region = Region(mode=self.mode.get())
        if self.saved is not None:
            try:
                self.region = Region(mode="file", keep=read_keep_mask(self.saved, self.frame_size))
            except (OSError, ValueError) as e:
                messagebox.showerror("Mask unreadable", f"{self.saved.name}: {e}")
        self._dirty = False
        self.render()

    def new_drawing(self) -> None:
        self.region = Region(mode=self.mode.get())
        self._dirty = False
        self.render()

    def save(self) -> None:
        if self.frame is None or not self.region.complete:
            messagebox.showwarning("Nothing to save", "Draw a rectangle, or a polygon closed with a right-click.")
            return
        target = self.targets[self.index]
        path = self.saved
        if path is None and target.mac and self.mask_dir is not None:
            path = self.mask_dir / f"{target.mac}.png"
        if path is None:
            name = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if not name:
                return
            path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = self.region.render(self.frame_size)
        if not cv2.imwrite(str(path), keep * np.uint8(255)):
            messagebox.showerror("Save failed", f"Could not write {path}")
            return
        self.saved, self._dirty = path, False
        self.region = Region(mode="file", keep=keep)
        logger.info("Saved %s (%.1f%% counted).", path, keep.mean() * 100)
        self.render()

    # --- drawing ----------------------------------------------------------

    def render(self) -> None:
        """The frame, the region's preview tint, and the editable outline."""
        self.canvas.delete("all")
        target = self.targets[self.index] if self.index >= 0 else None
        if self.frame is None:
            if target is not None:
                self.canvas.create_text(20, 20, anchor=tk.NW, fill="white", text=f"No frame from {target.label}")
            self._update_labels()
            return
        cw, ch = max(self.canvas.winfo_width(), 320), max(self.canvas.winfo_height(), 240)
        w, h = self.frame_size
        self.scale = min(cw / w, ch / h)
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        view = cv2.resize(self.frame, size, interpolation=cv2.INTER_AREA)
        if self.region.complete and self._drag is None:
            overlay_mask(view, mask_layers(self.region.render(self.frame_size), size))
        self._photo = tk.PhotoImage(data=ppm(view), format="PPM")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        self._draw_outline()
        self._update_labels()

    def _draw_outline(self) -> None:
        self.canvas.delete("shape")
        r, s = self.region, self.scale
        if r.mode == "rect" and r.box is not None:
            x0, y0, x1, y1 = (v * s for v in r.box)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2, tags="shape")
            for hx, hy in rect_handles(r.box).values():
                self._handle(hx * s, hy * s, "white")
        elif r.mode == "poly" and r.points:
            pts = [(x * s, y * s) for x, y in r.points]
            if r.closed:
                self.canvas.create_polygon(*[c for p in pts for c in p], outline="red", fill="", width=2, tags="shape")
            elif len(pts) > 1:
                self.canvas.create_line(*[c for p in pts for c in p], fill="red", width=2, tags="shape")
            for i, (x, y) in enumerate(pts):
                self._handle(x, y, "#fffacd" if i == 0 and not r.closed else "white")

    def _handle(self, x: float, y: float, fill: str) -> None:
        self.canvas.create_rectangle(
            x - HANDLE, y - HANDLE, x + HANDLE, y + HANDLE, fill=fill, outline="red", tags="shape"
        )

    def _update_labels(self) -> None:
        if self.index < 0:
            return
        t = self.targets[self.index]
        saved = self.saved.name if self.saved else "none"
        self.status.config(
            text=f"[{self.index + 1}/{len(self.targets)}] {t.label}   MAC {t.mac or '?'}   saved mask: {saved}"
            + ("   (unsaved changes)" if self._dirty else "")
        )
        hints = {
            "rect": "Drag on empty space to draw; drag inside to move; drag handles to resize.",
            "poly": "Click to add points, right-click to close; then drag points or the body; right-click a point to delete it.",
            "file": "Showing the saved mask. Draw a new rectangle or polygon to replace it.",
        }
        self.hint.config(text=hints[self.region.mode])

    # --- mouse ------------------------------------------------------------

    def _image_xy(self, event) -> tuple[int, int]:
        w, h = self.frame_size
        return (
            round(min(max(event.x / self.scale, 0), w)),
            round(min(max(event.y / self.scale, 0), h)),
        )

    def _hit_handle(self, event) -> str | None:
        if self.region.box is None:
            return None
        for key, (hx, hy) in rect_handles(self.region.box).items():
            if abs(event.x - hx * self.scale) <= HANDLE and abs(event.y - hy * self.scale) <= HANDLE:
                return key
        return None

    def _hit_point(self, event) -> int | None:
        for i, (x, y) in enumerate(self.region.points or []):
            if abs(event.x - x * self.scale) <= HANDLE and abs(event.y - y * self.scale) <= HANDLE:
                return i
        return None

    def _inside(self, event) -> bool:
        r, (x, y) = self.region, self._image_xy(event)
        if r.mode == "rect" and r.box is not None:
            x0, y0, x1, y1 = normalized(r.box)
            return x0 < x < x1 and y0 < y < y1
        return r.mode == "poly" and r.complete and inside_polygon(x, y, r.points)

    def _start_shape(self) -> None:
        """First click on a saved mask: begin a new drawing in the selected mode."""
        if self.region.mode == "file":
            self.region = Region(mode=self.mode.get())

    def on_hover(self, event) -> None:
        if self.frame is None or self.region.mode == "file":
            self.canvas.config(cursor="cross")
        elif self.region.mode == "rect" and (handle := self._hit_handle(event)):
            self.canvas.config(cursor=RECT_HANDLES[handle][0])
        elif self.region.mode == "poly" and self.region.closed and self._hit_point(event) is not None:
            self.canvas.config(cursor="hand2")
        else:
            self.canvas.config(cursor="fleur" if self._inside(event) else "cross")

    def on_press(self, event) -> None:
        if self.frame is None:
            return
        self._start_shape()
        r, xy = self.region, self._image_xy(event)
        if r.mode == "rect":
            if handle := self._hit_handle(event):
                self._drag = ("handle", handle)
            elif self._inside(event):
                self._drag = ("move", xy, list(r.box))
            else:
                r.box = [*xy, *xy]
                self._drag = ("draw",)
        elif r.closed:
            i = self._hit_point(event)
            if i is not None:
                self._drag = ("point", i)
            elif self._inside(event):
                self._drag = ("move", xy, [list(p) for p in r.points])
        else:
            r.points = [*(r.points or []), list(xy)]
            self._dirty = True
        self._draw_outline()
        self._update_labels()

    def on_drag(self, event) -> None:
        if self._drag is None:
            return
        r, (x, y) = self.region, self._image_xy(event)
        w, h = self.frame_size
        kind = self._drag[0]
        if kind == "draw":
            r.box[2:] = [x, y]
        elif kind == "handle":
            _, ex, ey = RECT_HANDLES[self._drag[1]]
            if ex:
                r.box[0 if ex < 0 else 2] = x
            if ey:
                r.box[1 if ey < 0 else 3] = y
        elif kind == "point":
            r.points[self._drag[1]] = [x, y]
        elif kind == "move":
            (ax, ay), start = self._drag[1], self._drag[2]
            if r.mode == "rect":
                x0, y0, x1, y1 = normalized(start)
                dx = min(max(x - ax, -x0), w - x1)
                dy = min(max(y - ay, -y0), h - y1)
                r.box = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
            else:
                r.points = [[min(max(px + x - ax, 0), w), min(max(py + y - ay, 0), h)] for px, py in start]
        self._dirty = True
        self._draw_outline()

    def on_release(self, _event) -> None:
        if self._drag is None:
            return
        self._drag = None
        r = self.region
        if r.mode == "rect" and r.box is not None:
            r.box = normalized(r.box)
            if r.box[2] - r.box[0] < MIN_BOX or r.box[3] - r.box[1] < MIN_BOX:
                r.box = None
        self.render()

    def on_right(self, event) -> None:
        r = self.region
        if r.mode != "poly" or not r.points:
            return
        if not r.closed:
            if len(r.points) >= MIN_POINTS:
                r.closed = True
                self.render()
            return
        i = self._hit_point(event)
        if i is not None and len(r.points) > MIN_POINTS:
            del r.points[i]
            self._dirty = True
            self.render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", type=Path, help="draw on this image instead of the configured streams")
    parser.add_argument("--mac", help="with --image: the camera it shows (names the saved mask)")
    args = parser.parse_args(argv)
    setup_logging("INFO")

    if args.image is not None:
        from birdcount.deploy.settings import DeploySettings

        settings = DeploySettings.load()
        mac = normalize_mac(args.mac or "") or None
        targets = [Target(args.image.name, mac, lambda: cv2.imread(str(args.image)))]
        mask_dir, frame_size = settings.mask_dir, settings.frame_size
    else:
        try:
            targets, mask_dir, frame_size = stream_targets()
        except ValueError as e:  # topology / stream configuration
            print(e, file=sys.stderr)
            return 2
    root = tk.Tk()
    root.geometry("1100x720")
    Editor(root, targets, mask_dir, frame_size)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
