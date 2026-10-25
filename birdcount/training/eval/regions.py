"""Where in the frame the count goes wrong: region-level count errors.

Two ways to cut the density map into regions, one way to score them:

blob_regions   connected components of `density > min_density` (optionally
               merged across small gaps) — the warm areas of the heatmap. A
               region's count is the density integrated inside it: the
               model's own count decomposed by location, not a detection.
               Packed pile-ups connect into one blob, so for those...
grid_regions   ...a fixed rows x cols tiling always decomposes; tile counts
               sum to the frame total exactly.
assign_points  scores either: each human point is counted in the region its
               cell belongs to (optionally snapped to a blob a cell or two
               away, since a density peak and a click can disagree slightly).

Everything works on the native stride-8 density grid, so region counts are
true sums; `Region` carries frame-pixel geometry for drawing and reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from birdcount.training.data.targets import point_cells


@dataclass
class Region:
    id: int  # 1-based; labels == id
    count: float  # predicted birds (density sum)
    area_cells: int
    centroid: tuple[float, float]  # frame pixels
    bbox: tuple[float, float, float, float]  # frame pixels: x, y, width, height
    row: int | None = None  # grid tiles only, 1-based
    col: int | None = None
    gt: int | None = None  # set by assign_points
    error: float | None = None

    @property
    def abs_error(self) -> float | None:
        return None if self.error is None else abs(self.error)

    @property
    def relative_error(self) -> float | None:
        """Signed error as % of GT; None when the region holds no GT point."""
        return None if self.error is None or not self.gt else self.error / self.gt * 100.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["abs_error"], d["relative_error"] = self.abs_error, self.relative_error
        return {k: v for k, v in d.items() if v is not None or k in ("gt", "error", "relative_error")}


def _geometry(
    sel: np.ndarray, weights: np.ndarray | None, scale: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    ys, xs = np.nonzero(sel)
    if weights is not None and weights.sum() > 0:
        cx, cy = float((xs * weights).sum() / weights.sum()), float((ys * weights).sum() / weights.sum())
    else:
        cx, cy = float(xs.mean()), float(ys.mean())
    sx, sy = scale
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1  # cell edges, exclusive end
    return ((cx + 0.5) * sx, (cy + 0.5) * sy), (x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)


def blob_regions(
    density: np.ndarray, frame_size: tuple[int, int], *, min_density: float, merge: int = 0, min_count: float = 0.0
) -> tuple[np.ndarray, list[Region]]:
    """Connected warm areas, most crowded first.

    `merge` dilates the mask by that many cells first (gaps up to 2*merge
    cells close); counts still integrate the original density, so merging
    only re-absorbs sub-threshold mass. Regions under `min_count` birds are
    dropped. Returns (labels, regions) with labels 0 = background.
    """
    mask = (density > min_density).astype(np.uint8)
    if merge > 0:
        k = 2 * merge + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, raw = cv2.connectedComponents(mask, connectivity=8)

    found = []
    for i in range(1, n):
        sel = raw == i
        count = float(density[sel].sum())
        if count >= min_count:
            found.append((count, i))
    found.sort(key=lambda t: -t[0])

    gh, gw = density.shape
    scale = (frame_size[0] / gw, frame_size[1] / gh)
    labels = np.zeros(density.shape, dtype=np.int32)
    regions = []
    for new_id, (count, old_id) in enumerate(found, start=1):
        sel = raw == old_id
        labels[sel] = new_id
        centroid, bbox = _geometry(sel, density[sel], scale)
        regions.append(Region(id=new_id, count=count, area_cells=int(sel.sum()), centroid=centroid, bbox=bbox))
    return labels, regions


def grid_regions(
    density: np.ndarray, frame_size: tuple[int, int], rows: int, cols: int
) -> tuple[np.ndarray, list[Region]]:
    """A fixed rows x cols tiling, numbered row-major from 1; every tile kept."""
    gh, gw = density.shape
    if not (0 < rows <= gh and 0 < cols <= gw):
        raise ValueError(f"grid {rows}x{cols} does not fit the {gh}x{gw} density map")
    y_edges = np.linspace(0, gh, rows + 1).round().astype(int)
    x_edges = np.linspace(0, gw, cols + 1).round().astype(int)
    scale = (frame_size[0] / gw, frame_size[1] / gh)

    labels = np.zeros(density.shape, dtype=np.int32)
    regions = []
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = y_edges[r], y_edges[r + 1], x_edges[c], x_edges[c + 1]
            tile = r * cols + c + 1
            labels[y0:y1, x0:x1] = tile
            sel = labels == tile
            # Tile centre, not a density-weighted centroid: the number belongs
            # to the tile as a whole.
            centroid, bbox = _geometry(sel, None, scale)
            regions.append(
                Region(
                    id=tile,
                    count=float(density[y0:y1, x0:x1].sum()),
                    area_cells=int((y1 - y0) * (x1 - x0)),
                    centroid=centroid,
                    bbox=bbox,
                    row=r + 1,
                    col=c + 1,
                )
            )
    return labels, regions


def assign_points(labels: np.ndarray, regions: list[Region], points: np.ndarray, stride: int, *, snap: int = 0) -> int:
    """Set each region's GT and error from the points; returns how many points fell in no region.

    A point on background is moved to the nearest region within `snap` cells,
    if any; farther ones stay unassigned (likely missed birds).
    """
    gh, gw = labels.shape
    rows, cols = point_cells(points, stride, (gh, gw))
    owner = labels[rows, cols].copy()
    if snap > 0:
        for k in np.flatnonzero(owner == 0):
            y, x = int(rows[k]), int(cols[k])
            y0, x0 = max(0, y - snap), max(0, x - snap)
            ny, nx = np.nonzero(labels[y0 : y + snap + 1, x0 : x + snap + 1])
            if len(nx):
                nearest = int(np.argmin((nx + x0 - x) ** 2 + (ny + y0 - y) ** 2))
                owner[k] = labels[ny[nearest] + y0, nx[nearest] + x0]

    counts = np.bincount(owner, minlength=len(regions) + 1)
    for region in regions:
        region.gt = int(counts[region.id])
        region.error = region.count - region.gt
    return int(counts[0])
