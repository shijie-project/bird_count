"""Dataset hygiene on files: points in masked-out areas, duplicate images, baking masks in.

numpy + cv2 only.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from birdcount.core.camera_identity import extract_mac, index_by_mac
from birdcount.core.frames import read_image
from birdcount.core.masks import read_keep_mask
from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.training.labeling.ls_format import task_file


# -- points inside blacked-out regions --------------------------------------------------


def in_black_patch(image: np.ndarray, x: float, y: float, radius: int, black: int) -> bool:
    """Whether the (2r+1)^2 patch around (x, y) is black in every channel.

    A patch, not a pixel: a dark bird or shadow can have one black pixel, a
    masked-out region is solid black.
    """
    h, w = image.shape[:2]
    xi, yi = min(max(round(x), 0), w - 1), min(max(round(y), 0), h - 1)
    patch = image[max(yi - radius, 0) : yi + radius + 1, max(xi - radius, 0) : xi + radius + 1]
    return bool(patch.max() <= black)


@dataclass
class DropReport:
    before: int
    after: int
    per_image: list[tuple[str, int, int]]  # (image id, dropped, kept)
    missing: list[str]  # image ids whose file was not found (points kept)


def drop_masked_points(dataset: dict, images_dir: Path, *, radius: int = 1, black: int = 10) -> DropReport:
    """Remove, in place, every point lying in a solid-black (masked-out) patch of its image."""
    files = {Path(im["file_name"]).stem: im["file_name"] for im in dataset.get("images", [])}
    before = after = 0
    per_image, missing = [], []
    for ann in dataset.get("annotations", []):
        stem, points = str(ann["image_id"]), ann.get("points") or []
        before += len(points)
        path = images_dir / files[stem] if stem in files else None
        if path is None or not path.is_file():
            missing.append(stem)
            after += len(points)
            continue
        image = read_image(path)
        kept = [p for p in points if not in_black_patch(image, p[0], p[1], radius, black)]
        if len(kept) != len(points):
            per_image.append((stem, len(points) - len(kept), len(kept)))
        ann["points"] = kept
        after += len(kept)
    return DropReport(before, after, per_image, missing)


# -- duplicate images ------------------------------------------------------------------

_UPLOAD_HASH = re.compile(r"^[0-9a-f]{8}-")
_NUMBERED = re.compile(r"^\d{3}_(.+)$")  # "017_name.jpg", a renamed copy of "name.jpg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_usage(exports: list[list[dict]]) -> dict[str, tuple[int, int]]:
    """file name -> (tasks referencing it, annotation results on them), over LS exports."""
    usage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for tasks in exports:
        for task in tasks:
            name = task_file(task)
            if name:
                usage[name][0] += 1
                usage[name][1] += sum(len(a.get("result") or []) for a in task.get("annotations") or [])
    return {name: (t, r) for name, (t, r) in usage.items()}


def _keeper_rank(path: Path, usage: dict[str, tuple[int, int]]) -> tuple:
    """Prefer the copy labelling uses, then the cleaner name."""
    tasks, results = usage.get(path.name, (0, 0))
    return (
        results,
        tasks,
        not _UPLOAD_HASH.match(path.name),
        not _NUMBERED.match(path.name),
        -len(path.name),
        path.name.casefold(),
    )


def _signature(path: Path) -> tuple[np.ndarray, int]:
    """A 256x144 thumbnail and a 64-bit perceptual (DCT) hash."""
    thumb = cv2.resize(read_image(path), (256, 144), interpolation=cv2.INTER_AREA)
    small = cv2.resize(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))[:8, :8]
    bits = (dct > np.median(dct[1:])).reshape(-1)
    return thumb, sum(int(b) << i for i, b in enumerate(bits))


def duplicate_plan(images_dir: Path, usage: dict[str, tuple[int, int]] | None = None) -> list[dict]:
    """Which files duplicate another, and which copy to keep.

    Exact duplicates by SHA-256. Near duplicates only between `NNN_name` and
    `name` (a renamed re-encode), and only when nearly identical: perceptual
    hash distance <= 2 and mean thumbnail difference <= 0.5/255.
    """
    usage = usage or {}
    files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    digests = {p: sha256(p) for p in files}
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        groups[digests[p]].append(p)

    plan = []
    for members in (g for g in groups.values() if len(g) > 1):
        keeper = max(members, key=lambda p: _keeper_rank(p, usage))
        plan += [_row(p, keeper, usage, reason="exact-sha256", sha256=digests[p]) for p in members if p != keeper]

    planned = {row["duplicate"] for row in plan}
    by_name = {p.name: p for p in files}
    for numbered in files:
        match = _NUMBERED.match(numbered.name)
        plain = by_name.get(match.group(1)) if match else None
        if plain is None or digests[numbered] == digests[plain]:
            continue
        (a, hash_a), (b, hash_b) = _signature(numbered), _signature(plain)
        distance = (hash_a ^ hash_b).bit_count()
        mae = float(np.abs(a.astype(np.int16) - b).mean())
        if distance > 2 or mae > 0.5:
            continue
        keeper = max((numbered, plain), key=lambda p: _keeper_rank(p, usage))
        duplicate = plain if keeper is numbered else numbered
        if duplicate.name not in planned:
            plan.append(
                _row(
                    duplicate, keeper, usage, reason="near-reencoded", phash_distance=distance, thumb_mae=round(mae, 6)
                )
            )
    return sorted(plan, key=lambda row: row["duplicate"].casefold())


def _row(duplicate: Path, keeper: Path, usage: dict, **extra) -> dict:
    return {
        "duplicate": duplicate.name,
        "keeper": keeper.name,
        **extra,
        "duplicate_usage": usage.get(duplicate.name, (0, 0)),
        "keeper_usage": usage.get(keeper.name, (0, 0)),
    }


def quarantine(images_dir: Path, plan: list[dict], parent: Path) -> Path:
    """Move each planned duplicate into parent/<time>/ beside a manifest.json to undo it.

    Nothing is deleted. `parent` must lie inside `images_dir`.
    """
    images_dir, parent = images_dir.resolve(), parent.resolve()
    if parent == images_dir or images_dir not in parent.parents:
        raise ValueError("the quarantine folder must be inside the image folder")
    run = parent / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    manifest = {"source": str(images_dir), "created_at": datetime.now().astimezone().isoformat(), "moves": plan}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for row in plan:
        source, target = images_dir / row["duplicate"], run / row["duplicate"]
        if source.parent != images_dir or not source.is_file() or target.exists():
            raise ValueError(f"refusing to move {row['duplicate']}")
        shutil.move(source, target)
    return run


# -- baking region masks into images -----------------------------------------------------


@dataclass
class MaskReport:
    written: list[str]
    unmasked: list[str]  # no MAC in the name, or no mask for that MAC: not written


def apply_masks(images: list[Path], out_dir: Path, *, mask_dir: Path | None, mask: Path | None = None) -> MaskReport:
    """Write each image to `out_dir` with its ignored region painted black.

    The mask is `mask` for every image, or the camera's own from `mask_dir` by
    the MAC in the image name — the same masks and black/white rule as the
    live runtime (`core.masks`), resized to each image.
    """
    by_mac: dict[str, Path] = {}
    if mask is None and mask_dir is not None:
        by_mac, _ = index_by_mac(p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(p.parent.resolve() == out_dir.resolve() for p in images):
        raise ValueError("the output folder must not be an input folder: originals would be overwritten")

    keeps: dict[tuple[Path, int, int], np.ndarray] = {}
    report = MaskReport([], [])
    for path in images:
        source = mask or by_mac.get(extract_mac(path.name) or "")
        if source is None:
            report.unmasked.append(path.name)
            continue
        image = read_image(path)
        h, w = image.shape[:2]
        keep = keeps.get((source, w, h))
        if keep is None:
            keep = keeps[(source, w, h)] = read_keep_mask(source, (w, h))
        ok, buf = cv2.imencode(path.suffix, image * keep[..., None], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise OSError(f"cannot encode {path.name}")
        buf.tofile(out_dir / path.name)
        report.written.append(path.name)
    return report
