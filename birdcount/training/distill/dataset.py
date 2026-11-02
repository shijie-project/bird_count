"""Training crops of distillation pairs, and the report on held-out pairs."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from birdcount.core.model import OUTPUT_STRIDE


def read_index(root: Path) -> list[dict[str, str]]:
    with open(Path(root) / "index.csv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class DistillCrops(Dataset):
    """Random `crop` x `crop` crops of the delivered frames, with the matching cells of the teacher's density.

    Crops are aligned to the output stride so the density cells match exactly.
    Of three candidate positions the one holding most birds is taken 70 % of
    the time, so the crowded floor (where compression costs counts) dominates.
    Flips and quarter turns as in the labelled crops; no colour change: the
    target is the teacher's reading of these very pixels.
    """

    def __init__(self, root: Path, crop: int):
        if crop % OUTPUT_STRIDE:
            raise ValueError(f"crop {crop} must be a multiple of {OUTPUT_STRIDE}")
        self.root = Path(root) / "pairs"
        self.names = [r["file"] for r in read_index(root)]
        self.crop = crop

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, item: tuple[int, int]) -> dict:
        index, seed = item
        rng = np.random.default_rng(seed)
        name = self.names[index]
        image = cv2.imread(str(self.root / f"{name}.png"), cv2.IMREAD_COLOR)
        density = np.load(self.root / f"{name}.npy").astype(np.float32)
        s, c = OUTPUT_STRIDE, self.crop
        h, w = image.shape[:2]
        if h < c or w < c:
            raise ValueError(f"{name}: {w}x{h} frame is smaller than the {c} px crop")
        spots = [
            (int(rng.integers(0, (h - c) // s + 1)) * s, int(rng.integers(0, (w - c) // s + 1)) * s) for _ in range(3)
        ]
        if rng.random() < 0.7:
            spots.sort(key=lambda yx: -density[yx[0] // s : (yx[0] + c) // s, yx[1] // s : (yx[1] + c) // s].sum())
        y, x = spots[0]
        im = image[y : y + c, x : x + c]
        d = density[y // s : (y + c) // s, x // s : (x + c) // s]
        if rng.random() < 0.5:
            im, d = im[:, ::-1], d[:, ::-1]
        k = int(rng.integers(0, 4))
        im, d = np.rot90(im, k), np.rot90(d, k)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(im)),  # (crop, crop, 3) uint8 BGR
            "density": torch.from_numpy(np.ascontiguousarray(d))[None],  # (1, crop/8, crop/8)
        }


class Draws(Sampler[tuple[int, int]]):
    """An endless stream of (index, seed), drawn with replacement; reproducible from `seed`."""

    def __init__(self, size: int, seed: int):
        if size <= 0:
            raise ValueError("no distillation pairs")
        self.size, self.seed = size, seed

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, 7])
        while True:
            yield int(rng.integers(self.size)), int(rng.integers(0, 2**63 - 1))


def _collate(batch: list[dict]) -> dict:
    return {"image": torch.stack([b["image"] for b in batch]), "density": torch.stack([b["density"] for b in batch])}


def make_distill_loader(dataset: DistillCrops, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=Draws(len(dataset), seed),
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# --------------------------------------------------------------------------------------
# Held-out report
# --------------------------------------------------------------------------------------


def holdout_report(rows: list[dict[str, str]], counts: np.ndarray) -> dict[str, dict]:
    """Per stream setting: how far the model's counts on the delivered frames are from the teacher's on the clean ones.

    `dense_bias_pct` (crowded frames, where alarms are decided) is the headline;
    `before` is the same for the teacher itself on the delivered frames.
    """
    groups: dict[str, list[tuple[float, float, float, bool]]] = {}
    for r, c in zip(rows, counts, strict=True):
        groups.setdefault(r["setting"], []).append(
            (float(r["teacher"]), float(c), float(r["teacher_on_delivered"]), r["dense"] == "1")
        )

    def bias(a: np.ndarray, col: int, sel: np.ndarray) -> float | None:
        ref = a[sel, 0].sum()
        return round(float(100 * (a[sel, col] - a[sel, 0]).sum() / ref), 2) if sel.any() and ref > 0 else None

    out = {}
    for key, g in sorted(groups.items()):
        a = np.array(g, dtype=np.float64)
        dense = a[:, 3] > 0
        out[key] = {
            "n": len(a),
            "dense_n": int(dense.sum()),
            "dense_bias_pct": bias(a, 1, dense),
            "dense_bias_pct_before": bias(a, 2, dense),
            "mae": round(float(np.abs(a[:, 1] - a[:, 0]).mean()), 3),
            "mae_before": round(float(np.abs(a[:, 2] - a[:, 0]).mean()), 3),
        }
    return out


def count_holdout(predictor_for, root: Path, rows: list[dict[str, str]], batch: int = 16) -> np.ndarray:
    """The model's count of every held-out delivered frame; `predictor_for(mac)` -> (predictor, mask ids or None)."""
    pairs = Path(root) / "pairs"
    counts = np.empty(len(rows), np.float32)
    for mac in sorted({r["mac"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["mac"] == mac]
        pred, uses_mask = predictor_for(mac)
        for s in range(0, len(idx), batch):
            chunk = idx[s : s + batch]
            frames = [cv2.imread(str(pairs / f"{rows[i]['file']}.png"), cv2.IMREAD_COLOR) for i in chunk]
            counts[chunk] = pred.predict(frames, [0] * len(frames) if uses_mask else None).counts
    return counts
