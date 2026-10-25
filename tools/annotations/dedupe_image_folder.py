"""Find duplicate images and optionally move extras into a quarantine folder.

Exact duplicates use SHA-256. Near duplicates are intentionally conservative:
only ``NNN_name.ext`` / ``name.ext`` pairs are considered, and they must have a
perceptual-hash distance <= 2 plus mean resized RGB error <= 0.5/255.

Label Studio exports can be supplied to choose the keeper: a referenced and
more-completely annotated spelling wins over a cleaner but unused file name.
Nothing is deleted; ``--apply`` moves extras and writes a recovery manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LS_HASH_RE = re.compile(r"^[0-9a-f]{8}-")
NUMBER_PREFIX_RE = re.compile(r"^\d{3}_(.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ls_basename(value: str) -> str:
    if "?d=" in value:
        value = value.split("?d=", 1)[1]
    return Path(unquote(value).replace("\\", "/")).name


def load_usage(paths: list[str]) -> dict[str, tuple[int, int]]:
    usage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for raw_path in paths:
        tasks = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise ValueError(f"{raw_path}: expected a Label Studio task list")
        for task in tasks:
            name = ls_basename(str(task.get("data", {}).get("img", "")))
            if not name:
                continue
            annotations = task.get("annotations") or []
            usage[name][0] += 1
            usage[name][1] += sum(len(annotation.get("result") or []) for annotation in annotations)
    return {name: (values[0], values[1]) for name, values in usage.items()}


def keeper_rank(path: Path, usage: dict[str, tuple[int, int]]) -> tuple:
    tasks, points = usage.get(path.name, (0, 0))
    return (
        points,
        tasks,
        not bool(LS_HASH_RE.match(path.name)),
        not bool(NUMBER_PREFIX_RE.match(path.name)),
        -len(path.name),
        path.name.casefold(),
    )


def exact_groups(files: list[Path]) -> list[list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[sha256(path)].append(path)
    return [members for members in groups.values() if len(members) > 1]


def image_signature(path: Path) -> tuple[np.ndarray, int]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((256, 144), Image.Resampling.LANCZOS), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    bits = (dct > np.median(dct[1:])).reshape(-1)
    phash = sum(int(value) << index for index, value in enumerate(bits))
    return rgb, phash


def near_pairs(files: list[Path], exact_members: set[Path]) -> list[tuple[Path, Path, int, float]]:
    by_name = {path.name: path for path in files}
    pairs = []
    for numbered in files:
        match = NUMBER_PREFIX_RE.match(numbered.name)
        plain = by_name.get(match.group(1)) if match else None
        if plain is None or (
            numbered in exact_members and plain in exact_members and sha256(numbered) == sha256(plain)
        ):
            continue
        a, hash_a = image_signature(numbered)
        b, hash_b = image_signature(plain)
        distance = (hash_a ^ hash_b).bit_count()
        mae = float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        if distance <= 2 and mae <= 0.5:
            pairs.append((numbered, plain, distance, mae))
    return pairs


def build_plan(images_dir: Path, ls_json: list[str]) -> list[dict]:
    files = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    usage = load_usage(ls_json)
    exact = exact_groups(files)
    exact_members = {path for group in exact for path in group}
    plan = []

    for group in exact:
        keeper = max(group, key=lambda path: keeper_rank(path, usage))
        for duplicate in group:
            if duplicate != keeper:
                plan.append(
                    {
                        "duplicate": duplicate.name,
                        "keeper": keeper.name,
                        "reason": "exact-sha256",
                        "sha256": sha256(duplicate),
                        "duplicate_usage": usage.get(duplicate.name, (0, 0)),
                        "keeper_usage": usage.get(keeper.name, (0, 0)),
                    }
                )

    already_planned = {row["duplicate"] for row in plan}
    for numbered, plain, distance, mae in near_pairs(files, exact_members):
        keeper = max((numbered, plain), key=lambda path: keeper_rank(path, usage))
        duplicate = plain if keeper == numbered else numbered
        if duplicate.name in already_planned:
            continue
        plan.append(
            {
                "duplicate": duplicate.name,
                "keeper": keeper.name,
                "reason": "near-reencoded",
                "phash_distance": distance,
                "resized_rgb_mae": round(mae, 6),
                "duplicate_usage": usage.get(duplicate.name, (0, 0)),
                "keeper_usage": usage.get(keeper.name, (0, 0)),
            }
        )
    return sorted(plan, key=lambda row: row["duplicate"].casefold())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", help="flat image directory to deduplicate")
    parser.add_argument("--ls-json", action="append", default=[], help="LS export used to choose referenced keepers")
    parser.add_argument("--quarantine", help="quarantine parent (default: IMAGES_DIR/_duplicates_quarantine)")
    parser.add_argument("--apply", action="store_true", help="move duplicates; otherwise only print the plan")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"image directory not found: {images_dir}")
    plan = build_plan(images_dir, args.ls_json)
    print(f"Duplicate files: {len(plan)}")
    for row in plan:
        print(f"  {row['duplicate']}  -> keep {row['keeper']}  ({row['reason']})")
    if not args.apply:
        print("Dry run; nothing moved. Pass --apply to quarantine these files.")
        return
    if not plan:
        return

    quarantine_parent = Path(args.quarantine).resolve() if args.quarantine else images_dir / "_duplicates_quarantine"
    if quarantine_parent == images_dir or images_dir not in quarantine_parent.parents:
        raise SystemExit("quarantine must be a subdirectory of the image directory")
    run_dir = quarantine_parent / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": str(images_dir),
        "created_at": datetime.now().astimezone().isoformat(),
        "moves": plan,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for row in plan:
        source = (images_dir / row["duplicate"]).resolve()
        target = (run_dir / row["duplicate"]).resolve()
        if source.parent != images_dir or target.parent != run_dir:
            raise SystemExit(f"unsafe move target in plan: {row['duplicate']}")
        if not source.is_file() or target.exists():
            raise SystemExit(f"move precondition failed: {source} -> {target}")
        shutil.move(str(source), str(target))

    remaining = sum(1 for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    print(f"[ok] moved {len(plan)} duplicate(s) to {run_dir}")
    print(f"[ok] {remaining} image(s) remain")
    print(f"[ok] recovery manifest: {manifest_path}")


if __name__ == "__main__":
    main()
