"""Shared CLI and dataset plumbing for the three evaluation operations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader

from datasets import BirdDataset, collate
from datasets.transforms import DOWNSAMPLE_RATIO, build_val_transform


DEFAULT_DATA_PATH = "../data/annotated"


def add_data_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the identical dataset-selection controls used by every evaluator."""
    group = parser.add_argument_group("data")
    group.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH, help="dataset root containing images/ and annotations/"
    )
    group.add_argument("--split", default="val", choices=["val", "train", "all"])
    group.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="resize the longer edge to this many pixels before inference (0 = native resolution)",
    )
    group.add_argument("--num-workers", type=int, default=2)
    group.add_argument("--limit", type=int, default=0, help="only evaluate the first N images (0 = all)")
    group.add_argument(
        "--skip-unannotated",
        action="store_true",
        help="skip images that have no annotation (not in the JSON, or zero points) instead of scoring them as GT 0",
    )
    group.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="IMAGE_NAME",
        help="image file name(s) to exclude; extension optional. Skipped images are left out of overlays and metrics",
    )
    return group


def add_model_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add checkpoint/device flags with one spelling and one default everywhere."""
    group = parser.add_argument_group("model")
    group.add_argument("--ckpt", default=None, help="checkpoint path; defaults to MODEL_PATH from .env")
    group.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")
    group.add_argument(
        "--no-fuse",
        action="store_true",
        help="skip Conv+BN fusion (debug only; fusion is mathematically equivalent)",
    )
    return group


def add_output_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_help: str,
    overlay_help: str,
    legacy_overlay_flags: tuple[str, ...] = (),
    include_metrics_out: bool = False,
) -> argparse._ArgumentGroup:
    """Add the common output controls, retaining old flags as CLI aliases."""
    group = parser.add_argument_group("output")
    group.add_argument("--output-dir", default=None, help=output_help)
    group.add_argument(
        "--no-density-map",
        *legacy_overlay_flags,
        dest="no_density_map",
        action="store_true",
        help=overlay_help,
    )
    if include_metrics_out:
        group.add_argument(
            "--metrics-out",
            default=None,
            help="optional JSON file for the full metrics report; a relative path is resolved inside --output-dir",
        )
    group.add_argument("--seed", type=int, default=42)
    return group


def _without_skipped_images(
    image_paths: list[str], skip_names: list[str], image_dir: Path
) -> tuple[list[str], list[str]]:
    """Apply test.py-style extension-insensitive exclusions to a split."""
    if not skip_names:
        return image_paths, []

    def key(path: str | Path) -> str:
        return os.path.normcase(os.path.normpath(os.path.splitext(str(path))[0]))

    wanted = {name: key(image_dir / name) for name in skip_names}
    matched: set[str] = set()
    kept = []
    for path in image_paths:
        image_key = key(path)
        if image_key in wanted.values():
            matched.add(image_key)
        else:
            kept.append(path)
    return kept, [name for name, image_key in wanted.items() if image_key not in matched]


def build_dataset(args: argparse.Namespace) -> BirdDataset:
    """Build one consistently filtered evaluation dataset from the shared args."""
    if args.test_size < 0 or args.num_workers < 0 or args.limit < 0:
        raise ValueError("--test-size, --num-workers, and --limit must be >= 0")

    # An evaluation split always uses the deterministic validation transform.
    # Without this explicit transform, BirdDataset would augment/crop when the
    # selected folder is the training split.
    transform = build_val_transform(DOWNSAMPLE_RATIO, test_size=args.test_size)
    dataset = BirdDataset(
        args.data_path,
        split=args.split,
        transform=transform,
        test_size=args.test_size,
        skip_unannotated=args.skip_unannotated,
    )
    if args.skip:
        before = len(dataset.im_list)
        image_dir = Path(args.data_path) / "images" / args.split
        dataset.im_list, unmatched = _without_skipped_images(dataset.im_list, args.skip, image_dir)
        print(f"--skip: excluded {before - len(dataset.im_list)} image(s) from '{image_dir}'")
        if unmatched:
            print(f"  warning: {len(unmatched)} skip name(s) matched no image: {unmatched}")
    if args.limit:
        dataset.im_list = dataset.im_list[: args.limit]
    return dataset


def build_loader(args: argparse.Namespace, dataset: BirdDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
