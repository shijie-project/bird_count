"""Labelling tools: Label Studio exports to the training split, and project upkeep.

    python -m birdcount.training.labeling pull -o export.json            # the project's tasks, via the API
    python -m birdcount.training.labeling merge a.json b.json -o all_ls.json [--images-dir ../data/images/all]
    python -m birdcount.training.labeling convert all_ls.json -o ../data/annotations/all.json
    python -m birdcount.training.labeling split                          # annotations/all.json -> train / val
    python -m birdcount.training.labeling to-ls -o tasks.json            # annotations/all.json -> LS tasks

    python -m birdcount.training.labeling drop-masked --report-only      # points inside blacked-out areas
    python -m birdcount.training.labeling dedupe-images                  # plan; --apply quarantines
    python -m birdcount.training.labeling apply-mask <images|folders> --out DIR

    python -m birdcount.training.labeling dedupe-annotations             # report; --apply deletes
    python -m birdcount.training.labeling import external.json --check-only
    python -m birdcount.training.labeling prelabel outputs/eval/.../blobs.json

Paths default to the dataset under DATA_ROOT; the server, token and project
come from LABEL_STUDIO_* in .env. Commands that delete or overwrite work in
Label Studio only report unless given --apply.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.training.labeling import dataset, project
from birdcount.training.labeling.ls_format import count_points


if TYPE_CHECKING:
    from birdcount.training.labeling.client import LabelStudio
    from birdcount.training.labeling.settings import LabelingSettings


class _LazySettings:
    """The settings, read on first use: the commands that never need them (convert,
    a plain merge) start without pydantic, cv2 or httpx, like v2's scripts did."""

    def __init__(self):
        self._settings = None

    def __getattr__(self, name: str):
        if self._settings is None:
            from birdcount.core.log import setup_logging
            from birdcount.training.labeling.settings import LabelingSettings

            self._settings = LabelingSettings.load()
            setup_logging(self._settings.log_level)
        return getattr(self._settings, name)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_items(f, obj, indent: int | None, depth: int = 0) -> None:
    """Exactly `json.dump(obj, f, indent=indent, ensure_ascii=False)`, but each item
    of the top two levels is encoded on its own by the C encoder.

    `json.dumps` builds the whole file as one string (a 10 MB export is then in
    memory twice more), and `json.dump` streams through the pure-Python encoder,
    ~5x slower. Item by item is both fast and small.
    """
    if depth < 2 and isinstance(obj, (list, dict)) and obj:
        items = obj.items() if isinstance(obj, dict) else ((None, v) for v in obj)
        pad = "" if indent is None else "\n" + " " * (indent * (depth + 1))
        f.write("{" if isinstance(obj, dict) else "[")
        for i, (key, value) in enumerate(items):
            if i:
                f.write("," if indent is not None else ", ")
            f.write(pad)
            if key is not None:
                f.write(json.dumps(key, ensure_ascii=False) + ": ")
            write_json_items(f, value, indent, depth + 1)
        f.write(("" if indent is None else "\n" + " " * (indent * depth)) + ("}" if isinstance(obj, dict) else "]"))
        return
    text = json.dumps(obj, indent=indent, ensure_ascii=False)
    if indent is not None and depth:  # strings never hold a raw newline: json escapes it
        text = text.replace("\n", "\n" + " " * (indent * depth))
    f.write(text)


def _write_json(path: Path, payload, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        write_json_items(f, payload, indent)
    tmp.replace(path)


def _tasks(path: Path) -> list[dict]:
    tasks = _read_json(path)
    if not isinstance(tasks, list):
        raise ValueError(f"{path}: expected a JSON list of Label Studio tasks")
    return tasks


def _connect(s: LabelingSettings, args) -> tuple[LabelStudio, int]:
    from birdcount.training.labeling.client import LabelStudio

    project_id = args.project_id or s.label_studio_project_id
    if not project_id:
        raise SystemExit("no project: pass --project-id or set LABEL_STUDIO_PROJECT_ID")
    return LabelStudio(s.label_studio_url, s.label_studio_api_key.get_secret_value()), project_id


# -- file pipeline --------------------------------------------------------------------------


def cmd_pull(s: LabelingSettings, args) -> int:
    ls, project_id = _connect(s, args)
    with ls:
        tasks = ls.export(project_id)
    out = args.output or s.data_root / "export" / f"project{project_id}-{datetime.now():%Y%m%d-%H%M%S}.json"
    _write_json(out, tasks)
    print(f"{len(tasks)} tasks, {count_points(tasks)} points -> {out}")
    return 0


def cmd_merge(s: LabelingSettings, args) -> int:
    if not args.inputs and not args.images_dir:
        raise SystemExit("nothing to merge: give export files, or --images-dir to start from a folder")
    tasks, report = dataset.merge_exports([(str(p), _tasks(p)) for p in args.inputs], keep_all=args.keep_all)
    for name, n, points in report.inputs:
        print(f"  {name}: {n} tasks, {points} points")
    if report.duplicates:
        print(f"  {report.duplicates} repeated image(s) collapsed, later inputs winning", end="")
        print(f" ({report.kept_labelled} kept a labelled copy over a later empty one)" if report.kept_labelled else "")
    if args.images_dir:
        roster = dataset.add_missing_images(
            tasks, dataset.list_images(args.images_dir), s.image_prefix(args.images_dir)
        )
        print(
            f"  {args.images_dir}: {roster.images} images, {roster.covered} already labelled, {len(roster.added)} added empty"
        )
        for name in roster.added:
            print(f"    + {name}")
        if roster.twins:
            print(f"    {roster.twins} skipped as a second copy of an image already in the output")
        if roster.unmatched:
            print(f"    {roster.unmatched} task(s) kept whose image is not in the folder")
        for name in roster.ambiguous:
            print(f"    ? {name} matches several files; left unmatched")
    _write_json(args.output, tasks)
    print(f"{len(tasks)} tasks, {count_points(tasks)} points -> {args.output}")
    return 0


def cmd_convert(s: LabelingSettings, args) -> int:
    tasks = _tasks(args.export)
    data = dataset.export_to_dataset(tasks)
    _write_json(args.output, data)
    points = sum(len(a["points"]) for a in data["annotations"])
    print(
        f"{len(data['images'])} annotated images, {points} points -> {args.output} "
        f"({len(tasks) - len(data['images'])} unannotated task(s) left out)"
    )
    return 0


def cmd_split(s: LabelingSettings, args) -> int:
    data = _read_json(s.annotations_dir / "all.json")
    train, val, per_camera = dataset.split_images(data.get("images", []), args.val_ratio, args.seed)
    missing = []
    for name, images in (("train", train), ("val", val)):
        part = dataset.split_dataset(data, images)
        _write_json(s.annotations_dir / f"{name}.json", part)
        missing += dataset.place_images(images, s.images_dir, s.data_root / "images" / name, move=args.move)
        print(f"{name}: {len(images)} images, {len(part['annotations'])} annotations")
    print(f"  {'camera':14} train  val")
    for camera, (n_train, n_val) in per_camera.items():
        print(f"  {camera:14} {n_train:5} {n_val:4}")
    if missing:
        print(f"WARNING: {len(missing)} listed image(s) not in {s.images_dir}: {', '.join(missing[:10])}")
    return 0


def _all_json(s: LabelingSettings, path: Path | None) -> Path:
    return path or s.annotations_dir / "all.json"


def cmd_to_ls(s: LabelingSettings, args) -> int:
    source = _read_json(_all_json(s, args.src))
    tasks, orphans = dataset.dataset_to_tasks(source, s.image_prefix(s.images_dir), label=args.label)
    _write_json(args.output, tasks, indent=None)
    print(f"{len(tasks)} tasks, {count_points(tasks)} points -> {args.output}")
    if orphans:
        print(f"  {len(orphans)} annotation(s) without an image entry left out: {', '.join(orphans[:10])}")
    return 0


# -- dataset hygiene ----------------------------------------------------------------------


def cmd_drop_masked(s: LabelingSettings, args) -> int:
    from birdcount.training.labeling import cleanup

    source = _all_json(s, args.input)
    data = _read_json(source)
    r = cleanup.drop_masked_points(data, s.images_dir, radius=args.radius, black=args.black)
    print(f"Points: {r.before} -> {r.after} ({r.before - r.after} inside blacked-out areas)")
    for stem, dropped, kept in sorted(r.per_image, key=lambda t: -t[1]):
        print(f"  {stem}: -{dropped} (kept {kept})")
    if r.missing:
        print(f"WARNING: {len(r.missing)} image(s) not found; their points were kept unchecked")
    if args.report_only:
        print("--report-only: nothing written")
        return 0
    out = args.output or source.with_name(f"{source.stem}_clean.json")
    _write_json(out, data)
    print(f"-> {out}")
    return 0


def cmd_dedupe_images(s: LabelingSettings, args) -> int:
    from birdcount.training.labeling import cleanup

    folder = args.folder or s.images_dir
    usage = cleanup.label_usage([_tasks(p) for p in args.ls_json])
    plan = cleanup.duplicate_plan(folder, usage)
    print(f"{len(plan)} duplicate file(s) in {folder}")
    for row in plan:
        print(f"  {row['duplicate']}  -> keep {row['keeper']}  ({row['reason']})")
    if not args.apply:
        print("Dry run: nothing moved. --apply moves them to a quarantine folder (with a manifest to undo).")
        return 0
    if plan:
        run = cleanup.quarantine(folder, plan, args.quarantine or folder / "_duplicates_quarantine")
        print(f"moved {len(plan)} file(s) to {run} (manifest.json lists every move)")
    return 0


def cmd_apply_mask(s: LabelingSettings, args) -> int:
    from birdcount.training.labeling import cleanup

    images = []
    for path in args.inputs:
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            images.append(path)
        else:
            raise SystemExit(f"no such image or folder: {path}")
    if args.mask is None and s.mask_dir is None:
        raise SystemExit("no masks: pass --mask or set MASK_DIR")
    r = cleanup.apply_masks(images, args.out, mask_dir=s.mask_dir, mask=args.mask)
    print(f"{len(r.written)} masked image(s) -> {args.out}")
    if r.unmasked:
        print(f"  {len(r.unmasked)} without a mask for their camera, not written: {', '.join(r.unmasked[:10])}")
    return 0


# -- live project ---------------------------------------------------------------------------


def cmd_dedupe_annotations(s: LabelingSettings, args) -> int:
    ls, project_id = _connect(s, args)
    with ls:
        c = project.dedupe_annotations(ls, project_id, keep=args.keep, apply=args.apply)
    print(
        f"{c['tasks']} tasks: {c['single']} with one annotation, {c['none']} with none, {c['duplicated']} with several"
    )
    if args.apply:
        print(f"deleted {c['deleted']} annotation(s)" + (f", {c['failed']} failed" if c["failed"] else ""))
        return 1 if c["failed"] else 0
    print(f"{c['extra']} annotation(s) WOULD be deleted (keep: {args.keep}); --apply to delete")
    return 0


def cmd_import(s: LabelingSettings, args) -> int:
    index = project.ImageIndex(args.images_dir or s.images_dir, recursive=args.recursive)
    source = project.load_export(args.src)
    tasks, rejected = project.prepare_import(
        source,
        index,
        s.image_prefix(index.root),
        label=args.label,
        threshold=args.fuzzy_threshold,
        margin=args.fuzzy_margin,
    )
    results = sum(len(a["result"]) for t in tasks for a in t["annotations"])
    print(f"matched {len(tasks)}/{len(source)} task(s), {results} result(s)")
    for i, name, reason in rejected:
        print(f"  #{i} {name}: {reason}")
    if rejected and not args.allow_missing:
        raise SystemExit(f"{len(rejected)} task(s) unmatched; nothing imported (--allow-missing skips them)")
    if not tasks or args.check_only:
        print("nothing imported" + (" (--check-only)" if args.check_only else ""))
        return 0
    ls, project_id = _connect(s, args)
    with ls:
        c = project.import_annotations(ls, project_id, tasks, label=args.label, existing_only=args.existing_only)
    print(
        f"{c['matched']} existing task(s) got {c['annotations_added']} annotation(s); "
        f"{c['tasks_created']} new task(s); {c['skipped']} skipped (--existing-only)"
    )
    return 0


def cmd_prelabel(s: LabelingSettings, args) -> int:
    blobs = _read_json(args.blobs)
    ls, project_id = _connect(s, args)
    with ls:
        c = project.push_prelabels(
            ls,
            project_id,
            blobs,
            label=args.label,
            manual_label=args.manual_label,
            min_count=args.min_count,
            check_only=args.check_only,
        )
    print(
        f"{c['matched']} task(s), {c['points']} {args.label} point(s): {c['created']} created, {c['updated']} updated; "
        f"{c['missing']} image(s) without a task" + (" (--check-only: nothing changed)" if args.check_only else "")
    )
    return 0


# -- parser -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m birdcount.training.labeling",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    fmt = argparse.ArgumentDefaultsHelpFormatter

    def command(name: str, func, help: str) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=help, description=help, formatter_class=fmt)
        c.set_defaults(func=func)
        return c

    def live(c: argparse.ArgumentParser) -> None:
        c.add_argument("--project-id", type=int, default=None, help="default LABEL_STUDIO_PROJECT_ID")

    c = command("pull", cmd_pull, "Download the project's tasks and annotations (LS JSON export)")
    live(c)
    c.add_argument("-o", "--output", type=Path, default=None, help="default DATA_ROOT/export/project<id>-<time>.json")

    c = command("merge", cmd_merge, "Merge LS exports, one task per image; later exports win")
    c.add_argument("inputs", nargs="*", type=Path, help="exports, the most trusted last")
    c.add_argument("-o", "--output", type=Path, required=True)
    c.add_argument("--keep-all", action="store_true", help="keep repeated images")
    c.add_argument(
        "--images-dir", type=Path, default=None, help="add an empty task for every image here that nothing labels"
    )

    c = command("convert", cmd_convert, "LS export -> dataset json (annotated images only)")
    c.add_argument("export", type=Path)
    c.add_argument("-o", "--output", type=Path, required=True, help="e.g. DATA_ROOT/annotations/all.json")

    c = command("split", cmd_split, "annotations/all.json -> train/val, per camera, whole recording sessions")
    c.add_argument("--val-ratio", type=float, default=0.2, help="per-camera share of images in val")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--move", action="store_true", help="move images from images/all instead of copying")

    c = command("to-ls", cmd_to_ls, "Dataset json -> LS tasks with the points as annotations")
    c.add_argument("--src", type=Path, default=None, help="default DATA_ROOT/annotations/all.json")
    c.add_argument("-o", "--output", type=Path, required=True)
    c.add_argument("--label", default="chicken")

    c = command("drop-masked", cmd_drop_masked, "Drop points lying in blacked-out (masked) image areas")
    c.add_argument("--input", type=Path, default=None, help="default DATA_ROOT/annotations/all.json")
    c.add_argument("--output", type=Path, default=None, help="default <input>_clean.json")
    c.add_argument("--radius", type=int, default=1, help="the (2r+1)^2 patch around a point must be black")
    c.add_argument("--black", type=int, default=10, help="highest channel value still black")
    c.add_argument("--report-only", action="store_true")

    c = command("dedupe-images", cmd_dedupe_images, "Find duplicate images; --apply quarantines them")
    c.add_argument("folder", nargs="?", type=Path, default=None, help="default DATA_ROOT/images/all")
    c.add_argument("--ls-json", type=Path, action="append", default=[], help="exports: the labelled copy is kept")
    c.add_argument("--quarantine", type=Path, default=None, help="default <folder>/_duplicates_quarantine")
    c.add_argument("--apply", action="store_true", help="move the duplicates (nothing is deleted)")

    c = command("apply-mask", cmd_apply_mask, "Write copies of images with the ignored region painted black")
    c.add_argument("inputs", nargs="+", type=Path, help="images or folders")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--mask", type=Path, default=None, help="one mask for all; default each camera's in MASK_DIR")

    c = command("dedupe-annotations", cmd_dedupe_annotations, "Keep one annotation per task; --apply deletes")
    live(c)
    c.add_argument("--keep", choices=project.KEEP_STRATEGIES, default="most-points")
    c.add_argument("--apply", action="store_true", help="delete; without it nothing changes")

    c = command("import", cmd_import, "Import another machine's export onto this machine's images")
    live(c)
    c.add_argument("src", type=Path)
    c.add_argument("--images-dir", type=Path, default=None, help="default DATA_ROOT/images/all")
    c.add_argument("--recursive", action="store_true")
    c.add_argument("--fuzzy-threshold", type=float, default=project.FUZZY_THRESHOLD, help="0 = exact names only")
    c.add_argument("--fuzzy-margin", type=float, default=project.FUZZY_MARGIN)
    c.add_argument("--label", default="chicken", help="every imported keypoint gets this label")
    c.add_argument("--allow-missing", action="store_true", help="skip unmatched tasks instead of refusing")
    c.add_argument("--existing-only", action="store_true", help="never create tasks")
    c.add_argument("--check-only", action="store_true")

    c = command("prelabel", cmd_prelabel, "Density blobs from `eval --blobs` as predictions on existing tasks")
    live(c)
    c.add_argument("blobs", type=Path, help="blobs.json written by python -m birdcount.training.eval --blobs")
    c.add_argument("--label", default=project.PREDICTION_LABEL)
    c.add_argument("--manual-label", default="chicken", help="the human label, to find the keypoint control")
    c.add_argument("--min-count", type=float, default=0.5, help="skip blobs integrating fewer birds")
    c.add_argument("--check-only", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _LazySettings()
    try:
        return args.func(settings, args)
    except (RuntimeError, OSError, ValueError) as e:  # RuntimeError: LabelStudioError
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
