"""Keep exactly ONE annotation per task in a Label Studio project, delete the rest.

Label Studio accumulates a second annotation whenever a task is submitted again
(a re-import, a second annotator, a stray "Submit"), and the exporters
downstream only ever read the first one — so the extras are invisible in the
UI's counts yet silently decide what gets exported.

Which annotation survives is chosen by --keep:

  most-points  (default) the one with the most regions, i.e. the most complete
               labelling. Chosen as the default because the alternative silently
               throws away work: a stray empty re-submit is usually the NEWEST.
  newest       highest updated_at (falls back to id).
  oldest       lowest updated_at (falls back to id).

A cancelled (skipped) annotation never wins over a real one, whatever --keep says.

DELETION IS PERMANENT and there is no undo in Label Studio, so this script is a
dry run unless you pass --apply: by default it only prints what it would do.

Configuration comes from the environment / .env (see also tools/starter.sh):

    LABEL_STUDIO_URL=http://localhost:8080
    LABEL_STUDIO_API_KEY=<your token from Account & Settings>
    LABEL_STUDIO_PROJECT_ID=7          # optional, --project-id overrides

Usage:
    python tools/ls_ops/dedupe_annotations.py --project-id 7            # dry run: report only
    python tools/ls_ops/dedupe_annotations.py --project-id 7 --apply    # actually delete
    python tools/ls_ops/dedupe_annotations.py --project-id 7 --keep newest --apply
"""

import argparse
from typing import Any

from _common import connect, get_field, target_parser, task_label


KEEP_STRATEGIES = ("most-points", "newest", "oldest")


def annotation_summary(annotation: Any) -> dict:
    """The few fields the keep-strategies and the report need."""
    result = get_field(annotation, "result", []) or []
    return {
        "id": get_field(annotation, "id"),
        "points": len(result),
        "cancelled": bool(get_field(annotation, "was_cancelled", False)),
        "updated_at": str(get_field(annotation, "updated_at", "") or ""),
        "created_at": str(get_field(annotation, "created_at", "") or ""),
        "by": get_field(annotation, "completed_by", None),
    }


def choose_keeper(annotations: list[dict], strategy: str) -> dict:
    """Pick the annotation to keep. A real annotation always beats a cancelled one."""

    def sort_key(a: dict):
        # First element: not-cancelled wins regardless of strategy.
        rank = (not a["cancelled"],)
        if strategy == "most-points":
            return rank + (a["points"], a["updated_at"], a["id"] or 0)
        if strategy == "newest":
            return rank + (a["updated_at"], a["id"] or 0)
        # oldest: invert by negating the id and comparing timestamps ascending
        return rank + (_invert(a["updated_at"]), -(a["id"] or 0))

    return max(annotations, key=sort_key)


def _invert(text: str) -> tuple:
    """Sort key that orders strings descending inside an otherwise ascending max()."""
    return tuple(-ord(c) for c in text)


def dedupe_annotations(client, project_id: int, keep: str, apply: bool) -> int:
    """Keep one annotation per task; delete the others. Returns the exit code."""
    project = client.projects.get(id=project_id)
    tasks = list(client.tasks.list(project=project_id, fields="all"))
    print(f"Project '{get_field(project, 'title')}' (id={project_id}): {len(tasks)} task(s)")
    print(f"Keep strategy: {keep}    Mode: {'APPLY (deleting)' if apply else 'dry run (nothing deleted)'}")
    print("-" * 78)

    n_multi = n_deleted = n_failed = n_extra = 0
    n_single = n_none = 0

    for task in tasks:
        annotations = [annotation_summary(a) for a in get_field(task, "annotations", []) or []]
        if not annotations:
            n_none += 1
            continue
        if len(annotations) == 1:
            n_single += 1
            continue

        n_multi += 1
        keeper = choose_keeper(annotations, keep)
        drops = [a for a in annotations if a["id"] != keeper["id"]]
        n_extra += len(drops)

        detail = ", ".join(
            "#{id}({points}p{flag})".format(flag=", cancelled" if a["cancelled"] else "", **a) for a in annotations
        )
        dropped = ", ".join("#{id}".format(**a) for a in drops)
        print(f"  {task_label(task)}")
        print(f"      {len(annotations)} annotations: {detail}")
        print(f"      keep #{keeper['id']} ({keeper['points']} points), drop {dropped}")

        for annotation in drops:
            if not apply:
                continue
            try:
                client.annotations.delete(id=int(annotation["id"]))
                n_deleted += 1
            except Exception as exc:  # keep going; one bad id must not abort the pass
                n_failed += 1
                print(f"      ! failed to delete #{annotation['id']}: {type(exc).__name__}: {exc}")

    print("-" * 78)
    print(f"  tasks with 1 annotation : {n_single}")
    print(f"  tasks with none         : {n_none}")
    print(f"  tasks with duplicates   : {n_multi}")
    if apply:
        print(f"  annotations deleted     : {n_deleted}")
        if n_failed:
            print(f"  deletions failed        : {n_failed}")
    else:
        print(f"  annotations that WOULD be deleted: {n_extra}")
        if n_extra:
            print("  Nothing was deleted. Re-run with --apply to commit.")
    return 1 if n_failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        parents=[target_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("dedupe")
    g.add_argument(
        "--keep",
        choices=KEEP_STRATEGIES,
        default=KEEP_STRATEGIES[0],
        help="which annotation survives when a task has several (default: most-points)",
    )
    g.add_argument(
        "--apply",
        action="store_true",
        help="actually delete. WITHOUT THIS NOTHING IS DELETED — the run only reports.",
    )
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    client = connect(args)
    raise SystemExit(dedupe_annotations(client, args.project_id, args.keep, args.apply))


if __name__ == "__main__":
    main()
