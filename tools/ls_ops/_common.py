"""Shared plumbing for the Label Studio operations in this folder.

Every op is its own script — one file, one thing it does — and they all need the
same three answers first: which server, which project, whose token. Those flags
are defined once here as a parent parser so the ops stay identical to use and
the web UI renders the same "target" group for each of them.

These ops act on a *live* project through the API. The file-based pipeline that
turns exports into training JSON lives next door in tools/annotations/.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import dotenv


# ls_img_to_filename is the canonical "Label Studio img field -> file name"
# decoder and lives with the exporters; importing it beats keeping a second copy
# in step. Adding the folder to sys.path here (rather than relying on
# PYTHONPATH) keeps the ops runnable from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "annotations"))

from convert_ls_to_coco import ls_img_to_filename  # noqa: E402


dotenv.load_dotenv(dotenv_path=Path.cwd().joinpath(".env"))

DEFAULT_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
DEFAULT_PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "0") or 0)


def target_parser() -> argparse.ArgumentParser:
    """Parent parser holding the flags every op shares."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("target")
    g.add_argument(
        "--project-id",
        type=int,
        default=DEFAULT_PROJECT_ID or None,
        help="Label Studio project id (default: LABEL_STUDIO_PROJECT_ID from .env)",
    )
    g.add_argument("--url", default=DEFAULT_URL, help=f"Label Studio base URL (default: {DEFAULT_URL})")
    g.add_argument("--api-key", default=None, help="API token (default: LABEL_STUDIO_API_KEY from .env)")
    return p


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def connect(args: argparse.Namespace):
    """Client for the project named on the command line.

    The SDK is imported lazily so `--help` and the web UI's form generation work
    even where label-studio-sdk is not installed.
    """
    if not args.project_id:
        raise SystemExit("No project id. Pass --project-id or set LABEL_STUDIO_PROJECT_ID in .env.")

    api_key = args.api_key or os.getenv("LABEL_STUDIO_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "No API key. Set LABEL_STUDIO_API_KEY in .env (Label Studio > Account & Settings > Access Token)\n"
            "or pass --api-key."
        )
    try:
        from label_studio_sdk import LabelStudio
    except ImportError:
        raise SystemExit("label-studio-sdk is not installed. `pip install label-studio-sdk`")
    return LabelStudio(base_url=args.url, api_key=api_key)


def task_label(task: Any) -> str:
    """Image file name for a task, for the report.

    Uses the same unquote-then-basename helper as the export tools, so a
    "%5C"-encoded local-files URL prints as a plain file name.
    """
    data = get_field(task, "data", {}) or {}
    for key in ("img", "image", "url"):
        value = data.get(key) if isinstance(data, dict) else None
        if value:
            return ls_img_to_filename(str(value))
    return f"task#{get_field(task, 'id')}"
