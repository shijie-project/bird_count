"""Label Studio's task format: the one place that knows how LS spells things.

A task as exported or imported:

    {"data": {"img": "/data/local-files/?d=images%5Call%5C001.jpg"},
     "annotations": [{"result": [<keypoint>, ...], "was_cancelled": false, ...}],
     "predictions": [...]}

with one keypoint per bird:

    {"type": "keypointlabels", "from_name": "dot-1", "to_name": "img-1",
     "original_width": 1920, "original_height": 1080,
     "value": {"x": <percent>, "y": <percent>, "width": <marker size>, "keypointlabels": ["chicken"]}}

Coordinates are percentages of the image (0-100), so they survive resizing.

Images are referenced as `/data/local-files/?d=<path under LS's
LOCAL_FILES_DOCUMENT_ROOT>` (URL-quoted; on Windows LS writes backslashes), or
`/data/upload/<project>/<8 hex>-<name>` for files uploaded through the UI. One
picture can appear under both spellings, so images are matched by `image_key`,
which folds the upload hash away.

Stdlib only.
"""

from __future__ import annotations

import json
import ntpath
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote


IMAGE_FIELDS = ("img", "image", "url")
# LS's own keypoint marker size, as a percentage of the image width; generated
# points then look like hand-placed ones.
KEYPOINT_WIDTH = 0.37650602409638556
_UPLOAD_HASH = re.compile(r"^[0-9a-f]{8}-")


def file_name(ref: str) -> str:
    """'/data/local-files/?d=images%5Call%5C001.jpg' -> '001.jpg' (also uploads and plain paths)."""
    if "?d=" in ref:
        ref = ref.split("?d=", 1)[1]
    return ntpath.basename(unquote(ref).replace("/", "\\"))


def image_key(name: str) -> str:
    """Identity of an image name across spellings: no LS upload hash, case-folded.

    Callers match exact names first: a file on disk may legitimately start with
    eight hex digits and a dash.
    """
    return _UPLOAD_HASH.sub("", name, count=1).casefold()


def normalize_prefix(prefix: str) -> str:
    """Ensure a trailing separator, in the style the prefix already uses."""
    if not prefix or prefix.endswith(("/", "\\")):
        return prefix
    return prefix + ("\\" if "\\" in prefix else "/")


def local_files_url(relative: str, prefix: str) -> str:
    """`data.img` for an image at `prefix + relative` under LS's document root."""
    return "/data/local-files/?d=" + quote(normalize_prefix(prefix) + relative, safe="")


def task_image(task: dict) -> str:
    """The task's image reference, as stored.

    Looks at the usual field names first, then at any string that looks like an
    LS image URL: tasks imported against a mismatched labelling config keep
    their image under a placeholder key such as `$undefined$`.
    """
    data = task.get("data")
    if not isinstance(data, dict):
        return ""
    for key in IMAGE_FIELDS:
        if data.get(key):
            return str(data[key])
    for value in data.values():
        if isinstance(value, str) and value.startswith(("/data/", "http://", "https://")):
            return value
    return ""


def task_file(task: dict) -> str:
    """The task's image file name, or '' when it has none."""
    ref = task_image(task)
    return file_name(ref) if ref else ""


def pick_annotation(task: dict) -> list:
    """The result list of the first annotation that has results and was not cancelled."""
    annotations = task.get("annotations") or []
    for ann in annotations:
        if not ann.get("was_cancelled") and ann.get("result"):
            return ann["result"]
    return annotations[0].get("result") or [] if annotations else []


def task_points(task: dict) -> tuple[list[list[float]], tuple[int | None, int | None]]:
    """Keypoints in pixels, and the (width, height) they refer to."""
    width = height = None
    points = []
    for item in pick_annotation(task):
        if item.get("type") and item["type"] != "keypointlabels":
            continue
        value = item.get("value") or {}
        w, h = item.get("original_width"), item.get("original_height")
        if "x" not in value or "y" not in value or w is None or h is None:
            continue
        width, height = w, h
        points.append([value["x"] / 100.0 * w, value["y"] / 100.0 * h])
    return points, (width, height)


def count_points(tasks: list[dict]) -> int:
    """Keypoints over `tasks`, counted exactly as the dataset conversion counts them."""
    return sum(len(task_points(t)[0]) for t in tasks)


def keypoint(
    x: float,
    y: float,
    size: tuple[int, int],
    *,
    label: str,
    from_name: str = "dot-1",
    to_name: str = "img-1",
    origin: str = "manual",
    width: float = KEYPOINT_WIDTH,
    **extra,
) -> dict:
    """One keypoint result at pixel (x, y) of an image of `size` = (width, height)."""
    w, h = size
    return {
        "original_width": int(w),
        "original_height": int(h),
        "image_rotation": 0,
        "value": {
            "x": min(100.0, max(0.0, x / w * 100.0)),
            "y": min(100.0, max(0.0, y / h * 100.0)),
            "width": width,
            "keypointlabels": [label],
        },
        "from_name": from_name,
        "to_name": to_name,
        "type": "keypointlabels",
        "origin": origin,
        **extra,
    }


def relabel(results: list, label: str, *, from_name: str | None = None, to_name: str | None = None) -> list:
    """Copies of `results` with every keypoint labelled `label` (and moved to a control)."""
    out = json.loads(json.dumps(results))
    for item in out:
        if not isinstance(item, dict) or item.get("type") != "keypointlabels":
            continue
        if not isinstance(item.get("value"), dict):
            item["value"] = {}
        item["value"]["keypointlabels"] = [label]
        if from_name:
            item["from_name"] = from_name
        if to_name:
            item["to_name"] = to_name
    return out


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def keypoint_control(label_config: str, label: str) -> tuple[ET.Element, ET.Element, list[str]]:
    """(config root, the KeyPointLabels control, its labels) of a project config.

    The control is the one holding `label`, else the only control with a single
    label, else the only control at all. Anything else is ambiguous and raises.
    """
    try:
        root = ET.fromstring(label_config)
    except ET.ParseError as e:
        raise ValueError(f"the project's labelling config is invalid XML: {e}") from e
    controls = [el for el in root.iter() if _tag(el) == "KeyPointLabels"]
    labels = {
        id(c): [ch.attrib["value"] for ch in c if _tag(ch) == "Label" and ch.attrib.get("value")] for c in controls
    }
    found = (
        [c for c in controls if label in labels[id(c)]]
        or [c for c in controls if len(labels[id(c)]) == 1]
        or (controls if len(controls) == 1 else [])
    )
    if len(found) != 1:
        raise ValueError(
            f"the project needs exactly one KeyPointLabels control for {label!r}; found {len(found) or 'none'}"
        )
    control = found[0]
    if not control.attrib.get("name") or not control.attrib.get("toName"):
        raise ValueError("the project's KeyPointLabels control has no name or toName")
    return root, control, labels[id(control)]


def add_label(control: ET.Element, value: str, *, background: str) -> None:
    """Add a label to a control of a config parsed by `keypoint_control`."""
    tag = "Label" if "}" not in control.tag else control.tag.rsplit("}", 1)[0] + "}Label"
    ET.SubElement(control, tag, value=value, background=background)


def config_xml(root: ET.Element) -> str:
    return ET.tostring(root, encoding="unicode")
