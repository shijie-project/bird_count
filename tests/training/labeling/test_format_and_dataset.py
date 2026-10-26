import json

import numpy as np
import pytest

from birdcount.training.labeling import dataset
from birdcount.training.labeling.ls_format import (
    file_name,
    image_key,
    keypoint,
    keypoint_control,
    local_files_url,
    pick_annotation,
    task_file,
    task_points,
)

from .conftest import CONFIG


PREFIX = "images\\all\\"


def task(name: str, points=(), size=(1000, 500), *, upload: bool = False, cancelled_first: bool = False) -> dict:
    img = f"/data/upload/3/1a2b3c4d-{name}" if upload else local_files_url(name, PREFIX)
    result = [keypoint(x, y, size, label="chicken") for x, y in points]
    annotations = [{"result": result}]
    if cancelled_first:
        annotations.insert(0, {"result": [keypoint(1, 1, size, label="chicken")], "was_cancelled": True})
    return {"data": {"img": img}, "annotations": annotations if points or cancelled_first else []}


def test_image_references() -> None:
    assert file_name("/data/local-files/?d=images%5Call%5C001.jpg") == "001.jpg"
    assert file_name("/data/local-files/?d=images%2Fall%2Fa%20b.png") == "a b.png"
    assert file_name("/data/upload/2/61701da2-020_axis4.jpg") == "61701da2-020_axis4.jpg"
    assert image_key("61701da2-020_AXIS4.jpg") == image_key("020_axis4.jpg") == "020_axis4.jpg"
    assert local_files_url("x.jpg", "images\\all") == "/data/local-files/?d=images%5Call%5Cx.jpg"
    assert local_files_url("x.jpg", "images/all") == "/data/local-files/?d=images%2Fall%2Fx.jpg"
    # A task imported against a mismatched config keeps its image under a placeholder key.
    assert task_file({"data": {"$undefined$": "/data/local-files/?d=a%5Cb.jpg"}}) == "b.jpg"
    assert task_file({"data": {"text": "hello"}}) == ""


def test_points_come_from_the_first_real_annotation() -> None:
    t = task("a.jpg", [(100, 50), (250, 400)], cancelled_first=True)
    assert len(pick_annotation(t)) == 2
    points, size = task_points(t)
    assert np.allclose(points, [[100, 50], [250, 400]]) and size == (1000, 500)


def test_keypoint_control() -> None:
    _, control, labels = keypoint_control(CONFIG, "chicken")
    assert control.attrib["name"] == "dot-1" and labels == ["chicken"]
    two = CONFIG.replace(
        "</View>",
        '<KeyPointLabels name="k2" toName="img-1"><Label value="a"/><Label value="b"/></KeyPointLabels></View>',
    )
    assert keypoint_control(two, "chicken")[1].attrib["name"] == "dot-1"
    three = two.replace('value="chicken"', 'value="hen"').replace('<Label value="b"/>', "")
    with pytest.raises(ValueError, match="exactly one"):
        keypoint_control(three, "chicken")
    with pytest.raises(ValueError, match="invalid XML"):
        keypoint_control("<View>", "chicken")


def test_export_to_dataset_and_back() -> None:
    tasks = [task("a.jpg", [(100, 50)]), task("b.png"), task("c.jpg", [(1, 2), (3, 4)], size=(640, 480))]
    data = dataset.export_to_dataset(tasks)
    assert data["images"] == [
        {"id": 1, "file_name": "a.jpg", "width": 1000, "height": 500},
        {"id": 2, "file_name": "c.jpg", "width": 640, "height": 480},
    ]
    assert [a["image_id"] for a in data["annotations"]] == ["a", "c"]
    data["annotations"].append({"image_id": "ghost", "points": [[1, 1]]})
    back, orphans = dataset.dataset_to_tasks(data, PREFIX)
    assert orphans == ["ghost"]
    for a, b in zip(dataset.export_to_dataset(back)["annotations"], data["annotations"][:2], strict=True):
        assert a["image_id"] == b["image_id"] and np.allclose(a["points"], b["points"])
    assert set(back[0]) == {"data", "annotations"}  # LS assigns ids and times on import


def test_merge_later_wins_but_never_with_an_empty_copy() -> None:
    a = [task("x.jpg", [(1, 1)]), task("y.jpg", [(1, 1)])]
    b = [task("x.jpg", [(2, 2), (3, 3)]), task("y.jpg", upload=True), task("z.jpg")]
    merged, report = dataset.merge_exports([("a", a), ("b", b)])
    assert [task_file(t) for t in merged] == ["x.jpg", "y.jpg", "z.jpg"]
    assert len(task_points(merged[0])[0]) == 2  # b's x wins
    assert len(task_points(merged[1])[0]) == 1  # a's labelled y beats b's empty upload
    assert (report.duplicates, report.kept_labelled) == (2, 1)
    assert report.inputs == [("a", 2, 2), ("b", 3, 2)]
    everything, _ = dataset.merge_exports([("a", a), ("b", b)], keep_all=True)
    assert len(everything) == 5


def test_the_folder_roster_adds_unlabelled_images_once(tmp_path) -> None:
    for name in ("a.jpg", "b.jpg", "1a2b3c4d-c.jpg", "c.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    tasks = [task("a.jpg", [(1, 1)]), task("c.jpg", upload=True), task("gone.jpg")]
    roster = dataset.add_missing_images(tasks, dataset.list_images(tmp_path), PREFIX)
    # c.jpg's upload covers the exact file 1a2b3c4d-c.jpg; plain c.jpg is its twin.
    assert (roster.images, roster.covered, roster.added, roster.twins, roster.unmatched) == (4, 2, ["b.jpg"], 1, 1)
    assert tasks[-1] == {"data": {"img": local_files_url("b.jpg", PREFIX)}, "annotations": []}


def test_split_keeps_sessions_whole_and_every_camera_in_val() -> None:
    images, i = [], 0
    for mac in ("B8A44FD51C3C", "B8A44FD52B7F"):
        for session in range(5):
            for frame in range(4):
                i += 1
                images.append({"id": i, "file_name": f"axis1_{mac}_2026031{session}_101010_AB12_frame{frame}.jpg"})
    images += [{"id": 100 + k, "file_name": f"20260311_1230{k}_X.jpg"} for k in range(3)]  # MAC-less batch
    train, val, per_camera = dataset.split_images(images, 0.2, seed=3)
    assert len(train) + len(val) == len(images)
    assert per_camera["B8A44FD52B7F"] == (16, 4)

    def sessions(side):
        return {(dataset.camera_of(im["file_name"]), dataset.session_of(im["file_name"])) for im in side}

    assert not sessions(train) & sessions(val)
    assert dataset.camera_of("20260311_123456_X.jpg") == "B8A44FD51C3C"
    assert dataset.split_images(images, 0.2, seed=3)[1] == val  # deterministic
    with pytest.raises(ValueError):
        dataset.split_images(images, 1.5, 0)


def test_split_dataset_and_place_images(tmp_path) -> None:
    src, dst = tmp_path / "all", tmp_path / "val"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"a")
    dst.mkdir()
    (dst / "stale.jpg").write_bytes(b"old")
    data = {
        "images": [{"id": 2, "file_name": "a.jpg"}, {"id": 1, "file_name": "b.jpg"}],
        "annotations": [{"image_id": "a", "points": []}],
    }
    part = dataset.split_dataset(data, data["images"])
    assert [im["id"] for im in part["images"]] == [1, 2] and part["annotations"] == data["annotations"]
    assert dataset.place_images(data["images"], src, dst) == ["b.jpg"]
    assert sorted(p.name for p in dst.iterdir()) == ["a.jpg"]
    json.dumps(part)
