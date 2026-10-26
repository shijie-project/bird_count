import pytest

from birdcount.training.labeling import project
from birdcount.training.labeling.ls_format import keypoint, keypoint_control, local_files_url

from .conftest import CONFIG


def kp(n: int, label: str = "chicken") -> list[dict]:
    return [keypoint(10 * i, 10, (100, 100), label=label, from_name="kp", to_name="image") for i in range(n)]


def test_choose_keeper() -> None:
    anns = [
        {"id": 1, "result": kp(3), "updated_at": "2026-01-01"},
        {"id": 2, "result": kp(5), "updated_at": "2026-01-02", "was_cancelled": True},
        {"id": 3, "result": [], "updated_at": "2026-01-03"},
        {"id": 4, "result": kp(3), "updated_at": "2026-01-02"},
    ]
    assert project.choose_keeper(anns, "most-points")["id"] == 4  # a cancelled one never wins; ties go newest
    assert project.choose_keeper(anns, "newest")["id"] == 3
    assert project.choose_keeper(anns, "oldest")["id"] == 1
    assert project.choose_keeper(anns[1:2], "oldest")["id"] == 2  # only cancelled ones: keep one anyway


def test_dedupe_annotations_reports_then_deletes(fake, ls) -> None:
    fake.add_task("/x/a.jpg", [{"result": kp(2)}, {"result": kp(4)}, {"result": []}])
    fake.add_task("/x/b.jpg", [{"result": kp(1)}])
    fake.add_task("/x/c.jpg")
    log = []
    counts = project.dedupe_annotations(ls, 1, keep="most-points", apply=False, log=log.append)
    assert (counts["duplicated"], counts["extra"], counts["deleted"]) == (1, 2, 0)
    assert len(fake.annotations) == 4 and "keep" in log[0]
    counts = project.dedupe_annotations(ls, 1, keep="most-points", apply=True, log=log.append)
    assert counts["deleted"] == 2
    assert [len(a["result"]) for a in fake.tasks[min(fake.tasks)]["annotations"]] == [4]


def test_import_matches_local_images_and_adds_to_existing_tasks(fake, ls, tmp_path) -> None:
    for name in ("001_axis1_B8A44FD51C3C_frame1.jpg", "axis2_B8A44FD52B7F_frame9.jpg", "other.jpg"):
        (tmp_path / name).write_bytes(b"")
    existing = fake.add_task(local_files_url("001_axis1_B8A44FD51C3C_frame1.jpg", "images\\all\\"))
    source = [
        {
            "id": 77,
            "data": {"image": "/data/upload/9/deadbeef-001_axis1_B8A44FD51C3C_frame1.jpg"},
            "annotations": [{"id": 5, "result": kp(2, "hen"), "lead_time": 3.5}],
        },
        {
            "data": {"img": "/data/upload/9/axis2-B8A44FD52B7F-frame9.jpg"},
            "annotations": [{"result": kp(1)}],
        },  # renamed: fuzzy
        {"data": {"img": "/data/upload/9/nothing_like_it.jpg"}, "annotations": []},
    ]
    tasks, rejected = project.prepare_import(
        source, project.ImageIndex(tmp_path), "images\\all\\", label="chicken", log=lambda _: None
    )
    assert [r[0] for r in rejected] == [3]
    assert tasks[0]["data"] == {"img": existing["data"]["img"]}
    assert tasks[0]["annotations"][0] == {
        "result": kp(2, "chicken"),
        "was_cancelled": False,
        "ground_truth": False,
        "lead_time": 3.5,
    }  # foreign ids dropped, relabelled

    counts = project.import_annotations(ls, 1, tasks, label="chicken", log=lambda _: None)
    assert counts == {"matched": 1, "annotations_added": 1, "tasks_created": 1, "skipped": 0, "duplicate_tasks": 0}
    added = existing["annotations"][0]["result"]
    assert {(r["from_name"], r["to_name"]) for r in added} == {("dot-1", "img-1")}  # moved onto the project's control
    assert len(fake.tasks) == 2


def test_prelabels_are_created_once_then_updated(fake, ls) -> None:
    fake.add_task(local_files_url("a_B8A44FD51C3C.jpg", "images\\all\\"))
    blobs = {
        "params": {"frame_size": [1280, 720]},
        "images": [
            {
                "name": "a_B8A44FD51C3C",
                "size": [1920, 1080],
                "regions": [
                    {"id": 1, "count": 1.2, "centroid": [640.0, 360.0]},
                    {"id": 2, "count": 0.2, "centroid": [10.0, 10.0]},
                ],
            },
            {"name": "missing", "size": [1920, 1080], "regions": []},
        ],
    }
    counts = project.push_prelabels(ls, 1, blobs, check_only=True, log=lambda _: None)
    assert (counts["matched"], counts["points"], counts["created"], counts["missing"]) == (1, 1, 0, 1)
    assert "chicken-pred" not in fake.projects[1]["label_config"]

    project.push_prelabels(ls, 1, blobs, log=lambda _: None)
    assert "chicken-pred" in keypoint_control(fake.projects[1]["label_config"], "chicken-pred")[2]
    counts = project.push_prelabels(ls, 1, blobs, min_count=1.0, log=lambda _: None)
    assert (counts["created"], counts["updated"]) == (0, 1)
    (pred,) = fake.predictions.values()
    (point,) = pred["result"]
    assert point["value"]["x"] == pytest.approx(50.0) and point["original_width"] == 1920
    assert point["value"]["keypointlabels"] == ["chicken-pred"] and point["origin"] == "prediction"


def test_image_index_refuses_ambiguity(tmp_path) -> None:
    for name in ("1a2b3c4d-a.jpg", "5e6f7a8b-a.jpg", "abc_123.jpg", "abd_123.jpg"):
        (tmp_path / name).write_bytes(b"")
    index = project.ImageIndex(tmp_path)
    assert index.match("a.jpg")[1] == "ambiguous hash-free name match"
    assert index.match("5E6F7A8B-a.jpg")[0].name == "5e6f7a8b-a.jpg"
    path, note = index.match("abx_123.jpg")
    assert path is None and "ambiguous" in note
    assert index.match("abc_123.jpg", threshold=0)[0].name == "abc_123.jpg"
    with pytest.raises(FileNotFoundError):
        project.ImageIndex(tmp_path / "nope")


def test_config_fixture_is_valid() -> None:
    assert keypoint_control(CONFIG, "chicken")[1].attrib["toName"] == "img-1"
