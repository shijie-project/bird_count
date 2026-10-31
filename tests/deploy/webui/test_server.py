"""The farm backend's HTTP API on a real socket."""

from birdcount.deploy.webui import server as srv

from .conftest import DAY, LOOSE, PLACED, T0, request


def test_farm_and_live_status(served) -> None:
    status, farm = request(served, "GET", "/api/farm")
    assert status == 200 and farm["days"] == [DAY] and farm["placed"] is False
    assert farm["status"]["state"] == "running" and farm["status"]["cameras"][0]["level"] == 1
    assert {c["mac"] for c in farm["cameras"]} == {PLACED, LOOSE}
    assert request(served, "GET", "/api/status")[1]["runtime"] == {"sms": "dry run"}


def test_a_day_and_one_camera(served) -> None:
    status, day = request(served, "GET", f"/api/day/{DAY}")
    assert status == 200 and day["model"] == "best.pth" and day["span"] == [T0, T0 + 119]
    placed = day["cameras"][PLACED]
    assert placed["periods"] == [[T0 + 70, T0 + 120, 1]]
    assert (placed["alarms"], placed["checked"], placed["unchecked"]) == (1, 0, [T0 + 70])
    assert day["cameras"][LOOSE] == {"seconds": 0, "threshold": 120.0, "alarms": 0, "checked": 0, "unchecked": []}
    assert [a["type"] for a in day["actions"]] == ["level1"]
    status, cam = request(served, "GET", f"/api/day/{DAY}/camera/{PLACED.lower()}")
    assert status == 200 and cam["threshold"] == 100.0 and cam["clear_level"] == 90.0
    assert [(s["kind"], s["level"]) for s in cam["snapshots"]] == [("bin", 0), ("high", 0)]
    assert cam["actions"][0]["image"].startswith("events/") and cam["actions"][0]["verdict"] is None
    part = request(served, "GET", f"/api/day/{DAY}/camera/{PLACED}?from={T0 + 10}&to={T0 + 20}")[1]
    assert (part["curve"]["t"][0], part["curve"]["t"][-1], part["curve"]["step"]) == (
        T0 + 10,
        T0 + 20,
        1,
    )  # a zoomed stretch, second by second
    assert [s["t"] for s in part["snapshots"]] == [T0 + 10]  # the stretch's snapshots only
    assert request(served, "GET", f"/api/day/{DAY}/camera/{PLACED}?from=soon")[0] == 400
    assert request(served, "GET", f"/api/day/2026-01-01/camera/{PLACED}")[0] == 404
    assert request(served, "GET", f"/api/day/{DAY}/camera/000000000000")[0] == 404


def test_checking_alarms_round_trip(served) -> None:
    status, todo = request(served, "GET", f"/api/alarms?day={DAY}")
    assert status == 200 and (todo["total"], todo["checked"], len(todo["items"])) == (1, 0, 1)
    (event,) = todo["items"]
    assert (event["event_id"], event["mac"], event["level"], event["end"]) == ("axis1_x", PLACED, 1, None)
    assert event["pictures"][0]["type"] == "level1" and event["key"] == f"alarm:{event['pictures'][0]['path']}"
    origin = {"Origin": f"http://127.0.0.1:{served.port}"}
    body = {"day": DAY, "path": event["pictures"][0]["path"], "source": "alarm", "label": "normal"}
    assert request(served, "POST", "/api/labels", body, origin)[0] == 200
    assert request(served, "GET", f"/api/alarms?day={DAY}")[1]["items"] == []  # checked: off the to-do list
    (done,) = request(served, "GET", f"/api/alarms?day={DAY}&state=done")[1]["items"]
    assert done["label"] == "normal"
    assert request(served, "GET", f"/api/day/{DAY}")[1]["cameras"][PLACED]["unchecked"] == []
    cam = request(served, "GET", f"/api/day/{DAY}/camera/{PLACED}")[1]
    assert cam["actions"][0]["verdict"] == "normal"
    checks = request(served, "GET", "/api/calibration")[1]["checks"]
    assert checks[PLACED] == {"alarms": 1, "checked": 1, "real": 0, "false": 1}
    assert request(served, "GET", f"/api/alarms?day={DAY}&camera={LOOSE}")[1]["total"] == 0
    assert (
        request(served, "POST", "/api/labels", {"day": DAY, "path": "../alarm.json", "label": "pileup"}, origin)[0]
        == 400
    )
    assert request(served, "POST", "/api/labels", {**body, "label": "maybe"}, origin)[0] == 400


def test_only_snapshots_and_alarm_pictures_are_served(served) -> None:
    shot = f"snapshots/{PLACED}/070010_bin_40.jpg"
    status, body = request(served, "GET", f"/api/image?day={DAY}&path={shot}")
    assert status == 200 and body.startswith(b"\xff\xd8")
    assert request(served, "GET", f"/api/image?day={DAY}&path=meta.json")[0] == 403
    assert request(served, "GET", f"/api/image?day={DAY}&path=../../alarm.json")[0] == 403
    assert request(served, "GET", "/api/image?alarm=../alarm.json")[0] == 403
    alarm = f"events/axis1_{PLACED}/20261024T200110Z_abcd1234/level1_{T0 + 70}.jpg"
    assert request(served, "GET", f"/api/image?alarm={alarm}")[0] == 200


def test_requests_from_elsewhere_are_refused(served) -> None:
    assert request(served, "GET", "/api/farm", headers={"Host": "evil.example:80"})[0] == 403
    body = {"day": DAY, "path": f"snapshots/{PLACED}/070010_bin_40.jpg", "label": "pileup"}
    assert request(served, "POST", "/api/labels", body, {"Origin": "http://evil.example"})[0] == 403
    assert request(served, "POST", "/api/labels", body, {"Sec-Fetch-Site": "cross-site"})[0] == 403


def test_the_page_and_its_modules(served) -> None:
    status, page = request(served, "GET", "/")
    assert status == 200 and b"Pile-up watch" in page
    for name in (
        "app.css",
        "js/app.js",
        "js/farm.js",
        "js/camera.js",
        "js/cameras.js",
        "js/check.js",
        "js/chart.js",
        "js/util.js",
        "js/thresholds.js",
        "js/ratings.js",
        "js/gallery.js",
        "js/datepick.js",
        "js/masks.js",
    ):
        assert request(served, "GET", f"/static/{name}")[0] == 200, name
    assert request(served, "GET", "/static/../server.py")[0] == 404
    assert srv.STATIC_DIR.joinpath("index.html").is_file()


def test_proposals_and_model_ratings_reach_the_page(served) -> None:
    import json as _json

    from birdcount.deploy.calibrate import ratings
    from birdcount.deploy.calibrate.ratings import Candidate

    assert request(served, "GET", "/api/calibration")[1]["proposal"] is None
    served.calibration_dir = served.tmp / "calibration"
    served.calibration_dir.mkdir()
    (served.calibration_dir / "latest.json").write_text(_json.dumps({"proposals": [{"mac": PLACED, "suggested": 95}]}))
    served.calibration_model = "m"
    _, data = request(served, "GET", "/api/calibration")
    assert data["proposal"]["proposals"][0]["suggested"] == 95 and data["thresholds"][PLACED] == 100
    day_dir = served.logs.day_dir(DAY)
    shot = f"snapshots/{PLACED}/070100_high_110.jpg"
    ratings.append(
        day_dir, [ratings.row(Candidate(DAY, shot, day_dir / shot, PLACED, 110, "high", T0 + 60), "m", 4, "packed")]
    )
    snaps = request(served, "GET", f"/api/day/{DAY}/camera/{PLACED}")[1]["snapshots"]
    assert [s["rating"] for s in snaps] == [None, {"level": 4, "reason": "packed"}]
    data = request(served, "GET", f"/api/ratings?day={DAY}")[1]
    cam = data["cameras"][PLACED]
    assert cam["pictures"] == 3 and cam["threshold"] == 100  # two snapshots and one alarm picture
    assert cam["items"] == [
        {
            "path": shot,
            "source": "count_log",
            "t": T0 + 60,
            "count": 110,
            "kind": "high",
            "level": 4,
            "reason": "packed",
        }
    ]
    assert data["levels"]["5"]["name"] == "Crammed" and data["model"] == "m"
    assert data["cameras"][LOOSE]["items"] == []
    every = request(served, "GET", "/api/ratings?day=all")[1]
    assert every["days"] == [DAY] and every["cameras"][PLACED]["items"] == cam["items"]


def test_masks_are_served_for_the_labelling_page(served, tmp_path) -> None:
    from birdcount.deploy.calibrate.rubric import SHADE
    from birdcount.deploy.webui.server import Backend

    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / f"{PLACED}_mask.png").write_bytes(b"\x89PNG mask")
    b = Backend(
        served.config, served.topology, status_path=None, count_log_dir=None, evidence_dir=tmp_path, mask_dir=masks
    )
    assert b.masks == {PLACED: masks / f"{PLACED}_mask.png"}
    served.masks = b.masks
    status, farm = request(served, "GET", "/api/farm")
    assert farm["masks"] == [PLACED] and farm["mask_shade"] == SHADE
    assert farm["levels"]["4"]["name"] == "Full"
    status, body = request(served, "GET", f"/api/mask/{PLACED.lower()}")
    assert status == 200 and body.startswith(b"\x89PNG")
    assert request(served, "GET", f"/api/mask/{LOOSE}")[0] == 404


def test_a_mask_is_edited_as_polygons_and_saved_with_its_history(served, tmp_path) -> None:
    import cv2
    import numpy as np

    masks = tmp_path / "masks"
    masks.mkdir()
    old = np.zeros((72, 128, 3), np.uint8)
    old[10:60, 20:100] = 255
    cv2.imwrite(str(masks / f"{PLACED}_mask.png"), old)
    served.mask_dir = masks
    served.index_masks()
    origin = {"Origin": f"http://127.0.0.1:{served.port}"}

    status, info = request(served, "GET", f"/api/masks/{PLACED}")
    assert status == 200 and info["size"] == [128, 72] and info["exists"] and info["history"] == []
    (poly,) = info["polygons"]
    assert sorted(map(tuple, poly)) == [(20, 10), (20, 59), (99, 10), (99, 59)]  # the white rectangle's corners
    assert info["picture"]["path"].startswith(f"snapshots/{PLACED}/")

    triangle = [[0, 0], [127, 0], [0, 71]]
    status, saved = request(
        served, "POST", f"/api/masks/{PLACED}", {"polygons": [triangle], "picture": info["picture"]}, origin
    )
    assert status == 200 and saved["result"]["backup"] and len(saved["history"]) == 1
    reference = masks / "_reference" / f"{PLACED}_mask.jpg"  # the picture it was drawn on, for alignment checks
    shown = tmp_path / "count_log" / info["picture"]["day"] / info["picture"]["path"]
    assert saved["result"]["reference"] == reference.name and reference.read_bytes() == shown.read_bytes()
    new = cv2.imread(str(masks / f"{PLACED}_mask.png"))
    assert new[2, 2].tolist() == [255, 255, 255] and new[70, 126].tolist() == [0, 0, 0]
    assert 0.45 < saved["keep"] < 0.55

    bad = request(served, "POST", f"/api/masks/{PLACED}", {"polygons": [[[0, 0], [1, 1]]]}, origin)
    assert bad[0] == 400  # two points are no polygon; nothing written
    assert len(request(served, "GET", f"/api/masks/{PLACED}")[1]["history"]) == 1

    name = saved["history"][0]["name"]
    status, back = request(served, "POST", f"/api/masks/{PLACED}/restore", {"name": name}, origin)
    assert status == 200 and (cv2.imread(str(masks / f"{PLACED}_mask.png")) == old).all()
    assert len(back["history"]) == 2  # the triangle is kept too
    assert request(served, "POST", f"/api/masks/{PLACED}/restore", {"name": "../x.png"}, origin)[0] == 400
