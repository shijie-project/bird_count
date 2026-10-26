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
    assert placed["periods"] == [[T0 + 70, T0 + 120, 1]] and placed["to_label"] == [T0 + 10, T0 + 60]
    assert day["cameras"][LOOSE] == {"seconds": 0, "threshold": 120.0, "labelled": 0, "to_label": []}
    assert [a["type"] for a in day["actions"]] == ["level1"]
    status, cam = request(served, "GET", f"/api/day/{DAY}/camera/{PLACED.lower()}")
    assert status == 200 and cam["threshold"] == 100.0 and cam["clear_level"] == 90.0
    assert [(s["kind"], s["level"]) for s in cam["snapshots"]] == [("bin", 0), ("high", 0)]
    assert cam["actions"][0]["image"].startswith("events/")
    assert request(served, "GET", f"/api/day/2026-01-01/camera/{PLACED}")[0] == 404
    assert request(served, "GET", f"/api/day/{DAY}/camera/000000000000")[0] == 404


def test_labelling_round_trip(served) -> None:
    status, queue = request(served, "GET", f"/api/snapshots?day={DAY}&kind=high,bin,alarm")
    assert status == 200 and (queue["total"], queue["labelled"], len(queue["items"])) == (3, 0, 3)
    alarm = next(i for i in queue["items"] if i.get("source") == "alarm")
    origin = {"Origin": f"http://127.0.0.1:{served.port}"}
    shot = f"snapshots/{PLACED}/070100_high_110.jpg"
    assert request(served, "POST", "/api/labels", {"day": DAY, "path": shot, "label": "pileup"}, origin)[0] == 200
    assert (
        request(
            served,
            "POST",
            "/api/labels",
            {"day": DAY, "path": alarm["path"], "source": "alarm", "label": "normal"},
            origin,
        )[0]
        == 200
    )
    status, queue = request(served, "GET", f"/api/snapshots?day={DAY}&kind=high,bin,alarm")
    assert (queue["labelled"], [i["path"] for i in queue["items"]]) == (2, [f"snapshots/{PLACED}/070010_bin_40.jpg"])
    done = request(served, "GET", f"/api/snapshots?day={DAY}&kind=high,alarm&state=done")[1]["items"]
    assert {i["label"] for i in done} == {"pileup", "normal"}
    assert (
        request(served, "POST", "/api/labels", {"day": DAY, "path": "../alarm.json", "label": "pileup"}, origin)[0]
        == 400
    )
    assert request(served, "POST", "/api/labels", {"day": DAY, "path": shot, "label": "maybe"}, origin)[0] == 400


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
        "js/label.js",
        "js/chart.js",
        "js/util.js",
    ):
        assert request(served, "GET", f"/static/{name}")[0] == 200, name
    assert request(served, "GET", "/static/../server.py")[0] == 404
    assert srv.STATIC_DIR.joinpath("index.html").is_file()
