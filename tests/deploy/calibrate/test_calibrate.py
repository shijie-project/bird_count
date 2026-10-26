"""Calibration: the rubric's replies, the rater against a fake API, picking pictures, and the fit."""

import base64
import json
from datetime import datetime

import cv2
import httpx
import numpy as np
import pytest

from birdcount.deploy import countlog
from birdcount.deploy.calibrate import fit, ratings
from birdcount.deploy.calibrate.images import MaskShader
from birdcount.deploy.calibrate.rater import ClaudeRater, Picture, RaterError
from birdcount.deploy.calibrate.rubric import RUBRIC_VERSION, parse_reply


MAC = "B8A44FD51C3C"


def test_replies_are_read_strictly() -> None:
    assert parse_reply('{"level": 4, "reason": "pressed together"}') == (4, "pressed together")
    assert parse_reply('Sure: {"level": 2, "reason": "gaps"} done') == (2, "gaps")
    for bad in ("level 3", '{"level": 7}', '{"reason": "x"}', ""):
        with pytest.raises((ValueError, KeyError)):
            parse_reply(bad)


def _api(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _message(text):
    return {"content": [{"type": "text", "text": text}]}


def test_direct_requests_send_the_image_and_rubric_and_never_fail_the_run() -> None:
    seen = []

    def handler(request):
        body = json.loads(request.content)
        if isinstance(body["messages"][0]["content"], str):  # the probe for how to switch thinking off
            msg = 'To turn thinking off on this model, send "thinking": {"type": "between_tools"} instead'
            return httpx.Response(400, json={"error": {"message": msg}})
        assert body["thinking"] == {"type": "between_tools"}
        seen.append((request.headers["x-api-key"], body["model"], body["messages"][0]["content"][0]["type"]))
        if base64.b64decode(body["messages"][0]["content"][0]["source"]["data"]).endswith(b"bad"):
            return httpx.Response(529, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=_message('{"level": 3, "reason": "dense"}'))

    rater = ClaudeRater("key-123", "claude-test", batch=False, client=_api(handler))
    out = rater.rate([Picture("a", b"\xff\xd8good"), Picture("b", b"\xff\xd8bad")])
    assert out["a"] == (3, "dense") and out["b"].startswith("HTTP 529: overloaded")
    assert all(s == ("key-123", "claude-test", "image") for s in seen)


def test_the_batch_api_is_polled_until_it_ends() -> None:
    state = {"polls": 0}

    def handler(request):
        if request.url.path.endswith("/messages"):  # the probe: thinking can be "disabled" here
            return httpx.Response(200, json=_message("OK"))
        if request.method == "POST":
            assert json.loads(request.content)["requests"][0]["params"]["thinking"] == {"type": "disabled"}
            ids = [r["custom_id"] for r in json.loads(request.content)["requests"]]
            state["ids"] = ids
            return httpx.Response(200, json={"id": "b1", "processing_status": "in_progress"})
        if request.url.path.endswith("/b1"):
            state["polls"] += 1
            done = state["polls"] >= 2
            return httpx.Response(200, json={"id": "b1", "processing_status": "ended" if done else "in_progress",
                                             "results_url": "https://api.test/results.jsonl"})  # fmt: skip
        lines = [{"custom_id": state["ids"][0], "result": {"type": "succeeded", "message": _message('{"level": 5}')}},
                 {"custom_id": state["ids"][1], "result": {"type": "errored", "error": {"type": "invalid"}}}]  # fmt: skip
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    rater = ClaudeRater("k", "m", client=_api(handler), poll_sec=0)
    out = rater.rate([Picture("p0", b"x"), Picture("p1", b"y")])
    assert out["p0"] == (5, "") and out["p1"].startswith("errored")
    assert state["polls"] == 2


def test_no_key_no_rater() -> None:
    with pytest.raises(RaterError, match="ANTHROPIC_API_KEY"):
        ClaudeRater("", "m")


def _day(tmp_path, counts_by_bin):
    day = "2026-10-25"
    folder = tmp_path / day / "snapshots" / MAC
    folder.mkdir(parents=True)
    t = datetime(2026, 10, 25, 7, 0, 0).timestamp()
    for count, n in counts_by_bin.items():
        for i in range(n):
            (folder / countlog.snapshot_path(MAC, t + 60 * i + count, "bin", count).split("/")[-1]).write_bytes(b"j")
    return day


def test_each_bin_is_rated_until_it_holds_enough(tmp_path) -> None:
    day = _day(tmp_path, {35: 15, 145: 3})
    cands = ratings.candidates(tmp_path, [day], None)
    assert len(cands) == 18 and {c.mac for c in cands} == {MAC}
    picked = ratings.select(cands, set(), per_bin=10)
    assert sum(c.count == 35 for c in picked) == 10 and sum(c.count == 145 for c in picked) == 3
    rated = {c.key for c in picked}
    assert ratings.select(cands, rated, per_bin=10) == []
    assert len(ratings.select(cands, set(), per_bin=10, limit=4)) == 4


def test_ratings_round_trip_and_only_the_current_standard_counts(tmp_path) -> None:
    day = _day(tmp_path, {35: 2})
    c = ratings.candidates(tmp_path, [day], None)[0]
    ratings.append(tmp_path / day, [ratings.row(c, "m", 2, "gaps")])
    ratings.append(tmp_path / day, [ratings.row(c, "m", 3, "denser on a second look")])
    ratings.append(tmp_path / day, [{**ratings.row(c, "m", 5, "old rubric"), "rubric": RUBRIC_VERSION - 1}])
    (row,) = ratings.current(ratings.read(tmp_path / day), "m")
    assert (row["level"], row["reason"]) == ("3", "denser on a second look")
    assert ratings.current(ratings.read(tmp_path / day), "other") == []


def test_isotonic_pools_violators() -> None:
    assert fit.isotonic([0.0, 0.5, 0.2, 1.0], [1, 1, 1, 1]) == pytest.approx([0.0, 0.35, 0.35, 1.0])
    assert fit.isotonic([0.3, 0.1], [3, 1]) == pytest.approx([0.25, 0.25])


def _points(dense_from):
    pts = []
    for count in range(20, 200, 10):
        pts += [(count + 3, 5 if count >= dense_from else 2)] * 4
    return pts


def test_the_threshold_sits_where_pictures_turn_dense() -> None:
    p = fit.propose(MAC, 170, _points(dense_from=130))
    assert p.suggested == 130 and p.step == 153.0 and p.error_suggested == 0.0
    assert p.error_now > 0.2 and p.dense == 28 and p.note == ""
    assert [b["lo"] for b in p.bins][:2] == [20, 30] and p.bins[-1]["fitted"] == 1.0


def test_no_suggestion_without_both_sides() -> None:
    assert fit.propose(MAC, 100, []).note == "no ratings yet"
    assert "too few crammed" in fit.propose(MAC, 100, [(50, 2)] * 20).note
    assert fit.propose(MAC, 100, [(50, 2)] * 20).suggested is None
    mixed = [(c, 5 if i % 2 else 2) for i, c in enumerate(range(20, 200, 2))]
    assert "poorly" in fit.propose(MAC, 100, mixed).note


def test_alarm_pictures_are_scored() -> None:
    p = fit.propose(MAC, 100, _points(130), alarm_levels=[5, 4, 2])
    assert (p.alarm_pictures, p.alarm_dense) == (3, 1)  # 4 is full, not crammed


def test_the_unmasked_area_is_shaded(tmp_path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    mask = np.zeros((36, 64, 3), np.uint8)
    mask[:, :32] = 255
    cv2.imwrite(str(masks / f"{MAC}.png"), mask)
    shot = tmp_path / "s.jpg"
    cv2.imwrite(str(shot), np.full((36, 64, 3), 200, np.uint8))
    out = cv2.imdecode(np.frombuffer(MaskShader(masks).jpeg(shot, MAC), np.uint8), cv2.IMREAD_COLOR)
    assert out[:, :28].mean() > 190 and out[:, 36:].mean() < 70
    plain = cv2.imdecode(np.frombuffer(MaskShader(None).jpeg(shot, MAC), np.uint8), cv2.IMREAD_COLOR)
    assert plain.mean() > 190
