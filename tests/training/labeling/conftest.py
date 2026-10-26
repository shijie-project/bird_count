"""An in-memory Label Studio behind httpx.MockTransport: the endpoints the client uses."""

from __future__ import annotations

import base64
import json
import re
import time

import httpx
import pytest

from birdcount.training.labeling.client import LabelStudio


CONFIG = """<View>
  <KeyPointLabels name="dot-1" toName="img-1" strokeWidth="5">
    <Label value="chicken" background="red"/>
  </KeyPointLabels>
  <Image name="img-1" value="$img"/>
</View>"""


def jwt(claims: dict) -> str:
    def part(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'HS256', 'typ': 'JWT'})}.{part(claims)}.c2ln"


REFRESH = jwt({"token_type": "refresh", "exp": 9_999_999_999})


class FakeLabelStudio:
    def __init__(self, page_size_cap: int = 100):
        self.projects = {1: {"id": 1, "title": "Birds", "label_config": CONFIG}}
        self.tasks: dict[int, dict] = {}
        self.annotations: dict[int, dict] = {}
        self.predictions: dict[int, dict] = {}
        self.ids = iter(range(1000, 10**6))
        self.calls: list[tuple[str, str]] = []
        self.refreshes = 0
        self.access_ttl = 300
        self.page_size_cap = page_size_cap
        self._valid_access: set[str] = set()

    # -- state helpers -------------------------------------------------------------

    def add_task(self, img: str, annotations: list[dict] = (), project: int = 1) -> dict:
        task = {"id": next(self.ids), "project": project, "data": {"img": img}, "annotations": [], "predictions": []}
        self.tasks[task["id"]] = task
        for ann in annotations:
            self._annotate(task, ann)
        return task

    def _annotate(self, task: dict, ann: dict) -> dict:
        ann = {"id": next(self.ids), "result": [], "was_cancelled": False, "updated_at": "", **ann, "task": task["id"]}
        self.annotations[ann["id"]] = ann
        task["annotations"].append(ann)
        return ann

    def expire_access(self) -> None:
        self._valid_access.clear()

    # -- the server ------------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        if path == "/api/token/refresh/":
            if json.loads(request.content)["refresh"] != REFRESH:
                return httpx.Response(401, json={"detail": "bad refresh"})
            self.refreshes += 1
            access = jwt({"token_type": "access", "exp": time.time() + self.access_ttl, "n": self.refreshes})
            self._valid_access.add(access)
            return httpx.Response(200, json={"access": access})
        auth = request.headers.get("Authorization", "")
        if not (auth.startswith("Bearer ") and auth[7:] in self._valid_access) and auth != "Token legacy-token":
            return httpx.Response(401, json={"detail": "Authentication credentials were not provided."})
        body = json.loads(request.content) if request.content else None

        if m := re.fullmatch(r"/api/projects/(\d+)/", path):
            project = self.projects.get(int(m[1]))
            if project is None:
                return httpx.Response(404, json={"detail": "No Project matches the given query."})
            if method == "PATCH":
                project.update(body)
            return httpx.Response(200, json=project)
        if m := re.fullmatch(r"/api/projects/(\d+)/export", path):
            return httpx.Response(200, json=[t for t in self.tasks.values() if t["project"] == int(m[1])])
        if m := re.fullmatch(r"/api/projects/(\d+)/import", path):
            ids = []
            for t in body:
                task = self.add_task(t["data"]["img"], t.get("annotations", []), project=int(m[1]))
                ids.append(task["id"])
            return httpx.Response(201, json={"task_count": len(ids), "task_ids": ids})
        if path == "/api/tasks/":
            project = int(request.url.params["project"])
            if project not in self.projects:
                return httpx.Response(404, json={"detail": "No Project matches the given query."})
            page, size = int(request.url.params["page"]), min(int(request.url.params["page_size"]), self.page_size_cap)
            tasks = [t for t in self.tasks.values() if t["project"] == project]
            chunk = tasks[(page - 1) * size : page * size]
            if page > 1 and not chunk:
                return httpx.Response(404, json={"detail": "Invalid page."})
            return httpx.Response(200, json={"tasks": chunk, "total": len(tasks)})
        if m := re.fullmatch(r"/api/tasks/(\d+)/annotations/", path):
            return httpx.Response(201, json=self._annotate(self.tasks[int(m[1])], body))
        if m := re.fullmatch(r"/api/annotations/(\d+)/", path):
            ann = self.annotations.pop(int(m[1]))
            task = self.tasks[ann["task"]]
            task["annotations"] = [a for a in task["annotations"] if a["id"] != ann["id"]]
            return httpx.Response(204)
        if path == "/api/predictions/":
            pred = {"id": next(self.ids), **body}
            self.predictions[pred["id"]] = pred
            self.tasks[body["task"]]["predictions"].append(pred)
            return httpx.Response(201, json=pred)
        if m := re.fullmatch(r"/api/predictions/(\d+)/", path):
            self.predictions[int(m[1])].update(body)
            return httpx.Response(200, json=self.predictions[int(m[1])])
        return httpx.Response(404, json={"detail": f"no route {method} {path}"})


@pytest.fixture
def fake() -> FakeLabelStudio:
    return FakeLabelStudio()


@pytest.fixture
def ls(fake) -> LabelStudio:
    client = LabelStudio("http://ls.test", REFRESH, transport=httpx.MockTransport(fake.handler))
    yield client
    client.close()
