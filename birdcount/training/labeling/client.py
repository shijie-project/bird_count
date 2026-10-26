"""A small Label Studio REST client: the endpoints this project uses, on httpx.

Not the label-studio-sdk: that package pulls in opencv-python-headless (which
shadows the GUI build the monitor and mask editor need), pandas and nltk, for
a handful of calls.

Authentication takes either kind of LS token:

* a personal access token (a JWT refresh token, LS >= 1.17 — the default, as
  legacy tokens are disabled per organisation): exchanged for a short-lived
  access token at /api/token/refresh/, renewed before it expires or on a 401;
* a legacy token: sent as is.

The token is never logged or put in an error message.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from collections.abc import Iterator

import httpx


logger = logging.getLogger(__name__)

PAGE_SIZE = 100
_RENEW_EARLY = 30.0  # seconds before an access token's expiry to renew it


class LabelStudioError(RuntimeError):
    pass


def jwt_claims(token: str) -> dict | None:
    """The payload of a JWT (unverified), or None when `token` is not one."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (binascii.Error, ValueError):
        return None
    return claims if isinstance(claims, dict) else None


class LabelStudio:
    def __init__(self, url: str, token: str, *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None):
        if not token:
            raise LabelStudioError(
                "no Label Studio token: set LABEL_STUDIO_API_KEY (Label Studio > Account & Settings > Access Token)"
            )
        self._token = token
        self._refreshable = (jwt_claims(token) or {}).get("token_type") == "refresh"
        self._access: str | None = None
        self._access_expires = 0.0
        self._http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LabelStudio:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- plumbing --------------------------------------------------------------

    def _auth(self, *, renew: bool = False) -> dict[str, str]:
        if not self._refreshable:
            return {"Authorization": f"Token {self._token}"}
        if renew or self._access is None or time.time() > self._access_expires - _RENEW_EARLY:
            r = self._http.post("/api/token/refresh/", json={"refresh": self._token})
            if r.status_code != 200:
                raise LabelStudioError(f"Label Studio refused the access token (HTTP {r.status_code}); is it revoked?")
            self._access = r.json()["access"]
            self._access_expires = float((jwt_claims(self._access) or {}).get("exp", time.time() + 60))
        return {"Authorization": f"Bearer {self._access}"}

    def request(self, method: str, path: str, **kwargs):
        r = self._http.request(method, path, headers=self._auth(), **kwargs)
        if r.status_code == 401 and self._refreshable:
            r = self._http.request(method, path, headers=self._auth(renew=True), **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail") or r.text
            except ValueError:
                detail = r.text
            raise LabelStudioError(f"{method} {path}: HTTP {r.status_code}: {str(detail)[:300]}")
        return r.json() if r.content else None

    # -- projects ---------------------------------------------------------------

    def project(self, project_id: int) -> dict:
        return self.request("GET", f"/api/projects/{project_id}/")

    def update_project(self, project_id: int, **fields) -> dict:
        return self.request("PATCH", f"/api/projects/{project_id}/", json=fields)

    def export(self, project_id: int) -> list[dict]:
        """Every task with its annotations, in LS's JSON export format."""
        return self.request(
            "GET", f"/api/projects/{project_id}/export", params={"exportType": "JSON", "download_all_tasks": "true"}
        )

    def import_tasks(self, project_id: int, tasks: list[dict]) -> dict:
        return self.request(
            "POST",
            f"/api/projects/{project_id}/import",
            params={"commit_to_project": "true", "return_task_ids": "true"},
            json=tasks,
        )

    # -- tasks, annotations, predictions ------------------------------------------

    def iter_tasks(self, project_id: int) -> Iterator[dict]:
        """Every task of the project with annotations and predictions, page by page."""
        page, seen = 1, 0
        while True:
            try:
                body = self.request(
                    "GET",
                    "/api/tasks/",
                    params={"project": project_id, "page": page, "page_size": PAGE_SIZE, "fields": "all"},
                )
            except LabelStudioError as e:
                if "HTTP 404" in str(e) and page > 1:  # past the last page
                    return
                raise
            tasks = body.get("tasks", []) if isinstance(body, dict) else body
            yield from tasks
            seen += len(tasks)
            total = body.get("total") if isinstance(body, dict) else None
            if not tasks or (total is not None and seen >= total):
                return
            page += 1

    def tasks(self, project_id: int) -> list[dict]:
        return list(self.iter_tasks(project_id))

    def create_annotation(self, task_id: int, annotation: dict) -> dict:
        return self.request("POST", f"/api/tasks/{task_id}/annotations/", json=annotation)

    def delete_annotation(self, annotation_id: int) -> None:
        self.request("DELETE", f"/api/annotations/{annotation_id}/")

    def create_prediction(self, task_id: int, result: list, model_version: str) -> dict:
        return self.request(
            "POST", "/api/predictions/", json={"task": task_id, "result": result, "model_version": model_version}
        )

    def update_prediction(self, prediction_id: int, task_id: int, result: list, model_version: str) -> dict:
        return self.request(
            "PATCH",
            f"/api/predictions/{prediction_id}/",
            json={"task": task_id, "result": result, "model_version": model_version},
        )
