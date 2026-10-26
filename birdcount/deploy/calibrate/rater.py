"""Asking a vision model how crowded each picture is, against the fixed rubric.

`ClaudeRater` sends pictures to the Anthropic API: by default through the
Message Batches API (half price; results within hours, which suits a nightly
job), or one request per picture with `batch=False` (seconds; for a handful).
The API key is sent only in the request header and never logged.

A rater returns, per picture key, `(level, reason)` or an error message; one
bad answer never fails the run.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from birdcount.deploy.calibrate.rubric import PROMPT, SYSTEM, parse_reply


logger = logging.getLogger(__name__)

API = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
MAX_TOKENS = 120  # the JSON answer is ~40 tokens
BATCH_POLL_SEC = 30.0
BATCH_TIMEOUT_SEC = 24 * 3600.0  # the API's own limit for a batch
DIRECT_WORKERS = 4

Result = tuple[int, str] | str  # (level, reason), or why there is none


@dataclass(frozen=True)
class Picture:
    key: str  # unique within one call: letters, digits, '-' and '_', at most 64
    jpeg: bytes


class RaterError(RuntimeError):
    pass


class ClaudeRater:
    def __init__(self, api_key: str, model: str, *, batch: bool = True, client=None, poll_sec: float = BATCH_POLL_SEC):
        if not api_key:
            raise RaterError("no API key: set ANTHROPIC_API_KEY in .env")
        import httpx

        self.model = model
        self.batch = batch
        self.poll_sec = poll_sec
        headers = {"x-api-key": api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}
        self.client = client or httpx.Client(headers=headers, timeout=120.0)
        if client is not None:
            self.client.headers.update(headers)
        self.thinking: dict | None = None  # found by _no_thinking() before the first picture

    def _no_thinking(self) -> dict:
        """The request setting that keeps this model from thinking first.

        A model that thinks spends the small answer budget on thinking and
        returns no text; models spell "off" differently, and the API says
        which spelling it wants when given the wrong one.
        """
        if self.thinking is None:
            self.thinking = {"type": "disabled"}
            probe = {"model": self.model, "max_tokens": 5, "thinking": self.thinking,
                     "messages": [{"role": "user", "content": "Reply with OK."}]}  # fmt: skip
            r = self.client.post(f"{API}/messages", json=probe)
            if r.status_code == 400 and '"between_tools"' in _error_text(r):
                self.thinking = {"type": "between_tools"}
            elif r.status_code not in (200, 400):
                raise RaterError(f"the API refused a test request: HTTP {r.status_code}: {_error_text(r)}")
        return self.thinking

    def params(self, picture: Picture) -> dict:
        image = {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(picture.jpeg).decode()}
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "thinking": self._no_thinking(),
            "system": SYSTEM,
            "messages": [
                {"role": "user", "content": [{"type": "image", "source": image}, {"type": "text", "text": PROMPT}]}
            ],
        }

    def rate(self, pictures: list[Picture], on_progress: Callable[[str], None] | None = None) -> dict[str, Result]:
        if not pictures:
            return {}
        self._no_thinking()  # once, before the worker threads build requests
        return self._batch(pictures, on_progress) if self.batch else self._direct(pictures, on_progress)

    # --- one request per picture ----------------------------------------------------------------

    def _direct(self, pictures: list[Picture], on_progress) -> dict[str, Result]:
        def one(p: Picture) -> tuple[str, Result]:
            try:
                r = self.client.post(f"{API}/messages", json=self.params(p))
                if r.status_code != 200:
                    return p.key, f"HTTP {r.status_code}: {_error_text(r)}"
                return p.key, parse_reply(_text_of(r.json()))
            except Exception as e:  # one picture's failure is reported, not raised
                return p.key, f"{type(e).__name__}: {e}"

        out = {}
        with ThreadPoolExecutor(DIRECT_WORKERS) as pool:
            for i, (key, result) in enumerate(pool.map(one, pictures), 1):
                out[key] = result
                if on_progress and i % 20 == 0:
                    on_progress(f"{i}/{len(pictures)} rated")
        return out

    # --- the batch API --------------------------------------------------------------------------

    def _batch(self, pictures: list[Picture], on_progress) -> dict[str, Result]:
        requests = [{"custom_id": p.key, "params": self.params(p)} for p in pictures]
        r = self.client.post(f"{API}/messages/batches", json={"requests": requests})
        if r.status_code != 200:
            raise RaterError(f"creating the batch failed: HTTP {r.status_code}: {_error_text(r)}")
        batch = r.json()
        started = time.monotonic()
        while batch.get("processing_status") != "ended":
            if time.monotonic() - started > BATCH_TIMEOUT_SEC:
                raise RaterError(f"batch {batch.get('id')} did not finish in {BATCH_TIMEOUT_SEC / 3600:.0f} h")
            if on_progress:
                counts = batch.get("request_counts", {})
                on_progress(
                    f"batch {batch.get('id')}: {counts.get('succeeded', 0)} done, {counts.get('processing', 0)} waiting"
                )
            time.sleep(self.poll_sec)
            r = self.client.get(f"{API}/messages/batches/{batch['id']}")
            if r.status_code != 200:
                raise RaterError(f"reading batch {batch['id']} failed: HTTP {r.status_code}: {_error_text(r)}")
            batch = r.json()
        r = self.client.get(batch["results_url"])
        if r.status_code != 200:
            raise RaterError(f"downloading batch results failed: HTTP {r.status_code}")
        out: dict[str, Result] = {}
        for line in r.text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            res = item.get("result", {})
            try:
                out[item["custom_id"]] = (
                    parse_reply(_text_of(res["message"]))
                    if res.get("type") == "succeeded"
                    else f"{res.get('type')}: {res.get('error', '')}"
                )
            except (KeyError, ValueError) as e:
                out[item["custom_id"]] = f"{type(e).__name__}: {e}"
        return out


def _text_of(message: dict) -> str:
    return "".join(block.get("text", "") for block in message.get("content", []) if block.get("type") == "text")


def _error_text(response) -> str:
    try:
        return response.json().get("error", {}).get("message", "")[:300]
    except ValueError:
        return response.text[:300]
