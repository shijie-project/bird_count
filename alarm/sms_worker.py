from __future__ import annotations

import json
import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from urllib import error, request

from .models import AlarmAction, WorkerConfig
from .time_utils import safe_name, utc_iso


class WorkerNotifier:
    def __init__(self, config: WorkerConfig, mock_root: str | Path):
        self.config = config
        self.mock_root = Path(mock_root)

    def notify(self, action: AlarmAction, snapshot_path: str | Path) -> dict:
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            return {
                "sent": False,
                "dry_run": self.config.dry_run,
                "reason": f"snapshot not found: {snapshot_path}",
                "snapshot_path": str(snapshot_path),
            }
        if not self.config.enabled:
            return {
                "sent": False,
                "dry_run": True,
                "reason": "worker disabled",
                "snapshot_path": str(snapshot_path),
            }
        if self.config.dry_run:
            return self._mock_notify(action, Path(snapshot_path))

        api_key = os.environ.get(self.config.api_key_env, "")
        if not api_key:
            return {
                "sent": False,
                "dry_run": False,
                "reason": f"missing API key env var: {self.config.api_key_env}",
                "snapshot_path": str(snapshot_path),
            }

        endpoint = f"{self.config.base_url}/api/projects/{self.config.project_slug}/images"
        fields = {
            "message_body": action.message,
            "camera_id": action.camera_id,
            "event_id": action.event_id,
            "action_type": action.action_type,
            "level": "" if action.level is None else str(action.level),
            "count": f"{action.count:.3f}",
            "threshold": f"{action.threshold:.3f}",
            "timestamp": f"{action.timestamp:.3f}",
            "timestamp_iso": utc_iso(action.timestamp),
        }
        body, content_type = encode_multipart(fields, snapshot_path, file_field="file")
        req = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "X-API-Key": api_key,
            },
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"raw": text}
                return {
                    "sent": True,
                    "dry_run": False,
                    "status": resp.status,
                    "endpoint": endpoint,
                    "response": payload,
                }
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"raw": text}
            return {
                "sent": False,
                "dry_run": False,
                "status": exc.code,
                "reason": exc.reason,
                "endpoint": endpoint,
                "response": payload,
            }
        except Exception as exc:
            return {"sent": False, "dry_run": False, "reason": str(exc), "endpoint": endpoint}

    def _mock_notify(self, action: AlarmAction, snapshot_path: Path) -> dict:
        image_id = str(uuid.uuid4())
        object_key = f"{self.config.project_slug}/{image_id}-{safe_upload_filename(snapshot_path.name)}"
        image_url = f"{self.config.base_url}/projects/{self.config.project_slug}/images/{image_id}"
        body = render_mock_message(action.message, image_url)

        image_dir = self.mock_root / "images" / safe_name(self.config.project_slug)
        image_dir.mkdir(parents=True, exist_ok=True)
        copied_image = image_dir / f"{image_id}-{safe_upload_filename(snapshot_path.name)}"
        if snapshot_path.exists():
            shutil.copy2(snapshot_path, copied_image)

        upload_url = f"{self.config.base_url}/api/projects/{self.config.project_slug}/images"
        recipients = self.config.mock_recipients or ("+61400000000",)
        sms_results = [
            {
                "sent": True,
                "to": recipient,
                "provider": "mock-twilio",
                "message_id": f"SM{uuid.uuid4().hex[:24]}",
                "reason": "dry-run: simulated by Chicken Alarm",
            }
            for recipient in recipients
        ]
        response = {
            "id": image_id,
            "project": {"name": self.config.project_slug, "slug": self.config.project_slug},
            "image_url": image_url,
            "message": {
                "template_id": None,
                "body": body,
                "scheduled_for": None,
                "status": "mock_sent",
            },
            "sms": sms_results[0],
            "sms_results": sms_results,
            "api": {
                "upload_url": upload_url,
                "image_url_pattern": f"{self.config.base_url}/projects/{self.config.project_slug}/images/{{image_id}}",
            },
        }

        log_record = {
            "dry_run": True,
            "created_at": utc_iso(action.timestamp),
            "request": {
                "method": "POST",
                "url": upload_url,
                "headers": {"X-API-Key": "<dry-run-redacted>"},
                "form": {
                    "file": str(snapshot_path),
                    "message_body": action.message,
                    "camera_id": action.camera_id,
                    "event_id": action.event_id,
                    "action_type": action.action_type,
                    "level": "" if action.level is None else str(action.level),
                    "count": f"{action.count:.3f}",
                    "threshold": f"{action.threshold:.3f}",
                    "timestamp": f"{action.timestamp:.3f}",
                    "timestamp_iso": utc_iso(action.timestamp),
                },
            },
            "mock_r2": {
                "object_key": object_key,
                "image_path": str(copied_image),
                "image_url": image_url,
            },
            "response": response,
        }
        log_path = self.mock_root / "mock_sms_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        return {
            "sent": True,
            "dry_run": True,
            "status": 201,
            "response": response,
            "mock_log_path": str(log_path),
            "mock_image_path": str(copied_image),
        }


def safe_upload_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name).strip("-")
    return cleaned or "snapshot.png"


def render_mock_message(message: str, image_url: str) -> str:
    body = message.replace("{image_url}", image_url)
    return body if image_url in body else f"{body}\n{image_url}"


def encode_multipart(fields: dict[str, str], file_path: Path, file_field: str = "file") -> tuple[bytes, str]:
    boundary = f"----chicken-alarm-{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for key, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    lines.extend(
        [
            f"--{boundary}\r\n".encode(),
            (f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n').encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"
