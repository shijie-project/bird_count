"""Starting and stopping the runtime from another process: the farm backend.

Stop: a request file next to the status file (STATUS_PATH with the suffix
.stop) naming the pid it is meant for. The runtime looks for it twice a
second and shuts down exactly as the operator's Terminate button does
(devices off, last recording finished, logs flushed). A request left over
from an earlier run names another pid and is ignored, and every run clears
it at start.

Start: `python -m birdcount.deploy` from the project root, in a console
window of its own (Windows), so it runs on with its panels and its log
whether or not the farm backend does.

Force stop: the whole process tree, for a runtime that no longer answers a
stop request. It skips the orderly shutdown, so a plug can stay on.

stdlib only: the runtime imports this.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from birdcount.core.paths import PROJECT_ROOT


def stop_file(status_path: Path) -> Path:
    return status_path.with_suffix(".stop")


def request_stop(status_path: Path, pid: int) -> None:
    path = stop_file(status_path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"pid": pid, "at": time.time()}), encoding="utf-8")
    os.replace(tmp, path)


def stop_requested(path: Path, pid: int) -> bool:
    """Whether the request file at `path` asks the process `pid` to stop."""
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("pid", -1)) == pid
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False  # mid-replace, or not ours: the next look decides


def clear_stop(path: Path) -> None:
    path.unlink(missing_ok=True)


def start_runtime() -> subprocess.Popen:
    cmd = [sys.executable, "-m", "birdcount.deploy"]
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen(cmd, cwd=PROJECT_ROOT, creationflags=flags)
    return subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _QUERY_LIMITED = 0x1000
    _STILL_ACTIVE = 259
    _ACCESS_DENIED = 5

    def _image(pid: int) -> str | None:
        """The executable of a live process: '' when it exists but cannot be read, None when it is gone."""
        handle = _kernel32.OpenProcess(_QUERY_LIMITED, False, pid)
        if not handle:
            return "" if ctypes.get_last_error() == _ACCESS_DENIED else None
        try:
            code = wintypes.DWORD()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
                return None
            buf = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buf))
            ok = _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            return buf.value if ok else ""
        finally:
            _kernel32.CloseHandle(handle)

else:

    def _image(pid: int) -> str | None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            return ""
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return ""


def alive(pid: int | None) -> bool:
    return bool(pid) and _image(pid) is not None


def force_stop(pid: int) -> None:
    """End the runtime `pid` and its children at once; only a Python process, so a reused pid is left alone."""
    image = _image(pid)
    if image is None:
        return
    if "python" not in Path(image).name.lower():
        raise ProcessLookupError(f"process {pid} is {Path(image).name or 'unknown'}, not the runtime")
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
