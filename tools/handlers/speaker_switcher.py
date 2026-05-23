"""Manual GUI for poking a single speaker (play / stop / status).

Reads optional auth + timeout from environment variables — falls back to
the AXIS factory defaults if unset, so the tool still works out of the box
on a fresh deployment.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import scrolledtext

import httpx


# ================= Defaults (overridable via env) =================
USERNAME = os.environ.get("SPEAKER_USERNAME", "admin")
PASSWORD = os.environ.get("SPEAKER_PASSWORD", "admin")
AUDIO_FILE = os.environ.get("SPEAKER_AUDIO_FILE", "7MB.wav")
TIMEOUT_SEC = float(os.environ.get("SPEAKER_TIMEOUT", "5.0"))
# ==================================================================


def _url(ip: str, path: str) -> str:
    return f"http://{ip}/cgi-bin/{path}"


def _request(ip: str, endpoint: str, params: dict | None = None) -> httpx.Response:
    """Single-shot HTTP GET with basic auth; raises on non-2xx."""
    with httpx.Client(auth=(USERNAME, PASSWORD), timeout=TIMEOUT_SEC) as c:
        r = c.get(_url(ip, endpoint), params=params)
        r.raise_for_status()
        return r


def speaker_play(ip: str, filename: str) -> str:
    r = _request(ip, "audio_play", {"name": filename, "action": "start", "time": 1})
    return f"PLAY  {filename}  (HTTP {r.status_code})"


def speaker_stop(ip: str, filename: str) -> str:
    r = _request(ip, "audio_play", {"name": filename, "action": "stop", "time": 1})
    return f"STOP  {filename}  (HTTP {r.status_code})"


def speaker_status(ip: str, _filename: str) -> str:
    # `_filename` is accepted (and ignored) so every action shares one signature
    # — keeps `SpeakerGUI._run`'s dispatch table uniform.
    r = _request(ip, "audio_get_status")
    return f"STATUS\n{r.text.strip()}"


ActionFn = Callable[[str, str], str]


class SpeakerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Speaker Switcher")
        self.root.geometry("520x360")

        form = tk.Frame(root, padx=10, pady=10)
        form.pack(fill="x")

        tk.Label(form, text="IP:", width=10, anchor="w").grid(row=0, column=0, sticky="w")
        self.ip_var = tk.StringVar(value="192.168.0.150")
        tk.Entry(form, textvariable=self.ip_var).grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        tk.Label(form, text="File:", width=10, anchor="w").grid(row=1, column=0, sticky="w")
        self.file_var = tk.StringVar(value=AUDIO_FILE)
        tk.Entry(form, textvariable=self.file_var).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        form.columnconfigure(1, weight=1)

        btns = tk.Frame(root, padx=10)
        btns.pack(fill="x")
        self.play_btn = tk.Button(
            btns, text="PLAY", bg="#27ae60", fg="white", width=10, command=lambda: self._run("PLAY", speaker_play)
        )
        self.stop_btn = tk.Button(
            btns, text="STOP", bg="#c0392b", fg="white", width=10, command=lambda: self._run("STOP", speaker_stop)
        )
        self.status_btn = tk.Button(
            btns,
            text="STATUS",
            bg="#2c3e50",
            fg="white",
            width=10,
            command=lambda: self._run("STATUS", speaker_status),
        )
        self.play_btn.pack(side="left", padx=4, pady=6)
        self.stop_btn.pack(side="left", padx=4, pady=6)
        self.status_btn.pack(side="left", padx=4, pady=6)

        self.log = scrolledtext.ScrolledText(
            root, height=12, state="disabled", font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4"
        )
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for b in (self.play_btn, self.stop_btn, self.status_btn):
            b.configure(state=state)

    def _run(self, label: str, fn: ActionFn) -> None:
        ip = self.ip_var.get().strip()
        filename = self.file_var.get().strip() or AUDIO_FILE
        if not ip:
            self._append("[!] IP is empty.")
            return
        self._set_busy(True)
        self._append(f"--> [{ip}] {label}...")

        def worker() -> None:
            try:
                msg = fn(ip, filename)
                self.root.after(0, lambda: self._append(f"[{ip}] {msg}"))
            except Exception as exc:  # capture all paths under one bound name
                # Bind the message string at submit time so the lambda doesn't
                # close over a loop/scope variable that may have changed.
                err_text = f"[{ip}] ERROR: {exc}"
                self.root.after(0, lambda: self._append(err_text))
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    SpeakerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
