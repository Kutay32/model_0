"""
Manage a single background `harmonia_vision.benchmark` subprocess and capture logs (ring buffer).
"""

from __future__ import annotations

import collections
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


class TrainingHub:
    """One training job at a time (benchmark subprocess)."""

    def __init__(self, log_max_lines: int = 8000) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._logs: collections.deque[str] = collections.deque(maxlen=log_max_lines)
        self._started_at: float | None = None
        self._last_cmd: list[str] | None = None
        self._reader_done = threading.Event()

    def _append_log(self, line: str) -> None:
        self._logs.append(line.rstrip("\n\r"))

    def _read_stdout(self) -> None:
        if not self._proc or not self._proc.stdout:
            return
        try:
            for line in self._proc.stdout:
                self._append_log(line)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[hub] log reader error: {exc}")
        finally:
            self._reader_done.set()
            if self._proc and self._proc.stdout:
                try:
                    self._proc.stdout.close()
                except OSError:
                    pass

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            pid = self._proc.pid if self._proc else None
            exit_code = None if running or not self._proc else self._proc.returncode
            return {
                "running": running,
                "pid": pid,
                "exit_code": exit_code,
                "started_at_unix": self._started_at,
                "command": self._last_cmd,
            }

    def logs_tail(self, n: int = 400) -> list[str]:
        n = max(1, min(n, 8000))
        with self._lock:
            return list(self._logs)[-n:]

    def start(self, cmd: list[str]) -> dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("A training job is already running. Stop it first.")
            self._logs.clear()
            self._reader_done.clear()
            self._last_cmd = list(cmd)
            self._started_at = time.time()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                env=env,
                text=True,
                bufsize=1,
                stdin=subprocess.DEVNULL,
            )
            threading.Thread(target=self._read_stdout, daemon=True).start()
            return {"pid": self._proc.pid, "command": self._last_cmd}

    def stop(self, grace_s: float = 5.0) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return {"stopped": False, "message": "No active training process."}
            proc = self._proc
        proc.terminate()
        t0 = time.time()
        while time.time() - t0 < grace_s:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()
        self._reader_done.wait(timeout=2.0)
        return {"stopped": True, "exit_code": proc.returncode}


hub = TrainingHub()
