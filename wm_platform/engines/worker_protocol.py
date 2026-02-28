"""
Subprocess worker protocol for engine isolation.

Each engine adapter spawns a worker process in its respective venv.
Communication is JSON over stdin/stdout. Frames are passed as base64-encoded
numpy arrays or saved to shared temp files for large payloads.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np


def encode_frame(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_frame(b64: str) -> np.ndarray:
    buf = io.BytesIO(base64.b64decode(b64))
    return np.load(buf)


def _drain_stderr(pipe, collected: list[str]):
    """Read stderr in a background thread to prevent pipe buffer deadlocks."""
    try:
        for line in pipe:
            collected.append(line)
    except (ValueError, OSError):
        pass


class EngineWorkerClient:
    """Manages a long-running subprocess in a specific venv."""

    def __init__(self, python_bin: Path, worker_script: Path, env: Optional[dict] = None):
        self.python_bin = python_bin
        self.worker_script = worker_script
        self._proc: Optional[subprocess.Popen] = None
        self._env = env
        self._stderr_lines: list[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.alive:
            return
        self._stderr_lines = []
        self._proc = subprocess.Popen(
            [str(self.python_bin), str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._env,
        )
        self._stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(self._proc.stderr, self._stderr_lines),
            daemon=True,
        )
        self._stderr_thread.start()

    def send_command(self, cmd: str, **kwargs) -> dict[str, Any]:
        if not self.alive:
            raise RuntimeError("Worker process is not running")
        msg = json.dumps({"cmd": cmd, **kwargs})
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if not line:
                err = "".join(self._stderr_lines[-50:])
                raise RuntimeError(f"Worker died. stderr (last 50 lines):\n{err}")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def stop(self) -> None:
        if self.alive:
            try:
                self.send_command("shutdown")
            except Exception:
                pass
            self._proc.terminate()
            self._proc.wait(timeout=10)
            self._proc = None


def worker_main_loop(handler):
    """Standard main loop for worker processes. `handler` takes a dict, returns a dict."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("cmd") == "shutdown":
                print(json.dumps({"status": "ok", "msg": "shutting down"}), flush=True)
                break
            response = handler(request)
            print(json.dumps(response), flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}), flush=True)
