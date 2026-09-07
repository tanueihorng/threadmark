#!/usr/bin/env python3
"""Keep the Mac awake while recording and report host risks worth warning about."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_HOLDERS: set[str] = set()
_PROCESS: subprocess.Popen[bytes] | None = None
_BATTERY_CACHE: tuple[float, dict[str, Any]] = (0.0, {})
BATTERY_TTL = 20.0


def hold_awake(session_id: str) -> None:
    """Block idle sleep and display sleep for as long as a session is recording."""
    global _PROCESS
    with _LOCK:
        _HOLDERS.add(session_id)
        if _PROCESS is None or _PROCESS.poll() is not None:
            try:
                _PROCESS = subprocess.Popen(
                    ["caffeinate", "-dimsu"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except (OSError, FileNotFoundError):
                _PROCESS = None


def release_awake(session_id: str) -> None:
    global _PROCESS
    with _LOCK:
        _HOLDERS.discard(session_id)
        if not _HOLDERS and _PROCESS is not None:
            _PROCESS.terminate()
            _PROCESS = None


def awake_held() -> bool:
    return bool(_HOLDERS) and _PROCESS is not None and _PROCESS.poll() is None


def battery() -> dict[str, Any]:
    global _BATTERY_CACHE
    stamp, cached = _BATTERY_CACHE
    if cached and time.monotonic() - stamp < BATTERY_TTL:
        return cached
    try:
        output = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=4
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    percent = re.search(r"(\d+)%", output)
    charging = "AC Power" in output or "charging" in output.lower()
    report = {
        "available": bool(percent),
        "percent": int(percent.group(1)) if percent else None,
        "on_power": charging,
    }
    _BATTERY_CACHE = (time.monotonic(), report)
    return report


def host_report(directory: Path, recorded_seconds: float = 0.0) -> dict[str, Any]:
    """Disk, battery and sleep state, plus the warnings a long meeting needs."""
    usage = shutil.disk_usage(directory)
    free_gb = usage.free / 1024**3
    power = battery()
    # 16 kHz mono PCM is 32 kB/s, roughly 0.115 GB per hour.
    hours_left = free_gb / 0.12
    warnings: list[str] = []
    if free_gb < 1.0:
        warnings.append(f"Only {free_gb:.1f} GB of disk is free — recording may stop soon.")
    elif hours_left < 2:
        warnings.append(f"Disk space allows roughly {hours_left:.1f} more hours of recording.")
    if power.get("available") and not power.get("on_power") and (power.get("percent") or 100) < 30:
        warnings.append(f"Battery is at {power['percent']}% and not charging — connect power.")
    if recorded_seconds > 3600 and not awake_held():
        warnings.append("Sleep prevention is not active; keep the Mac awake manually.")
    return {
        "free_gb": round(free_gb, 1),
        "recording_hours_left": round(hours_left, 1),
        "battery": power,
        "sleep_prevented": awake_held(),
        "warnings": warnings,
    }
