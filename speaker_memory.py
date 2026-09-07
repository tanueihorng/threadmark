#!/usr/bin/env python3
"""Local speaker identity memory.

pyannote returns one embedding per detected speaker. Storing those against a
name lets later meetings recognise the same voice instead of starting again at
SPEAKER_00. Everything stays in a single JSON file inside the project.
"""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

MATCH = 0.62
POSSIBLE = 0.45

_LOCK = threading.Lock()


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class SpeakerMemory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"profiles": []}

    def _save(self, data: dict[str, Any]) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def names(self) -> list[str]:
        return sorted(profile["name"] for profile in self._load()["profiles"])

    def identify(self, embedding: list[float] | None) -> dict[str, Any]:
        """Best stored match for one voice, with an explicit confidence."""
        if not embedding:
            return {"name": None, "similarity": 0.0, "certainty": "unknown"}
        best_name, best_score = None, 0.0
        for profile in self._load()["profiles"]:
            score = _cosine(embedding, profile.get("embedding", []))
            if score > best_score:
                best_name, best_score = profile["name"], score
        if best_score >= MATCH:
            certainty = "confident"
        elif best_score >= POSSIBLE:
            certainty = "possible"
        else:
            best_name, certainty = None, "unknown"
        return {"name": best_name, "similarity": round(best_score, 3), "certainty": certainty}

    def remember(self, name: str, embedding: list[float] | None, session_id: str) -> None:
        """Blend a confirmed voice into the stored profile for that name."""
        name = name.strip()
        if not name or not embedding:
            return
        with _LOCK:
            data = self._load()
            for profile in data["profiles"]:
                if profile["name"].casefold() == name.casefold():
                    count = profile.get("samples", 1)
                    profile["embedding"] = [
                        (old * count + new) / (count + 1)
                        for old, new in zip(profile["embedding"], embedding)
                    ]
                    profile["samples"] = count + 1
                    profile["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
                    sessions = profile.setdefault("sessions", [])
                    if session_id not in sessions:
                        sessions.append(session_id)
                    break
            else:
                data["profiles"].append({
                    "name": name, "embedding": list(embedding), "samples": 1,
                    "sessions": [session_id],
                    "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
                })
            self._save(data)

    def forget(self, name: str) -> bool:
        with _LOCK:
            data = self._load()
            remaining = [p for p in data["profiles"] if p["name"].casefold() != name.strip().casefold()]
            removed = len(remaining) != len(data["profiles"])
            if removed:
                data["profiles"] = remaining
                self._save(data)
            return removed
