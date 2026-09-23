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

    @staticmethod
    def _verdict(name: str | None, score: float) -> dict[str, Any]:
        if score >= MATCH:
            certainty = "confident"
        elif score >= POSSIBLE:
            certainty = "possible"
        else:
            name, certainty = None, "unknown"
        return {"name": name, "similarity": round(score, 3), "certainty": certainty}

    def rank(self, embedding: list[float] | None) -> list[tuple[str, float]]:
        """Every stored voice scored against this one, best first."""
        if not embedding:
            return []
        scored = [
            (profile["name"], _cosine(embedding, profile.get("embedding", [])))
            for profile in self._load()["profiles"]
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def identify(self, embedding: list[float] | None) -> dict[str, Any]:
        """Best stored match for one voice, with an explicit confidence."""
        ranked = self.rank(embedding)
        if not ranked:
            return {"name": None, "similarity": 0.0, "certainty": "unknown"}
        return self._verdict(*ranked[0])

    def assign(self, embeddings: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
        """Match a meeting's voices to stored people, one person per voice.

        Scoring each voice on its own lets two different speakers both come back
        as the same person — the transcript then credits one participant with
        someone else's words. Claiming the strongest pairs first makes that
        impossible: the runner-up gets its next-best name, or none.
        """
        pairs = sorted(
            ((score, label, name)
             for label, embedding in embeddings.items()
             for name, score in self.rank(embedding)),
            key=lambda item: item[0], reverse=True,
        )
        taken: set[str] = set()
        best: dict[str, tuple[str, float]] = {}
        for score, label, name in pairs:
            if label in best or name.casefold() in taken:
                continue
            best[label] = (name, score)
            taken.add(name.casefold())
        return {
            label: self._verdict(*best[label]) if label in best
            else {"name": None, "similarity": 0.0, "certainty": "unknown"}
            for label in embeddings
        }

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
