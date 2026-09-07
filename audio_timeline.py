#!/usr/bin/env python3
"""Assemble recorded chunks into one continuous WAV without timeline drift.

Chunks are written by the browser at a fixed cadence. If an upload is lost or a
chunk is truncated, plain concatenation shortens the file and silently shifts
every later word timestamp and speaker turn. This module rebuilds the timeline
from chunk indices instead, padding missing or short audio with silence so that
chunk `i` always starts at `i * CHUNK_SECONDS`.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

CHUNK_SECONDS = 6.0
TARGET_RATE = 16000
# Encoders round chunk lengths by a few samples; only a real loss is worth reporting.
GAP_TOLERANCE = 0.05


def _meta(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.getnframes(),
        )


def _write(path: Path, channels: int, width: int, rate: int, frames: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(frames)


def _silence(path: Path, channels: int, width: int, rate: int, frames: int) -> None:
    _write(path, channels, width, rate, b"\x00" * (frames * channels * width))


def _read_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def build_recording(directory: Path) -> dict[str, Any]:
    """Return the continuous 16 kHz mono recording plus a report of any gaps."""
    chunk_dir = directory / "chunks"
    present = {
        int(path.stem.split("-")[1]): path
        for path in sorted(chunk_dir.glob("chunk-*.wav"))
        if path.stat().st_size > 44
    }
    if not present:
        raise RuntimeError("No audio chunks were recorded")

    reference = present[min(present)]
    channels, width, rate, _ = _meta(reference)
    expected_frames = round(rate * CHUNK_SECONDS)
    last_index = max(present)

    staging = directory / "timeline"
    staging.mkdir(exist_ok=True)
    for stale in staging.glob("*.wav"):
        stale.unlink()

    ordered: list[Path] = []
    gaps: list[dict[str, Any]] = []
    recovered_frames = 0

    for index in range(last_index + 1):
        start = index * CHUNK_SECONDS
        source = present.get(index)
        staged = staging / f"part-{index:06d}.wav"
        if source is None:
            _silence(staged, channels, width, rate, expected_frames)
            gaps.append({
                "index": index, "start": start, "end": start + CHUNK_SECONDS,
                "seconds": CHUNK_SECONDS, "reason": "missing segment",
            })
            recovered_frames += expected_frames
            ordered.append(staged)
            continue

        chunk_channels, chunk_width, chunk_rate, frames = _meta(source)
        consistent = (chunk_channels, chunk_width, chunk_rate) == (channels, width, rate)
        final = index == last_index
        if not consistent:
            gaps.append({
                "index": index, "start": start, "end": start + frames / max(chunk_rate, 1),
                "seconds": 0.0, "reason": "format changed mid-recording",
            })
            ordered.append(source)
            continue
        if not final and frames < expected_frames:
            missing = expected_frames - frames
            padding = b"\x00" * (missing * channels * width)
            _write(staged, channels, width, rate, _read_frames(source) + padding)
            if missing / rate > GAP_TOLERANCE:
                gaps.append({
                    "index": index, "start": start + frames / rate,
                    "end": start + CHUNK_SECONDS, "seconds": missing / rate,
                    "reason": "truncated segment",
                })
                recovered_frames += missing
            ordered.append(staged)
            continue
        if not final and frames > expected_frames:
            _write(staged, channels, width, rate, _read_frames(source)[: expected_frames * channels * width])
            ordered.append(staged)
            continue
        ordered.append(source)

    concat_file = directory / "chunks.txt"
    concat_file.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in ordered), encoding="utf-8"
    )
    output = directory / "recording.wav"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-vn", "-ac", "1", "-ar", str(TARGET_RATE), str(output),
    ], check=True)

    _, _, out_rate, out_frames = _meta(output)
    return {
        "path": output,
        "duration": out_frames / out_rate,
        "gaps": gaps,
        "repaired_seconds": round(recovered_frames / rate, 2),
        "expected_segments": last_index + 1,
        "present_segments": len(present),
    }
