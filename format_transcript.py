#!/usr/bin/env python3
"""Convert an MLX Whisper JSON result to readable timestamped Markdown."""

import argparse
import json
from pathlib import Path


def timestamp(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Transcript")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    lines = [
        f"# {args.title}",
        "",
        f"Language: `{data.get('language', 'unknown')}`",
        "",
        "> This is an automated transcript. Check the recording before relying on names, numbers, or commitments.",
        "",
    ]
    for segment in data.get("segments", []):
        text = segment.get("text", "").strip()
        if text:
            lines.extend((f"**[{timestamp(segment['start'])}]**", "", text, ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
