#!/usr/bin/env python3
"""Run local pyannote Community-1 diarization and save JSON plus RTTM."""

import argparse
import json
from pathlib import Path

from pyannote.audio import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-speakers", type=int)
    args = parser.parse_args()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=True,
    )

    options = {}
    if args.num_speakers is not None:
        options["num_speakers"] = args.num_speakers

    result = pipeline(str(args.audio), **options)
    diarization = result.speaker_diarization
    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, speaker in diarization
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"turns": turns}, indent=2) + "\n")
    with args.output.with_suffix(".rttm").open("w") as handle:
        diarization.write_rttm(handle)

    speakers = sorted({turn["speaker"] for turn in turns})
    print(f"Saved {len(turns)} turns across {len(speakers)} speakers.")
    print(f"Speakers: {', '.join(speakers)}")


if __name__ == "__main__":
    main()
