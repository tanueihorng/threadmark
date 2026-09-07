#!/usr/bin/env python3
"""Threadmark — a private, local meeting recorder that marks what matters."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import threading
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import accuracy
import meeting_context
import power
from audio_timeline import CHUNK_SECONDS, build_recording
from speaker_memory import SpeakerMemory

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RECORDINGS = ROOT / "recordings"
LIVE_MODEL = "mlx-community/whisper-small-mlx"
FINAL_MODELS = {
    "small": "mlx-community/whisper-small-mlx",
    "turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",
}
REFINE_MODEL = os.environ.get("THREADMARK_REFINE_MODEL", "")
LIVE_CONTEXT_SECONDS = 1.0

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
RECORDINGS.mkdir(exist_ok=True)
SPEAKERS = SpeakerMemory(ROOT / "speakers.json")


@dataclass
class SessionState:
    session_id: str
    language: str | None
    created_at: str
    directory: Path
    final_model: str = "turbo-q4"
    vocabulary: str = ""
    num_speakers: int | None = None
    self_correct: bool = True
    chunks: list[dict[str, Any]] = field(default_factory=list)
    status: str = "recording"
    stage: str = "Listening"
    error: str | None = None
    result: list[dict[str, Any]] | None = None
    review: list[dict[str, Any]] = field(default_factory=list)
    refinement: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    speakers: dict[str, Any] = field(default_factory=dict)
    flags: list[dict[str, Any]] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    audio_path: str | None = None


class SessionRequest(BaseModel):
    language: str = "en"
    final_model: str = "turbo-q4"
    vocabulary: str = ""
    num_speakers: int = 0
    self_correct: bool = True


class RenameRequest(BaseModel):
    label: str
    name: str
    remember: bool = True


class SegmentEdit(BaseModel):
    text: str


class FlagRequest(BaseModel):
    offset: float


SESSIONS: dict[str, SessionState] = {}
MODEL_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

app = FastAPI(title="Threadmark", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def manifest_path(state: SessionState) -> Path:
    return state.directory / "manifest.json"


def save_state(state: SessionState) -> None:
    with STATE_LOCK:
        payload = asdict(state)
        payload["directory"] = state.directory.name
        temp = state.directory / "manifest.tmp"
        temp.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
        temp.replace(manifest_path(state))


def load_sessions() -> None:
    known = {field_name for field_name in SessionState.__dataclass_fields__}
    for path in RECORDINGS.glob("*/manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["directory"] = path.parent
            state = SessionState(**{key: value for key, value in payload.items() if key in known})
            if state.status in {"recording", "queued", "finalizing"}:
                state.status = "interrupted"
                state.stage = "Recording recovered — ready to finalize"
                for chunk in state.chunks:
                    if chunk.get("status") in {"queued", "transcribing"}:
                        chunk["status"] = "skipped"
                        chunk["error"] = "Live preview skipped after restart; final audio is intact"
            SESSIONS[state.session_id] = state
            save_state(state)
        except (OSError, TypeError, ValueError):
            continue


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


load_sessions()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health(session_id: str | None = None) -> dict[str, Any]:
    state = SESSIONS.get(session_id or "")
    recorded = len(state.chunks) * CHUNK_SECONDS if state else 0.0
    report = power.host_report(RECORDINGS, recorded)
    return {"status": "ok", **report}


@app.get("/api/speakers")
def known_speakers() -> dict[str, Any]:
    return {"names": SPEAKERS.names()}


@app.post("/api/sessions")
def create_session(request: SessionRequest) -> dict[str, str]:
    if request.final_model not in FINAL_MODELS:
        raise HTTPException(status_code=400, detail="Unknown final transcription model")
    if shutil.disk_usage(RECORDINGS).free < 1024**3:
        raise HTTPException(status_code=507, detail="At least 1 GB of free disk space is required")
    session_id = uuid.uuid4().hex[:12]
    directory = RECORDINGS / session_id
    (directory / "chunks").mkdir(parents=True)
    state = SessionState(
        session_id=session_id,
        language=None if request.language == "auto" else request.language,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        directory=directory,
        final_model=request.final_model,
        vocabulary=request.vocabulary.strip(),
        num_speakers=request.num_speakers or None,
        self_correct=request.self_correct,
    )
    SESSIONS[session_id] = state
    save_state(state)
    power.hold_awake(session_id)
    return {"session_id": session_id, "status": state.status}


def get_session(session_id: str) -> SessionState:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Recording session not found")
    return state


def vocabulary_terms(state: SessionState) -> dict[str, Any]:
    return accuracy.parse_vocabulary(state.vocabulary)


# --------------------------------------------------------------------------- #
# Live preview
# --------------------------------------------------------------------------- #

def transcribe(path: Path, language: str | None, model: str, **options: Any) -> dict[str, Any]:
    import mlx_whisper

    with MODEL_LOCK:
        return mlx_whisper.transcribe(
            str(path), path_or_hf_repo=model, word_timestamps=True,
            language=language, verbose=None, **options,
        )


def _join_wav(previous: Path, current: Path, destination: Path, seconds: float) -> float:
    """Prepend the tail of the previous chunk so a sentence is not cut in half."""
    with wave.open(str(previous), "rb") as reader:
        channels, width, rate = reader.getnchannels(), reader.getsampwidth(), reader.getframerate()
        total = reader.getnframes()
        keep = min(total, int(rate * seconds))
        reader.setpos(total - keep)
        lead = reader.readframes(keep)
    with wave.open(str(current), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (channels, width, rate):
            return 0.0
        body = reader.readframes(reader.getnframes())
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(lead + body)
    return keep / rate


def transcribe_live_chunk(state: SessionState, index: int, path: Path) -> None:
    item = next((chunk for chunk in state.chunks if chunk["index"] == index), None)
    if item is None:
        return
    scratch: Path | None = None
    try:
        item["status"] = "transcribing"
        save_state(state)
        previous = next((chunk for chunk in state.chunks if chunk["index"] == index - 1), None)
        source, lead = path, 0.0
        previous_path = path.with_name(f"chunk-{index - 1:06d}.wav")
        if index and previous_path.exists():
            scratch = state.directory / f"live-{index:06d}.wav"
            lead = _join_wav(previous_path, path, scratch, LIVE_CONTEXT_SECONDS)
            if lead:
                source = scratch
        prompt = accuracy.build_prompt(
            vocabulary_terms(state)["terms"], (previous or {}).get("text", "")[-300:]
        )
        result = transcribe(
            source, state.language, LIVE_MODEL,
            initial_prompt=prompt, condition_on_previous_text=False,
        )
        words = accuracy.collect_words(result)
        kept = [word for word in words if word["end"] > lead + 0.05] if lead else words
        text = " ".join(word["text"] for word in kept).strip() or result.get("text", "").strip()
        confidence = round(sum(w["probability"] for w in kept) / len(kept), 3) if kept else 0.0
        item.update(
            text=text, language=result.get("language"), status="ready", confidence=confidence,
            uncertain=sum(1 for word in kept if word["probability"] < accuracy.WORD_LOW),
        )
    except Exception as exc:
        item.update(status="error", error=str(exc))
    finally:
        if scratch and scratch.exists():
            scratch.unlink(missing_ok=True)
        save_state(state)


@app.post("/api/sessions/{session_id}/chunks")
async def upload_chunk(
    session_id: str,
    audio: UploadFile = File(...),
    index: int = Form(...),
    offset: float = Form(...),
) -> dict[str, Any]:
    state = get_session(session_id)
    if state.status == "interrupted":
        state.status = "recording"
        state.stage = "Recording resumed after server restart"
        save_state(state)
    if state.status != "recording":
        raise HTTPException(status_code=409, detail="Recording has already stopped")
    existing = next((chunk for chunk in state.chunks if chunk["index"] == index), None)
    if existing:
        return existing
    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty audio chunk")
    if len(payload) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio chunk is unexpectedly large")
    path = state.directory / "chunks" / f"chunk-{index:06d}.wav"
    path.write_bytes(payload)
    item = {
        "index": index, "offset": offset, "text": "", "language": state.language,
        "filename": path.name, "status": "queued", "confidence": None, "uncertain": 0,
    }
    state.chunks.append(item)
    state.chunks.sort(key=lambda chunk: chunk["index"])
    power.hold_awake(session_id)
    save_state(state)
    task = asyncio.create_task(asyncio.to_thread(transcribe_live_chunk, state, index, path))
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
    return item


# --------------------------------------------------------------------------- #
# Speaker alignment
# --------------------------------------------------------------------------- #

def choose_speaker(start: float, end: float, turns: list[dict[str, Any]]) -> tuple[str, float]:
    """Return the best speaker for a word plus how well the turn actually covers it."""
    midpoint = (start + end) / 2
    duration = max(end - start, 0.01)
    best_speaker, best_overlap = "SPEAKER_UNKNOWN", 0.0
    nearest_speaker, nearest_distance = "SPEAKER_UNKNOWN", float("inf")
    for turn in turns:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, turn["speaker"]
        distance = 0.0 if turn["start"] <= midpoint <= turn["end"] else min(
            abs(midpoint - turn["start"]), abs(midpoint - turn["end"])
        )
        if distance < nearest_distance:
            nearest_distance, nearest_speaker = distance, turn["speaker"]
    if best_overlap > 0:
        return best_speaker, min(1.0, best_overlap / duration)
    if nearest_distance <= 1.0:
        return nearest_speaker, max(0.15, 0.5 - nearest_distance / 2)
    return "SPEAKER_UNKNOWN", 0.0


def group_words(words: list[dict[str, Any]], turns: list[dict[str, Any]],
                display: dict[str, str]) -> list[dict[str, Any]]:
    for word in words:
        label, confidence = choose_speaker(word["start"], word["end"], turns)
        word["label"] = label
        word["speaker_confidence"] = round(confidence, 3)
    groups: list[dict[str, Any]] = []
    for word in words:
        last = groups[-1] if groups else None
        if (last and last["label"] == word["label"]
                and word["start"] - last["end"] <= 2.5 and len(last["words"]) < 70):
            last["words"].append(word)
            last["end"] = word["end"]
        else:
            groups.append({
                "start": word["start"], "end": word["end"],
                "label": word["label"], "words": [word],
            })
    for group in groups:
        members = group["words"]
        group["text"] = " ".join(word["text"] for word in members).strip()
        group["confidence"] = round(sum(w["probability"] for w in members) / len(members), 3)
        group["speaker_confidence"] = round(
            sum(w["speaker_confidence"] for w in members) / len(members), 3
        )
        group["speaker"] = display.get(group["label"], group["label"])
        group["flags"] = sorted({flag for word in members for flag in word.get("flags", [])})
        group["edited"] = False
        keep = ("start", "end", "text", "probability", "flags", "alternative", "corrected_from")
        group["words"] = [
            {key: word[key] for key in keep if key in word} for word in members
        ]
    return groups


# --------------------------------------------------------------------------- #
# Finalization
# --------------------------------------------------------------------------- #

def rebuild_review(state: SessionState) -> None:
    items = accuracy.review_items(
        state.result or [], vocabulary_terms(state)["people"],
        state.quality.get("gaps", []), state.flags,
    )
    state.review = [item for item in items if item["id"] not in state.resolved]


def write_outputs(state: SessionState) -> None:
    payload = asdict(state)
    payload["directory"] = state.directory.name
    people = vocabulary_terms(state)["people"]
    arguments = (payload, state.result or [], state.review, people, state.quality, state.speakers)
    context = meeting_context.build_context(*arguments)
    bundle = meeting_context.build_bundle(*arguments)
    transcript = meeting_context.build_transcript(payload, state.result or [], state.quality)
    (state.directory / "context.md").write_text(context, encoding="utf-8")
    (state.directory / "context.json").write_text(
        json.dumps(bundle, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    (state.directory / "transcript.md").write_text(transcript, encoding="utf-8")
    (state.directory / "transcript.json").write_text(
        json.dumps({
            "segments": state.result, "review": state.review,
            "refinement": state.refinement, "quality": state.quality,
            "speakers": state.speakers,
        }, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def finalize_session(state: SessionState) -> None:
    try:
        state.status = "finalizing"
        while any(chunk.get("status") in {"queued", "transcribing"} for chunk in state.chunks):
            state.stage = "Completing live transcript backlog"
            save_state(state)
            time.sleep(0.25)

        state.stage = "Rebuilding the continuous recording"
        save_state(state)
        assembled = build_recording(state.directory)
        recording = assembled["path"]
        state.audio_path = str(recording.relative_to(ROOT))
        state.quality = {
            "duration": round(assembled["duration"], 2),
            "gaps": assembled["gaps"],
            "repaired_seconds": assembled["repaired_seconds"],
            "expected_segments": assembled["expected_segments"],
            "present_segments": assembled["present_segments"],
        }
        save_state(state)

        vocabulary = vocabulary_terms(state)
        state.stage = "Creating the higher-accuracy transcript"
        save_state(state)
        import mlx.core as mx

        transcription = transcribe(
            recording, state.language, FINAL_MODELS[state.final_model],
            initial_prompt=accuracy.build_prompt(vocabulary["terms"]),
        )
        (state.directory / "transcription.json").write_text(
            json.dumps(transcription, indent=2, default=json_default) + "\n", encoding="utf-8"
        )
        words = accuracy.collect_words(transcription)

        corrections = accuracy.correct_terms(words, vocabulary["terms"])
        windows: list[dict[str, Any]] = []
        report: list[dict[str, Any]] = []
        if state.self_correct:
            windows = accuracy.suspect_windows(words, assembled["gaps"])
            refine_model = REFINE_MODEL or FINAL_MODELS[state.final_model]

            def decode(start: float, end: float, prompt: str | None) -> dict[str, Any]:
                return transcribe(
                    recording, state.language, refine_model,
                    clip_timestamps=[start, end], initial_prompt=prompt,
                    temperature=0.0, condition_on_previous_text=False,
                )

            def progress(position: int, total: int) -> None:
                state.stage = f"Re-checking uncertain passage {position} of {total}"
                save_state(state)

            words, report = accuracy.refine_windows(
                words, windows, decode, vocabulary["terms"], progress
            )
            corrections += accuracy.correct_terms(words, vocabulary["terms"])
        state.refinement = report
        mx.clear_cache()
        gc.collect()

        state.stage = "Identifying speakers"
        save_state(state)
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=True)
        options = {"num_speakers": state.num_speakers} if state.num_speakers else {}
        output = pipeline(str(recording), **options)
        diarization = output.speaker_diarization
        turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, speaker in diarization
        ]
        embeddings: dict[str, list[float]] = {}
        vectors = getattr(output, "speaker_embeddings", None)
        if vectors is not None:
            for position, label in enumerate(diarization.labels()):
                try:
                    embeddings[label] = [float(value) for value in vectors[position]]
                except (IndexError, TypeError, ValueError):
                    continue
        (state.directory / "diarization.json").write_text(
            json.dumps({"turns": turns, "embeddings": embeddings}, indent=2) + "\n", encoding="utf-8"
        )
        del pipeline, output
        gc.collect()

        state.stage = "Matching voices to known speakers"
        save_state(state)
        state.speakers = {}
        display: dict[str, str] = {}
        for label in sorted({turn["speaker"] for turn in turns}):
            match = SPEAKERS.identify(embeddings.get(label))
            name = match["name"] if match["certainty"] == "confident" else label
            display[label] = name
            state.speakers[label] = {
                "label": label, "name": name, "suggested": match["name"],
                "certainty": match["certainty"], "similarity": match["similarity"],
                "seconds": round(sum(t["end"] - t["start"] for t in turns if t["speaker"] == label), 1),
                "embedding": embeddings.get(label, []),
            }

        state.stage = "Aligning words, speakers and review flags"
        save_state(state)
        groups = group_words(words, turns, display)
        state.result = groups
        confidences = [word["probability"] for group in groups for word in group["words"]]
        state.quality.update({
            "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "low_confidence_words": sum(1 for value in confidences if value < accuracy.WORD_LOW),
            "total_words": len(confidences),
            "vocabulary_corrections": corrections,
            "windows": len(windows),
            "refined": len(report),
            "replaced": sum(1 for item in report if item["outcome"] == "replaced"),
            "disputed": sum(1 for item in report if item["outcome"] == "disputed"),
            "confirmed": sum(1 for item in report if item["outcome"] == "confirmed"),
        })
        rebuild_review(state)
        write_outputs(state)
        state.status, state.stage = "complete", "Ready"
        save_state(state)
    except Exception as exc:
        state.status, state.stage, state.error = "error", "Finalization failed", str(exc)
        save_state(state)
    finally:
        power.release_awake(state.session_id)


@app.post("/api/sessions/{session_id}/finish")
async def finish_session(session_id: str) -> dict[str, str]:
    state = get_session(session_id)
    if state.status in {"queued", "finalizing", "complete"}:
        return {"status": state.status}
    if state.status not in {"recording", "interrupted", "error"}:
        raise HTTPException(status_code=409, detail="Session cannot be finalized")
    if not state.chunks:
        raise HTTPException(status_code=409, detail="No audio was recorded")
    state.status, state.stage, state.error = "queued", "Waiting to finalize", None
    power.release_awake(session_id)
    save_state(state)
    task = asyncio.create_task(asyncio.to_thread(finalize_session, state))
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
    return {"status": "queued"}


@app.get("/api/sessions/{session_id}")
def session_status(session_id: str) -> dict[str, Any]:
    state = get_session(session_id)
    if state.status != "recording":
        power.release_awake(session_id)
    if state.status == "complete" and (
        any("id" not in item for item in state.review)
        or not (state.directory / "context.json").exists()
    ):
        # Sessions finalized by an earlier build predate review ids and the
        # structured export; bring them up to date on first read.
        rebuild_review(state)
        write_outputs(state)
        save_state(state)
    pending = sum(chunk.get("status") in {"queued", "transcribing"} for chunk in state.chunks)
    ready = sum(chunk.get("status") == "ready" for chunk in state.chunks)
    complete = state.status == "complete"
    return {
        "session_id": state.session_id, "status": state.status, "stage": state.stage,
        "error": state.error, "chunks": state.chunks, "accepted_chunks": len(state.chunks),
        "ready_chunks": ready, "pending_chunks": pending, "result": state.result,
        "final_model": state.final_model, "review": state.review,
        "flags": state.flags, "resolved": len(state.resolved),
        "refinement": state.refinement, "quality": state.quality,
        "speakers": [
            {key: value for key, value in speaker.items() if key != "embedding"}
            for speaker in state.speakers.values()
        ],
        "known_speakers": SPEAKERS.names() if complete else [],
        "host": power.host_report(RECORDINGS, len(state.chunks) * CHUNK_SECONDS),
        "audio_url": f"/api/sessions/{state.session_id}/audio.wav" if state.audio_path else None,
        "export_url": f"/api/sessions/{state.session_id}/context.md" if complete else None,
        "transcript_url": f"/api/sessions/{state.session_id}/transcript.md" if complete else None,
        "bundle_url": f"/api/sessions/{state.session_id}/context.json" if complete else None,
    }


@app.post("/api/sessions/{session_id}/speakers")
def rename_speaker(session_id: str, request: RenameRequest) -> dict[str, Any]:
    state = get_session(session_id)
    speaker = state.speakers.get(request.label)
    if state.status != "complete" or speaker is None:
        raise HTTPException(status_code=409, detail="That speaker is not available yet")
    name = request.name.strip() or request.label
    speaker["name"] = name
    speaker["certainty"] = "named"
    for group in state.result or []:
        if group["label"] == request.label:
            group["speaker"] = name
    if request.remember and name != request.label:
        SPEAKERS.remember(name, speaker.get("embedding"), state.session_id)
    rebuild_review(state)
    write_outputs(state)
    save_state(state)
    return {"status": "ok", "name": name, "known_speakers": SPEAKERS.names()}


@app.post("/api/sessions/{session_id}/segments/{index}")
def edit_segment(session_id: str, index: int, request: SegmentEdit) -> dict[str, Any]:
    state = get_session(session_id)
    if state.status != "complete" or not state.result or index >= len(state.result):
        raise HTTPException(status_code=409, detail="That segment is not available")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Corrected text cannot be empty")
    return _apply_segment(state, index, text, "edited")


def _apply_segment(state: SessionState, index: int, text: str, flag: str) -> dict[str, Any]:
    group = (state.result or [])[index]
    group["text"], group["edited"] = text, True
    group["flags"] = sorted(set(group.get("flags", [])) | {flag})
    resolved = {item["id"] for item in state.review if item.get("segment") == index}
    state.resolved = sorted(set(state.resolved) | resolved)
    rebuild_review(state)
    write_outputs(state)
    save_state(state)
    return {"status": "ok", "segment": group, "review": state.review}


@app.post("/api/sessions/{session_id}/segments/{index}/accept")
def accept_alternative(session_id: str, index: int) -> dict[str, Any]:
    """Take the second pass's reading of a disputed passage."""
    state = get_session(session_id)
    if state.status != "complete" or not state.result or index >= len(state.result):
        raise HTTPException(status_code=409, detail="That segment is not available")
    alternative = accuracy.alternative_text(state.result[index])
    if not alternative:
        raise HTTPException(status_code=409, detail="That passage has no alternative reading")
    return _apply_segment(state, index, alternative, "accepted")


@app.post("/api/sessions/{session_id}/review/{item_id}/resolve")
def resolve_review(session_id: str, item_id: str) -> dict[str, Any]:
    """Mark a flagged passage as checked, without changing the transcript."""
    state = get_session(session_id)
    state.resolved = sorted(set(state.resolved) | {item_id})
    rebuild_review(state)
    write_outputs(state)
    save_state(state)
    return {"status": "ok", "review": state.review}


@app.post("/api/sessions/{session_id}/flags")
def flag_moment(session_id: str, request: FlagRequest) -> dict[str, Any]:
    """Bookmark the current moment while recording."""
    state = get_session(session_id)
    if state.status not in {"recording", "interrupted"}:
        raise HTTPException(status_code=409, detail="Moments can only be flagged while recording")
    state.flags.append({"offset": max(0.0, request.offset)})
    save_state(state)
    return {"status": "ok", "flags": len(state.flags)}


@app.get("/api/sessions/{session_id}/audio.wav")
def session_audio(session_id: str) -> FileResponse:
    state = get_session(session_id)
    path = state.directory / "recording.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="The continuous recording is not ready yet")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/sessions/{session_id}/context.md")
def export_context(session_id: str) -> FileResponse:
    return _export(session_id, "context.md", f"meeting-context-{session_id}.md")


@app.get("/api/sessions/{session_id}/transcript.md")
def export_transcript(session_id: str) -> FileResponse:
    return _export(session_id, "transcript.md", f"meeting-transcript-{session_id}.md")


@app.get("/api/sessions/{session_id}/context.json")
def export_bundle(session_id: str) -> FileResponse:
    return _export(session_id, "context.json", f"meeting-context-{session_id}.json",
                   "application/json")


def _export(session_id: str, filename: str, download: str,
            media_type: str = "text/markdown; charset=utf-8") -> FileResponse:
    state = get_session(session_id)
    path = state.directory / filename
    if state.status != "complete" or not path.exists():
        raise HTTPException(status_code=409, detail="The export is not ready yet")
    return FileResponse(path, media_type=media_type, filename=download)
