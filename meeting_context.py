#!/usr/bin/env python3
"""Turn a speaker-labelled transcript into agent context that cites its source.

Every extracted decision, commitment and open question keeps the timestamp and
speaker it came from, so an agent reading `context.md` can always fall back to
the evidence instead of trusting a summary.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from accuracy import DEADLINE_PATTERN, MARK_LOW, MONEY_PATTERN, WORD_LOW, normalize

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")

DECISION = re.compile(
    r"\b(?:we (?:decided|agreed|settled|concluded)|it(?:'s| is) (?:decided|agreed|settled)|"
    r"decision is|we(?:'re| are) going (?:with|ahead)|let(?:'s| us) go with|sign(?:ed)? off|"
    r"approved|we(?:'ll| will) (?:proceed|go) with|final(?:ised|ized)|no[- ]go|green ?light)\b",
    re.I,
)
ACTION = re.compile(
    r"\b(?:action item|to[- ]do|takeaway|i(?:'ll| will)|we(?:'ll| will)|you(?:'ll| will)|"
    r"(?:he|she|they)(?:'ll| will)|going to|gonna|need(?:s)? to|have to|has to|must|should|"
    r"let(?:'s| us)|please|follow(?:ing)? up|take (?:this|that|it) (?:on|over)|own(?:s|ing)? (?:this|that))\b",
    re.I,
)
QUESTION = re.compile(
    r"\b(?:not sure|unclear|to be (?:confirmed|decided)|tbc|tbd|open question|we(?:'ll| will) (?:check|find out)|"
    r"need(?:s)? to (?:check|confirm|verify|clarify)|does anyone know|who owns|pending|blocked (?:on|by)|waiting (?:on|for))\b",
    re.I,
)
RISK = re.compile(
    r"\b(?:risk|concern|worried|issue|problem|blocker|delay|slip(?:ping|page)?|escalat|compliance|breach|outage|penalt)\w*\b",
    re.I,
)
REQUIREMENT = re.compile(
    r"\b(?:requirement|must have|mandatory|policy|regulat\w+|standard|SLA|criteria|scope includes|out of scope)\b",
    re.I,
)

OWNER_PRONOUN = re.compile(r"\b(i|we|you|he|she|they)\b", re.I)


def clock(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]


def _owner(sentence: str, speaker: str, people: list[str]) -> str:
    keys = {normalize(name): name for name in people}
    for token in re.findall(r"[A-Za-z']+", sentence):
        name = keys.get(normalize(token))
        if name:
            return name
    pronoun = OWNER_PRONOUN.search(sentence)
    if pronoun and pronoun.group(1).casefold() in {"i", "we"}:
        return speaker
    return "unassigned"


def _deadline(sentence: str) -> str | None:
    match = DEADLINE_PATTERN.search(sentence)
    return match.group(0) if match else None


ITEM_TYPES = ("decision", "action", "question", "risk", "requirement", "amount")


def _classify(sentence: str) -> list[str]:
    """A sentence can be several things at once: a decision that is also a risk."""
    kinds: list[str] = []
    if DECISION.search(sentence):
        kinds.append("decision")
    elif ACTION.search(sentence):
        kinds.append("action")
    if QUESTION.search(sentence) or sentence.rstrip().endswith("?"):
        kinds.append("question")
    if RISK.search(sentence):
        kinds.append("risk")
    if REQUIREMENT.search(sentence):
        kinds.append("requirement")
    if MONEY_PATTERN.search(sentence):
        kinds.append("amount")
    return kinds


def extract(groups: list[dict[str, Any]], people: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Classify each sentence, keeping a citation back to speaker and timestamp.

    Every item carries the evidence it came from — segment index, time range and
    the verbatim quote — plus how confident the transcript was there and whether
    a person has since verified that passage. An agent reading this can weigh a
    claim instead of taking all of them equally.
    """
    found: dict[str, list[dict[str, Any]]] = {f"{kind}s": [] for kind in ITEM_TYPES}
    for index, group in enumerate(groups):
        for sentence in _sentences(group["text"]):
            if len(sentence) < 12:
                continue
            confidence = round(float(group.get("confidence", 1.0)), 3)
            verified = bool(group.get("edited"))
            citation = {
                "speaker": group["speaker"], "start": group["start"], "end": group["end"],
                "timestamp": clock(group["start"]), "text": sentence,
                "segment": index, "confidence": confidence, "verified": verified,
                "uncertain": confidence < WORD_LOW and not verified,
            }
            for kind in _classify(sentence):
                item = dict(citation)
                if kind == "action":
                    item["owner"] = _owner(sentence, group["speaker"], people)
                    item["due"] = _deadline(sentence)
                found[f"{kind}s"].append(item)
    return {key: _dedupe(value) for key, value in found.items()}


def participants(groups: list[dict[str, Any]], speakers: dict[str, Any]) -> list[dict[str, Any]]:
    """Who spoke, how much, and whether the name is a person's or a placeholder."""
    totals: dict[str, dict[str, Any]] = {}
    for group in groups:
        entry = totals.setdefault(group["label"], {
            "id": group["label"], "name": group["speaker"],
            "seconds": 0.0, "segments": 0, "words": 0,
        })
        entry["seconds"] += max(0.0, group["end"] - group["start"])
        entry["segments"] += 1
        entry["words"] += len(group.get("words") or group["text"].split())
    spoken = sum(entry["seconds"] for entry in totals.values()) or 1.0
    people = []
    for entry in totals.values():
        known = speakers.get(entry["id"], {})
        people.append({
            **entry,
            "seconds": round(entry["seconds"], 1),
            "share": round(entry["seconds"] / spoken, 3),
            "named": entry["name"] != entry["id"],
            "voice_match": known.get("certainty", "unknown"),
        })
    return sorted(people, key=lambda entry: entry["seconds"], reverse=True)


USAGE_RULES = [
    "Cite every claim as [timestamp] Speaker, using the values on the item you used.",
    "Treat this as evidence of what was said, not as a decision that was ratified.",
    "Never present an item with verified=false and confidence below 0.55 as fact; "
    "say the transcript is unclear there and give the timestamp.",
    "Text marked with a question mark in guillemets was transcribed with low confidence.",
    "An action with owner 'unassigned' has no owner in the recording. Do not invent one.",
    "Do not answer an open question on the speakers' behalf; surface it as open.",
    "Anything under needs_review was flagged and not resolved by a person.",
    "When items conflict, prefer the later timestamp and say that the record disagrees.",
]


def build_bundle(state: dict[str, Any], groups: list[dict[str, Any]], review: list[dict[str, Any]],
                 people: list[str], quality: dict[str, Any],
                 speakers: dict[str, Any] | None = None) -> dict[str, Any]:
    """The machine-readable companion to context.md."""
    found = extract(groups, people)
    words = quality.get("total_words", 0)
    low = quality.get("low_confidence_words", 0)
    items: list[dict[str, Any]] = []
    for kind in ITEM_TYPES:
        for position, entry in enumerate(found[f"{kind}s"], start=1):
            items.append({
                "id": f"{kind}-{position}", "type": kind,
                "text": entry["text"], "speaker": entry["speaker"],
                "timestamp": entry["timestamp"], "start": round(entry["start"], 2),
                "end": round(entry["end"], 2), "segment": entry["segment"],
                "confidence": entry["confidence"], "verified": entry["verified"],
                **({"owner": entry["owner"], "due": entry["due"]} if kind == "action" else {}),
                "evidence": {
                    "quote": entry["text"],
                    "audio": f"recording.wav#t={entry['start']:.2f},{entry['end']:.2f}",
                },
            })
    return {
        "schema": "threadmark.context/1",
        "meeting": {
            "recorded_at": state["created_at"],
            "duration_seconds": quality.get("duration", 0),
            "duration": clock(quality.get("duration", 0)),
            "language": state["language"] or "auto-detected",
            "model": state["final_model"],
        },
        "reliability": {
            "mean_word_confidence": quality.get("mean_confidence", 0),
            "high_confidence_share": round(1 - low / words, 3) if words else 0.0,
            "words": words, "low_confidence_words": low,
            "passages_rechecked": quality.get("refined", 0),
            "replaced": quality.get("replaced", 0),
            "disputed": quality.get("disputed", 0),
            "confirmed": quality.get("confirmed", 0),
            "audio_gaps": quality.get("gaps", []),
            "repaired_seconds": quality.get("repaired_seconds", 0),
            "human_verified_segments": sum(1 for group in groups if group.get("edited")),
            "unresolved_review_items": len(review),
        },
        "participants": participants(groups, speakers or {}),
        "items": items,
        "needs_review": [
            {"id": item["id"], "timestamp": clock(item["start"]), "start": round(item["start"], 2),
             "speaker": item["speaker"], "reasons": item["labels"], "text": item["text"]}
            for item in review
        ],
        "usage": {
            "citation_format": "[{timestamp}] {speaker}",
            "audio_reference": "recording.wav#t=<start>,<end>",
            "rules": USAGE_RULES,
        },
    }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Whisper repeats itself across segment boundaries; keep the fullest version."""
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda entry: len(entry["text"]), reverse=True):
        key = normalize(item["text"])
        if any(_overlap(key, normalize(other["text"])) >= 0.6 for other in kept):
            continue
        kept.append(item)
    return sorted(kept, key=lambda entry: entry["start"])


def _overlap(shorter: str, longer: str) -> float:
    """How much of one sentence is already contained in another."""
    if not shorter:
        return 1.0
    match = difflib.SequenceMatcher(None, shorter, longer, autojunk=False).find_longest_match(
        0, len(shorter), 0, len(longer)
    )
    return match.size / len(shorter)


def _cite(item: dict[str, Any]) -> str:
    mark = " *(low confidence — verify)*" if item.get("uncertain") else ""
    return f"- [{item['timestamp']}] **{item['speaker']}** — {item['text']}{mark}"


def build_context(state: dict[str, Any], groups: list[dict[str, Any]], review: list[dict[str, Any]],
                  people: list[str], quality: dict[str, Any],
                  speakers: dict[str, Any] | None = None) -> str:
    """The agent-facing digest: extracted items first, evidence always attached."""
    bundle = build_bundle(state, groups, review, people, quality, speakers)
    found = extract(groups, people)
    reliability = bundle["reliability"]
    lines = [
        "# Meeting context", "",
        "> Produced by Threadmark from a local recording. Items below are extracted",
        "> heuristically and each cites the speaker and second it came from.",
        "> `context.json` carries the same content in machine-readable form.", "",
        "## How to use this record", "",
        "- Cite every claim as `[timestamp] Speaker` — for example `[00:01:12] Wei Hong`.",
        *(f"- {rule}" for rule in USAGE_RULES[1:]),
        "",
        "## Reliability", "",
        f"- Transcript duration: {bundle['meeting']['duration']} · language "
        f"{bundle['meeting']['language']} · model `{bundle['meeting']['model']}`",
        f"- Mean word confidence: {reliability['mean_word_confidence']:.2f} "
        f"({reliability['high_confidence_share']:.0%} of {reliability['words']} words above the doubt threshold)",
        f"- Passages re-transcribed: {reliability['passages_rechecked']} "
        f"({reliability['replaced']} replaced, {reliability['confirmed']} confirmed, "
        f"{reliability['disputed']} still disputed)",
        f"- Audio repaired: {reliability['repaired_seconds']}s across "
        f"{len(reliability['audio_gaps'])} gap(s)",
        f"- Verified by a person: {reliability['human_verified_segments']} passage(s)",
        f"- Still flagged for review: {reliability['unresolved_review_items']} item(s)", "",
        "## Participants", "",
    ]
    for person in bundle["participants"]:
        naming = person["name"] if person["named"] else f"{person['id']} (unnamed)"
        lines.append(
            f"- **{naming}** — {person['share']:.0%} of speaking time "
            f"({person['seconds']}s across {person['segments']} turns)"
        )
    lines.append("")

    def section(title: str, key: str, empty: str) -> None:
        lines.extend([f"## {title}", ""])
        items = found[key]
        if not items:
            lines.extend([empty, ""])
            return
        for item in items:
            mark = _quality_mark(item)
            if key == "actions":
                due = f" · due **{item['due']}**" if item.get("due") else ""
                lines.append(
                    f"- **{item['owner']}**{due} — {item['text']} "
                    f"[{item['timestamp']}, {item['speaker']}]{mark}"
                )
            else:
                lines.append(f"- [{item['timestamp']}] **{item['speaker']}** — {item['text']}{mark}")
        lines.append("")

    section("Decisions", "decisions", "_No explicit decision language was detected._")
    section("Action items", "actions", "_No commitments were detected._")
    section("Open questions", "questions", "_No open questions were detected._")
    section("Risks and concerns", "risks", "_None detected._")
    section("Requirements and constraints", "requirements", "_None detected._")
    section("Amounts mentioned", "amounts", "_None detected._")

    lines.extend(["## Needs human review", ""])
    if review:
        for item in review[:120]:
            labels = ", ".join(item["labels"])
            lines.append(f"- [{clock(item['start'])}] {item['speaker']} — {labels}: {item['text'][:220]}")
        if len(review) > 120:
            lines.append(f"- _…and {len(review) - 120} further flagged passages in the review panel._")
    else:
        lines.append("_Nothing is outstanding._")
    lines.extend([
        "", "## Full transcript", "",
        "See `transcript.md` for the complete speaker-labelled record, and "
        "`recording.wav` for the audio each timestamp refers to.", "",
    ])
    return "\n".join(lines)


def _quality_mark(item: dict[str, Any]) -> str:
    if item.get("verified"):
        return " ✓ *verified by a person*"
    if item.get("uncertain"):
        return f" ⚠ *transcript confidence {item['confidence']:.2f} — verify before relying on this*"
    return ""


def build_transcript(state: dict[str, Any], groups: list[dict[str, Any]], quality: dict[str, Any]) -> str:
    """The complete record, with uncertain words marked rather than hidden."""
    lines = [
        "# Meeting transcript", "",
        f"Recorded {state['created_at']} · {clock(quality.get('duration', 0))} · "
        f"model `{state['final_model']}`", "",
        "> Words wrapped in `⟨?⟩` were transcribed with low confidence. Passages marked",
        "> **[disputed]** differ between transcription passes and were left as first heard.", "",
    ]
    for group in groups:
        if group.get("edited"):
            how = "second pass accepted" if "accepted" in group.get("flags", []) else "corrected by hand"
            lines.extend([
                f"### [{clock(group['start'])}] {group['speaker']} *({how})*", "",
                group["text"], "",
            ])
            continue
        marked: list[str] = []
        for word in group["words"]:
            text = word["text"]
            if word["probability"] < MARK_LOW or "disputed" in word.get("flags", []):
                text = f"⟨?⟩{text}"
            marked.append(text)
        disputed = " **[disputed]**" if any("disputed" in w.get("flags", []) for w in group["words"]) else ""
        lines.extend([
            f"### [{clock(group['start'])}] {group['speaker']}{disputed}", "",
            " ".join(marked).replace("⟨?⟩ ", "⟨?⟩"), "",
        ])
    return "\n".join(lines)
