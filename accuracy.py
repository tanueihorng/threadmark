#!/usr/bin/env python3
"""Confidence scoring, vocabulary conditioning and selective re-decoding.

Whisper already reports how unsure it was: per-segment `avg_logprob`,
`no_speech_prob` and `compression_ratio`, and a per-word `probability`. This
module keeps that evidence instead of discarding it, turns it into review flags,
and re-decodes only the passages that look wrong.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Callable, Iterable

WORD_LOW = 0.55
WORD_VERY_LOW = 0.35
# Whisper is routinely unsure about function words it gets right, so only the
# genuinely doubtful ones are marked in the exported text.
MARK_LOW = 0.4
SEGMENT_LOGPROB_LOW = -0.85
SEGMENT_NO_SPEECH = 0.55
SEGMENT_COMPRESSION = 2.2

WINDOW_PAD = 1.5
WINDOW_MERGE_GAP = 2.5
WINDOW_MIN = 4.0
WINDOW_MAX = 40.0
MAX_WINDOWS = 14
MAX_REFINE_SECONDS = 300.0
# Re-checking should cost a fraction of the first pass, not more than it. A
# three-minute recording was previously allowed 300 s of re-decoding — longer
# than the recording — because the budget ignored how much audio there was.
REFINE_SHARE = 0.2
MIN_REFINE_SECONDS = 30.0

PROMPT_CHARS = 780
# Ordinary English words are never "near misses" for a glossary term: correcting
# "more" to "MOE" is worse than leaving it alone.
COMMON_WORDS = frozenset("""
about above after again against all also and any are because been before being
below between both but came can come could did does doing down during each even
every few first for from further get give going good got had has have having her
here hers him his how into its itself just like made make many may me more most
much must never new next not now off once one only other our out over own point
right said same say see she should since some still such take than that the their
them then there these they thing think this those three through time too two under
until very want was way well went were what when where which while who whom why
will with within without would year yes you your
""".split())
FUZZY_CUTOFF = 0.86
ACRONYM_CUTOFF = 0.95
# A multi-word name is matched on the whole joined phrase, so a single wrong
# letter matters less than it does in one short token; but an invented phrase is
# a bigger lie than an invented word, so the bar is still high.
PHRASE_CUTOFF = 0.88
PHRASE_MAX_TOKENS = 4
PHRASE_LENGTH_SLACK = 0.2
# Two passes this close apart are the same reading, not a disagreement worth surfacing.
SAME_ENOUGH = 0.92

NUMBER_PATTERN = re.compile(r"\b\d[\d,.]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion)\b", re.I)
MONEY_PATTERN = re.compile(r"(?:[$€£¥]|\b(?:RM|USD|SGD|MYR|EUR|GBP))\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:k|m|bn|million|billion|dollars?|ringgit|euros?|pounds?)\b", re.I)
DEADLINE_PATTERN = re.compile(
    r"\b(?:by|before|due|deadline|no later than|eod|eow|cob)\b|"
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|"
    r"\b(?:today|tomorrow|tonight|next week|next month|this week|end of (?:the )?(?:day|week|month|quarter|year))\b|"
    r"\bq[1-4]\b|\b\d{1,2}(?:st|nd|rd|th)\b|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]+", "", text)


# --------------------------------------------------------------------------- #
# Meeting vocabulary
# --------------------------------------------------------------------------- #

def parse_vocabulary(raw: str) -> dict[str, Any]:
    """Read a free-form 'People: a, b / Projects: c / Terms: d' block."""
    people: list[str] = []
    other: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        label, _, remainder = line.partition(":")
        bucket = other
        if remainder and label.strip().casefold() in {"people", "names", "attendees", "speakers", "participants"}:
            bucket = people
        values = remainder if remainder else line
        for term in re.split(r"[,;/|]| and ", values):
            term = term.strip().strip(".")
            if 1 < len(term) <= 48 and term not in bucket:
                bucket.append(term)
    terms = people + [term for term in other if term not in people]
    return {"people": people, "terms": terms}


def build_prompt(terms: Iterable[str], tail: str = "") -> str | None:
    """Whisper conditions on ~224 prompt tokens; keep well inside that budget."""
    listed = ", ".join(dict.fromkeys(term for term in terms if term))
    parts = []
    if listed:
        parts.append(f"Glossary: {listed}.")
    if tail:
        parts.append(tail.strip())
    prompt = " ".join(parts).strip()
    if not prompt:
        return None
    if len(prompt) > PROMPT_CHARS:
        prompt = prompt[-PROMPT_CHARS:]
    return prompt


def _strip(text: str) -> str:
    return text.strip().strip(".,!?;:\"'()[]")


def _record_correction(word: dict[str, Any], stripped: str, replacement: str,
                       corrections: list[dict[str, Any]]) -> None:
    corrections.append({
        "start": word["start"], "end": word["end"],
        "from": stripped, "to": replacement,
    })
    word["text"] = word["text"].replace(stripped, replacement, 1)
    flags = word.setdefault("flags", [])
    if "vocabulary" not in flags:
        flags.append("vocabulary")
    word.setdefault("corrected_from", stripped)


def correct_phrases(words: list[dict[str, Any]], terms: list[str],
                    corrections: list[dict[str, Any]]) -> set[int]:
    """Snap runs of words onto known names, rewriting word boundaries if needed.

    Single-token matching cannot see any of this: neither half of a person's
    name is a near miss for the whole thing, and "trust bridge" is not a near
    miss for "TrustBridge" until the two words are joined. Matching the joined
    phrase is also safer than matching a token, because one wrong letter carries
    less weight across a whole name than across a short word.

    `words` is rewritten in place, since a match can change how many words the
    passage contains. The returned indices are positions in the rewritten list
    that later single-token correction must leave alone.
    """
    candidates: list[tuple[str, list[str]]] = []
    for term in terms:
        tokens = [token for token in term.split() if token]
        if tokens and len(tokens) <= PHRASE_MAX_TOKENS:
            candidates.append((normalize(term), tokens))
    if not candidates:
        return set()

    claimed: set[int] = set()
    index = 0
    while index < len(words):
        window_keys = [normalize(_strip(word["text"])) for word in words[index:index + PHRASE_MAX_TOKENS]]
        best: tuple[float, int, list[str]] | None = None
        for key, tokens in candidates:
            for size in range(1, len(window_keys) + 1):
                # One word against one term is the single-token pass's job.
                if size == 1 and len(tokens) == 1:
                    continue
                window = window_keys[:size]
                if not all(window):
                    continue
                joined = "".join(window)
                if joined == key:
                    ratio = 1.0
                else:
                    # A run made only of ordinary English words is almost
                    # certainly ordinary English, not a mangled project name.
                    if all(part in COMMON_WORDS for part in window):
                        continue
                    # Without a length guard a high ratio lets an extra word be
                    # swallowed: "and trust bridge" would become "TrustBridge".
                    if abs(len(joined) - len(key)) > PHRASE_LENGTH_SLACK * max(len(joined), len(key)):
                        continue
                    ratio = difflib.SequenceMatcher(None, joined, key).ratio()
                    if ratio < PHRASE_CUTOFF:
                        continue
                if best is None or ratio > best[0] or (ratio == best[0] and size < best[1]):
                    best = (ratio, size, tokens)
        if best is None:
            index += 1
            continue

        _, size, tokens = best
        block = words[index:index + size]
        surface = [_strip(word["text"]) for word in block]
        if surface == tokens:
            claimed.update(range(index, index + size))
            index += size
            continue

        if len(tokens) == size:
            for offset, token in enumerate(tokens):
                if surface[offset] != token:
                    _record_correction(block[offset], surface[offset], token, corrections)
            rewritten = block
        else:
            # The word boundaries themselves were wrong, so the passage is
            # rebuilt with one word per token and the span shared between them.
            start_at, end_at = block[0]["start"], block[-1]["end"]
            step = (end_at - start_at) / len(tokens)
            probability = round(sum(word["probability"] for word in block) / len(block), 4)
            flags = sorted({flag for word in block for flag in word.get("flags", [])} | {"vocabulary"})
            corrections.append({
                "start": start_at, "end": end_at,
                "from": " ".join(surface), "to": " ".join(tokens),
            })
            rewritten = [{
                "start": round(start_at + position * step, 3),
                "end": round(start_at + (position + 1) * step, 3),
                "text": token, "probability": probability,
                "segment": block[0].get("segment", {}), "flags": list(flags),
                "corrected_from": " ".join(surface),
            } for position, token in enumerate(tokens)]
            words[index:index + size] = rewritten

        claimed.update(range(index, index + len(rewritten)))
        index += len(rewritten)
    return claimed


def correct_terms(words: list[dict[str, Any]], terms: list[str]) -> list[dict[str, Any]]:
    """Snap near-miss transcriptions onto known names and jargon.

    Two failure modes matter here. Missing a real fix ("Digibunk" for "Digibank")
    loses information; inventing one ("more" heard as "MOE") corrupts the record.
    The second is worse, so ordinary English words are never rewritten and short
    acronyms have to match almost exactly.
    """
    corrections: list[dict[str, Any]] = []
    # Multi-word names first: "way hong" must become "Wei Hong" as a unit rather
    # than have "hong" fuzzed onto some unrelated single-token term.
    claimed = correct_phrases(words, terms, corrections)

    lookup: dict[str, str] = {}
    for term in terms:
        if " " not in term:
            lookup.setdefault(normalize(term), term)
    keys = list(lookup)

    for position, word in enumerate(words):
        if position in claimed:
            continue
        stripped = _strip(word["text"])
        if len(stripped) < 3 or not stripped.isalpha():
            continue
        key = normalize(stripped)
        if key in lookup:
            # Same word, wrong casing: "moe" should be written "MOE".
            if lookup[key] != stripped:
                _record_correction(word, stripped, lookup[key], corrections)
            continue
        if key in COMMON_WORDS:
            continue
        match = difflib.get_close_matches(key, keys, n=1, cutoff=FUZZY_CUTOFF)
        if not match:
            continue
        replacement = lookup[match[0]]
        ratio = difflib.SequenceMatcher(None, key, match[0]).ratio()
        acronym = replacement.isupper() and len(replacement) <= 6
        if acronym and ratio < ACRONYM_CUTOFF:
            continue
        _record_correction(word, stripped, replacement, corrections)
    return corrections


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #

def collect_words(transcription: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten Whisper output, carrying segment-level uncertainty onto words."""
    words: list[dict[str, Any]] = []
    for segment in transcription.get("segments", []):
        stats = {
            "avg_logprob": float(segment.get("avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(segment.get("no_speech_prob", 0.0) or 0.0),
            "compression_ratio": float(segment.get("compression_ratio", 0.0) or 0.0),
        }
        segment_flags = []
        if stats["avg_logprob"] < SEGMENT_LOGPROB_LOW:
            segment_flags.append("weak-segment")
        if stats["no_speech_prob"] > SEGMENT_NO_SPEECH:
            segment_flags.append("maybe-silence")
        if stats["compression_ratio"] > SEGMENT_COMPRESSION:
            segment_flags.append("repetitive-segment")
        for word in segment.get("words", []):
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            probability = float(word.get("probability", 1.0) or 0.0)
            flags = list(segment_flags)
            if probability < WORD_VERY_LOW:
                flags.append("very-low-confidence")
            elif probability < WORD_LOW:
                flags.append("low-confidence")
            words.append({
                "start": float(word["start"]), "end": float(word["end"]), "text": text,
                "probability": round(probability, 4), "segment": stats, "flags": flags,
            })
    words.sort(key=lambda word: (word["start"], word["end"]))
    return words


def repetition_spans(words: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Find stuck loops: a repeated word, or a repeated short phrase."""
    spans: list[tuple[float, float]] = []
    keys = [normalize(word["text"]) for word in words]
    index = 0
    while index < len(words):
        run = index
        while run + 1 < len(words) and keys[run + 1] and keys[run + 1] == keys[index]:
            run += 1
        if run - index >= 3 and len(keys[index]) > 1:
            spans.append((words[index]["start"], words[run]["end"]))
            index = run + 1
            continue
        index += 1
    for size in range(2, 7):
        position = 0
        while position + size * 3 <= len(words):
            block = keys[position:position + size]
            repeats = 1
            while keys[position + repeats * size: position + (repeats + 1) * size] == block:
                repeats += 1
            if repeats >= (3 if size < 3 else 2) and any(block):
                spans.append((words[position]["start"], words[position + repeats * size - 1]["end"]))
                position += repeats * size
                continue
            position += 1
    return spans


def refine_budget(duration: float) -> float:
    """How many seconds of audio the self-correction pass may decode again."""
    if duration <= 0:
        return MAX_REFINE_SECONDS
    return min(MAX_REFINE_SECONDS, max(MIN_REFINE_SECONDS, duration * REFINE_SHARE))


def suspect_windows(words: list[dict[str, Any]], gaps: list[dict[str, Any]] | None = None,
                    duration: float = 0.0) -> list[dict[str, Any]]:
    """Group flagged words into a few short passages worth transcribing again."""
    raw: list[dict[str, Any]] = []
    for word in words:
        reasons = [flag for flag in word["flags"] if flag != "vocabulary"]
        if reasons:
            raw.append({"start": word["start"], "end": word["end"], "reasons": set(reasons)})
    for start, end in repetition_spans(words):
        raw.append({"start": start, "end": end, "reasons": {"repetition"}})
    for gap in gaps or []:
        raw.append({
            "start": max(0.0, gap["start"] - 2.0), "end": gap["end"] + 2.0,
            "reasons": {"audio-gap"},
        })
    if not raw:
        return []
    raw.sort(key=lambda item: item["start"])

    windows: list[dict[str, Any]] = []
    for item in raw:
        start = max(0.0, item["start"] - WINDOW_PAD)
        end = item["end"] + WINDOW_PAD
        if windows and start - windows[-1]["end"] <= WINDOW_MERGE_GAP and end - windows[-1]["start"] <= WINDOW_MAX:
            windows[-1]["end"] = max(windows[-1]["end"], end)
            windows[-1]["reasons"] |= item["reasons"]
        else:
            windows.append({"start": start, "end": end, "reasons": set(item["reasons"])})

    scored = []
    for window in windows:
        window["end"] = min(window["end"], window["start"] + WINDOW_MAX)
        if window["end"] - window["start"] < WINDOW_MIN:
            window["end"] = window["start"] + WINDOW_MIN
        covered = [word for word in words if word["start"] < window["end"] and word["end"] > window["start"]]
        flagged = [word for word in covered if word["flags"]]
        window["reasons"] = sorted(window["reasons"])
        window["words"] = len(covered)
        window["score"] = len(flagged) + 3 * len(set(window["reasons"]) & {"repetition", "audio-gap", "repetitive-segment"})
        scored.append(window)

    scored.sort(key=lambda window: window["score"], reverse=True)
    chosen: list[dict[str, Any]] = []
    budget = refine_budget(duration)
    for window in scored:
        length = window["end"] - window["start"]
        if len(chosen) >= MAX_WINDOWS or length > budget:
            continue
        budget -= length
        chosen.append(window)
    chosen.sort(key=lambda window: window["start"])
    return chosen


# --------------------------------------------------------------------------- #
# Selective re-decoding
# --------------------------------------------------------------------------- #

def _mean_probability(words: list[dict[str, Any]]) -> float:
    values = [word["probability"] for word in words]
    return sum(values) / len(values) if values else 0.0


def _text_of(words: list[dict[str, Any]]) -> str:
    return " ".join(word["text"] for word in words).strip()


def mark_differences(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
    """Flag only the words the second pass actually heard differently."""
    left = [normalize(word["text"]) for word in before]
    right = [normalize(word["text"]) for word in after]
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    for tag, start, end, other_start, other_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        replaced = before[start:end]
        alternative = _text_of(after[other_start:other_end])
        if not replaced:
            # Pure insertion: anchor it to the preceding word so accepting the
            # second pass adds the words instead of swallowing that one.
            anchor = before[max(0, start - 1):start]
            if not anchor:
                continue
            replaced = anchor
            alternative = f"{anchor[0]['text']} {alternative}".strip()
        for word in replaced:
            if "disputed" not in word["flags"]:
                word["flags"].append("disputed")
            word["alternative"] = alternative or "(nothing)"


def refine_windows(
    words: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    decode: Callable[[float, float, str | None], dict[str, Any]],
    terms: list[str],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-transcribe each suspect window with tighter decoding and more context.

    The replacement is only accepted when the second pass is measurably more
    confident. Otherwise the original text is kept and marked as disputed, so an
    uncertain passage is surfaced rather than silently overwritten.
    """
    report: list[dict[str, Any]] = []
    for position, window in enumerate(windows, start=1):
        if progress:
            progress(position, len(windows))
        before = [w for w in words if w["start"] < window["end"] and w["end"] > window["start"]]
        if not before:
            continue
        lead = _text_of([w for w in words if w["end"] <= window["start"]][-40:])
        prompt = build_prompt(terms, lead[-400:])
        try:
            attempt = decode(window["start"], window["end"], prompt)
        except Exception as error:  # a failed window must not lose the transcript
            report.append({
                "start": window["start"], "end": window["end"], "reasons": window["reasons"],
                "outcome": "failed", "detail": str(error), "before": _text_of(before), "after": "",
            })
            continue
        after = [
            word for word in collect_words(attempt)
            if word["start"] < window["end"] and word["end"] > window["start"]
        ]
        original_text, new_text = _text_of(before), _text_of(after)
        original_score, new_score = _mean_probability(before), _mean_probability(after)
        similarity = difflib.SequenceMatcher(
            None, normalize(original_text), normalize(new_text)
        ).ratio() if (original_text or new_text) else 1.0
        changed = similarity < SAME_ENOUGH
        improved = bool(after) and new_score >= original_score + 0.02

        entry = {
            "start": round(window["start"], 2), "end": round(window["end"], 2),
            "reasons": window["reasons"], "before": original_text, "after": new_text,
            "confidence_before": round(original_score, 3), "confidence_after": round(new_score, 3),
            "similarity": round(similarity, 3),
        }
        if improved and changed:
            for word in after:
                word.setdefault("flags", []).append("re-decoded")
            words = [w for w in words if not (w["start"] < window["end"] and w["end"] > window["start"])]
            words.extend(after)
            words.sort(key=lambda word: (word["start"], word["end"]))
            entry["outcome"] = "replaced"
        elif changed:
            mark_differences(before, after)
            entry["outcome"] = "disputed"
        else:
            for word in before:
                if "confirmed" not in word["flags"]:
                    word["flags"].append("confirmed")
            entry["outcome"] = "confirmed"
        report.append(entry)
    return words, report


# --------------------------------------------------------------------------- #
# Review surface
# --------------------------------------------------------------------------- #

REVIEW_LABELS = {
    "flagged": "You flagged this",
    "very-low-confidence": "Very low confidence",
    "low-confidence": "Low confidence",
    "weak-segment": "Weak audio",
    "maybe-silence": "Possible non-speech",
    "repetitive-segment": "Repetition",
    "disputed": "Models disagree",
    "vocabulary": "Corrected to vocabulary",
    "audio-gap": "Audio gap",
    "number": "Number",
    "money": "Amount",
    "deadline": "Date or deadline",
    "name": "Name",
    "speaker-uncertain": "Uncertain speaker",
}


def alternative_text(group: dict[str, Any]) -> str | None:
    """What the second pass heard, rebuilt as a full replacement for the passage."""
    words = group.get("words", [])
    if not any("disputed" in word.get("flags", []) for word in words):
        return None
    parts: list[str] = []
    previous: str | None = None
    for word in words:
        alternative = word.get("alternative") if "disputed" in word.get("flags", []) else None
        if alternative is None:
            parts.append(word["text"])
            previous = None
        elif alternative != previous:
            if alternative != "(nothing)":
                parts.append(alternative)
            previous = alternative
    text = " ".join(parts).strip()
    return text or None


def review_items(groups: list[dict[str, Any]], people: list[str], gaps: list[dict[str, Any]],
                 flags: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Everything a human should glance at before trusting the transcript."""
    items: list[dict[str, Any]] = []
    people_keys = {normalize(name) for name in people}
    for index, group in enumerate(groups):
        if group.get("edited"):
            continue
        text = group["text"]
        kinds: list[str] = []
        if any("disputed" in word.get("flags", []) for word in group["words"]):
            kinds.append("disputed")
        if any("vocabulary" in word.get("flags", []) for word in group["words"]):
            kinds.append("vocabulary")
        if group["confidence"] < WORD_LOW:
            kinds.append("low-confidence")
        elif any(word["probability"] < WORD_VERY_LOW for word in group["words"]):
            kinds.append("very-low-confidence")
        if MONEY_PATTERN.search(text):
            kinds.append("money")
        elif NUMBER_PATTERN.search(text):
            kinds.append("number")
        if DEADLINE_PATTERN.search(text):
            kinds.append("deadline")
        if people_keys and any(normalize(token) in people_keys for token in re.findall(r"[A-Za-z']+", text)):
            kinds.append("name")
        if group["speaker"] == "SPEAKER_UNKNOWN" or group.get("speaker_confidence", 1.0) < 0.6:
            kinds.append("speaker-uncertain")
        if kinds:
            items.append({
                "id": f"{group['start']:.2f}:{index}",
                "segment": index, "start": group["start"], "end": group["end"],
                "speaker": group["speaker"], "text": text,
                "kinds": sorted(set(kinds)),
                "labels": [REVIEW_LABELS.get(kind, kind) for kind in sorted(set(kinds))],
                "confidence": group["confidence"],
            })
    for gap in gaps:
        items.append({
            "id": f"{gap['start']:.2f}:gap", "segment": None,
            "start": gap["start"], "end": gap["end"],
            "speaker": "—", "text": f"{gap['seconds']:.1f}s of audio was {gap['reason']} and padded with silence.",
            "kinds": ["audio-gap"], "labels": [REVIEW_LABELS["audio-gap"]], "confidence": 0.0,
        })
    for flag in flags or []:
        items.append({
            "id": f"{flag['offset']:.2f}:flag", "segment": None,
            "start": flag["offset"], "end": flag["offset"],
            "speaker": "—", "text": "You flagged this moment while recording.",
            "kinds": ["flagged"], "labels": [REVIEW_LABELS["flagged"]], "confidence": 0.0,
        })
    items.sort(key=lambda item: item["start"])
    for item in items:
        if item["segment"] is not None:
            item["alternative"] = alternative_text(groups[item["segment"]])
    return items
