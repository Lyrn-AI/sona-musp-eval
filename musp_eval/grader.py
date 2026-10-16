"""Deterministic grader for MuSP-Bench answers.

MuSP-Bench encodes tolerance in the ``answers`` column (every accepted string).
This module only canonicalizes strings so trivially different spellings match.
No enharmonic equivalence, no partial credit.

Two modes:

* ``strict``  - unicode accidentals, case, whitespace, punctuation, markdown,
  ``M:SS`` -> ``MM:SS``, ``measure`` -> ``bar``, list splitting.
* ``lenient`` - strict plus: key/mode synonyms (``Am`` == ``A minor``),
  ordinal words (``4th`` == ``fourth``), ``bar 25, beat 1`` == ``bar 25 beat 1``,
  a bare integer or timestamp pulled out of a short phrase when the accepted
  answers are bare integers or timestamps, leading articles dropped.

Timestamps are exact in both modes; +-1 s tolerance is already in ``answers``
where the authors wanted it.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

ACCIDENTALS = {"♭": "b", "♯": "#", "\U0001d12a": "##", "\U0001d12b": "bb", "♮": ""}
DASHES = {"–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-"}
QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}

ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
    "6th": "sixth", "7th": "seventh", "8th": "eighth", "9th": "ninth", "10th": "tenth",
    "11th": "eleventh", "12th": "twelfth", "13th": "thirteenth", "14th": "fourteenth",
    "15th": "fifteenth",
}

ANSWER_LINE = re.compile(
    r"^\s*(?:[-*>]\s*)?\**\s*(?:final\s+)?answer\s*\**\s*[:：]\s*\**(.+?)\**\s*$",
    re.IGNORECASE | re.MULTILINE,
)
TIMESTAMP = re.compile(r"(?<![\d:])(\d{1,3}):(\d{2})(?:\.\d+)?(?![\d:])")
INTEGER = re.compile(r"(?<![\w.-])-?\d+(?![\w.])")
LIST_SEP = re.compile(r"\s*(?:,|;|\band\b|&)\s*")
UNORDERED = {"list", "list ordered alphabetically"}


@dataclass
class GradeResult:
    correct: bool
    extracted: str
    normalized: list[str]
    accepted_normalized: list[list[str]] = field(default_factory=list)
    mode: str = "strict"


def extract_answer(text: str) -> str:
    """Return the last ``Answer: ...`` line, else the last non-empty line."""
    if not text:
        return ""
    hits = ANSWER_LINE.findall(text)
    if hits:
        return hits[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _basic(s: str, case_sensitive: bool = False) -> str:
    s = unicodedata.normalize("NFKC", s)
    for table in (ACCIDENTALS, DASHES, QUOTES):
        for k, v in table.items():
            s = s.replace(k, v)
    s = re.sub(r"[`*_]", "", s)
    s = s.strip().strip("\"' \t")
    if not case_sensitive:
        s = s.lower()
    s = TIMESTAMP.sub(lambda m: f"{int(m.group(1)):02d}:{m.group(2)}", s)
    s = re.sub(r"\bmeasures?\b", "bar", s, flags=re.IGNORECASE)
    s = re.sub(r"\bbars\b", "bar", s, flags=re.IGNORECASE)
    s = re.sub(r"[()\[\]{}\"'!?]", " ", s)
    s = re.sub(r"\.+$", "", s.strip())
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*([:/])\s*", r"\1", s)
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", s)
    return s


def _lenient(s: str, answer_object_type: str, answer_format: str) -> str:
    fmt = f"{answer_object_type} {answer_format}".lower()
    s = re.sub(r"^the\s+", "", s)  # not "a"/"an": "a" is a pitch
    s = re.sub(r"\bbar (\d+),\s*beat\b", r"bar \1 beat", s)
    for k, v in ORDINALS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    if "key" in fmt or "tonality" in fmt or "mode" in fmt or "scale" in fmt:
        s = re.sub(r"\b([a-g](?:#|b)?)\s*(?:m|min|minor|-minor)\b", r"\1 minor", s)
        s = re.sub(r"\b([a-g](?:#|b)?)\s*(?:maj|major|-major)\b", r"\1 major", s)
    return s


def _split(s: str) -> list[str]:
    parts = [p.strip() for p in LIST_SEP.split(s) if p and p.strip()]
    return parts or [s]


def normalize(
    s: str,
    answer_object_type: str = "",
    answer_format: str = "",
    answer_quantity: str = "Single",
    mode: str = "strict",
) -> list[str]:
    """Canonicalize one answer string into an ordered list of items."""
    case_sensitive = "roman" in answer_format.lower()
    s = _basic(s, case_sensitive=case_sensitive)
    if mode == "lenient":
        s = _lenient(s, answer_object_type, answer_format)
    items = _split(s)
    if answer_quantity.strip().lower() in UNORDERED:
        items = sorted(items)
    return items


def _pull_bare(extracted: str, accepted: list[list[str]]) -> str | None:
    """Lenient only: if every accepted answer is a single integer or a single
    timestamp and the model's phrase contains exactly one such token, use it."""
    flat = [a for lst in accepted for a in lst]
    if not flat or any(len(lst) != 1 for lst in accepted):
        return None
    if all(INTEGER.fullmatch(a) for a in flat):
        found = INTEGER.findall(extracted)
        return found[0] if len(found) == 1 else None
    if all(re.fullmatch(r"\d{2}:\d{2}", a) for a in flat):
        found = TIMESTAMP.findall(extracted)
        return f"{int(found[0][0]):02d}:{found[0][1]}" if len(found) == 1 else None
    return None


def grade(
    model_output: str,
    answers: str | list[str],
    answer_object_type: str = "",
    answer_format: str = "",
    answer_quantity: str = "Single",
    mode: str = "strict",
) -> GradeResult:
    """Grade one model output against the accepted ``answers`` list."""
    accepted_raw = json.loads(answers) if isinstance(answers, str) else list(answers)
    accepted = [
        normalize(a, answer_object_type, answer_format, answer_quantity, mode)
        for a in accepted_raw
        if isinstance(a, str) and a.strip()
    ]
    extracted = extract_answer(model_output)
    got = normalize(extracted, answer_object_type, answer_format, answer_quantity, mode)
    correct = got in accepted
    if not correct and mode == "lenient":
        bare = _pull_bare(extracted, accepted)
        if bare is not None:
            got = [bare]
            correct = got in accepted
    return GradeResult(correct=correct, extracted=extracted, normalized=got,
                       accepted_normalized=accepted, mode=mode)


def grade_row(model_output: str, row, mode: str = "strict") -> GradeResult:
    """Convenience wrapper for a pandas row / dict from questions.csv."""
    g = row.get if hasattr(row, "get") else row.__getitem__
    return grade(
        model_output,
        g("answers"),
        answer_object_type=str(g("answer_object_type") or ""),
        answer_format=str(g("answer_format") or ""),
        answer_quantity=str(g("answer_quantity") or "Single"),
        mode=mode,
    )
