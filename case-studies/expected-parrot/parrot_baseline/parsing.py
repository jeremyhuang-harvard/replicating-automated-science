"""Robust extraction of one JSON object from noisy LLM output, and validation against the survey."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .schemas import Question

MAX_TEXT_CHARS = 1000
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class ParseError(ValueError):
    """Model output could not be turned into valid survey answers."""


def _snippet(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _balanced_end(text: str, start: int) -> int | None:
    """Index just past the brace-balanced object starting at text[start], honoring JSON strings."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _load(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", candidate))


def extract_json_object(raw: str, expected_keys: Sequence[str] = ()) -> dict[str, Any]:
    """Return the JSON object in `raw` that best matches `expected_keys` (ties go to the last one)."""
    text = _THINK.sub("", raw or "")
    if not text.strip():
        raise ParseError("model output was empty")
    candidates: list[dict[str, Any]] = []
    saw_unclosed = False
    pos = text.find("{")
    while pos != -1:
        end = _balanced_end(text, pos)
        if end is None:
            saw_unclosed = True
            break
        try:
            obj = _load(text[pos:end])
        except json.JSONDecodeError:
            pos = text.find("{", pos + 1)
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
        pos = text.find("{", end)
    if not candidates:
        reason = "JSON object appears truncated (unclosed brace)" if saw_unclosed else "no JSON object found"
        raise ParseError(f"{reason} in model output: {_snippet(text)!r}")

    def score(obj: dict[str, Any]) -> int:
        inner = obj.get("answers") if isinstance(obj.get("answers"), dict) else obj
        return sum(k in inner for k in expected_keys)

    best = max(range(len(candidates)), key=lambda i: (score(candidates[i]), i))
    return candidates[best]


def parse_answers(raw: str, questions: Sequence[Question]) -> dict[str, int | str]:
    """Extract and validate one answer per question; coerce "4"/4.0 to 4, strip and bound text."""
    obj = extract_json_object(raw, [q.question_id for q in questions])
    if isinstance(obj.get("answers"), dict):
        obj = obj["answers"]
    missing = [q.question_id for q in questions if q.question_id not in obj]
    if missing:
        raise ParseError(f"missing answers for: {', '.join(missing)}")
    answers: dict[str, int | str] = {}
    for q in questions:
        value = obj[q.question_id]
        if q.kind == "rating":
            answers[q.question_id] = _rating(q.question_id, value)
        else:
            if not isinstance(value, str) or not value.strip():
                raise ParseError(f"{q.question_id} must be non-empty text, got {value!r}")
            answers[q.question_id] = value.strip()[:MAX_TEXT_CHARS]
    return answers


def _rating(question_id: str, value: Any) -> int:
    if isinstance(value, str) and re.fullmatch(r"\s*[1-5]\s*", value):
        value = int(value)
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ParseError(f"{question_id} must be an integer 1-5, got {value!r}")
    return value
