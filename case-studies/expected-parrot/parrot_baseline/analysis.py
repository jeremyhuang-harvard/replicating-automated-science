"""Aggregation: means, rating distributions, representative quotes, and warnings."""

from __future__ import annotations

from statistics import mean
from typing import Any

from .schemas import Response, StudyManifest

MIN_SEGMENT_N = 5

LIMITATIONS = [
    "These are simulated hypotheses, not human evidence. No real consumer was surveyed.",
    "Synthetic personas reflect the prompt, the model and its training data; they can be confidently wrong, "
    "overly agreeable, and less varied than real people.",
    "Rating levels are not calibrated to real purchase behavior; compare concepts against each other, "
    "never read a mean as a forecast.",
    "Segments are defined by explicit product attitudes written by the study author, not measured populations.",
    "Validate any promising signal with a human panel before making decisions.",
]


def _rounded_mean(values: list[int]) -> float:
    return round(mean(values), 2)


def summarize(manifest: StudyManifest, responses: list[Response]) -> dict[str, Any]:
    ratings = [q.question_id for q in manifest.questions if q.kind == "rating"]
    texts = [q.question_id for q in manifest.questions if q.kind == "text"]
    segments = list(dict.fromkeys(p.segment for p in manifest.panel))
    ok, response_warnings = _reconcile(manifest, responses)
    by_seg = {s: [r for r in ok if r.segment == s] for s in segments}

    def means(group: list[Response]) -> dict[str, float]:
        return {q: _rounded_mean([r.answers[q] for r in group]) for q in ratings}

    def dist(group: list[Response]) -> dict[str, dict[str, int]]:
        return {q: {str(v): sum(r.answers[q] == v for r in group) for v in range(1, 6)} for q in ratings}

    present = {s: g for s, g in by_seg.items() if g}
    summary: dict[str, Any] = {
        "counts": {
            "panel": len(manifest.panel),
            "ok": len(ok),
            "failed": len(manifest.panel) - len(ok),
            "by_segment": {s: len(g) for s, g in by_seg.items()},
        },
        "means": {"overall": means(ok) if ok else {}, "by_segment": {s: means(g) for s, g in present.items()}},
        "distributions": {"overall": dist(ok), "by_segment": {s: dist(g) for s, g in present.items()}},
        "quotes": _quotes(present, ratings, texts),
        "warnings": response_warnings + _segment_warnings(by_seg),
        "limitations": list(LIMITATIONS),
    }
    return summary


def _quotes(by_seg: dict[str, list[Response]], ratings: list[str], texts: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    """Per segment, quote the respondent whose first rating is closest to the segment mean (ties: lowest id)."""
    quotes: dict[str, dict[str, dict[str, str]]] = {}
    for seg, group in by_seg.items():
        anchor = ratings[0] if ratings else None
        target = mean(r.answers[anchor] for r in group) if anchor else 0
        pick = min(group, key=lambda r: (abs(r.answers[anchor] - target) if anchor else 0, r.persona_id))
        quotes[seg] = {t: {"persona_id": pick.persona_id, "text": pick.answers[t]} for t in texts}
    return quotes


def _reconcile(manifest: StudyManifest, responses: list[Response]) -> tuple[list[Response], list[str]]:
    """Return one valid response per expected persona, using manifest-owned segments."""
    expected = {p.persona_id: p for p in manifest.panel}
    grouped: dict[str, list[Response]] = {}
    warnings: list[str] = []
    for response in responses:
        if response.persona_id not in expected:
            warnings.append(f"Unknown persona {response.persona_id} excluded.")
            continue
        grouped.setdefault(response.persona_id, []).append(response)

    reconciled: list[Response] = []
    for persona in manifest.panel:
        candidates = grouped.get(persona.persona_id, [])
        if not candidates:
            warnings.append(f"Response missing for persona {persona.persona_id}.")
        elif len(candidates) > 1:
            warnings.append(f"Duplicate responses for persona {persona.persona_id} excluded.")
        elif not candidates[0].ok:
            warnings.append(f"Persona {persona.persona_id} excluded: {candidates[0].error}")
        else:
            response = candidates[0]
            reconciled.append(Response(response.persona_id, persona.segment, response.answers, response.raw_output))
    return reconciled, warnings


def _segment_warnings(by_seg: dict[str, list[Response]]) -> list[str]:
    warnings = []
    for seg, group in by_seg.items():
        if not group:
            warnings.append(f"Segment {seg} has no valid responses.")
        elif len(group) < MIN_SEGMENT_N:
            warnings.append(f"Segment {seg} has a small sample (n={len(group)}); treat its means as anecdotal.")
    return warnings
