"""Run one study end to end and persist manifest.json, responses.json, summary.json, report.html."""

from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import summarize
from .engines import Engine
from .panel import default_panel
from .parsing import ParseError, parse_answers
from .prompts import build_persona_prompt
from .report import render_report
from .schemas import Concept, Response, StudyManifest
from .survey import default_questions

ARTIFACTS = ("manifest.json", "responses.json", "summary.json", "report.html")


@dataclass(frozen=True)
class StudyResult:
    manifest: StudyManifest
    responses: list[Response]
    summary: dict[str, Any]
    paths: dict[str, Path]


def run_study(
    concept: Concept,
    engine: Engine,
    output_dir: str | Path,
    panel_size: int = 12,
    seed: int = 42,
    now: datetime | None = None,
    study_id: str | None = None,
) -> StudyResult:
    """Engine-level failures (EngineError) propagate before anything is written; a single persona's
    unusable output is recorded as an error Response and surfaced in the summary warnings."""
    now = now or datetime.now(timezone.utc)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    panel, questions = default_panel(panel_size), default_questions()
    prompts = {p.persona_id: build_persona_prompt(concept, p, questions) for p in panel}
    manifest = StudyManifest(
        study_id=study_id or f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}",
        created_at=now.isoformat(),
        engine=engine.name,
        model=engine.model,
        seed=seed,
        concept=concept,
        panel=panel,
        questions=questions,
        prompts=prompts,
        engine_config=getattr(engine, "configuration", lambda: {})(),
        runtime={
            "parrot_baseline": __version__,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    )

    responses = []
    for persona in panel:
        raw = engine.generate(concept, persona, questions, prompts[persona.persona_id], seed)
        try:
            responses.append(Response(persona.persona_id, persona.segment, parse_answers(raw, questions), raw))
        except ParseError as exc:
            responses.append(Response(persona.persona_id, persona.segment, {}, raw, error=str(exc)))

    summary = summarize(manifest, responses)
    contents = {
        "manifest.json": _dump(manifest.to_dict()),
        "responses.json": _dump([r.to_dict() for r in responses]),
        "summary.json": _dump(summary),
        "report.html": render_report(manifest, responses, summary),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        for name, text in contents.items():
            _write_text_synced(staging / name, text)
        _fsync_directory(staging)
        saved_empty_destination: Path | None = None
        if out.exists():
            # Preserve a caller-created empty directory (including its metadata)
            # so a failed commit can restore it exactly.
            saved_empty_destination = staging.with_name(staging.name + ".destination")
            out.replace(saved_empty_destination)
        try:
            staging.replace(out)
        except OSError:
            if saved_empty_destination is not None and not out.exists():
                saved_empty_destination.replace(out)
                _fsync_directory(out.parent)
            raise
        if saved_empty_destination is not None:
            saved_empty_destination.rmdir()
        _fsync_directory(out.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return StudyResult(manifest, responses, summary, {name: out / name for name in ARTIFACTS})


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _write_text_synced(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as artifact:
        artifact.write(text)
        artifact.flush()
        os.fsync(artifact.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
