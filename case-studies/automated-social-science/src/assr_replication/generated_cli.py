"""CLI for the bounded generated-experiment pipeline."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generated_experiments import (
    analyze_study,
    enumerate_factorial,
    finalize_design,
    run_study,
    validate_finalized_design,
)
from .providers import MLXInProcessProvider

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pipeline(
    initial_dir: Path,
    output_dir: Path,
    *,
    provider: MLXInProcessProvider,
    model: str,
    base_seed: int,
) -> Path:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    design_paths = sorted(initial_dir.glob("*.json"))
    if len(design_paths) != 4:
        raise ValueError(f"expected exactly four initial designs, found {len(design_paths)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for study_index, design_path in enumerate(design_paths):
        slug = design_path.stem
        initial = _read_json(design_path)
        final_dir = output_dir / "finalizations" / slug
        final_path = final_dir / "finalized.json"
        if final_path.exists():
            finalized = validate_finalized_design(initial, _read_json(final_path))
        else:
            finalized = finalize_design(
                initial,
                provider=provider,
                model=model,
                output_dir=final_dir,
                seed=base_seed + study_index * 10_000,
            )
        cells = enumerate_factorial(finalized)
        if len(cells) != 27 or len({cell["cell_id"] for cell in cells}) != 27:
            raise ValueError(f"{slug}: expected 27 unique cells")
        _write_json(output_dir / "factorials" / f"{slug}.json", cells)
        prepared.append((slug, finalized, cells))
    if sum(len(item[2]) for item in prepared) != 108:
        raise ValueError("expected exactly 108 total cells")

    checkpoints: list[dict[str, Any]] = []
    for study_index, (slug, finalized, cells) in enumerate(prepared):
        study_started = time.monotonic()
        study_dir = output_dir / "studies" / slug
        run_study(
            finalized,
            cells,
            study_dir,
            provider=provider,
            model=model,
            base_seed=base_seed + 100_000 + study_index * 1_000_000,
        )
        rows = _read_jsonl(study_dir / "cells.jsonl")
        analysis = analyze_study(finalized, rows)
        _write_json(study_dir / "analysis.json", analysis)
        checkpoint = {
            "study": slug,
            "runtime_seconds": time.monotonic() - study_started,
            "cells": len(rows),
            "complete": analysis["counts"]["complete"],
            "valid_measurements": analysis["counts"]["valid_measurements"],
        }
        checkpoints.append(checkpoint)
        _write_json(output_dir / "checkpoint.json", {"completed_studies": checkpoints})

    manifest = {
        "schema_version": 1,
        "run_type": "generated-factorial-local-model",
        "status": "complete",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "model": model,
        "provider": provider.name,
        "provider_configuration": dict(provider.configuration),
        "subject_temperature": 0.3,
        "measurement_temperature": 0.0,
        "thinking_enabled": False,
        "tools": False,
        "retrieval": False,
        "base_seed": base_seed,
        "retry_policy": {
            "finalization": "up to 3 attempts for provider, JSON, or validation failure; retain all raw calls",
            "subject": "up to 2 attempts only for provider error or empty response; retain all raw calls and failures",
            "measurement": "up to 3 attempts for provider, strict-JSON, or schema failure; retain all raw calls and failures",
        },
        "counts": {"studies": 4, "cells_per_study": 27, "total_cells": 108},
        "checkpoints": checkpoints,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-seed", type=int, default=2026101400)
    parser.add_argument("--max-tokens", type=int, default=3200)
    args = parser.parse_args()
    provider = MLXInProcessProvider.load(args.model, max_tokens=args.max_tokens)
    print(run_pipeline(args.initial_dir, args.output_dir, provider=provider, model=args.model, base_seed=args.base_seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
