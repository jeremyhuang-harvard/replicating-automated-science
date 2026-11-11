"""Generate and validate one scientist design per canonical scenario."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .call_log import JsonlCallLogger
from .hashing import stable_hash
from .models import ChatProvider, ChatRequest
from .scientist_design import DesignValidationError, design_hash, validate_design
from .scientist_prompt import render_scientist_messages

CANONICAL_SCENARIOS = (
    "two people bargaining over a mug",
    "a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud",
    "A person is interviewing for a job as a lawyer",
    "3 bidders participating in an auction for a piece of art starting at fifty dollars",
)
SCENARIO_SLUGS = (
    "mug-bargaining",
    "bail-setting",
    "lawyer-interview",
    "art-auction",
)
MAX_ATTEMPTS = 3
SCIENTIST_TEMPERATURE = 0.3
ProviderFactory = Callable[[int], ChatProvider]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_scientist_designs(
    output_dir: Path,
    *,
    provider_factory: ProviderFactory,
    model: str,
    scenarios: Sequence[str] = CANONICAL_SCENARIOS,
    seeds: Sequence[int],
) -> Path:
    """Accept the first valid design for each scenario, preserving every attempt."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if len(seeds) < len(scenarios) * MAX_ATTEMPTS:
        raise ValueError("seeds must provide three distinct values per scenario")
    if len(set(seeds)) != len(seeds):
        raise ValueError("all recorded seeds must be distinct")

    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    logger = JsonlCallLogger(output_dir / "raw_calls.jsonl")
    scenario_reports: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []
    seed_cursor = 0
    provider_name: str | None = None
    provider_configurations: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios):
        slug = SCENARIO_SLUGS[scenario_index] if scenarios == CANONICAL_SCENARIOS else f"scenario-{scenario_index + 1:02d}"
        attempts: list[dict[str, Any]] = []
        accepted: dict[str, Any] | None = None
        accepted_hash: str | None = None
        design_file: str | None = None

        for attempt_number in range(1, MAX_ATTEMPTS + 1):
            seed = seeds[seed_cursor]
            seed_cursor += 1
            provider = provider_factory(seed)
            provider_name = provider.name
            provider_configurations.append(dict(provider.configuration))
            request = ChatRequest(
                model=model,
                messages=render_scientist_messages(scenario),
                metadata={
                    "task": "scientist-design",
                    "scenario_index": scenario_index,
                    "scenario": scenario,
                    "attempt": attempt_number,
                    "seed": seed,
                    "thinking_enabled": False,
                },
                temperature=SCIENTIST_TEMPERATURE,
            )
            response = provider.complete(request)
            call_record = logger.append(request, response)
            prompt_hashes.append(str(call_record["prompt_hash"]))
            attempt_report: dict[str, Any] = {
                "attempt": attempt_number,
                "seed": seed,
                "call_hash": call_record["call_hash"],
            }

            if response.error is not None or response.text is None:
                attempt_report.update(status="provider_error", error=response.error or "missing response text")
                attempts.append(attempt_report)
                break
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as exc:
                attempt_report.update(status="malformed_json", error=str(exc))
                attempts.append(attempt_report)
                continue
            try:
                accepted = validate_design(parsed, expected_scenario=scenario)
            except DesignValidationError as exc:
                attempt_report.update(status="schema_invalid", error=str(exc))
                attempts.append(attempt_report)
                continue

            accepted_hash = design_hash(accepted)
            design_file = f"designs/{scenario_index + 1:02d}-{slug}.json"
            (output_dir / "designs").mkdir(exist_ok=True)
            _write_json(output_dir / design_file, accepted)
            attempt_report.update(status="valid", design_hash=accepted_hash)
            attempts.append(attempt_report)
            break

        scenario_reports.append(
            {
                "scenario_index": scenario_index,
                "scenario": scenario,
                "attempts": attempts,
                "valid": accepted is not None,
                "design_file": design_file,
                "design_hash": accepted_hash,
            }
        )

    total_attempts = sum(len(item["attempts"]) for item in scenario_reports)
    valid_designs = sum(bool(item["valid"]) for item in scenario_reports)
    report = {
        "schema_version": 1,
        "summary": {
            "scenarios": len(scenarios),
            "valid_designs": valid_designs,
            "total_attempts": total_attempts,
        },
        "scenarios": scenario_reports,
    }
    _write_json(output_dir / "validation_report.json", report)
    manifest = {
        "schema_version": 1,
        "run_type": "canonical-scientist-design-generation",
        "status": "complete" if valid_designs == len(scenarios) else "complete_with_invalid_scenarios",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider_name,
        "temperature": SCIENTIST_TEMPERATURE,
        "thinking_enabled": False,
        "max_attempts_per_scenario": MAX_ATTEMPTS,
        "retry_policy": "retry only malformed JSON or schema-invalid output; accept first valid",
        "scenarios": list(scenarios),
        "seeds_used": [attempt["seed"] for item in scenario_reports for attempt in item["attempts"]],
        "provider_configurations": provider_configurations[:total_attempts],
        "prompt_hashes": prompt_hashes,
        "scenario_set_hash": stable_hash(list(scenarios)),
        "artifacts": {
            "raw_calls": "raw_calls.jsonl",
            "validation_report": "validation_report.json",
            "design_directory": "designs",
        },
        "counts": report["summary"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir
