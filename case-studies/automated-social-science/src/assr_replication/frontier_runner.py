"""General runner for the frontier-planned four-study experiment bundle."""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .call_log import JsonlCallLogger
from .hashing import stable_hash
from .models import ChatMessage, ChatProvider, ChatRequest

BUNDLE_SCHEMA_VERSION = "frontier-design-bundle-v1"
EXPECTED_SCENARIOS = (
    "two people bargaining over a mug",
    "a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud",
    "A person is interviewing for a job as a lawyer",
    "3 bidders participating in an auction for a piece of art starting at fifty dollars",
)
OUTCOME_TYPES = {"binary", "number", "nullable_number", "bounded_number", "category"}
SUBJECT_TEMPERATURE = 0.3
MEASUREMENT_TEMPERATURE = 0.0
SUBJECT_MAX_ATTEMPTS = 2
MEASUREMENT_MAX_ATTEMPTS = 3
PROMPT_VERSION = "frontier-prompts-v3"
SUBJECT_SYSTEM_TEMPLATE = (
    "Scenario: {scenario}\n\nYou are {agent_name}. Role: {role}\n\nGoal: {goal}\n\n"
    "Constraints:\n- {constraints}\n\n[optional public instructions]\n\n[optional own private instructions]\n\n"
    "Act only as your assigned role. Produce only your next public utterance. Do not use tools or retrieval."
)
MEASUREMENT_SYSTEM_PROMPT = (
    "You are a measurement oracle, not an interaction participant. Use only the public transcript. "
    "Return one strict JSON object with exactly two keys: outcomes and evidence. outcomes must contain exactly "
    "the declared outcome names. Follow every declared type and bound. evidence must be one non-empty string "
    "or list of public transcript excerpts (an exact outcome-keyed evidence object is also accepted). Return no "
    "markdown or commentary. Do not use tools or retrieval."
)
_TEMPLATE_FIELD = re.compile(r"\$\{([^{}]+)\}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_bundle(value: Any) -> dict[str, Any]:
    """Validate and return an isolated copy of the fixed frontier design bundle."""
    _require(isinstance(value, dict), "bundle must be an object")
    _require(value.get("schema_version") == BUNDLE_SCHEMA_VERSION, "invalid bundle schema_version")
    studies = value.get("studies")
    _require(isinstance(studies, list) and len(studies) == 4, "bundle must contain exactly four studies")
    _require([study.get("scenario") for study in studies if isinstance(study, dict)] == list(EXPECTED_SCENARIOS), "bundle must preserve the four fixed scenario seeds in order")
    study_ids: set[str] = set()
    for study in studies:
        _require(isinstance(study, dict), "study must be an object")
        study_id = study.get("study_id")
        _require(isinstance(study_id, str) and study_id and study_id not in study_ids, "study_id must be non-empty and unique")
        study_ids.add(study_id)
        agents = study.get("agents")
        _require(isinstance(agents, list) and 1 <= len(agents) <= 4, f"{study_id}: invalid agents")
        agent_names = [agent.get("name") for agent in agents if isinstance(agent, dict)]
        _require(len(agent_names) == len(agents) and len(set(agent_names)) == len(agents), f"{study_id}: agent names must be unique")
        for agent in agents:
            _require(set(agent) == {"name", "role", "goal", "constraints"}, f"{study_id}: invalid agent fields")
            _require(all(isinstance(agent[field], str) and agent[field].strip() for field in ("name", "role", "goal")), f"{study_id}: invalid agent text")
            _require(isinstance(agent["constraints"], list) and agent["constraints"] and all(isinstance(item, str) and item.strip() for item in agent["constraints"]), f"{study_id}: invalid constraints")
        factors = study.get("factors")
        _require(isinstance(factors, list) and len(factors) == 5, f"{study_id}: expected five factors")
        factor_names = [factor.get("name") for factor in factors if isinstance(factor, dict)]
        _require(len(factor_names) == 5 and len(set(factor_names)) == 5, f"{study_id}: factor names must be unique")
        for factor in factors:
            _require(set(factor) == {"name", "prediction", "levels"}, f"{study_id}: invalid factor fields")
            _require(isinstance(factor["prediction"], str) and factor["prediction"].strip(), f"{study_id}: invalid prediction")
            levels = factor["levels"]
            _require(isinstance(levels, list) and len(levels) == 3, f"{study_id}: every factor needs three levels")
            _require(len({repr(level.get("value")) for level in levels if isinstance(level, dict)}) == 3, f"{study_id}: factor levels must be unique")
            for level in levels:
                _require(isinstance(level, dict) and "value" in level, f"{study_id}: invalid factor level")
                instruction_keys = set(level) - {"value"}
                _require(instruction_keys in ({"public"}, {"private"}, {"public_template"}), f"{study_id}: level must have exactly one instruction form")
                if "public" in level:
                    _require(isinstance(level["public"], str) and level["public"].strip(), f"{study_id}: invalid public instruction")
                if "private" in level:
                    private = level["private"]
                    _require(isinstance(private, dict) and private and set(private) <= set(agent_names), f"{study_id}: invalid private recipients")
                    _require(all(isinstance(text, str) and text.strip() for text in private.values()), f"{study_id}: invalid private instruction")
                if "public_template" in level:
                    template = level["public_template"]
                    _require(isinstance(template, str) and template.strip(), f"{study_id}: invalid public_template")
                    for field in _TEMPLATE_FIELD.findall(template):
                        base = field.removesuffix(" range")
                        _require(base in factor_names, f"{study_id}: unknown template field {field}")
        interaction = study.get("interaction")
        _require(isinstance(interaction, dict) and set(interaction) == {"order", "max_turns"}, f"{study_id}: invalid interaction")
        _require(interaction["order"] == agent_names, f"{study_id}: interaction order must contain each agent once")
        _require(isinstance(interaction["max_turns"], int) and interaction["max_turns"] > 0, f"{study_id}: invalid max_turns")
        outcomes = study.get("outcomes")
        _require(isinstance(outcomes, list) and outcomes, f"{study_id}: outcomes required")
        outcome_names: set[str] = set()
        for outcome in outcomes:
            _require(isinstance(outcome, dict), f"{study_id}: invalid outcome")
            name = outcome.get("name")
            outcome_type = outcome.get("type")
            _require(isinstance(name, str) and name and name not in outcome_names, f"{study_id}: outcome names must be unique")
            outcome_names.add(name)
            _require(outcome_type in OUTCOME_TYPES, f"{study_id}: unsupported outcome type")
            required = {"name", "type", "question"}
            if outcome_type == "bounded_number":
                required |= {"minimum", "maximum"}
                _require(isinstance(outcome.get("minimum"), (int, float)) and isinstance(outcome.get("maximum"), (int, float)) and outcome["minimum"] <= outcome["maximum"], f"{study_id}: invalid bounds")
            if outcome_type == "category":
                required |= {"allowed"}
                _require(isinstance(outcome.get("allowed"), list) and outcome["allowed"] and len(set(outcome["allowed"])) == len(outcome["allowed"]), f"{study_id}: invalid categories")
            _require(set(outcome) == required, f"{study_id}: invalid outcome fields")
            _require(isinstance(outcome["question"], str) and outcome["question"].strip(), f"{study_id}: invalid outcome question")
    return copy.deepcopy(value)


def generate_oa_cells(study: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate the OA(27,5,3,2) from five pairwise-independent F3 columns."""
    factors = study["factors"]
    _require(len(factors) == 5 and all(len(factor["levels"]) == 3 for factor in factors), "OA requires five three-level factors")
    cells: list[dict[str, Any]] = []
    for row_number, (a, b, c) in enumerate(itertools.product(range(3), repeat=3), start=1):
        indices = [a, b, c, (a + b) % 3, (a + 2 * b) % 3]
        treatments = {factor["name"]: factor["levels"][index]["value"] for factor, index in zip(factors, indices, strict=True)}
        payload = {"study_id": study["study_id"], "indices": indices, "treatments": treatments}
        cells.append(
            {
                "cell_id": f"{study['study_id']}-{row_number:02d}",
                "indices": indices,
                "treatments": treatments,
                "cell_hash": stable_hash(payload),
            }
        )
    _require(len(cells) == 27 and len({cell["cell_id"] for cell in cells}) == 27, "OA must contain 27 unique cells")
    for column in range(5):
        _require(all(sum(cell["indices"][column] == level for cell in cells) == 9 for level in range(3)), "OA level imbalance")
    for left in range(5):
        for right in range(left + 1, 5):
            _require(
                all(sum((cell["indices"][left], cell["indices"][right]) == (a, b) for cell in cells) == 3 for a in range(3) for b in range(3)),
                "OA pair imbalance",
            )
    return cells


def _format_number(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _expand_public_template(template: str, treatments: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        field = match.group(1)
        if field.endswith(" range"):
            value = treatments[field.removesuffix(" range")]
            _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"range template requires numeric value: {field}")
            return f"{_format_number(value - 25)}–${_format_number(value + 25)}"
        value = treatments[field]
        return _format_number(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)

    return _TEMPLATE_FIELD.sub(replace, template)


def materialize_cell(study: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """Resolve a planned cell into public instructions and per-agent private data."""
    public: list[str] = []
    private: dict[str, list[str]] = {agent["name"]: [] for agent in study["agents"]}
    for factor, index in zip(study["factors"], cell["indices"], strict=True):
        level = factor["levels"][index]
        if "public" in level:
            public.append(level["public"])
        elif "public_template" in level:
            public.append(_expand_public_template(level["public_template"], cell["treatments"]))
        else:
            for agent, instruction in level["private"].items():
                private[agent].append(instruction)
    return {
        **copy.deepcopy(cell),
        "public_instructions": public,
        "private_instructions": private,
        "materialized_hash": stable_hash(
            {
                "study_id": study["study_id"],
                "cell_hash": cell["cell_hash"],
                "public_instructions": public,
                "private_instructions": private,
            }
        ),
    }


def render_agent_messages(
    study: dict[str, Any],
    cell: dict[str, Any],
    agent_name: str,
    transcript: list[dict[str, Any]],
    *,
    turn: int,
) -> tuple[ChatMessage, ...]:
    """Render a privacy-minimized subject prompt for one acting agent."""
    agent = next((item for item in study["agents"] if item["name"] == agent_name), None)
    _require(agent is not None, f"unknown agent: {agent_name}")
    sections = [
        f"Scenario: {study['scenario']}",
        f"You are {agent_name}. Role: {agent['role']}",
        f"Goal: {agent['goal']}",
        "Constraints:\n- " + "\n- ".join(agent["constraints"]),
    ]
    if cell["public_instructions"]:
        sections.append("Public instructions:\n- " + "\n- ".join(cell["public_instructions"]))
    own_private = cell["private_instructions"].get(agent_name, [])
    if own_private:
        sections.append("Your private instructions (do not quote them verbatim):\n- " + "\n- ".join(own_private))
    sections.append("Act only as your assigned role. Produce only your next public utterance. Do not use tools or retrieval.")
    public_history = "\n".join(f"{item['agent']}: {item['text']}" for item in transcript) or "(no prior public turns)"
    return (
        ChatMessage(role="system", content="\n\n".join(sections)),
        ChatMessage(
            role="user",
            content=f"Public transcript so far:\n{public_history}\n\nThis is public turn {turn + 1} of at most {study['interaction']['max_turns']}. Give your next public utterance.",
        ),
    )


def parse_typed_measurement(text: str, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the exact multi-outcome JSON envelope and enforce declared types."""
    parsed = json.loads(text)
    _require(isinstance(parsed, dict) and set(parsed) == {"outcomes", "evidence"}, "measurement must contain exactly outcomes and evidence")
    names = {outcome["name"] for outcome in outcomes}
    values = parsed["outcomes"]
    evidence = parsed["evidence"]
    _require(isinstance(values, dict) and set(values) == names, "measurement outcome keys must match exactly")
    evidence_by_outcome = isinstance(evidence, dict)
    if evidence_by_outcome:
        _require(set(evidence) == names, "measurement evidence keys must match exactly")
    else:
        valid_global_evidence = isinstance(evidence, str) and bool(evidence.strip())
        if isinstance(evidence, list):
            valid_global_evidence = bool(evidence) and all(isinstance(item, str) and item.strip() for item in evidence)
        _require(valid_global_evidence, "measurement evidence must be a non-empty string, list, or exact outcome map")
    normalized: dict[str, Any] = {}
    for outcome in outcomes:
        name = outcome["name"]
        value = values[name]
        outcome_type = outcome["type"]
        if outcome_type == "binary":
            _require(type(value) is int and value in (0, 1), f"{name} must be binary 0 or 1")
            normalized[name] = value
        elif outcome_type in {"number", "bounded_number"}:
            _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), f"{name} must be a finite number")
            numeric = float(value)
            if outcome_type == "bounded_number":
                _require(float(outcome["minimum"]) <= numeric <= float(outcome["maximum"]), f"{name} is outside bounds")
            normalized[name] = numeric
        elif outcome_type == "nullable_number":
            _require(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))), f"{name} must be null or a finite number")
            normalized[name] = None if value is None else float(value)
        else:
            _require(isinstance(value, str) and value in outcome["allowed"], f"{name} must be an allowed category")
            normalized[name] = value
        if evidence_by_outcome:
            support = evidence[name]
            valid_support = isinstance(support, str) and bool(support.strip())
            if isinstance(support, list):
                valid_support = bool(support) and all(isinstance(item, str) and item.strip() for item in support)
            _require(valid_support, f"{name} evidence must be a non-empty string or list")
    return {"outcomes": normalized, "evidence": copy.deepcopy(evidence)}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_snapshot(path: Path, value: Any) -> None:
    if path.exists():
        _require(json.loads(path.read_text(encoding="utf-8")) == value, f"snapshot mismatch: {path}")
        return
    _write_json(path, value)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _existing_cell_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(json.loads(line)["cell_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _measurement_messages(
    study: dict[str, Any],
    transcript: list[dict[str, Any]],
    repair_feedback: str | None = None,
) -> tuple[ChatMessage, ...]:
    outcome_contract = []
    for outcome in study["outcomes"]:
        item = {key: outcome[key] for key in ("name", "type", "question")}
        if outcome["type"] == "bounded_number":
            item.update(minimum=outcome["minimum"], maximum=outcome["maximum"])
        if outcome["type"] == "category":
            item["allowed"] = outcome["allowed"]
        outcome_contract.append(item)
    transcript_text = "\n".join(f"{item['agent']}: {item['text']}" for item in transcript)
    user = "Outcome contract:\n" + json.dumps(outcome_contract, ensure_ascii=False, sort_keys=True) + "\n\nPublic transcript:\n" + transcript_text
    messages = [ChatMessage(role="system", content=MEASUREMENT_SYSTEM_PROMPT), ChatMessage(role="user", content=user)]
    if repair_feedback is not None:
        messages.append(
            ChatMessage(
                role="user",
                content="Your previous measurement was rejected. Correct only the JSON/type error and return the full two-key object again.\n" + repair_feedback,
            )
        )
    return tuple(messages)


def derive_cell_summary(study: dict[str, Any], cell: dict[str, Any], measurement: dict[str, Any] | None) -> dict[str, Any] | None:
    if measurement is None:
        return None
    values = measurement["outcomes"]
    study_id = study["study_id"]
    if study_id == "frontier-mug-bargaining":
        return {"deal": values["deal"] == 1, "price": values["price"]}
    if study_id == "frontier-bail-setting":
        return {"bail_amount": values["bail_amount"]}
    if study_id == "frontier-lawyer-interview":
        return {"hire": values["hire"] == 1, "rating": values["rating"]}
    if study_id == "frontier-art-auction":
        winner = values["winner"]
        valuations = {name: float(cell["treatments"][f"{name} valuation"]) for name in ("Bidder A", "Bidder B", "Bidder C")}
        maximum = max(valuations.values())
        winner_value = 0.0 if winner == "no sale" else valuations[winner]
        efficiency = winner_value / maximum if maximum else None
        return {
            "final_price": values["final_price"],
            "winner": winner,
            "winner_valuation": winner_value,
            "maximum_valuation": maximum,
            "allocative_efficiency": efficiency,
            "efficient_allocation": efficiency == 1.0,
        }
    raise ValueError(f"unsupported study_id: {study_id}")


def run_experiment(
    bundle: dict[str, Any],
    selected_cells: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    provider: ChatProvider,
    model: str,
    base_seed: int,
    wall_clock_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Run selected OA cells with append-only cell/call logs and deterministic seeds."""
    bundle = validate_bundle(bundle)
    selected_ids = set(selected_cells)
    studies_by_id = {study["study_id"]: study for study in bundle["studies"]}
    _require(selected_ids <= set(studies_by_id), "selected cells contain an unknown study")
    allowed_by_study = {study_id: {cell["cell_id"]: cell for cell in generate_oa_cells(study)} for study_id, study in studies_by_id.items()}
    for study_id, cells in selected_cells.items():
        _require(len({cell["cell_id"] for cell in cells}) == len(cells), f"{study_id}: duplicate selected cell")
        for cell in cells:
            _require(allowed_by_study[study_id].get(cell["cell_id"]) == cell, f"{study_id}: selected cell does not match canonical OA")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_snapshot = copy.deepcopy(bundle)
    config_snapshot = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "prompt_templates": {
            "subject_system_template": SUBJECT_SYSTEM_TEMPLATE,
            "measurement_system_prompt": MEASUREMENT_SYSTEM_PROMPT,
        },
        "model": model,
        "provider": provider.name,
        "provider_configuration": dict(provider.configuration),
        "base_seed": base_seed,
        "subject_temperature": SUBJECT_TEMPERATURE,
        "measurement_temperature": MEASUREMENT_TEMPERATURE,
        "thinking_enabled": False,
        "tools": False,
        "retrieval": False,
        "subject_max_attempts": SUBJECT_MAX_ATTEMPTS,
        "measurement_max_attempts": MEASUREMENT_MAX_ATTEMPTS,
    }
    _write_snapshot(output_dir / "design_bundle.json", bundle_snapshot)
    _write_snapshot(output_dir / "prompt_config_snapshot.json", config_snapshot)
    _write_snapshot(
        output_dir / "cell_plan.json",
        {study_id: cells for study_id, cells in selected_cells.items()},
    )
    prior_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) if (output_dir / "manifest.json").exists() else None
    started_at = prior_manifest["started_at"] if prior_manifest else datetime.now(timezone.utc).isoformat()
    run_started = time.monotonic()
    invocation_started_at = datetime.now(timezone.utc).isoformat()
    cells_started = 0
    cells_completed = 0
    cells_failed = 0
    cells_skipped = 0
    budget_exceeded = False

    for study_index, study in enumerate(bundle["studies"]):
        study_id = study["study_id"]
        study_cells = selected_cells.get(study_id, [])
        if not study_cells:
            continue
        study_dir = output_dir / "studies" / study_id
        rows_path = study_dir / "cells.jsonl"
        logger = JsonlCallLogger(study_dir / "raw_calls.jsonl")
        done = _existing_cell_ids(rows_path)
        for cell in study_cells:
            if cell["cell_id"] in done:
                cells_skipped += 1
                continue
            if wall_clock_budget_seconds is not None and time.monotonic() - run_started >= wall_clock_budget_seconds:
                budget_exceeded = True
                break
            cells_started += 1
            cell_started = time.monotonic()
            transcript: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            interaction_failed = False
            materialized = materialize_cell(study, cell)
            canonical_position = int(cell["cell_id"].rsplit("-", 1)[1]) - 1
            for turn in range(study["interaction"]["max_turns"]):
                agent = study["interaction"]["order"][turn % len(study["interaction"]["order"])]
                accepted = False
                for attempt in range(1, SUBJECT_MAX_ATTEMPTS + 1):
                    seed = base_seed + study_index * 1_000_000 + canonical_position * 10_000 + turn * 10 + attempt
                    request = ChatRequest(
                        model=model,
                        messages=render_agent_messages(study, materialized, agent, transcript, turn=turn),
                        temperature=SUBJECT_TEMPERATURE,
                        metadata={
                            "task": "subject-turn",
                            "study_id": study_id,
                            "cell_id": cell["cell_id"],
                            "cell_hash": cell["cell_hash"],
                            "agent": agent,
                            "turn": turn,
                            "attempt": attempt,
                            "seed": seed,
                            "max_tokens": 192,
                            "thinking_enabled": False,
                            "tools": False,
                            "retrieval": False,
                        },
                    )
                    response = provider.complete(request)
                    record = logger.append(request, response)
                    if response.error is None and response.text is not None and response.text.strip():
                        transcript.append({"turn": turn, "agent": agent, "text": response.text.strip(), "seed": seed, "call_hash": record["call_hash"]})
                        accepted = True
                        break
                    failures.append({"stage": "subject", "turn": turn, "agent": agent, "attempt": attempt, "seed": seed, "error": response.error or "empty response", "call_hash": record["call_hash"]})
                if not accepted:
                    interaction_failed = True
                    break

            measurement: dict[str, Any] | None = None
            measurement_attempts = 0
            measurement_error: str | None = None
            repair_feedback: str | None = None
            if not interaction_failed:
                for attempt in range(1, MEASUREMENT_MAX_ATTEMPTS + 1):
                    measurement_attempts = attempt
                    seed = base_seed + study_index * 1_000_000 + canonical_position * 10_000 + 9_000 + attempt
                    request = ChatRequest(
                        model=model,
                        messages=_measurement_messages(study, transcript, repair_feedback),
                        temperature=MEASUREMENT_TEMPERATURE,
                        metadata={
                            "task": "measurement",
                            "study_id": study_id,
                            "cell_id": cell["cell_id"],
                            "cell_hash": cell["cell_hash"],
                            "attempt": attempt,
                            "seed": seed,
                            "max_tokens": 384,
                            "thinking_enabled": False,
                            "tools": False,
                            "retrieval": False,
                            "outcome_contract": copy.deepcopy(study["outcomes"]),
                        },
                    )
                    response = provider.complete(request)
                    record = logger.append(request, response)
                    if response.error is not None or response.text is None:
                        measurement_error = response.error or "missing measurement response"
                    else:
                        try:
                            measurement = parse_typed_measurement(response.text, study["outcomes"])
                            measurement_error = None
                            break
                        except (json.JSONDecodeError, ValueError) as exc:
                            measurement_error = str(exc)
                    failures.append({"stage": "measurement", "attempt": attempt, "seed": seed, "error": measurement_error, "call_hash": record["call_hash"]})
                    previous_output = response.text if response.text is not None else "(no text returned)"
                    repair_feedback = f"Validation error: {measurement_error}\nPrevious output:\n{previous_output}"

            status = "complete" if not interaction_failed and measurement is not None else "failed"
            if status == "complete":
                cells_completed += 1
            else:
                cells_failed += 1
            row_without_hash = {
                "schema_version": 1,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.monotonic() - cell_started,
                "study_id": study_id,
                "study_hash": stable_hash(study),
                "cell_id": cell["cell_id"],
                "cell_hash": cell["cell_hash"],
                "indices": cell["indices"],
                "treatments": cell["treatments"],
                "public_instructions": materialized["public_instructions"],
                "status": status,
                "transcript": transcript,
                "measurement": measurement,
                "measurement_attempts": measurement_attempts,
                "measurement_error": measurement_error,
                "derived": derive_cell_summary(study, cell, measurement),
                "failures": failures,
            }
            _append_jsonl(rows_path, {**row_without_hash, "record_hash": stable_hash(row_without_hash)})
            done.add(cell["cell_id"])
        if budget_exceeded:
            break

    status = "budget_exceeded" if budget_exceeded else "complete"
    invocation = {
        "started_at": invocation_started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - run_started,
        "status": status,
        "cells_started": cells_started,
        "cells_completed": cells_completed,
        "cells_failed": cells_failed,
        "cells_skipped": cells_skipped,
    }
    _append_jsonl(output_dir / "run_events.jsonl", invocation)
    manifest = {
        "schema_version": 1,
        "run_type": "frontier-planned-local-subjects",
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_hash": stable_hash(bundle_snapshot),
        "config_hash": stable_hash(config_snapshot),
        "plan_hash": stable_hash(selected_cells),
        "planned_cells": sum(len(cells) for cells in selected_cells.values()),
        "wall_clock_budget_seconds": wall_clock_budget_seconds,
        "latest_invocation": invocation,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return invocation


def _summarize_values(values: list[Any], outcome_type: str) -> dict[str, Any]:
    non_null = [value for value in values if value is not None]
    if outcome_type == "category":
        counts: dict[str, int] = {}
        for value in non_null:
            counts[str(value)] = counts.get(str(value), 0) + 1
        return {
            "type": "category",
            "n": len(values),
            "counts": counts,
            "proportions": {key: count / len(non_null) for key, count in counts.items()} if non_null else {},
        }
    numeric = [float(value) for value in non_null]
    summary: dict[str, Any] = {
        "type": outcome_type,
        "n": len(values),
        "non_null": len(numeric),
        "mean": sum(numeric) / len(numeric) if numeric else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }
    return summary


def analyze_frontier_study(study: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce descriptive one-draw outcome and factor-level marginal summaries."""
    valid = [row for row in rows if row.get("status") == "complete" and row.get("measurement") is not None]
    outcome_types = {outcome["name"]: outcome["type"] for outcome in study["outcomes"]}
    if study["study_id"] == "frontier-art-auction":
        outcome_types["allocative_efficiency"] = "bounded_number"
        outcome_types["efficient_allocation"] = "binary"

    def row_value(row: dict[str, Any], name: str) -> Any:
        if name in row["measurement"]["outcomes"]:
            return row["measurement"]["outcomes"][name]
        value = row["derived"][name]
        if isinstance(value, bool):
            return int(value)
        return value

    overall = {name: _summarize_values([row_value(row, name) for row in valid], outcome_type) for name, outcome_type in outcome_types.items()}
    marginals: dict[str, list[dict[str, Any]]] = {}
    for factor_index, factor in enumerate(study["factors"]):
        levels: list[dict[str, Any]] = []
        for level_index, level in enumerate(factor["levels"]):
            subset = [row for row in valid if row["indices"][factor_index] == level_index]
            levels.append(
                {
                    "level": level["value"],
                    "level_index": level_index,
                    "n": len(subset),
                    "outcomes": {
                        name: _summarize_values([row_value(row, name) for row in subset], outcome_type)
                        for name, outcome_type in outcome_types.items()
                    },
                }
            )
        marginals[factor["name"]] = levels

    measured = [row["measurement"]["outcomes"] for row in valid]
    derived = [row["derived"] for row in valid]
    study_id = study["study_id"]
    if study_id == "frontier-mug-bargaining":
        deal_values = [item["deal"] for item in measured]
        prices = [item["price"] for item in measured if item["price"] is not None]
        study_summary = {
            "deal_rate": sum(deal_values) / len(deal_values) if deal_values else None,
            "mean_price_on_recorded_deals": sum(prices) / len(prices) if prices else None,
        }
    elif study_id == "frontier-bail-setting":
        values = [float(item["bail_amount"]) for item in measured]
        study_summary = {"mean_bail_amount": sum(values) / len(values) if values else None}
    elif study_id == "frontier-lawyer-interview":
        hires = [item["hire"] for item in measured]
        ratings = [float(item["rating"]) for item in measured]
        study_summary = {
            "hire_rate": sum(hires) / len(hires) if hires else None,
            "mean_rating": sum(ratings) / len(ratings) if ratings else None,
        }
    elif study_id == "frontier-art-auction":
        prices = [float(item["final_price"]) for item in measured]
        winner_counts: dict[str, int] = {}
        for item in measured:
            winner_counts[item["winner"]] = winner_counts.get(item["winner"], 0) + 1
        efficiencies = [float(item["allocative_efficiency"]) for item in derived]
        study_summary = {
            "mean_final_price": sum(prices) / len(prices) if prices else None,
            "winner_counts": winner_counts,
            "mean_allocative_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else None,
            "efficient_allocation_rate": sum(bool(item["efficient_allocation"]) for item in derived) / len(derived) if derived else None,
        }
    else:
        raise ValueError(f"unsupported study_id: {study_id}")
    return {
        "schema_version": 1,
        "study_id": study_id,
        "caveat": "Descriptive one-draw local-model evidence: one model draw per OA cell, not repeated sampling and not human behavior.",
        "counts": {"rows": len(rows), "complete": len(valid), "failed": len(rows) - len(valid)},
        "outcome_summaries": overall,
        "study_summary": study_summary,
        "main_effect_marginals": marginals,
    }
