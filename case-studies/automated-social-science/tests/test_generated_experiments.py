import copy
import json
from pathlib import Path

import pytest

from assr_replication.generated_experiments import (
    FinalizationValidationError,
    analyze_study,
    enumerate_factorial,
    parse_measurement,
    run_study,
    validate_finalized_design,
)
from assr_replication.models import ChatResponse


def initial_design() -> dict[str, object]:
    return {
        "schema_version": "scientist-design-v1",
        "scenario": "test scenario",
        "agents": [
            {"name": "A", "role": "speaker", "goal": "speak", "constraints": ["be concise"]},
            {"name": "B", "role": "listener", "goal": "listen", "constraints": ["be concise"]},
        ],
        "outcome": {
            "name": "score",
            "operationalization": "numeric score",
            "measurement_question": "What is the score?",
            "respondent": "A and B",
        },
        "causes": [
            {"name": "c1", "hypothesis": "c1 raises score", "predicted_direction": "increase", "treatment_values": ["l1", "m1", "h1"]},
            {"name": "c2", "hypothesis": "c2 lowers score", "predicted_direction": "decrease", "treatment_values": ["l2", "m2", "h2"]},
            {"name": "c3", "hypothesis": "c3 raises score", "predicted_direction": "increase", "treatment_values": ["l3", "m3", "h3"]},
        ],
        "information_scope": {"public": ["base public"], "private": [{"agent": "A", "information": ["base private A"]}]},
        "interaction": {"order": ["A", "B"], "max_turns": 3, "stopping_rule": "three turns"},
    }


def finalized_design() -> dict[str, object]:
    original = initial_design()
    return {
        "schema_version": "execution-design-v1",
        "source_design": copy.deepcopy(original),
        "execution": {
            "respondent": "oracle",
            "turn_order": ["A", "B"],
            "treatment_mappings": [
                {"cause": "c1", "scope": "public", "agent": None, "values": [{"value": v, "instruction": f"Public c1 is {v}."} for v in ["l1", "m1", "h1"]]},
                {"cause": "c2", "scope": "private", "agent": "A", "values": [{"value": v, "instruction": f"Private c2 is {v}."} for v in ["l2", "m2", "h2"]]},
                {"cause": "c3", "scope": "private", "agent": "B", "values": [{"value": v, "instruction": f"Private c3 is {v}."} for v in ["l3", "m3", "h3"]]},
            ],
        },
    }


def test_finalization_preserves_substantive_design_exactly() -> None:
    original = initial_design()
    final = finalized_design()
    assert validate_finalized_design(original, final) == final

    changed = copy.deepcopy(final)
    changed["source_design"]["causes"][0]["treatment_values"][2] = "different"
    with pytest.raises(FinalizationValidationError, match="source_design"):
        validate_finalized_design(original, changed)


def test_finalizer_accepts_qwen_execution_only_and_wraps_original(tmp_path: Path) -> None:
    from assr_replication.generated_experiments import finalize_design

    class Provider:
        name = "scripted"
        configuration = {}

        def complete(self, request):
            execution = finalized_design()["execution"]
            return ChatResponse(self.name, request.model, json.dumps(execution), {"ok": True})

    result = finalize_design(initial_design(), provider=Provider(), model="local", output_dir=tmp_path, seed=1)
    assert result["source_design"] == initial_design()
    assert result["execution"] == finalized_design()["execution"]
    diff = json.loads((tmp_path / "diff.json").read_text())
    assert diff["substantive_changes"] == []


def test_factorial_is_complete_unique_and_ordered() -> None:
    cells = enumerate_factorial(finalized_design())
    assert len(cells) == 27
    assert len({cell["cell_id"] for cell in cells}) == 27
    assert cells[0]["indices"] == [0, 0, 0]
    assert cells[-1]["indices"] == [2, 2, 2]


def test_private_treatments_are_isolated_and_resume_is_idempotent(tmp_path: Path) -> None:
    class Provider:
        name = "scripted"
        configuration = {"thinking_enabled": False}

        def __init__(self) -> None:
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            if request.metadata["task"] == "measurement":
                return ChatResponse(self.name, request.model, '{"value": 5, "evidence": "stated score"}', {"ok": True})
            return ChatResponse(self.name, request.model, f"turn {request.metadata['turn']}", {"ok": True})

    provider = Provider()
    cells = enumerate_factorial(finalized_design())[:1]
    run_study(finalized_design(), cells, tmp_path, provider=provider, model="local", base_seed=100)
    first_count = len(provider.requests)
    run_study(finalized_design(), cells, tmp_path, provider=provider, model="local", base_seed=100)
    assert len(provider.requests) == first_count

    subject_requests = [r for r in provider.requests if r.metadata["task"] == "subject-turn"]
    a_prompt = next(r.messages[0].content for r in subject_requests if r.metadata["agent"] == "A")
    b_prompt = next(r.messages[0].content for r in subject_requests if r.metadata["agent"] == "B")
    assert "Private c2 is l2." in a_prompt and "Private c3 is l3." not in a_prompt
    assert "Private c3 is l3." in b_prompt and "Private c2 is l2." not in b_prompt


def test_raw_failures_are_retained_and_measurement_retries(tmp_path: Path) -> None:
    class Provider:
        name = "flaky"
        configuration = {}

        def __init__(self) -> None:
            self.measurements = 0

        def complete(self, request):
            if request.metadata["task"] == "measurement":
                self.measurements += 1
                if self.measurements == 1:
                    return ChatResponse(self.name, request.model, "not json", {"raw": "not json"})
                return ChatResponse(self.name, request.model, '{"value": 7.5, "evidence": ["line 1"]}', {"raw": "ok"})
            if request.metadata["turn"] == 0 and request.metadata["attempt"] == 1:
                return ChatResponse(self.name, request.model, None, {"partial": "x"}, "boom")
            return ChatResponse(self.name, request.model, "ok", {"raw": "ok"})

    provider = Provider()
    run_study(finalized_design(), enumerate_factorial(finalized_design())[:1], tmp_path, provider=provider, model="local", base_seed=100)
    raw = [json.loads(line) for line in (tmp_path / "raw_calls.jsonl").read_text().splitlines()]
    assert any(record["response"]["error"] == "boom" for record in raw)
    assert any(record["response"]["text"] == "not json" for record in raw)
    result = json.loads((tmp_path / "cells.jsonl").read_text().splitlines()[0])
    assert result["measurement"] == {"value": 7.5, "evidence": ["line 1"]}
    assert result["measurement_attempts"] == 2


def test_measurement_parser_is_strict() -> None:
    assert parse_measurement('{"value": 4, "evidence": "quote"}') == {"value": 4.0, "evidence": "quote"}
    with pytest.raises(ValueError):
        parse_measurement('{"value": "4", "evidence": "quote"}')
    with pytest.raises(ValueError):
        parse_measurement('{"value": 4, "evidence": "quote", "extra": true}')


def test_simple_analysis_reports_contrasts_and_additive_ols() -> None:
    cells = enumerate_factorial(finalized_design())
    rows = []
    for cell in cells:
        i, j, k = cell["indices"]
        rows.append({"cell_id": cell["cell_id"], "status": "complete", "indices": [i, j, k], "measurement": {"value": 10 + 2*i - 3*j + 4*k, "evidence": "synthetic"}})
    result = analyze_study(finalized_design(), rows)
    assert result["counts"] == {"cells": 27, "complete": 27, "valid_measurements": 27}
    assert result["contrasts"] == {"c1": 4.0, "c2": -6.0, "c3": 8.0}
    assert result["ols"]["coefficients"] == pytest.approx({"intercept": 10.0, "c1": 2.0, "c2": -3.0, "c3": 4.0})
    assert result["direction_agreement"] == {"c1": True, "c2": True, "c3": True}
