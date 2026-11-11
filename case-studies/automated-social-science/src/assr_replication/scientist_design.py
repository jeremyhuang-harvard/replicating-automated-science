"""Versioned scientist-design validation and hashing."""

from __future__ import annotations

from typing import Any

from .hashing import stable_hash

SCHEMA_VERSION = "scientist-design-v1"
PREDICTED_DIRECTIONS = {"increase", "decrease", "non_monotonic", "uncertain"}


class DesignValidationError(ValueError):
    """Raised when a scientist design violates the frozen v1 contract."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DesignValidationError(f"{path} must be an object")
    return value


def _list(value: Any, path: str, minimum: int, maximum: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise DesignValidationError(f"{path} must be an array")
    if len(value) < minimum or (maximum is not None and len(value) > maximum):
        bounds = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
        raise DesignValidationError(f"{path} must contain {bounds} items")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DesignValidationError(f"{path} must be a non-empty string")
    return value


def _keys(value: dict[str, Any], path: str, expected: set[str]) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        raise DesignValidationError(f"{path} missing required field: {sorted(missing)[0]}")
    if extra:
        raise DesignValidationError(f"{path} has unsupported field: {sorted(extra)[0]}")


def validate_design(value: Any, *, expected_scenario: str | None = None) -> dict[str, Any]:
    design = _mapping(value, "design")
    _keys(
        design,
        "design",
        {"schema_version", "scenario", "agents", "outcome", "causes", "information_scope", "interaction"},
    )
    if design["schema_version"] != SCHEMA_VERSION:
        raise DesignValidationError(f"schema_version must equal {SCHEMA_VERSION}")
    scenario = _text(design["scenario"], "scenario")
    if expected_scenario is not None and scenario != expected_scenario:
        raise DesignValidationError("scenario must exactly match the supplied scenario")

    agents = _list(design["agents"], "agents", 1, 4)
    agent_names: list[str] = []
    for index, raw_agent in enumerate(agents):
        agent = _mapping(raw_agent, f"agents[{index}]")
        _keys(agent, f"agents[{index}]", {"name", "role", "goal", "constraints"})
        name = _text(agent["name"], f"agents[{index}].name")
        agent_names.append(name)
        _text(agent["role"], f"agents[{index}].role")
        _text(agent["goal"], f"agents[{index}].goal")
        constraints = _list(agent["constraints"], f"agents[{index}].constraints", 1)
        for item_index, item in enumerate(constraints):
            _text(item, f"agents[{index}].constraints[{item_index}]")
    if len(set(agent_names)) != len(agent_names):
        raise DesignValidationError("agents must have unique names")

    outcome = _mapping(design["outcome"], "outcome")
    outcome_fields = {"name", "operationalization", "measurement_question", "respondent"}
    _keys(outcome, "outcome", outcome_fields)
    for field in outcome_fields:
        _text(outcome[field], f"outcome.{field}")

    causes = _list(design["causes"], "causes", 1, 3)
    cause_names: list[str] = []
    for index, raw_cause in enumerate(causes):
        cause = _mapping(raw_cause, f"causes[{index}]")
        fields = {"name", "hypothesis", "predicted_direction", "treatment_values"}
        _keys(cause, f"causes[{index}]", fields)
        cause_names.append(_text(cause["name"], f"causes[{index}].name"))
        _text(cause["hypothesis"], f"causes[{index}].hypothesis")
        if cause["predicted_direction"] not in PREDICTED_DIRECTIONS:
            raise DesignValidationError(f"causes[{index}].predicted_direction is invalid")
        treatments = _list(cause["treatment_values"], f"causes[{index}].treatment_values", 2, 3)
        if any(isinstance(item, (dict, list)) or item is None for item in treatments):
            raise DesignValidationError(f"causes[{index}].treatment_values must be concrete scalar values")
        if len({stable_hash(item) for item in treatments}) != len(treatments):
            raise DesignValidationError(f"causes[{index}].treatment_values must be distinct")
    if len(set(cause_names)) != len(cause_names):
        raise DesignValidationError("causes must have unique names")

    scope = _mapping(design["information_scope"], "information_scope")
    _keys(scope, "information_scope", {"public", "private"})
    for index, item in enumerate(_list(scope["public"], "information_scope.public", 0)):
        _text(item, f"information_scope.public[{index}]")
    for index, raw_private in enumerate(_list(scope["private"], "information_scope.private", 0)):
        private = _mapping(raw_private, f"information_scope.private[{index}]")
        _keys(private, f"information_scope.private[{index}]", {"agent", "information"})
        agent = _text(private["agent"], f"information_scope.private[{index}].agent")
        if agent not in agent_names:
            raise DesignValidationError(f"information_scope.private[{index}].agent must name an agent")
        for item_index, item in enumerate(_list(private["information"], f"information_scope.private[{index}].information", 1)):
            _text(item, f"information_scope.private[{index}].information[{item_index}]")

    interaction = _mapping(design["interaction"], "interaction")
    _keys(interaction, "interaction", {"order", "max_turns", "stopping_rule"})
    order = _list(interaction["order"], "interaction.order", 1)
    for index, item in enumerate(order):
        if _text(item, f"interaction.order[{index}]") not in agent_names:
            raise DesignValidationError(f"interaction.order[{index}] must name an agent")
    max_turns = interaction["max_turns"]
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or not 1 <= max_turns <= 8:
        raise DesignValidationError("interaction.max_turns must be an integer from 1 to 8")
    _text(interaction["stopping_rule"], "interaction.stopping_rule")
    return design


def design_hash(design: Any) -> str:
    """Hash a validated design using canonical JSON serialization."""
    return stable_hash(validate_design(design))
