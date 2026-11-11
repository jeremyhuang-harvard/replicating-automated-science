"""Typed, validated records shared by every stage of a study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

SCHEMA_VERSION = 1
QUESTION_KINDS = ("rating", "text")


class SchemaError(ValueError):
    """Raised when a record fails validation."""


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class Concept:
    title: str
    description: str

    def __post_init__(self) -> None:
        _require_text("concept title", self.title)
        _require_text("concept description", self.description)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Concept:
        return cls(title=data["title"], description=data["description"])


@dataclass(frozen=True)
class Persona:
    """A synthetic respondent defined by explicit product attitudes (1-5), not demographics."""

    persona_id: str
    name: str
    segment: str
    description: str
    attitudes: dict[str, int]

    def __post_init__(self) -> None:
        for name in ("persona_id", "name", "segment", "description"):
            _require_text(f"persona {name}", getattr(self, name))
        for key, value in self.attitudes.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise SchemaError(f"attitude {key!r} must be an integer 1-5, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        return cls(**data)


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    kind: str  # "rating" (integer 1-5) or "text" (open-ended)

    def __post_init__(self) -> None:
        _require_text("question_id", self.question_id)
        _require_text("question text", self.text)
        if self.kind not in QUESTION_KINDS:
            raise SchemaError(f"question kind must be one of {QUESTION_KINDS}, got {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        return cls(**data)


@dataclass(frozen=True)
class Response:
    """One persona's answers. `error` is set (and `answers` empty) when the engine output was unusable."""

    persona_id: str
    segment: str
    answers: dict[str, int | str]
    raw_output: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Response:
        return cls(**data)


@dataclass(frozen=True)
class StudyManifest:
    study_id: str
    created_at: str  # timezone-aware ISO-8601, UTC
    engine: str
    model: str
    seed: int
    concept: Concept
    panel: list[Persona]
    questions: list[Question]
    prompts: dict[str, str] = field(default_factory=dict)  # persona_id -> exact prompt sent
    engine_config: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("study_id", self.study_id)
        try:
            stamp = datetime.fromisoformat(self.created_at)
        except (TypeError, ValueError):
            raise SchemaError(f"created_at must be an ISO-8601 timestamp, got {self.created_at!r}") from None
        if stamp.utcoffset() != timedelta(0):
            raise SchemaError("created_at must be timezone-aware UTC")
        if not self.panel:
            raise SchemaError("panel must contain at least one persona")
        if not self.questions:
            raise SchemaError("questions must not be empty")
        if len({p.persona_id for p in self.panel}) != len(self.panel):
            raise SchemaError("persona ids must be unique")
        if len({q.question_id for q in self.questions}) != len(self.questions):
            raise SchemaError("question ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudyManifest:
        return cls(
            study_id=data["study_id"],
            created_at=data["created_at"],
            engine=data["engine"],
            model=data["model"],
            seed=data["seed"],
            concept=Concept.from_dict(data["concept"]),
            panel=[Persona.from_dict(p) for p in data["panel"]],
            questions=[Question.from_dict(q) for q in data["questions"]],
            prompts=data.get("prompts", {}),
            engine_config=data.get("engine_config", {}),
            runtime=data.get("runtime", {}),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )
