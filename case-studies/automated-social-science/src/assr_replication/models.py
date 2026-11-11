"""Provider-neutral chat request and response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    temperature: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [message.to_dict() for message in self.messages],
            "metadata": dict(self.metadata),
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class ChatResponse:
    provider: str
    model: str
    text: str | None
    raw_response: Any
    error: str | None = None

    @classmethod
    def from_raw(
        cls,
        *,
        provider: str,
        model: str,
        raw_response: Any,
    ) -> "ChatResponse":
        text = raw_response.get("text") if isinstance(raw_response, dict) else None
        if isinstance(text, str):
            return cls(provider, model, text, raw_response)
        return cls(
            provider,
            model,
            None,
            raw_response,
            "response did not contain string field 'text'",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "text": self.text,
            "raw_response": self.raw_response,
            "error": self.error,
        }


class ChatProvider(Protocol):
    name: str

    @property
    def configuration(self) -> Mapping[str, Any]:
        """Return non-secret provider settings for run metadata."""
        ...

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Complete one provider-neutral chat request."""
        ...
