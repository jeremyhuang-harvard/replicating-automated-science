"""Append-only raw provider-call logging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .hashing import stable_hash
from .models import ChatRequest, ChatResponse


class JsonlCallLogger:
    """Persist one complete request/response record per appended JSONL line."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, request: ChatRequest, response: ChatResponse) -> dict[str, object]:
        request_data = request.to_dict()
        response_data = response.to_dict()
        prompt_payload = {
            "model": request.model,
            "messages": [message.to_dict() for message in request.messages],
            "temperature": request.temperature,
        }
        record: dict[str, object] = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": stable_hash(prompt_payload),
            "request": request_data,
            "response": response_data,
        }
        record["call_hash"] = stable_hash(
            {"prompt_hash": record["prompt_hash"], "response": response_data}
        )
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return record
