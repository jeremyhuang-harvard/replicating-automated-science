"""Tiny bargaining smoke workflow used to verify the local harness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .call_log import JsonlCallLogger
from .hashing import stable_hash
from .models import ChatMessage, ChatProvider, ChatRequest
from .providers import DeterministicMockProvider

_SCENARIO = "Two people bargaining over a mug"
_AGENTS = {
    "buyer": {"budget_usd": 75},
    "seller": {"minimum_price_usd": 50},
}
_TURNS = ("buyer", "seller", "buyer")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_bargaining_smoke(
    output_dir: Path,
    provider: ChatProvider,
    *,
    model: str,
) -> Path:
    """Run three bargaining turns and write a self-describing local artifact set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "transcript.jsonl", "raw_calls.jsonl"):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    started_at = datetime.now(timezone.utc).isoformat()
    run_config = {
        "scenario": _SCENARIO,
        "provider": provider.name,
        "provider_config": dict(provider.configuration),
        "model": model,
        "turns": list(_TURNS),
    }
    run_id = stable_hash(run_config)[:16]
    logger = JsonlCallLogger(output_dir / "raw_calls.jsonl")
    transcript: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []

    with (output_dir / "transcript.jsonl").open("a", encoding="utf-8") as transcript_file:
        for turn, agent in enumerate(_TURNS):
            private_attributes = _AGENTS[agent]
            public_history = "\n".join(
                f"{item['agent']}: {item['content']}" for item in transcript
            ) or "(none yet)"
            system_prompt = (
                f"You are the {agent} in this scenario: {_SCENARIO}. "
                f"Your private attributes are {json.dumps(private_attributes, sort_keys=True)}. "
                "Never claim access to another agent's private attributes."
            )
            user_prompt = (
                f"Public transcript:\n{public_history}\n"
                "Make the next concise bargaining statement."
            )
            request = ChatRequest(
                model=model,
                messages=(
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ),
                metadata={"run_id": run_id, "agent": agent, "turn": turn},
            )
            response = provider.complete(request)
            call_record = logger.append(request, response)
            prompt_hashes.append(str(call_record["prompt_hash"]))
            entry = {
                "turn": turn,
                "agent": agent,
                "content": response.text,
                "response_error": response.error,
                "call_hash": call_record["call_hash"],
            }
            transcript.append(entry)
            transcript_file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            transcript_file.flush()

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **run_config,
        "call_count": len(prompt_hashes),
        "transcript_count": len(transcript),
        "prompt_hashes": prompt_hashes,
        "privacy_policy": "own-private-attributes-plus-public-transcript",
        "artifacts": {
            "manifest": "manifest.json",
            "raw_calls": "raw_calls.jsonl",
            "transcript": "transcript.jsonl",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def run_mock_smoke(output_dir: Path) -> Path:
    return run_bargaining_smoke(
        output_dir,
        DeterministicMockProvider(),
        model="mock-bargainer-v1",
    )
