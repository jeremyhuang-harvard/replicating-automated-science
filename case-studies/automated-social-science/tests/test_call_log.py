import json
from pathlib import Path

from assr_replication.call_log import JsonlCallLogger
from assr_replication.models import ChatMessage, ChatRequest, ChatResponse


def _request(turn: int) -> ChatRequest:
    return ChatRequest(
        model="mock-bargainer-v1",
        messages=(ChatMessage(role="user", content=f"turn {turn}"),),
        metadata={"turn": turn},
    )


def _response(turn: int) -> ChatResponse:
    return ChatResponse(
        provider="deterministic-mock",
        model="mock-bargainer-v1",
        text=f"reply {turn}",
        raw_response={"reply": turn},
    )


def test_jsonl_logger_appends_without_rewriting_existing_records(tmp_path: Path) -> None:
    path = tmp_path / "raw_calls.jsonl"
    logger = JsonlCallLogger(path)

    logger.append(_request(1), _response(1))
    first_bytes = path.read_bytes()
    logger.append(_request(2), _response(2))

    final_bytes = path.read_bytes()
    records = [json.loads(line) for line in final_bytes.splitlines()]
    assert final_bytes.startswith(first_bytes)
    assert len(records) == 2
    assert records[0]["request"]["metadata"] == {"turn": 1}
    assert records[1]["request"]["metadata"] == {"turn": 2}
    assert records[0]["prompt_hash"] != records[1]["prompt_hash"]
