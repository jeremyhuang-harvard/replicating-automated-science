import json
from pathlib import Path

from assr_replication.models import ChatRequest, ChatResponse
from assr_replication.scientist_runner import generate_scientist_designs


SCENARIO = "two people bargaining over a mug"


def design(outcome_name: str) -> dict[str, object]:
    return {
        "schema_version": "scientist-design-v1",
        "scenario": SCENARIO,
        "agents": [
            {"name": "buyer", "role": "buyer", "goal": "buy", "constraints": ["budget"]},
            {"name": "seller", "role": "seller", "goal": "sell", "constraints": ["ownership"]},
        ],
        "outcome": {
            "name": outcome_name,
            "operationalization": "recorded agreement result",
            "measurement_question": "What was agreed?",
            "respondent": "recorder",
        },
        "causes": [
            {
                "name": "time pressure",
                "hypothesis": "More pressure changes agreement.",
                "predicted_direction": "increase",
                "treatment_values": ["low", "high"],
            }
        ],
        "information_scope": {"public": ["a mug is traded"], "private": []},
        "interaction": {
            "order": ["buyer", "seller"],
            "max_turns": 4,
            "stopping_rule": "agreement or cap",
        },
    }


class ScriptedProvider:
    name = "scripted"

    def __init__(self, seed: int, replies: list[str], seen_seeds: list[int]) -> None:
        self.seed = seed
        self.replies = replies
        self.seen_seeds = seen_seeds

    @property
    def configuration(self) -> dict[str, object]:
        return {"seed": self.seed, "temperature": 0.3, "thinking": False}

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.seen_seeds.append(self.seed)
        text = self.replies.pop(0)
        return ChatResponse("scripted", request.model, text, {"text": text})


def test_first_valid_design_is_selected_and_failed_attempt_is_preserved(tmp_path: Path) -> None:
    replies = ["not json", json.dumps(design("first valid")), json.dumps(design("later valid"))]
    seen_seeds: list[int] = []

    def provider_factory(seed: int) -> ScriptedProvider:
        return ScriptedProvider(seed, replies, seen_seeds)

    output_dir = tmp_path / "scientist-run"
    result = generate_scientist_designs(
        output_dir,
        provider_factory=provider_factory,
        model="local/qwen",
        scenarios=(SCENARIO,),
        seeds=(101, 102, 103),
    )

    assert result == output_dir
    assert seen_seeds == [101, 102]
    raw_records = [json.loads(line) for line in (output_dir / "raw_calls.jsonl").read_text().splitlines()]
    assert len(raw_records) == 2
    assert raw_records[0]["response"]["text"] == "not json"
    assert [record["request"]["metadata"]["seed"] for record in raw_records] == [101, 102]

    report = json.loads((output_dir / "validation_report.json").read_text())
    assert report["summary"] == {"scenarios": 1, "valid_designs": 1, "total_attempts": 2}
    assert report["scenarios"][0]["attempts"][0]["status"] == "malformed_json"
    assert report["scenarios"][0]["attempts"][1]["status"] == "valid"

    parsed_path = output_dir / report["scenarios"][0]["design_file"]
    parsed = json.loads(parsed_path.read_text())
    assert parsed["outcome"]["name"] == "first valid"
    assert report["scenarios"][0]["design_hash"]


def test_schema_invalid_output_retries_but_provider_error_does_not(tmp_path: Path) -> None:
    invalid = design("invalid")
    invalid["interaction"]["max_turns"] = 9

    class ErrorAfterInvalid(ScriptedProvider):
        def complete(self, request: ChatRequest) -> ChatResponse:
            self.seen_seeds.append(self.seed)
            if len(self.seen_seeds) == 1:
                text = json.dumps(invalid)
                return ChatResponse("scripted", request.model, text, {"text": text})
            return ChatResponse("scripted", request.model, None, {"stderr": "failed"}, "provider failed")

    seen_seeds: list[int] = []

    def provider_factory(seed: int) -> ErrorAfterInvalid:
        return ErrorAfterInvalid(seed, [], seen_seeds)

    output_dir = tmp_path / "failed-run"
    generate_scientist_designs(
        output_dir,
        provider_factory=provider_factory,
        model="local/qwen",
        scenarios=(SCENARIO,),
        seeds=(1, 2, 3),
    )

    report = json.loads((output_dir / "validation_report.json").read_text())
    assert seen_seeds == [1, 2]
    assert [attempt["status"] for attempt in report["scenarios"][0]["attempts"]] == [
        "schema_invalid",
        "provider_error",
    ]
    assert report["summary"]["valid_designs"] == 0
