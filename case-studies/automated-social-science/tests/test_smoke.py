import json
from pathlib import Path

from assr_replication.smoke import run_mock_smoke


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_mock_smoke_writes_complete_manifest_transcript_and_raw_calls(tmp_path: Path) -> None:
    run_dir = run_mock_smoke(tmp_path / "smoke-run")

    assert run_dir == tmp_path / "smoke-run"
    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json",
        "transcript.jsonl",
        "raw_calls.jsonl",
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    transcript = _read_jsonl(run_dir / "transcript.jsonl")
    raw_calls = _read_jsonl(run_dir / "raw_calls.jsonl")

    assert manifest["status"] == "complete"
    assert manifest["provider"] == "deterministic-mock"
    assert manifest["call_count"] == manifest["transcript_count"] == 3
    assert len(manifest["prompt_hashes"]) == len(raw_calls) == len(transcript) == 3
    assert transcript[-1]["content"] == "Agreed. I will pay $60 for the mug."
    assert manifest["artifacts"] == {
        "manifest": "manifest.json",
        "raw_calls": "raw_calls.jsonl",
        "transcript": "transcript.jsonl",
    }


def test_private_attributes_are_only_visible_to_the_owning_agent(tmp_path: Path) -> None:
    run_dir = run_mock_smoke(tmp_path / "privacy-run")
    raw_calls = _read_jsonl(run_dir / "raw_calls.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    for call in raw_calls:
        request = call["request"]
        agent = request["metadata"]["agent"]
        rendered = json.dumps(request["messages"], sort_keys=True)
        if agent == "buyer":
            assert "budget_usd" in rendered
            assert "minimum_price_usd" not in rendered
        else:
            assert "minimum_price_usd" in rendered
            assert "budget_usd" not in rendered

    transcript_text = (run_dir / "transcript.jsonl").read_text(encoding="utf-8")
    assert "budget_usd" not in transcript_text
    assert "minimum_price_usd" not in transcript_text
    assert manifest["privacy_policy"] == "own-private-attributes-plus-public-transcript"
