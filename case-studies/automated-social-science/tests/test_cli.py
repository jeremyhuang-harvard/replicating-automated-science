import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from assr_replication.cli import main
from assr_replication.scientist_runner import CANONICAL_SCENARIOS


def test_cli_smoke_command_writes_artifacts(tmp_path: Path, capsys: object) -> None:
    output_dir = tmp_path / "cli-smoke"

    exit_code = main(["smoke", "--output-dir", str(output_dir)])

    assert exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "deterministic-mock"
    assert manifest["status"] == "complete"


def test_cli_local_smoke_uses_qwen_defaults_with_injected_runner(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="local bargaining reply\n", stderr="")

    output_dir = tmp_path / "local-smoke"
    exit_code = main(
        ["local-smoke", "--output-dir", str(output_dir)],
        mlx_process_runner=fake_runner,
    )

    assert exit_code == 0
    assert len(seen) == 3
    assert all("mlx-community/Qwen3-8B-4bit" in argv for argv in seen)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "mlx-cli"
    assert manifest["model"] == "mlx-community/Qwen3-8B-4bit"
    assert manifest["provider_config"] == {
        "executable": "mlx_lm.generate",
        "max_tokens": 128,
        "temperature": 0.0,
        "seed": 0,
        "chat_template_config": {"enable_thinking": False},
    }
    raw_calls = [
        json.loads(line)
        for line in (output_dir / "raw_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert raw_calls[0]["response"]["raw_response"]["returncode"] == 0


def test_cli_generate_designs_runs_all_canonical_scenarios_with_distinct_seeds(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        seen.append(args)
        prompt = args[args.index("--prompt") + 1]
        scenario = next(item for item in CANONICAL_SCENARIOS if item in prompt)
        payload = {
            "schema_version": "scientist-design-v1",
            "scenario": scenario,
            "agents": [
                {"name": "actor", "role": "participant", "goal": "complete the interaction", "constraints": ["follow the protocol"]}
            ],
            "outcome": {
                "name": "completion",
                "operationalization": "whether the interaction completes",
                "measurement_question": "Did the interaction complete?",
                "respondent": "recorder",
            },
            "causes": [
                {
                    "name": "time available",
                    "hypothesis": "More time increases completion.",
                    "predicted_direction": "increase",
                    "treatment_values": ["short", "long"],
                }
            ],
            "information_scope": {"public": ["the scenario"], "private": []},
            "interaction": {"order": ["actor"], "max_turns": 2, "stopping_rule": "completion or cap"},
        }
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

    output_dir = tmp_path / "design-run"
    exit_code = main(
        ["generate-designs", "--output-dir", str(output_dir), "--base-seed", "700"],
        mlx_process_runner=fake_runner,
    )

    assert exit_code == 0
    assert len(seen) == 4
    seeds = [args[args.index("--seed") + 1] for args in seen]
    assert seeds == ["700", "701", "702", "703"]
    assert all(args[args.index("--temp") + 1] == "0.3" for args in seen)
    assert all('{"enable_thinking":false}' in args for args in seen)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["counts"] == {"scenarios": 4, "valid_designs": 4, "total_attempts": 4}
    assert len(list((output_dir / "designs").glob("*.json"))) == 4
