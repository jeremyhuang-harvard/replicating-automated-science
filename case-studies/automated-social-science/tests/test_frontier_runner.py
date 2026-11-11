import json
from pathlib import Path

import pytest

from assr_replication.frontier_runner import (
    analyze_frontier_study,
    derive_cell_summary,
    generate_oa_cells,
    materialize_cell,
    parse_typed_measurement,
    render_agent_messages,
    run_experiment,
    validate_bundle,
)
from assr_replication.models import ChatResponse
from assr_replication.frontier_cli import build_parser as build_frontier_parser, main as frontier_main


DESIGN_PATH = Path("designs/frontier-planned-v1/designs.json")
EXPECTED_SCENARIOS = [
    "two people bargaining over a mug",
    "a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud",
    "A person is interviewing for a job as a lawyer",
    "3 bidders participating in an auction for a piece of art starting at fifty dollars",
]


def load_bundle() -> dict:
    return json.loads(DESIGN_PATH.read_text(encoding="utf-8"))


def test_validate_bundle_accepts_fixed_four_scenarios_and_rejects_drift() -> None:
    bundle = load_bundle()
    validated = validate_bundle(bundle)
    assert [study["scenario"] for study in validated["studies"]] == EXPECTED_SCENARIOS
    assert all(len(study["factors"]) == 5 for study in validated["studies"])
    assert all(len(factor["levels"]) == 3 for study in validated["studies"] for factor in study["factors"])

    bundle["studies"][0]["scenario"] = "changed scenario"
    with pytest.raises(ValueError, match="fixed scenario"):
        validate_bundle(bundle)


def test_finite_field_oa_has_balanced_levels_pairs_and_108_total_cells() -> None:
    bundle = validate_bundle(load_bundle())
    all_cells = []
    for study in bundle["studies"]:
        cells = generate_oa_cells(study)
        assert len(cells) == len({cell["cell_id"] for cell in cells}) == 27
        for factor_index in range(5):
            assert [cell["indices"][factor_index] for cell in cells].count(0) == 9
            assert [cell["indices"][factor_index] for cell in cells].count(1) == 9
            assert [cell["indices"][factor_index] for cell in cells].count(2) == 9
        for left in range(5):
            for right in range(left + 1, 5):
                pairs = [(cell["indices"][left], cell["indices"][right]) for cell in cells]
                assert all(pairs.count((a, b)) == 3 for a in range(3) for b in range(3))
        all_cells.extend(cells)
    assert len(all_cells) == 108


def test_public_templates_expand_ranges_and_exact_values() -> None:
    auction = validate_bundle(load_bundle())["studies"][3]
    cells = generate_oa_cells(auction)
    ranges = materialize_cell(auction, next(cell for cell in cells if cell["indices"][:3] == [1, 0, 2] and cell["indices"][4] == 1))
    assert ranges["public_instructions"][-1] == (
        "The auctioneer publicly announces valuation ranges: Bidder A is in $175–$225, "
        "Bidder B is in $75–$125, and Bidder C is in $275–$325. Exact valuations remain private."
    )
    exact = materialize_cell(auction, next(cell for cell in cells if cell["indices"][:3] == [0, 1, 2] and cell["indices"][4] == 2))
    assert exact["public_instructions"][-1] == (
        "The auctioneer publicly announces the exact valuations: Bidder A $100, Bidder B $200, Bidder C $300."
    )


def test_agent_prompt_contains_only_public_history_and_own_private_information() -> None:
    mug = validate_bundle(load_bundle())["studies"][0]
    cell = materialize_cell(mug, generate_oa_cells(mug)[0])
    messages = render_agent_messages(mug, cell, "Buyer", [{"agent": "Seller", "text": "Public offer: $5"}], turn=2)
    prompt = "\n".join(message.content for message in messages)
    assert mug["scenario"] in prompt
    assert "Your maximum total willingness to pay is $6." in prompt
    assert "Public offer: $5" in prompt
    assert "You will not voluntarily sell for less than $4." not in prompt
    assert mug["factors"][0]["prediction"] not in prompt
    assert mug["outcomes"][0]["question"] not in prompt
    assert json.dumps(mug, sort_keys=True) not in prompt


def test_typed_multi_outcome_measurement_is_strict() -> None:
    outcomes = [
        {"name": "binary", "type": "binary", "question": "binary?"},
        {"name": "number", "type": "number", "question": "number?"},
        {"name": "nullable", "type": "nullable_number", "question": "nullable?"},
        {"name": "bounded", "type": "bounded_number", "minimum": 0, "maximum": 10, "question": "bounded?"},
        {"name": "category", "type": "category", "allowed": ["a", "b"], "question": "category?"},
    ]
    text = json.dumps(
        {
            "outcomes": {"binary": 1, "number": 2.5, "nullable": None, "bounded": 10, "category": "b"},
            "evidence": {"binary": "yes", "number": ["2.5"], "nullable": "none", "bounded": "10", "category": "b"},
        }
    )
    assert parse_typed_measurement(text, outcomes)["outcomes"] == {
        "binary": 1,
        "number": 2.5,
        "nullable": None,
        "bounded": 10.0,
        "category": "b",
    }
    invalid_payloads = [
        text.replace('"binary": 1', '"binary": true'),
        text.replace('"bounded": 10', '"bounded": 11'),
        text.replace('"category": "b"', '"category": "c"', 1),
        text[:-1] + ', "extra": true}',
    ]
    for invalid in invalid_payloads:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_typed_measurement(invalid, outcomes)


def test_typed_measurement_accepts_global_evidence_excerpt_list() -> None:
    outcomes = [{"name": "score", "type": "number", "question": "score?"}]
    parsed = parse_typed_measurement(
        '{"outcomes":{"score":4},"evidence":["Agent: explicit score 4"]}',
        outcomes,
    )
    assert parsed == {"outcomes": {"score": 4.0}, "evidence": ["Agent: explicit score 4"]}


def test_run_is_append_only_resumable_and_preserves_malformed_attempts(tmp_path: Path) -> None:
    bundle = validate_bundle(load_bundle())
    mug = bundle["studies"][0]
    selected = {mug["study_id"]: generate_oa_cells(mug)[:1]}

    class FlakyProvider:
        name = "flaky"
        configuration = {"local": True}

        def __init__(self) -> None:
            self.requests = []
            self.subject_attempts = 0
            self.measurement_attempts = 0

        def complete(self, request):
            self.requests.append(request)
            if request.metadata["task"] == "subject-turn":
                self.subject_attempts += 1
                if self.subject_attempts == 1:
                    return ChatResponse(self.name, request.model, None, {"partial": "bad"}, "temporary failure")
                return ChatResponse(self.name, request.model, f"public turn {request.metadata['turn']}", {"ok": True})
            self.measurement_attempts += 1
            if self.measurement_attempts == 1:
                return ChatResponse(self.name, request.model, "not json", {"raw": "not json"})
            return ChatResponse(
                self.name,
                request.model,
                '{"outcomes":{"deal":1,"price":6},"evidence":{"deal":"agreement","price":"$6"}}',
                {"ok": True},
            )

    provider = FlakyProvider()
    result = run_experiment(bundle, selected, tmp_path, provider=provider, model="local/model", base_seed=100)
    assert result["status"] == "complete"
    assert len(provider.requests) == mug["interaction"]["max_turns"] + 3
    row_path = tmp_path / "studies" / mug["study_id"] / "cells.jsonl"
    row = json.loads(row_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["measurement"]["outcomes"] == {"deal": 1, "price": 6.0}
    assert row["derived"] == {"deal": True, "price": 6.0}
    assert len(row["failures"]) == 2
    assert row["runtime_seconds"] >= 0
    assert row["recorded_at"].endswith("+00:00")
    raw_path = tmp_path / "studies" / mug["study_id"] / "raw_calls.jsonl"
    raw_before = raw_path.read_text(encoding="utf-8")
    assert '"text": "not json"' in raw_before and '"error": "temporary failure"' in raw_before
    assert (tmp_path / "manifest.json").exists()
    snapshot = json.loads((tmp_path / "prompt_config_snapshot.json").read_text(encoding="utf-8"))
    assert "subject_system_template" in snapshot["prompt_templates"]
    assert "measurement_system_prompt" in snapshot["prompt_templates"]
    assert (tmp_path / "design_bundle.json").exists()

    measurement_requests = [request for request in provider.requests if request.metadata["task"] == "measurement"]
    assert "not json" in measurement_requests[1].messages[-1].content
    assert "rejected" in measurement_requests[1].messages[-1].content.lower()

    run_experiment(bundle, selected, tmp_path, provider=provider, model="local/model", base_seed=100)
    assert raw_path.read_text(encoding="utf-8") == raw_before
    assert len(row_path.read_text(encoding="utf-8").splitlines()) == 1


def test_wall_clock_budget_stops_before_starting_a_new_cell(tmp_path: Path) -> None:
    bundle = validate_bundle(load_bundle())
    mug = bundle["studies"][0]
    selected = {mug["study_id"]: generate_oa_cells(mug)[:2]}

    class UnusedProvider:
        name = "unused"
        configuration = {}

        def complete(self, request):
            raise AssertionError("budget check must happen before provider call")

    result = run_experiment(
        bundle,
        selected,
        tmp_path,
        provider=UnusedProvider(),
        model="local/model",
        base_seed=1,
        wall_clock_budget_seconds=0,
    )
    assert result["status"] == "budget_exceeded"
    assert result["cells_started"] == 0


def test_analysis_handles_categorical_treatments_and_auction_efficiency() -> None:
    auction = validate_bundle(load_bundle())["studies"][3]
    cells = generate_oa_cells(auction)
    example = next(cell for cell in cells if cell["treatments"]["Bidder A valuation"] == 100 and cell["treatments"]["Bidder B valuation"] == 200 and cell["treatments"]["Bidder C valuation"] == 300)
    derived = derive_cell_summary(
        auction,
        example,
        {"outcomes": {"final_price": 175.0, "winner": "Bidder B"}, "evidence": {"final_price": "175", "winner": "B"}},
    )
    assert derived["allocative_efficiency"] == pytest.approx(2 / 3)
    assert derived["efficient_allocation"] is False

    rows = []
    for cell in cells:
        regime = cell["treatments"]["Information regime"]
        price = {"private": 100.0, "public ranges": 150.0, "public exact": 200.0}[regime]
        winner = {"private": "Bidder A", "public ranges": "Bidder B", "public exact": "Bidder C"}[regime]
        measurement = {"outcomes": {"final_price": price, "winner": winner}, "evidence": {"final_price": "price", "winner": "winner"}}
        rows.append({**cell, "status": "complete", "measurement": measurement, "derived": derive_cell_summary(auction, cell, measurement)})
    analysis = analyze_frontier_study(auction, rows)
    assert "one model draw per OA cell" in analysis["caveat"]
    regime_marginals = analysis["main_effect_marginals"]["Information regime"]
    assert [entry["level"] for entry in regime_marginals] == ["private", "public ranges", "public exact"]
    assert [entry["outcomes"]["final_price"]["mean"] for entry in regime_marginals] == [100.0, 150.0, 200.0]
    assert regime_marginals[0]["outcomes"]["winner"]["counts"] == {"Bidder A": 9}
    assert analysis["study_summary"]["mean_final_price"] == 150.0


def test_frontier_cli_exposes_all_commands_and_mock_pilot_is_bounded(tmp_path: Path) -> None:
    parser = build_frontier_parser()
    for command in ("validate", "dry-run", "pilot", "full-run", "analyze"):
        assert parser.parse_args([command, "--help"]) if False else command in parser._subparsers._group_actions[0].choices

    dry_dir = tmp_path / "dry"
    assert frontier_main(["dry-run", "--design-file", str(DESIGN_PATH), "--output-dir", str(dry_dir)]) == 0
    plan = json.loads((dry_dir / "cell_plan.json").read_text(encoding="utf-8"))
    assert sum(len(cells) for cells in plan.values()) == 108

    pilot_dir = tmp_path / "pilot"
    assert frontier_main(
        [
            "pilot",
            "--design-file",
            str(DESIGN_PATH),
            "--output-dir",
            str(pilot_dir),
            "--provider",
            "mock",
            "--base-seed",
            "10",
        ]
    ) == 0
    rows = list(pilot_dir.glob("studies/*/cells.jsonl"))
    assert len(rows) == 4
    assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in rows) == 4
    assert all(json.loads(path.read_text(encoding="utf-8").splitlines()[0])["status"] == "complete" for path in rows)
