"""CLI for validating, piloting, running, and analyzing frontier designs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .frontier_runner import analyze_frontier_study, generate_oa_cells, run_experiment, validate_bundle
from .hashing import stable_hash
from .providers import FrontierMockProvider, MLXInProcessProvider

DEFAULT_DESIGN_FILE = Path("designs/frontier-planned-v1/designs.json")
DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_PILOT_OUTPUT = Path("outputs/frontier-pilot-2026-10-14")
DEFAULT_FULL_OUTPUT = Path("outputs/frontier-planned-qwen-subjects-2026-10-14")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _add_design_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--design-file", type=Path, default=DEFAULT_DESIGN_FILE)


def _add_run_arguments(parser: argparse.ArgumentParser, default_output: Path) -> None:
    _add_design_argument(parser)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--provider", choices=("mlx", "mock"), default="mlx")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--base-seed", type=int, default=2026101400)
    parser.add_argument("--wall-clock-budget-seconds", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assr-frontier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the fixed bundle and OA invariants")
    _add_design_argument(validate)

    dry_run = subparsers.add_parser("dry-run", help="write the deterministic 108-cell plan without inference")
    _add_design_argument(dry_run)
    dry_run.add_argument("--output-dir", type=Path, default=Path("outputs/frontier-dry-run-2026-10-14"))

    pilot = subparsers.add_parser("pilot", help="run exactly one cell per study (four total)")
    _add_run_arguments(pilot, DEFAULT_PILOT_OUTPUT)

    full_run = subparsers.add_parser("full-run", help="run or resume all 108 cells")
    _add_run_arguments(full_run, DEFAULT_FULL_OUTPUT)

    analyze = subparsers.add_parser("analyze", help="analyze available terminal cell records")
    _add_design_argument(analyze)
    analyze.add_argument("--output-dir", type=Path, default=DEFAULT_FULL_OUTPUT)
    return parser


def _load_bundle(path: Path) -> dict[str, Any]:
    return validate_bundle(_read_json(path))


def _plans(bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    plan = {study["study_id"]: generate_oa_cells(study) for study in bundle["studies"]}
    if sum(len(cells) for cells in plan.values()) != 108:
        raise ValueError("expected exactly 108 planned cells")
    return plan


def _provider(args: argparse.Namespace):
    if args.provider == "mock":
        return FrontierMockProvider()
    return MLXInProcessProvider.load(args.model, max_tokens=args.max_tokens)


def _analyze(bundle: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for study in bundle["studies"]:
        study_dir = output_dir / "studies" / study["study_id"]
        rows = _read_jsonl(study_dir / "cells.jsonl")
        report = analyze_frontier_study(study, rows)
        _write_json(study_dir / "analysis.json", report)
        reports[study["study_id"]] = report
    aggregate = {
        "schema_version": 1,
        "caveat": "Descriptive one-draw local-model evidence; not human behavior.",
        "studies": reports,
    }
    _write_json(output_dir / "analysis.json", aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = _load_bundle(args.design_file)
    plan = _plans(bundle)
    if args.command == "validate":
        report = {
            "valid": True,
            "bundle_hash": stable_hash(bundle),
            "studies": 4,
            "cells_per_study": 27,
            "total_cells": 108,
            "scenarios": [study["scenario"] for study in bundle["studies"]],
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "dry-run":
        _write_json(args.output_dir / "design_bundle.json", bundle)
        _write_json(args.output_dir / "cell_plan.json", plan)
        _write_json(
            args.output_dir / "validation_report.json",
            {"valid": True, "bundle_hash": stable_hash(bundle), "plan_hash": stable_hash(plan), "total_cells": 108},
        )
        print(args.output_dir)
        return 0
    if args.command in {"pilot", "full-run"}:
        selected = {study_id: cells[:1] for study_id, cells in plan.items()} if args.command == "pilot" else plan
        result = run_experiment(
            bundle,
            selected,
            args.output_dir,
            provider=_provider(args),
            model=args.model,
            base_seed=args.base_seed,
            wall_clock_budget_seconds=args.wall_clock_budget_seconds,
        )
        print(json.dumps({"output_dir": str(args.output_dir), **result}, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "analyze":
        aggregate = _analyze(bundle, args.output_dir)
        print(json.dumps({"output_dir": str(args.output_dir), "studies": len(aggregate["studies"])}, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
