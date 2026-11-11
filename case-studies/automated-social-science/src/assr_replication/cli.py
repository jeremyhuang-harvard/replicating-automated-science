"""Command-line entry points for local replication smoke runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .providers import MLXCLIProvider, ProcessRunner
from .scientist_runner import MAX_ATTEMPTS, SCIENTIST_TEMPERATURE, generate_scientist_designs
from .smoke import run_bargaining_smoke, run_mock_smoke

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen3-8B-4bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assr-replication")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the deterministic no-network smoke")
    smoke.add_argument("--output-dir", type=Path, default=Path("outputs/mock-smoke"))

    local = subparsers.add_parser("local-smoke", help="run smoke with local MLX inference")
    local.add_argument("--output-dir", type=Path, default=Path("outputs/local-smoke"))
    local.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    local.add_argument("--max-tokens", type=int, default=128)
    local.add_argument("--temp", type=float, default=0.0)
    local.add_argument("--seed", type=int, default=0)
    local.add_argument("--executable", default="mlx_lm.generate")

    designs = subparsers.add_parser(
        "generate-designs",
        help="generate one first-valid scientist design for each canonical scenario",
    )
    designs.add_argument("--output-dir", type=Path, default=Path("outputs/canonical-designs"))
    designs.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    designs.add_argument("--max-tokens", type=int, default=2200)
    designs.add_argument("--base-seed", type=int, default=20241001)
    designs.add_argument("--executable", default="mlx_lm.generate")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    mlx_process_runner: ProcessRunner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        output_dir = run_mock_smoke(args.output_dir)
        print(output_dir)
        return 0
    if args.command == "local-smoke":
        provider_options = {
            "executable": args.executable,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "seed": args.seed,
        }
        if mlx_process_runner is not None:
            provider_options["process_runner"] = mlx_process_runner
        provider = MLXCLIProvider(**provider_options)
        output_dir = run_bargaining_smoke(
            args.output_dir,
            provider,
            model=args.model,
        )
        print(output_dir)
        return 0
    if args.command == "generate-designs":
        def provider_factory(seed: int) -> MLXCLIProvider:
            provider_options = {
                "executable": args.executable,
                "max_tokens": args.max_tokens,
                "temperature": SCIENTIST_TEMPERATURE,
                "seed": seed,
            }
            if mlx_process_runner is not None:
                provider_options["process_runner"] = mlx_process_runner
            return MLXCLIProvider(**provider_options)

        seeds = tuple(args.base_seed + offset for offset in range(4 * MAX_ATTEMPTS))
        output_dir = generate_scientist_designs(
            args.output_dir,
            provider_factory=provider_factory,
            model=args.model,
            seeds=seeds,
        )
        print(output_dir)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
