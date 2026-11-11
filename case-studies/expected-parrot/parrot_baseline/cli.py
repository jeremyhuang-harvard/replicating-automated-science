"""Command line: `demo` (offline sample study) and `run` (your concept, demo or mlx engine)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engines import DemoEngine, EngineError
from .mlx_engine import DEFAULT_MODEL, MlxEngine
from .report import LABEL
from .runner import StudyResult, run_study
from .schemas import Concept, SchemaError

DEMO_CONCEPT = Concept(
    title="Lasagna Cola",
    description=(
        "A zero-sugar cola with savory tomato, basil and parmesan notes, served ice cold. "
        "A bold, limited-edition twist on soda for people who like unexpected flavors."
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parrot_baseline.cli", description="Simulated consumer concept test (hypotheses, not human evidence).")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the built-in sample concept with the offline demo engine")
    demo.add_argument("--output-dir", default=".artifacts/demo")
    demo.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run", help="run your own concept")
    run.add_argument("--concept-title", required=True)
    run.add_argument("--concept-description", required=True)
    run.add_argument("--engine", choices=("demo", "mlx"), required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--panel-size", type=int, default=12)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--model", default=DEFAULT_MODEL, help="MLX model id (mlx engine only)")
    return parser


def _report(result: StudyResult) -> int:
    counts = result.summary["counts"]
    print(f"Study {result.manifest.study_id}: engine={result.manifest.engine} model={result.manifest.model} "
          f"valid={counts['ok']}/{counts['panel']}")
    for name, path in result.paths.items():
        print(f"  {name}: {path}")
    for warning in result.summary["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"{LABEL}.")
    return 0 if counts["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = run_study(DEMO_CONCEPT, DemoEngine(), Path(args.output_dir), seed=args.seed)
        else:
            concept = Concept(args.concept_title, args.concept_description)
            engine = DemoEngine() if args.engine == "demo" else MlxEngine(model=args.model)
            result = run_study(concept, engine, Path(args.output_dir), panel_size=args.panel_size, seed=args.seed)
    except SchemaError as exc:
        parser.error(str(exc))  # exits 2
    except (EngineError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return _report(result)


if __name__ == "__main__":
    sys.exit(main())
